"""E4 reference: multi-start long-run PT-MALA with a Laplace-mixture SNIS cross-check.

The E4 target is a 24-dimensional two-component coupled quartic chain at
``beta = 8``. It has no closed-form marginal, no exact sampler, and no analytic
order-parameter density, so the "ground truth" every E4 metric is scored
against is a *numerical* estimate built here and frozen on disk.

Two independent estimates are constructed and cross-checked.

**Primary -- multi-start parallel-tempered MALA.** ``n_runs`` long runs, one
initialised from each of the four refined homogeneous phases, each carrying
``chains_per_run`` chains and a geometric inverse-temperature ladder from
``beta`` down to ``beta_min``. Every replica moves by MH-corrected MALA at its
own ``beta_k``, with the reverse proposal drift recomputed at the proposal
point; the swap is the usual product-target involution. After burn-in the cold
replica's unweighted 24-D configurations become the official reference sample
bank, tagged with run id, chain id, and iteration index.

**Independent cross-check -- Laplace-mixture SNIS.** A four-component Gaussian
mixture centred on the coherent states, with covariances from the regularised
inverse Hessians at ``beta``, is used as an importance proposal. The weights are
used *directly* for weighted estimates: SNIS is a weighted cross-check, not a
second unweighted sample bank.

Uncertainty discipline
----------------------
Reference uncertainty is estimated by independent chain or by non-overlapping
block, never by treating correlated draws from one cold chain as i.i.d.
replicates. The relative susceptibility and relative correlation cross-check
standard errors come from a hierarchical bootstrap applied directly to the whole
statistic (resampling independent PT blocks and independent SNIS runs and
recomputing the relative difference per replicate), never from combining
elementwise standard errors.

Acceptance
----------
Every gate in ``configs/experiments/E4_reference_acceptance.yaml`` is evaluated
and written to ``reference_validation.json`` with its threshold, observed value,
standard error, block length or bootstrap definition, bootstrap seed, bootstrap
replicate count, verdict, and a diagnostic message. ``reference_validated`` is
set only when every gate passes; otherwise :class:`ReferenceValidationError` is
raised after the validation file has been written, so the failure is both fatal
and inspectable.

Saved representations
---------------------
``reference_samples_24d.npz``
    The PT-MALA bank with run/chain identifiers, iteration index, per-sample
    observables, and run diagnostics.
``reference_order_parameter_grid.npz``
    ``p*(m_x, m_y)`` on a fixed grid and ``F*(m_x, m_y) = -beta^-1 log p* + C``.
    This is a NUMERICAL reference estimate obtained by projecting the PT-MALA
    samples onto the order-parameter plane -- it is not an analytic grid and
    carries the sampling uncertainty of the bank it came from.
``reference_snis_weighted_24d.npz``
    Proposal draws, normalized weights, phase labels, ESS, maximum weight, and
    coverage diagnostics.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import copy
import math

import numpy as np
import torch
import yaml

from .. import metrics, observables
from ..device import resolve_device
from ..observables import OUTSIDE_LABEL, GradientFlowBasinMap2D
from ..potentials import PHASES, site_potential_grad
from ..samplers import geometric_ladder
from .base import (Reference, check_positive_int, check_seed, frozen_generator,
                   load_npz, read_json, save_npz, write_json)

#: Repository root, used to resolve the acceptance path recorded in E4.yaml.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

SAMPLES_FILE = "reference_samples_24d.npz"
GRID_FILE = "reference_order_parameter_grid.npz"
SNIS_FILE = "reference_snis_weighted_24d.npz"
VALIDATION_FILE = "reference_validation.json"
ACCEPTANCE_FILE = "reference_acceptance.yaml"
BASIN_CACHE_FILE = "reference_basin_map_order_parameter.npz"

#: Defaults for sizes that are not pinned in E4.yaml. Every one of them can be
#: overridden from the ``reference:`` block, and the resolved value is recorded.
_BASIN_MAP_DEFAULTS = {"bound": 4.0, "n_grid": 600, "dt_flow": 1.5e-4,
                       "n_flow": 20_000}
_GRID_DEFAULTS = {"lo": -2.5, "hi": 2.5, "n_bins": 200,
                  "kde_bandwidth_rule": "silverman_2d"}
_SNIS_CHUNK = 50_000


class ReferenceValidationError(RuntimeError):
    """A reference failed at least one frozen acceptance gate.

    Carries the failing gate records so a caller can print them and exit
    nonzero. A reference that raises this is never an official reference: it is
    written to disk for inspection with ``reference_validated: false``.
    """

    def __init__(self, failed_gates: Sequence[dict], *, directory=None) -> None:
        self.failed_gates = [dict(gate) for gate in failed_gates]
        self.directory = None if directory is None else str(directory)
        lines = [f"  - {gate['metric']}: observed={gate['observed_value']!r} "
                 f"threshold={gate['threshold']!r} ({gate['diagnostic_message']})"
                 for gate in self.failed_gates]
        location = ("" if self.directory is None
                    else f"\nvalidation record: {self.directory}/{VALIDATION_FILE}")
        super().__init__(
            f"the E4 reference failed {len(self.failed_gates)} acceptance "
            f"gate(s):\n" + "\n".join(lines) + location)


# ============================================================ config plumbing
def _resolve_acceptance_path(value) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (_REPOSITORY_ROOT / path)


def _load_acceptance(config: Mapping, acceptance_path=None) -> tuple[dict, Path]:
    """Load the frozen acceptance gates. Every threshold comes from this file."""
    if acceptance_path is None:
        acceptance_path = config["reference"]["acceptance"]
    path = _resolve_acceptance_path(acceptance_path)
    with open(path, encoding="utf-8") as handle:
        acceptance = yaml.safe_load(handle)
    if str(acceptance.get("experiment")) != "E4":
        raise ValueError(
            f"acceptance file {path} declares experiment "
            f"{acceptance.get('experiment')!r}, expected 'E4'")
    return acceptance, path


def _resolved(section: Mapping | None, defaults: Mapping) -> dict:
    out = dict(defaults)
    out.update({key: value for key, value in dict(section or {}).items()
                if key in defaults})
    return out


def _progress(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[E4 reference] {message}", flush=True)


# ============================================================= gate records
def _finite(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _gate(metric: str, threshold, observed, *, direction: str,
          standard_error=None, block_length=None, bootstrap_definition=None,
          bootstrap_seed=None, bootstrap_replicates=None, message: str = "",
          **extra) -> dict:
    """One acceptance-gate record with every field the validation file needs.

    ``direction`` is ``"min"`` when the observed value must be at least the
    threshold and ``"max"`` when it must be at most the threshold. A non-finite
    observation never passes.
    """
    if direction not in ("min", "max"):
        raise ValueError("direction must be 'min' or 'max'")
    value = _finite(observed)
    limit = _finite(threshold)
    if math.isnan(value) or math.isnan(limit):
        passed = False
    elif direction == "min":
        passed = value >= limit
    else:
        passed = value <= limit
    record = {
        "metric": str(metric),
        "direction": direction,
        "threshold": limit,
        "observed_value": value,
        "standard_error": (None if standard_error is None
                           else _finite(standard_error)),
        "block_length": (None if block_length is None else int(block_length)),
        "bootstrap_definition": bootstrap_definition,
        "bootstrap_seed": (None if bootstrap_seed is None
                           else int(bootstrap_seed)),
        "bootstrap_replicates": (None if bootstrap_replicates is None
                                 else int(bootstrap_replicates)),
        "passed": bool(passed),
        "diagnostic_message": str(message),
    }
    record.update(extra)
    return record


def _tolerance_gate(metric: str, difference, *, floor: float, multiplier: float,
                    combined_se, message: str, **extra) -> dict:
    """``|difference| <= max(floor, multiplier * combined_se)``."""
    se = _finite(combined_se)
    scaled = float("nan") if math.isnan(se) else multiplier * se
    tolerance = float(floor) if math.isnan(se) else max(float(floor), scaled)
    return _gate(metric, tolerance, abs(_finite(difference)), direction="max",
                 standard_error=se, message=message,
                 absolute_floor=float(floor),
                 combined_se_multiplier=float(multiplier), **extra)


# ========================================================== feature algebra
class _Layout:
    """Index layout of the per-sample feature vector.

    Every derived statistic the gates and the frozen targets need is a smooth
    function of mass-weighted sums of these features, so one summary row per
    independent unit (a PT block, a SNIS run) is enough to recompute the whole
    statistic on a bootstrap replicate.
    """

    def __init__(self, n_phases: int, n_lags: int) -> None:
        names = ["inside"]
        names += [f"phase_{index}" for index in range(n_phases)]
        names += ["e", "e2", "mx", "my", "mx2", "mxmy", "my2", "s", "s2",
                  "G", "G2"]
        names += [f"C{lag}" for lag in range(n_lags)]
        names += ["kink"]
        self.names = tuple(names)
        self.index = {name: position for position, name in enumerate(names)}
        self.n_features = len(names)
        self.n_phases = int(n_phases)
        self.n_lags = int(n_lags)

    def __getitem__(self, name: str) -> int:
        return self.index[name]


def _feature_matrix(layout: _Layout, *, labels, energies, m, coherences,
                    correlations, kinks) -> torch.Tensor:
    """Assemble the ``(N, F)`` feature matrix from per-sample observables."""
    n = energies.shape[0]
    device, dtype = energies.device, energies.dtype
    features = torch.zeros(n, layout.n_features, dtype=dtype, device=device)
    inside = (labels != OUTSIDE_LABEL)
    features[:, layout["inside"]] = inside.to(dtype)
    for phase in range(layout.n_phases):
        features[:, layout[f"phase_{phase}"]] = (labels == phase).to(dtype)
    mx, my = m[:, 0], m[:, 1]
    squared = mx * mx + my * my
    features[:, layout["e"]] = energies
    features[:, layout["e2"]] = energies * energies
    features[:, layout["mx"]] = mx
    features[:, layout["my"]] = my
    features[:, layout["mx2"]] = mx * mx
    features[:, layout["mxmy"]] = mx * my
    features[:, layout["my2"]] = my * my
    features[:, layout["s"]] = squared
    features[:, layout["s2"]] = squared * squared
    features[:, layout["G"]] = coherences
    features[:, layout["G2"]] = coherences * coherences
    for lag in range(layout.n_lags):
        features[:, layout[f"C{lag}"]] = correlations[:, lag]
    features[:, layout["kink"]] = kinks
    return features


def _summary_row(features: torch.Tensor, mass: torch.Tensor | None
                 ) -> np.ndarray:
    """``[A, A2, sum_i a_i f_i]`` for a group of samples with masses ``a_i``."""
    if mass is None:
        total = float(features.shape[0])
        total_squared = float(features.shape[0])
        sums = features.sum(dim=0)
    else:
        total = float(mass.sum().item())
        total_squared = float((mass * mass).sum().item())
        sums = (mass.unsqueeze(1) * features).sum(dim=0)
    return np.concatenate([[total, total_squared],
                           sums.detach().cpu().numpy().astype(float)])


def _derive(row: np.ndarray, layout: _Layout, *, beta: float,
            n_sites: int) -> dict:
    """Every derived statistic from one summary row.

    The mass-weighted algebra reduces to the unweighted estimators exactly:
    with ``a_i = 1`` the denominator ``A - A2/A`` is ``n - 1``, reproducing the
    ``N-1`` convention of :mod:`src.observables`; with ``a_i`` proportional to
    SNIS weights it is ``A (1 - sum_k wbar_k^2)``, the reliability-weight
    denominator the weighted estimators use.
    """
    row = np.asarray(row, dtype=float)
    total, total_squared, sums = row[0], row[1], row[2:]
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("a summary row must carry positive total mass")
    denominator = total - total_squared / total

    def mean(name: str) -> float:
        return float(sums[layout[name]] / total)

    def covariance(name: str, mean_first: float, mean_second: float) -> float:
        return float((sums[layout[name]] - total * mean_first * mean_second)
                     / denominator)

    inside_mass = float(sums[layout["inside"]])
    probabilities = np.zeros(layout.n_phases, dtype=float)
    if inside_mass > 0.0:
        for phase in range(layout.n_phases):
            probabilities[phase] = float(
                sums[layout[f"phase_{phase}"]] / inside_mass)
    mean_e = mean("e")
    variance_e = covariance("e2", mean_e, mean_e)
    mean_mx, mean_my = mean("mx"), mean("my")
    cov = np.array([[covariance("mx2", mean_mx, mean_mx),
                     covariance("mxmy", mean_mx, mean_my)],
                    [covariance("mxmy", mean_mx, mean_my),
                     covariance("my2", mean_my, mean_my)]], dtype=float)
    mean_G = mean("G")
    variance_G = covariance("G2", mean_G, mean_G)
    correlation = np.array([mean(f"C{lag}") for lag in range(layout.n_lags)],
                           dtype=float)
    mean_s, mean_s2 = mean("s"), mean("s2")
    binder = (float("nan") if mean_s <= 0.0
              else 1.0 - mean_s2 / (2.0 * mean_s * mean_s))
    return {
        "phase_probabilities": probabilities,
        "phase_outside_mass": float(1.0 - inside_mass / total),
        "order_parameter_mean": np.array([mean_mx, mean_my], dtype=float),
        "susceptibility": beta * n_sites * cov,
        "energy_per_site_mean": mean_e,
        "energy_per_site_variance": variance_e,
        "energy_per_site_sd": float(math.sqrt(max(variance_e, 0.0))),
        "coherence_mean": mean_G,
        "coherence_sd": float(math.sqrt(max(variance_G, 0.0))),
        "two_point_correlation": correlation,
        "connected_two_point_correlation":
            correlation - (mean_mx ** 2 + mean_my ** 2),
        "kink_density_mean": mean("kink"),
        "heat_capacity_per_site": beta * beta * n_sites * variance_e,
        "binder_cumulant": binder,
    }


#: Order in which derived statistics are flattened for the bootstrap.
_STATISTIC_ORDER = (
    "phase_probabilities", "phase_outside_mass", "order_parameter_mean",
    "susceptibility", "energy_per_site_mean", "energy_per_site_variance",
    "energy_per_site_sd", "coherence_mean", "coherence_sd",
    "two_point_correlation", "connected_two_point_correlation",
    "kink_density_mean", "heat_capacity_per_site", "binder_cumulant",
)


def _stats_vector(stats: Mapping) -> np.ndarray:
    return np.concatenate([np.atleast_1d(np.asarray(stats[name], dtype=float))
                           .reshape(-1) for name in _STATISTIC_ORDER])


def _unflatten_like(vector: np.ndarray, template: Mapping) -> dict:
    out, cursor = {}, 0
    for name in _STATISTIC_ORDER:
        reference = np.atleast_1d(np.asarray(template[name], dtype=float))
        size = reference.size
        block = np.asarray(vector[cursor:cursor + size], dtype=float)
        cursor += size
        out[name] = (float(block[0]) if np.ndim(template[name]) == 0
                     else block.reshape(reference.shape))
    return out


# ========================================================= Laplace proposal
class _LaplaceMixture:
    """``q(x) = sum_s omega_s N(x; mu_s, Sigma_s)`` around the coherent states.

    ``Sigma_s = (H_s + reg I)^-1 / beta`` is the regularised inverse Hessian at
    ``beta``: the regularisation is added to the Hessian diagonal *before*
    inversion, and both the value and the fact that it was applied are recorded.
    ``omega_s`` are the Laplace weights ``exp(-beta V(mu_s)) det(H_s + reg I)^-1/2``
    normalised to sum to one.
    """

    def __init__(self, means: torch.Tensor, hessians: torch.Tensor,
                 energies: torch.Tensor, beta: float,
                 regularization: float) -> None:
        if means.ndim != 2 or hessians.ndim != 3:
            raise ValueError("means must be (K, d) and hessians (K, d, d)")
        self.beta = float(beta)
        self.regularization = float(regularization)
        self.means = means
        n_components, dimension = means.shape
        self.n_components, self.d = int(n_components), int(dimension)
        eye = torch.eye(self.d, dtype=means.dtype, device=means.device)
        regularised = hessians + self.regularization * eye
        regularised = 0.5 * (regularised + regularised.transpose(-1, -2))
        sign, logdet = torch.linalg.slogdet(regularised)
        if not bool((sign > 0).all().item()):
            raise ValueError(
                "the regularised coherent Hessians must be positive definite; "
                "increase hessian_regularization")
        self.regularised_hessians = regularised
        covariance = torch.linalg.inv(regularised) / self.beta
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        self.covariances = covariance
        self.cholesky = torch.linalg.cholesky(covariance)
        self.precisions = self.beta * regularised
        log_weights = -self.beta * energies - 0.5 * logdet
        self.log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
        self.weights = torch.exp(self.log_weights)
        self.cumulative_weights = torch.cumsum(self.weights, dim=0)
        _, logdet_covariance = torch.linalg.slogdet(covariance)
        self.log_normalizers = -0.5 * (self.d * math.log(2.0 * math.pi)
                                       + logdet_covariance)

    def sample(self, n: int, generator: torch.Generator) -> torch.Tensor:
        device, dtype = self.means.device, self.means.dtype
        gen_device = generator.device
        u = torch.rand(n, generator=generator, device=gen_device,
                       dtype=dtype).to(device)
        z = torch.randn(n, self.d, generator=generator, device=gen_device,
                        dtype=dtype).to(device)
        component = torch.searchsorted(
            self.cumulative_weights.contiguous(), u.contiguous()
        ).clamp_(max=self.n_components - 1)
        x = self.means[component].clone()
        for index in range(self.n_components):
            mask = component == index
            if bool(mask.any().item()):
                x[mask] = x[mask] + z[mask] @ self.cholesky[index].T
        return x, component

    def log_q(self, x: torch.Tensor, chunk: int = _SNIS_CHUNK) -> torch.Tensor:
        parts = []
        for start in range(0, x.shape[0], int(chunk)):
            block = x[start:start + int(chunk)]
            difference = block.unsqueeze(1) - self.means.unsqueeze(0)
            quadratic = torch.einsum("bkd,kde,bke->bk", difference,
                                     self.precisions, difference)
            parts.append(torch.logsumexp(
                self.log_weights + self.log_normalizers - 0.5 * quadratic,
                dim=1))
        return torch.cat(parts)

    def describe(self) -> dict:
        return {
            "family": "laplace_gaussian_mixture",
            "n_components": self.n_components,
            "dimension": self.d,
            "beta": self.beta,
            "hessian_regularization": self.regularization,
            "covariance_rule": "Sigma_s = (H_s + reg I)^-1 / beta",
            "weight_rule": "omega_s ~ exp(-beta V(mu_s)) det(H_s + reg I)^-1/2",
            "component_weights": self.weights.detach().cpu().tolist(),
            "log_component_weights": self.log_weights.detach().cpu().tolist(),
        }


# ================================================================ PT-MALA
def _run_pt_mala(target, settings: Mapping, *, verbose: bool) -> dict:
    """Multi-start parallel-tempered MH-corrected MALA.

    Runs are batched as a leading tensor dimension ``(runs, replicas, chains, d)``
    so the whole ensemble advances in one kernel per operation, but every run
    keeps its own ``torch.Generator``, so the runs are independent random
    streams with individually recorded seeds.
    """
    device = target.device
    dtype = torch.float64
    dimension = int(target.d)
    coherent = target.extras["coherent_states"].to(device=device, dtype=dtype)
    phases = list(target.extras["phases"])

    n_runs = check_positive_int(settings["n_runs"], "pt_mala.n_runs")
    chains = check_positive_int(settings["chains_per_run"],
                                "pt_mala.chains_per_run")
    n_replicas = check_positive_int(settings["n_replicas"],
                                    "pt_mala.n_replicas")
    swap_interval = check_positive_int(settings["swap_interval"],
                                       "pt_mala.swap_interval")
    total_steps = check_positive_int(settings["total_steps"],
                                     "pt_mala.total_steps")
    burn_in = int(settings["burn_in_steps"])
    thinning = check_positive_int(settings["thinning"], "pt_mala.thinning")
    dt = float(settings["dt"])
    beta_min = float(settings["beta_min"])
    seed_base = check_seed(settings["seed_base"], "pt_mala.seed_base")
    init_sigma = float(settings.get("init_sigma", 0.05))
    init_phases = list(settings.get("init_phases", phases))
    if burn_in < 0 or burn_in >= total_steps:
        raise ValueError("burn_in_steps must satisfy 0 <= burn_in < total_steps")
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("pt_mala.dt must be finite and positive")
    if not 0.0 < beta_min <= target.beta:
        raise ValueError("pt_mala.beta_min must lie in (0, beta]")
    if len(init_phases) < n_runs:
        raise ValueError(
            f"pt_mala.init_phases lists {len(init_phases)} phases but "
            f"{n_runs} runs were requested")
    n_checkpoints = (total_steps - burn_in) // thinning
    if n_checkpoints < 4:
        raise ValueError(
            "the PT schedule must retain at least four saved checkpoints per "
            f"chain, got {n_checkpoints}")

    phase_indices = [phases.index(name) for name in init_phases[:n_runs]]
    seeds = [seed_base + index for index in range(n_runs)]
    generators = [frozen_generator(device, seed) for seed in seeds]
    betas = geometric_ladder(target.beta, beta_min, n_replicas, device)
    proposal_variance = (2.0 * dt / betas).reshape(1, n_replicas, 1, 1)
    proposal_scale = proposal_variance.sqrt()
    inverse_variance = 1.0 / (2.0 * proposal_variance.reshape(1, n_replicas, 1))
    beta_column = betas.reshape(1, n_replicas, 1)

    x = torch.stack([
        coherent[phase_indices[run]] + init_sigma * torch.randn(
            (n_replicas, chains, dimension), generator=generators[run],
            device=device, dtype=dtype)
        for run in range(n_runs)])
    energy, force = target.value_and_force(x)

    accepted = torch.zeros(n_replicas, dtype=torch.float64, device=device)
    proposed = torch.zeros(n_replicas, dtype=torch.float64, device=device)
    swap_accepted = torch.zeros(max(n_replicas - 1, 1), dtype=torch.float64,
                                device=device)
    swap_proposed = torch.zeros(max(n_replicas - 1, 1), dtype=torch.float64,
                                device=device)
    saved, saved_steps = [], []
    report_every = max(1, total_steps // 20)

    for step in range(1, total_steps + 1):
        noise = torch.stack([
            torch.randn((n_replicas, chains, dimension), generator=generator,
                        device=device, dtype=dtype)
            for generator in generators])
        mean_forward = x + dt * force
        y = mean_forward + proposal_scale * noise
        energy_y, force_y = target.value_and_force(y)
        mean_reverse = y + dt * force_y
        forward = ((y - mean_forward) ** 2).sum(-1)
        reverse = ((x - mean_reverse) ** 2).sum(-1)
        log_alpha = (-beta_column * (energy_y - energy)
                     + (forward - reverse) * inverse_variance)
        uniform = torch.stack([
            torch.rand((n_replicas, chains), generator=generator,
                       device=device, dtype=dtype)
            for generator in generators])
        finite = torch.isfinite(y).all(dim=-1) & torch.isfinite(log_alpha)
        accept = (torch.log(uniform) < log_alpha) & finite
        x = torch.where(accept.unsqueeze(-1), y, x)
        energy = torch.where(accept, energy_y, energy)
        force = torch.where(accept.unsqueeze(-1), force_y, force)
        accepted += accept.to(dtype).sum(dim=(0, 2))
        proposed += float(n_runs * chains)

        if step % swap_interval == 0:
            offset = (step // swap_interval) % 2
            pairs = list(range(offset, n_replicas - 1, 2))
            if pairs:
                swap_uniform = torch.stack([
                    torch.rand((len(pairs), chains), generator=generator,
                               device=device, dtype=dtype)
                    for generator in generators])
                for position, lower in enumerate(pairs):
                    log_swap = ((betas[lower] - betas[lower + 1])
                                * (energy[:, lower] - energy[:, lower + 1]))
                    swap = torch.log(swap_uniform[:, position]) < log_swap
                    column = swap.unsqueeze(-1)
                    new_lower = torch.where(column, x[:, lower + 1], x[:, lower])
                    new_upper = torch.where(column, x[:, lower], x[:, lower + 1])
                    x[:, lower], x[:, lower + 1] = new_lower, new_upper
                    energy_lower = torch.where(swap, energy[:, lower + 1],
                                               energy[:, lower])
                    energy_upper = torch.where(swap, energy[:, lower],
                                               energy[:, lower + 1])
                    energy[:, lower] = energy_lower
                    energy[:, lower + 1] = energy_upper
                    force_lower = torch.where(column, force[:, lower + 1],
                                              force[:, lower])
                    force_upper = torch.where(column, force[:, lower],
                                              force[:, lower + 1])
                    force[:, lower], force[:, lower + 1] = force_lower, force_upper
                    swap_accepted[lower] += swap.to(dtype).sum()
                    swap_proposed[lower] += float(n_runs * chains)

        if step > burn_in and (step - burn_in) % thinning == 0:
            saved.append(x[:, 0].clone())
            saved_steps.append(step)

        if verbose and step % report_every == 0:
            _progress(True, f"PT-MALA step {step}/{total_steps} "
                            f"({100.0 * step / total_steps:.0f}%), "
                            f"{len(saved)} checkpoints")

    bank = torch.stack(saved)                      # (T, runs, chains, d)
    n_saved = bank.shape[0]
    # Chain-major ordering: sample i belongs to chain i // T at checkpoint i % T,
    # so a non-overlapping block is a contiguous slice of one chain.
    configurations = bank.permute(1, 2, 0, 3).reshape(
        n_runs * chains * n_saved, dimension).contiguous()
    run_id = np.repeat(np.arange(n_runs), chains * n_saved)
    chain_id = np.tile(np.repeat(np.arange(chains), n_saved), n_runs)
    iteration = np.tile(np.asarray(saved_steps, dtype=np.int64),
                        n_runs * chains)
    return {
        "configurations": configurations,
        "run_id": run_id.astype(np.int64),
        "chain_id": chain_id.astype(np.int64),
        "iteration_step": iteration,
        "checkpoint_index": np.tile(np.arange(n_saved), n_runs * chains),
        "n_runs": n_runs,
        "chains_per_run": chains,
        "n_chains": n_runs * chains,
        "n_checkpoints": n_saved,
        "betas": betas,
        "seeds": seeds,
        "phase_indices": phase_indices,
        "init_phases": init_phases[:n_runs],
        "init_sigma": init_sigma,
        "dt": dt,
        "beta_min": beta_min,
        "n_replicas": n_replicas,
        "swap_interval": swap_interval,
        "burn_in_steps": burn_in,
        "total_steps": total_steps,
        "thinning": thinning,
        "saved_steps": np.asarray(saved_steps, dtype=np.int64),
        "mala_acceptance": (accepted / proposed.clamp_min(1.0)
                            ).detach().cpu().numpy(),
        "swap_acceptance": (swap_accepted / swap_proposed.clamp_min(1.0)
                            ).detach().cpu().numpy(),
    }


# ============================================================ per-sample pass
def _per_sample_observables(target, x: torch.Tensor, *, basin_map,
                            site_minima) -> dict:
    """Every per-sample E4 observable, computed once through :mod:`src.observables`."""
    m = observables.order_parameter(target, x)
    return {
        "order_parameter": m,
        "labels": observables.phase_labels(m, basin_map),
        "energy_per_site": observables.energy_per_site(target, x),
        "coherence": observables.coherence(target, x),
        "two_point_correlation": observables.two_point_correlation(target, x),
        "kink_density": observables.kink_density(target, x, site_minima),
    }


def _canonical_estimates(target, per_sample: Mapping, *, weights=None,
                         n_phases: int) -> dict:
    """Point estimates through the canonical aggregators in :mod:`src.observables`.

    The summary algebra in :func:`_derive` is only ever used for bootstrap
    replicates; the reported values come from these functions so the reference
    and the metrics that consume it share one definition.
    """
    m = per_sample["order_parameter"]
    labels = per_sample["labels"]
    energies = per_sample["energy_per_site"]
    coherences = per_sample["coherence"]
    correlations = per_sample["two_point_correlation"]
    kinks = per_sample["kink_density"]
    n_sites = int(target.potential.n_sites)
    if weights is None:
        probabilities, outside = observables.phase_probabilities(labels, n_phases)
        mean_m = observables.order_parameter_mean(m)
        correlation = observables.two_point_correlation_mean(correlations)
        record = {
            "phase_probabilities": probabilities,
            "phase_outside_mass": outside,
            "order_parameter_mean": mean_m,
            "susceptibility": observables.susceptibility(m, target.beta, n_sites),
            "energy_per_site_mean": observables.energy_per_site_mean(energies),
            "energy_per_site_variance":
                observables.energy_per_site_variance(energies),
            "coherence_mean": observables.coherence_mean(coherences),
            "two_point_correlation": correlation,
            "kink_density_mean": observables.kink_density_mean(kinks),
            "heat_capacity_per_site": observables.heat_capacity_per_site(
                energies * n_sites, target.beta, n_sites),
            "binder_cumulant": observables.binder_cumulant(m),
        }
    else:
        probabilities, outside = observables.phase_probabilities_weighted(
            labels, weights, n_phases)
        mean_m = observables.order_parameter_mean_weighted(m, weights)
        correlation = observables.two_point_correlation_mean_weighted(
            correlations, weights)
        record = {
            "phase_probabilities": probabilities,
            "phase_outside_mass": outside,
            "order_parameter_mean": mean_m,
            "susceptibility": observables.susceptibility_weighted(
                m, weights, target.beta, n_sites),
            "energy_per_site_mean":
                observables.energy_per_site_mean_weighted(energies, weights),
            "energy_per_site_variance":
                observables.energy_per_site_variance_weighted(energies, weights),
            "coherence_mean": observables.coherence_mean_weighted(
                coherences, weights),
            "two_point_correlation": correlation,
            "kink_density_mean": observables.kink_density_mean_weighted(
                kinks, weights),
            "heat_capacity_per_site": observables.heat_capacity_per_site_weighted(
                energies * n_sites, weights, target.beta, n_sites),
            "binder_cumulant": observables.binder_cumulant_weighted(m, weights),
        }
    record["connected_two_point_correlation"] = (
        observables.connected_two_point_correlation(correlation, mean_m))
    variance = float(record["energy_per_site_variance"])
    record["energy_per_site_sd"] = math.sqrt(max(variance, 0.0))
    if weights is None:
        coherence_variance = float(
            observables.energy_per_site_variance(coherences).item())
    else:
        coherence_variance = float(
            observables.energy_per_site_variance_weighted(
                coherences, weights).item())
    record["coherence_sd"] = math.sqrt(max(coherence_variance, 0.0))
    return {key: (float(value.item()) if isinstance(value, torch.Tensor)
                  and value.ndim == 0 else value)
            for key, value in record.items()}


def _as_numpy_stats(stats: Mapping) -> dict:
    out = {}
    for key, value in stats.items():
        if isinstance(value, torch.Tensor):
            out[key] = (float(value.item()) if value.ndim == 0
                        else value.detach().cpu().numpy().astype(float))
        elif isinstance(value, np.ndarray):
            out[key] = value.astype(float)
        else:
            out[key] = float(value)
    return out


# ==================================================================== SNIS
def _run_snis(target, settings: Mapping, *, mixture, basin_map, site_minima,
              layout: _Layout, n_phases: int, verbose: bool) -> dict:
    """Independent Laplace-mixture SNIS runs with directly weighted estimates."""
    device = target.device
    n_runs = check_positive_int(settings["n_runs"], "snis.n_runs")
    per_run = check_positive_int(settings["proposals_per_run"],
                                 "snis.proposals_per_run")
    seed_base = check_seed(settings["seed_base"], "snis.seed_base")
    seeds = [seed_base + index for index in range(n_runs)]

    proposals, log_weights, components = [], [], []
    for index, seed in enumerate(seeds):
        generator = frozen_generator(device, seed)
        x, component = mixture.sample(per_run, generator)
        log_q = mixture.log_q(x)
        log_w = -target.beta * target.value(x, cost_class="baseline") - log_q
        if not bool(torch.isfinite(log_w).all().item()):
            bad = int((~torch.isfinite(log_w)).sum().item())
            raise FloatingPointError(
                f"SNIS run {index} produced {bad}/{per_run} nonfinite log "
                "weights; the proposal covariance is degenerate")
        proposals.append(x)
        log_weights.append(log_w)
        components.append(component)
        _progress(verbose, f"SNIS run {index + 1}/{n_runs} drawn "
                           f"({per_run} proposals)")

    x = torch.cat(proposals)
    log_w = torch.cat(log_weights)
    component = torch.cat(components)
    run_id = np.repeat(np.arange(n_runs), per_run)
    per_sample = _per_sample_observables(target, x, basin_map=basin_map,
                                         site_minima=site_minima)
    features = _feature_matrix(
        layout, labels=per_sample["labels"],
        energies=per_sample["energy_per_site"],
        m=per_sample["order_parameter"], coherences=per_sample["coherence"],
        correlations=per_sample["two_point_correlation"],
        kinks=per_sample["kink_density"])

    pooled_weights = metrics.normalize_log_weights(log_w)
    shift = float(log_w.max().item())
    unnormalized = torch.exp(log_w - shift)
    run_index = torch.as_tensor(run_id, device=device)
    summaries = np.stack([
        _summary_row(features[run_index == index],
                     unnormalized[run_index == index])
        for index in range(n_runs)])
    if not np.all(summaries[:, 0] > 0.0):
        raise FloatingPointError(
            "at least one SNIS run underflowed to zero total weight; the runs "
            "are not on a common weight scale")
    return {
        "proposals": x,
        "log_weights": log_w,
        "weights": pooled_weights,
        "unnormalized_shift": shift,
        "unnormalized_weights": unnormalized,
        "component": component,
        "run_id": run_id.astype(np.int64),
        "per_sample": per_sample,
        "features": features,
        "summaries": summaries,
        "n_runs": n_runs,
        "proposals_per_run": per_run,
        "seeds": seeds,
    }


def _influence_estimates(features: torch.Tensor, weights: torch.Tensor,
                         layout: _Layout, *, beta: float, n_sites: int) -> dict:
    """SNIS point estimates with delta-method (influence-function) standard errors.

    ``se(theta) = sqrt(sum_i wbar_i^2 psi_i^2)`` with ``psi_i`` the influence of
    proposal ``i``. The draws inside one run are independent, so this is a
    legitimate within-run standard error; it is never used for the PT chains.
    """
    w = weights.reshape(-1).to(features.dtype)
    w = w / w.sum()
    w2 = w * w
    sums = (w.unsqueeze(1) * features).sum(dim=0)

    def se(influence: torch.Tensor) -> float:
        return float(torch.sqrt((w2 * influence * influence).sum()).item())

    inside = features[:, layout["inside"]]
    inside_mass = float(sums[layout["inside"]].item())
    probabilities, probability_ses = [], []
    for phase in range(layout.n_phases):
        indicator = features[:, layout[f"phase_{phase}"]]
        value = (0.0 if inside_mass <= 0.0
                 else float(sums[layout[f"phase_{phase}"]].item()) / inside_mass)
        probabilities.append(value)
        probability_ses.append(
            float("nan") if inside_mass <= 0.0
            else se((indicator - value * inside) / inside_mass))
    mean_e = float(sums[layout["e"]].item())
    mean_G = float(sums[layout["G"]].item())
    mean_mx = float(sums[layout["mx"]].item())
    mean_my = float(sums[layout["my"]].item())
    reliability = float((1.0 - w2.sum()).item())
    centered_x = features[:, layout["mx"]] - mean_mx
    centered_y = features[:, layout["my"]] - mean_my
    products = {
        (0, 0): centered_x * centered_x,
        (0, 1): centered_x * centered_y,
        (1, 1): centered_y * centered_y,
    }
    covariance = np.zeros((2, 2), dtype=float)
    susceptibility_se = np.zeros((2, 2), dtype=float)
    scale = beta * n_sites
    for (row, column), product in products.items():
        value = float((w * product).sum().item()) / reliability
        error = se(scale * (product - value * reliability) / reliability)
        covariance[row, column] = covariance[column, row] = value
        susceptibility_se[row, column] = susceptibility_se[column, row] = error
    return {
        "phase_probabilities": np.asarray(probabilities, dtype=float),
        "phase_probabilities_se": np.asarray(probability_ses, dtype=float),
        "energy_per_site_mean": mean_e,
        "energy_per_site_mean_se": se(features[:, layout["e"]] - mean_e),
        "coherence_mean": mean_G,
        "coherence_mean_se": se(features[:, layout["G"]] - mean_G),
        "order_parameter_mean": np.asarray([mean_mx, mean_my], dtype=float),
        "order_parameter_mean_se": np.asarray(
            [se(centered_x), se(centered_y)], dtype=float),
        "susceptibility": scale * covariance,
        "susceptibility_se": susceptibility_se,
    }


# ====================================================== PT chain diagnostics
def _chain_series(values: torch.Tensor, n_chains: int,
                  n_checkpoints: int) -> np.ndarray:
    """Reshape a chain-major per-sample vector into ``(n_chains, n_checkpoints)``."""
    return (values.detach().cpu().numpy().astype(float)
            .reshape(n_chains, n_checkpoints))


def _observable_series(per_sample: Mapping, labels: np.ndarray, *,
                       continuous: Sequence[str], indicators: Sequence[str],
                       phases: Sequence[str], n_chains: int,
                       n_checkpoints: int) -> dict:
    """The observable set F of the acceptance file, as per-chain series."""
    sources = {
        "energy_per_site": per_sample["energy_per_site"],
        "mx": per_sample["order_parameter"][:, 0],
        "my": per_sample["order_parameter"][:, 1],
        "coherence": per_sample["coherence"],
        "kink_density": per_sample["kink_density"],
    }
    series = {}
    for name in continuous:
        if name not in sources:
            raise KeyError(
                f"acceptance file lists continuous observable {name!r}, which "
                f"the reference does not provide (known: {sorted(sources)})")
        series[name] = _chain_series(sources[name], n_chains, n_checkpoints)
    label_series = labels.reshape(n_chains, n_checkpoints)
    for name in indicators:
        phase_name = name.split("phase_", 1)[-1]
        if phase_name not in phases:
            raise KeyError(
                f"acceptance file lists indicator {name!r}, which names no "
                f"phase in {list(phases)}")
        index = list(phases).index(phase_name)
        series[name] = (label_series == index).astype(float)
    return series


def _entry_events(label_series: np.ndarray, n_phases: int,
                  min_consecutive: int) -> np.ndarray:
    """Entry events per phase, aggregated over the cold chains.

    An entry event into phase ``s`` requires the label to CHANGE to ``s`` and
    then persist for at least ``min_consecutive`` consecutive saved
    checkpoints. Maximal constant stretches are used, so an oscillation that
    never settles contributes nothing.
    """
    counts = np.zeros(n_phases, dtype=np.int64)
    for chain in label_series:
        if chain.size == 0:
            continue
        boundaries = np.flatnonzero(np.diff(chain)) + 1
        starts = np.concatenate([[0], boundaries])
        stops = np.concatenate([boundaries, [chain.size]])
        for position, (start, stop) in enumerate(zip(starts, stops)):
            label = int(chain[start])
            if position == 0 or label == OUTSIDE_LABEL:
                continue
            if (stop - start) >= min_consecutive:
                counts[label] += 1
    return counts


def _pooled_block_mcse(series: np.ndarray, block_length: int) -> float:
    """Batch-means MCSE of the pooled mean over equally long independent chains.

    Each chain's own batch-means standard error comes from
    :func:`metrics.block_mcse`; the pooled mean is the average of the chain
    means, so its variance is the average of theirs divided by the chain count.
    Chains are never concatenated: a block must never straddle two chains.
    """
    errors = []
    for chain in series:
        if chain.size // int(block_length) < 2:
            return float("nan")
        errors.append(metrics.block_mcse(chain, int(block_length)))
    values = np.asarray(errors, dtype=float)
    if not np.all(np.isfinite(values)):
        return float("nan")
    return float(math.sqrt(float((values ** 2).sum())) / values.size)


def _block_units(n_chains: int, n_checkpoints: int,
                 block_length: int) -> np.ndarray:
    """Sample-index blocks: contiguous runs of ``block_length`` inside one chain."""
    per_chain = n_checkpoints // int(block_length)
    if per_chain < 1:
        return np.zeros((0, int(block_length)), dtype=np.int64)
    offsets = (np.arange(n_chains)[:, None] * n_checkpoints
               + np.arange(per_chain)[None, :] * int(block_length))
    inner = np.arange(int(block_length))[None, None, :]
    return (offsets[:, :, None] + inner).reshape(-1, int(block_length))


def _block_summaries(features: torch.Tensor, blocks: np.ndarray) -> np.ndarray:
    """One unweighted summary row per non-overlapping block."""
    if blocks.shape[0] == 0:
        return np.zeros((0, features.shape[1] + 2), dtype=float)
    index = torch.as_tensor(blocks.reshape(-1), device=features.device)
    grouped = features[index].reshape(blocks.shape[0], blocks.shape[1],
                                      features.shape[1])
    sums = grouped.sum(dim=1).detach().cpu().numpy().astype(float)
    mass = np.full((blocks.shape[0], 1), float(blocks.shape[1]))
    return np.concatenate([mass, mass, sums], axis=1)


def _bootstrap_statistics(summaries: np.ndarray, layout: _Layout, *,
                          beta: float, n_sites: int, replicates: int,
                          seed: int, template: Mapping) -> dict:
    """Block-bootstrap standard errors for every derived statistic at once.

    :func:`metrics.hierarchical_bootstrap` drives the resampling so the frozen
    seeding and unit semantics are shared with the cross-check bootstrap; each
    replicate recomputes the *whole* statistic vector from the resampled block
    sums, and the returned per-replicate scalar is unused.
    """
    collected: list[np.ndarray] = []
    unit_ids = list(range(summaries.shape[0]))

    def statistic(resampled: Mapping) -> float:
        rows = summaries[np.asarray(resampled["pt_blocks"], dtype=np.int64)]
        collected.append(_stats_vector(
            _derive(rows.sum(axis=0), layout, beta=beta, n_sites=n_sites)))
        return 0.0

    metrics.hierarchical_bootstrap(statistic, {"pt_blocks": unit_ids},
                                   int(replicates), int(seed))
    draws = np.asarray(collected, dtype=float)
    return _unflatten_like(draws.std(axis=0, ddof=1), template)


# ============================================================== PT-MALA gates
def _pt_gates(acceptance: Mapping, *, pt, series: Mapping, label_series,
              phases: Sequence[str], continuous: Sequence[str],
              indicators: Sequence[str], block_length: int,
              point: Mapping, standard_errors: Mapping,
              bootstrap: Mapping) -> list[dict]:
    gates = acceptance["pt_mala_gates"]
    uncertainty = acceptance["uncertainty"]
    n_phases = len(phases)
    n_chains = int(pt["n_chains"])
    n_checkpoints = int(pt["n_checkpoints"])
    records: list[dict] = []

    records.append(_gate(
        "pt_independent_runs", gates["min_independent_runs"], pt["n_runs"],
        direction="min",
        message=(f"{pt['n_runs']} independent long runs, "
                 f"{pt['chains_per_run']} chains each, seeds {pt['seeds']}")))

    if bool(gates.get("init_from_each_refined_phase", True)):
        distinct = len(set(pt["phase_indices"]))
        records.append(_gate(
            "pt_init_from_each_refined_phase", n_phases, distinct,
            direction="min",
            message=("runs initialised from refined homogeneous phases "
                     f"{pt['init_phases']} with jitter sigma "
                     f"{pt['init_sigma']}")))

    visited = np.asarray([
        np.unique(chain[chain != OUTSIDE_LABEL]).size for chain in label_series])
    if bool(gates.get("every_cold_chain_visits_all_phases", True)):
        records.append(_gate(
            "pt_every_cold_chain_visits_all_phases", n_phases,
            int(visited.min()) if visited.size else 0, direction="min",
            message=(f"minimum over {n_chains} cold chains of the number of "
                     f"distinct phases visited after burn-in; per-chain counts "
                     f"{visited.tolist()}")))

    min_consecutive = int(gates["entry_event_min_consecutive_checkpoints"])
    entries = _entry_events(label_series, n_phases, min_consecutive)
    for index, name in enumerate(phases):
        records.append(_gate(
            f"pt_entry_events_phase_{name}",
            gates["min_entry_events_per_phase"], int(entries[index]),
            direction="min",
            message=("entry events into this phase in the aggregated cold "
                     f"chains; an entry requires a label change persisting for "
                     f">= {min_consecutive} consecutive saved checkpoints")))

    for name in list(continuous) + list(indicators):
        records.append(_gate(
            f"pt_split_rhat_{name}", gates["max_split_rhat"],
            metrics.split_rhat(series[name]), direction="max",
            message=(f"{gates['rhat_kind']} split R-hat over {n_chains} cold "
                     f"chains of {n_checkpoints} saved checkpoints")))

    for name in continuous:
        records.append(_gate(
            f"pt_bulk_ess_{name}", gates["min_bulk_ess"],
            metrics.bulk_ess(series[name]), direction="min",
            message=f"rank-normalized bulk ESS pooled over {n_chains} chains"))
        records.append(_gate(
            f"pt_tail_ess_{name}", gates["min_tail_ess"],
            metrics.tail_ess(series[name]), direction="min",
            message=f"tail ESS (min of the 5% and 95% quantile indicators)"))
    for name in indicators:
        records.append(_gate(
            f"pt_phase_indicator_ess_{name}",
            gates["min_phase_indicator_ess"], metrics.bulk_ess(series[name]),
            direction="min",
            message=f"bulk ESS of the phase indicator over {n_chains} chains"))

    for name in continuous:
        mcse = _pooled_block_mcse(series[name], block_length)
        pooled = series[name].reshape(-1)
        sd = float(pooled.std(ddof=1)) if pooled.size > 1 else float("nan")
        fraction = (float("nan") if not math.isfinite(sd) or sd <= 0.0
                    else mcse / sd)
        records.append(_gate(
            f"pt_block_mcse_fraction_{name}",
            gates["max_block_mcse_fraction_of_sd"], fraction, direction="max",
            standard_error=mcse, block_length=block_length,
            message=(f"batch-means MCSE {mcse:.6g} as a fraction of the "
                     f"reference standard deviation {sd:.6g}; "
                     f"{uncertainty['block_rule']} blocks")))

    chains_per_run = int(pt["chains_per_run"])
    half = n_checkpoints // 2
    for run in range(int(pt["n_runs"])):
        rows = slice(run * chains_per_run, (run + 1) * chains_per_run)
        for name in continuous:
            first = series[name][rows, :half]
            second = series[name][rows, n_checkpoints - half:]
            mean_first, mean_second = float(first.mean()), float(second.mean())
            se_first = _pooled_block_mcse(first, block_length)
            se_second = _pooled_block_mcse(second, block_length)
            combined = math.sqrt(se_first ** 2 + se_second ** 2) \
                if math.isfinite(se_first) and math.isfinite(se_second) \
                else float("nan")
            ratio = (float("nan") if not math.isfinite(combined) or combined <= 0.0
                     else abs(mean_first - mean_second) / combined)
            records.append(_gate(
                f"pt_half_run_consistency_{name}_run{run}",
                gates["max_half_run_difference_in_combined_se"], ratio,
                direction="max", standard_error=combined,
                block_length=block_length,
                message=(f"run {run}: first half {mean_first:.6g} vs second "
                         f"half {mean_second:.6g}, difference in combined "
                         f"batch-means standard errors")))
    return records


def _block_length_gate(acceptance: Mapping, *, block_length: int,
                       n_blocks_per_chain: int, n_blocks_total: int,
                       max_tau: float, n_checkpoints: int,
                       required_checkpoints: int) -> dict:
    uncertainty = acceptance["uncertainty"]
    minimum = int(uncertainty["min_effective_blocks"])
    passed = n_blocks_per_chain >= minimum
    message = (
        f"L_block = ceil({uncertainty['block_length_multiplier']} * "
        f"max_f tau_int(f)) = {block_length} with max_f tau_int(f) = "
        f"{max_tau:.4g} saved checkpoints; {n_blocks_per_chain} "
        f"{uncertainty['block_rule']} blocks per chain "
        f"({n_blocks_total} in total)")
    if not passed:
        message += (
            f". The block length must NOT be shrunk to satisfy this gate: the "
            f"PT reference run must be EXTENDED to at least "
            f"{required_checkpoints} saved checkpoints per chain "
            f"(currently {n_checkpoints}), i.e. total_steps must grow to about "
            f"burn_in_steps + {required_checkpoints} * thinning")
    return _gate("block_length_effective_blocks", minimum, n_blocks_per_chain,
                 direction="min", block_length=block_length, message=message,
                 max_integrated_autocorrelation_time=_finite(max_tau),
                 n_blocks_total=int(n_blocks_total),
                 n_checkpoints_per_chain=int(n_checkpoints))


# ================================================================ SNIS gates
def _snis_gates(acceptance: Mapping, *, snis, layout: _Layout,
                phases: Sequence[str], beta: float, n_sites: int,
                coherence_decile: float) -> tuple[list[dict], dict]:
    gates = acceptance["snis_gates"]
    coverage_required = gates["require_coverage"]
    weights = snis["weights"]
    features = snis["features"]
    labels = snis["per_sample"]["labels"]
    n_proposals = int(weights.numel())
    records: list[dict] = []

    records.append(_gate(
        "snis_independent_runs", gates["min_independent_runs"], snis["n_runs"],
        direction="min",
        message=(f"{snis['n_runs']} independent proposal runs of "
                 f"{snis['proposals_per_run']} draws, seeds {snis['seeds']}")))

    ess = metrics.importance_sampling_ess(weights)
    fraction = ess / n_proposals
    max_weight = float(weights.max().item())
    records.append(_gate("snis_total_ess", gates["min_total_ess"], ess,
                         direction="min",
                         message=(f"Kish ESS of the pooled normalized weights "
                                  f"over {n_proposals} proposals")))
    records.append(_gate("snis_ess_fraction", gates["min_ess_fraction"],
                         fraction, direction="min",
                         message=f"ESS / n_proposals = {ess:.6g} / {n_proposals}"))
    records.append(_gate("snis_max_normalized_weight",
                         gates["max_normalized_weight"], max_weight,
                         direction="max",
                         message="largest pooled normalized weight"))

    effective_counts = []
    for index, name in enumerate(phases):
        mask = labels == index
        count = metrics.weighted_effective_count(weights, mask)
        effective_counts.append(count)
        records.append(_gate(
            f"snis_weighted_effective_count_phase_{name}",
            gates["min_weighted_effective_count_per_phase"], count,
            direction="min",
            message=("N_eff,s = (sum_{i in s} wbar_i)^2 / sum_{i in s} wbar_i^2 "
                     "via metrics.weighted_effective_count")))

    proposal_counts = np.asarray(
        [int((labels == index).sum().item()) for index in range(len(phases))])
    if bool(coverage_required.get("all_four_phases", True)):
        records.append(_gate(
            "snis_coverage_all_phases", 1, int(proposal_counts.min()),
            direction="min",
            message=(f"smallest proposal count over the {len(phases)} phases; "
                     f"per-phase proposal counts {proposal_counts.tolist()}")))
    kink_covered = int((features[:, layout["kink"]] > 0.0).sum().item())
    if bool(coverage_required.get("nonzero_kink_configurations", True)):
        records.append(_gate(
            "snis_coverage_nonzero_kink", 1, kink_covered, direction="min",
            message="proposals carrying at least one kinked neighbour pair"))
    decile_covered = int(
        (features[:, layout["G"]] >= coherence_decile).sum().item())
    if bool(coverage_required.get("coherence_upper_decile", True)):
        records.append(_gate(
            "snis_coverage_coherence_upper_decile", 1, decile_covered,
            direction="min",
            message=(f"proposals with coherence G >= {coherence_decile:.6g}, "
                     "the 90th percentile of the PT-MALA coherence")))

    run_index = torch.as_tensor(snis["run_id"], device=features.device)
    per_run = []
    for index in range(int(snis["n_runs"])):
        mask = run_index == index
        per_run.append(_influence_estimates(
            features[mask],
            metrics.normalize_log_weights(snis["log_weights"][mask]),
            layout, beta=beta, n_sites=n_sites))

    multiplier = float(gates["max_run_difference_in_combined_se"])

    def agreement(name: str, extract, error, message: str) -> None:
        worst, detail = 0.0, []
        for first in range(len(per_run)):
            detail.append(float(np.max(np.abs(np.atleast_1d(
                np.asarray(extract(per_run[first]), dtype=float))))))
            for second in range(first + 1, len(per_run)):
                a = np.atleast_1d(np.asarray(extract(per_run[first]),
                                             dtype=float)).reshape(-1)
                b = np.atleast_1d(np.asarray(extract(per_run[second]),
                                             dtype=float)).reshape(-1)
                sa = np.atleast_1d(np.asarray(error(per_run[first]),
                                              dtype=float)).reshape(-1)
                sb = np.atleast_1d(np.asarray(error(per_run[second]),
                                              dtype=float)).reshape(-1)
                combined = np.sqrt(sa ** 2 + sb ** 2)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(combined > 0.0,
                                     np.abs(a - b) / combined, np.inf)
                worst = max(worst, float(np.nanmax(ratio)))
        records.append(_gate(
            f"snis_run_agreement_{name}", multiplier, worst, direction="max",
            message=message))

    agreement("phase_probability",
              lambda run: run["phase_probabilities"],
              lambda run: run["phase_probabilities_se"],
              "largest standardized pairwise difference between independent "
              "SNIS runs over the four phase probabilities, delta-method SEs")
    agreement("energy_per_site",
              lambda run: run["energy_per_site_mean"],
              lambda run: run["energy_per_site_mean_se"],
              "largest standardized pairwise difference between independent "
              "SNIS runs on the mean energy per site")
    agreement("susceptibility",
              lambda run: run["susceptibility"],
              lambda run: run["susceptibility_se"],
              "largest standardized pairwise difference between independent "
              "SNIS runs over the four susceptibility matrix entries "
              "(elementwise delta-method SEs; the cross-check statistic itself "
              "uses a whole-statistic hierarchical bootstrap)")
    agreement("coherence_mean",
              lambda run: run["coherence_mean"],
              lambda run: run["coherence_mean_se"],
              "largest standardized pairwise difference between independent "
              "SNIS runs on the mean coherence")

    diagnostics = {
        "n_proposals": n_proposals,
        "importance_sampling_ess": ess,
        "ess_fraction": fraction,
        "max_normalized_weight": max_weight,
        "weighted_effective_count_per_phase": effective_counts,
        "proposal_phase_counts": proposal_counts.tolist(),
        "proposal_phase_fractions": (proposal_counts / n_proposals).tolist(),
        "coverage_nonzero_kink_count": kink_covered,
        "coverage_coherence_upper_decile_count": decile_covered,
        "coherence_upper_decile_threshold": float(coherence_decile),
        "per_run": [{key: value for key, value in _as_numpy_stats(run).items()}
                    for run in per_run],
    }
    return records, diagnostics


# ========================================================== cross-check gates
def _cross_check_gates(acceptance: Mapping, *, phases: Sequence[str],
                       pt_point: Mapping, pt_se: Mapping,
                       snis_point: Mapping, snis_influence: Mapping,
                       relative_susceptibility: float,
                       relative_susceptibility_se: float,
                       relative_correlation: float,
                       relative_correlation_se: float,
                       bootstrap: Mapping, block_length: int) -> list[dict]:
    gates = acceptance["cross_check_gates"]
    definition = (
        "hierarchical bootstrap over "
        f"{list(bootstrap['resample_units'])}: each replicate resamples the "
        "independent PT blocks and the independent SNIS runs with replacement "
        "and recomputes the whole relative statistic")
    records: list[dict] = []

    phase_gate = gates["phase_probability"]
    for index, name in enumerate(phases):
        pt_value = float(np.asarray(pt_point["phase_probabilities"])[index])
        snis_value = float(np.asarray(snis_point["phase_probabilities"])[index])
        se = math.sqrt(
            float(np.asarray(pt_se["phase_probabilities"])[index]) ** 2
            + float(np.asarray(
                snis_influence["phase_probabilities_se"])[index]) ** 2)
        records.append(_tolerance_gate(
            f"cross_check_phase_probability_{name}",
            pt_value - snis_value, floor=float(phase_gate["absolute_floor"]),
            multiplier=float(phase_gate["combined_se_multiplier"]),
            combined_se=se, block_length=block_length,
            bootstrap_definition=None,
            message=(f"PT {pt_value:.6g} vs SNIS {snis_value:.6g}; PT SE from "
                     "the block bootstrap, SNIS SE from the delta method"),
            pt_value=pt_value, snis_value=snis_value))

    energy_gate = gates["energy_per_site"]
    pt_energy = float(pt_point["energy_per_site_mean"])
    snis_energy = float(snis_point["energy_per_site_mean"])
    scale = max(abs(pt_energy), float(pt_point["energy_per_site_sd"]), 1e-12)
    se = math.sqrt(float(pt_se["energy_per_site_mean"]) ** 2
                   + float(snis_influence["energy_per_site_mean_se"]) ** 2)
    records.append(_tolerance_gate(
        "cross_check_energy_per_site", pt_energy - snis_energy,
        floor=float(energy_gate["relative_floor"]) * scale,
        multiplier=float(energy_gate["combined_se_multiplier"]),
        combined_se=se, block_length=block_length,
        message=(f"PT {pt_energy:.6g} vs SNIS {snis_energy:.6g}; floor is "
                 f"{energy_gate['relative_floor']} * "
                 f"max(|e|, sigma_e, 1e-12) = {scale:.6g} "
                 f"({energy_gate['scale_rule']})"),
        scale=scale, pt_value=pt_energy, snis_value=snis_energy))

    magnetization_gate = gates["magnetization"]
    for index, component in enumerate(("mx", "my")):
        pt_value = float(np.asarray(pt_point["order_parameter_mean"])[index])
        snis_value = float(np.asarray(snis_point["order_parameter_mean"])[index])
        se = math.sqrt(
            float(np.asarray(pt_se["order_parameter_mean"])[index]) ** 2
            + float(np.asarray(
                snis_influence["order_parameter_mean_se"])[index]) ** 2)
        records.append(_tolerance_gate(
            f"cross_check_magnetization_{component}", pt_value - snis_value,
            floor=float(magnetization_gate["absolute_floor"]),
            multiplier=float(magnetization_gate["combined_se_multiplier"]),
            combined_se=se, block_length=block_length,
            message=f"PT <{component}> {pt_value:.6g} vs SNIS {snis_value:.6g}",
            pt_value=pt_value, snis_value=snis_value))

    susceptibility_gate = gates["susceptibility"]
    records.append(_tolerance_gate(
        "cross_check_susceptibility_relative_frobenius",
        relative_susceptibility,
        floor=float(susceptibility_gate["relative_frobenius_floor"]),
        multiplier=float(susceptibility_gate["combined_se_multiplier"]),
        combined_se=relative_susceptibility_se,
        bootstrap_definition=definition,
        bootstrap_seed=int(bootstrap["seed"]),
        bootstrap_replicates=int(bootstrap["replicates"]),
        message=("||chi_PT - chi_SNIS||_F / ||chi_PT||_F; the standard error "
                 "is a hierarchical bootstrap of the whole statistic, never "
                 "assembled from elementwise standard errors")))

    coherence_gate = gates["coherence_mean"]
    pt_coherence = float(pt_point["coherence_mean"])
    snis_coherence = float(snis_point["coherence_mean"])
    coherence_scale = max(abs(pt_coherence), float(pt_point["coherence_sd"]),
                          1e-12)
    se = math.sqrt(float(pt_se["coherence_mean"]) ** 2
                   + float(snis_influence["coherence_mean_se"]) ** 2)
    records.append(_tolerance_gate(
        "cross_check_coherence_mean", pt_coherence - snis_coherence,
        floor=float(coherence_gate["relative_floor"]) * coherence_scale,
        multiplier=float(coherence_gate["combined_se_multiplier"]),
        combined_se=se, block_length=block_length,
        message=(f"PT {pt_coherence:.6g} vs SNIS {snis_coherence:.6g}; floor "
                 f"is {coherence_gate['relative_floor']} * "
                 f"max(|G|, sigma_G, 1e-12) = {coherence_scale:.6g} "
                 f"({coherence_gate['scale_rule']})"),
        scale=coherence_scale, pt_value=pt_coherence, snis_value=snis_coherence))

    correlation_gate = gates["two_point_correlation"]
    records.append(_tolerance_gate(
        "cross_check_two_point_correlation_relative_l2", relative_correlation,
        floor=float(correlation_gate["relative_l2_floor"]),
        multiplier=float(correlation_gate["combined_se_multiplier"]),
        combined_se=relative_correlation_se,
        bootstrap_definition=definition,
        bootstrap_seed=int(bootstrap["seed"]),
        bootstrap_replicates=int(bootstrap["replicates"]),
        message=("sqrt(sum_r (C_PT(r) - C_SNIS(r))^2 / sum_r C_PT(r)^2); the "
                 "standard error is a hierarchical bootstrap of the whole "
                 "statistic")))
    return records


# ================================================== whole-statistic bootstrap
def _cross_check_bootstrap(pt_summaries: np.ndarray, snis_summaries: np.ndarray,
                           layout: _Layout, *, beta: float, n_sites: int,
                           replicates: int, seed: int) -> dict:
    """Hierarchical bootstrap of the two relative cross-check statistics.

    Each replicate resamples the independent PT blocks and the independent SNIS
    runs with replacement, rebuilds both summary rows, and recomputes
    ``||chi_PT - chi_SNIS||_F / ||chi_PT||_F`` and the relative correlation L2
    from scratch. Nothing is assembled from elementwise standard errors.
    """
    correlation_draws: list[float] = []

    def statistic(resampled: Mapping) -> float:
        pt_row = pt_summaries[
            np.asarray(resampled["pt_blocks"], dtype=np.int64)].sum(axis=0)
        snis_row = snis_summaries[
            np.asarray(resampled["snis_runs"], dtype=np.int64)].sum(axis=0)
        pt_stats = _derive(pt_row, layout, beta=beta, n_sites=n_sites)
        snis_stats = _derive(snis_row, layout, beta=beta, n_sites=n_sites)
        correlation_draws.append(_relative_l2(
            snis_stats["two_point_correlation"],
            pt_stats["two_point_correlation"]))
        return _relative_frobenius(snis_stats["susceptibility"],
                                   pt_stats["susceptibility"])

    draws = metrics.hierarchical_bootstrap(
        statistic,
        {"pt_blocks": list(range(pt_summaries.shape[0])),
         "snis_runs": list(range(snis_summaries.shape[0]))},
        int(replicates), int(seed))
    correlation = np.asarray(correlation_draws, dtype=float)
    return {
        "susceptibility_relative_frobenius_se": float(draws.std(ddof=1)),
        "two_point_correlation_relative_l2_se": float(correlation.std(ddof=1)),
        "susceptibility_relative_frobenius_replicates": draws,
        "two_point_correlation_relative_l2_replicates": correlation,
    }


def _relative_frobenius(estimate, reference) -> float:
    return observables.relative_frobenius_error(
        torch.as_tensor(np.asarray(estimate, dtype=float)),
        torch.as_tensor(np.asarray(reference, dtype=float)))


def _relative_l2(estimate, reference) -> float:
    return observables.correlation_relative_l2(
        torch.as_tensor(np.asarray(estimate, dtype=float)),
        torch.as_tensor(np.asarray(reference, dtype=float)))


# =================================================== order-parameter density
def _order_parameter_grid(m: torch.Tensor, *, beta: float, settings: Mapping,
                          n_chains: int, n_checkpoints: int) -> dict:
    """``p*(m_x, m_y)`` and ``F*`` by projecting the PT-MALA bank onto the plane.

    This is a NUMERICAL reference estimate, not an analytic grid: the density is
    the binned empirical measure of the frozen PT sample bank, so it carries
    that bank's sampling error. The per-cell uncertainty is estimated ACROSS
    INDEPENDENT COLD CHAINS -- never by treating one chain's correlated draws as
    i.i.d. replicates.
    """
    lo, hi = float(settings["lo"]), float(settings["hi"])
    n_bins = check_positive_int(settings["n_bins"], "order_parameter_grid.n_bins")
    if not hi > lo:
        raise ValueError("order_parameter_grid.hi must exceed .lo")
    device = m.device
    edges = torch.linspace(lo, hi, n_bins + 1, dtype=torch.float64, device=device)
    centers = 0.5 * (edges[1:] + edges[:-1])
    width = float((edges[1] - edges[0]).item())
    cell_area = width * width

    inside = ((m >= lo) & (m <= hi)).all(dim=1)
    index = torch.clamp(((m - lo) / width).to(torch.long), 0, n_bins - 1)
    flat = (index[:, 0] * n_bins + index[:, 1])[inside]
    counts = torch.bincount(flat, minlength=n_bins * n_bins).reshape(n_bins, n_bins)
    n_inside = int(inside.sum().item())
    if n_inside < 1:
        raise ValueError("no PT sample lands inside the order-parameter grid")
    density = counts.to(torch.float64) / (float(n_inside) * cell_area)

    chain_index = (torch.arange(m.shape[0], device=device) // n_checkpoints)
    per_chain = torch.zeros(n_chains, n_bins * n_bins, dtype=torch.float64,
                            device=device)
    for chain in range(n_chains):
        mask = inside & (chain_index == chain)
        total = int(mask.sum().item())
        if total:
            per_chain[chain] = torch.bincount(
                (index[:, 0] * n_bins + index[:, 1])[mask],
                minlength=n_bins * n_bins).to(torch.float64) / (total * cell_area)
    standard_error = (per_chain.std(dim=0, unbiased=True)
                      / math.sqrt(n_chains)).reshape(n_bins, n_bins)

    positive = density > 0.0
    free_energy = torch.full_like(density, float("nan"))
    free_energy[positive] = -(1.0 / beta) * torch.log(density[positive])
    offset = float(free_energy[positive].min().item())
    free_energy = free_energy - offset

    sigma = float(m[inside].std(dim=0, unbiased=True).mean().item())
    bandwidth = max(sigma * float(n_inside) ** (-1.0 / 6.0), 1e-6)
    kde = metrics.kde_on_grid_2d(m[inside], centers, centers, bandwidth)
    return {
        "m_x_edges": edges, "m_y_edges": edges,
        "m_x_centers": centers, "m_y_centers": centers,
        "counts": counts, "p_star": density, "p_star_standard_error":
            standard_error, "free_energy": free_energy,
        "free_energy_offset": offset, "kde_p_star": kde,
        "kde_bandwidth": bandwidth, "cell_area": cell_area,
        "n_samples": int(m.shape[0]), "n_samples_inside": n_inside,
        "outside_grid_fraction": 1.0 - n_inside / float(m.shape[0]),
        "binning_rule": (f"uniform {n_bins} x {n_bins} histogram on "
                         f"[{lo}, {hi}]^2, normalised to a density over the "
                         "grid domain"),
        "kde_bandwidth_rule": str(settings["kde_bandwidth_rule"]),
        "uncertainty_rule": ("per-cell standard error across the "
                             f"{n_chains} independent cold chains"),
        "estimate_kind": "numerical_estimate_from_pt_mala_samples",
        "note": ("This is a numerical reference estimate obtained by "
                 "projecting the PT-MALA sample bank onto the order-parameter "
                 "plane. It is NOT an analytic grid and carries the sampling "
                 "uncertainty of that bank."),
    }


# =================================================================== the class
class CoupledQuarticChainReference(Reference):
    """The frozen E4 ground truth: a validated PT-MALA bank plus a SNIS cross-check.

    Every number this object exposes is a numerical estimate, not an analytic
    result. The 24-D bank is the cold-replica output of multi-start
    parallel-tempered MALA; the order-parameter density is that bank projected
    onto ``(m_x, m_y)`` and binned; the SNIS arm is a *weighted* cross-check
    whose weights must be used directly rather than resampled into a second
    unweighted bank.
    """

    kind = "multistart_pt_mala_with_snis_crosscheck"
    experiment_id = "E4"

    def __init__(self, *, target, phases, sample_bank, order_parameter_bank,
                 run_id, chain_id, iteration_step, checkpoint_index, pt, snis,
                 grid, basin_map, sw2_projections, mmd_bandwidth,
                 observable_targets, validation_records, reference_validated,
                 acceptance, acceptance_path, provenance, extras=None,
                 describe_cache=None) -> None:
        self.target = target
        self.device = target.device
        self.beta = float(target.beta)
        self.n_sites = int(target.potential.n_sites)
        self.phases = list(phases)
        self.n_phases = len(self.phases)
        self.sample_bank = sample_bank
        self.order_parameter_bank = order_parameter_bank
        self.run_id = np.asarray(run_id, dtype=np.int64)
        self.chain_id = np.asarray(chain_id, dtype=np.int64)
        self.iteration_step = np.asarray(iteration_step, dtype=np.int64)
        self.checkpoint_index = np.asarray(checkpoint_index, dtype=np.int64)
        self.pt = dict(pt)
        self.snis = dict(snis)
        self.grid = dict(grid)
        self.basin_map = basin_map
        self.sw2_projections = sw2_projections
        self.mmd_bandwidth = float(mmd_bandwidth)
        self.observable_targets = dict(observable_targets)
        self.validation_records = [dict(record) for record in validation_records]
        self.reference_validated = bool(reference_validated)
        self.acceptance = copy.deepcopy(dict(acceptance))
        self.acceptance_path = str(acceptance_path)
        self._provenance = copy.deepcopy(dict(provenance))
        self.extras = dict(extras or {})
        self._describe_cache = (None if describe_cache is None
                                else copy.deepcopy(dict(describe_cache)))
        self.last_sample_record: dict | None = None

    # -- identity ----------------------------------------------------------
    @property
    def provenance(self) -> dict:
        """The configuration-determined identity, without touching ``describe``.

        Overridden so the hash can be embedded in ``describe()`` itself without
        the base-class round trip recursing.
        """
        return copy.deepcopy(self._provenance)

    @property
    def failed_gates(self) -> list[dict]:
        return [record for record in self.validation_records
                if not record["passed"]]

    def describe(self) -> dict:
        if self._describe_cache is not None:
            return copy.deepcopy(self._describe_cache)
        failed = self.failed_gates
        payload = {
            "experiment_id": self.experiment_id,
            "kind": self.kind,
            "reference_validated": self.reference_validated,
            "provenance": self._provenance,
            "sample_bank": {
                "n_samples": int(self.sample_bank.shape[0]),
                "dimension": int(self.sample_bank.shape[1]),
                "source": "pt_mala_cold_replica_unweighted",
                "sampling_policy": (
                    "sample(n) draws without replacement when n <= bank size "
                    "and with replacement otherwise; the choice is recorded in "
                    "the returned draw record"),
            },
            "pt_mala": self.pt,
            "snis": self.snis,
            "order_parameter_grid": {
                key: value for key, value in self.grid.items()
                if key not in ("m_x_edges", "m_y_edges", "m_x_centers",
                               "m_y_centers", "counts", "p_star",
                               "p_star_standard_error", "free_energy",
                               "kde_p_star")},
            "basin_map": self.basin_map.cache_provenance(),
            "metric_inputs": {
                "mmd_bandwidth": self.mmd_bandwidth,
                "mmd_bandwidth_rule": "median_heuristic_on_reference_bank",
                "sw2_projections": {
                    "dimension": int(self.sw2_projections.shape[1]),
                    "n_projections": int(self.sw2_projections.shape[0]),
                },
            },
            "observable_targets": self.observable_targets,
            "validation": {
                "acceptance_file": self.acceptance_path,
                "n_gates": len(self.validation_records),
                "n_failed": len(failed),
                "failed_gates": [record["metric"] for record in failed],
            },
            "notes": [
                "The order-parameter grid is a numerical estimate obtained by "
                "projecting the PT-MALA bank, not an analytic density.",
                "The SNIS arm is a weighted cross-check; its weights are used "
                "directly and it is not a second unweighted sample bank.",
                "Reference uncertainty is estimated by independent chain or "
                "non-overlapping block; correlated draws from one cold chain "
                "are never treated as i.i.d. replicates.",
            ],
        }
        payload["provenance_hash"] = self.provenance_hash
        return payload

    # -- persistence -------------------------------------------------------
    def save(self, directory) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_npz(directory / SAMPLES_FILE,
                 configurations=self.sample_bank,
                 order_parameter=self.order_parameter_bank,
                 run_id=self.run_id, chain_id=self.chain_id,
                 iteration_step=self.iteration_step,
                 checkpoint_index=self.checkpoint_index,
                 phase_label=self.extras["pt_labels"],
                 energy_per_site=self.extras["pt_energy_per_site"],
                 coherence=self.extras["pt_coherence"],
                 kink_density=self.extras["pt_kink_density"],
                 two_point_correlation=self.extras["pt_two_point_correlation"],
                 betas=np.asarray(self.pt["betas"], dtype=float),
                 mala_acceptance=np.asarray(self.pt["mala_acceptance"],
                                            dtype=float),
                 swap_acceptance=np.asarray(self.pt["swap_acceptance"],
                                            dtype=float),
                 block_length=np.asarray(self.pt["block_length"],
                                         dtype=np.int64),
                 n_blocks_per_chain=np.asarray(self.pt["n_blocks_per_chain"],
                                               dtype=np.int64),
                 n_chains=np.asarray(self.pt["n_chains"], dtype=np.int64),
                 n_checkpoints=np.asarray(self.pt["n_checkpoints"],
                                          dtype=np.int64),
                 saved_steps=np.asarray(self.pt["saved_steps"], dtype=np.int64))
        save_npz(directory / GRID_FILE,
                 **{key: value for key, value in self.grid.items()
                    if isinstance(value, (torch.Tensor, np.ndarray))},
                 beta=np.asarray(self.beta, dtype=float),
                 free_energy_offset=np.asarray(self.grid["free_energy_offset"],
                                               dtype=float),
                 kde_bandwidth=np.asarray(self.grid["kde_bandwidth"],
                                          dtype=float),
                 cell_area=np.asarray(self.grid["cell_area"], dtype=float),
                 n_samples=np.asarray(self.grid["n_samples"], dtype=np.int64),
                 n_samples_inside=np.asarray(self.grid["n_samples_inside"],
                                             dtype=np.int64),
                 outside_grid_fraction=np.asarray(
                     self.grid["outside_grid_fraction"], dtype=float))
        snis_arrays = {
            "proposals": self.extras["snis_proposals"],
            "log_weights": self.extras["snis_log_weights"],
            "normalized_weights": self.extras["snis_weights"],
            "run_id": np.asarray(self.snis["run_id"], dtype=np.int64),
            "component": self.extras["snis_component"],
            "phase_label": self.extras["snis_labels"],
            "order_parameter": self.extras["snis_order_parameter"],
            "energy_per_site": self.extras["snis_energy_per_site"],
            "coherence": self.extras["snis_coherence"],
            "kink_density": self.extras["snis_kink_density"],
            "component_means": self.extras["snis_component_means"],
            "component_covariances": self.extras["snis_component_covariances"],
            "component_weights": self.extras["snis_component_weights"],
            "hessian_regularization": np.asarray(
                self.snis["proposal"]["hessian_regularization"], dtype=float),
            "importance_sampling_ess": np.asarray(
                self.snis["diagnostics"]["importance_sampling_ess"], dtype=float),
            "ess_fraction": np.asarray(
                self.snis["diagnostics"]["ess_fraction"], dtype=float),
            "max_normalized_weight": np.asarray(
                self.snis["diagnostics"]["max_normalized_weight"], dtype=float),
            "weighted_effective_count_per_phase": np.asarray(
                self.snis["diagnostics"]["weighted_effective_count_per_phase"],
                dtype=float),
            "proposal_phase_counts": np.asarray(
                self.snis["diagnostics"]["proposal_phase_counts"],
                dtype=np.int64),
            "weighted_phase_probabilities": np.asarray(
                self.snis["weighted_estimates"]["phase_probabilities"],
                dtype=float),
            "coverage_nonzero_kink_count": np.asarray(
                self.snis["diagnostics"]["coverage_nonzero_kink_count"],
                dtype=np.int64),
            "coverage_coherence_upper_decile_count": np.asarray(
                self.snis["diagnostics"][
                    "coverage_coherence_upper_decile_count"], dtype=np.int64),
            "coherence_upper_decile_threshold": np.asarray(
                self.snis["diagnostics"]["coherence_upper_decile_threshold"],
                dtype=float),
        }
        if self.extras.get("snis_sir") is not None:
            snis_arrays.update({
                "sir_indices": self.extras["snis_sir"]["indices"],
                "sir_seed": np.asarray(self.extras["snis_sir"]["seed"],
                                       dtype=np.int64),
                "sir_unique_fraction": np.asarray(
                    self.extras["snis_sir"]["unique_fraction"], dtype=float),
                "sir_duplicate_fraction": np.asarray(
                    self.extras["snis_sir"]["duplicate_fraction"], dtype=float),
            })
        save_npz(directory / SNIS_FILE, **snis_arrays)
        write_json(directory / VALIDATION_FILE, {
            "experiment": self.experiment_id,
            "reference_kind": self.kind,
            "acceptance_file": self.acceptance_path,
            "acceptance": self.acceptance,
            "reference_validated": self.reference_validated,
            "n_gates": len(self.validation_records),
            "n_failed": len(self.failed_gates),
            "gates": self.validation_records,
        })
        with open(directory / ACCEPTANCE_FILE, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.acceptance, handle, sort_keys=False,
                           default_flow_style=False)
        self.write_describe(directory)

    @classmethod
    def load(cls, directory, target, device=None) -> "CoupledQuarticChainReference":
        directory = Path(directory)
        device = resolve_device(target.device if device is None else device)
        payload = read_json(directory / Path("reference.json"))
        samples = load_npz(directory / SAMPLES_FILE)
        grid = load_npz(directory / GRID_FILE)
        snis_arrays = load_npz(directory / SNIS_FILE)
        validation = read_json(directory / VALIDATION_FILE)

        def tensor(array):
            return torch.as_tensor(np.asarray(array), dtype=torch.float64,
                                   device=device)

        phases = list(target.extras["phases"])
        basin_map = _build_basin_map(
            target, payload["provenance"]["basin_map"], device,
            cache=directory / BASIN_CACHE_FILE)
        grid_record = dict(payload["order_parameter_grid"])
        grid_record.update({key: tensor(grid[key]) for key in
                            ("m_x_edges", "m_y_edges", "m_x_centers",
                             "m_y_centers", "p_star", "p_star_standard_error",
                             "free_energy", "kde_p_star")})
        grid_record["counts"] = torch.as_tensor(np.asarray(grid["counts"]),
                                                device=device)
        sample_bank = tensor(samples["configurations"])
        extras = {
            "pt_labels": np.asarray(samples["phase_label"], dtype=np.int64),
            "pt_energy_per_site": tensor(samples["energy_per_site"]),
            "pt_coherence": tensor(samples["coherence"]),
            "pt_kink_density": tensor(samples["kink_density"]),
            "pt_two_point_correlation": tensor(samples["two_point_correlation"]),
            "snis_proposals": tensor(snis_arrays["proposals"]),
            "snis_log_weights": tensor(snis_arrays["log_weights"]),
            "snis_weights": tensor(snis_arrays["normalized_weights"]),
            "snis_component": np.asarray(snis_arrays["component"],
                                         dtype=np.int64),
            "snis_labels": np.asarray(snis_arrays["phase_label"],
                                      dtype=np.int64),
            "snis_order_parameter": tensor(snis_arrays["order_parameter"]),
            "snis_energy_per_site": tensor(snis_arrays["energy_per_site"]),
            "snis_coherence": tensor(snis_arrays["coherence"]),
            "snis_kink_density": tensor(snis_arrays["kink_density"]),
            "snis_component_means": tensor(snis_arrays["component_means"]),
            "snis_component_covariances":
                tensor(snis_arrays["component_covariances"]),
            "snis_component_weights": tensor(snis_arrays["component_weights"]),
            "snis_sir": None,
        }
        projections = metrics.make_projections(
            2, int(payload["metric_inputs"]["sw2_projections"]["n_projections"]),
            int(payload["provenance"]["metrics"]["sw2_projection_seed"]), device)
        return cls(
            target=target, phases=phases, sample_bank=sample_bank,
            order_parameter_bank=tensor(samples["order_parameter"]),
            run_id=samples["run_id"], chain_id=samples["chain_id"],
            iteration_step=samples["iteration_step"],
            checkpoint_index=samples["checkpoint_index"],
            pt=payload["pt_mala"], snis=payload["snis"], grid=grid_record,
            basin_map=basin_map, sw2_projections=projections,
            mmd_bandwidth=float(payload["metric_inputs"]["mmd_bandwidth"]),
            observable_targets=_restore_targets(payload["observable_targets"],
                                                device),
            validation_records=validation["gates"],
            reference_validated=bool(validation["reference_validated"]),
            acceptance=validation["acceptance"],
            acceptance_path=validation["acceptance_file"],
            provenance=payload["provenance"], extras=extras,
            describe_cache=payload)

    # -- draws -------------------------------------------------------------
    def sample(self, n: int, generator: torch.Generator) -> torch.Tensor:
        """``(n, 24)`` draws from the frozen PT-MALA bank.

        Without replacement while the bank is large enough, with replacement
        otherwise; which one was used is recorded in
        :attr:`last_sample_record`.
        """
        n = check_positive_int(n, "n")
        size = int(self.sample_bank.shape[0])
        gen_device = generator.device
        replacement = n > size
        if replacement:
            index = torch.randint(0, size, (n,), generator=generator,
                                  device=gen_device)
        else:
            index = torch.randperm(size, generator=generator,
                                   device=gen_device)[:n]
        self.last_sample_record = {
            "n": int(n), "bank_size": size, "with_replacement": bool(replacement),
            "policy": ("with replacement: the request exceeds the bank"
                       if replacement else "without replacement"),
        }
        return self.sample_bank[index.to(self.sample_bank.device)]

    def snis_sir_resample(self, n: int, seed: int) -> dict:
        """SIR resamples of the SNIS pool, for scatter plots only.

        SIR is never used for an estimate -- the weights are used directly --
        so this records the resampling seed, the unique fraction, and the
        duplicate fraction alongside the draws.
        """
        n = check_positive_int(n, "n")
        seed = check_seed(seed, "seed")
        weights = self.extras["snis_weights"]
        generator = frozen_generator(weights.device, seed)
        index = torch.multinomial(weights, n, replacement=True,
                                  generator=generator)
        unique = int(torch.unique(index).numel())
        record = {
            "indices": index.detach().cpu().numpy().astype(np.int64),
            "seed": int(seed), "n": int(n),
            "unique_fraction": unique / float(n),
            "duplicate_fraction": 1.0 - unique / float(n),
        }
        self.extras["snis_sir"] = record
        return record


def _restore_targets(payload: Mapping, device) -> dict:
    out = {}
    for name, record in payload.items():
        entry = dict(record)
        for key in ("value", "standard_error"):
            value = entry.get(key)
            if isinstance(value, list):
                entry[key] = torch.as_tensor(np.asarray(value, dtype=float),
                                             dtype=torch.float64, device=device)
        out[name] = entry
    return out


# ================================================================ the builder
def _build_basin_map(target, settings: Mapping, device, cache=None
                     ) -> GradientFlowBasinMap2D:
    """Gradient-flow phase map on the order-parameter plane.

    For a homogeneous configuration ``q_i = v`` the order parameter is ``v`` and
    the chain energy reduces to the site potential ``W(v)``, so ``W`` is the
    landscape whose basins define the four phases. The domain must reach at
    least ``+-4`` so a double phase-to-phase jump of the order parameter cannot
    be clamped into a boundary basin.
    """
    bound = float(settings["bound"])
    if bound < 4.0:
        raise ValueError(
            f"the order-parameter basin map needs bounds of at least +-4 so a "
            f"double jump cannot be clamped, got +-{bound}")
    coefficients = target.potential.coefficients
    return GradientFlowBasinMap2D(
        lambda v: site_potential_grad(v, coefficients),
        target.extras["refined_site_minima"], [-bound, -bound], [bound, bound],
        n_grid=int(settings["n_grid"]), device=device,
        cache=None if cache is None else str(cache),
        dt_flow=float(settings["dt_flow"]), n_flow=int(settings["n_flow"]))


def _nan_like(template: Mapping) -> dict:
    out = {}
    for name in _STATISTIC_ORDER:
        value = np.asarray(template[name], dtype=float)
        out[name] = (float("nan") if value.ndim == 0
                     else np.full(value.shape, float("nan")))
    return out


_EXPERIMENT_IDS = frozenset({"E1", "E2", "E3", "E4"})


def build_reference(*args, **kwargs) -> CoupledQuarticChainReference:
    """Build (or reuse) the frozen E4 reference.

    Accepts the experiment configuration, the ``Target``, and the output
    directory in any order, as keywords or positionally, so it can be driven
    either by ``src.references.build_or_load`` or directly from a script.

    Raises :class:`ReferenceValidationError` when any frozen acceptance gate
    fails. The validation record is written first, so the failure is
    inspectable and the caller exits nonzero without an official reference
    having been promoted.
    """
    config = kwargs.pop("config", None)
    target = kwargs.pop("target", None)
    directory = kwargs.pop("directory", None)
    for value in args:
        if isinstance(value, Mapping) and config is None:
            config = value
        elif hasattr(value, "no_count") and hasattr(value, "potential") \
                and target is None:
            target = value
        elif isinstance(value, (str, Path)):
            if str(value) in _EXPERIMENT_IDS:
                continue
            if directory is None:
                directory = value
    device = kwargs.pop("device", None)
    rebuild = bool(kwargs.pop("rebuild", False))
    verbose = bool(kwargs.pop("verbose", False))
    write = bool(kwargs.pop("save", True))
    acceptance_path = kwargs.pop("acceptance_path", None)
    if config is None or target is None:
        raise TypeError("build_reference needs the experiment config and target")
    device = resolve_device(target.device if device is None else device)
    directory = None if directory is None else Path(directory)

    acceptance, resolved_acceptance_path = _load_acceptance(config,
                                                            acceptance_path)
    reference_config = dict(config["reference"])
    basin_settings = _resolved(reference_config.get("basin_map"),
                               _BASIN_MAP_DEFAULTS)
    grid_settings = _resolved(reference_config.get("order_parameter_grid"),
                              _GRID_DEFAULTS)
    bootstrap = dict(acceptance["uncertainty"]["bootstrap"])
    declared = dict(reference_config.get("bootstrap") or {})
    for key in ("replicates", "seed"):
        if key in declared and int(declared[key]) != int(bootstrap[key]):
            raise ValueError(
                f"reference.bootstrap.{key} = {declared[key]} in the experiment "
                f"config disagrees with the frozen acceptance file "
                f"({bootstrap[key]}); the two must declare the same bootstrap")
    provenance = _provenance_record(
        config, target, acceptance, resolved_acceptance_path,
        basin_settings=basin_settings, grid_settings=grid_settings)

    if directory is not None and not rebuild:
        existing = _try_load(directory, target, device, provenance)
        if existing is not None:
            _progress(verbose, f"reusing the stored reference in {directory}")
            return existing

    with target.no_count():
        reference = _construct(
            config, target, device, acceptance=acceptance,
            acceptance_path=resolved_acceptance_path, provenance=provenance,
            basin_settings=basin_settings, grid_settings=grid_settings,
            bootstrap=bootstrap, directory=directory, verbose=verbose)
    if directory is not None and write:
        reference.save(directory)
        _progress(verbose, f"wrote the reference artifacts to {directory}")
    if not reference.reference_validated:
        raise ReferenceValidationError(reference.failed_gates,
                                       directory=directory)
    return reference


def _try_load(directory: Path, target, device, provenance: Mapping):
    from .base import stored_provenance_hash
    from ..results import stable_hash

    required = (SAMPLES_FILE, GRID_FILE, SNIS_FILE, VALIDATION_FILE,
                "reference.json")
    if not all((directory / name).is_file() for name in required):
        return None
    if stored_provenance_hash(directory) != stable_hash(provenance):
        return None
    try:
        stored = CoupledQuarticChainReference.load(directory, target, device)
    except (OSError, KeyError, ValueError):
        return None
    return stored if stored.reference_validated else None


def _provenance_record(config: Mapping, target, acceptance: Mapping,
                       acceptance_path, *, basin_settings: Mapping,
                       grid_settings: Mapping) -> dict:
    """The configuration-determined identity of this reference."""
    reference_config = copy.deepcopy(dict(config["reference"]))
    metric_config = copy.deepcopy(dict(config.get("metrics", {})))
    return {
        "experiment_id": "E4",
        "kind": CoupledQuarticChainReference.kind,
        "target": copy.deepcopy(dict(config["target"])),
        "beta": float(target.beta),
        "n_sites": int(target.potential.n_sites),
        "phases": list(target.extras["phases"]),
        "reference": reference_config,
        "acceptance_file": str(acceptance_path),
        "acceptance": copy.deepcopy(dict(acceptance)),
        "basin_map": dict(basin_settings),
        "order_parameter_grid": dict(grid_settings),
        "metrics": {
            "mmd_bandwidth_rule": (metric_config.get("mmd", {})
                                   .get("bandwidth_rule",
                                        "median_heuristic_on_reference_bank")),
            "mmd_bandwidth_reference_points": int(
                metric_config.get("mmd", {}).get("bandwidth_reference_points",
                                                 4096)),
            "mmd_bandwidth_seed": int(
                metric_config.get("mmd", {}).get("bandwidth_seed", 99)),
            "sw2_n_projections": int(
                metric_config.get("sw2", {}).get("n_projections", 512)),
            "sw2_projection_seed": int(
                metric_config.get("sw2", {}).get("projection_seed", 777)),
        },
    }


def build_annealed_smc_fallback(*args, **kwargs):
    """Optional third-party check. Disabled, and deliberately not implemented.

    This arm is enabled only after BOTH the PT-MALA reference and the SNIS
    cross-check have been extended (longer runs, better temperature mixing, a
    better proposal) and still disagree. It is never a tie-breaker between two
    under-converged references, and averaging the two or picking the nicer one
    is not permitted.

    If it is ever enabled it must start from a PROPER base distribution ``q_0``
    with ``gamma_lambda(x) = q_0(x)^(1-lambda) exp(-lambda beta V(x))``. It must
    never start from an unbounded uniform at ``beta_0 = 0``: on R^24 that is not
    a probability distribution, so the first weight update is meaningless.
    """
    config = None
    for value in args:
        if isinstance(value, Mapping):
            config = value
            break
    config = kwargs.get("config", config) or {}
    settings = dict((config.get("reference", {}) or {})
                    .get("fallback_annealed_smc", {}) or {})
    enabled = bool(settings.get("enabled", False))
    raise NotImplementedError(
        "the annealed-SMC fallback is an optional third-party check that is "
        f"enabled only after PT and SNIS have both been extended and still "
        f"disagree (fallback_annealed_smc.enabled = {enabled}, off by "
        "default). If it is enabled it must start from a proper base "
        "distribution q_0 with gamma_lambda(x) = q_0(x)^(1-lambda) "
        "exp(-lambda beta V(x)), never from an unbounded uniform at beta_0 = 0.")


def _construct(config: Mapping, target, device, *, acceptance, acceptance_path,
               provenance, basin_settings, grid_settings, bootstrap, directory,
               verbose) -> CoupledQuarticChainReference:
    reference_config = dict(config["reference"])
    phases = list(target.extras["phases"])
    n_phases = len(phases)
    n_sites = int(target.potential.n_sites)
    beta = float(target.beta)
    layout = _Layout(n_phases, n_sites // 2 + 1)
    site_minima = target.extras["refined_site_minima"]
    continuous = list(acceptance["observables"]["continuous"])
    indicators = list(acceptance["observables"]["indicators"])
    multiplier = float(acceptance["uncertainty"]["block_length_multiplier"])
    replicates = int(bootstrap["replicates"])
    seed = int(bootstrap["seed"])

    _progress(verbose, "building the order-parameter basin map")
    basin_map = _build_basin_map(
        target, basin_settings, device,
        cache=None if directory is None else Path(directory) / BASIN_CACHE_FILE)

    _progress(verbose, "running multi-start PT-MALA")
    pt = _run_pt_mala(target, reference_config["pt_mala"], verbose=verbose)
    bank = pt["configurations"]
    n_chains, n_checkpoints = int(pt["n_chains"]), int(pt["n_checkpoints"])

    _progress(verbose, f"scoring {bank.shape[0]} PT samples")
    pt_per_sample = _per_sample_observables(target, bank, basin_map=basin_map,
                                            site_minima=site_minima)
    pt_features = _feature_matrix(
        layout, labels=pt_per_sample["labels"],
        energies=pt_per_sample["energy_per_site"],
        m=pt_per_sample["order_parameter"],
        coherences=pt_per_sample["coherence"],
        correlations=pt_per_sample["two_point_correlation"],
        kinks=pt_per_sample["kink_density"])
    pt_point = _as_numpy_stats(_canonical_estimates(target, pt_per_sample,
                                                    n_phases=n_phases))
    _check_summary_algebra(pt_point, _derive(_summary_row(pt_features, None),
                                             layout, beta=beta, n_sites=n_sites),
                           "PT-MALA")

    labels = pt_per_sample["labels"].detach().cpu().numpy().astype(np.int64)
    label_series = labels.reshape(n_chains, n_checkpoints)
    series = _observable_series(pt_per_sample, labels, continuous=continuous,
                                indicators=indicators, phases=phases,
                                n_chains=n_chains, n_checkpoints=n_checkpoints)

    per_chain_series = {f"{name}::chain{index}": series[name][index]
                        for name in series for index in range(n_chains)}
    block_length = metrics.recommended_block_length(per_chain_series, multiplier)
    taus = [metrics.autocorrelation_time(values)
            for values in per_chain_series.values()]
    finite_taus = [value for value in taus if np.isfinite(value)]
    max_tau = max(finite_taus) if finite_taus else float("nan")
    blocks = _block_units(n_chains, n_checkpoints, block_length)
    n_blocks_per_chain = n_checkpoints // block_length
    pt_summaries = _block_summaries(pt_features, blocks)
    _progress(verbose, f"block length {block_length} -> {n_blocks_per_chain} "
                       f"blocks per chain ({pt_summaries.shape[0]} total)")

    if pt_summaries.shape[0] >= 2:
        pt_se = _bootstrap_statistics(pt_summaries, layout, beta=beta,
                                      n_sites=n_sites, replicates=replicates,
                                      seed=seed, template=pt_point)
    else:
        pt_se = _nan_like(pt_point)

    _progress(verbose, "drawing the Laplace-mixture SNIS cross-check")
    snis_settings = dict(reference_config["snis"])
    energies = target.value(target.extras["coherent_states"],
                            cost_class="baseline")
    mixture = _LaplaceMixture(target.extras["coherent_states"],
                              target.extras["coherent_hessians"], energies,
                              beta, float(snis_settings["hessian_regularization"]))
    snis = _run_snis(target, snis_settings, mixture=mixture, basin_map=basin_map,
                     site_minima=site_minima, layout=layout, n_phases=n_phases,
                     verbose=verbose)
    snis_point = _as_numpy_stats(_canonical_estimates(
        target, snis["per_sample"], weights=snis["weights"], n_phases=n_phases))
    _check_summary_algebra(
        snis_point, _derive(snis["summaries"].sum(axis=0), layout, beta=beta,
                            n_sites=n_sites), "SNIS")
    snis_influence = _influence_estimates(snis["features"], snis["weights"],
                                          layout, beta=beta, n_sites=n_sites)

    coherence_decile = float(
        torch.quantile(pt_per_sample["coherence"], 0.9).item())
    relative_susceptibility = _relative_frobenius(
        snis_point["susceptibility"], pt_point["susceptibility"])
    relative_correlation = _relative_l2(snis_point["two_point_correlation"],
                                        pt_point["two_point_correlation"])
    if pt_summaries.shape[0] >= 2:
        _progress(verbose, f"hierarchical bootstrap ({replicates} replicates)")
        cross = _cross_check_bootstrap(pt_summaries, snis["summaries"], layout,
                                       beta=beta, n_sites=n_sites,
                                       replicates=replicates, seed=seed)
    else:
        cross = {"susceptibility_relative_frobenius_se": float("nan"),
                 "two_point_correlation_relative_l2_se": float("nan")}

    records = _pt_gates(
        acceptance, pt=pt, series=series, label_series=label_series,
        phases=phases, continuous=continuous, indicators=indicators,
        block_length=block_length, point=pt_point, standard_errors=pt_se,
        bootstrap=bootstrap)
    records.append(_block_length_gate(
        acceptance, block_length=block_length,
        n_blocks_per_chain=n_blocks_per_chain,
        n_blocks_total=int(pt_summaries.shape[0]), max_tau=max_tau,
        n_checkpoints=n_checkpoints,
        required_checkpoints=int(acceptance["uncertainty"]
                                 ["min_effective_blocks"]) * block_length))
    snis_records, snis_diagnostics = _snis_gates(
        acceptance, snis=snis, layout=layout, phases=phases, beta=beta,
        n_sites=n_sites, coherence_decile=coherence_decile)
    records.extend(snis_records)
    records.extend(_cross_check_gates(
        acceptance, phases=phases, pt_point=pt_point, pt_se=pt_se,
        snis_point=snis_point, snis_influence=snis_influence,
        relative_susceptibility=relative_susceptibility,
        relative_susceptibility_se=cross["susceptibility_relative_frobenius_se"],
        relative_correlation=relative_correlation,
        relative_correlation_se=cross["two_point_correlation_relative_l2_se"],
        bootstrap=bootstrap, block_length=block_length))

    _progress(verbose, "projecting the order-parameter density")
    grid = _order_parameter_grid(
        pt_per_sample["order_parameter"], beta=beta, settings=grid_settings,
        n_chains=n_chains, n_checkpoints=n_checkpoints)

    metric_settings = provenance["metrics"]
    projections = metrics.make_projections(
        2, int(metric_settings["sw2_n_projections"]),
        int(metric_settings["sw2_projection_seed"]), device)
    bandwidth = metrics.median_heuristic(
        pt_per_sample["order_parameter"],
        max_points=int(metric_settings["mmd_bandwidth_reference_points"]),
        seed=int(metric_settings["mmd_bandwidth_seed"]))

    se_definition = (f"block bootstrap over {pt_summaries.shape[0]} "
                     f"non-overlapping PT blocks of {block_length} saved "
                     f"checkpoints")
    observable_targets = {
        name: {
            "value": (float(pt_point[name]) if np.ndim(pt_point[name]) == 0
                      else torch.as_tensor(np.asarray(pt_point[name]),
                                           dtype=torch.float64, device=device)),
            "standard_error": (float(pt_se[name]) if np.ndim(pt_se[name]) == 0
                               else torch.as_tensor(np.asarray(pt_se[name]),
                                                    dtype=torch.float64,
                                                    device=device)),
            "standard_error_method": se_definition,
            "bootstrap_seed": seed,
            "bootstrap_replicates": replicates,
            "block_length": int(block_length),
        }
        for name in _STATISTIC_ORDER
    }

    pt_record = {
        key: pt[key] for key in
        ("n_runs", "chains_per_run", "n_chains", "n_checkpoints", "n_replicas",
         "swap_interval", "burn_in_steps", "total_steps", "thinning", "dt",
         "beta_min", "seeds", "init_phases", "init_sigma", "phase_indices")}
    pt_record.update({
        "betas": pt["betas"].detach().cpu().tolist(),
        "mala_acceptance_per_replica": pt["mala_acceptance"].tolist(),
        "swap_acceptance_per_pair": pt["swap_acceptance"].tolist(),
        "block_length": int(block_length),
        "n_blocks_per_chain": int(n_blocks_per_chain),
        "n_blocks_total": int(pt_summaries.shape[0]),
        "max_integrated_autocorrelation_time": _finite(max_tau),
        "block_rule": acceptance["uncertainty"]["block_rule"],
        "kernel": "mh_corrected_mala_with_recomputed_reverse_drift",
        "saved_steps": pt["saved_steps"],
        "estimates": pt_point,
        "standard_errors": pt_se,
        "phase_visits_per_chain": [
            int(np.unique(chain[chain != OUTSIDE_LABEL]).size)
            for chain in label_series],
        "entry_events_per_phase": _entry_events(
            label_series, n_phases,
            int(acceptance["pt_mala_gates"]
                ["entry_event_min_consecutive_checkpoints"])).tolist(),
    })
    snis_record = {
        "n_runs": snis["n_runs"], "proposals_per_run": snis["proposals_per_run"],
        "seeds": snis["seeds"], "proposal": mixture.describe(),
        "diagnostics": snis_diagnostics,
        "weighted_estimates": snis_point,
        "influence_standard_errors": _as_numpy_stats(snis_influence),
        "usage": ("weights are used directly for weighted estimates; SNIS is "
                  "not a second unweighted sample bank"),
        "cross_check": {
            "susceptibility_relative_frobenius": relative_susceptibility,
            "susceptibility_relative_frobenius_se":
                cross["susceptibility_relative_frobenius_se"],
            "two_point_correlation_relative_l2": relative_correlation,
            "two_point_correlation_relative_l2_se":
                cross["two_point_correlation_relative_l2_se"],
            "bootstrap_seed": seed, "bootstrap_replicates": replicates,
            "bootstrap_units": list(bootstrap["resample_units"]),
        },
    }
    extras = {
        "pt_labels": labels,
        "pt_energy_per_site": pt_per_sample["energy_per_site"],
        "pt_coherence": pt_per_sample["coherence"],
        "pt_kink_density": pt_per_sample["kink_density"],
        "pt_two_point_correlation": pt_per_sample["two_point_correlation"],
        "snis_proposals": snis["proposals"],
        "snis_log_weights": snis["log_weights"],
        "snis_weights": snis["weights"],
        "snis_component": snis["component"].detach().cpu().numpy(),
        "snis_labels": snis["per_sample"]["labels"].detach().cpu().numpy(),
        "snis_order_parameter": snis["per_sample"]["order_parameter"],
        "snis_energy_per_site": snis["per_sample"]["energy_per_site"],
        "snis_coherence": snis["per_sample"]["coherence"],
        "snis_kink_density": snis["per_sample"]["kink_density"],
        "snis_component_means": mixture.means,
        "snis_component_covariances": mixture.covariances,
        "snis_component_weights": mixture.weights,
        "snis_sir": None,
    }
    reference = CoupledQuarticChainReference(
        target=target, phases=phases, sample_bank=bank,
        order_parameter_bank=pt_per_sample["order_parameter"],
        run_id=pt["run_id"], chain_id=pt["chain_id"],
        iteration_step=pt["iteration_step"],
        checkpoint_index=pt["checkpoint_index"], pt=pt_record, snis=snis_record,
        grid=grid, basin_map=basin_map, sw2_projections=projections,
        mmd_bandwidth=bandwidth, observable_targets=observable_targets,
        validation_records=records,
        reference_validated=all(record["passed"] for record in records),
        acceptance=acceptance, acceptance_path=acceptance_path,
        provenance=provenance, extras=extras)
    sir_size = int(snis_settings.get("sir_scatter_size", 0) or 0)
    if sir_size > 0:
        reference.snis_sir_resample(
            sir_size, int(snis_settings.get("sir_scatter_seed",
                                            snis["seeds"][0] + 7919)))
    return reference


def _check_summary_algebra(canonical: Mapping, derived: Mapping,
                           arm: str) -> None:
    """The bootstrap algebra must reproduce the canonical point estimates.

    :mod:`src.observables` owns every estimator definition; the summary algebra
    exists only so a bootstrap replicate can be recomputed from block sums. If
    the two ever disagree the reference is inconsistent and must not be used.
    """
    for name in _STATISTIC_ORDER:
        left = np.atleast_1d(np.asarray(canonical[name], dtype=float))
        right = np.atleast_1d(np.asarray(derived[name], dtype=float))
        scale = max(1.0, float(np.max(np.abs(left))) if left.size else 1.0)
        if not np.allclose(left, right, rtol=1e-8, atol=1e-8 * scale):
            raise AssertionError(
                f"the {arm} summary algebra disagrees with "
                f"src.observables on {name!r}: {left} vs {right}")
