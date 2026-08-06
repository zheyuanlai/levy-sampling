"""E1 reference: high-precision inverse-CDF ground truth for the double well.

``pi(x) ~ exp(-beta (x^2 - 1)^2)`` is one dimensional, so the reference is not a
sample bank with a sampling error -- it is the distribution function itself,
built by trapezoidal quadrature on a uniform grid over the numerical box. Every
E1 metric that needs a CDF (KS, CDF-L2, W1) reads it from :attr:`.cdf` exactly;
only the two-sample metrics touch :attr:`.sample_bank`, and that bank is frozen
once at a seed read from the YAML so every method is scored against the same
draws.

The box ``[-5.2, 5.2]`` is not an approximation at ``beta = 8``: the integrand
at the boundary is ``exp(-8 * 678)``, which underflows float64. The validation
block measures that, together with the discretisation error of the grid, and
records the reference-vs-reference W2 at the production particle count -- the
sampling noise floor below which no method's W2 can be interpreted.
"""
from __future__ import annotations

from pathlib import Path
import math

import torch

from ..results import stable_hash
from .base import (REFERENCE_JSON, Reference, as_tensor,
                   check_positive_int, check_seed, frozen_generator,
                   import_metrics, load_npz, passed_check, read_json,
                   save_npz, write_json)

KIND = "grid_inverse_cdf_1d"
EXPERIMENT_ID = "E1"

#: Frozen default seeds. A YAML key of the same name overrides them.
DEFAULT_BANK_SEED = 424242
DEFAULT_SELF_W2_SEED = 909090

#: Points at which two CDFs built on different grids are compared.
COMPARISON_GRID_POINTS = 20001

#: Validation tolerances, overridable under ``reference.validation.tolerances``.
#: The grid tolerances are ~1e4 times the trapezoidal error expected at the
#: coarsest requested grid; the self-W2 tolerance is a sanity bound, not a
#: quality gate -- the value itself is the noise floor being reported.
DEFAULT_TOLERANCES = {
    "grid_refinement_cdf_max_abs": 1e-6,
    "grid_refinement_moment_abs": 1e-6,
    "grid_refinement_basin_mass_abs": 1e-8,
    "wide_box_cdf_max_abs": 1e-6,
    "wide_box_moment_abs": 1e-6,
    "wide_box_basin_mass_abs": 1e-8,
    "outside_narrow_box_mass": 1e-12,
    "self_w2_mean": 0.5,
}

GRID_FILE = "reference_grid.npz"
BANK_FILE = "reference_samples.npz"
VALIDATION_FILE = "reference_validation.json"


# ============================================================ grid quadrature
class _Quadrature:
    """Normalized density and CDF of ``exp(-beta V)`` on a uniform 1-D grid."""

    def __init__(self, target, lo: float, hi: float, n_grid: int) -> None:
        device = target.device
        self.grid = torch.linspace(float(lo), float(hi), int(n_grid),
                                   dtype=torch.float64, device=device)
        log_p = target.log_target(self.grid.unsqueeze(-1), cost_class="baseline")
        unnormalized = torch.exp(log_p - log_p.max())
        widths = self.grid[1:] - self.grid[:-1]
        increments = 0.5 * (unnormalized[1:] + unnormalized[:-1]) * widths
        cumulative = torch.cumsum(increments, dim=0)
        mass = cumulative[-1].clone()
        if not bool(torch.isfinite(mass)) or float(mass.item()) <= 0.0:
            raise ValueError("the E1 quadrature has no mass on the requested box")
        self.pdf = unnormalized / mass
        cdf = torch.cat([torch.zeros(1, dtype=torch.float64, device=device),
                         cumulative / mass])
        # The CUDA scan reassociates the additions, so a cumsum of nonnegative
        # terms can dip by an ulp where the density has underflowed. searchsorted
        # requires a sorted array, so the running maximum is taken and the
        # endpoints are pinned.
        self.cdf = torch.cummax(cdf, dim=0).values.clamp(0.0, 1.0)
        self.cdf[0] = 0.0
        self.cdf[-1] = 1.0

    def cdf_on(self, x: torch.Tensor) -> torch.Tensor:
        return interpolate_cdf(self.grid, self.cdf, x)

    def moments(self, orders) -> dict[str, float]:
        return grid_moments(self.grid, self.pdf, orders)

    def basin_masses(self) -> dict[str, float]:
        left = float(self.cdf_on(torch.zeros(1, dtype=torch.float64,
                                             device=self.grid.device))[0].item())
        return {"left": left, "right": 1.0 - left}


def interpolate_cdf(grid: torch.Tensor, cdf: torch.Tensor,
                    x: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear CDF value at arbitrary points, clamped to ``[0, 1]``."""
    points = torch.as_tensor(x, dtype=torch.float64, device=grid.device).reshape(-1)
    index = torch.clamp(torch.searchsorted(grid, points), 1, grid.numel() - 1)
    left, right = grid[index - 1], grid[index]
    lower, upper = cdf[index - 1], cdf[index]
    fraction = (points - left) / torch.clamp(right - left, min=1e-300)
    values = lower + fraction * (upper - lower)
    values = torch.where(points <= grid[0], torch.zeros_like(values), values)
    values = torch.where(points >= grid[-1], torch.ones_like(values), values)
    return values.reshape(x.shape if isinstance(x, torch.Tensor) else (-1,))


def grid_moments(grid: torch.Tensor, pdf: torch.Tensor,
                 orders) -> dict[str, float]:
    """``E[x^k]`` by trapezoidal quadrature of the normalized grid density."""
    widths = grid[1:] - grid[:-1]
    out = {}
    for order in orders:
        integrand = pdf * grid ** int(order)
        value = (0.5 * (integrand[1:] + integrand[:-1]) * widths).sum()
        out[f"m{int(order)}"] = float(value.item())
    return out


def inverse_cdf_sample(grid: torch.Tensor, cdf: torch.Tensor, n: int,
                       generator: torch.Generator) -> torch.Tensor:
    """``(n, 1)`` inverse-CDF draws by binary search plus linear interpolation."""
    n = check_positive_int(n, "n")
    u = torch.rand(n, generator=generator, device=grid.device,
                   dtype=torch.float64)
    index = torch.clamp(torch.searchsorted(cdf, u), 1, grid.numel() - 1)
    lower, upper = cdf[index - 1], cdf[index]
    fraction = (u - lower) / torch.clamp(upper - lower, min=1e-300)
    left, right = grid[index - 1], grid[index]
    return (left + fraction * (right - left)).unsqueeze(1)


# ================================================================= the class
class DoubleWellReference(Reference):
    """Grid inverse-CDF reference for ``pi ~ exp(-beta (x^2 - 1)^2)``.

    Frozen at construction: the quadrature grid, its normalized density and
    CDF, the sample bank (``sample_bank_size`` draws at ``bank_seed``), the
    grid-quadrature moments and basin masses, and the one-off validation
    record. Nothing here is recomputed per run.
    """

    kind = KIND
    experiment_id = EXPERIMENT_ID

    def __init__(self, *, target, provenance: dict, grid: torch.Tensor,
                 pdf: torch.Tensor, cdf: torch.Tensor,
                 sample_bank: torch.Tensor, measured: dict,
                 validation: dict) -> None:
        self.target = target
        self.beta = float(target.beta)
        self.device = grid.device
        self._provenance = dict(provenance)
        self._measured = dict(measured)
        self.validation = dict(validation)
        self.grid = grid
        self.pdf = pdf
        self.cdf = cdf
        self.sample_bank = sample_bank
        self.bounds = tuple(float(value) for value in provenance["bounds"])
        self.n_grid = int(provenance["n_grid"])
        self.bank_seed = int(provenance["bank_seed"])
        self.moments = dict(measured["moments"])
        self.basin_masses = dict(measured["basin_masses"])
        self.basin_labels = list(measured["basin_labels"])

    # -- construction ------------------------------------------------------
    @staticmethod
    def provenance_for(config: dict, target) -> dict:
        """The configuration-determined identity of an E1 reference."""
        block = config["reference"]
        bounds = [float(value) for value in block["bounds"]]
        if len(bounds) != 2 or not bounds[1] > bounds[0]:
            raise ValueError(f"reference.bounds must be [lo, hi] with hi > lo, "
                             f"got {block['bounds']!r}")
        validation = dict(block.get("validation") or {})
        tolerances = {**DEFAULT_TOLERANCES,
                      **(validation.get("tolerances") or {})}
        return {
            "experiment_id": EXPERIMENT_ID,
            "kind": KIND,
            "potential": target.potential.name,
            "beta": float(target.beta),
            "dimension": int(target.d),
            "bounds": bounds,
            "n_grid": check_positive_int(block["n_grid"], "reference.n_grid"),
            "sample_bank_size": check_positive_int(
                block["sample_bank_size"], "reference.sample_bank_size"),
            "bank_seed": check_seed(block.get("bank_seed", DEFAULT_BANK_SEED),
                                    "reference.bank_seed"),
            "validation": {
                "grid_sizes": [check_positive_int(value, "grid_sizes entry")
                               for value in validation.get("grid_sizes", [])],
                "wide_bounds": [float(value)
                                for value in validation.get("wide_bounds", [])],
                "moment_orders": [int(value) for value in
                                  validation.get("moment_orders", [1, 2, 3, 4])],
                "self_w2_replicates": int(
                    validation.get("self_w2_replicates", 0)),
                "self_w2_seed": check_seed(
                    validation.get("self_w2_seed", DEFAULT_SELF_W2_SEED),
                    "reference.validation.self_w2_seed"),
                "self_w2_particles": check_positive_int(
                    config["protocol"]["particles"], "protocol.particles"),
                "comparison_grid_points": COMPARISON_GRID_POINTS,
                "tolerances": {key: float(value)
                               for key, value in sorted(tolerances.items())},
            },
        }

    @classmethod
    def build(cls, config: dict, target, directory: Path | None = None, *,
              device=None, verbose: bool = False) -> "DoubleWellReference":
        provenance = cls.provenance_for(config, target)
        lo, hi = provenance["bounds"]
        with target.no_count():
            if verbose:
                print(f"[E1] quadrature on {provenance['n_grid']} points")
            quadrature = _Quadrature(target, lo, hi, provenance["n_grid"])
            orders = provenance["validation"]["moment_orders"]
            moments = quadrature.moments(orders)
            basin_masses = quadrature.basin_masses()
            generator = frozen_generator(target.device, provenance["bank_seed"])
            sample_bank = inverse_cdf_sample(
                quadrature.grid, quadrature.cdf,
                provenance["sample_bank_size"], generator)
            if verbose:
                print("[E1] validating")
            validation = _validate(target, provenance, quadrature, sample_bank)
        measured = {
            "moments": moments,
            "basin_masses": basin_masses,
            "basin_labels": ["left", "right"],
            "grid_spacing": float((hi - lo) / (provenance["n_grid"] - 1)),
            "sample_bank_shape": list(sample_bank.shape),
            "sample_bank_mean": float(sample_bank.mean().item()),
            "build_device": str(target.device),
        }
        return cls(target=target, provenance=provenance, grid=quadrature.grid,
                   pdf=quadrature.pdf, cdf=quadrature.cdf,
                   sample_bank=sample_bank, measured=measured,
                   validation=validation)

    # -- contract ----------------------------------------------------------
    def describe(self) -> dict:
        record = {
            "experiment_id": EXPERIMENT_ID,
            "kind": KIND,
            "provenance": dict(self._provenance),
            "provenance_hash": stable_hash(self._provenance),
            **self._measured,
            "validation": dict(self.validation),
        }
        return record

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_npz(directory / GRID_FILE, grid=self.grid, pdf=self.pdf,
                 cdf=self.cdf)
        save_npz(directory / BANK_FILE, sample_bank=self.sample_bank)
        write_json(directory / VALIDATION_FILE, self.validation)
        self.write_describe(directory)

    @classmethod
    def load(cls, directory: Path, target, device) -> "DoubleWellReference":
        directory = Path(directory)
        payload = read_json(directory / REFERENCE_JSON)
        arrays = load_npz(directory / GRID_FILE)
        bank = load_npz(directory / BANK_FILE)
        measured = {key: value for key, value in payload.items()
                    if key not in ("provenance", "provenance_hash",
                                   "validation", "experiment_id", "kind")}
        return cls(
            target=target,
            provenance=payload["provenance"],
            grid=as_tensor(arrays["grid"], device),
            pdf=as_tensor(arrays["pdf"], device),
            cdf=as_tensor(arrays["cdf"], device),
            sample_bank=as_tensor(bank["sample_bank"], device),
            measured=measured,
            validation=payload["validation"])

    def sample(self, n: int, generator: torch.Generator) -> torch.Tensor:
        """``(n, 1)`` exact inverse-CDF draws from the grid distribution."""
        return inverse_cdf_sample(self.grid, self.cdf, n, generator)

    # -- reference quantities ---------------------------------------------
    def exact_cdf_on(self, x: torch.Tensor) -> torch.Tensor:
        """The reference CDF at arbitrary points, same shape as ``x``."""
        return interpolate_cdf(self.grid, self.cdf, x)

    @property
    def basin_mass_tensor(self) -> torch.Tensor:
        """Basin masses ordered as ``basin_labels``, shape ``(2,)``."""
        return torch.tensor([self.basin_masses[label]
                             for label in self.basin_labels],
                            dtype=torch.float64, device=self.device)

    @property
    def moment_tensor(self) -> torch.Tensor:
        orders = self._provenance["validation"]["moment_orders"]
        return torch.tensor([self.moments[f"m{int(order)}"] for order in orders],
                            dtype=torch.float64, device=self.device)


# ================================================================ validation
def _validate(target, provenance: dict, production: _Quadrature,
              sample_bank: torch.Tensor) -> dict:
    """One-off refinement, box-width, and sampling-noise validation."""
    settings = provenance["validation"]
    tolerances = settings["tolerances"]
    lo, hi = provenance["bounds"]
    orders = settings["moment_orders"]
    comparison = torch.linspace(lo, hi, settings["comparison_grid_points"],
                                dtype=torch.float64, device=target.device)
    checks: list[dict] = []

    # -- grid refinement ---------------------------------------------------
    grid_sizes = sorted(set(settings["grid_sizes"]))
    refinement = {"grid_sizes": grid_sizes, "finest": None, "per_grid": []}
    if grid_sizes:
        finest_size = grid_sizes[-1]
        finest = _Quadrature(target, lo, hi, finest_size)
        finest_cdf = finest.cdf_on(comparison)
        finest_moments = finest.moments(orders)
        finest_masses = finest.basin_masses()
        refinement["finest"] = finest_size
        worst_cdf = worst_moment = worst_mass = 0.0
        for size in grid_sizes:
            quadrature = (finest if size == finest_size
                          else _Quadrature(target, lo, hi, size))
            cdf_difference = float(
                (quadrature.cdf_on(comparison) - finest_cdf).abs().max().item())
            moment_difference = {
                key: abs(value - finest_moments[key])
                for key, value in quadrature.moments(orders).items()}
            mass_difference = {
                key: abs(value - finest_masses[key])
                for key, value in quadrature.basin_masses().items()}
            refinement["per_grid"].append({
                "n_grid": size,
                "cdf_max_abs_difference": cdf_difference,
                "moment_abs_difference": moment_difference,
                "basin_mass_abs_difference": mass_difference,
            })
            worst_cdf = max(worst_cdf, cdf_difference)
            worst_moment = max([worst_moment, *moment_difference.values()])
            worst_mass = max([worst_mass, *mass_difference.values()])
        checks += [
            passed_check("grid_refinement_cdf", worst_cdf,
                         tolerances["grid_refinement_cdf_max_abs"],
                         statistic="max_k max_x |F_k(x) - F_finest(x)|"),
            passed_check("grid_refinement_moments", worst_moment,
                         tolerances["grid_refinement_moment_abs"],
                         statistic="max_k max_j |E_k[x^j] - E_finest[x^j]|"),
            passed_check("grid_refinement_basin_masses", worst_mass,
                         tolerances["grid_refinement_basin_mass_abs"],
                         statistic="max_k max_b |p_k(b) - p_finest(b)|"),
        ]

    # -- wider box ---------------------------------------------------------
    wide_bounds = settings["wide_bounds"]
    wide: dict = {"bounds": wide_bounds}
    if len(wide_bounds) == 2:
        wide_lo, wide_hi = float(wide_bounds[0]), float(wide_bounds[1])
        if wide_lo > lo or wide_hi < hi:
            raise ValueError(
                "reference.validation.wide_bounds must contain reference.bounds")
        wide_quadrature = _Quadrature(target, wide_lo, wide_hi,
                                      provenance["n_grid"])
        edges = torch.tensor([lo, hi], dtype=torch.float64,
                             device=target.device)
        inside = wide_quadrature.cdf_on(edges)
        inside_mass = float((inside[1] - inside[0]).item())
        outside_mass = max(0.0, 1.0 - inside_mass)
        conditional = ((wide_quadrature.cdf_on(comparison) - inside[0])
                       / max(inside_mass, 1e-300))
        cdf_difference = float(
            (conditional - production.cdf_on(comparison)).abs().max().item())
        wide_moments = wide_quadrature.moments(orders)
        wide_masses = wide_quadrature.basin_masses()
        production_moments = production.moments(orders)
        production_masses = production.basin_masses()
        moment_difference = {key: abs(value - production_moments[key])
                             for key, value in wide_moments.items()}
        mass_difference = {key: abs(value - production_masses[key])
                           for key, value in wide_masses.items()}
        wide.update({
            "n_grid": provenance["n_grid"],
            "outside_narrow_box_mass": outside_mass,
            "cdf_max_abs_difference": cdf_difference,
            "moment_abs_difference": moment_difference,
            "basin_mass_abs_difference": mass_difference,
            "note": ("the wide-box CDF is renormalized to the narrow box before "
                     "comparison, so this isolates truncation from "
                     "discretisation"),
        })
        checks += [
            passed_check("wide_box_outside_mass", outside_mass,
                         tolerances["outside_narrow_box_mass"],
                         statistic="1 - (F_wide(hi) - F_wide(lo))"),
            passed_check("wide_box_cdf", cdf_difference,
                         tolerances["wide_box_cdf_max_abs"],
                         statistic="max_x |F_wide|box(x) - F_box(x)|"),
            passed_check("wide_box_moments", max(moment_difference.values()),
                         tolerances["wide_box_moment_abs"],
                         statistic="max_j |E_wide[x^j] - E_box[x^j]|"),
            passed_check("wide_box_basin_masses", max(mass_difference.values()),
                         tolerances["wide_box_basin_mass_abs"],
                         statistic="max_b |p_wide(b) - p_box(b)|"),
        ]

    # -- reference-vs-reference floors for every primary metric ------------
    replicates = int(settings["self_w2_replicates"])
    particles = int(settings["self_w2_particles"])
    self_w2: dict = {"replicates": replicates, "particles": particles,
                     "seed": settings["self_w2_seed"]}
    sampling_floors: dict = {
        "replicates": replicates,
        "particles": particles,
        "seed": settings["self_w2_seed"],
    }

    def summarize(values):
        mean = sum(values) / len(values)
        variance = (sum((value - mean) ** 2 for value in values)
                    / max(len(values) - 1, 1))
        return {"mean": mean, "sd": math.sqrt(variance),
                "min": min(values), "max": max(values),
                "values": values}

    if replicates > 0:
        (w2_exact_1d, mmd2_biased, ks_distance_cdf,
         median_heuristic) = import_metrics(
            "w2_exact_1d", "mmd2_biased", "ks_distance_cdf",
            "median_heuristic")
        take = min(particles, int(sample_bank.shape[0]))
        bank_index = torch.linspace(
            0, sample_bank.shape[0] - 1, take,
            device=sample_bank.device).round().long()
        metric_bank = sample_bank[bank_index]
        mmd_bandwidth = median_heuristic(metric_bank)
        w2_values, mmd_values, ks_values = [], [], []
        for index in range(replicates):
            first = frozen_generator(target.device,
                                     settings["self_w2_seed"] + 2 * index)
            second = frozen_generator(target.device,
                                      settings["self_w2_seed"] + 2 * index + 1)
            a = inverse_cdf_sample(production.grid, production.cdf, particles,
                                   first)
            b = inverse_cdf_sample(production.grid, production.cdf, particles,
                                   second)
            w2_values.append(w2_exact_1d(a, b))
            mmd_values.append(mmd2_biased(a, b, mmd_bandwidth))
            ks_values.extend([
                ks_distance_cdf(a[:, 0], production.grid, production.cdf),
                ks_distance_cdf(b[:, 0], production.grid, production.cdf),
            ])
        self_w2.update(summarize(w2_values))
        self_w2["note"] = (
            "reference-vs-reference W2 at the production particle count")
        sampling_floors.update({
            "mmd_bandwidth": mmd_bandwidth,
            "mmd_bandwidth_rule": (
                "median heuristic on the same frozen reference subsample "
                "used by runtime E1 metrics"),
            "W2_exact_1d": summarize(w2_values),
            "MMD2_biased": summarize(mmd_values),
            "KS": summarize(ks_values),
        })
        checks.append(passed_check(
            "self_w2_noise_floor", self_w2["mean"],
            tolerances["self_w2_mean"],
            statistic="mean W2(bank_a, bank_b)"))

    return {
        "grid_refinement": refinement,
        "wide_box": wide,
        "self_w2": self_w2,
        "sampling_floors": sampling_floors,
        "checks": checks,
        "validated": all(check["passed"] for check in checks),
    }


def build_reference(config: dict, target, directory: Path | None = None, *,
                    device=None, verbose: bool = False) -> DoubleWellReference:
    """Entry point named by ``reference.builder`` in ``configs/E1.yaml``."""
    return DoubleWellReference.build(config, target, directory, device=device,
                                     verbose=verbose)
