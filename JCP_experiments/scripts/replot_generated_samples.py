#!/usr/bin/env python3
"""Regenerate sample figures without rerunning any experiment sampler.

Method samples are read from ``results/<experiment>/positions.csv`` and written
to a separate figure set:

* E1: one figure per method with a smooth reference density above a rug/strip of
  that method's saved samples.
* E2--E4: one scatter figure per method and one standalone smooth reference
  density figure per experiment.

Reference densities are evaluated directly where the saved experiment has a
tractable target density (E1--E3).  For E4, the script regenerates only the same
fixed-seed 200,000-point direct-SNIS importance-reference construction used by
the experiment, caches its weighted qbar cloud, and evaluates a
phase-conditioned binned KDE.  This is reference integration, not a sampler
trajectory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import logsumexp


JCP_ROOT = Path(__file__).resolve().parents[1]
if str(JCP_ROOT) not in sys.path:
    sys.path.insert(0, str(JCP_ROOT))

from src.plotting import (  # noqa: E402
    METHOD_STYLE,
    SIMPLE_LABELS,
    apply_style,
    load_positions_csv,
)


EXPERIMENTS = ("double_well", "mog40", "mb3well_10d", "coupled_phi4")

# Fixed scientific display domains.  These are the declared sampling/basin-map
# domains, not min/max ranges inferred from the reference draws.  Keeping them
# fixed exposes leakage and boundary contact instead of cropping them away.
DOMAINS = {
    "double_well": ((-5.2, 5.2),),
    "mog40": ((-65.0, 65.0), (-65.0, 65.0)),
    "mb3well_10d": ((-2.0, 1.9), (-1.3, 2.6)),
    # qbar can leave the [-4, 4]^2 basin-map domain.  Use the full numerical
    # envelope so Raw-CP leakage remains visible instead of being cropped.
    "coupled_phi4": ((-5.0, 5.0), (-5.0, 5.0)),
}

AXIS_LABELS = {
    "double_well": (r"$x$",),
    "mog40": (r"$x_1$", r"$x_2$"),
    "mb3well_10d": (r"$z_1$", r"$z_2$"),
    "coupled_phi4": (r"$\bar q_1$", r"$\bar q_2$"),
}

REFERENCE_LABELS = {
    "double_well": "exact target density",
    "mog40": "exact Gaussian-mixture density",
    "mb3well_10d": "target density in active plane",
    "coupled_phi4": "direct-SNIS weighted qbar density",
}


def _manifest_path(path: Path) -> str:
    """Use repository-relative paths for portable manifests when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(JCP_ROOT).as_posix()
    except ValueError:
        return str(resolved)


# Atom-bank size A per finite-bank experiment. The realised LSC arms are one
# family labelled "LSC-CP-RA (k)" by the atoms used per step: single-atom RA is
# k=1, multi-atom MA is k=A. E1/E2 have continuous jump laws (no finite bank),
# so their single realised arm stays plain "LSC-CP-RA".
_ARM_ATOMS = {"mb3well_10d": 4, "coupled_phi4": 8}


def _method_label(method: str, experiment: str, manifest: dict) -> str:
    A = _ARM_ATOMS.get(experiment)
    if method == "LSC-CP-RA":
        return "LSC-CP-RA (1)" if A else "LSC-CP-RA"
    if method == "LSC-CP-MA":
        return f"LSC-CP-RA ({A})" if A else "LSC-CP-MA"
    overrides = (manifest.get("plot") or {}).get("label_overrides") or {}
    return overrides.get(method, SIMPLE_LABELS.get(method, method))


def _method_color(method: str) -> str:
    return METHOD_STYLE.get(method, {}).get("color", "#444444")


def _normalise_density_1d(x: np.ndarray, density: np.ndarray) -> np.ndarray:
    area = float(np.trapezoid(density, x))
    if not math.isfinite(area) or area <= 0:
        raise ValueError("reference density has non-positive integral")
    return density / area


def _normalise_density_2d(
    x: np.ndarray, y: np.ndarray, density: np.ndarray
) -> np.ndarray:
    area = float(np.trapezoid(np.trapezoid(density, x, axis=1), y, axis=0))
    if not math.isfinite(area) or area <= 0:
        raise ValueError("reference density has non-positive integral")
    return density / area


def _reference_density_1d(
    experiment: str, manifest: dict, grid_size: int
) -> tuple[np.ndarray, np.ndarray]:
    if experiment != "double_well":
        raise ValueError(f"no 1-D density rule for {experiment}")
    beta = float(manifest["config"]["beta"])
    (xlim,) = DOMAINS[experiment]
    x = np.linspace(*xlim, max(1200, grid_size * 4))
    log_density = -beta * (x * x - 1.0) ** 2
    density = np.exp(log_density - log_density.max())
    return x, _normalise_density_1d(x, density)


def _exact_mog40_density(
    results_dir: Path, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    modes_path = results_dir / "modes.csv"
    if not modes_path.is_file():
        raise FileNotFoundError(f"missing E2 modes: {modes_path}")
    modes = np.loadtxt(modes_path, delimiter=",", skiprows=1)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    density = np.zeros_like(xx)
    coefficient = 1.0 / (len(modes) * 2.0 * math.pi)
    for mu_x, mu_y in modes:
        density += np.exp(-0.5 * ((xx - mu_x) ** 2 + (yy - mu_y) ** 2))
    return coefficient * density


def _exact_mb3_density(
    beta: float, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    import torch

    from src.potentials import mb3_2d

    xx, yy = np.meshgrid(x, y, indexing="xy")
    points = torch.as_tensor(
        np.column_stack([xx.ravel(), yy.ravel()]), dtype=torch.float64
    )
    with torch.no_grad():
        potential = mb3_2d(points).cpu().numpy().reshape(xx.shape)
    log_density = -beta * potential
    density = np.exp(log_density - np.nanmax(log_density))
    return _normalise_density_2d(x, y, density)


def _e4_reference_signature(
    manifest: dict, n_proposals: int, seed: int
) -> dict:
    from src.potentials import CoupledPhi4, PHI4_MINIMA, PHI4_W_COEFFS

    return {
        "schema_version": 1,
        "method": "direct_snis_weighted_qbar",
        "backend": "torch_cpu",
        "dtype": "float64",
        "n_proposals": int(n_proposals),
        "seed": int(seed),
        "beta": float(manifest["config"]["beta"]),
        "n_sites": int(CoupledPhi4.Ns),
        "dimension": int(CoupledPhi4.d),
        "kappa": float(CoupledPhi4.kappa),
        "delta": 1.0 / float(CoupledPhi4.Ns),
        "site_potential_coefficients": {
            key: float(value) for key, value in PHI4_W_COEFFS.items()
        },
        "phase_initial_minima": {
            phase: [float(value) for value in values[0]]
            for phase, values in PHI4_MINIMA.items()
        },
    }


def _load_e4_reference_cache(
    cache_path: Path, expected_signature: dict
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            qbar = np.asarray(cache["qbar"], dtype=np.float64)
            weights = np.asarray(cache["weights"], dtype=np.float64)
            metadata = json.loads(str(cache["metadata_json"].item()))
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None
    if metadata.get("signature") != expected_signature:
        return None
    if qbar.shape != (expected_signature["n_proposals"], 2):
        return None
    if weights.shape != (expected_signature["n_proposals"],):
        return None
    if (
        not np.isfinite(qbar).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0.0)
        or not np.isclose(weights.sum(), 1.0, rtol=1e-10, atol=1e-12)
    ):
        return None
    return qbar, weights, metadata


def _write_e4_reference_cache(
    cache_path: Path,
    qbar: np.ndarray,
    weights: np.ndarray,
    metadata: dict,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=cache_path.parent,
            prefix=f".{cache_path.stem}-",
            suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        np.savez_compressed(
            temporary_path,
            qbar=np.asarray(qbar, dtype=np.float64),
            weights=np.asarray(weights, dtype=np.float64),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        os.replace(temporary_path, cache_path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _build_e4_direct_snis_reference(
    manifest: dict,
    n_proposals: int,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Regenerate the experiment's fixed direct-SNIS qbar reference on CPU.

    Proposal generation is done once at full size to preserve one-shot CPU RNG
    ordering.  Torch RNG streams are backend-specific, so this reproduces the
    production construction and seed but not the archived CUDA draws bitwise.
    Exact target and proposal log densities are evaluated in bounded-memory
    batches.
    """
    import torch
    from torch.autograd.functional import hessian

    from src.potentials import (
        CoupledPhi4,
        PHI4_MINIMA,
        newton_refine,
        phi4_W,
        phi4_W_grad,
    )
    from src.references import LaplaceMixture

    beta = float(manifest["config"]["beta"])
    dtype = torch.float64
    device = torch.device("cpu")
    potential = CoupledPhi4()
    phases = ("--", "-+", "+-", "++")
    minima = []
    for phase in phases:
        initial = torch.tensor(
            PHI4_MINIMA[phase][0], dtype=dtype, device=device
        )
        minima.append(newton_refine(phi4_W_grad, initial))
    minima_2d = torch.stack(minima)
    means = (
        minima_2d.unsqueeze(1)
        .expand(len(phases), potential.Ns, 2)
        .reshape(len(phases), potential.d)
        .contiguous()
    )
    hessians = []
    for mean in means:
        current = hessian(
            lambda state: potential._V_raw(state.unsqueeze(0))[0],
            mean.clone(),
        )
        hessians.append(0.5 * (current + current.T))
    hessians_24d = torch.stack(hessians)
    energies = phi4_W(minima_2d)
    proposal = LaplaceMixture(means, hessians_24d, energies, beta)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    with torch.no_grad():
        # Equivalent to LaplaceMixture.sample, but apply the four shared
        # Cholesky factors component-wise instead of materialising one 24x24
        # factor per proposal.
        components = torch.multinomial(
            proposal.weights.expand(n_proposals, -1),
            1,
            generator=generator,
        ).squeeze(1)
        noise = torch.randn(
            n_proposals,
            proposal.d,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        points = torch.empty_like(noise)
        for component in range(proposal.means.shape[0]):
            mask = components == component
            points[mask] = proposal.means[component] + torch.einsum(
                "ij,nj->ni",
                proposal.chol[component],
                noise[mask],
            )
        del components, noise
        qbar = (
            points.reshape(n_proposals, potential.Ns, 2)
            .mean(dim=1)
            .cpu()
            .numpy()
        )
        log_weights = np.empty(n_proposals, dtype=np.float64)
        for start in range(0, n_proposals, batch_size):
            stop = min(start + batch_size, n_proposals)
            block = points[start:stop]
            block_log_weights = (
                -beta * potential._V_raw(block) - proposal.log_q(block)
            )
            log_weights[start:stop] = block_log_weights.cpu().numpy()

    if not np.isfinite(log_weights).all():
        n_bad = int((~np.isfinite(log_weights)).sum())
        raise FloatingPointError(
            f"E4 direct-SNIS produced {n_bad}/{n_proposals} nonfinite log weights"
        )
    weights = np.exp(log_weights - logsumexp(log_weights))
    ess = float(1.0 / np.sum(weights * weights))
    entropy_ess = float(
        np.exp(-np.sum(weights * np.log(np.maximum(weights, 1e-300))))
    )
    phase_labels = (qbar[:, 0] >= 0.0).astype(np.int64) * 2
    phase_labels += (qbar[:, 1] >= 0.0).astype(np.int64)
    phase_masses = [
        float(weights[phase_labels == label].sum()) for label in range(4)
    ]
    weighted_mean = np.sum(weights[:, None] * qbar, axis=0)
    production_diagnostics = (
        (manifest.get("reference") or {}).get("reference_diagnostics") or {}
    )
    metadata = {
        "signature": _e4_reference_signature(manifest, n_proposals, seed),
        "runtime": {
            "torch_version": str(torch.__version__),
            "device": str(device),
            "rng_backend": "torch_cpu",
        },
        "diagnostics": {
            "proposal_ess": ess,
            "proposal_ess_fraction": ess / n_proposals,
            "entropy_ess": entropy_ess,
            "max_normalized_weight": float(weights.max()),
            "weighted_qbar_mean": [float(value) for value in weighted_mean],
            "weighted_sign_phase_masses": phase_masses,
            "production_manifest_proposal_ess": production_diagnostics.get(
                "proposal_ess"
            ),
        },
    }
    return qbar, weights, metadata


def _load_or_build_e4_reference(
    manifest: dict,
    cache_path: Path,
    *,
    n_proposals: int,
    seed: int,
    batch_size: int,
    refresh: bool,
) -> tuple[np.ndarray, np.ndarray, dict, bool]:
    signature = _e4_reference_signature(manifest, n_proposals, seed)
    cached = None if refresh else _load_e4_reference_cache(cache_path, signature)
    if cached is not None:
        qbar, weights, metadata = cached
        return qbar, weights, metadata, True
    qbar, weights, metadata = _build_e4_direct_snis_reference(
        manifest,
        n_proposals,
        seed,
        batch_size,
    )
    _write_e4_reference_cache(cache_path, qbar, weights, metadata)
    return qbar, weights, metadata, False


def _weighted_phase_conditioned_kde_reference_density(
    reference: np.ndarray,
    weights: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Evaluate a binned, full-covariance KDE for each E4 sign phase.

    Phase conditioning prevents the separation among the four modes from
    inflating a global bandwidth and creating artificial bridges.  Each phase
    uses a weighted Scott bandwidth based on its conditional effective sample
    size; FFT convolution makes the 200,000-point estimate inexpensive.
    """
    if reference.shape != (weights.size, 2):
        raise ValueError("E4 qbar reference and weights have incompatible shapes")
    if len(x) < 2 or len(y) < 2:
        raise ValueError("E4 reference grid must have at least two points per axis")
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    if (
        not np.allclose(np.diff(x), dx)
        or not np.allclose(np.diff(y), dy)
        or dx <= 0.0
        or dy <= 0.0
    ):
        raise ValueError("E4 reference KDE requires a uniform increasing grid")

    x_edges = np.concatenate(
        ([x[0] - 0.5 * dx], 0.5 * (x[:-1] + x[1:]), [x[-1] + 0.5 * dx])
    )
    y_edges = np.concatenate(
        ([y[0] - 0.5 * dy], 0.5 * (y[:-1] + y[1:]), [y[-1] + 0.5 * dy])
    )
    kernel_x = np.arange(-(len(x) - 1), len(x), dtype=np.float64) * dx
    kernel_y = np.arange(-(len(y) - 1), len(y), dtype=np.float64) * dy
    kernel_xx, kernel_yy = np.meshgrid(kernel_x, kernel_y, indexing="xy")
    kernel_points = np.stack([kernel_xx, kernel_yy], axis=-1)

    phase = (reference[:, 0] >= 0.0).astype(np.int64) * 2
    phase += (reference[:, 1] >= 0.0).astype(np.int64)
    density = np.zeros((len(y), len(x)), dtype=np.float64)
    bandwidth_floor_squared = (0.65 * min(dx, dy)) ** 2

    for label in range(4):
        mask = phase == label
        block = reference[mask]
        block_weights = weights[mask]
        phase_mass = float(block_weights.sum())
        if block.shape[0] < 3 or phase_mass <= 0.0:
            continue
        conditional_weights = block_weights / phase_mass
        mean = np.sum(conditional_weights[:, None] * block, axis=0)
        centered = block - mean
        covariance = (
            centered * conditional_weights[:, None]
        ).T @ centered
        effective_n = float(1.0 / np.sum(conditional_weights**2))
        scott_factor = effective_n ** (-1.0 / 6.0)
        bandwidth_covariance = covariance * scott_factor**2
        eigenvalues, eigenvectors = np.linalg.eigh(bandwidth_covariance)
        eigenvalues = np.maximum(eigenvalues, bandwidth_floor_squared)
        bandwidth_covariance = (
            eigenvectors * eigenvalues[None, :]
        ) @ eigenvectors.T
        inverse_covariance = np.linalg.inv(bandwidth_covariance)
        quadratic = np.einsum(
            "...i,ij,...j->...",
            kernel_points,
            inverse_covariance,
            kernel_points,
        )
        kernel = np.exp(-0.5 * quadratic)
        kernel /= float(kernel.sum() * dx * dy)
        histogram, _, _ = np.histogram2d(
            block[:, 1],
            block[:, 0],
            bins=(y_edges, x_edges),
            weights=block_weights,
        )
        density += fftconvolve(histogram, kernel, mode="same")

    density = np.maximum(density, 0.0)
    return _normalise_density_2d(x, y, density)


def _reference_density_2d(
    experiment: str,
    results_dir: Path,
    reference: np.ndarray,
    manifest: dict,
    grid_size: int,
    *,
    e4_cache_path: Path | None,
    e4_reference_proposals: int,
    e4_reference_seed: int,
    e4_reference_batch_size: int,
    refresh_e4_reference: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    xlim, ylim = DOMAINS[experiment]
    effective_grid_size = (
        max(grid_size, 801) if experiment == "coupled_phi4" else grid_size
    )
    x = np.linspace(*xlim, effective_grid_size)
    y = np.linspace(*ylim, effective_grid_size)
    details: dict = {}
    if experiment == "mog40":
        density = _exact_mog40_density(results_dir, x, y)
    elif experiment == "mb3well_10d":
        density = _exact_mb3_density(float(manifest["config"]["beta"]), x, y)
    elif experiment == "coupled_phi4":
        if e4_cache_path is None:
            raise ValueError("E4 reference cache path is required")
        qbar, weights, metadata, cache_reused = _load_or_build_e4_reference(
            manifest,
            e4_cache_path,
            n_proposals=e4_reference_proposals,
            seed=e4_reference_seed,
            batch_size=e4_reference_batch_size,
            refresh=refresh_e4_reference,
        )
        density = _weighted_phase_conditioned_kde_reference_density(
            qbar, weights, x, y
        )
        details = {
            "reference_cache": _manifest_path(e4_cache_path),
            "reference_cache_reused": cache_reused,
            "reference_grid_size": effective_grid_size,
            "reference_metadata": metadata,
        }
    else:
        raise ValueError(f"no 2-D density rule for {experiment}")
    return x, y, density, details


def _save_figure(
    fig: plt.Figure, base: Path, *, overwrite: bool, dpi: int
) -> list[str]:
    outputs = [base.with_suffix(".png"), base.with_suffix(".pdf")]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing figure(s): {names}")
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs[0], dpi=dpi, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    plt.close(fig)
    return [_manifest_path(path) for path in outputs]


def _plot_e1_method(
    method: str,
    samples: np.ndarray,
    reference_x: np.ndarray,
    reference_density: np.ndarray,
    label: str,
    output_dir: Path,
    *,
    overwrite: bool,
    dpi: int,
) -> list[str]:
    apply_style()
    fig, (ax_density, ax_rug) = plt.subplots(
        2,
        1,
        figsize=(4.8, 3.75),
        sharex=True,
        gridspec_kw={"height_ratios": [4.0, 0.8], "hspace": 0.04},
    )
    ax_density.plot(
        reference_x,
        reference_density,
        color="#666666",
        lw=2.1,
    )
    ax_density.set_ylabel("density")
    ax_density.set_ylim(bottom=0.0)
    ax_density.set_title(label, fontsize=10)
    ax_density.tick_params(labelbottom=False)

    seed = sum((i + 1) * ord(char) for i, char in enumerate(method))
    rng = np.random.default_rng(seed)
    n_display = min(4000, samples.shape[0])
    display_idx = rng.choice(samples.shape[0], size=n_display, replace=False)
    displayed = samples[display_idx]
    jitter = rng.uniform(0.12, 0.88, size=n_display)
    ax_rug.scatter(
        displayed[:, 0],
        jitter,
        s=1.2,
        alpha=0.14,
        linewidths=0,
        color=_method_color(method),
        rasterized=True,
    )
    ax_rug.set_xlim(*DOMAINS["double_well"][0])
    ax_rug.set_ylim(0.0, 1.0)
    ax_rug.set_yticks([])
    ax_rug.set_xlabel(AXIS_LABELS["double_well"][0])
    ax_rug.grid(False)

    return _save_figure(
        fig,
        output_dir / f"double_well_samples_{method}",
        overwrite=overwrite,
        dpi=dpi,
    )


def _plot_2d_samples(
    experiment: str,
    method: str,
    samples: np.ndarray,
    label: str,
    output_dir: Path,
    *,
    overwrite: bool,
    dpi: int,
) -> list[str]:
    apply_style()
    fig, ax = plt.subplots(figsize=(4.25, 4.0))
    ax.scatter(
        samples[:, 0],
        samples[:, 1],
        s=1.5,
        alpha=0.24,
        linewidths=0,
        color=_method_color(method),
        rasterized=True,
    )
    xlim, ylim = DOMAINS[experiment]
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(AXIS_LABELS[experiment][0])
    ax.set_ylabel(AXIS_LABELS[experiment][1])
    ax.set_title(label, fontsize=10)
    ax.grid(False)
    return _save_figure(
        fig,
        output_dir / f"{experiment}_samples_{method}",
        overwrite=overwrite,
        dpi=dpi,
    )


def _plot_2d_reference_density(
    experiment: str,
    x: np.ndarray,
    y: np.ndarray,
    density: np.ndarray,
    output_dir: Path,
    *,
    overwrite: bool,
    dpi: int,
) -> list[str]:
    apply_style()
    fig, ax = plt.subplots(figsize=(4.55, 4.0))
    xx, yy = np.meshgrid(x, y, indexing="xy")
    mesh = ax.pcolormesh(
        xx,
        yy,
        density,
        cmap="viridis",
        shading="auto",
        rasterized=True,
    )
    positive = density[density > 0]
    if positive.size:
        levels = np.linspace(float(positive.min()), float(density.max()), 12)[1:-1]
        if levels.size:
            ax.contour(xx, yy, density, levels=levels, colors="white", linewidths=0.35)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("probability density")
    xlim, ylim = DOMAINS[experiment]
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(AXIS_LABELS[experiment][0])
    ax.set_ylabel(AXIS_LABELS[experiment][1])
    ax.set_title("Reference", fontsize=10)
    ax.grid(False)
    return _save_figure(
        fig,
        output_dir / f"{experiment}_reference_density",
        overwrite=overwrite,
        dpi=dpi,
    )


def replot_experiment(
    experiment: str,
    results_root: Path,
    output_root: Path,
    *,
    overwrite: bool,
    dpi: int,
    grid_size: int,
    e4_reference_proposals: int,
    e4_reference_seed: int,
    e4_reference_batch_size: int,
    refresh_e4_reference: bool,
) -> dict:
    results_dir = results_root / experiment
    positions_path = results_dir / "positions.csv"
    manifest_path = results_dir / "manifest.json"
    if not positions_path.is_file():
        raise FileNotFoundError(f"missing positions CSV: {positions_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")

    positions = load_positions_csv(str(positions_path))
    if "reference" not in positions:
        raise ValueError(f"{positions_path} has no reference block")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    methods = [name for name in positions if name != "reference"]
    output_dir = output_root / experiment
    written: list[str] = []

    if experiment == "double_well":
        reference_x, reference_density = _reference_density_1d(
            experiment, manifest, grid_size
        )
        for method in methods:
            written.extend(
                _plot_e1_method(
                    method,
                    positions[method],
                    reference_x,
                    reference_density,
                    _method_label(method, experiment, manifest),
                    output_dir,
                    overwrite=overwrite,
                    dpi=dpi,
                )
            )
    else:
        for method in methods:
            written.extend(
                _plot_2d_samples(
                    experiment,
                    method,
                    positions[method],
                    _method_label(method, experiment, manifest),
                    output_dir,
                    overwrite=overwrite,
                    dpi=dpi,
                )
            )
        e4_cache_path = (
            output_dir / "direct_snis_qbar_reference.npz"
            if experiment == "coupled_phi4"
            else None
        )
        x, y, density, reference_details = _reference_density_2d(
            experiment,
            results_dir,
            positions["reference"],
            manifest,
            grid_size,
            e4_cache_path=e4_cache_path,
            e4_reference_proposals=e4_reference_proposals,
            e4_reference_seed=e4_reference_seed,
            e4_reference_batch_size=e4_reference_batch_size,
            refresh_e4_reference=refresh_e4_reference,
        )
        written.extend(
            _plot_2d_reference_density(
                experiment,
                x,
                y,
                density,
                output_dir,
                overwrite=overwrite,
                dpi=dpi,
            )
        )

    report = {
        "experiment": experiment,
        "source": _manifest_path(positions_path),
        "methods": methods,
        "reference_density": REFERENCE_LABELS[experiment],
        "figure_pairs": len(written) // 2,
        "files": written,
    }
    if experiment != "double_well":
        report.update(reference_details)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENTS,
        default=list(EXPERIMENTS),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=JCP_ROOT / "results",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=JCP_ROOT / "figures" / "generated_samples",
    )
    parser.add_argument("--grid-size", type=int, default=401)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--e4-reference-proposals", type=int, default=200_000)
    parser.add_argument("--e4-reference-seed", type=int, default=31_337)
    parser.add_argument("--e4-reference-batch-size", type=int, default=25_000)
    parser.add_argument("--refresh-e4-reference", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.grid_size < 100:
        raise ValueError("--grid-size must be at least 100")
    if args.dpi < 72:
        raise ValueError("--dpi must be at least 72")
    if args.e4_reference_proposals < 1:
        raise ValueError("--e4-reference-proposals must be positive")
    if args.e4_reference_batch_size < 1:
        raise ValueError("--e4-reference-batch-size must be positive")

    reports = []
    for experiment in args.experiments:
        report = replot_experiment(
            experiment,
            args.results_root.resolve(),
            args.output_root.resolve(),
            overwrite=args.overwrite,
            dpi=args.dpi,
            grid_size=args.grid_size,
            e4_reference_proposals=args.e4_reference_proposals,
            e4_reference_seed=args.e4_reference_seed,
            e4_reference_batch_size=args.e4_reference_batch_size,
            refresh_e4_reference=args.refresh_e4_reference,
        )
        reports.append(report)
        print(
            f"{experiment}: wrote {report['figure_pairs']} figure pairs "
            f"from {report['source']}"
        )

    manifest_path = args.output_root.resolve() / "generated_sample_plots_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest_path}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "source_kind": (
                    "saved method positions.csv plus direct-SNIS E4 reference "
                    "integration; no sampler rerun"
                ),
                "reports": reports,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
