#!/usr/bin/env python3
"""Regenerate manuscript sample/reference comparisons without rerunning samplers.

Method samples are read from ``results/<experiment>/positions.csv``:

* E1: two overlays, respectively comparing the reference and every manuscript
  method by probability density function (PDF) and cumulative distribution
  function (CDF).
* E2--E4: one three-row figure per experiment.  The reference density occupies
  the middle of the first row; the six manuscript methods occupy a 2-by-3 grid
  in the second and third rows.

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

# Matplotlib needs a writable cache in the managed desktop environment.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/jcp-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import fftconvolve
from scipy.special import logsumexp


JCP_ROOT = Path(__file__).resolve().parents[1]
if str(JCP_ROOT) not in sys.path:
    sys.path.insert(0, str(JCP_ROOT))

from src.plotting import (  # noqa: E402
    load_positions_csv,
)
from src.manuscript import EXPERIMENTS as RELEASE_EXPERIMENTS  # noqa: E402


EXPERIMENTS = tuple(RELEASE_EXPERIMENTS)

MANUSCRIPT_METHODS = {
    key: spec.methods for key, spec in RELEASE_EXPERIMENTS.items()
}

# Same method encodings as replot_manuscript_figures.py.
METHOD_STYLE = {
    "ULA": dict(color="#767676", linestyle="-", marker="o"),
    "BAOAB": dict(color="#42949E", linestyle="--", marker="s"),
    "FLA": dict(color="#009E73", linestyle=":", marker="^"),
    "PT": dict(color="#9A4D8E", linestyle="-.", marker="D"),
    "CP": dict(color="#B64342", linestyle=(0, (5, 2)), marker="X"),
    "LSC-CP": dict(color="#0F4D92", linestyle="-", marker="*"),
    "LSC-CP-RA": dict(
        color="#3775BA", linestyle=(0, (3, 1, 1, 1)), marker="P"
    ),
    "LSC-CP-MA": dict(
        color="#3775BA", linestyle=(0, (3, 1, 1, 1)), marker="P"
    ),
}

# Fixed scientific display domains.  These are the declared sampling/basin-map
# domains, not min/max ranges inferred from the reference draws.  Keeping them
# fixed exposes leakage and boundary contact instead of cropping them away.
DOMAINS = {
    # Manuscript zoom: the relevant well/barrier structure lies in [-2, 2].
    # Empirical densities are still divided by the full saved sample count, so
    # mass outside the displayed range is not silently renormalized away.
    "double_well": ((-2.0, 2.0),),
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
    if method == "BAOAB":
        return "ULD"
    if method == "CP":
        return "Raw-CP"
    A = _ARM_ATOMS.get(experiment)
    if method == "LSC-CP-RA":
        return "LSC-CP-RA (1)" if A else "LSC-CP-RA"
    if method == "LSC-CP-MA":
        return f"LSC-CP-RA ({A})" if A else "LSC-CP-MA"
    return method


def _method_color(method: str) -> str:
    return METHOD_STYLE.get(method, {}).get("color", "#444444")


def _apply_manuscript_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "lines.linewidth": 2.2,
        "legend.frameon": False,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.3,
        "ytick.major.width": 1.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })


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
    fig: plt.Figure,
    basename: str,
    output_root: Path,
    *,
    overwrite: bool,
    dpi: int,
) -> list[str]:
    outputs = [
        output_root / "png" / f"{basename}.png",
        output_root / "pdf" / f"{basename}.pdf",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing figure(s): {names}")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs[0], dpi=dpi, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    plt.close(fig)
    return [_manifest_path(path) for path in outputs]


def _empirical_density_1d(
    samples: np.ndarray,
    grid: np.ndarray,
    bandwidth: float = 0.04,
) -> np.ndarray:
    dx = float(grid[1] - grid[0])
    edges = np.concatenate((
        [grid[0] - 0.5 * dx],
        0.5 * (grid[:-1] + grid[1:]),
        [grid[-1] + 0.5 * dx],
    ))
    counts, _ = np.histogram(samples[:, 0], bins=edges)
    smoothed = gaussian_filter1d(
        counts.astype(float),
        sigma=max(bandwidth / dx, 0.75),
        mode="constant",
    )
    return smoothed / (samples.shape[0] * dx)


def _plot_e1_overlays(
    positions: dict[str, np.ndarray],
    methods: tuple[str, ...],
    manifest: dict,
    reference_x: np.ndarray,
    reference_density: np.ndarray,
    output_root: Path,
    *,
    overwrite: bool,
    dpi: int,
) -> list[str]:
    _apply_manuscript_style()
    outputs: list[str] = []

    fig_pdf, ax_pdf = plt.subplots(figsize=(7.4, 4.7))
    ax_pdf.plot(
        reference_x,
        reference_density,
        color="#222222",
        linestyle="-",
        linewidth=2.8,
        label="Reference",
        zorder=5,
    )
    for method in methods:
        style = METHOD_STYLE[method]
        density = _empirical_density_1d(positions[method], reference_x)
        ax_pdf.plot(
            reference_x,
            density,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=max(1, len(reference_x) // 14),
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=_method_label(method, "double_well", manifest),
        )
    ax_pdf.set_xlim(*DOMAINS["double_well"][0])
    ax_pdf.set_ylim(bottom=0.0)
    ax_pdf.set_xlabel(AXIS_LABELS["double_well"][0])
    ax_pdf.set_ylabel("Probability density")
    ax_pdf.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        columnspacing=1.2,
        handlelength=2.4,
    )
    fig_pdf.tight_layout(rect=(0, 0, 1, 0.88))
    outputs.extend(_save_figure(
        fig_pdf,
        "double_well_generated_pdf",
        output_root,
        overwrite=overwrite,
        dpi=dpi,
    ))

    reference_cdf = cumulative_trapezoid(
        reference_density, reference_x, initial=0.0
    )
    reference_cdf /= reference_cdf[-1]
    fig_cdf, ax_cdf = plt.subplots(figsize=(7.4, 4.7))
    ax_cdf.plot(
        reference_x,
        reference_cdf,
        color="#222222",
        linestyle="-",
        linewidth=2.8,
        label="Reference",
        zorder=5,
    )
    for method in methods:
        style = METHOD_STYLE[method]
        values = np.sort(positions[method][:, 0])
        cdf = np.searchsorted(values, reference_x, side="right") / values.size
        ax_cdf.plot(
            reference_x,
            cdf,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=max(1, len(reference_x) // 14),
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=_method_label(method, "double_well", manifest),
        )
    ax_cdf.set_xlim(*DOMAINS["double_well"][0])
    ax_cdf.set_ylim(-0.015, 1.015)
    ax_cdf.set_xlabel(AXIS_LABELS["double_well"][0])
    ax_cdf.set_ylabel("Cumulative probability")
    ax_cdf.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        columnspacing=1.2,
        handlelength=2.4,
    )
    fig_cdf.tight_layout(rect=(0, 0, 1, 0.88))
    outputs.extend(_save_figure(
        fig_cdf,
        "double_well_generated_cdf",
        output_root,
        overwrite=overwrite,
        dpi=dpi,
    ))
    return outputs


def _empirical_density_2d(
    samples: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    smoothing_bins: float = 1.4,
) -> np.ndarray:
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    x_edges = np.concatenate((
        [x[0] - 0.5 * dx],
        0.5 * (x[:-1] + x[1:]),
        [x[-1] + 0.5 * dx],
    ))
    y_edges = np.concatenate((
        [y[0] - 0.5 * dy],
        0.5 * (y[:-1] + y[1:]),
        [y[-1] + 0.5 * dy],
    ))
    counts, _, _ = np.histogram2d(
        samples[:, 1],
        samples[:, 0],
        bins=(y_edges, x_edges),
    )
    smoothed = gaussian_filter(
        counts.astype(float),
        sigma=smoothing_bins,
        mode="constant",
    )
    # Divide by the full sample count rather than the in-domain count, so any
    # escaped mass remains visible as missing density instead of being silently
    # renormalized away.
    return smoothed / (samples.shape[0] * dx * dy)


def _plot_2d_density_grid(
    experiment: str,
    positions: dict[str, np.ndarray],
    methods: tuple[str, ...],
    manifest: dict,
    x: np.ndarray,
    y: np.ndarray,
    reference_density: np.ndarray,
    output_root: Path,
    *,
    overwrite: bool,
    dpi: int,
) -> list[str]:
    _apply_manuscript_style()
    method_densities = {
        method: _empirical_density_2d(positions[method], x, y)
        for method in methods
    }
    positive_blocks = [
        block[block > 0]
        for block in [reference_density, *method_densities.values()]
        if np.any(block > 0)
    ]
    if not positive_blocks:
        raise ValueError(f"{experiment} has no positive density values")
    all_positive = np.concatenate(positive_blocks)
    vmax = float(np.quantile(all_positive, 0.997))
    vmax = max(vmax, float(reference_density.max()) * 0.25)
    vmin = max(vmax * 1.0e-4, np.finfo(float).tiny)
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

    fig = plt.figure(figsize=(11.0, 10.1))
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=(1.0, 1.0, 1.0),
        hspace=0.34,
        wspace=0.24,
    )
    reference_ax = fig.add_subplot(grid[0, 1])
    method_axes = [
        fig.add_subplot(grid[row, col])
        for row in (1, 2)
        for col in range(3)
    ]
    xx, yy = np.meshgrid(x, y, indexing="xy")
    xlim, ylim = DOMAINS[experiment]

    def draw_panel(ax, density: np.ndarray, title: str) -> None:
        ax.pcolormesh(
            xx,
            yy,
            np.maximum(density, vmin),
            cmap="viridis",
            norm=norm,
            shading="auto",
            rasterized=True,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(AXIS_LABELS[experiment][0])
        ax.set_ylabel(AXIS_LABELS[experiment][1])
        ax.set_title(title, pad=4)
        ax.grid(False)

    draw_panel(reference_ax, reference_density, "Reference")
    for ax, method in zip(method_axes, methods):
        draw_panel(
            ax,
            method_densities[method],
            _method_label(method, experiment, manifest),
        )

    # Empty first-row side cells preserve the requested centered reference.
    for col in (0, 2):
        empty_ax = fig.add_subplot(grid[0, col])
        empty_ax.axis("off")

    scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis")
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=[reference_ax, *method_axes],
        fraction=0.025,
        pad=0.025,
        aspect=35,
    )
    colorbar.set_label("Probability density")
    return _save_figure(
        fig,
        f"{experiment}_generated_density_grid",
        output_root,
        overwrite=overwrite,
        dpi=dpi,
    )


def _plot_2d_scatter_grid(
    experiment: str,
    positions: dict[str, np.ndarray],
    methods: tuple[str, ...],
    manifest: dict,
    x: np.ndarray,
    y: np.ndarray,
    reference_density: np.ndarray,
    output_root: Path,
    *,
    overwrite: bool,
    dpi: int,
) -> list[str]:
    """Reference density above a 2-by-3 grid of generated sample points."""
    _apply_manuscript_style()
    positive = reference_density[reference_density > 0]
    if not positive.size:
        raise ValueError(f"{experiment} reference has no positive density values")
    vmax = float(reference_density.max())
    vmin = max(vmax * 1.0e-4, float(positive.min()))
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

    fig = plt.figure(figsize=(11.0, 10.1))
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=(1.0, 1.0, 1.0),
        hspace=0.34,
        wspace=0.24,
    )
    reference_ax = fig.add_subplot(grid[0, 1])
    method_axes = [
        fig.add_subplot(grid[row, col])
        for row in (1, 2)
        for col in range(3)
    ]
    xx, yy = np.meshgrid(x, y, indexing="xy")
    mesh = reference_ax.pcolormesh(
        xx,
        yy,
        np.maximum(reference_density, vmin),
        cmap="viridis",
        norm=norm,
        shading="auto",
        rasterized=True,
    )
    xlim, ylim = DOMAINS[experiment]

    def format_panel(ax, title: str) -> None:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(AXIS_LABELS[experiment][0])
        ax.set_ylabel(AXIS_LABELS[experiment][1])
        ax.set_title(title, pad=4)
        ax.grid(False)

    format_panel(reference_ax, "Reference density")
    for ax, method in zip(method_axes, methods):
        samples = positions[method]
        ax.scatter(
            samples[:, 0],
            samples[:, 1],
            s=2.0,
            alpha=0.20,
            linewidths=0,
            color=_method_color(method),
            rasterized=True,
        )
        format_panel(ax, _method_label(method, experiment, manifest))

    # Keep the density reference centered.  Its colorbar occupies a narrow
    # strip inside the otherwise empty first-row right cell and does not imply
    # a color scale for the single-color scatter panels.
    left_empty = fig.add_subplot(grid[0, 0])
    left_empty.axis("off")
    colorbar_grid = grid[0, 2].subgridspec(1, 2, width_ratios=(1.0, 4.5))
    colorbar_ax = fig.add_subplot(colorbar_grid[0, 0])
    right_empty = fig.add_subplot(colorbar_grid[0, 1])
    right_empty.axis("off")
    colorbar = fig.colorbar(mesh, cax=colorbar_ax)
    colorbar.set_label("Reference probability density")

    return _save_figure(
        fig,
        f"{experiment}_generated_scatter_grid",
        output_root,
        overwrite=overwrite,
        dpi=dpi,
    )


def replot_experiment(
    experiment: str,
    results_root: Path,
    output_root: Path,
    cache_root: Path,
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

    methods = MANUSCRIPT_METHODS[experiment]
    missing = [method for method in methods if method not in positions]
    if missing:
        raise ValueError(
            f"{positions_path} is missing manuscript methods: {missing}"
        )
    written: list[str] = []

    if experiment == "double_well":
        reference_x, reference_density = _reference_density_1d(
            experiment, manifest, grid_size
        )
        written.extend(_plot_e1_overlays(
            positions,
            methods,
            manifest,
            reference_x,
            reference_density,
            output_root,
            overwrite=overwrite,
            dpi=dpi,
        ))
        reference_details = {}
    else:
        e4_cache_path = (
            cache_root / "coupled_phi4" / "direct_snis_qbar_reference.npz"
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
        written.extend(_plot_2d_density_grid(
            experiment,
            positions,
            methods,
            manifest,
            x,
            y,
            density,
            output_root,
            overwrite=overwrite,
            dpi=dpi,
        ))
        written.extend(_plot_2d_scatter_grid(
            experiment,
            positions,
            methods,
            manifest,
            x,
            y,
            density,
            output_root,
            overwrite=overwrite,
            dpi=dpi,
        ))

    report = {
        "experiment": experiment,
        "source": _manifest_path(positions_path),
        "methods": list(methods),
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
        default=JCP_ROOT / "figures",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=JCP_ROOT / "cache" / "generated_samples",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=JCP_ROOT / "cache" / "generated_samples"
        / "generated_sample_plots_manifest.json",
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
            args.cache_root.resolve(),
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

    manifest_path = args.manifest_path.resolve()
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
