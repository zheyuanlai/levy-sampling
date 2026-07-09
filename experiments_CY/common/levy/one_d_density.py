"""One-dimensional target, KDE, and drift diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def trapz_weights(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a one-dimensional grid with at least two points")
    w = np.empty_like(x, dtype=float)
    dx = np.diff(x)
    if not np.all(dx > 0):
        raise ValueError("x must be strictly increasing")
    w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    return w


def normalize_mixture_weights(raw_weights, *, atol=1e-12):
    raw = np.asarray(raw_weights, dtype=float)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("raw_weights must be a nonempty vector")
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw_weights must be finite")
    if not np.all(raw > 0):
        raise ValueError("raw_weights must be positive")
    total = float(np.sum(raw))
    if total <= 0:
        raise ValueError("raw_weights must have positive sum")
    weights = raw / total
    if not np.isclose(float(np.sum(weights)), 1.0, atol=atol, rtol=0.0):
        raise ValueError("normalized weights do not sum to one")
    return weights


def weighted_mean_std(grid, density, weights=None):
    x = np.asarray(grid, dtype=float)
    p = np.asarray(density, dtype=float)
    w = trapz_weights(x) if weights is None else np.asarray(weights, dtype=float)
    mass = float(np.sum(p * w))
    if mass <= 0 or not np.isfinite(mass):
        raise ValueError("density must have positive finite mass")
    mean = float(np.sum(x * p * w) / mass)
    var = float(np.sum((x - mean) ** 2 * p * w) / mass)
    return mean, math.sqrt(max(var, 0.0))


def target_scott_bandwidth(grid, target_density, n_particles, *, min_grid_factor=2.0):
    x = np.asarray(grid, dtype=float)
    _, sigma = weighted_mean_std(x, target_density)
    dx = float(np.median(np.diff(x)))
    h = 1.06 * sigma * max(int(n_particles), 1) ** (-1.0 / 5.0)
    return float(max(h, min_grid_factor * dx))


def target_local_scott_bandwidth(
    grid,
    target_density,
    n_particles,
    *,
    core_mass=0.80,
    min_component_mass=0.03,
    min_grid_factor=2.0,
):
    """Scott bandwidth using target-local high-density components.

    A target-wide standard deviation is dominated by mode separation in
    multimodal low-temperature targets.  This rule identifies high-density
    connected components carrying ``core_mass`` of the target and uses the
    mass-weighted within-component variance as the normal-reference scale.  It
    is deterministic, target-based, method-independent, and fixed over time for
    an experiment.
    """

    x = np.asarray(grid, dtype=float)
    p = np.asarray(target_density, dtype=float)
    if x.ndim != 1 or p.shape != x.shape or x.size < 3:
        raise ValueError("grid and target_density must be one-dimensional arrays of equal length")
    if not (0.0 < core_mass < 1.0):
        raise ValueError("core_mass must be in (0, 1)")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("target_density must be finite and nonnegative")
    w = trapz_weights(x)
    mass = float(np.sum(p * w))
    if mass <= 0 or not np.isfinite(mass):
        raise ValueError("target_density must have positive finite mass")
    p = p / mass
    dx = float(np.median(np.diff(x)))

    weighted_mass = p * w
    order = np.argsort(p)[::-1]
    csum = np.cumsum(weighted_mass[order])
    cutoff_index = int(np.searchsorted(csum, core_mass, side="left"))
    cutoff_index = min(max(cutoff_index, 0), len(order) - 1)
    level = float(p[order[cutoff_index]])
    mask = p >= level

    component_vars = []
    component_masses = []
    start = None
    for i, active in enumerate(mask):
        if active and start is None:
            start = i
        if start is not None and ((not active) or i == len(mask) - 1):
            stop = i if not active else i + 1
            comp = slice(start, stop)
            cmass = float(np.sum(p[comp] * w[comp]))
            if cmass >= min_component_mass:
                mean = float(np.sum(x[comp] * p[comp] * w[comp]) / cmass)
                var = float(np.sum((x[comp] - mean) ** 2 * p[comp] * w[comp]) / cmass)
                component_vars.append(max(var, 0.0))
                component_masses.append(cmass)
            start = None

    if component_vars:
        masses = np.asarray(component_masses, dtype=float)
        vars_ = np.asarray(component_vars, dtype=float)
        local_var = float(np.sum(masses * vars_) / np.sum(masses))
        sigma = math.sqrt(max(local_var, 0.0))
    else:
        _, sigma = weighted_mean_std(x, p, weights=w)

    h = 1.06 * sigma * max(int(n_particles), 1) ** (-1.0 / 5.0)
    return float(max(h, min_grid_factor * dx))


def central_interval_from_cdf(grid, cdf, mass=0.98):
    x = np.asarray(grid, dtype=float)
    F = np.asarray(cdf, dtype=float)
    if not (0.0 < mass < 1.0):
        raise ValueError("mass must be in (0, 1)")
    lo_q = 0.5 * (1.0 - mass)
    hi_q = 1.0 - lo_q
    lo = float(np.interp(lo_q, F, x))
    hi = float(np.interp(hi_q, F, x))
    omitted = float((1.0 - hi_q) + lo_q)
    return lo, hi, omitted


def binned_gaussian_kde_on_grid(samples, grid, bandwidth):
    """Deterministic Gaussian KDE evaluated on an equally spaced grid.

    Samples are first binned to the grid spacing, then convolved with a
    normalized Gaussian kernel.  The returned density is renormalized on the
    displayed grid; the pre-renormalization integral is returned separately.
    """

    x = np.asarray(grid, dtype=float)
    z = np.asarray(samples, dtype=float).ravel()
    if x.ndim != 1 or x.size < 3:
        raise ValueError("grid must be one-dimensional with at least three points")
    dx_arr = np.diff(x)
    dx = float(np.median(dx_arr))
    if not np.allclose(dx_arr, dx, rtol=1e-5, atol=1e-12):
        raise ValueError("grid must be approximately equally spaced")
    h = float(bandwidth)
    if h <= 0 or not np.isfinite(h):
        raise ValueError("bandwidth must be positive and finite")
    edges = np.concatenate(([x[0] - 0.5 * dx], 0.5 * (x[:-1] + x[1:]), [x[-1] + 0.5 * dx]))
    counts, _ = np.histogram(z[np.isfinite(z)], bins=edges)
    mass = counts.astype(float) / max(int(np.sum(counts)), 1)
    radius = max(1, int(math.ceil(4.0 * h / dx)))
    offsets = np.arange(-radius, radius + 1, dtype=float) * dx
    kernel = np.exp(-0.5 * (offsets / h) ** 2)
    kernel = kernel / np.sum(kernel)
    smooth_mass = np.convolve(mass, kernel, mode="same")
    density = smooth_mass / dx
    w = trapz_weights(x)
    integral_before = float(np.sum(density * w))
    if integral_before > 0 and np.isfinite(integral_before):
        density = density / integral_before
    integral_after = float(np.sum(density * w))
    return density, integral_before, integral_after


def binned_gaussian_kde_on_grid_with_diagnostics(samples, grid, bandwidth, *, renormalize=True):
    """Grid-binned Gaussian KDE with explicit out-of-grid tail accounting."""

    x = np.asarray(grid, dtype=float)
    z = np.asarray(samples, dtype=float).ravel()
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        raise ValueError("samples must contain at least one finite value")
    dx_arr = np.diff(x)
    dx = float(np.median(dx_arr))
    if not np.allclose(dx_arr, dx, rtol=1e-5, atol=1e-12):
        raise ValueError("grid must be approximately equally spaced")
    h = float(bandwidth)
    if h <= 0 or not np.isfinite(h):
        raise ValueError("bandwidth must be positive and finite")
    edges = np.concatenate(([x[0] - 0.5 * dx], 0.5 * (x[:-1] + x[1:]), [x[-1] + 0.5 * dx]))
    counts, _ = np.histogram(finite, bins=edges)
    in_grid = int(np.sum(counts))
    tail_mass = float(1.0 - in_grid / max(int(finite.size), 1))
    mass = counts.astype(float) / max(int(finite.size), 1)
    radius = max(1, int(math.ceil(4.0 * h / dx)))
    offsets = np.arange(-radius, radius + 1, dtype=float) * dx
    kernel = np.exp(-0.5 * (offsets / h) ** 2)
    kernel = kernel / np.sum(kernel)
    smooth_mass = np.convolve(mass, kernel, mode="same")
    density = smooth_mass / dx
    w = trapz_weights(x)
    integral_before = float(np.sum(density * w))
    if renormalize and integral_before > 0 and np.isfinite(integral_before):
        density = density / integral_before
    integral_after = float(np.sum(density * w))
    return density, {
        "sample_count": int(finite.size),
        "in_grid_count": int(in_grid),
        "tail_mass_outside_grid": tail_mass,
        "integral_before_grid_renormalization": integral_before,
        "integral": integral_after,
        "bandwidth": h,
    }


def density_histogram_on_bins(samples, bin_edges):
    """Density-normalized histogram and bin masses for fixed bins."""

    z = np.asarray(samples, dtype=float).ravel()
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin_edges must be one-dimensional with at least two entries")
    finite = z[np.isfinite(z)]
    hist, _ = np.histogram(finite, bins=edges, density=True)
    widths = np.diff(edges)
    masses = hist * widths
    return hist, masses, float(np.sum(masses))


def cdf_from_density_on_grid(grid, density):
    x = np.asarray(grid, dtype=float)
    p = np.asarray(density, dtype=float)
    if x.ndim != 1 or p.shape != x.shape:
        raise ValueError("grid and density must have the same one-dimensional shape")
    w = trapz_weights(x)
    cdf = np.cumsum(p * w)
    total = float(cdf[-1]) if cdf.size else 0.0
    if total <= 0 or not np.isfinite(total):
        raise ValueError("density must have positive finite integral")
    cdf = cdf / total
    cdf[-1] = 1.0
    return cdf


def empirical_cdf_on_grid(samples, grid):
    z = np.sort(np.asarray(samples, dtype=float).ravel())
    z = z[np.isfinite(z)]
    if z.size == 0:
        raise ValueError("samples must contain at least one finite value")
    x = np.asarray(grid, dtype=float)
    return np.searchsorted(z, x, side="right") / float(z.size)


def kde_bin_masses_from_grid(kde_density, grid, bin_edges):
    q = np.asarray(kde_density, dtype=float)
    x = np.asarray(grid, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)
    masses = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x <= hi)
        if np.count_nonzero(mask) < 2:
            masses.append(0.0)
        else:
            masses.append(float(np.trapz(q[mask], x[mask])))
    return np.asarray(masses, dtype=float)


def kde_histogram_bin_mass_l1(samples, grid, bandwidth, bin_edges):
    kde, diag = binned_gaussian_kde_on_grid_with_diagnostics(samples, grid, bandwidth)
    _, hist_masses, hist_integral = density_histogram_on_bins(samples, bin_edges)
    kde_masses = kde_bin_masses_from_grid(kde, grid, bin_edges)
    # Renormalize the quadrature bin masses on the same bins for an apples-to-apples L1.
    total = float(np.sum(kde_masses))
    if total > 0:
        kde_masses = kde_masses / total
    l1 = float(np.sum(np.abs(kde_masses - hist_masses)))
    return {
        "histogram_integral": float(hist_integral),
        "KDE_integral": float(diag["integral"]),
        "KDE_tail_mass_outside_plot": float(diag["tail_mass_outside_grid"]),
        "KDE_vs_histogram_bin_mass_L1": l1,
        "sample_count": int(diag["sample_count"]),
    }


@dataclass(frozen=True)
class DensityDiagnosticConfig:
    bandwidth: float
    chi_interval: tuple[float, float]
    chi_omitted_mass: float
    chi_target_min: float
    quadrature: str = "composite trapezoid on reference grid"


def density_errors_on_grid(kde_density, target_density, grid, config: DensityDiagnosticConfig):
    q = np.asarray(kde_density, dtype=float)
    p = np.asarray(target_density, dtype=float)
    x = np.asarray(grid, dtype=float)
    w = trapz_weights(x)
    diff2 = (q - p) ** 2
    l2 = float(np.sum(diff2 * w))
    lo, hi = config.chi_interval
    mask = (x >= lo) & (x <= hi)
    if not np.any(mask):
        raise ValueError("chi-square interval does not intersect grid")
    if np.any(p[mask] <= 0):
        raise ValueError("target density must be positive on chi-square interval")
    chi = float(np.sum(diff2[mask] / p[mask] * w[mask]))
    if l2 < -1e-14 or chi < -1e-14:
        raise ValueError("density diagnostics must be nonnegative")
    return max(l2, 0.0), max(chi, 0.0)


def density_metric_bundle(kde_density, target_density, grid, config: DensityDiagnosticConfig, samples=None):
    q = np.asarray(kde_density, dtype=float)
    p = np.asarray(target_density, dtype=float)
    x = np.asarray(grid, dtype=float)
    if q.shape != p.shape or q.shape != x.shape:
        raise ValueError("kde_density, target_density, and grid must have matching shapes")
    w = trapz_weights(x)
    diff = q - p
    l1 = float(np.sum(np.abs(diff) * w))
    l2_squared = float(np.sum(diff * diff * w))
    lo, hi = config.chi_interval
    mask = (x >= lo) & (x <= hi)
    if not np.any(mask):
        raise ValueError("chi-square interval does not intersect grid")
    if np.any(p[mask] <= 0):
        raise ValueError("target density must be positive on chi-square interval")
    chi = float(np.sum(diff[mask] * diff[mask] / p[mask] * w[mask]))
    target_cdf = cdf_from_density_on_grid(x, p)
    kde_cdf = cdf_from_density_on_grid(x, q)
    cdf_sup = float(np.max(np.abs(kde_cdf - target_cdf)))
    out = {
        "L1_density_error": max(l1, 0.0),
        "L2_density_error": max(l2_squared, 0.0),
        "truncated_KDE_chi2": max(chi, 0.0),
        "KDE_CDF_sup_error": cdf_sup,
    }
    if samples is not None:
        emp_cdf = empirical_cdf_on_grid(samples, x)
        out["empirical_CDF_sup_error"] = float(np.max(np.abs(emp_cdf - target_cdf)))
        out["KDE_vs_empirical_CDF_sup"] = float(np.max(np.abs(kde_cdf - emp_cdf)))
    return out


def sample_from_grid_density(grid, target_density, n_samples, rng):
    x = np.asarray(grid, dtype=float)
    p = np.asarray(target_density, dtype=float)
    w = trapz_weights(x)
    prob = p * w
    total = float(np.sum(prob))
    if total <= 0 or not np.isfinite(total):
        raise ValueError("target_density must have positive finite mass")
    prob = prob / total
    ids = rng.choice(len(x), size=int(n_samples), p=prob)
    return x[ids]


def estimate_kde_bias_floor(
    grid,
    target_density,
    bandwidth,
    n_particles,
    config: DensityDiagnosticConfig,
    *,
    n_replicates=16,
    seed=20261708,
):
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(int(n_replicates)):
        samples = sample_from_grid_density(grid, target_density, n_particles, rng)
        kde, diag = binned_gaussian_kde_on_grid_with_diagnostics(samples, grid, bandwidth)
        metrics = density_metric_bundle(kde, target_density, grid, config, samples=samples)
        metrics.update(
            replicate=rep,
            KDE_integral=float(diag["integral"]),
            KDE_tail_mass_outside_plot=float(diag["tail_mass_outside_grid"]),
            KDE_bandwidth=float(bandwidth),
        )
        rows.append(metrics)
    return rows


def summarize_density_diagnostics(rows):
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metric_cols = [
        "L1_density_error",
        "L2_density_error",
        "truncated_KDE_chi2",
        "KDE_CDF_sup_error",
        "empirical_CDF_sup_error",
        "KDE_vs_empirical_CDF_sup",
        "KDE_integral",
    ]
    metric_cols = [col for col in metric_cols if col in df.columns]
    grouped = df.groupby(["experiment", "method", "time"], as_index=False)
    out = grouped[metric_cols].agg(["mean", "std", "count"])
    out.columns = ["_".join([c for c in col if c]).rstrip("_") for col in out.columns.to_flat_index()]
    out = out.reset_index()
    for col in metric_cols:
        count = np.maximum(out[f"{col}_count"].to_numpy(dtype=float), 1.0)
        out[f"{col}_se"] = out[f"{col}_std"].fillna(0.0).to_numpy(dtype=float) / np.sqrt(count)
    meta_cols = [
        "KDE_bandwidth",
        "chi_interval_left",
        "chi_interval_right",
        "chi_omitted_target_mass",
        "chi_target_density_min",
    ]
    meta = df.groupby(["experiment", "method", "time"], as_index=False)[meta_cols].first()
    return out.merge(meta, on=["experiment", "method", "time"], how="left")
