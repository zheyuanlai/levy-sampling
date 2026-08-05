"""Generic sampling metrics: pure functions, no experiment knowledge.

Everything here is a plain function of arrays. There is no file I/O, no
plotting, no printing, no global random state, and no reference to any
particular experiment, sampler, or target. Torch entry points work on CPU and
CUDA tensors and compute in float64; the MCMC diagnostics and the bootstrap
work on NumPy arrays.

Determinism: every routine that subsamples takes an explicit seed and draws
from a local CPU generator, so results do not depend on the device the data
happens to live on.

Naming discipline: EMC means one thing only, the normalized Shannon entropy of
an occupancy vector (:func:`entropic_mode_coverage`). The related quantity
exp(H)/K is :func:`effective_mode_fraction` and is never called EMC.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math

import numpy as np
import torch

__all__ = [
    # distances between samples
    "w2_exact_1d",
    "make_projections",
    "sliced_w2",
    "mmd2_biased",
    "mmd2_unbiased",
    "median_heuristic",
    # one-dimensional distribution comparisons
    "ks_distance_samples",
    "ks_distance_cdf",
    "empirical_cdf_on_grid",
    "cdf_l2",
    "w1_from_cdf",
    # categorical / occupancy
    "occupancy",
    "entropic_mode_coverage",
    "effective_mode_fraction",
    "jensen_shannon_divergence",
    "total_variation",
    "max_absolute_error",
    "occupancy_ratio",
    # two-dimensional density
    "kde_on_grid_1d",
    "kde_on_grid_2d",
    "squared_hellinger_grid",
    # weighted (self-normalized importance sampling) helpers
    "normalize_log_weights",
    "weighted_mean",
    "weighted_covariance",
    "weighted_category_probabilities",
    "importance_sampling_ess",
    "weighted_effective_count",
    # MCMC diagnostics
    "autocorrelation_time",
    "effective_sample_size",
    "split_rhat",
    "split_rhat_components",
    "bulk_ess",
    "tail_ess",
    "block_mcse",
    "recommended_block_length",
    # bootstrap
    "hierarchical_bootstrap",
]

#: Side length of a pairwise kernel block. Chosen so one block is at most
#: 1024*1024 float64 entries (8 MB) regardless of the sample size.
_PAIRWISE_CHUNK = 1024

#: Number of source points evaluated per KDE block.
_KDE_CHUNK = 4096

#: Tolerance on ``sum(w) == 1`` for helpers documented to take normalized
#: weights. Summing 1e6 float64 weights accumulates at most ~1e-10 of error.
_WEIGHT_SUM_TOL = 1e-8


# --------------------------------------------------------------- internals
def _to_f64(x: torch.Tensor, name: str) -> torch.Tensor:
    """Tensor view of ``x`` in float64, without changing its device."""
    t = torch.as_tensor(x)
    if not torch.is_floating_point(t):
        t = t.to(torch.float64)
    elif t.dtype is not torch.float64:
        t = t.to(torch.float64)
    if not bool(torch.isfinite(t).all()):
        raise ValueError(f"{name} must be finite")
    return t


def _flat_1d(x: torch.Tensor, name: str) -> torch.Tensor:
    """Flatten an ``(n,)`` or ``(n, 1)`` sample to ``(n,)``, float64."""
    t = _to_f64(x, name)
    if t.ndim == 2 and t.shape[1] == 1:
        t = t.reshape(-1)
    elif t.ndim != 1:
        raise ValueError(f"{name} must have shape (n,) or (n, 1), got {tuple(t.shape)}")
    if t.numel() == 0:
        raise ValueError(f"{name} must be nonempty")
    return t


def _cloud_2d(x: torch.Tensor, name: str) -> torch.Tensor:
    """Coerce a point cloud to ``(n, d)``, float64."""
    t = _to_f64(x, name)
    if t.ndim == 1:
        t = t[:, None]
    if t.ndim != 2:
        raise ValueError(f"{name} must have shape (n,) or (n, d), got {tuple(t.shape)}")
    if t.shape[0] == 0:
        raise ValueError(f"{name} must be nonempty")
    return t


def _grid_1d(grid: torch.Tensor, name: str = "grid") -> torch.Tensor:
    """Validate a strictly increasing one-dimensional grid."""
    g = _to_f64(grid, name)
    if g.ndim != 1 or g.numel() < 2:
        raise ValueError(f"{name} must be one-dimensional with at least two points")
    if not bool(torch.all(g[1:] > g[:-1])):
        raise ValueError(f"{name} must be strictly increasing")
    return g


def _uniform_spacing(grid: torch.Tensor, name: str) -> float:
    """Spacing of a uniform grid; raises when the grid is not uniform."""
    diffs = grid[1:] - grid[:-1]
    step = float(diffs.mean().item())
    if float((diffs - step).abs().max().item()) > 1e-9 * max(abs(step), 1.0):
        raise ValueError(f"{name} must be uniformly spaced")
    return step


def _positive_bandwidth(bandwidth: float) -> float:
    h = float(bandwidth)
    if not math.isfinite(h) or h <= 0:
        raise ValueError("bandwidth must be finite and strictly positive")
    return h


def _probability_vector(p: torch.Tensor, name: str,
                        min_categories: int = 1) -> torch.Tensor:
    """Validate a categorical distribution and renormalize it to sum one."""
    t = _to_f64(p, name)
    if t.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional probability vector")
    if t.numel() < min_categories:
        raise ValueError(f"{name} needs at least {min_categories} categories")
    if bool((t < 0).any()):
        raise ValueError(f"{name} must be non-negative")
    total = t.sum()
    if not bool(total > 0):
        raise ValueError(f"{name} must have positive total mass")
    return t / total


def _matched_pair(p: torch.Tensor, q: torch.Tensor,
                  min_categories: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    pp = _probability_vector(p, "p", min_categories)
    qq = _probability_vector(q, "q", min_categories)
    if pp.shape != qq.shape:
        raise ValueError("p and q must have the same number of categories")
    return pp, qq.to(pp.device)


def _normalized_weights(weights: torch.Tensor, n: int | None = None) -> torch.Tensor:
    """Validate weights that are documented to be already normalized."""
    w = _to_f64(weights, "weights").reshape(-1)
    if w.numel() == 0:
        raise ValueError("weights must be nonempty")
    if n is not None and w.numel() != n:
        raise ValueError(f"expected {n} weights, got {w.numel()}")
    if bool((w < 0).any()):
        raise ValueError("weights must be non-negative")
    if abs(float(w.sum().item()) - 1.0) > _WEIGHT_SUM_TOL:
        raise ValueError("weights must be normalized to sum one")
    return w


def _cpu_permutation(n: int, k: int, seed: int, device: torch.device) -> torch.Tensor:
    """First ``k`` entries of a seeded permutation of ``range(n)``.

    The permutation is drawn on the CPU so the subsample depends only on the
    seed, never on whether the data sits on a GPU.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return torch.randperm(n, generator=gen)[:k].to(device)


def _trapz(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Trapezoidal integral of ``f`` over the (possibly nonuniform) grid ``x``."""
    return (0.5 * (f[1:] + f[:-1]) * (x[1:] - x[:-1])).sum()


def _sorted_quantiles(sorted_values: torch.Tensor,
                      probs: torch.Tensor) -> torch.Tensor:
    """Linearly interpolated empirical quantiles of values sorted along dim 0.

    ``sorted_values`` has shape ``(n, ...)``; the result has shape
    ``(len(probs), ...)``. Probability ``p`` maps to the fractional order
    statistic ``p * (n - 1)``, the standard "linear" quantile convention.
    """
    n = sorted_values.shape[0]
    pos = probs.to(sorted_values.device) * (n - 1)
    lo = torch.floor(pos).to(torch.long).clamp(0, n - 1)
    hi = torch.ceil(pos).to(torch.long).clamp(0, n - 1)
    frac = (pos - lo.to(pos.dtype)).reshape((-1,) + (1,) * (sorted_values.ndim - 1))
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def _w2_from_sorted(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """Column-wise 1-D W2 between clouds sorted along dim 0.

    Equal sample sizes use the exact sorted (monotone) coupling. Unequal sizes
    are compared at ``L = min(n, m)`` evenly spaced probability levels
    ``(i + 0.5) / L``, an ``O(L)`` quantile approximation of the exact
    quantile-function integral. Returns one W2 per column.
    """
    n, m = xs.shape[0], ys.shape[0]
    if n == m:
        diff = xs - ys
    else:
        levels = min(n, m)
        probs = (torch.arange(levels, dtype=torch.float64, device=xs.device) + 0.5) / levels
        diff = _sorted_quantiles(xs, probs) - _sorted_quantiles(ys, probs)
    return torch.sqrt((diff * diff).mean(dim=0))


# ------------------------------------------------- distances between samples
def w2_exact_1d(x: torch.Tensor, y: torch.Tensor) -> float:
    """Exact one-dimensional Wasserstein-2 distance by sorting.

    ``W2 = sqrt(mean_i (x_(i) - y_(i))^2)`` over the order statistics of two
    equally sized samples, which is the exact optimal coupling in 1-D. Inputs
    have shape ``(n,)`` or ``(n, 1)``. When the sizes differ, the samples are
    compared at ``min(n, m)`` evenly spaced probability levels instead (see
    :func:`sliced_w2`).
    """
    xs = torch.sort(_flat_1d(x, "x")).values[:, None]
    ys = torch.sort(_flat_1d(y, "y").to(xs.device)).values[:, None]
    return float(_w2_from_sorted(xs, ys)[0].item())


def make_projections(d: int, n_projections: int, seed: int,
                     device: str | torch.device) -> torch.Tensor:
    """Deterministic ``(n_projections, d)`` matrix of unit-norm random rows.

    Rows are standard normal vectors rescaled to unit length, i.e. uniform on
    the sphere. They are drawn on the CPU and moved to ``device``, so the same
    seed gives the same projections on CPU and on GPU.
    """
    if d < 1:
        raise ValueError("d must be at least 1")
    if n_projections < 1:
        raise ValueError("n_projections must be at least 1")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    theta = torch.randn(int(n_projections), int(d), generator=gen, dtype=torch.float64)
    norms = theta.norm(dim=1, keepdim=True)
    theta = theta / norms.clamp_min(1e-300)
    return theta.to(torch.device(device))


def sliced_w2(x: torch.Tensor, y: torch.Tensor,
              projections: torch.Tensor) -> float:
    """Sliced Wasserstein-2: ``mean_l W2(<theta_l, X>, <theta_l, Y>)``.

    This is the mean of the per-projection W2 values, not the square root of
    the mean of their squares. Each projected pair is compared exactly by
    sorting when the sample sizes match; otherwise both projected clouds are
    evaluated at ``min(n, m)`` evenly spaced probability levels
    ``(i + 0.5) / min(n, m)`` and compared quantile by quantile.
    """
    xs = _cloud_2d(x, "x")
    ys = _cloud_2d(y, "y").to(xs.device)
    proj = _to_f64(projections, "projections").to(xs.device)
    if proj.ndim != 2:
        raise ValueError("projections must have shape (n_projections, d)")
    if proj.shape[1] != xs.shape[1] or proj.shape[1] != ys.shape[1]:
        raise ValueError("projections must have one column per sample dimension")
    xp = torch.sort(xs @ proj.T, dim=0).values
    yp = torch.sort(ys @ proj.T, dim=0).values
    return float(_w2_from_sorted(xp, yp).mean().item())


def _rbf_cross_sum(a: torch.Tensor, b: torch.Tensor, gamma: float) -> torch.Tensor:
    """Sum of ``exp(-gamma ||a_i - b_j||^2)`` over all pairs, in blocks."""
    total = torch.zeros((), dtype=torch.float64, device=a.device)
    for i0 in range(0, a.shape[0], _PAIRWISE_CHUNK):
        ai = a[i0:i0 + _PAIRWISE_CHUNK]
        for j0 in range(0, b.shape[0], _PAIRWISE_CHUNK):
            bj = b[j0:j0 + _PAIRWISE_CHUNK]
            total = total + torch.exp(-gamma * torch.cdist(ai, bj) ** 2).sum()
    return total


def _rbf_self_sums(a: torch.Tensor, gamma: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``(sum over all pairs, sum over off-diagonal pairs)`` for one sample.

    The diagonal is removed by subtracting the computed diagonal entries rather
    than by assuming ``k(x, x) == 1``, so the off-diagonal sum is exact even if
    the distance backend returns a nonzero self-distance.
    """
    total = torch.zeros((), dtype=torch.float64, device=a.device)
    offdiag = torch.zeros((), dtype=torch.float64, device=a.device)
    for i0 in range(0, a.shape[0], _PAIRWISE_CHUNK):
        ai = a[i0:i0 + _PAIRWISE_CHUNK]
        for j0 in range(0, a.shape[0], _PAIRWISE_CHUNK):
            aj = a[j0:j0 + _PAIRWISE_CHUNK]
            block = torch.exp(-gamma * torch.cdist(ai, aj) ** 2)
            block_sum = block.sum()
            total = total + block_sum
            if i0 == j0:
                block_sum = block_sum - torch.diagonal(block).sum()
            offdiag = offdiag + block_sum
    return total, offdiag


def _mmd_inputs(x: torch.Tensor, y: torch.Tensor,
                bandwidth: float) -> tuple[torch.Tensor, torch.Tensor, float]:
    xs = _cloud_2d(x, "x")
    ys = _cloud_2d(y, "y").to(xs.device)
    if xs.shape[1] != ys.shape[1]:
        raise ValueError("x and y must have the same dimension")
    h = _positive_bandwidth(bandwidth)
    return xs, ys, 0.5 / (h * h)


def mmd2_biased(x: torch.Tensor, y: torch.Tensor, bandwidth: float) -> float:
    """Biased (V-statistic) squared MMD with an RBF kernel.

    ``k(a, b) = exp(-||a - b||^2 / (2 * bandwidth^2))`` and

        MMD2_b = sum_ij k(x_i,x_j)/n^2 - 2 sum_ij k(x_i,y_j)/(n m)
                 + sum_ij k(y_i,y_j)/m^2,

    diagonal terms included. This is exactly ``||mu_x - mu_y||_H^2`` for the
    empirical mean embeddings, so it is non-negative up to round-off. The
    pairwise sums are accumulated in blocks; the result matches the unblocked
    computation to floating-point round-off.
    """
    xs, ys, gamma = _mmd_inputs(x, y, bandwidth)
    n, m = xs.shape[0], ys.shape[0]
    sxx, _ = _rbf_self_sums(xs, gamma)
    syy, _ = _rbf_self_sums(ys, gamma)
    sxy = _rbf_cross_sum(xs, ys, gamma)
    return float((sxx / (n * n) - 2.0 * sxy / (n * m) + syy / (m * m)).item())


def mmd2_unbiased(x: torch.Tensor, y: torch.Tensor, bandwidth: float) -> float:
    """Unbiased (U-statistic) squared MMD with an RBF kernel.

    ``k(a, b) = exp(-||a - b||^2 / (2 * bandwidth^2))`` and

        MMD2_u = sum_{i!=j} k(x_i,x_j)/(n(n-1))
                 - 2 sum_ij k(x_i,y_j)/(n m)
                 + sum_{i!=j} k(y_i,y_j)/(m(m-1)),

    with the diagonal excluded from both within-sample terms. The estimator is
    unbiased for the squared MMD and can therefore be slightly negative when
    the two samples come from the same distribution. Requires n, m >= 2.
    """
    xs, ys, gamma = _mmd_inputs(x, y, bandwidth)
    n, m = xs.shape[0], ys.shape[0]
    if n < 2 or m < 2:
        raise ValueError("the unbiased MMD needs at least two points per sample")
    _, sxx = _rbf_self_sums(xs, gamma)
    _, syy = _rbf_self_sums(ys, gamma)
    sxy = _rbf_cross_sum(xs, ys, gamma)
    value = sxx / (n * (n - 1)) - 2.0 * sxy / (n * m) + syy / (m * (m - 1))
    return float(value.item())


def median_heuristic(y: torch.Tensor, max_points: int = 4096,
                     seed: int = 99) -> float:
    """Median pairwise Euclidean distance on a seeded subsample of ``y``.

    The median runs over the strict upper triangle of the distance matrix of at
    most ``max_points`` points drawn without replacement using ``seed``. The
    usual use is to freeze an MMD bandwidth once on a reference sample. Rows
    are processed in blocks so the full square matrix is never materialized.
    """
    ys = _cloud_2d(y, "y")
    if ys.shape[0] < 2:
        raise ValueError("the median heuristic needs at least two points")
    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    n = min(int(max_points), ys.shape[0])
    sub = ys[_cpu_permutation(ys.shape[0], n, seed, ys.device)]
    columns = torch.arange(n, device=ys.device)
    parts = []
    for i0 in range(0, n, _PAIRWISE_CHUNK):
        rows = torch.arange(i0, min(i0 + _PAIRWISE_CHUNK, n), device=ys.device)
        block = torch.cdist(sub[i0:i0 + _PAIRWISE_CHUNK], sub)
        parts.append(block[columns[None, :] > rows[:, None]])
    median = float(torch.cat(parts).median().item())
    if median <= 0:
        raise ValueError("degenerate sample: the median pairwise distance is zero")
    return median


# ------------------------------------ one-dimensional distribution comparisons
def empirical_cdf_on_grid(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Empirical CDF ``F_n(t) = #{x_i <= t} / n`` evaluated on ``grid``."""
    xs = torch.sort(_flat_1d(x, "x")).values
    g = _grid_1d(grid).to(xs.device)
    counts = torch.searchsorted(xs, g.contiguous(), right=True)
    return counts.to(torch.float64) / xs.numel()


def ks_distance_samples(x: torch.Tensor, reference: torch.Tensor) -> float:
    """Two-sample Kolmogorov-Smirnov distance ``sup_t |F_x(t) - F_ref(t)|``.

    Both empirical CDFs are step functions, so the supremum is attained at one
    of the pooled sample points and is computed exactly there. This is a
    distance, not a hypothesis test: no p-value and no sample-size scaling is
    applied.
    """
    xs = torch.sort(_flat_1d(x, "x")).values
    rs = torch.sort(_flat_1d(reference, "reference").to(xs.device)).values
    pooled = torch.cat([xs, rs]).contiguous()
    fx = torch.searchsorted(xs, pooled, right=True).to(torch.float64) / xs.numel()
    fr = torch.searchsorted(rs, pooled, right=True).to(torch.float64) / rs.numel()
    return float((fx - fr).abs().max().item())


def _cdf_difference(x: torch.Tensor, grid: torch.Tensor,
                    target_cdf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    g = _grid_1d(grid)
    target = _to_f64(target_cdf, "target_cdf").reshape(-1).to(g.device)
    if target.shape != g.shape:
        raise ValueError("target_cdf must have one value per grid point")
    empirical = empirical_cdf_on_grid(x, g)
    return empirical - target.to(empirical.device), g.to(empirical.device)


def ks_distance_cdf(x: torch.Tensor, grid: torch.Tensor,
                    target_cdf: torch.Tensor) -> float:
    """``max_g |F_n(g) - F*(g)|`` over the supplied grid.

    A grid-restricted Kolmogorov-Smirnov distance against a known CDF; it is a
    lower bound on the supremum over the whole line.
    """
    diff, _ = _cdf_difference(x, grid, target_cdf)
    return float(diff.abs().max().item())


def cdf_l2(x: torch.Tensor, grid: torch.Tensor,
           target_cdf: torch.Tensor) -> float:
    """``sqrt(int (F_n - F*)^2 dt)`` by the trapezoidal rule on ``grid``."""
    diff, g = _cdf_difference(x, grid, target_cdf)
    return float(torch.sqrt(_trapz(diff * diff, g).clamp_min(0.0)).item())


def w1_from_cdf(x: torch.Tensor, grid: torch.Tensor,
                target_cdf: torch.Tensor) -> float:
    """``int |F_n - F*| dt`` by the trapezoidal rule, i.e. the 1-D W1 distance.

    Only the mass inside the grid range is counted, so the value is a lower
    bound on W1 when either distribution puts mass outside ``grid``.
    """
    diff, g = _cdf_difference(x, grid, target_cdf)
    return float(_trapz(diff.abs(), g).item())


# ------------------------------------------------ categorical / occupancy
def occupancy(labels: torch.Tensor, n_categories: int) -> torch.Tensor:
    """Normalized category counts ``p_k = #{labels == k} / n``, shape ``(K,)``."""
    k = int(n_categories)
    if k < 1:
        raise ValueError("n_categories must be at least 1")
    idx = torch.as_tensor(labels).reshape(-1)
    if idx.numel() == 0:
        raise ValueError("labels must be nonempty")
    if torch.is_floating_point(idx):
        raise ValueError("labels must be integer category indices")
    idx = idx.to(torch.long)
    if bool((idx < 0).any()) or bool((idx >= k).any()):
        raise ValueError(f"labels must lie in [0, {k})")
    counts = torch.bincount(idx, minlength=k).to(torch.float64)
    return counts / idx.numel()


def entropic_mode_coverage(p: torch.Tensor) -> float:
    """EMC: the normalized Shannon entropy of an occupancy vector.

    ``EMC = -sum_k p_k log p_k / log K`` with the convention ``0 log 0 = 0``.
    EMC is 1 for a uniform occupancy over the K categories and 0 for a point
    mass. This is the only quantity in this module allowed to be called EMC.
    """
    q = _probability_vector(p, "p", min_categories=2)
    nz = q[q > 0]
    entropy = float(-(nz * torch.log(nz)).sum().item())
    return entropy / math.log(q.numel())


def effective_mode_fraction(p: torch.Tensor) -> float:
    """``exp(H(p)) / K``: the perplexity of ``p`` as a fraction of K.

    NOT EMC. This is a different functional of the same occupancy vector and
    must never be plotted, tabulated, or labelled as EMC; use
    :func:`entropic_mode_coverage` for that. Both equal 1 at the uniform
    occupancy but they disagree everywhere else (for a point mass this returns
    1/K while EMC returns 0).
    """
    q = _probability_vector(p, "p", min_categories=2)
    nz = q[q > 0]
    entropy = float(-(nz * torch.log(nz)).sum().item())
    return math.exp(entropy) / q.numel()


def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor,
                              base: str = "e") -> float:
    """Jensen-Shannon divergence ``0.5 KL(p||m) + 0.5 KL(q||m)``, ``m=(p+q)/2``.

    ``base="e"`` (default) returns nats and is bounded by ``log 2``;
    ``base="2"`` returns bits and is bounded by 1. Convention ``0 log 0 = 0``,
    so terms with zero mass are dropped rather than made infinite.
    """
    if base not in ("e", "2"):
        raise ValueError("base must be 'e' or '2'")
    pp, qq = _matched_pair(p, q)
    m = 0.5 * (pp + qq)
    log = torch.log2 if base == "2" else torch.log

    def _kl(a: torch.Tensor) -> torch.Tensor:
        mask = a > 0
        return (a[mask] * log(a[mask] / m[mask])).sum()

    return float((0.5 * _kl(pp) + 0.5 * _kl(qq)).clamp_min(0.0).item())


def total_variation(p: torch.Tensor, q: torch.Tensor) -> float:
    """Total variation distance ``0.5 * sum_k |p_k - q_k|``, in ``[0, 1]``."""
    pp, qq = _matched_pair(p, q)
    return float((0.5 * (pp - qq).abs().sum()).item())


def max_absolute_error(p: torch.Tensor, q: torch.Tensor) -> float:
    """Largest per-category discrepancy ``max_k |p_k - q_k|``."""
    pp, qq = _matched_pair(p, q)
    return float((pp - qq).abs().max().item())


def occupancy_ratio(p_hat: torch.Tensor, p_star: torch.Tensor,
                    floor: float = 0.0) -> torch.Tensor:
    """Elementwise occupancy ratio ``p_hat_k / max(p_star_k, floor)``.

    With the default ``floor = 0`` a zero-mass target category yields ``inf``
    when the sampler put mass there and ``nan`` when it did not (0/0 is
    genuinely undefined), so an unreachable category can never masquerade as a
    ratio of 1. A positive ``floor`` caps the reported ratio instead.
    """
    if floor < 0 or not math.isfinite(floor):
        raise ValueError("floor must be finite and non-negative")
    hat, star = _matched_pair(p_hat, p_star)
    den = star if floor <= 0 else star.clamp_min(float(floor))
    out = torch.full_like(hat, float("nan"))
    positive = den > 0
    out[positive] = hat[positive] / den[positive]
    out[~positive & (hat > 0)] = float("inf")
    return out


# ------------------------------------------------- two-dimensional density
def kde_on_grid_1d(x: torch.Tensor, grid: torch.Tensor,
                   bandwidth: float) -> torch.Tensor:
    """Gaussian KDE of ``x`` evaluated on ``grid`` and normalized on it.

    ``rho(g) = sum_i exp(-(g - x_i)^2 / (2 h^2)) / (n h sqrt(2 pi))``, then
    divided by its trapezoidal integral over ``grid`` so the returned density
    integrates to 1 on the grid. Contributions are accumulated in blocks of
    source points.
    """
    xs = _flat_1d(x, "x")
    g = _grid_1d(grid).to(xs.device)
    h = _positive_bandwidth(bandwidth)
    acc = torch.zeros(g.numel(), dtype=torch.float64, device=xs.device)
    for start in range(0, xs.numel(), _KDE_CHUNK):
        block = xs[start:start + _KDE_CHUNK]
        z = (g[:, None] - block[None, :]) / h
        acc = acc + torch.exp(-0.5 * z * z).sum(dim=1)
    rho = acc / (xs.numel() * h * math.sqrt(2.0 * math.pi))
    return rho / _trapz(rho, g).clamp_min(1e-300)


def kde_on_grid_2d(points: torch.Tensor, grid_x: torch.Tensor,
                   grid_y: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """Isotropic Gaussian KDE on a tensor grid, returned as ``(nx, ny)``.

    ``rho(gx, gy) = sum_i exp(-((gx - x_i)^2 + (gy - y_i)^2) / (2 h^2))
    / (n * 2 pi h^2)``, evaluated at every ``(grid_x[i], grid_y[j])`` and then
    divided by ``sum(rho) * dA`` so that ``sum(rho) * dA == 1``. Both grids
    must be uniformly spaced because a single cell area ``dA = dx * dy`` is
    used. The separable kernel is accumulated in blocks of source points.
    """
    pts = _cloud_2d(points, "points")
    if pts.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    gx = _grid_1d(grid_x, "grid_x").to(pts.device)
    gy = _grid_1d(grid_y, "grid_y").to(pts.device)
    h = _positive_bandwidth(bandwidth)
    dx = _uniform_spacing(gx, "grid_x")
    dy = _uniform_spacing(gy, "grid_y")

    acc = torch.zeros(gx.numel(), gy.numel(), dtype=torch.float64, device=pts.device)
    scale = -0.5 / (h * h)
    for start in range(0, pts.shape[0], _KDE_CHUNK):
        block = pts[start:start + _KDE_CHUNK]
        ax = torch.exp(scale * (gx[:, None] - block[None, :, 0]) ** 2)
        ay = torch.exp(scale * (gy[:, None] - block[None, :, 1]) ** 2)
        acc = acc + ax @ ay.T
    rho = acc / (pts.shape[0] * 2.0 * math.pi * h * h)
    mass = rho.sum() * (dx * dy)
    if not bool(mass > 0):
        raise ValueError("the KDE has no mass on the supplied grid")
    return rho / mass


def squared_hellinger_grid(p: torch.Tensor, q: torch.Tensor,
                           cell_area: float) -> float:
    """Squared Hellinger distance ``1 - sum_g sqrt(p_g q_g) * dA``.

    Both inputs are densities already normalized on the same grid, i.e.
    ``sum(p) * dA == 1``. The Bhattacharyya coefficient can exceed 1 by
    round-off when the two densities coincide, so tiny negative results are
    clamped to 0.
    """
    area = float(cell_area)
    if not math.isfinite(area) or area <= 0:
        raise ValueError("cell_area must be finite and strictly positive")
    pp = _to_f64(p, "p")
    qq = _to_f64(q, "q").to(pp.device)
    if pp.shape != qq.shape:
        raise ValueError("p and q must be defined on the same grid")
    if bool((pp < 0).any()) or bool((qq < 0).any()):
        raise ValueError("densities must be non-negative")
    overlap = float((torch.sqrt(pp * qq).sum() * area).item())
    return max(0.0, 1.0 - overlap)


# ------------------------- weighted (self-normalized importance sampling)
def normalize_log_weights(log_w: torch.Tensor) -> torch.Tensor:
    """Stable softmax of log weights: ``w_i = exp(l_i - max l) / sum_j (...)``.

    Entries equal to ``-inf`` receive weight exactly 0. At least one entry must
    be finite; NaN entries are rejected.
    """
    lw = torch.as_tensor(log_w).reshape(-1)
    if not torch.is_floating_point(lw) or lw.dtype is not torch.float64:
        lw = lw.to(torch.float64)
    if lw.numel() == 0:
        raise ValueError("log_w must be nonempty")
    if bool(torch.isnan(lw).any()) or bool(torch.isposinf(lw).any()):
        raise ValueError("log_w must not contain NaN or +inf")
    finite = lw[torch.isfinite(lw)]
    if finite.numel() == 0:
        raise ValueError("log_w must contain at least one finite entry")
    shifted = torch.exp(lw - finite.max())
    total = shifted.sum()
    if not bool(total > 0):
        raise ValueError("log weights underflow to zero total mass")
    return shifted / total


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``sum_i w_i v_i`` for already normalized weights ``w``.

    ``values`` of shape ``(n,)`` gives a scalar tensor; shape ``(n, k)`` gives
    a ``(k,)`` tensor.
    """
    v = _to_f64(values, "values")
    if v.ndim not in (1, 2) or v.shape[0] == 0:
        raise ValueError("values must have shape (n,) or (n, k)")
    w = _normalized_weights(weights, v.shape[0]).to(v.device)
    if v.ndim == 1:
        return (w * v).sum()
    return (w[:, None] * v).sum(dim=0)


def weighted_covariance(values: torch.Tensor,
                        weights: torch.Tensor) -> torch.Tensor:
    """``sum_i w_i (v_i - mu)(v_i - mu)^T`` for normalized weights, shape (k,k).

    This is the plug-in self-normalized estimator with no small-sample or
    ``1 - sum w^2`` bias correction. ``values`` of shape ``(n,)`` is treated as
    ``k = 1`` and returns a ``(1, 1)`` tensor.
    """
    v = _to_f64(values, "values")
    if v.ndim == 1:
        v = v[:, None]
    if v.ndim != 2 or v.shape[0] == 0:
        raise ValueError("values must have shape (n,) or (n, k)")
    w = _normalized_weights(weights, v.shape[0]).to(v.device)
    centered = v - (w[:, None] * v).sum(dim=0, keepdim=True)
    return (centered * w[:, None]).T @ centered


def weighted_category_probabilities(labels: torch.Tensor, n_categories: int,
                                    weights: torch.Tensor) -> torch.Tensor:
    """Weighted occupancy ``p_k = sum_{i: label_i = k} w_i``, shape ``(K,)``.

    Weights are already normalized, so the result sums to 1 by construction.
    """
    k = int(n_categories)
    if k < 1:
        raise ValueError("n_categories must be at least 1")
    idx = torch.as_tensor(labels).reshape(-1)
    if idx.numel() == 0:
        raise ValueError("labels must be nonempty")
    if torch.is_floating_point(idx):
        raise ValueError("labels must be integer category indices")
    idx = idx.to(torch.long)
    if bool((idx < 0).any()) or bool((idx >= k).any()):
        raise ValueError(f"labels must lie in [0, {k})")
    w = _normalized_weights(weights, idx.numel()).to(idx.device)
    out = torch.zeros(k, dtype=torch.float64, device=w.device)
    out.scatter_add_(0, idx.to(w.device), w)
    return out


def importance_sampling_ess(weights: torch.Tensor) -> float:
    """Kish effective sample size ``1 / sum_i w_i^2`` for normalized weights.

    Ranges from 1 (all mass on one sample) to ``n`` (uniform weights).
    """
    w = _normalized_weights(weights)
    denom = float((w * w).sum().item())
    if denom <= 0:
        raise ValueError("weights must have positive squared mass")
    return 1.0 / denom


def weighted_effective_count(weights: torch.Tensor,
                             mask: torch.Tensor) -> float:
    """Effective number of weighted samples inside ``mask``.

    FROZEN FORMULA, do not reformulate:

        n_eff(mask) = (sum_{i in mask} w_i)^2 / sum_{i in mask} w_i^2

    with ``w`` the normalized weights. This is the exact expression the
    acceptance gates are written against; it is scale invariant, equals the
    number of masked items when the masked weights are all equal, and is 0 for
    an empty mask.
    """
    w = _to_f64(weights, "weights").reshape(-1)
    if w.numel() == 0:
        raise ValueError("weights must be nonempty")
    if bool((w < 0).any()):
        raise ValueError("weights must be non-negative")
    m = torch.as_tensor(mask).reshape(-1).to(w.device)
    if m.dtype is not torch.bool:
        raise ValueError("mask must be a boolean tensor")
    if m.shape != w.shape:
        raise ValueError("mask must have one entry per weight")
    selected = w[m]
    if selected.numel() == 0:
        return 0.0
    numerator = selected.sum() ** 2
    denominator = (selected * selected).sum()
    if not bool(denominator > 0):
        return 0.0
    return float((numerator / denominator).item())


# ------------------------------------------------------- MCMC diagnostics
def _acf_1d(x: np.ndarray) -> np.ndarray:
    """Biased autocorrelation estimate used by Geyer's positive sequence."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return np.asarray([], dtype=float)
    x = x - x.mean()
    n = x.size
    if np.allclose(x, 0.0):
        return np.asarray([1.0])
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n].real
    return acf / acf[0]


def autocorrelation_time(x: np.ndarray) -> float:
    """Integrated autocorrelation time by Geyer's initial positive sequence.

    With ``Gamma_k = rho[2k] + rho[2k+1]``, pair sums are truncated before the
    first non-positive value and monotonised by cumulative minima, giving
    ``tau = -1 + 2 sum_k Gamma_k`` bounded below by 1. A constant series has no
    observed decorrelation and returns ``inf``; a series with fewer than two
    finite draws returns ``nan``.
    """
    values = np.asarray(x, dtype=float).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values)):
        return float("nan")
    if np.allclose(values, values[0]):
        return float("inf")
    rho = _acf_1d(values)
    n_pairs = rho.size // 2
    if n_pairs == 0:
        return 1.0
    paired = rho[:2 * n_pairs].reshape(n_pairs, 2).sum(axis=1)
    positive: list[float] = []
    for gamma in paired:
        if not np.isfinite(gamma) or gamma <= 0:
            break
        positive.append(float(gamma))
    if not positive:
        return 1.0
    monotone = np.minimum.accumulate(np.asarray(positive, dtype=float))
    return float(max(-1.0 + 2.0 * monotone.sum(), 1.0))


def effective_sample_size(x: np.ndarray) -> float:
    """Total ESS ``sum_c n / tau_c`` over independent chains.

    ``x`` has shape ``(n_draws,)`` or ``(n_chains, n_draws)``. Each chain
    contributes ``n_draws / tau`` with ``tau`` from
    :func:`autocorrelation_time`; a constant chain contributes 0. Nonfinite or
    too-short input is undefined and returns ``nan`` rather than being
    silently dropped.
    """
    series = np.atleast_2d(np.asarray(x, dtype=float))
    if series.ndim != 2 or series.shape[1] < 2 or not np.all(np.isfinite(series)):
        return float("nan")
    total = 0.0
    for chain in series:
        tau = autocorrelation_time(chain)
        if np.isnan(tau):
            return float("nan")
        if np.isfinite(tau):
            total += chain.size / tau
    return float(total)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """One-based average ranks with exact tie handling."""
    x = np.asarray(x, dtype=float).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(x.size, dtype=float)
    start = 0
    while start < x.size:
        stop = start + 1
        while stop < x.size and sorted_x[stop] == sorted_x[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _norm_ppf(p: np.ndarray) -> np.ndarray:
    """Inverse standard-normal CDF (Acklam's rational approximation, ~1e-9)."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    cc = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    out = np.empty_like(p)
    lo = p < plow
    hi = p > phigh
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(p[lo]))
    out[lo] = (((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5]) / \
              ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    q = p[mid] - 0.5
    r = q * q
    out[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = np.sqrt(-2 * np.log(1 - p[hi]))
    out[hi] = -(((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5]) / \
               ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    return out


def _rank_normalize(chains: np.ndarray) -> np.ndarray:
    """Rank-normalize pooled draws to z-scores (Vehtari et al. 2021)."""
    flat = np.asarray(chains, dtype=float).reshape(-1)
    ranks = _average_ranks(flat)
    # Blom's offset; S + 1/4 is the correct denominator.
    z = _norm_ppf((ranks - 3.0 / 8.0) / (flat.size + 1.0 / 4.0))
    return z.reshape(np.asarray(chains).shape)


def _basic_rhat(chains: np.ndarray) -> float:
    """Classical between/within-chain variance ratio on transformed data."""
    chains = np.asarray(chains, dtype=float)
    _, n = chains.shape
    chain_means = chains.mean(axis=1)
    between = n * chain_means.var(ddof=1)
    within = chains.var(axis=1, ddof=1).mean()
    if within <= 0:
        return 1.0 if between <= 0 else float("inf")
    var_plus = (n - 1.0) / n * within + between / n
    return float(np.sqrt(max(var_plus / within, 0.0)))


def split_rhat_components(chains: np.ndarray) -> tuple[float, float]:
    """Rank-normalized bulk and folded split-R-hat components.

    ``chains`` has shape ``(n_chains, n_draws)``. Each chain is split in half,
    the halves are treated as separate chains, and R-hat is computed on the
    rank-normalized draws (bulk) and on the rank-normalized absolute deviations
    from the pooled median (folded). Returns ``(nan, nan)`` when the input is
    nonfinite, too short, has fewer than two chains, or is entirely constant.
    """
    chains = np.atleast_2d(np.asarray(chains, dtype=float))
    if not np.all(np.isfinite(chains)):
        return float("nan"), float("nan")
    n_chains, n_draws = chains.shape
    half = n_draws // 2
    if n_chains < 2 or half < 2:
        return float("nan"), float("nan")
    split = np.concatenate([chains[:, :half], chains[:, n_draws - half:]], axis=0)
    if np.all(split == split.flat[0]):
        # R-hat is undefined, not evidence of convergence, when every retained
        # chain is the same constant series.
        return float("nan"), float("nan")
    bulk = _basic_rhat(_rank_normalize(split))
    folded = _basic_rhat(_rank_normalize(np.abs(split - np.median(split))))
    return bulk, folded


def split_rhat(chains: np.ndarray) -> float:
    """Maximum of the bulk and folded rank-normalized split-R-hat.

    Average ranks are used for ties, which is essential for discrete basin
    indicators. The folded component detects scale and tail non-convergence
    that a location-only rank diagnostic can miss.
    """
    bulk, folded = split_rhat_components(chains)
    if np.isnan(bulk) or np.isnan(folded):
        return float("nan")
    return float(max(bulk, folded))


def bulk_ess(chains: np.ndarray) -> float:
    """ESS of the rank-normalized draws (Vehtari et al. 2021, bulk-ESS).

    All draws are pooled, replaced by their rank-normalized z-scores, reshaped
    back into chains, and passed to :func:`effective_sample_size`. Rank
    normalization makes the diagnostic robust to heavy tails and to
    non-normality of the marginal. Chains are not split here: the per-chain
    ESS sum uses only within-chain autocorrelation, so splitting would change
    only the (unused) between-chain variance term; use :func:`split_rhat` for
    the between-chain part of the diagnosis.
    """
    series = np.atleast_2d(np.asarray(chains, dtype=float))
    if series.ndim != 2 or series.shape[1] < 2 or not np.all(np.isfinite(series)):
        return float("nan")
    return effective_sample_size(_rank_normalize(series))


def tail_ess(chains: np.ndarray) -> float:
    """Tail-ESS: ``min`` of the ESS at the 5% and 95% quantiles.

    Following Vehtari et al. (2021), for each of the pooled 5% and 95%
    empirical quantiles ``q`` the indicator series ``I(x <= q)`` is formed,
    rank-normalized, and its ESS computed; the reported value is the smaller of
    the two. (Rank-normalizing a binary series is an affine map, so it leaves
    the ESS unchanged; it is kept to match the published definition.) A
    constant indicator, e.g. when no draw crosses a quantile, contributes 0.
    """
    series = np.atleast_2d(np.asarray(chains, dtype=float))
    if series.ndim != 2 or series.shape[1] < 2 or not np.all(np.isfinite(series)):
        return float("nan")
    flat = series.reshape(-1)
    values = []
    for level in (0.05, 0.95):
        indicator = (series <= np.quantile(flat, level)).astype(float)
        values.append(effective_sample_size(_rank_normalize(indicator)))
    if any(np.isnan(v) for v in values):
        return float("nan")
    return float(min(values))


def block_mcse(x: np.ndarray, block_length: int) -> float:
    """Batch-means Monte Carlo standard error of the mean of ``x``.

    The series is cut into ``n_blocks = len(x) // block_length`` non-overlapping
    blocks (a trailing partial block is discarded) and

        MCSE = sd(block means, ddof=1) / sqrt(n_blocks).

    Raises ValueError when fewer than two whole blocks fit. Nonfinite input
    returns ``nan``.
    """
    values = np.asarray(x, dtype=float).reshape(-1)
    length = int(block_length)
    if length < 1:
        raise ValueError("block_length must be at least 1")
    n_blocks = values.size // length
    if n_blocks < 2:
        raise ValueError(
            f"need at least two whole blocks, got {n_blocks} "
            f"for {values.size} draws at block_length={length}"
        )
    if not np.all(np.isfinite(values)):
        return float("nan")
    means = values[:n_blocks * length].reshape(n_blocks, length).mean(axis=1)
    return float(means.std(ddof=1) / math.sqrt(n_blocks))


def recommended_block_length(series: Mapping[str, np.ndarray],
                             multiplier: float = 2.0) -> int:
    """``ceil(multiplier * max_f tau_int(f))`` over the supplied named series.

    The block length for :func:`block_mcse` must exceed the slowest
    autocorrelation among the observables that will be blocked, so the maximum
    is taken over all of them. The result is at least 1. Series whose IAT is
    not finite (constant, or too short to estimate) carry no usable timescale
    and are skipped; if no series yields a finite IAT, that is an error.
    """
    if multiplier <= 0 or not math.isfinite(multiplier):
        raise ValueError("multiplier must be finite and positive")
    if not series:
        raise ValueError("at least one named series is required")
    taus = [autocorrelation_time(values) for values in series.values()]
    finite = [t for t in taus if np.isfinite(t)]
    if not finite:
        raise ValueError("no supplied series has a finite autocorrelation time")
    return max(1, int(math.ceil(float(multiplier) * max(finite))))


# ---------------------------------------------------------------- bootstrap
def hierarchical_bootstrap(statistic: Callable[[dict[str, list]], float],
                           groups: Mapping[str, Sequence],
                           replicates: int,
                           seed: int) -> np.ndarray:
    """Nonparametric bootstrap that resamples several exchangeable units.

    ``groups`` maps a unit name to the list of independent objects of that kind
    (for example ``{"pt_blocks": [...], "snis_runs": [...]}``). Each replicate
    resamples every group with replacement to its original size and evaluates
    ``statistic(resampled_groups) -> float`` on the resampled collection; the
    array of ``replicates`` values is returned.

    The point is that the *whole* statistic is recomputed on each replicate, so
    the standard error of a nonlinear whole-object quantity (a relative
    Frobenius difference, a ratio, a max over categories) comes out directly.
    Such an SE must never be assembled from elementwise SEs, which would ignore
    the correlations between the elements and the nonlinearity of the map.
    """
    n_replicates = int(replicates)
    if n_replicates < 1:
        raise ValueError("replicates must be at least 1")
    if not groups:
        raise ValueError("at least one group is required")
    items = {}
    for name, unit in groups.items():
        unit_list = list(unit)
        if not unit_list:
            raise ValueError(f"group '{name}' is empty")
        items[name] = unit_list
    rng = np.random.default_rng(int(seed))
    out = np.empty(n_replicates, dtype=float)
    for r in range(n_replicates):
        resampled = {
            name: [unit[i] for i in rng.integers(0, len(unit), size=len(unit))]
            for name, unit in items.items()
        }
        out[r] = float(statistic(resampled))
    return out
