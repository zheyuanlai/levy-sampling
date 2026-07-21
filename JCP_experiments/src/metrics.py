"""Metrics: W2 (exact 1D / sliced / Hungarian spot-check), TV (occupancy and
density), MMD (frozen-bandwidth Gaussian, biased V-statistic), EMC, EJS,
bias floors, nonfinite fraction.

Conventions: reference sample size equals the run's N; sliced-W2 projections
are drawn ONCE from a fixed seed and reused across all times and methods;
the MMD bandwidth is frozen once by the median heuristic on the reference
sample and never recomputed (per-frame bandwidths would make curves
non-comparable).
"""
from __future__ import annotations

import math

import numpy as np
import torch


# ------------------------------------------------------------------- W2
def w2_exact_1d(x: torch.Tensor, y: torch.Tensor) -> float:
    """Exact 1D W2 via sorted coupling; x, y: (N,) or (N,1)."""
    xs = torch.sort(x.reshape(-1)).values
    ys = torch.sort(y.reshape(-1)).values
    n = min(xs.shape[0], ys.shape[0])
    return float(torch.sqrt(((xs[:n] - ys[:n]) ** 2).mean()).item())


def make_projections(d: int, L: int = 200, seed: int = 777,
                     device: str | torch.device = "cuda") -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    theta = torch.randn(L, d, generator=gen, device=device, dtype=torch.float64)
    return theta / theta.norm(dim=1, keepdim=True)


def sliced_w2(x: torch.Tensor, y: torch.Tensor, projections: torch.Tensor) -> float:
    """SW2^2 = mean_l W2^2(<theta_l, X>, <theta_l, Y>); bias floor ~ N^{-1/2}."""
    xp = torch.sort(x @ projections.T, dim=0).values          # (N, L)
    yp = torch.sort(y @ projections.T, dim=0).values
    n = min(xp.shape[0], yp.shape[0])
    return float(torch.sqrt(((xp[:n] - yp[:n]) ** 2).mean()).item())


def hungarian_w2(x: torch.Tensor, y: torch.Tensor, m: int = 500,
                 seed: int = 123) -> float:
    """Exact W2 on an m-subsample via the Hungarian algorithm (terminal
    spot-check in 2D only; exact W2 in higher d would be swamped by its
    N^{-1/d} bias floor)."""
    from scipy.optimize import linear_sum_assignment
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    ix = torch.randperm(x.shape[0], generator=gen, device=x.device)[:m]
    iy = torch.randperm(y.shape[0], generator=gen, device=y.device)[:m]
    xs, ys = x[ix], y[iy]
    cost = ((xs.unsqueeze(1) - ys.unsqueeze(0)) ** 2).sum(-1).cpu().numpy()
    r, c = linear_sum_assignment(cost)
    return float(math.sqrt(cost[r, c].mean()))


# ------------------------------------------------------------------- TV
def occupancy(labels: torch.Tensor, K: int) -> torch.Tensor:
    """Empirical occupancy p_hat over a K-cell partition."""
    return torch.bincount(labels, minlength=K).to(torch.float64) / labels.shape[0]


def occupancy_tv(p_hat: torch.Tensor, p_star: torch.Tensor) -> float:
    """TV on the partition: a LOWER BOUND on the full TV."""
    return float(0.5 * (p_hat - p_star).abs().sum().item())


def basin_kl_target_to_empirical(p_hat: torch.Tensor, p_star: torch.Tensor,
                                 pseudocount: float = 0.0) -> float:
    """Basin KL ``D_KL(p_star || p_hat)`` in nats.

    This orientation penalizes a sampler that misses a target basin. A small
    probability-space pseudocount may be supplied to keep the finite-sample
    estimate finite; production uses the Jeffreys-scale value ``0.5 / N`` and
    records that convention with the metric.
    """
    p_hat = torch.as_tensor(p_hat, dtype=torch.float64)
    p_star = torch.as_tensor(p_star, device=p_hat.device, dtype=torch.float64)
    if p_hat.shape != p_star.shape:
        raise ValueError("p_hat and p_star must have the same shape")
    if not math.isfinite(pseudocount) or pseudocount < 0:
        raise ValueError("pseudocount must be finite and non-negative")
    if (not bool(torch.isfinite(p_hat).all())
            or not bool(torch.isfinite(p_star).all())
            or bool((p_hat < 0).any()) or bool((p_star < 0).any())
            or not bool(p_hat.sum() > 0) or not bool(p_star.sum() > 0)):
        raise ValueError("basin probabilities must be finite, non-negative, and nonzero")
    q = p_hat / p_hat.sum()
    if pseudocount > 0:
        q = q + float(pseudocount)
        q = q / q.sum()
    p = p_star / p_star.sum()
    mask = p > 0
    if bool((q[mask] <= 0).any()):
        return float("inf")
    kl = (p[mask] * torch.log(p[mask] / q[mask])).sum()
    return float(torch.clamp(kl, min=0.0).item())


def density_tv_1d(x: torch.Tensor, bin_edges: torch.Tensor,
                  target_bin_mass: torch.Tensor) -> float:
    """200-bin density TV against the exact pi on a box (E1). A genuine
    density TV, unlike the occupancy TV."""
    idx = torch.bucketize(x.reshape(-1), bin_edges[1:-1])
    p_hat = torch.bincount(idx, minlength=target_bin_mass.shape[0]).to(torch.float64)
    p_hat = p_hat / p_hat.sum()
    return float(0.5 * (p_hat - target_bin_mass).abs().sum().item())


# ------------------------------------------------------------------- MMD
def median_heuristic(y: torch.Tensor, max_points: int = 2048, seed: int = 99) -> float:
    gen = torch.Generator(device=y.device)
    gen.manual_seed(seed)
    idx = torch.randperm(y.shape[0], generator=gen, device=y.device)[:max_points]
    ys = y[idx]
    dists = torch.cdist(ys, ys)
    iu = torch.triu_indices(ys.shape[0], ys.shape[0], offset=1, device=y.device)
    return float(dists[iu[0], iu[1]].median().item())


def mmd_biased(x: torch.Tensor, y: torch.Tensor, bandwidth: float) -> float:
    """Gaussian-kernel biased V-statistic; exactly ||mu_X - mu_Y||_H^2 >= 0.
    k(x,y) = exp(-||x-y||^2 / (2 sigma^2)) with sigma = bandwidth (frozen)."""
    g = 0.5 / bandwidth**2
    kxx = torch.exp(-g * torch.cdist(x, x) ** 2).mean()
    kyy = torch.exp(-g * torch.cdist(y, y) ** 2).mean()
    kxy = torch.exp(-g * torch.cdist(x, y) ** 2).mean()
    mmd2 = kxx - 2.0 * kxy + kyy
    return float(torch.sqrt(torch.clamp(mmd2, min=0.0)).item())


# --------------------------------------------------------------- EMC / EJS
def emc(p_hat: torch.Tensor) -> float:
    """EMC = exp(H(p_hat)) / K. Optimal value is exp(H(p_star))/K, which is 1
    only for uniform p_star; always plot the target line."""
    p = p_hat[p_hat > 0]
    H = float(-(p * torch.log(p)).sum().item())
    return math.exp(H) / p_hat.shape[0]


def _kl_bits(p: torch.Tensor, q: torch.Tensor) -> float:
    mask = p > 0
    return float((p[mask] * torch.log2(p[mask] / q[mask])).sum().item())


def ejs(p_hat: torch.Tensor, p_star: torch.Tensor) -> float:
    """Base-2 Jensen-Shannon divergence between occupancy and target
    (Blessing et al., arXiv:2406.07423, App. A.3). In [0,1]; 0 iff equal;
    1 iff disjoint support; quadratic near the target so it stays
    informative where TV saturates."""
    m = 0.5 * (p_hat + p_star)
    return 0.5 * _kl_bits(p_hat, m) + 0.5 * _kl_bits(p_star, m)


# ------------------------------------------------------------- nonfinite
def nonfinite_count(x: torch.Tensor) -> int:
    """Number of particles with at least one nonfinite coordinate."""
    return int((~torch.isfinite(x)).any(dim=-1).sum().item())


def nonfinite_frac(x: torch.Tensor) -> float:
    """Fraction of particles counted by :func:`nonfinite_count`.

    This must be identically zero.  Nothing is filtered because metrics on
    survivors would introduce survivorship bias.
    """
    return float((~torch.isfinite(x)).any(dim=-1).to(torch.float64).mean().item())


# ------------------------------------------------------------ bias floors
# ==================================================== chemistry-native metrics
def binned_probabilities(
    samples: torch.Tensor,
    edges: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    smooth: float = 0.0,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Histogram probabilities on a one- or multi-dimensional rectangular grid.

    Samples has shape (N,), (N, 1), or (N, d). For one dimension, edges may be
    a single tensor; for d > 1 it is a sequence of d edge tensors. The returned
    tensor has one axis per collective variable. Samples outside the supplied
    rectangle are excluded rather than accumulated in boundary bins.

    Smooth is a non-negative pseudocount per bin. Production free-energy
    estimates should report this value because it determines the finite penalty
    assigned to an empirically empty bin. Optional importance weights are
    rescaled to sum to the sample count before adding the pseudocount, so the
    smoothing convention remains in count units and is invariant to a global
    rescaling of the weights.
    """
    if smooth < 0:
        raise ValueError("smooth must be non-negative")
    if isinstance(edges, torch.Tensor):
        edge_list = (edges,)
    else:
        edge_list = tuple(edges)
    if not edge_list:
        raise ValueError("at least one edge tensor is required")
    if any(e.ndim != 1 or e.numel() < 2 for e in edge_list):
        raise ValueError("each edge tensor must be one-dimensional with >=2 entries")
    if any(not bool(torch.all(e[1:] > e[:-1])) for e in edge_list):
        raise ValueError("bin edges must be strictly increasing")

    x = samples
    if not torch.is_floating_point(x):
        x = x.to(torch.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[1] != len(edge_list):
        raise ValueError(
            f"samples have shape {tuple(samples.shape)}, but {len(edge_list)} "
            "edge tensors were supplied"
        )
    if x.shape[0] == 0:
        raise ValueError("cannot histogram an empty sample")
    weights = None
    if sample_weights is not None:
        weights = torch.as_tensor(
            sample_weights, device=x.device, dtype=torch.float64).reshape(-1)
        if weights.shape != (x.shape[0],):
            raise ValueError("sample_weights must contain one weight per sample")
        if (not bool(torch.isfinite(weights).all())
                or bool((weights < 0).any())
                or not bool(weights.sum() > 0)):
            raise ValueError("sample_weights must be finite, non-negative, and nonzero")
        # Work in effective count units; multiplying all input weights by a
        # constant therefore cannot change either the histogram or smoothing.
        weights = weights * (x.shape[0] / weights.sum())

    shape = tuple(int(e.numel() - 1) for e in edge_list)
    valid = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    per_dim: list[torch.Tensor] = []
    for j, edge in enumerate(edge_list):
        edge = edge.to(device=x.device, dtype=x.dtype)
        # Internal edges map the closed upper endpoint to the final bin; the
        # explicit validity mask rejects true underflow and overflow.
        valid &= (x[:, j] >= edge[0]) & (x[:, j] <= edge[-1])
        per_dim.append(torch.bucketize(
            x[:, j].contiguous(), edge[1:-1], right=True
        ))
    if not bool(valid.any()) and smooth == 0:
        raise ValueError("no samples fall inside the supplied histogram domain")

    linear = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    stride = 1
    for idx, n_bin in zip(reversed(per_dim), reversed(shape)):
        linear += idx.to(torch.long) * stride
        stride *= n_bin
    if weights is None:
        counts = torch.bincount(
            linear[valid], minlength=math.prod(shape)).to(torch.float64)
    else:
        counts = torch.zeros(math.prod(shape), dtype=torch.float64, device=x.device)
        counts.scatter_add_(0, linear[valid], weights[valid])
    counts = counts.reshape(shape) + float(smooth)
    total = counts.sum()
    if not bool(total > 0):
        raise ValueError("histogram has zero total mass")
    return counts / total


def binned_probabilities_with_outside(
    samples: torch.Tensor,
    edges: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    smooth: float = 0.0,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """Flattened cell probabilities with an appended off-grid cell.

    Returns (p, outside_mass). `p` has prod(shape) + 1 entries, the last being
    the off-grid cell, and the pseudocount `smooth` is applied identically to
    every entry including that one. `outside_mass` is the raw, unsmoothed
    fraction of weight landing outside the grid.

    `binned_probabilities` drops off-grid samples and renormalises, so a sampler
    that leaks mass has its worst-placed mass simply removed from the score.
    Keeping the off-grid cell makes the free-energy comparison mass-conserving.
    """
    edge_list = (edges,) if isinstance(edges, torch.Tensor) else tuple(edges)
    x = samples
    if not torch.is_floating_point(x):
        x = x.to(torch.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] == 0:
        raise ValueError("cannot histogram an empty sample")
    w = None
    if sample_weights is not None:
        w = torch.as_tensor(sample_weights, device=x.device,
                            dtype=torch.float64).reshape(-1)
        if w.shape != (x.shape[0],):
            raise ValueError("sample_weights must contain one weight per sample")
        w = w * (x.shape[0] / w.sum())

    inside = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
    for j, edge in enumerate(edge_list):
        edge = edge.to(device=x.device, dtype=x.dtype)
        inside &= (x[:, j] >= edge[0]) & (x[:, j] <= edge[-1])

    total = float(x.shape[0]) if w is None else float(w.sum().item())
    out_w = (float((~inside).sum().item()) if w is None
             else float(w[~inside].sum().item()))
    outside_mass = out_w / total

    n_cells = math.prod(int(e.numel() - 1) for e in edge_list)
    if not bool(inside.any()):
        counts = torch.zeros(n_cells, dtype=torch.float64, device=x.device)
    else:
        p_in = binned_probabilities(
            x[inside], edge_list, smooth=0.0,
            sample_weights=None if w is None else w[inside])
        counts = p_in.reshape(-1) * (total - out_w)
    counts = torch.cat([
        counts,
        torch.tensor([out_w], dtype=torch.float64, device=x.device),
    ]) + float(smooth)
    return counts / counts.sum(), outside_mass


def _bin_volumes(
    edges: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Cell volumes for a rectangular histogram, with histogram-shaped output."""
    edge_list = (edges,) if isinstance(edges, torch.Tensor) else tuple(edges)
    widths = [(e[1:] - e[:-1]).to(device=device, dtype=dtype) for e in edge_list]
    volume = widths[0]
    for width in widths[1:]:
        volume = volume.unsqueeze(-1) * width.reshape((1,) * volume.ndim + (-1,))
    return volume


def reduced_free_energy(
    probability: torch.Tensor,
    bin_volume: torch.Tensor | None = None,
    probability_floor: float = 0.0,
) -> torch.Tensor:
    """Reduced free energy beta*F = -log(p/|C|), in k_B T units.

    No additive shift is applied. Comparisons must align profiles with a fitted
    additive constant rather than independently pinning their noisy minima.
    Zero-probability bins are infinite unless probability_floor is positive.
    """
    p = torch.as_tensor(probability, dtype=torch.float64)
    if not bool(torch.isfinite(p).all()) or bool((p < 0).any()):
        raise ValueError("probabilities must be finite and non-negative")
    if not bool(p.sum() > 0):
        raise ValueError("probabilities must have positive total mass")
    p = p / p.sum()
    if probability_floor < 0:
        raise ValueError("probability_floor must be non-negative")
    if probability_floor > 0:
        p = p.clamp_min(float(probability_floor))
        p = p / p.sum()
    if bin_volume is None:
        volume = torch.ones_like(p)
    else:
        volume = torch.as_tensor(bin_volume, device=p.device, dtype=p.dtype)
        if (volume.shape != p.shape
                or not bool(torch.isfinite(volume).all())
                or bool((volume <= 0).any())):
            raise ValueError(
                "bin_volume must be finite, positive, and match probability"
            )
    return -torch.log(p / volume)


def free_energy_rmse_from_probabilities(
    p_hat: torch.Tensor,
    p_ref: torch.Tensor,
    *,
    pi_min: float = 0.0,
    weights: str | torch.Tensor = "uniform",
    bin_volume: torch.Tensor | None = None,
    probability_floor: float = 1e-300,
    always_keep: torch.Tensor | None = None,
) -> float:
    """Additive-constant-aligned free-energy RMSE in k_B T units.

    For A_hat=-log(p_hat/|C|), A_ref=-log(p_ref/|C|), and their weighted mean
    difference c, this returns sqrt(sum_i w_i*(A_hat_i-A_ref_i-c)^2/sum_i w_i).
    Uniform weights measure FES shape equally; reference weights emphasize
    thermodynamically common regions. Only p_ref >= pi_min bins are retained.

    `always_keep` is a boolean mask of cells exempt from the pi_min cut. It
    exists for the off-grid cell, whose reference mass is legitimately tiny but
    whose empirical mass is the sharpest signature of a leaking sampler; cutting
    it on reference mass would hide exactly the defect it measures.
    """
    p_hat = torch.as_tensor(p_hat, dtype=torch.float64)
    p_ref = torch.as_tensor(p_ref, device=p_hat.device, dtype=torch.float64)
    if p_hat.shape != p_ref.shape:
        raise ValueError("p_hat and p_ref must have the same shape")
    if pi_min < 0:
        raise ValueError("pi_min must be non-negative")
    if (not bool(torch.isfinite(p_ref).all())
            or bool((p_ref < 0).any())
            or not bool(p_ref.sum() > 0)):
        raise ValueError(
            "p_ref must be finite, non-negative, and have positive mass"
        )
    p_ref_norm = p_ref / p_ref.sum()
    # Zero-reference-mass cells have undefined reference free energy and must
    # never enter the FES norm, including when pi_min is exactly zero.
    mask = (p_ref_norm > 0) & (p_ref_norm >= float(pi_min))
    if always_keep is not None:
        keep = torch.as_tensor(always_keep, device=p_hat.device, dtype=torch.bool)
        if keep.shape != p_hat.shape:
            raise ValueError("always_keep must match the probability shape")
        # still require positive reference mass: -log(0) has no finite target
        mask = mask | (keep & (p_ref_norm > 0))
    if not bool(mask.any()):
        raise ValueError("pi_min excludes every reference bin")
    a_hat = reduced_free_energy(p_hat, bin_volume, probability_floor)
    a_ref = reduced_free_energy(p_ref_norm, bin_volume, probability_floor)
    delta = (a_hat - a_ref)[mask]
    # With no probability floor, a missed supported bin has infinite rather
    # than NaN free-energy error.
    if bool(torch.isinf(delta).any()):
        return float("inf")
    if not bool(torch.isfinite(delta).all()):
        raise ValueError("free-energy difference contains NaN")
    if isinstance(weights, str):
        if weights == "uniform":
            w = torch.ones_like(delta)
        elif weights == "reference":
            w = p_ref_norm[mask]
        else:
            raise ValueError("weights must be 'uniform', 'reference', or a tensor")
    else:
        w_all = torch.as_tensor(weights, device=p_hat.device, dtype=torch.float64)
        if w_all.shape != p_hat.shape:
            raise ValueError("tensor weights must match the probability shape")
        w = w_all[mask]
    if not bool(torch.isfinite(w).all()) or bool((w < 0).any()) or not bool(w.sum() > 0):
        raise ValueError("weights must be finite, non-negative, and have positive mass")
    offset = (w * delta).sum() / w.sum()
    return float(torch.sqrt((w * (delta - offset).square()).sum() / w.sum()).item())


def free_energy_profile(cv: torch.Tensor, edges: torch.Tensor, beta: float,
                        smooth: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Physical free-energy profile and bin probabilities for a 1D CV.

    F=-beta^{-1}log(p/bin_width) is returned in energy units and shifted to
    min(F)=0 for plotting. Use free_energy_profile_error for aligned RMSE in
    k_B T units.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    p = binned_probabilities(cv, edges, smooth=smooth)
    volume = _bin_volumes(edges, device=p.device, dtype=p.dtype)
    F = reduced_free_energy(p, volume) / float(beta)
    return F - F.min(), p


def free_energy_profile_error(cv: torch.Tensor, edges: torch.Tensor, beta: float,
                              ref_F: torch.Tensor, ref_p: torch.Tensor,
                              pi_min: float) -> float:
    """Aligned free-energy RMSE in k_B T units on supported reference bins.

    The legacy signature is retained because experiment factories cache ref_F.
    The error is computed from bin probabilities, so beta cancels after
    conversion to reduced units. ref_F is shape-validated for compatibility.
    """
    if beta <= 0:
        raise ValueError("beta must be positive")
    if ref_F.shape != ref_p.shape:
        raise ValueError("ref_F and ref_p must have the same shape")
    _, p_hat = free_energy_profile(cv, edges, beta)
    volume = _bin_volumes(edges, device=p_hat.device, dtype=p_hat.dtype)
    return free_energy_rmse_from_probabilities(
        p_hat, ref_p, pi_min=pi_min, weights="uniform", bin_volume=volume
    )


def basin_rel_mass_error(p_hat: torch.Tensor, p_star: torch.Tensor,
                         eps: float = 1e-12) -> tuple[float, float]:
    """Per-basin relative mass error: (max_k |p_hat_k - p*_k| / p*_k, L1 sum)."""
    rel = (p_hat - p_star).abs() / (p_star + eps)
    return float(rel.max().item()), float((p_hat - p_star).abs().sum().item())


def observable_error(v_hat: torch.Tensor, ref_mean: float,
                     ref_var: float) -> tuple[float, float]:
    """(|<V> - <V>_pi|, |Var(V) - Var_pi(V)|) for an energy sample v_hat=(N,)."""
    return (abs(float(v_hat.mean().item()) - ref_mean),
            abs(float(v_hat.var(unbiased=True).item()) - ref_var))


def energy_hist_overlap(E: torch.Tensor, edges: torch.Tensor,
                        ref_hist: torch.Tensor) -> float:
    """Histogram overlap with out-of-grid sample mass counted as non-overlap.

    ``ref_hist`` is normalized on the frozen reference grid.  Empirical energy
    values outside that grid must not be clamped into an edge bin or discarded
    and renormalized, either of which would hide energetic tail failures.
    """
    values = E.reshape(-1)
    if values.numel() == 0:
        raise ValueError("energy sample must be nonempty")
    valid = (values >= edges[0]) & (values <= edges[-1])
    idx = torch.bucketize(values[valid], edges[1:-1], right=True)
    h = torch.bincount(idx, minlength=edges.shape[0] - 1).to(torch.float64)
    h = h / values.numel()
    return float(torch.minimum(h, ref_hist).sum().item())


def ksd_imq(x: torch.Tensor, score: torch.Tensor, c: float = 1.0,
            beta_k: float = -0.5, max_points: int = 512, seed: int = 17) -> float:
    """Kernel Stein discrepancy (IMQ kernel k=(c^2+||x-y||^2)^beta_k), V-statistic
    on an m-subsample. score = grad log pi = -beta grad V, evaluated at x.

    NB (documented blind spot): KSD is INSENSITIVE to mode-imbalance -- a chain
    fully trapped in one well can have small KSD. Secondary metric only; use the
    basin-aware metrics for the failure mode we actually care about."""
    n = x.shape[0]
    if n > max_points:
        g = torch.Generator(device=x.device); g.manual_seed(seed)
        idx = torch.randperm(n, generator=g, device=x.device)[:max_points]
        x, score = x[idx], score[idx]
    r = x.unsqueeze(1) - x.unsqueeze(0)                      # (m, m, d)
    s = (r * r).sum(-1)                                      # (m, m) = ||x-y||^2
    base = c * c + s
    d = x.shape[1]
    sx_sy = score @ score.T                                  # (m, m)
    sxmsy_r = ((score.unsqueeze(1) - score.unsqueeze(0)) * r).sum(-1)  # (sx - sy).r
    k0 = (base ** beta_k) * sx_sy \
        - 2.0 * beta_k * base ** (beta_k - 1.0) * sxmsy_r \
        - 2.0 * beta_k * d * base ** (beta_k - 1.0) \
        - 4.0 * beta_k * (beta_k - 1.0) * base ** (beta_k - 2.0) * s
    val = k0.mean()
    return float(torch.sqrt(torch.clamp(val, min=0.0)).item())


# ============================ 1D density / CDF metrics (collaborator parity)
def _trapz(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (0.5 * (f[1:] + f[:-1]) * (x[1:] - x[:-1])).sum()


def _torch_interp(xq: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    idx = torch.searchsorted(xp, xq).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    t = (xq - x0) / (x1 - x0).clamp(min=1e-30)
    return f0 + t * (f1 - f0)


def kde_on_grid(cv: torch.Tensor, x_grid: torch.Tensor, bandwidth: float) -> torch.Tensor:
    """Gaussian-KDE density of `cv` evaluated on x_grid, normalised on the grid."""
    z = (x_grid.unsqueeze(1) - cv.reshape(1, -1)) / bandwidth
    rho = torch.exp(-0.5 * z * z).sum(1) / (cv.numel() * bandwidth * math.sqrt(2.0 * math.pi))
    return rho / _trapz(rho, x_grid).clamp(min=1e-300)


def density_cdf_metrics(cv: torch.Tensor, x_grid: torch.Tensor,
                        target_pdf: torch.Tensor, target_cdf: torch.Tensor,
                        bandwidth: float, chi_mask: torch.Tensor) -> dict:
    """1D density/CDF errors of an empirical sample against a target given on
    x_grid (all along a collective variable cv = 1D). Returns:
      W1        = int|F_hat - F*| dx           (= CDF L1; collaborator 'W1')
      CDF_sup   = sup|F_hat - F*|              (Kolmogorov-Smirnov; 'CDF_sup')
      cdf_L2    = sqrt(int (F_hat-F*)^2 dx)    (Cramer-von-Mises-like)
      pdf_L1    = int|rho_hat - rho*| dx       (= 2 x density-TV)
      pdf_L2    = sqrt(int (rho_hat-rho*)^2 dx)
      KDE_chi2  = int (rho_hat-rho*)^2/rho* dx  on [1%,99%] (collaborator 'KDE_chi2')
    """
    cv = cv.reshape(-1)
    s = torch.sort(cv).values
    Femp = torch.searchsorted(s, x_grid, right=True).to(torch.float64) / cv.numel()
    dC = Femp - target_cdf
    rho = kde_on_grid(cv, x_grid, bandwidth)
    dP = rho - target_pdf
    m = chi_mask
    return {
        "W1_cdf": float(_trapz(dC.abs(), x_grid).item()),
        "CDF_sup": float(dC.abs().max().item()),
        "cdf_L2": float(torch.sqrt(_trapz(dC * dC, x_grid).clamp(min=0)).item()),
        "pdf_L1": float(_trapz(dP.abs(), x_grid).item()),
        "pdf_L2": float(torch.sqrt(_trapz(dP * dP, x_grid).clamp(min=0)).item()),
        "KDE_chi2": float(_trapz((dP[m] ** 2) / target_pdf[m].clamp(min=1e-300),
                                 x_grid[m]).item()),
    }


def bin_chi2_pit(cv: torch.Tensor, x_grid: torch.Tensor, target_cdf: torch.Tensor,
                 n_bins: int) -> float:
    """PIT chi-squared: under the target, F*(X) ~ Uniform[0,1]. Bin F*(cv) into
    n_bins equal bins and compare to 1/n_bins (collaborator 'bin_chi2_M')."""
    u = _torch_interp(cv.reshape(-1), x_grid, target_cdf).clamp(0.0, 1.0)
    idx = (u * n_bins).long().clamp(0, n_bins - 1)
    counts = torch.bincount(idx, minlength=n_bins).to(torch.float64)
    p = counts / counts.sum()
    return float((n_bins * ((p - 1.0 / n_bins) ** 2).sum()).item())


def well_tv(p_hat: torch.Tensor, p_star: torch.Tensor) -> float:
    """Occupancy TV on the well partition (collaborator 'well_TV'); for 2 equal
    wells this is |p_hat[0] - 0.5|, identical to occupancy_tv there."""
    return float(0.5 * (p_hat - p_star).abs().sum().item())


# ==================================================== MCMC convergence (post-hoc)
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


def iat_1d(x: np.ndarray, c: float | None = None) -> float:
    """Conservative IAT using Geyer's initial-positive/monotone sequence.

    With Gamma_k = rho[2k] + rho[2k+1], pair sums are truncated before the
    first non-positive value and monotonised by cumulative minima. The estimate
    is tau = -1 + 2 sum_k Gamma_k and is conservatively bounded below by one.
    A constant series has no observed decorrelation and returns infinity;
    series with fewer than two finite draws return NaN.

    The c argument is accepted only for compatibility with the former Sokal
    estimator and has no effect.
    """
    del c
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
    tau = -1.0 + 2.0 * monotone.sum()
    return float(max(tau, 1.0))


def ess_from_series(series: np.ndarray) -> float:
    """Sum per-chain effective sample sizes for independent chains.

    Series has shape (n_chains, n_draws). A constant chain contributes zero
    ESS. Nonfinite or too-short chains make the result undefined (NaN) rather
    than being silently dropped.
    """
    series = np.atleast_2d(np.asarray(series, dtype=float))
    if series.shape[1] < 2 or not np.all(np.isfinite(series)):
        return float("nan")
    total = 0.0
    for chain in series:
        tau = iat_1d(chain)
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


def _basic_rhat(chains: np.ndarray) -> float:
    """Classical between/within-chain variance ratio on transformed data."""
    chains = np.asarray(chains, dtype=float)
    _, n = chains.shape
    chain_means = chains.mean(axis=1)
    B = n * chain_means.var(ddof=1)
    W = chains.var(axis=1, ddof=1).mean()
    if W <= 0:
        return 1.0 if B <= 0 else float("inf")
    var_plus = (n - 1.0) / n * W + B / n
    return float(np.sqrt(max(var_plus / W, 0.0)))


def _rank_normalize(chains: np.ndarray) -> np.ndarray:
    flat = np.asarray(chains, dtype=float).reshape(-1)
    ranks = _average_ranks(flat)
    # Blom's offset; S + 1/4 is the correct denominator.
    z = _norm_ppf((ranks - 3.0 / 8.0) / (flat.size + 1.0 / 4.0))
    return z.reshape(np.asarray(chains).shape)


def split_rhat_components(chains: np.ndarray) -> tuple[float, float]:
    """Return rank-normalized bulk and folded split-R-hat components."""
    chains = np.atleast_2d(np.asarray(chains, dtype=float))
    if not np.all(np.isfinite(chains)):
        return float("nan"), float("nan")
    M, N = chains.shape
    h = N // 2
    if M < 2 or h < 2:
        return float("nan"), float("nan")
    split = np.concatenate([chains[:, :h], chains[:, N - h:]], axis=0)
    if np.all(split == split.flat[0]):
        # R-hat is undefined, not evidence of convergence, when every retained
        # chain is the same constant series.
        return float("nan"), float("nan")
    bulk = _basic_rhat(_rank_normalize(split))
    folded = np.abs(split - np.median(split))
    tail = _basic_rhat(_rank_normalize(folded))
    return bulk, tail


def split_rhat(chains: np.ndarray) -> float:
    """Maximum of bulk and folded rank-normalized split-R-hat.

    Average ranks are used for ties, which is essential for discrete basin
    indicators. The folded component detects scale/tail non-convergence that
    can be missed by a location-only rank diagnostic.
    """
    bulk, folded = split_rhat_components(chains)
    if np.isnan(bulk) or np.isnan(folded):
        return float("nan")
    return float(max(bulk, folded))


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
    lo = p < plow; hi = p > phigh; mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(p[lo]))
    out[lo] = (((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5]) / \
              ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    q = p[mid] - 0.5; r = q * q
    out[mid] = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = np.sqrt(-2 * np.log(1 - p[hi]))
    out[hi] = -(((((cc[0]*q+cc[1])*q+cc[2])*q+cc[3])*q+cc[4])*q+cc[5]) / \
               ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    return out


def round_trips(labels_t: np.ndarray, home: int, far: int) -> float:
    """Mean observed home->far->home trips per chain.

    A chain is armed only after ``home`` has actually been observed, so a
    far-start followed by home is not spuriously counted as a round trip.
    ``labels_t`` has shape ``(T, chains)`` (or ``(T,)`` for one chain).
    """
    labels_t = np.atleast_2d(labels_t.T).T if labels_t.ndim == 1 else labels_t
    total = 0
    C = labels_t.shape[1]
    for c in range(C):
        seq = labels_t[:, c]
        state = "seek_home"; trips = 0
        for lab in seq:
            if state == "seek_home" and lab == home:
                state = "home"
            elif state == "home" and lab == far:
                state = "far"
            elif state == "far" and lab == home:
                state = "home"; trips += 1
        total += trips
    return total / max(C, 1)


def first_passage_observations(labels_t: np.ndarray, home: int, far: int,
                               dt: float, steps_per_frame: int
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Right-censored home-to-far passage observations.

    Rows of labels_t are observations at times 0, Delta, ..., (T-1)Delta,
    where Delta=dt*steps_per_frame. Only chains initially in home are eligible.
    Returned arrays are (time, event_observed); an unobserved hit is censored at
    the final recorded time.
    """
    labels_t = np.asarray(labels_t)
    labels_t = labels_t if labels_t.ndim == 2 else labels_t[:, None]
    if labels_t.ndim != 2 or labels_t.shape[0] < 2:
        raise ValueError("labels_t must contain at least two time points")
    if dt <= 0 or steps_per_frame <= 0:
        raise ValueError("dt and steps_per_frame must be positive")
    T, _ = labels_t.shape
    delta = float(dt) * int(steps_per_frame)
    horizon = (T - 1) * delta
    times: list[float] = []
    events: list[bool] = []
    for c in range(labels_t.shape[1]):
        if labels_t[0, c] != home:
            continue
        hit = np.flatnonzero(labels_t[1:, c] == far)
        if hit.size:
            times.append(float(hit[0] + 1) * delta)
            events.append(True)
        else:
            times.append(horizon)
            events.append(False)
    return np.asarray(times, dtype=float), np.asarray(events, dtype=bool)


def kaplan_meier_rmst(times: np.ndarray, events: np.ndarray,
                      tau: float | None = None) -> float:
    """Kaplan-Meier restricted mean survival/passage time up to tau."""
    times = np.asarray(times, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=bool).reshape(-1)
    if times.shape != events.shape or times.size == 0:
        return float("nan")
    if not np.all(np.isfinite(times)) or np.any(times < 0):
        raise ValueError("times must be finite and non-negative")
    if tau is None:
        tau = float(times.max())
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and non-negative")
    tau = float(tau)

    survival = 1.0
    area = 0.0
    previous = 0.0
    at_risk = int(times.size)
    for t_value in np.unique(times[times <= tau]):
        t = float(t_value)
        area += survival * (t - previous)
        at_time = times == t
        n_event = int(np.sum(events & at_time))
        n_censor = int(np.sum((~events) & at_time))
        if at_risk > 0 and n_event:
            survival *= 1.0 - n_event / at_risk
        at_risk -= n_event + n_censor
        previous = t
    if previous < tau:
        area += survival * (tau - previous)
    return float(area)


def exponential_waiting_time_mle(labels_t: np.ndarray, home: int, far: int,
                                 dt: float, steps_per_frame: int) -> float:
    """Exponential waiting-time MLE: total exposure divided by event count.

    This model-dependent quantity has an explicit name so it is not confused
    with a general empirical mean first-passage time.
    """
    times, events = first_passage_observations(
        labels_t, home, far, dt, steps_per_frame
    )
    if times.size == 0:
        return float("nan")
    n_event = int(events.sum())
    return float(times.sum() / n_event) if n_event else float("inf")


def committed_mfpt(labels_t: np.ndarray, home: int, far: int, dt: float,
                   steps_per_frame: int) -> float:
    """Kaplan-Meier restricted mean home-to-far passage time.

    This backwards-compatible name now reports a nonparametric RMST at the
    recorded horizon, correctly accounting for right censoring and filtering
    to chains initially committed to home. It is an algorithmic mixing
    diagnostic, not a physical reaction time.
    """
    times, events = first_passage_observations(
        labels_t, home, far, dt, steps_per_frame
    )
    if times.size == 0:
        return float("nan")
    # Restrict at the declared common observation horizon, not at the latest
    # event.  If every eligible chain hits early, ``times.max()`` is shorter
    # than the experiment horizon and would silently change the estimand.
    n_frames = np.asarray(labels_t).shape[0]
    horizon = (n_frames - 1) * float(dt) * int(steps_per_frame)
    return kaplan_meier_rmst(times, events, tau=horizon)


def bias_floors(sample_ref, two_sample_fns: dict, one_sample_fns: dict, n: int,
                replicates: int = 20, seed0: int = 5000,
                device: str | torch.device = "cuda") -> dict:
    """For each metric: the metric between two INDEPENDENT reference samples
    of size n (two_sample_fns: name -> fn(x, y)), or the metric of one fresh
    reference sample against the target (one_sample_fns: name -> fn(x), for
    occupancy-type metrics), averaged over `replicates`. Reported as
    mean/std; plotted as a dashed line on every panel. Without this, every
    plateau is uninterpretable."""
    vals: dict[str, list[float]] = {k: [] for k in {**two_sample_fns, **one_sample_fns}}
    for rep in range(replicates):
        g1 = torch.Generator(device=device)
        g1.manual_seed(seed0 + 31 * rep)
        g2 = torch.Generator(device=device)
        g2.manual_seed(seed0 + 31 * rep + 17)
        xa, xb = sample_ref(n, g1), sample_ref(n, g2)
        for name, fn in two_sample_fns.items():
            vals[name].append(fn(xa, xb))
        for name, fn in one_sample_fns.items():
            vals[name].append(fn(xa))
    return {name: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
            for name, v in vals.items()}
