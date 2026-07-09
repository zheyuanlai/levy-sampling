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
def nonfinite_frac(x: torch.Tensor) -> float:
    """Must be identically zero; metrics on survivors only would be
    survivorship bias, so nothing is ever filtered."""
    return float((~torch.isfinite(x)).any(dim=-1).to(torch.float64).mean().item())


# ------------------------------------------------------------ bias floors
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
