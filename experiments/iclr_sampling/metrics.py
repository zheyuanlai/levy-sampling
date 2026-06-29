"""Sampling-quality metrics.

Distance metrics
----------------
* ``mmd_rbf``       biased multi-kernel Gaussian MMD (median-heuristic bandwidth),
                    reported as MMD (= sqrt of MMD^2).
* ``w2_exact``      exact empirical W2 via the Hungarian assignment (low/moderate
                    dimension and sample size).
* ``sliced_w2``     sliced 2-Wasserstein for high-dimensional targets (ManyWell).
* ``sinkhorn_w2``   entropic-OT W2 via POT, when available.

Mode metrics
------------
* ``mode_coverage`` fraction of target modes with empirical mass above
                    max(1/n, 0.001 * target_weight); also the integer count.
* ``mode_kl``       KL(p_target || p_empirical) with epsilon smoothing (missing
                    target modes are penalised strongly).
* EMC              we standardise on  EMC = exp(-KL(p_target || p_empirical))
                    in (0, 1]; we additionally report ``emc_notebook`` matching
                    each reference notebook's own EMC convention.
* ManyWell count metrics: per-block marginal KL, deep-count distribution KL vs
  Binomial(n_blocks, p_deep), and the corresponding count-EMC.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

try:
    import ot as pot
    HAS_POT = True
except Exception:
    HAS_POT = False


# --------------------------------------------------------------------------- #
# Distance metrics
# --------------------------------------------------------------------------- #
def mmd_rbf(X: torch.Tensor, Y: torch.Tensor,
            mults=(0.25, 0.5, 1.0, 2.0, 4.0)) -> float:
    """Biased multi-kernel Gaussian MMD; returns MMD (not squared)."""
    Dxx = torch.cdist(X, X); Dyy = torch.cdist(Y, Y); Dxy = torch.cdist(X, Y)
    pool = torch.cat([X, Y], dim=0)
    Dpp = torch.cdist(pool, pool)
    N = pool.shape[0]
    iu = torch.triu_indices(N, N, 1, device=X.device)
    base = Dpp[iu[0], iu[1]].median().clamp_min(1e-6)
    out = 0.0
    for mu in mults:
        h2 = 2.0 * (base * mu) ** 2
        kxx = torch.exp(-Dxx ** 2 / h2).mean()
        kyy = torch.exp(-Dyy ** 2 / h2).mean()
        kxy = torch.exp(-Dxy ** 2 / h2).mean()
        out = out + (kxx + kyy - 2 * kxy)
    mmd2 = float(out / len(mults))
    return math.sqrt(max(mmd2, 0.0))


def w2_exact(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Exact empirical 2-Wasserstein via Hungarian assignment (equal sizes)."""
    C = (torch.cdist(X, Y) ** 2).detach().cpu().numpy()
    ri, ci = linear_sum_assignment(C)
    return math.sqrt(C[ri, ci].mean())


def sliced_w2(X: torch.Tensor, Y: torch.Tensor, n_proj: int = 200,
              seed: int = 0) -> float:
    """Sliced 2-Wasserstein distance (mean over random 1-D projections)."""
    d = X.shape[1]
    g = torch.Generator(device=X.device); g.manual_seed(seed)
    theta = torch.randn(d, n_proj, generator=g, device=X.device, dtype=X.dtype)
    theta = theta / theta.norm(dim=0, keepdim=True).clamp_min(1e-12)
    xp = (X @ theta)        # (n, n_proj)
    yp = (Y @ theta)
    xp, _ = torch.sort(xp, dim=0)
    yp, _ = torch.sort(yp, dim=0)
    n = min(xp.shape[0], yp.shape[0])
    w2_sq = ((xp[:n] - yp[:n]) ** 2).mean()
    return math.sqrt(max(float(w2_sq), 0.0))


def sinkhorn_w2(X: torch.Tensor, Y: torch.Tensor, reg: float = 1.0) -> float:
    """Entropic-OT (Sinkhorn) W2 via POT; returns NaN if POT unavailable."""
    if not HAS_POT:
        return float("nan")
    Xn = X.detach().cpu().numpy(); Yn = Y.detach().cpu().numpy()
    a = np.ones(len(Xn)) / len(Xn); b = np.ones(len(Yn)) / len(Yn)
    M = pot.dist(Xn, Yn, metric="sqeuclidean")
    cost = pot.sinkhorn2(a, b, M, reg)
    return math.sqrt(max(float(cost), 0.0))


# --------------------------------------------------------------------------- #
# Mode metrics (discrete-mode targets)
# --------------------------------------------------------------------------- #
def _empirical_mode_freq(labels: torch.Tensor, n_modes: int) -> np.ndarray:
    counts = torch.bincount(labels.reshape(-1), minlength=n_modes).float()
    return (counts / counts.sum().clamp_min(1)).cpu().numpy()


def mode_coverage(p_emp: np.ndarray, p_tgt: np.ndarray, n_samples: int):
    """Fraction (and count) of target modes whose empirical mass clears
    max(1/n_samples, 0.001 * target_weight)."""
    thr = np.maximum(1.0 / max(n_samples, 1), 0.001 * p_tgt)
    covered = (p_emp >= thr) & (p_tgt > 0)
    n_cov = int(covered.sum())
    frac = n_cov / int((p_tgt > 0).sum())
    return frac, n_cov


def kl_target_emp(p_tgt: np.ndarray, p_emp: np.ndarray, n_samples: int) -> float:
    """KL(p_target || p_empirical) with a 1/n_samples smoothing floor."""
    floor = 1.0 / max(n_samples, 1)
    pe = p_emp + floor
    pe = pe / pe.sum()
    mask = p_tgt > 0
    return float(np.sum(p_tgt[mask] * (np.log(p_tgt[mask]) - np.log(pe[mask]))))


def entropy_emc_notebook(p_emp: np.ndarray, n_modes: int) -> float:
    """MoG40-notebook EMC: exp(H(p_emp)) / n_modes  (entropy-ratio form)."""
    mask = p_emp > 0
    H = -float(np.sum(p_emp[mask] * np.log(p_emp[mask])))
    return math.exp(H) / n_modes


# --------------------------------------------------------------------------- #
# ManyWell count metrics
# --------------------------------------------------------------------------- #
def _bernoulli_kl(p: float, q: float, eps: float = 1e-9) -> float:
    p = min(max(p, eps), 1 - eps); q = min(max(q, eps), 1 - eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def manywell_count_metrics(target, X: torch.Tensor) -> Dict[str, float]:
    """Per-block marginal KL, deep-count KL vs Binomial, and count-EMC."""
    from scipy.stats import binom

    deep = target.block_deep(X)                              # (n, n_blocks)
    n = X.shape[0]
    nb = target.n_blocks
    p_deep = target.p_deep

    # per-block marginal KL averaged over blocks
    emp_block = deep.float().mean(0).cpu().numpy()           # (n_blocks,)
    block_kl = float(np.mean([_bernoulli_kl(p_deep, float(q)) for q in emp_block]))
    overall_deep_frac = float(deep.float().mean().item())

    # deep-count distribution vs Binomial(nb, p_deep)
    K = deep.sum(-1).reshape(-1)                             # (n,)
    pmf_emp = torch.bincount(K, minlength=nb + 1).float()
    pmf_emp = (pmf_emp / pmf_emp.sum()).cpu().numpy()
    pmf_tgt = binom.pmf(np.arange(nb + 1), nb, p_deep)
    floor = 1.0 / max(n, 1)
    pe = pmf_emp + floor; pe = pe / pe.sum()
    mask = pmf_tgt > 1e-12
    count_kl = float(np.sum(pmf_tgt[mask] * (np.log(pmf_tgt[mask]) - np.log(pe[mask]))))
    count_emc = math.exp(-count_kl)

    # coverage over the (nb+1) count values that carry non-trivial target mass
    sig = pmf_tgt > (1.0 / (nb + 1)) * 0.01
    cov = float(np.mean(pmf_emp[sig] > floor)) if sig.any() else float("nan")
    n_cov = int(np.sum(pmf_emp[sig] > floor)) if sig.any() else 0

    return {"block_marginal_kl": block_kl, "overall_deep_frac": overall_deep_frac,
            "count_mode_kl": count_kl, "count_emc": count_emc,
            "mode_coverage": cov, "n_covered_modes": n_cov,
            "mode_kl": count_kl, "emc": count_emc}


def manywell_config_emc(target, X: torch.Tensor) -> float:
    """ManyWell-notebook EMC: exp(-KL(p_hat_config || p_star_config)) over the
    2^n_blocks deep/shallow configurations (product Bernoulli target)."""
    nb = target.n_blocks
    deep = target.block_deep(X).long()                      # (n, nb)
    pow2 = (2 ** torch.arange(nb, device=X.device)).long()
    codes = (deep * pow2).sum(1)
    uniq, counts = torch.unique(codes, return_counts=True)
    p_hat = counts.float() / counts.sum()
    k = torch.zeros_like(uniq); u = uniq.clone()
    for _ in range(nb):
        k += (u & 1); u >>= 1
    lp = math.log(target.p_deep); lq = math.log1p(-target.p_deep)
    log_pstar = k.float() * lp + (nb - k).float() * lq
    kl = float((p_hat * (torch.log(p_hat) - log_pstar)).sum().item())
    return math.exp(-kl)


# --------------------------------------------------------------------------- #
# Top-level per-seed dispatcher
# --------------------------------------------------------------------------- #
def seed_metrics(target, X: torch.Tensor, ref: Optional[torch.Tensor],
                 gen: torch.Generator, cfg: Dict) -> Dict:
    """Compute all applicable metrics for one seed's final samples ``X`` (n, d).

    ``cfg`` controls subsample sizes and the W2 backend.  Returns a dict whose
    keys match the raw-CSV schema; inapplicable metrics are NaN.
    """
    out: Dict[str, float] = {}
    n = X.shape[0]
    nonfinite = int((~torch.isfinite(X)).sum().item())
    out["nonfinite_count"] = nonfinite
    Xf = X[torch.isfinite(X).all(dim=1)]
    if Xf.shape[0] == 0:
        return {**out, "mmd": float("nan"), "w2_or_sliced_w2": float("nan")}

    # ---- distance metrics vs reference ---- #
    w2_sub = int(cfg.get("w2_sub", 256))
    mmd_sub = int(cfg.get("mmd_sub", 1024))
    if ref is not None:
        m = min(mmd_sub, Xf.shape[0], ref.shape[0])
        ip = torch.randperm(Xf.shape[0], generator=gen, device=X.device)[:m]
        ir = torch.randperm(ref.shape[0], generator=gen, device=X.device)[:m]
        out["mmd"] = mmd_rbf(Xf[ip], ref[ir])

        mw = min(w2_sub, Xf.shape[0], ref.shape[0])
        jp = torch.randperm(Xf.shape[0], generator=gen, device=X.device)[:mw]
        jr = torch.randperm(ref.shape[0], generator=gen, device=X.device)[:mw]
        if getattr(target, "high_dim", False):
            out["w2_or_sliced_w2"] = sliced_w2(Xf[jp], ref[jr],
                                               n_proj=int(cfg.get("n_proj", 200)))
            out["w2_metric"] = "sliced_w2"
        else:
            out["w2_or_sliced_w2"] = w2_exact(Xf[jp], ref[jr])
            out["w2_metric"] = "w2_exact_hungarian"
    else:
        out["mmd"] = float("nan"); out["w2_or_sliced_w2"] = float("nan")
        out["w2_metric"] = "none"

    # ---- mode metrics ---- #
    if getattr(target, "name", "").startswith("manywell"):
        out.update(manywell_count_metrics(target, Xf))
        out["emc_notebook"] = manywell_config_emc(target, Xf)
    elif getattr(target, "has_modes", False):
        labels = target.assign_modes(Xf)
        p_emp = _empirical_mode_freq(labels, target.n_modes)
        p_tgt = target.target_mode_weights(X.device).cpu().numpy()
        frac, n_cov = mode_coverage(p_emp, p_tgt, n)
        kl = kl_target_emp(p_tgt, p_emp, n)
        out["mode_coverage"] = frac
        out["n_covered_modes"] = n_cov
        out["mode_kl"] = kl
        out["emc"] = math.exp(-kl)
        out["emc_notebook"] = entropy_emc_notebook(p_emp, target.n_modes)
        out["count_mode_kl"] = float("nan")
        out["count_emc"] = float("nan")
        out["block_marginal_kl"] = float("nan")
    return out
