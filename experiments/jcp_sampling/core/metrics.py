
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


def basin_population_metrics(labels: torch.Tensor, target_probs: Optional[torch.Tensor]) -> dict:
    labels = labels.reshape(-1).detach()
    if target_probs is None:
        return {"basin_population_error": float("nan"), "basin_kl": float("nan"), "n_basins_seen": int(torch.unique(labels).numel())}
    n = int(target_probs.numel())
    counts = torch.bincount(labels.clamp_min(0), minlength=n).float()[:n]
    emp = counts / counts.sum().clamp_min(1.0)
    tgt = target_probs.to(device=emp.device, dtype=emp.dtype)
    err = 0.5 * torch.abs(emp - tgt).sum()
    floor = 1.0 / max(1, int(labels.numel()))
    pe = (emp + floor); pe = pe / pe.sum()
    kl = (tgt * (torch.log(tgt.clamp_min(1e-12)) - torch.log(pe.clamp_min(1e-12)))).sum()
    return {"basin_population_error": float(err.item()), "basin_kl": float(kl.item()),
            "n_basins_seen": int((counts > 0).sum().item())}


def transition_matrix(label_trace: np.ndarray, n_basins: int) -> np.ndarray:
    M = np.zeros((n_basins, n_basins), dtype=float)
    arr = np.asarray(label_trace, dtype=int).reshape(-1)
    if arr.size < 2:
        return M
    for a, b in zip(arr[:-1], arr[1:]):
        if 0 <= a < n_basins and 0 <= b < n_basins:
            M[a, b] += 1
    row = M.sum(axis=1, keepdims=True)
    return np.divide(M, row, out=np.zeros_like(M), where=row > 0)


def autocorr_fft(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2 or np.var(x) == 0:
        return np.ones(1)
    x = x - x.mean()
    n = x.size
    f = np.fft.rfft(x, n=2*n)
    ac = np.fft.irfft(f * np.conjugate(f))[:n]
    ac /= ac[0]
    return ac


def integrated_autocorrelation_time(x, method: str = "initial_positive") -> float:
    ac = autocorr_fft(x)
    if ac.size <= 1:
        return 1.0
    if method == "initial_positive":
        pos = ac[1:]
        m = np.argmax(pos < 0)
        if m == 0 and not (pos[0] < 0):
            m = min(len(pos), 1000)
        return float(max(1.0, 1.0 + 2.0 * np.sum(pos[:m])))
    return float(max(1.0, 1.0 + 2.0 * np.sum(ac[1:])))


def ess_from_iat(n_samples: int, iat: float) -> float:
    return float(max(1.0, n_samples / max(float(iat), 1.0)))


def manywell_metrics(target, X: torch.Tensor) -> dict:
    deep = target.block_deep(X).float()
    emp = deep.mean(0).detach().cpu().numpy()
    p = float(target.p_deep)
    eps = 1e-9
    block_kl = np.mean([p * math.log(max(p, eps) / max(q, eps)) + (1-p) * math.log(max(1-p, eps) / max(1-q, eps)) for q in emp])
    counts = deep.sum(-1).long()
    nb = target.n_blocks
    hist = torch.bincount(counts, minlength=nb+1).float().cpu().numpy(); hist /= max(hist.sum(), 1.0)
    from math import comb
    tgt = np.array([comb(nb, k) * (p**k) * ((1-p)**(nb-k)) for k in range(nb+1)], dtype=float)
    pe = hist + 1.0 / max(1, X.shape[0]); pe /= pe.sum()
    mask = tgt > 1e-12
    count_kl = float(np.sum(tgt[mask] * (np.log(tgt[mask]) - np.log(pe[mask]))))
    return {"block_marginal_kl": float(block_kl), "count_mode_kl": count_kl,
            "count_emc": float(math.exp(-count_kl)), "deep_count_mean": float(counts.float().mean().item())}


def free_energy_rmse_2d(target, X: torch.Tensor, nbins: int = 40, min_count: int = 5) -> float:
    """RMSE between the empirical 2D free energy and the true surface, over well-sampled bins.

    For a 2D configuration-space target the free-energy surface is F(x,y) = V(x,y) + const,
    so the empirical F_hat = -beta^-1 log p_hat is compared directly to V on a coarse grid,
    restricted to bins the sampler actually populated (>= min_count) and aligned by their mean.
    A coarse grid + count threshold avoids the empty-bin blow-up that made the previous
    fine-grid version return a near-constant, non-discriminating value at practical sample sizes.
    """
    if not hasattr(target, "domain"):
        return float("nan")
    xmin, xmax, ymin, ymax = target.domain
    # periodic (torus) targets accumulate state in unwrapped coordinates; wrap into the domain
    Xd = target._wrap(X) if hasattr(target, "_wrap") else X
    x = Xd.detach().cpu().numpy()
    hx = np.linspace(xmin, xmax, nbins + 1)
    hy = np.linspace(ymin, ymax, nbins + 1)
    H, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=[hx, hy])
    cx = 0.5 * (hx[:-1] + hx[1:]); cy = 0.5 * (hy[:-1] + hy[1:])
    CX, CY = np.meshgrid(cx, cy, indexing="ij")
    pts = torch.tensor(np.stack([CX.ravel(), CY.ravel()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        Vref = target.potential(pts).cpu().numpy().reshape(nbins, nbins)
    mask = H >= int(min_count)
    if int(mask.sum()) < 5:
        return float("nan")
    Fh = np.where(H > 0, -np.log(np.maximum(H, 1e-12)) / target.beta, np.nan)
    fh = Fh[mask] - np.nanmean(Fh[mask])
    fr = Vref[mask] - np.nanmean(Vref[mask])
    return float(np.sqrt(np.mean((fh - fr) ** 2)))


def cdf_sup_error(samples, ref) -> float:
    """Kolmogorov sup-distance between the empirical CDFs of a 1D sample and the reference."""
    s = np.sort(np.asarray(samples, dtype=float)); s = s[np.isfinite(s)]
    r = np.sort(np.asarray(ref, dtype=float)); r = r[np.isfinite(r)]
    if s.size == 0 or r.size == 0:
        return float("nan")
    grid = np.sort(np.concatenate([s, r]))
    Fs = np.searchsorted(s, grid, side="right") / s.size
    Fr = np.searchsorted(r, grid, side="right") / r.size
    return float(np.max(np.abs(Fs - Fr)))


def density_l1_error(samples, ref, nbins: int = 100, pad: float = 0.25) -> float:
    """L1 distance between histogram densities of a 1D sample and the reference on a shared grid."""
    s = np.asarray(samples, dtype=float); s = s[np.isfinite(s)]
    r = np.asarray(ref, dtype=float); r = r[np.isfinite(r)]
    if s.size == 0 or r.size == 0:
        return float("nan")
    lo = min(s.min(), r.min()) - pad; hi = max(s.max(), r.max()) + pad
    edges = np.linspace(lo, hi, int(nbins) + 1)
    hs, _ = np.histogram(s, bins=edges, density=True)
    hr, _ = np.histogram(r, bins=edges, density=True)
    return float(np.sum(np.abs(hs - hr)) * (edges[1] - edges[0]))


def basin_tv_series(label_hist, target_probs) -> np.ndarray:
    """Per-recorded-time total variation between empirical basin occupation and the target."""
    H = np.asarray(label_hist)
    if H.ndim == 1:
        H = H[:, None]
    tp = np.asarray(target_probs, dtype=float); n = tp.size
    out = np.empty(H.shape[0], dtype=float)
    for t in range(H.shape[0]):
        counts = np.bincount(np.clip(H[t], 0, n - 1), minlength=n)[:n].astype(float)
        emp = counts / max(counts.sum(), 1.0)
        out[t] = 0.5 * np.abs(emp - tp).sum()
    return out


def threshold_time(tv_series, times, tau: float = 0.1) -> float:
    """First recorded time at which the TV series is at or below ``tau`` (nan if never)."""
    tv = np.asarray(tv_series, dtype=float); times = np.asarray(times, dtype=float)
    idx = np.where(tv <= float(tau))[0]
    return float(times[idx[0]]) if idx.size else float("nan")


def first_all_basin_coverage_time(label_hist, times, n_basins: int):
    """Mean over chains of the first time a chain has visited all ``n_basins`` basins.

    Returns (mean_coverage_time_over_covered_chains, fraction_of_chains_that_covered).
    """
    H = np.asarray(label_hist); times = np.asarray(times, dtype=float)
    if H.ndim == 1:
        H = H[:, None]
    T, N = H.shape
    first_seen = np.full((int(n_basins), N), np.inf)
    for b in range(int(n_basins)):
        hit = (H == b)
        has = hit.any(0)
        idx = np.argmax(hit, axis=0)
        first_seen[b] = np.where(has, times[np.clip(idx, 0, T - 1)], np.inf)
    cover = first_seen.max(0)
    finite = np.isfinite(cover)
    mean_cov = float(cover[finite].mean()) if finite.any() else float("nan")
    return mean_cov, float(finite.mean())


def observable_bias_metrics(target, X: torch.Tensor, ref: torch.Tensor) -> dict:
    out = {}
    obs = target.observables(X)
    refobs = target.observables(ref)
    for k, v in obs.items():
        if v.ndim == 1 and k in refobs:
            out[f"observable_bias_{k}"] = float((v.mean() - refobs[k].mean()).abs().item())
    return out


def mixing_metrics(label_hist, cv_series, iat_stride: int = 1, dt: float = 0.0, burn_frac: float = 0.5) -> dict:
    """Slow-mode IAT/ESS and basin-transition diagnostics.

    ``label_hist`` is an integer array of shape (T, N): the basin label of each of the N
    independent chains at T recorded times (spacing ``iat_stride`` integrator steps), used for
    robust transition counting. ``cv_series`` is the length-T ensemble mean of a continuous,
    non-saturating slow collective variable (``potential.slow_cv``); its integrated
    autocorrelation time is the per-chain slow-mode IAT (for N independent chains the normalized
    autocorrelation of the ensemble mean equals the per-chain one in expectation).

    A sampler that never changes any basin label (zero transitions) does not sample the slow
    mode on this budget: ESS is reported as 0 with ``mixing_frozen=1``. Otherwise ESS is the
    number of recorded configurations (N * T_eq) divided by the IAT. The escaped-fraction is
    kept only as a reported diagnostic (it saturates in high dimensions and is not used for
    the frozen decision or the IAT, which was a prior bug).
    """
    H = np.asarray(label_hist)
    if H.ndim == 1:
        H = H[:, None]
    stride = max(1, int(iat_stride))
    T, N = H.shape
    out = {"iat": float("nan"), "ess": 0.0, "n_transitions_total": 0,
           "transition_rate_per_step": 0.0, "transition_rate_per_time": 0.0,
           "mixing_frozen": 1, "escaped_fraction_final": 0.0}
    if T < 2 or N < 1:
        return out
    L0 = H[0]
    out["escaped_fraction_final"] = float(np.mean(H[-1] != L0))
    changes = int(np.sum(H[1:] != H[:-1]))
    denom = N * (T - 1) * stride
    out["n_transitions_total"] = changes
    out["transition_rate_per_step"] = float(changes / denom) if denom else 0.0
    if dt and dt > 0:
        out["transition_rate_per_time"] = float(out["transition_rate_per_step"] / dt)
    if changes == 0:
        # slow mode never changes basin on this budget: not sampled
        out["iat"] = float(T * stride)
        out["ess"] = 0.0
        out["mixing_frozen"] = 1
        return out
    cv = np.asarray(cv_series, dtype=float).reshape(-1)
    b = int(burn_frac * cv.size)
    series = cv[b:]
    if series.size < 4 or float(np.var(series)) < 1e-12:
        # transitions occur but the ensemble CV mean is flat (fast, symmetric mixing):
        # treat recorded configurations as effectively decorrelated
        out["iat"] = float(stride)
        out["ess"] = float(N * max(series.size, 1))
        out["mixing_frozen"] = 0
        return out
    iat_rec = integrated_autocorrelation_time(series)
    out["iat"] = float(iat_rec * stride)
    out["ess"] = float(N * series.size / max(iat_rec, 1.0))
    out["mixing_frozen"] = 0
    return out


def compute_metric_bundle(target, X: torch.Tensor, ref: torch.Tensor, label_hist, cv_series, diag: dict,
                          runtime_sec: float, iat_stride: int = 1, dt: float = 0.0) -> dict:
    Xf = X[torch.isfinite(X).all(dim=-1)]
    out = {"n_samples": int(X.shape[0]), "n_finite": int(Xf.shape[0]),
           "nonfinite_count": int(X.numel() - Xf.numel()), "runtime_sec": float(runtime_sec)}
    if Xf.shape[0] == 0:
        return out
    labels = target.basin_labels(Xf)
    target_probs = target.target_basin_probs(device=X.device)
    out.update(basin_population_metrics(labels, target_probs))
    if target.name.startswith("manywell"):
        out.update(manywell_metrics(target, Xf))
    out["free_energy_rmse"] = free_energy_rmse_2d(target, Xf)
    out.update(observable_bias_metrics(target, Xf, ref))
    out.update(mixing_metrics(label_hist, cv_series, iat_stride=iat_stride, dt=dt))
    # 1D target-fidelity distances (double well, triple well)
    if int(getattr(target, "dim", 0)) == 1:
        out["cdf_sup_error"] = cdf_sup_error(Xf[:, 0].detach().cpu().numpy(), ref[:, 0].detach().cpu().numpy())
        out["density_l1_error"] = density_l1_error(Xf[:, 0].detach().cpu().numpy(), ref[:, 0].detach().cpu().numpy())
    # Time-resolved TV relaxation and all-basin coverage from the per-chain label history
    if target_probs is not None:
        H = np.asarray(label_hist)
        stride = max(1, int(iat_stride))
        times = np.arange(H.shape[0]) * stride * float(dt)
        tv = basin_tv_series(H, target_probs.detach().cpu().numpy())
        out["basin_tv_final"] = float(tv[-1]) if tv.size else float("nan")
        out["threshold_time_tv"] = threshold_time(tv, times, tau=0.1)
        cov, cov_frac = first_all_basin_coverage_time(H, times, int(target_probs.numel()))
        out["coverage_time_all_basins"] = cov
        out["coverage_fraction"] = cov_frac
    ess = float(out["ess"])
    out["ess_per_sec"] = ess / max(float(runtime_sec), 1e-12)
    for k, v in diag.items():
        out[k] = v
    out["ess_per_gradient_eval"] = ess / max(float(out.get("grad_evals", 0) or 0), 1.0)
    out["ess_per_potential_eval"] = ess / max(float(out.get("pot_evals", 0) or 0), 1.0)
    out["ess_per_levy_quadrature_eval"] = ess / max(float(out.get("levy_quadrature_evals", 0) or 0), 1.0)
    return out
