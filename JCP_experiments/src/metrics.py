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
# ==================================================== chemistry-native metrics
def free_energy_profile(cv: torch.Tensor, edges: torch.Tensor, beta: float,
                        smooth: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Free-energy profile F(xi) = -beta^{-1} log p(xi) along a collective
    variable, on the given (uniform) bin edges, Laplace-smoothed (so empty bins
    read a large but finite F). F is shifted so min F = 0 (free energy is defined
    up to a constant). Returns (F on bin centres, normalised bin probability p).
    """
    idx = torch.bucketize(cv.reshape(-1), edges[1:-1])
    counts = torch.bincount(idx, minlength=edges.shape[0] - 1).to(torch.float64)
    p = (counts + smooth)
    p = p / p.sum()
    F = -(1.0 / beta) * torch.log(p)
    return F - F.min(), p


def free_energy_profile_error(cv: torch.Tensor, edges: torch.Tensor, beta: float,
                              ref_F: torch.Tensor, ref_p: torch.Tensor,
                              pi_min: float) -> float:
    """sup_xi |F_hat(xi) - F*(xi)| in units of k_B T = 1/beta, restricted to
    bins where the reference probability ref_p >= pi_min (else the sup is
    dominated by empty-bin noise). Local methods missing a well plateau at that
    well's free-energy deficit -- devastating on log-y."""
    F, _ = free_energy_profile(cv, edges, beta)
    mask = ref_p >= pi_min
    if not bool(mask.any()):
        return 0.0
    return float((F[mask] - ref_F[mask]).abs().max().item())


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
    """Overlap int min(rho_hat_E, rho*_E) dE in [0,1] (1 = identical). ref_hist
    is the frozen reference energy histogram (normalised, same edges)."""
    idx = torch.bucketize(E.reshape(-1), edges[1:-1])
    h = torch.bincount(idx, minlength=edges.shape[0] - 1).to(torch.float64)
    h = h / h.sum()
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
    x = x - x.mean()
    n = x.size
    if np.allclose(x, 0.0):
        return np.ones(1)
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n].real
    return acf / acf[0]


def iat_1d(x: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time, Sokal automatic windowing:
    tau(M) = 1 + 2 sum_{k=1}^M rho_k, window at smallest M with M >= c*tau(M)."""
    rho = _acf_1d(np.asarray(x, dtype=float))
    tau = 1.0 + 2.0 * np.cumsum(rho[1:])
    m = np.arange(1, tau.size + 1)
    win = np.where(m >= c * tau)[0]
    idx = win[0] if win.size else tau.size - 1
    return float(max(tau[idx], 1.0))


def ess_from_series(series: np.ndarray) -> float:
    """ESS = (total samples) / mean IAT over chains. series: (n_chains, n_draws)."""
    series = np.atleast_2d(series)
    taus = [iat_1d(series[i]) for i in range(series.shape[0])]
    return float(series.size / np.mean(taus))


def split_rhat(chains: np.ndarray) -> float:
    """Rank-normalized split-R-hat (Vehtari et al. 2021). chains: (M, N)."""
    chains = np.atleast_2d(np.asarray(chains, dtype=float))
    M, N = chains.shape
    h = N // 2
    if h < 2:
        return float("nan")
    split = np.concatenate([chains[:, :h], chains[:, h:2 * h]], axis=0)  # (2M, h)
    flat = split.reshape(-1)
    order = flat.argsort(kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, flat.size + 1)
    z = _norm_ppf((ranks - 3.0 / 8.0) / (flat.size - 0.25)).reshape(split.shape)
    m = z.mean(axis=1)
    B = h * m.var(ddof=1)
    W = z.var(axis=1, ddof=1).mean()
    if W <= 0:
        return float("nan")
    var_plus = (h - 1.0) / h * W + B / h
    return float(np.sqrt(var_plus / W))


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
    """Mean number of home->far->home round trips per chain. labels_t: (T, chains)
    of basin indices."""
    labels_t = np.atleast_2d(labels_t.T).T if labels_t.ndim == 1 else labels_t
    total = 0
    C = labels_t.shape[1]
    for c in range(C):
        seq = labels_t[:, c]
        state = "home"; trips = 0
        for lab in seq:
            if state == "home" and lab == far:
                state = "far"
            elif state == "far" and lab == home:
                state = "home"; trips += 1
        total += trips
    return total / max(C, 1)


def committed_mfpt(labels_t: np.ndarray, home: int, far: int, dt: float,
                   steps_per_frame: int) -> float:
    """Mean first-passage time home->far (frames -> physical time), censored at
    the end. labels_t: (T, chains)."""
    labels_t = labels_t if labels_t.ndim == 2 else labels_t[:, None]
    T, C = labels_t.shape
    times, n_exit = 0.0, 0
    horizon = T * steps_per_frame * dt
    for c in range(C):
        hit = np.where(labels_t[:, c] == far)[0]
        if hit.size:
            times += (hit[0] + 1) * steps_per_frame * dt; n_exit += 1
        else:
            times += horizon
    return times / n_exit if n_exit else float("inf")


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
