#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from high_dim_output.benchmark_metrics import (
    BENCHMARK_METHODS,
    clip_samples_to_box,
    compute_benchmark_metrics,
    get_default_benchmark_config,
    init_benchmark_history,
    make_metric_rng,
    plot_benchmark_metrics_figure,
    sample_from_2d_grid_density,
    save_benchmark_metadata_json,
    save_benchmark_metrics_csv,
)
from flmc_utils import step_flmc_2d

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Example: Ring Potential
#   V(x,y) = (1 - x^2 - y^2)^2 + y^2/(x^2+y^2)
# ============================================================

R2_EPS = 1e-12
EXPO_CLIP = 60.0
S_CLIP = 40.0
POT_CLIP = 8.0
STATE_CLIP = 64.0
STATE_CLIP_BOUNDS = ((-STATE_CLIP, STATE_CLIP), (-STATE_CLIP, STATE_CLIP))
RING_MODE_DESCRIPTORS = np.array(
    [
        [-1.0, 0.0],
        [1.0, 0.0],
    ],
    dtype=float,
)

# ---------------- Potential & grad ----------------

def _clip_xy_for_eval(x, y, bound=POT_CLIP):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=bound, neginf=-bound)
    y = np.nan_to_num(y, nan=0.0, posinf=bound, neginf=-bound)
    return np.clip(x, -bound, bound), np.clip(y, -bound, bound)


def sanitize_state(X):
    """
    Catastrophic numerical-safety guard for rare outlier states.

    This clipping is not part of the paper-faithful FLMC formula; it only keeps
    the benchmark numerically finite when a sampler produces extreme excursions.
    """
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=STATE_CLIP, neginf=-STATE_CLIP)
    return np.clip(X, -STATE_CLIP, STATE_CLIP)

def V_ring(x, y):
    x, y = _clip_xy_for_eval(x, y)
    r2 = x * x + y * y
    r2s = r2 + R2_EPS
    return (1.0 - r2) ** 2 + (y * y) / r2s

def gradV_ring(x, y):
    x, y = _clip_xy_for_eval(x, y)
    r2 = x * x + y * y
    r2s = r2 + R2_EPS
    r4s = r2s * r2s
    dVx = 4.0 * (r2 - 1.0) * x - 2.0 * x * (y * y) / r4s
    dVy = 4.0 * (r2 - 1.0) * y + 2.0 * y * (x * x) / r4s
    return dVx, dVy

# ---------------- Utilities ----------------

def bilinear_interp(x, y, gx, gy, F):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=gx[-1], neginf=gx[0])
    y = np.nan_to_num(y, nan=0.0, posinf=gy[-1], neginf=gy[0])
    x = np.clip(x, gx[0], gx[-1])
    y = np.clip(y, gy[0], gy[-1])
    dx = gx[1] - gx[0]
    dy = gy[1] - gy[0]
    
    ix = np.clip(np.searchsorted(gx, x, side="right") - 1, 0, gx.size - 2)
    iy = np.clip(np.searchsorted(gy, y, side="right") - 1, 0, gy.size - 2)
    
    x1, y1 = gx[ix], gy[iy]
    wx = (x - x1) / (dx + 1e-12)
    wy = (y - y1) / (dy + 1e-12)
    
    return (1-wx)*(1-wy)*F[iy, ix] + wx*(1-wy)*F[iy, ix+1] + \
           (1-wx)*wy*F[iy+1, ix] + wx*wy*F[iy+1, ix+1]

def sample_from_pi_grid(rng, pi, gx, gy, n_samples):
    ny, nx = pi.shape
    dx, dy = gx[1]-gx[0], gy[1]-gy[0]
    w = (pi * dx * dy).ravel()
    w /= w.sum()
    idx = rng.choice(w.size, size=n_samples, p=w)
    iy, ix = np.unravel_index(idx, pi.shape)
    return np.stack([gx[ix] + (rng.random(n_samples)-0.5)*dx, 
                     gy[iy] + (rng.random(n_samples)-0.5)*dy], axis=1)

# ---------------- Density Estimation & Errors ----------------

def gaussian_kernel_1d(sigma, dx, truncate=4.0):
    m = int(np.ceil(truncate * sigma / dx))
    xs = np.arange(-m, m + 1) * dx
    ker = np.exp(-0.5 * (xs / sigma) ** 2)
    return ker / (np.sum(ker) * dx + 1e-300)

def smooth2d_separable(P, ker_x, ker_y):
    tmp = np.array([np.convolve(row, ker_x, mode="same") for row in P])
    return np.array([np.convolve(tmp[:, j], ker_y, mode="same") for j in range(P.shape[1])]).T

def density_on_grid(samples, gx, gy, do_smooth=True, smoothing_sigma=0.08):
    samples = sanitize_state(samples)
    dx, dy = gx[1] - gx[0], gy[1] - gy[0]
    bins_x = np.concatenate([gx - dx/2, [gx[-1] + dx/2]])
    bins_y = np.concatenate([gy - dy/2, [gy[-1] + dy/2]])
    H, _, _ = np.histogram2d(samples[:,1], samples[:,0], bins=[bins_y, bins_x])
    P = H / (samples.shape[0] * dx * dy + 1e-12)
    if do_smooth:
        ker = gaussian_kernel_1d(smoothing_sigma, dx)
        P = smooth2d_separable(P, ker, ker)
    return np.maximum(P, 1e-300) / (P.sum() * dx * dy)

def compute_grid_errors(dens, pi, dx, dy):
    diff = np.abs(dens - pi)
    l1_err = np.sum(diff) * dx * dy
    l2_err = np.sqrt(np.sum(diff**2) * dx * dy)
    return diff, l1_err, l2_err

# ---------------- Precompute Fields ----------------

def precompute_pi_drift_score_on_grid(eps, gx, gy, lam, sigma_L, multipliers, pm):
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    V = V_ring(X, Y)
    logpi = -2.0 * V / eps
    logpi -= np.max(logpi)
    pi = np.exp(np.clip(logpi, -EXPO_CLIP, 0.0))
    pi /= (np.sum(pi) * (gx[1]-gx[0]) * (gy[1]-gy[0]) + 1e-300)

    dVx, dVy = gradV_ring(X, Y)
    bx, by = -dVx, -dVy

    Sx, Sy = np.zeros_like(pi), np.zeros_like(pi)
    thetas = np.linspace(0.05, 1.0, 17)
    ang = np.linspace(0, 2*np.pi, 12, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    
    for k, pk in enumerate(pm):
        z_mag = sigma_L * multipliers[k]
        for u in dirs:
            zx, zy = z_mag * u[0], z_mag * u[1]
            accx, accy = 0, 0
            for th in thetas:
                dlog_m = -2.0*(V_ring(X-th*zx, Y-th*zy)-V)/eps
                dlog_p = -2.0*(V_ring(X+th*zx, Y+th*zy)-V)/eps
                rm = np.exp(np.clip(dlog_m, -EXPO_CLIP, EXPO_CLIP))
                rp = np.exp(np.clip(dlog_p, -EXPO_CLIP, EXPO_CLIP))
                term = 0.5 * (rm - rp)
                accx += term * zx
                accy += term * zy
            Sx += (pk/len(dirs)) * (accx/len(thetas))
            Sy += (pk/len(dirs)) * (accy/len(thetas))
            
    return pi, bx, by, lam*np.clip(Sx, -S_CLIP, S_CLIP), lam*np.clip(Sy, -S_CLIP, S_CLIP)

# ---------------- Simulation Steps ----------------

def step_diff(X, dt, eps, gx, gy, bx_g, by_g, rng):
    X = sanitize_state(X)
    bx = bilinear_interp(X[:,0], X[:,1], gx, gy, bx_g)
    by = bilinear_interp(X[:,0], X[:,1], gx, gy, by_g)
    norm = np.sqrt(bx**2 + by**2) + 1e-8
    drift = dt * np.stack([bx, by], axis=1) / (1.0 + dt*norm)[:,None]
    X_new = X + drift + np.sqrt(eps*dt) * rng.standard_normal(X.shape)
    return sanitize_state(X_new)

def step_levy(X, dt, eps, gx, gy, bx_g, by_g, Sx_g, Sy_g, rng, lam, sigma_L, mults, pm):
    X = sanitize_state(X)
    bx = bilinear_interp(X[:,0], X[:,1], gx, gy, bx_g)
    by = bilinear_interp(X[:,0], X[:,1], gx, gy, by_g)
    sx = bilinear_interp(X[:,0], X[:,1], gx, gy, Sx_g)
    sy = bilinear_interp(X[:,0], X[:,1], gx, gy, Sy_g)
    
    drift_x, drift_y = bx - sx, by - sy
    norm = np.sqrt(drift_x**2 + drift_y**2) + 1e-8
    X_new = X + dt * np.stack([drift_x, drift_y], axis=1) / (1.0 + dt*norm)[:,None]
    X_new += np.sqrt(eps*dt) * rng.standard_normal(X.shape)
    
    n_jumps = rng.poisson(lam * dt, size=X.shape[0])
    idx = np.where(n_jumps > 0)[0]
    if len(idx) > 0:
        for i in idx:
            k = n_jumps[i]
            m_choice = rng.choice(mults, size=k, p=pm)
            ang = rng.random(k) * 2 * np.pi
            jx = np.sum(sigma_L * m_choice * np.cos(ang))
            jy = np.sum(sigma_L * m_choice * np.sin(ang))
            mag = np.sqrt(jx**2 + jy**2) + 1e-8
            scale = np.minimum(1.0, 3.2/mag) # jump cap
            X_new[i] += np.array([jx*scale, jy*scale])
    return sanitize_state(X_new)

def logpi_ring_xy(x, y, eps):
    return -2.0 * V_ring(x, y) / eps

def grad_logpi_ring_xy(x, y, eps):
    dVx, dVy = gradV_ring(x, y)
    scale = -2.0 / eps
    return scale * dVx, scale * dVy


def gradU_ring_xy(x, y, eps):
    """
    Paper-faithful FLMC mapping for the ring benchmark.

    Target:
        pi(x, y) ∝ exp(-2 V(x, y) / eps)
    Therefore:
        U(x, y) = -log pi(x, y) = 2 V(x, y) / eps + const
        gradU(x, y) = (2 / eps) gradV(x, y)
    and the FLMC drift is
        -c_alpha * gradU(x, y) = -(2 c_alpha / eps) gradV(x, y).
    """
    gx, gy = grad_logpi_ring_xy(x, y, eps)
    return -gx, -gy

def step_mala(X, dt, eps, rng):
    X = sanitize_state(X)
    x = X[:, 0]
    y = X[:, 1]
    gx, gy = grad_logpi_ring_xy(x, y, eps)
    grad = np.stack([gx, gy], axis=1)

    mean = X + 0.5 * dt * grad
    proposal = sanitize_state(mean + np.sqrt(dt) * rng.standard_normal(X.shape))

    logp_x = logpi_ring_xy(x, y, eps)
    logp_y = logpi_ring_xy(proposal[:, 0], proposal[:, 1], eps)

    gyx, gyy = grad_logpi_ring_xy(proposal[:, 0], proposal[:, 1], eps)
    grad_y = np.stack([gyx, gyy], axis=1)
    mean_y = proposal + 0.5 * dt * grad_y

    log_q_y_given_x = -np.sum((proposal - mean) ** 2, axis=1) / (2.0 * dt)
    log_q_x_given_y = -np.sum((X - mean_y) ** 2, axis=1) / (2.0 * dt)
    log_alpha = (logp_y + log_q_x_given_y) - (logp_x + log_q_y_given_x)

    accept = np.log(rng.random(X.shape[0])) < log_alpha
    X_new = X.copy()
    X_new[accept] = proposal[accept]
    return sanitize_state(X_new), float(accept.mean())

def step_malevy(X, dt, eps, rng, lam, sigma_L, mults, pm, jump_cap=3.2):
    X_mid, _ = step_mala(X, dt, eps, rng)
    proposal = X_mid.copy()
    n_jumps = rng.poisson(lam * dt, size=X.shape[0])
    idx = np.where(n_jumps > 0)[0]
    if len(idx) > 0:
        for i in idx:
            k = int(n_jumps[i])
            m_choice = rng.choice(mults, size=k, p=pm)
            ang = rng.random(k) * 2 * np.pi
            jx = np.sum(sigma_L * m_choice * np.cos(ang))
            jy = np.sum(sigma_L * m_choice * np.sin(ang))
            mag = np.sqrt(jx**2 + jy**2) + 1e-8
            scale = np.minimum(1.0, jump_cap / mag)
            proposal[i] += np.array([jx * scale, jy * scale])

    logp_x = logpi_ring_xy(X_mid[:, 0], X_mid[:, 1], eps)
    logp_y = logpi_ring_xy(proposal[:, 0], proposal[:, 1], eps)
    log_alpha = logp_y - logp_x
    accept = np.log(rng.random(X.shape[0])) < log_alpha
    X_new = X_mid.copy()
    X_new[accept] = proposal[accept]
    return X_new, float(accept.mean())
# ---------------- Plotting Functions ----------------

def save_figure_both(fig, out_base, dpi=200):
    fig.savefig(f"{out_base}.png", dpi=dpi)
    fig.savefig(f"{out_base}.pdf")
    plt.close(fig)


def shared_density_norm(arrs, use_log=False, gamma=0.5):
    vmax = max(float(np.max(a)) for a in arrs)
    if use_log:
        min_pos = min(float(np.min(a[a > 0])) for a in arrs)
        vmin = max(min_pos, vmax * 1e-6)
        return mcolors.LogNorm(vmin=vmin, vmax=vmax)
    return mcolors.PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)


def plot_metrics_figure(t, mean, std, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    methods = [
        ("l1_d", "ULA", "C0"),
        ("l1_m", "MALA", "C2"),
        ("l1_flmc", "FLMC", "tab:orange"),
        ("l1_l", "LSBMC", "C3"),
    ]
    for key, label, color in methods:
        axes[0].errorbar(t, mean[key], yerr=std[key], fmt="-", color=color, label=label, alpha=0.9, capsize=2)
    axes[0].set_title("L1 Error")
    axes[0].set_xlabel("Time")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    methods = [
        ("l2_d", "ULA", "C0"),
        ("l2_m", "MALA", "C2"),
        ("l2_flmc", "FLMC", "tab:orange"),
        ("l2_l", "LSBMC", "C3"),
    ]
    for key, label, color in methods:
        axes[1].errorbar(t, mean[key], yerr=std[key], fmt="-", color=color, label=label, alpha=0.9, capsize=2)
    axes[1].set_title("L2 Error")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Ring: Metrics Comparison")
    fig.tight_layout()
    save_figure_both(fig, os.path.join(out_dir, "metrics"))


def plot_density_figure(gx, gy, dens, title, slug, out_dir, norm, init_point=None):
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    extent = [gx[0], gx[-1], gy[0], gy[-1]]

    im = ax.imshow(dens, origin="lower", extent=extent, cmap="viridis", norm=norm, aspect="equal")
    if init_point is not None:
        ax.scatter(
            init_point[0],
            init_point[1],
            marker="*",
            s=120,
            c="red",
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Density")
    fig.tight_layout()
    save_figure_both(fig, os.path.join(out_dir, slug))

# ---------------- Main ----------------

def run_simulation(
    seed,
    eps,
    dt,
    T,
    N,
    gx,
    gy,
    dx,
    dy,
    lam,
    sigma_L,
    mults,
    pm,
    pi,
    bx,
    by,
    Sx,
    Sy,
    alpha=1.5,
    mala_dt=None,
    benchmark_ref_samples=None,
    benchmark_config=None,
    benchmark_mode_descriptors=None,
    return_benchmark_history=False,
):
    rng = np.random.default_rng(seed)

    X_diff = sanitize_state(np.array([-1.0, 0.0]) + 0.05 * rng.standard_normal((N, 2)))
    X_levy = X_diff.copy()
    X_flmc = X_diff.copy()
    X_mala = X_diff.copy()

    history = {
        't': [],
        'l1_d': [], 'l1_l': [], 'l1_flmc': [], 'l1_m': [],
        'l2_d': [], 'l2_l': [], 'l2_flmc': [], 'l2_m': []
    }
    benchmark_metric_names = [
        "benchmark_sinkhorn_ot_cost",
        "benchmark_sinkhorn_divergence",
        "benchmark_mmd_squared",
        "benchmark_mmd",
    ]
    if (
        benchmark_config is not None
        and benchmark_mode_descriptors is not None
        and bool(benchmark_config.get("emc_enabled", True))
    ):
        benchmark_metric_names.append("benchmark_emc")
    benchmark_history = None
    benchmark_eval_steps = set()
    benchmark_lower = np.array([gx[0], gy[0]], dtype=float)
    benchmark_upper = np.array([gx[-1], gy[-1]], dtype=float)
    if mala_dt is None:
        mala_dt = dt
    acc_sum = 0.0
    acc_count = 0

    steps = int(T/dt)
    check = max(1, steps // 20)
    if benchmark_ref_samples is not None and benchmark_config is not None:
        benchmark_history = init_benchmark_history(benchmark_metric_names)
        benchmark_eval_steps = build_benchmark_eval_steps(
            steps,
            benchmark_config.get("benchmark_num_checkpoints", 7),
        )

    for i in range(steps+1):
        if i % check == 0 or i == steps:
            t = i * dt
            history['t'].append(t)

            dd = density_on_grid(X_diff, gx, gy)
            dl = density_on_grid(X_levy, gx, gy)
            dflmc = density_on_grid(X_flmc, gx, gy)
            dm = density_on_grid(X_mala, gx, gy)
            _, l1d, l2d = compute_grid_errors(dd, pi, dx, dy)
            _, l1l, l2l = compute_grid_errors(dl, pi, dx, dy)
            _, l1flmc, l2flmc = compute_grid_errors(dflmc, pi, dx, dy)
            _, l1m, l2m = compute_grid_errors(dm, pi, dx, dy)

            history['l1_d'].append(l1d); history['l1_l'].append(l1l); history['l1_flmc'].append(l1flmc)
            history['l2_d'].append(l2d); history['l2_l'].append(l2l); history['l2_flmc'].append(l2flmc)
            history['l1_m'].append(l1m); history['l2_m'].append(l2m)

        if benchmark_history is not None and i in benchmark_eval_steps:
            t = i * dt
            benchmark_history["t"].append(t)
            method_samples = {
                "ula": clip_samples_to_box(X_diff, benchmark_lower, benchmark_upper),
                "mala": clip_samples_to_box(X_mala, benchmark_lower, benchmark_upper),
                "flmc": clip_samples_to_box(X_flmc, benchmark_lower, benchmark_upper),
                "lsbmc": clip_samples_to_box(X_levy, benchmark_lower, benchmark_upper),
            }
            for method_idx, (slug, _, _) in enumerate(BENCHMARK_METHODS):
                metric_rng = make_metric_rng(benchmark_config["metric_seed"], seed, i, method_idx)
                values = compute_benchmark_metrics(
                    method_samples[slug],
                    benchmark_ref_samples,
                    benchmark_config,
                    metric_rng,
                    metric_prefix="benchmark",
                    mode_descriptors=benchmark_mode_descriptors,
                    context=f"ring seed={seed} t={t:.4f} method={slug}",
                )
                for metric_name in benchmark_metric_names:
                    benchmark_history[f"{metric_name}_{slug}"].append(
                        values.get(metric_name, float("nan"))
                    )

        X_diff = step_diff(X_diff, dt, eps, gx, gy, bx, by, rng)
        X_levy = step_levy(X_levy, dt, eps, gx, gy, bx, by, Sx, Sy, rng, lam, sigma_L, mults, pm)
        X_flmc = step_flmc_2d(
            sanitize_state(X_flmc),
            dt,
            alpha,
            lambda x, y: gradU_ring_xy(x, y, eps),
            rng,
            clip_bounds=STATE_CLIP_BOUNDS,
        )
        X_flmc = sanitize_state(X_flmc)
        X_mala, acc = step_mala(X_mala, mala_dt, eps, rng)
        acc_sum += acc
        acc_count += 1

    acc_rate = acc_sum / max(acc_count, 1)
    if return_benchmark_history:
        return history, benchmark_history, X_diff, X_levy, X_flmc, X_mala, acc_rate
    return history, X_diff, X_levy, X_flmc, X_mala, acc_rate

def aggregate_histories(histories, keys):
    stacked = {k: np.stack([np.array(h[k]) for h in histories], axis=0) for k in keys}
    mean = {k: stacked[k].mean(axis=0) for k in keys}
    std = {k: stacked[k].std(axis=0) for k in keys}
    return mean, std


def build_benchmark_eval_steps(steps, num_checkpoints):
    num_checkpoints = max(2, int(num_checkpoints))
    return set(np.unique(np.linspace(0, steps, num_checkpoints, dtype=int)).tolist())

def main():
    eps, dt, T, N = 0.35, 0.0015, 40.0, 5000
    gx, gy = np.linspace(-2.2, 2.2, 240), np.linspace(-2.2, 2.2, 240)
    dx, dy = gx[1]-gx[0], gy[1]-gy[0]
    lam, sigma_L = 1.6, 1.25
    mults, pm = [1.0, 1.7, 2.4], [0.70, 0.22, 0.08]
    num_seeds = 5
    seeds = list(range(num_seeds))

    print("Precomputing fields...")
    pi, bx, by, Sx, Sy = precompute_pi_drift_score_on_grid(eps, gx, gy, lam, sigma_L, mults, pm)

    alpha = 1.5
    benchmark_config = get_default_benchmark_config({
        "benchmark_num_checkpoints": 21,
        "sinkhorn_method": "sinkhorn_stabilized",
        "sinkhorn_epsilon": 0.25,
        "sinkhorn_max_iter": 300,
        "sinkhorn_tol": 1e-5,
        "sinkhorn_subsample_size": 128,
        "mmd_subsample_size": 192,
        "emc_tau": 0.2,
        "emc_enabled": True,
    })
    benchmark_ref_size = max(
        512,
        int(benchmark_config["sinkhorn_subsample_size"]),
        int(benchmark_config["mmd_subsample_size"]),
    )
    rng_benchmark = np.random.default_rng(benchmark_config["metric_seed"])
    benchmark_ref = sample_from_2d_grid_density(
        rng_benchmark,
        pi,
        gx,
        gy,
        benchmark_ref_size,
    )

    print(f"Simulating {num_seeds} seeds...")
    histories = []
    benchmark_histories = []
    first_final = None
    acc_rates = []
    for seed in seeds:
        history, benchmark_history, X_diff, X_levy, X_flmc, X_mala, acc_rate = run_simulation(
            seed,
            eps,
            dt,
            T,
            N,
            gx,
            gy,
            dx,
            dy,
            lam,
            sigma_L,
            mults,
            pm,
            pi,
            bx,
            by,
            Sx,
            Sy,
            alpha=alpha,
            benchmark_ref_samples=benchmark_ref,
            benchmark_config=benchmark_config,
            benchmark_mode_descriptors=RING_MODE_DESCRIPTORS,
            return_benchmark_history=True,
        )
        histories.append(history)
        benchmark_histories.append(benchmark_history)
        acc_rates.append(acc_rate)
        if first_final is None:
            first_final = (X_diff, X_levy, X_flmc, X_mala)

    t = np.array(histories[0]['t'])
    keys = [
        'l1_d', 'l1_l', 'l1_flmc', 'l1_m',
        'l2_d', 'l2_l', 'l2_flmc', 'l2_m'
    ]
    mean, std = aggregate_histories(histories, keys)
    benchmark_metric_names = [
        "benchmark_sinkhorn_ot_cost",
        "benchmark_sinkhorn_divergence",
        "benchmark_mmd_squared",
        "benchmark_mmd",
    ]
    if bool(benchmark_config.get("emc_enabled", True)):
        benchmark_metric_names.append("benchmark_emc")
    benchmark_keys = [
        f"{metric_name}_{slug}"
        for metric_name in benchmark_metric_names
        for slug, _, _ in BENCHMARK_METHODS
    ]
    benchmark_mean, benchmark_std = aggregate_histories(benchmark_histories, benchmark_keys)
    benchmark_t = np.array(benchmark_histories[0]["t"])

    out_dir = os.path.join(THIS_DIR, "ring_output")
    os.makedirs(out_dir, exist_ok=True)
    plot_metrics_figure(t, mean, std, out_dir)
    benchmark_base = os.path.join(out_dir, "benchmark_metrics_ring")
    plot_benchmark_metrics_figure(
        benchmark_t,
        benchmark_mean,
        benchmark_std,
        ["benchmark_sinkhorn_divergence", "benchmark_mmd", "benchmark_emc"],
        benchmark_base,
        "Ring: Benchmark Metrics",
        metric_labels={
            "benchmark_sinkhorn_divergence": "Sinkhorn",
            "benchmark_mmd": "MMD",
            "benchmark_emc": "EMC",
        },
    )
    save_benchmark_metrics_csv(
        f"{benchmark_base}.csv",
        benchmark_t,
        benchmark_mean,
        benchmark_std,
        benchmark_metric_names,
    )
    save_benchmark_metadata_json(
        os.path.join(out_dir, "metrics_benchmark_ring.json"),
        "ring",
        benchmark_config,
        benchmark_metric_names,
        mode_descriptors=RING_MODE_DESCRIPTORS,
        extra_metadata={
            "benchmark_reference_size": int(benchmark_ref.shape[0]),
            "benchmark_num_checkpoints": int(benchmark_t.size),
            "emc_applicable": True,
            "emc_descriptor_description": "Two dominant minima of the ring potential at (-1, 0) and (1, 0).",
        },
    )

    X_diff, X_levy, X_flmc, X_mala = first_final
    dens_d = density_on_grid(X_diff, gx, gy)
    dens_l = density_on_grid(X_levy, gx, gy)
    dens_flmc = density_on_grid(X_flmc, gx, gy)
    dens_m = density_on_grid(X_mala, gx, gy)
    norm = shared_density_norm([pi, dens_d, dens_l, dens_flmc, dens_m], use_log=False, gamma=0.5)
    init_point = np.array([-1.0, 0.0])
    plot_density_figure(gx, gy, pi, "Ring: True Density", "true_density", out_dir, norm)
    plot_density_figure(gx, gy, dens_d, "Ring: ULA Density", "ula_density", out_dir, norm, init_point=init_point)
    plot_density_figure(gx, gy, dens_m, "Ring: MALA Density", "mala_density", out_dir, norm, init_point=init_point)
    plot_density_figure(gx, gy, dens_flmc, "Ring: FLMC Density", "flmc_density", out_dir, norm, init_point=init_point)
    plot_density_figure(gx, gy, dens_l, "Ring: LSBMC Density", "lsbmc_density", out_dir, norm, init_point=init_point)
    
    avg_acc = float(np.mean(acc_rates)) if acc_rates else 0.0
    print(
        "Done. Saved metrics and density figures to:\n"
        f"{out_dir}"
    )
    print(f"Saved benchmark metrics to: {benchmark_base}.png/.pdf/.csv")
    print(f"MALA mean acceptance rate: {avg_acc:.3f}")

if __name__ == "__main__":
    main()
