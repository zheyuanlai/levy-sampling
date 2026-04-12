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
# 1. 4-Well Müller-Brown Potential (Scaled)
# ============================================================

EXPO_CLIP = 60.0
LOGR_CLIP = 30.0
S_CLIP = 80.0
SCALE_V = 0.05
MUELLER_MODE_DESCRIPTORS = np.array(
    [
        [0.01930266, 0.47927555],
        [0.96190327, 0.01893948],
        [-0.79793766, -0.49426091],
        [-0.50016830, 1.49979865],
    ],
    dtype=float,
)

MUELLER_PARAMS = np.array(
    [
        [-200, -1, 0, -10, 1, 0],
        [-200, -1, 0, -10, 0, 0.5],
        [-200, -6.5, 11, -6.5, -0.5, 1.5],
        [-200, -3, 0, -3, -0.8, -0.5],
    ]
)


def V_mueller(x, y):
    x, y = np.asarray(x), np.asarray(y)
    V = np.zeros_like(x)
    for p in MUELLER_PARAMS:
        A, a, b, c, x0, y0 = p
        V += (A * SCALE_V) * np.exp(a * (x - x0) ** 2 + b * (x - x0) * (y - y0) + c * (y - y0) ** 2)
    return V


def gradV_mueller(x, y):
    x, y = np.asarray(x), np.asarray(y)
    dVx, dVy = np.zeros_like(x), np.zeros_like(x)
    for p in MUELLER_PARAMS:
        A, a, b, c, x0, y0 = p
        dx, dy = x - x0, y - y0
        arg = a * dx**2 + b * dx * dy + c * dy**2
        exp_term = (A * SCALE_V) * np.exp(arg)
        dVx += exp_term * (2 * a * dx + b * dy)
        dVy += exp_term * (b * dx + 2 * c * dy)
    return dVx, dVy


def V_mueller_flmc(x, y):
    # Manuscript uses the rescaled convention b = -(1/2) grad V for this example.
    return 0.5 * V_mueller(x, y)


def gradV_mueller_flmc(x, y):
    dVx, dVy = gradV_mueller(x, y)
    return 0.5 * dVx, 0.5 * dVy


# ============================================================
# 2. Utilities & Density Estimation
# ============================================================

def bilinear_interp(x, y, gx, gy, F):
    x = np.clip(x, gx[0], gx[-1])
    y = np.clip(y, gy[0], gy[-1])
    dx = gx[1] - gx[0]
    dy = gy[1] - gy[0]
    ix = np.clip(np.searchsorted(gx, x, side="right") - 1, 0, gx.size - 2)
    iy = np.clip(np.searchsorted(gy, y, side="right") - 1, 0, gy.size - 2)
    x1, y1 = gx[ix], gy[iy]
    wx = (x - x1) / (dx + 1e-12)
    wy = (y - y1) / (dy + 1e-12)
    f11, f21 = F[iy, ix], F[iy, ix + 1]
    f12, f22 = F[iy + 1, ix], F[iy + 1, ix + 1]
    return (1 - wx) * (1 - wy) * f11 + wx * (1 - wy) * f21 + (1 - wx) * wy * f12 + wx * wy * f22


def gaussian_kernel_1d(sigma, dx, truncate=4.0):
    m = int(np.ceil(truncate * sigma / dx))
    xs = np.arange(-m, m + 1) * dx
    ker = np.exp(-0.5 * (xs / sigma) ** 2)
    return ker / (np.sum(ker) * dx + 1e-300)


def smooth2d_separable(P, ker_x, ker_y):
    tmp = np.array([np.convolve(row, ker_x, mode="same") for row in P])
    out = np.array([np.convolve(tmp[:, j], ker_y, mode="same") for j in range(P.shape[1])]).T
    return out


def density_on_grid(samples, gx, gy, do_smooth=True):
    dx = gx[1] - gx[0]
    dy = gy[1] - gy[0]
    bins_x = np.concatenate([gx - dx / 2, [gx[-1] + dx / 2]])
    bins_y = np.concatenate([gy - dy / 2, [gy[-1] + dy / 2]])
    H, _, _ = np.histogram2d(samples[:, 1], samples[:, 0], bins=[bins_y, bins_x])
    P = H / (samples.shape[0] * dx * dy + 1e-12)
    if do_smooth:
        ker = gaussian_kernel_1d(0.08, dx)
        P = smooth2d_separable(P, ker, ker)
    P = np.maximum(P, 1e-300)
    return P / (P.sum() * dx * dy)


# ============================================================
# 3. Error Metrics
# ============================================================

def compute_errors(dens, pi, dx, dy):
    abs_err_map = np.abs(dens - pi)
    l1_err = np.sum(abs_err_map) * dx * dy
    l2_err = np.sqrt(np.sum((dens - pi) ** 2) * dx * dy)
    return abs_err_map, l1_err, l2_err


# ============================================================
# 4. Simulation Kernels
# ============================================================

def precompute_pi_b_S(eps, gx, gy, lam, sigma_L, multipliers, pm):
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    V = V_mueller(X, Y)
    logpi = -(V - np.min(V)) / (eps**2)
    pi_unn = np.exp(np.clip(logpi, -EXPO_CLIP, 0))
    pi = pi_unn / (np.sum(pi_unn) * (gx[1] - gx[0]) * (gy[1] - gy[0]))

    dVx, dVy = gradV_mueller(X, Y)
    bx, by = -0.5 * dVx, -0.5 * dVy

    Sx, Sy = np.zeros_like(pi), np.zeros_like(pi)
    thetas = np.linspace(0.05, 1.0, 20)
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)

    for k, prob_k in enumerate(pm):
        jump_mag = sigma_L * multipliers[k]
        for u in dirs:
            dz_x, dz_y = jump_mag * u[0], jump_mag * u[1]
            t_acc_x, t_acc_y = 0, 0
            for theta in thetas:
                Vp = V_mueller(X + theta * dz_x, Y + theta * dz_y)
                Vm = V_mueller(X - theta * dz_x, Y - theta * dz_y)
                rp = np.exp(np.clip(-(Vp - V) / (eps**2), -LOGR_CLIP, LOGR_CLIP))
                rm = np.exp(np.clip(-(Vm - V) / (eps**2), -LOGR_CLIP, LOGR_CLIP))
                t_acc_x += 0.5 * (rm - rp) * dz_x
                t_acc_y += 0.5 * (rm - rp) * dz_y
            Sx += (prob_k / len(dirs)) * (t_acc_x / len(thetas))
            Sy += (prob_k / len(dirs)) * (t_acc_y / len(thetas))

    return pi, bx, by, lam * np.clip(Sx, -S_CLIP, S_CLIP), lam * np.clip(Sy, -S_CLIP, S_CLIP)


def step_diff(X, dt, eps, gx, gy, bx_g, by_g, rng):
    bx = bilinear_interp(X[:, 0], X[:, 1], gx, gy, bx_g)
    by = bilinear_interp(X[:, 0], X[:, 1], gx, gy, by_g)
    factor = 1.0 / (1.0 + dt * (np.sqrt(bx**2 + by**2) + 1e-8))
    X_new = X + dt * np.stack([bx, by], axis=1) * factor[:, None]
    return X_new + eps * np.sqrt(dt) * rng.standard_normal(X.shape)


def step_levy(X, dt, eps, gx, gy, bx_g, by_g, Sx_g, Sy_g, rng, lam, sigma_L, mults, pm):
    bx = bilinear_interp(X[:, 0], X[:, 1], gx, gy, bx_g)
    by = bilinear_interp(X[:, 0], X[:, 1], gx, gy, by_g)
    sx = bilinear_interp(X[:, 0], X[:, 1], gx, gy, Sx_g)
    sy = bilinear_interp(X[:, 0], X[:, 1], gx, gy, Sy_g)
    dx = bx - sx
    dy = by - sy
    factor = 1.0 / (1.0 + dt * (np.sqrt(dx**2 + dy**2) + 1e-8))
    X_new = X + dt * np.stack([dx, dy], axis=1) * factor[:, None]
    X_new += eps * np.sqrt(dt) * rng.standard_normal(X.shape)

    n_jumps = rng.poisson(lam * dt, size=X.shape[0])
    for i in np.where(n_jumps > 0)[0]:
        k = int(n_jumps[i])
        m_choice = rng.choice(mults, size=k, p=pm)
        ang = rng.random(k) * 2 * np.pi
        jx = np.sum(sigma_L * m_choice * np.cos(ang))
        jy = np.sum(sigma_L * m_choice * np.sin(ang))
        X_new[i, 0] += jx
        X_new[i, 1] += jy
    return X_new


def logpi_mueller(x, y, eps):
    return -V_mueller(x, y) / (eps**2)


def grad_logpi_mueller(x, y, eps):
    dVx, dVy = gradV_mueller(x, y)
    scale = -1.0 / (eps**2)
    return scale * dVx, scale * dVy


def step_mala(X, dt, eps, rng):
    x = X[:, 0]
    y = X[:, 1]
    gx, gy = grad_logpi_mueller(x, y, eps)
    grad = np.stack([gx, gy], axis=1)

    mean = X + 0.5 * dt * grad
    proposal = mean + np.sqrt(dt) * rng.standard_normal(X.shape)

    logp_x = logpi_mueller(x, y, eps)
    logp_y = logpi_mueller(proposal[:, 0], proposal[:, 1], eps)

    gyx, gyy = grad_logpi_mueller(proposal[:, 0], proposal[:, 1], eps)
    grad_y = np.stack([gyx, gyy], axis=1)
    mean_y = proposal + 0.5 * dt * grad_y

    log_q_y_given_x = -np.sum((proposal - mean) ** 2, axis=1) / (2.0 * dt)
    log_q_x_given_y = -np.sum((X - mean_y) ** 2, axis=1) / (2.0 * dt)
    log_alpha = (logp_y + log_q_x_given_y) - (logp_x + log_q_y_given_x)

    accept = np.log(rng.random(X.shape[0])) < log_alpha
    X_new = X.copy()
    X_new[accept] = proposal[accept]
    return X_new, float(accept.mean())


def step_malevy(X, dt, eps, rng, lam, sigma_L, mults, pm):
    # Retained for reference; not used in the active benchmark workflow.
    X_mid, _ = step_mala(X, dt, eps, rng)
    proposal = X_mid.copy()
    n_jumps = rng.poisson(lam * dt, size=X.shape[0])
    for i in np.where(n_jumps > 0)[0]:
        k = int(n_jumps[i])
        m_choice = rng.choice(mults, size=k, p=pm)
        ang = rng.random(k) * 2 * np.pi
        proposal[i, 0] += np.sum(sigma_L * m_choice * np.cos(ang))
        proposal[i, 1] += np.sum(sigma_L * m_choice * np.sin(ang))

    logp_x = logpi_mueller(X_mid[:, 0], X_mid[:, 1], eps)
    logp_y = logpi_mueller(proposal[:, 0], proposal[:, 1], eps)
    log_alpha = logp_y - logp_x
    accept = np.log(rng.random(X.shape[0])) < log_alpha
    X_new = X_mid.copy()
    X_new[accept] = proposal[accept]
    return X_new, float(accept.mean())


def sample_from_pi(rng, pi, gx, gy, N):
    flat_pi = pi.ravel() / pi.sum()
    indices = rng.choice(flat_pi.size, size=N, p=flat_pi)
    y_idx, x_idx = np.unravel_index(indices, pi.shape)
    dx = gx[1] - gx[0]
    dy = gy[1] - gy[0]
    return np.stack(
        [
            gx[x_idx] + (rng.random(N) - 0.5) * dx,
            gy[y_idx] + (rng.random(N) - 0.5) * dy,
        ],
        axis=1,
    )


# ============================================================
# 5. Plotting
# ============================================================

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

    methods_l1 = [
        ("l1_d", "ULA", "C0"),
        ("l1_m", "MALA", "C2"),
        ("l1_flmc", "FLMC", "tab:orange"),
        ("l1_l", "LSBMC", "C3"),
    ]
    for key, label, color in methods_l1:
        axes[0].errorbar(t, mean[key], yerr=std[key], fmt="-", color=color, label=label, alpha=0.9, capsize=2)
    axes[0].set_title("L1 Error")
    axes[0].set_xlabel("Time")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    methods_l2 = [
        ("l2_d", "ULA", "C0"),
        ("l2_m", "MALA", "C2"),
        ("l2_flmc", "FLMC", "tab:orange"),
        ("l2_l", "LSBMC", "C3"),
    ]
    for key, label, color in methods_l2:
        axes[1].errorbar(t, mean[key], yerr=std[key], fmt="-", color=color, label=label, alpha=0.9, capsize=2)
    axes[1].set_title("L2 Error")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Müller-Brown: Metrics Comparison")
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


# ============================================================
# 6. Main Execution
# ============================================================

def run_simulation(
    seed,
    eps,
    dt,
    T,
    N,
    gx,
    gy,
    dx_grid,
    dy_grid,
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

    X_diff = np.zeros((N, 2)) + np.array([1.0, 0.0]) + rng.standard_normal((N, 2)) * 0.1
    X_levy = X_diff.copy()
    X_flmc = X_diff.copy()
    X_mala = X_diff.copy()

    metrics = {
        "t": [],
        "l1_d": [],
        "l1_l": [],
        "l1_flmc": [],
        "l1_m": [],
        "l2_d": [],
        "l2_l": [],
        "l2_flmc": [],
        "l2_m": [],
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
    benchmark_lower = np.array([gx[0], gy[0]], dtype=float)
    benchmark_upper = np.array([gx[-1], gy[-1]], dtype=float)
    if benchmark_ref_samples is not None and benchmark_config is not None:
        benchmark_history = init_benchmark_history(benchmark_metric_names)
    if mala_dt is None:
        mala_dt = dt
    acc_sum = 0.0
    acc_count = 0

    steps = int(T / dt)
    check_interval = max(1, steps // 20)

    for i in range(steps + 1):
        if i % check_interval == 0 or i == steps:
            t = i * dt
            metrics["t"].append(t)

            d_dens = density_on_grid(X_diff, gx, gy)
            l_dens = density_on_grid(X_levy, gx, gy)
            f_dens = density_on_grid(X_flmc, gx, gy)
            m_dens = density_on_grid(X_mala, gx, gy)
            _, l1_d, l2_d = compute_errors(d_dens, pi, dx_grid, dy_grid)
            _, l1_l, l2_l = compute_errors(l_dens, pi, dx_grid, dy_grid)
            _, l1_f, l2_f = compute_errors(f_dens, pi, dx_grid, dy_grid)
            _, l1_m, l2_m = compute_errors(m_dens, pi, dx_grid, dy_grid)
            metrics["l1_d"].append(l1_d)
            metrics["l1_l"].append(l1_l)
            metrics["l1_flmc"].append(l1_f)
            metrics["l1_m"].append(l1_m)
            metrics["l2_d"].append(l2_d)
            metrics["l2_l"].append(l2_l)
            metrics["l2_flmc"].append(l2_f)
            metrics["l2_m"].append(l2_m)

            if benchmark_history is not None:
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
                        context=f"mueller seed={seed} t={t:.4f} method={slug}",
                    )
                    for metric_name in benchmark_metric_names:
                        benchmark_history[f"{metric_name}_{slug}"].append(
                            values.get(metric_name, float("nan"))
                        )

        X_diff = step_diff(X_diff, dt, eps, gx, gy, bx, by, rng)
        X_levy = step_levy(X_levy, dt, eps, gx, gy, bx, by, Sx, Sy, rng, lam, sigma_L, mults, pm)
        X_flmc = step_flmc_2d(
            X_flmc,
            dt,
            alpha,
            eps,
            V_mueller_flmc,
            gradV_mueller_flmc,
            rng,
        )
        X_mala, acc = step_mala(X_mala, mala_dt, eps, rng)
        acc_sum += acc
        acc_count += 1

    acc_rate = acc_sum / max(acc_count, 1)
    if return_benchmark_history:
        return metrics, benchmark_history, X_diff, X_levy, X_flmc, X_mala, acc_rate
    return metrics, X_diff, X_levy, X_flmc, X_mala, acc_rate


def aggregate_histories(histories, keys):
    stacked = {k: np.stack([np.array(h[k]) for h in histories], axis=0) for k in keys}
    mean = {k: stacked[k].mean(axis=0) for k in keys}
    std = {k: stacked[k].std(axis=0) for k in keys}
    return mean, std


def main():
    eps, dt, T, N = 0.7, 2e-4, 2.0, 3000
    gx, gy = np.linspace(-1.8, 1.8, 200), np.linspace(-1.2, 2.2, 200)
    dx_grid, dy_grid = gx[1] - gx[0], gy[1] - gy[0]
    lam, sigma_L, mults, pm = 8.0, 1.2, [0.8, 1.2], [0.5, 0.5]
    num_seeds = 5
    seeds = list(range(num_seeds))

    print("Precomputing Pi and Score...")
    pi, bx, by, Sx, Sy = precompute_pi_b_S(eps, gx, gy, lam, sigma_L, mults, pm)

    alpha = 1.5
    benchmark_config = get_default_benchmark_config({
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
        1500,
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
        metrics, benchmark_history, X_diff, X_levy, X_flmc, X_mala, acc_rate = run_simulation(
            seed,
            eps,
            dt,
            T,
            N,
            gx,
            gy,
            dx_grid,
            dy_grid,
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
            benchmark_mode_descriptors=MUELLER_MODE_DESCRIPTORS,
            return_benchmark_history=True,
        )
        histories.append(metrics)
        benchmark_histories.append(benchmark_history)
        acc_rates.append(acc_rate)
        if first_final is None:
            first_final = (X_diff, X_levy, X_flmc, X_mala)

    t = np.array(histories[0]["t"])
    keys = [
        "l1_d",
        "l1_l",
        "l1_flmc",
        "l1_m",
        "l2_d",
        "l2_l",
        "l2_flmc",
        "l2_m",
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

    out_dir = os.path.join(THIS_DIR, "mueller_output")
    os.makedirs(out_dir, exist_ok=True)
    plot_metrics_figure(t, mean, std, out_dir)
    benchmark_base = os.path.join(out_dir, "benchmark_metrics_mueller")
    plot_benchmark_metrics_figure(
        t,
        benchmark_mean,
        benchmark_std,
        ["benchmark_sinkhorn_divergence", "benchmark_mmd", "benchmark_emc"],
        benchmark_base,
        "Müller-Brown: Benchmark Metrics",
        metric_labels={
            "benchmark_sinkhorn_divergence": "Sinkhorn",
            "benchmark_mmd": "MMD",
            "benchmark_emc": "EMC",
        },
    )
    save_benchmark_metrics_csv(
        f"{benchmark_base}.csv",
        t,
        benchmark_mean,
        benchmark_std,
        benchmark_metric_names,
    )
    save_benchmark_metadata_json(
        os.path.join(out_dir, "metrics_benchmark_mueller.json"),
        "mueller",
        benchmark_config,
        benchmark_metric_names,
        mode_descriptors=MUELLER_MODE_DESCRIPTORS,
        extra_metadata={
            "benchmark_reference_size": int(benchmark_ref.shape[0]),
            "emc_applicable": True,
            "emc_descriptor_description": "Four explicit local minima of the Müller-Brown potential.",
        },
    )

    X_diff, X_levy, X_flmc, X_mala = first_final
    d_dens = density_on_grid(X_diff, gx, gy)
    l_dens = density_on_grid(X_levy, gx, gy)
    f_dens = density_on_grid(X_flmc, gx, gy)
    m_dens = density_on_grid(X_mala, gx, gy)
    norm = shared_density_norm([pi, d_dens, l_dens, f_dens, m_dens], use_log=False, gamma=0.5)
    init_point = np.array([1.0, 0.0])
    plot_density_figure(gx, gy, pi, "Müller-Brown: True Density", "true_density", out_dir, norm)
    plot_density_figure(gx, gy, d_dens, "Müller-Brown: ULA Density", "ula_density", out_dir, norm, init_point=init_point)
    plot_density_figure(gx, gy, m_dens, "Müller-Brown: MALA Density", "mala_density", out_dir, norm, init_point=init_point)
    plot_density_figure(gx, gy, f_dens, "Müller-Brown: FLMC Density", "flmc_density", out_dir, norm, init_point=init_point)
    plot_density_figure(gx, gy, l_dens, "Müller-Brown: LSBMC Density", "lsbmc_density", out_dir, norm, init_point=init_point)

    avg_acc = float(np.mean(acc_rates)) if acc_rates else 0.0
    print("Complete. Generated metrics and density figures.")
    print(out_dir)
    print(f"Saved benchmark metrics to: {benchmark_base}.png/.pdf/.csv")
    print(f"MALA mean acceptance rate: {avg_acc:.3f}")


if __name__ == "__main__":
    main()
