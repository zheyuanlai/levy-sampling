#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# Optional: POT (Python Optimal Transport)
try:
    import ot  # pip install POT
    HAS_POT = True
except Exception:
    HAS_POT = False

# ============================================================
# Example: Ring Potential
#   V(x,y) = (1 - x^2 - y^2)^2 + y^2/(x^2+y^2)
# ============================================================

R2_EPS = 1e-12
EXPO_CLIP = 60.0
S_CLIP = 40.0

# ---------------- Potential & grad ----------------

def V_ring(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    r2 = x * x + y * y
    r2s = r2 + R2_EPS
    return (1.0 - r2) ** 2 + (y * y) / r2s

def gradV_ring(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    r2 = x * x + y * y
    r2s = r2 + R2_EPS
    r4s = r2s * r2s
    dVx = 4.0 * (r2 - 1.0) * x - 2.0 * x * (y * y) / r4s
    dVy = 4.0 * (r2 - 1.0) * y + 2.0 * y * (x * x) / r4s
    return dVx, dVy

# ---------------- Utilities ----------------

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
    bx = bilinear_interp(X[:,0], X[:,1], gx, gy, bx_g)
    by = bilinear_interp(X[:,0], X[:,1], gx, gy, by_g)
    norm = np.sqrt(bx**2 + by**2) + 1e-8
    drift = dt * np.stack([bx, by], axis=1) / (1.0 + dt*norm)[:,None]
    return X + drift + np.sqrt(eps*dt) * rng.standard_normal(X.shape)

def step_levy(X, dt, eps, gx, gy, bx_g, by_g, Sx_g, Sy_g, rng, lam, sigma_L, mults, pm):
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
    return X_new

def logpi_ring_xy(x, y, eps):
    return -2.0 * V_ring(x, y) / eps

def grad_logpi_ring_xy(x, y, eps):
    dVx, dVy = gradV_ring(x, y)
    scale = -2.0 / eps
    return scale * dVx, scale * dVy

def step_mala(X, dt, eps, rng):
    x = X[:, 0]
    y = X[:, 1]
    gx, gy = grad_logpi_ring_xy(x, y, eps)
    grad = np.stack([gx, gy], axis=1)

    mean = X + 0.5 * dt * grad
    proposal = mean + np.sqrt(dt) * rng.standard_normal(X.shape)

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
    return X_new, float(accept.mean())

def step_malevy(X, dt, eps, rng, lam, sigma_L, mults, pm, jump_cap=3.2):
    # Stage 1: local MALA move (gradient-informed)
    X_mid, _ = step_mala(X, dt, eps, rng)

    # Stage 2: symmetric Lévy jump MH move (global)
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
def wasserstein2(X, Y, rng, m=400):
    A = X[rng.choice(X.shape[0], m, replace=False)]
    B = Y[rng.choice(Y.shape[0], m, replace=False)]
    if HAS_POT:
        M = ot.dist(A, B, metric='sqeuclidean')
        return np.sqrt(ot.sinkhorn2([], [], M, reg=0.05, numItermax=5000))
    else:
        return 0.0 # Fallback

# ---------------- Plotting Functions ----------------

def plot_errors_over_time(t, mean, std, out_prefix):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Wasserstein-2
    axes[0].errorbar(t, mean['w_d'], yerr=std['w_d'], fmt='b--', label='Diffusion', alpha=0.9, capsize=2)
    axes[0].errorbar(t, mean['w_l'], yerr=std['w_l'], fmt='r-', label='Lévy PF', alpha=0.9, capsize=2)
    axes[0].errorbar(t, mean['w_m'], yerr=std['w_m'], fmt='g-', label='MALA', alpha=0.9, capsize=2)
    axes[0].errorbar(t, mean['w_ml'], yerr=std['w_ml'], fmt='k-', label='MALA-Levy', alpha=0.9, capsize=2)
    axes[0].set_title('Wasserstein-2 Distance ($W_2$)')
    axes[0].set_xlabel('Time')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. L1 Error (MAE)
    axes[1].errorbar(t, mean['l1_d'], yerr=std['l1_d'], fmt='b--', label='Diffusion', alpha=0.9, capsize=2)
    axes[1].errorbar(t, mean['l1_l'], yerr=std['l1_l'], fmt='r-', label='Lévy PF', alpha=0.9, capsize=2)
    axes[1].errorbar(t, mean['l1_m'], yerr=std['l1_m'], fmt='g-', label='MALA', alpha=0.9, capsize=2)
    axes[1].errorbar(t, mean['l1_ml'], yerr=std['l1_ml'], fmt='k-', label='MALA-Levy', alpha=0.9, capsize=2)
    axes[1].set_title('Mean Absolute Error ($L^1$)')
    axes[1].set_xlabel('Time')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 3. L2 Error (RMSE)
    axes[2].errorbar(t, mean['l2_d'], yerr=std['l2_d'], fmt='b--', label='Diffusion', alpha=0.9, capsize=2)
    axes[2].errorbar(t, mean['l2_l'], yerr=std['l2_l'], fmt='r-', label='Lévy PF', alpha=0.9, capsize=2)
    axes[2].errorbar(t, mean['l2_m'], yerr=std['l2_m'], fmt='g-', label='MALA', alpha=0.9, capsize=2)
    axes[2].errorbar(t, mean['l2_ml'], yerr=std['l2_ml'], fmt='k-', label='MALA-Levy', alpha=0.9, capsize=2)
    axes[2].set_title('Root Mean Square Error ($L^2$)')
    axes[2].set_xlabel('Time')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_errors_convergence.png", dpi=200)
    plt.close()

def plot_spatial_heatmaps(gx, gy, diff_dens, levy_dens, mala_dens, malevy_dens, pi, out_prefix):
    # Calculate Abs Errors
    err_d = np.abs(diff_dens - pi)
    err_l = np.abs(levy_dens - pi)
    err_m = np.abs(mala_dens - pi)
    err_ml = np.abs(malevy_dens - pi)
    
    # Common color scale for fairness
    vmax = max(np.max(err_d), np.max(err_l), np.max(err_m), np.max(err_ml))
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    extent = [gx[0], gx[-1], gy[0], gy[-1]]
    
    # Diffusion Map
    im1 = axes[0].imshow(err_d, origin='lower', extent=extent, cmap='hot', vmin=0, vmax=vmax)
    axes[0].set_title(f'Spatial Error: Diffusion\nMax Err: {np.max(err_d):.2f}')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Levy Map
    im2 = axes[1].imshow(err_l, origin='lower', extent=extent, cmap='hot', vmin=0, vmax=vmax)
    axes[1].set_title(f'Spatial Error: Lévy PF\nMax Err: {np.max(err_l):.2f}')
    axes[1].set_xlabel('x')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # MALA Map
    im3 = axes[2].imshow(err_m, origin='lower', extent=extent, cmap='hot', vmin=0, vmax=vmax)
    axes[2].set_title(f'Spatial Error: MALA\nMax Err: {np.max(err_m):.2f}')
    axes[2].set_xlabel('x')
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # MALA-Levy Map
    im4 = axes[3].imshow(err_ml, origin='lower', extent=extent, cmap='hot', vmin=0, vmax=vmax)
    axes[3].set_title(f'Spatial Error: MALA-Levy\nMax Err: {np.max(err_ml):.2f}')
    axes[3].set_xlabel('x')
    fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_spatial_error.png", dpi=200)
    plt.close()

# ---------------- Main ----------------

def run_simulation(seed, eps, dt, T, N, gx, gy, dx, dy, lam, sigma_L, mults, pm, pi, bx, by, Sx, Sy, mala_dt=None):
    rng = np.random.default_rng(seed)
    ref_samples = sample_from_pi_grid(rng, pi, gx, gy, 2000)

    # Init trapped in Left Well (-1, 0)
    X_diff = np.array([-1.0, 0.0]) + 0.05 * rng.standard_normal((N, 2))
    X_levy = X_diff.copy()
    X_mala = X_diff.copy()
    X_malevy = X_diff.copy()

    history = {
        't': [],
        'w_d': [], 'w_l': [], 'w_m': [], 'w_ml': [],
        'l1_d': [], 'l1_l': [], 'l1_m': [], 'l1_ml': [],
        'l2_d': [], 'l2_l': [], 'l2_m': [], 'l2_ml': []
    }
    if mala_dt is None:
        mala_dt = dt
    acc_sum = 0.0
    acc_count = 0
    acc_sum_ml = 0.0
    acc_count_ml = 0

    steps = int(T/dt)
    check = steps // 20 

    for i in range(steps+1):
        if i % check == 0 or i == steps:
            t = i * dt
            history['t'].append(t)

            # W2
            wd = wasserstein2(X_diff, ref_samples, rng)
            wl = wasserstein2(X_levy, ref_samples, rng)
            wm = wasserstein2(X_mala, ref_samples, rng)
            wml = wasserstein2(X_malevy, ref_samples, rng)
            history['w_d'].append(wd); history['w_l'].append(wl); history['w_m'].append(wm); history['w_ml'].append(wml)

            # Grid Errors
            dd = density_on_grid(X_diff, gx, gy)
            dl = density_on_grid(X_levy, gx, gy)
            dm = density_on_grid(X_mala, gx, gy)
            dml = density_on_grid(X_malevy, gx, gy)
            _, l1d, l2d = compute_grid_errors(dd, pi, dx, dy)
            _, l1l, l2l = compute_grid_errors(dl, pi, dx, dy)
            _, l1m, l2m = compute_grid_errors(dm, pi, dx, dy)
            _, l1ml, l2ml = compute_grid_errors(dml, pi, dx, dy)

            history['l1_d'].append(l1d); history['l1_l'].append(l1l)
            history['l2_d'].append(l2d); history['l2_l'].append(l2l)
            history['l1_m'].append(l1m); history['l2_m'].append(l2m)
            history['l1_ml'].append(l1ml); history['l2_ml'].append(l2ml)

        X_diff = step_diff(X_diff, dt, eps, gx, gy, bx, by, rng)
        X_levy = step_levy(X_levy, dt, eps, gx, gy, bx, by, Sx, Sy, rng, lam, sigma_L, mults, pm)
        X_mala, acc = step_mala(X_mala, mala_dt, eps, rng)
        acc_sum += acc
        acc_count += 1
        X_malevy, acc_ml = step_malevy(X_malevy, dt, eps, rng, lam, sigma_L, mults, pm)
        acc_sum_ml += acc_ml
        acc_count_ml += 1

    acc_rate = acc_sum / max(acc_count, 1)
    acc_rate_ml = acc_sum_ml / max(acc_count_ml, 1)
    return history, X_diff, X_levy, X_mala, X_malevy, acc_rate, acc_rate_ml

def aggregate_histories(histories, keys):
    stacked = {k: np.stack([np.array(h[k]) for h in histories], axis=0) for k in keys}
    mean = {k: stacked[k].mean(axis=0) for k in keys}
    std = {k: stacked[k].std(axis=0) for k in keys}
    return mean, std

def main():
    # Setup
    eps, dt, T, N = 0.35, 0.0015, 40.0, 5000
    gx, gy = np.linspace(-2.2, 2.2, 240), np.linspace(-2.2, 2.2, 240)
    dx, dy = gx[1]-gx[0], gy[1]-gy[0]
    
    # Levy Params
    lam, sigma_L = 1.6, 1.25
    mults, pm = [1.0, 1.7, 2.4], [0.70, 0.22, 0.08]
    num_seeds = 5
    seeds = list(range(num_seeds))

    print("Precomputing fields...")
    pi, bx, by, Sx, Sy = precompute_pi_drift_score_on_grid(eps, gx, gy, lam, sigma_L, mults, pm)

    print(f"Simulating {num_seeds} seeds...")
    histories = []
    first_final = None
    acc_rates = []
    acc_rates_ml = []
    for seed in seeds:
        history, X_diff, X_levy, X_mala, X_malevy, acc_rate, acc_rate_ml = run_simulation(
            seed, eps, dt, T, N, gx, gy, dx, dy, lam, sigma_L, mults, pm, pi, bx, by, Sx, Sy
        )
        histories.append(history)
        acc_rates.append(acc_rate)
        acc_rates_ml.append(acc_rate_ml)
        if first_final is None:
            first_final = (X_diff, X_levy, X_mala, X_malevy)

    t = np.array(histories[0]['t'])
    keys = [
        'w_d', 'w_l', 'w_m', 'w_ml',
        'l1_d', 'l1_l', 'l1_m', 'l1_ml',
        'l2_d', 'l2_l', 'l2_m', 'l2_ml'
    ]
    mean, std = aggregate_histories(histories, keys)

    # Final Plots
    out_prefix = "ring_final"
    
    # 1. Convergence Metrics Plot
    plot_errors_over_time(t, mean, std, out_prefix)
    
    # 2. Spatial Error Map Plot
    X_diff, X_levy, X_mala, X_malevy = first_final
    dens_d = density_on_grid(X_diff, gx, gy)
    dens_l = density_on_grid(X_levy, gx, gy)
    dens_m = density_on_grid(X_mala, gx, gy)
    dens_ml = density_on_grid(X_malevy, gx, gy)
    plot_spatial_heatmaps(gx, gy, dens_d, dens_l, dens_m, dens_ml, pi, out_prefix)
    
    avg_acc = float(np.mean(acc_rates)) if acc_rates else 0.0
    print(f"Done. Saved:\n1. {out_prefix}_errors_convergence.png\n2. {out_prefix}_spatial_error.png")
    print(f"MALA mean acceptance rate: {avg_acc:.3f}")
    avg_acc_ml = float(np.mean(acc_rates_ml)) if acc_rates_ml else 0.0
    print(f"MALA-Levy mean acceptance rate: {avg_acc_ml:.3f}")

if __name__ == "__main__":
    main()
