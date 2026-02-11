#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import ot  # POT
except Exception as e:
    raise RuntimeError("POT (ot) is required: pip install POT")

# ============================================================
# 1. Symmetric 4-Well Potential
# ============================================================

EXPO_CLIP = 60.0
LOGR_CLIP = 30.0
S_CLIP = 80.0

def V_fourwell(x, y, a=1.0):
    x, y = np.asarray(x), np.asarray(y)
    return (x**2 - a**2)**2 + (y**2 - a**2)**2

def gradV_fourwell(x, y, a=1.0):
    x, y = np.asarray(x), np.asarray(y)
    dVx = 4.0 * x * (x**2 - a**2)
    dVy = 4.0 * y * (y**2 - a**2)
    return dVx, dVy

# ============================================================
# 2. Utilities & Density Estimation
# ============================================================

def bilinear_interp(x, y, gx, gy, F):
    x, y = np.clip(x, gx[0], gx[-1]), np.clip(y, gy[0], gy[-1])
    dx, dy = gx[1] - gx[0], gy[1] - gy[0]
    ix = np.clip(np.searchsorted(gx, x, side="right") - 1, 0, gx.size - 2)
    iy = np.clip(np.searchsorted(gy, y, side="right") - 1, 0, gy.size - 2)
    x1, y1 = gx[ix], gy[iy]
    wx, wy = (x - x1) / (dx + 1e-12), (y - y1) / (dy + 1e-12)
    return (1-wx)*(1-wy)*F[iy, ix] + wx*(1-wy)*F[iy, ix+1] + (1-wx)*wy*F[iy+1, ix] + wx*wy*F[iy+1, ix+1]

def gaussian_kernel_1d(sigma, dx):
    m = int(np.ceil(4.0 * sigma / dx))
    xs = np.arange(-m, m + 1) * dx
    ker = np.exp(-0.5 * (xs / sigma) ** 2)
    return ker / (np.sum(ker) * dx + 1e-300)

def smooth2d_separable(P, ker_x, ker_y):
    tmp = np.array([np.convolve(row, ker_x, mode="same") for row in P])
    return np.array([np.convolve(tmp[:, j], ker_y, mode="same") for j in range(P.shape[1])]).T

def density_on_grid(samples, gx, gy, do_smooth=True):
    dx, dy = gx[1] - gx[0], gy[1] - gy[0]
    bins_x = np.concatenate([gx - dx/2, [gx[-1] + dx/2]])
    bins_y = np.concatenate([gy - dy/2, [gy[-1] + dy/2]])
    H, _, _ = np.histogram2d(samples[:, 1], samples[:, 0], bins=[bins_y, bins_x])
    P = H / (samples.shape[0] * dx * dy + 1e-12)
    if do_smooth:
        ker = gaussian_kernel_1d(0.05, dx)
        P = smooth2d_separable(P, ker, ker)
    return np.maximum(P, 1e-300) / (P.sum() * dx * dy)

# ============================================================
# 3. Error Metrics
# ============================================================

def compute_errors(dens, pi, dx, dy):
    abs_err_map = np.abs(dens - pi)
    mae = np.sum(abs_err_map) * dx * dy
    l2_err = np.sqrt(np.sum((dens - pi)**2) * dx * dy)
    return abs_err_map, mae, l2_err

# ============================================================
# 4. Oracle Score & Simulation Steps
# ============================================================

def precompute_pi_b_S(eps, gx, gy, a, lam, sigma_L, mults, pm):
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    V = V_fourwell(X, Y, a)
    logpi = -V / (eps**2)
    logpi -= np.max(logpi)
    pi = np.exp(np.clip(logpi, -EXPO_CLIP, 0))
    pi /= (np.sum(pi) * (gx[1]-gx[0]) * (gy[1]-gy[0]))
    
    dVx, dVy = gradV_fourwell(X, Y, a)
    bx, by = -0.5 * dVx, -0.5 * dVy
    
    Sx, Sy = np.zeros_like(pi), np.zeros_like(pi)
    thetas = np.linspace(0.05, 1.0, 15)
    angs = np.linspace(0, 2*np.pi, 12, endpoint=False)
    dirs = np.stack([np.cos(angs), np.sin(angs)], axis=1)
    
    for k, pk in enumerate(pm):
        z_mag = sigma_L * mults[k]
        for u in dirs:
            zx, zy = z_mag * u[0], z_mag * u[1]
            tx, ty = 0, 0
            for th in thetas:
                rp = np.exp(np.clip(-(V_fourwell(X+th*zx, Y+th*zy, a)-V)/(eps**2), -LOGR_CLIP, LOGR_CLIP))
                rm = np.exp(np.clip(-(V_fourwell(X-th*zx, Y-th*zy, a)-V)/(eps**2), -LOGR_CLIP, LOGR_CLIP))
                tx += 0.5 * (rm - rp) * zx
                ty += 0.5 * (rm - rp) * zy
            Sx += (pk / len(dirs)) * (tx / len(thetas))
            Sy += (pk / len(dirs)) * (ty / len(thetas))
            
    return pi, bx, by, lam * np.clip(Sx, -S_CLIP, S_CLIP), lam * np.clip(Sy, -S_CLIP, S_CLIP)

def step_diff(X, dt, eps, gx, gy, bx_g, by_g, rng):
    bx, by = bilinear_interp(X[:,0], X[:,1], gx, gy, bx_g), bilinear_interp(X[:,0], X[:,1], gx, gy, by_g)
    return X + dt * np.stack([bx, by], axis=1) + eps * np.sqrt(dt) * rng.standard_normal(X.shape)

def step_levy(X, dt, eps, gx, gy, bx_g, by_g, Sx_g, Sy_g, rng, lam, sigma_L, mults, pm):
    bx, by = bilinear_interp(X[:,0], X[:,1], gx, gy, bx_g), bilinear_interp(X[:,0], X[:,1], gx, gy, by_g)
    sx, sy = bilinear_interp(X[:,0], X[:,1], gx, gy, Sx_g), bilinear_interp(X[:,0], X[:,1], gx, gy, Sy_g)
    X_new = X + dt * np.stack([bx - sx, by - sy], axis=1) + eps * np.sqrt(dt) * rng.standard_normal(X.shape)
    
    n_jumps = rng.poisson(lam * dt, size=X.shape[0])
    for i in np.where(n_jumps > 0)[0]:
        m = rng.choice(mults, size=n_jumps[i], p=pm)
        ang = rng.random(n_jumps[i]) * 2 * np.pi
        X_new[i, 0] += np.sum(sigma_L * m * np.cos(ang))
        X_new[i, 1] += np.sum(sigma_L * m * np.sin(ang))
    return X_new

def logpi_fourwell_xy(x, y, eps, a):
    return -V_fourwell(x, y, a=a) / (eps ** 2)

def grad_logpi_fourwell_xy(x, y, eps, a):
    dVx, dVy = gradV_fourwell(x, y, a=a)
    scale = -1.0 / (eps ** 2)
    return scale * dVx, scale * dVy

def step_mala(X, dt, eps, a, rng):
    x = X[:, 0]
    y = X[:, 1]
    gx, gy = grad_logpi_fourwell_xy(x, y, eps, a=a)
    grad = np.stack([gx, gy], axis=1)

    mean = X + 0.5 * dt * grad
    proposal = mean + np.sqrt(dt) * rng.standard_normal(X.shape)

    logp_x = logpi_fourwell_xy(x, y, eps, a=a)
    logp_y = logpi_fourwell_xy(proposal[:, 0], proposal[:, 1], eps, a=a)

    gyx, gyy = grad_logpi_fourwell_xy(proposal[:, 0], proposal[:, 1], eps, a=a)
    grad_y = np.stack([gyx, gyy], axis=1)
    mean_y = proposal + 0.5 * dt * grad_y

    log_q_y_given_x = -np.sum((proposal - mean) ** 2, axis=1) / (2.0 * dt)
    log_q_x_given_y = -np.sum((X - mean_y) ** 2, axis=1) / (2.0 * dt)
    log_alpha = (logp_y + log_q_x_given_y) - (logp_x + log_q_y_given_x)

    accept = np.log(rng.random(X.shape[0])) < log_alpha
    X_new = X.copy()
    X_new[accept] = proposal[accept]
    return X_new, float(accept.mean())

def step_malevy(X, dt, eps, a, rng, lam, sigma_L, mults, pm):
    # Stage 1: local MALA move (gradient-informed)
    X_mid, _ = step_mala(X, dt, eps, a, rng)

    # Stage 2: symmetric Lévy jump MH move (global)
    proposal = X_mid.copy()
    n_jumps = rng.poisson(lam * dt, size=X.shape[0])
    for i in np.where(n_jumps > 0)[0]:
        k = int(n_jumps[i])
        m_choice = rng.choice(mults, size=k, p=pm)
        ang = rng.random(k) * 2 * np.pi
        proposal[i, 0] += np.sum(sigma_L * m_choice * np.cos(ang))
        proposal[i, 1] += np.sum(sigma_L * m_choice * np.sin(ang))

    logp_x = logpi_fourwell_xy(X_mid[:, 0], X_mid[:, 1], eps, a=a)
    logp_y = logpi_fourwell_xy(proposal[:, 0], proposal[:, 1], eps, a=a)
    log_alpha = logp_y - logp_x
    accept = np.log(rng.random(X.shape[0])) < log_alpha
    X_new = X_mid.copy()
    X_new[accept] = proposal[accept]
    return X_new, float(accept.mean())

# ============================================================
# 5. Main Simulation
# ============================================================

def run_simulation(seed, a, eps, dt, T, N, gx, gy, dx, dy, lam, sigma_L, mults, pm, pi, bx, by, Sx, Sy, mala_dt=None):
    rng = np.random.default_rng(seed)

    # Target distribution samples for W2
    flat_pi = pi.ravel() / pi.sum()
    idx = rng.choice(flat_pi.size, size=2000, p=flat_pi)
    y_idx, x_idx = np.unravel_index(idx, pi.shape)
    ref = np.stack([gx[x_idx], gy[y_idx]], axis=1)

    # Init in one well (a, a)
    X_diff = np.array([a, a]) + 0.1 * rng.standard_normal((N, 2))
    X_levy = X_diff.copy()
    X_mala = X_diff.copy()
    X_malevy = X_diff.copy()

    history = {
        't': [],
        'w2_d': [], 'w2_l': [], 'w2_m': [], 'w2_ml': [],
        'mae_d': [], 'mae_l': [], 'mae_m': [], 'mae_ml': [],
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

    for i in range(steps + 1):
        if i % check == 0:
            t = i * dt
            history['t'].append(t)

            # W2 Distance
            sub = 800
            M_d = ot.dist(X_diff[rng.choice(N, sub)], ref[rng.choice(ref.shape[0], sub)], metric='sqeuclidean')
            M_l = ot.dist(X_levy[rng.choice(N, sub)], ref[rng.choice(ref.shape[0], sub)], metric='sqeuclidean')
            M_m = ot.dist(X_mala[rng.choice(N, sub)], ref[rng.choice(ref.shape[0], sub)], metric='sqeuclidean')
            M_ml = ot.dist(X_malevy[rng.choice(N, sub)], ref[rng.choice(ref.shape[0], sub)], metric='sqeuclidean')
            history['w2_d'].append(np.sqrt(ot.sinkhorn2([], [], M_d, reg=0.05)))
            history['w2_l'].append(np.sqrt(ot.sinkhorn2([], [], M_l, reg=0.05)))
            history['w2_m'].append(np.sqrt(ot.sinkhorn2([], [], M_m, reg=0.05)))
            history['w2_ml'].append(np.sqrt(ot.sinkhorn2([], [], M_ml, reg=0.05)))

            # Grid Errors
            d_dens = density_on_grid(X_diff, gx, gy)
            l_dens = density_on_grid(X_levy, gx, gy)
            m_dens = density_on_grid(X_mala, gx, gy)
            ml_dens = density_on_grid(X_malevy, gx, gy)
            _, md, l2d = compute_errors(d_dens, pi, dx, dy)
            _, ml, l2l = compute_errors(l_dens, pi, dx, dy)
            _, mm, l2m = compute_errors(m_dens, pi, dx, dy)
            _, mml, l2ml = compute_errors(ml_dens, pi, dx, dy)
            history['mae_d'].append(md); history['mae_l'].append(ml)
            history['l2_d'].append(l2d); history['l2_l'].append(l2l)
            history['mae_m'].append(mm); history['l2_m'].append(l2m)
            history['mae_ml'].append(mml); history['l2_ml'].append(l2ml)

        X_diff = step_diff(X_diff, dt, eps, gx, gy, bx, by, rng)
        X_levy = step_levy(X_levy, dt, eps, gx, gy, bx, by, Sx, Sy, rng, lam, sigma_L, mults, pm)
        X_mala, acc = step_mala(X_mala, mala_dt, eps, a, rng)
        acc_sum += acc
        acc_count += 1
        X_malevy, acc_ml = step_malevy(X_malevy, dt, eps, a, rng, lam, sigma_L, mults, pm)
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
    # Params
    a, eps, dt, T, N = 1.0, 0.5, 0.005, 15.0, 5000
    gx, gy = np.linspace(-1.8, 1.8, 200), np.linspace(-1.8, 1.8, 200)
    dx, dy = gx[1]-gx[0], gy[1]-gy[0]
    lam, sigma_L, mults, pm = 1.2, 1.0, [1.0, 2.0], [0.85, 0.15]
    num_seeds = 5
    seeds = list(range(num_seeds))

    print("Precomputing fields...")
    pi, bx, by, Sx, Sy = precompute_pi_b_S(eps, gx, gy, a, lam, sigma_L, mults, pm)

    print(f"Starting simulations over {num_seeds} seeds...")
    histories = []
    first_final = None
    acc_rates = []
    acc_rates_ml = []
    for seed in seeds:
        history, X_diff, X_levy, X_mala, X_malevy, acc_rate, acc_rate_ml = run_simulation(
            seed, a, eps, dt, T, N, gx, gy, dx, dy, lam, sigma_L, mults, pm, pi, bx, by, Sx, Sy
        )
        histories.append(history)
        acc_rates.append(acc_rate)
        acc_rates_ml.append(acc_rate_ml)
        if first_final is None:
            first_final = (X_diff, X_levy, X_mala, X_malevy)

    t = np.array(histories[0]['t'])
    keys = [
        'w2_d', 'w2_l', 'w2_m', 'w2_ml',
        'mae_d', 'mae_l', 'mae_m', 'mae_ml',
        'l2_d', 'l2_l', 'l2_m', 'l2_ml'
    ]
    mean, std = aggregate_histories(histories, keys)

    # ============================================================
    # 6. Visualization
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].errorbar(t, mean['w2_d'], yerr=std['w2_d'], fmt='b--', label='Diff', capsize=2, alpha=0.9)
    axes[0].errorbar(t, mean['w2_l'], yerr=std['w2_l'], fmt='r-', label='Levy', capsize=2, alpha=0.9)
    axes[0].errorbar(t, mean['w2_m'], yerr=std['w2_m'], fmt='g-', label='MALA', capsize=2, alpha=0.9)
    axes[0].errorbar(t, mean['w2_ml'], yerr=std['w2_ml'], fmt='k-', label='MALA-Levy', capsize=2, alpha=0.9)
    axes[0].set_title('W2 Distance'); axes[0].legend()
    
    axes[1].errorbar(t, mean['mae_d'], yerr=std['mae_d'], fmt='b--', label='Diff', capsize=2, alpha=0.9)
    axes[1].errorbar(t, mean['mae_l'], yerr=std['mae_l'], fmt='r-', label='Levy', capsize=2, alpha=0.9)
    axes[1].errorbar(t, mean['mae_m'], yerr=std['mae_m'], fmt='g-', label='MALA', capsize=2, alpha=0.9)
    axes[1].errorbar(t, mean['mae_ml'], yerr=std['mae_ml'], fmt='k-', label='MALA-Levy', capsize=2, alpha=0.9)
    axes[1].set_title('MAE ($L^1$) Error')
    axes[1].legend()
    
    axes[2].errorbar(t, mean['l2_d'], yerr=std['l2_d'], fmt='b--', label='Diff', capsize=2, alpha=0.9)
    axes[2].errorbar(t, mean['l2_l'], yerr=std['l2_l'], fmt='r-', label='Levy', capsize=2, alpha=0.9)
    axes[2].errorbar(t, mean['l2_m'], yerr=std['l2_m'], fmt='g-', label='MALA', capsize=2, alpha=0.9)
    axes[2].errorbar(t, mean['l2_ml'], yerr=std['l2_ml'], fmt='k-', label='MALA-Levy', capsize=2, alpha=0.9)
    axes[2].set_title('$L^2$ (RMSE) Error')
    axes[2].legend()
    plt.tight_layout()
    plt.savefig("fourwell_errors.png", dpi=200)
    plt.close()

    # Spatial Error Map
    X_diff, X_levy, X_mala, X_malevy = first_final
    d_dens = density_on_grid(X_diff, gx, gy)
    l_dens = density_on_grid(X_levy, gx, gy)
    m_dens = density_on_grid(X_mala, gx, gy)
    ml_dens = density_on_grid(X_malevy, gx, gy)
    err_d, _, _ = compute_errors(d_dens, pi, dx, dy)
    err_l, _, _ = compute_errors(l_dens, pi, dx, dy)
    err_m, _, _ = compute_errors(m_dens, pi, dx, dy)
    err_ml, _, _ = compute_errors(ml_dens, pi, dx, dy)
    vmax = max(np.max(err_d), np.max(err_l), np.max(err_m), np.max(err_ml))
    
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    im1 = ax[0].imshow(err_d, origin='lower', extent=[gx[0],gx[-1],gy[0],gy[-1]], cmap='inferno', vmin=0, vmax=vmax)
    ax[0].set_title("Diffusion Spatial Error (Stuck in 1 Well)"); fig.colorbar(im1, ax=ax[0])
    im2 = ax[1].imshow(err_l, origin='lower', extent=[gx[0],gx[-1],gy[0],gy[-1]], cmap='inferno', vmin=0, vmax=vmax)
    ax[1].set_title("Lévy PF Spatial Error (Mixed)"); fig.colorbar(im2, ax=ax[1])
    im3 = ax[2].imshow(err_m, origin='lower', extent=[gx[0],gx[-1],gy[0],gy[-1]], cmap='inferno', vmin=0, vmax=vmax)
    ax[2].set_title("MALA Spatial Error"); fig.colorbar(im3, ax=ax[2])
    im4 = ax[3].imshow(err_ml, origin='lower', extent=[gx[0],gx[-1],gy[0],gy[-1]], cmap='inferno', vmin=0, vmax=vmax)
    ax[3].set_title("MALA-Levy Spatial Error"); fig.colorbar(im4, ax=ax[3])
    plt.savefig("fourwell_spatial_error.png", dpi=200)
    plt.close()
    
    print("Done. Check fourwell_errors.png and fourwell_spatial_error.png")
    avg_acc = float(np.mean(acc_rates)) if acc_rates else 0.0
    print(f"MALA mean acceptance rate: {avg_acc:.3f}")
    avg_acc_ml = float(np.mean(acc_rates_ml)) if acc_rates_ml else 0.0
    print(f"MALA-Levy mean acceptance rate: {avg_acc_ml:.3f}")

if __name__ == "__main__":
    main()
