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
# 1. 4-Well Müller-Brown Potential (Scaled)
# ============================================================

EXPO_CLIP = 60.0
LOGR_CLIP = 30.0
S_CLIP = 80.0
SCALE_V = 0.05 

MUELLER_PARAMS = np.array([
    [-200, -1,   0,   -10,  1,    0],      # Well 1 (Right)
    [-200, -1,   0,   -10,  0,    0.5],    # Well 2 (Center)
    [-200, -6.5, 11,  -6.5, -0.5, 1.5],    # Well 3 (Top Left)
    [-200, -3,   0,   -3,   -0.8, -0.5]    # Well 4 (Bot Left)
])

def V_mueller(x, y):
    x, y = np.asarray(x), np.asarray(y)
    V = np.zeros_like(x)
    for p in MUELLER_PARAMS:
        A, a, b, c, x0, y0 = p
        V += (A * SCALE_V) * np.exp(a*(x - x0)**2 + b*(x - x0)*(y - y0) + c*(y - y0)**2)
    return V

def gradV_mueller(x, y):
    x, y = np.asarray(x), np.asarray(y)
    dVx, dVy = np.zeros_like(x), np.zeros_like(x)
    for p in MUELLER_PARAMS:
        A, a, b, c, x0, y0 = p
        dx, dy = x - x0, y - y0
        arg = a*dx**2 + b*dx*dy + c*dy**2
        exp_term = (A * SCALE_V) * np.exp(arg)
        dVx += exp_term * (2*a*dx + b*dy)
        dVy += exp_term * (b*dx + 2*c*dy)
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
    f11, f21 = F[iy, ix], F[iy, ix+1]
    f12, f22 = F[iy+1, ix], F[iy+1, ix+1]
    return (1-wx)*(1-wy)*f11 + wx*(1-wy)*f21 + (1-wx)*wy*f12 + wx*wy*f22

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
    dx, dy = gx[1] - gx[0], gy[1] - gy[0]
    bins_x = np.concatenate([gx - dx/2, [gx[-1] + dx/2]])
    bins_y = np.concatenate([gy - dy/2, [gy[-1] + dy/2]])
    H, _, _ = np.histogram2d(samples[:,1], samples[:,0], bins=[bins_y, bins_x])
    P = H / (samples.shape[0] * dx * dy + 1e-12)
    if do_smooth:
        ker = gaussian_kernel_1d(0.08, dx)
        P = smooth2d_separable(P, ker, ker)
    P = np.maximum(P, 1e-300)
    return P / (P.sum() * dx * dy)

# ============================================================
# 3. New Error Metrics Function
# ============================================================

def compute_errors(dens, pi, dx, dy):
    """Calculates point-wise Absolute Error, MAE (L1), and RMSE (L2)."""
    abs_err_map = np.abs(dens - pi)
    mae = np.sum(abs_err_map) * dx * dy
    l2_err = np.sqrt(np.sum((dens - pi)**2) * dx * dy)
    return abs_err_map, mae, l2_err

# ============================================================
# 4. Simulation Kernels
# ============================================================

def precompute_pi_b_S(eps, gx, gy, lam, sigma_L, multipliers, pm):
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    V = V_mueller(X, Y)
    logpi = -(V - np.min(V)) / (eps**2)
    pi_unn = np.exp(np.clip(logpi, -EXPO_CLIP, 0))
    pi = pi_unn / (np.sum(pi_unn) * (gx[1]-gx[0]) * (gy[1]-gy[0]))
    
    dVx, dVy = gradV_mueller(X, Y)
    bx, by = -0.5 * dVx, -0.5 * dVy
    
    Sx, Sy = np.zeros_like(pi), np.zeros_like(pi)
    thetas = np.linspace(0.05, 1.0, 20)
    ang = np.linspace(0, 2*np.pi, 16, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    
    for k, prob_k in enumerate(pm):
        jump_mag = sigma_L * multipliers[k]
        for u in dirs:
            dz_x, dz_y = jump_mag * u[0], jump_mag * u[1]
            t_acc_x, t_acc_y = 0, 0
            for theta in thetas:
                Vp = V_mueller(X + theta*dz_x, Y + theta*dz_y)
                Vm = V_mueller(X - theta*dz_x, Y - theta*dz_y)
                rp = np.exp(np.clip(-(Vp - V)/(eps**2), -LOGR_CLIP, LOGR_CLIP))
                rm = np.exp(np.clip(-(Vm - V)/(eps**2), -LOGR_CLIP, LOGR_CLIP))
                t_acc_x += 0.5 * (rm - rp) * dz_x
                t_acc_y += 0.5 * (rm - rp) * dz_y
            Sx += (prob_k / len(dirs)) * (t_acc_x / 20)
            Sy += (prob_k / len(dirs)) * (t_acc_y / 20)
            
    return pi, bx, by, lam*np.clip(Sx, -S_CLIP, S_CLIP), lam*np.clip(Sy, -S_CLIP, S_CLIP)

def step_diff(X, dt, eps, gx, gy, bx_g, by_g, rng):
    bx = bilinear_interp(X[:,0], X[:,1], gx, gy, bx_g)
    by = bilinear_interp(X[:,0], X[:,1], gx, gy, by_g)
    factor = 1.0 / (1.0 + dt * (np.sqrt(bx**2 + by**2) + 1e-8))
    X_new = X + dt * np.stack([bx, by], axis=1) * factor[:,None]
    return X_new + eps * np.sqrt(dt) * rng.standard_normal(X.shape)

def step_levy(X, dt, eps, gx, gy, bx_g, by_g, Sx_g, Sy_g, rng, lam, sigma_L, mults, pm):
    bx, by = bilinear_interp(X[:,0],X[:,1],gx,gy,bx_g), bilinear_interp(X[:,0],X[:,1],gx,gy,by_g)
    sx, sy = bilinear_interp(X[:,0],X[:,1],gx,gy,Sx_g), bilinear_interp(X[:,0],X[:,1],gx,gy,Sy_g)
    dx, dy = bx - sx, by - sy
    factor = 1.0 / (1.0 + dt * (np.sqrt(dx**2 + dy**2) + 1e-8))
    X_new = X + dt * np.stack([dx, dy], axis=1) * factor[:,None]
    X_new += eps * np.sqrt(dt) * rng.standard_normal(X.shape)
    
    n_jumps = rng.poisson(lam * dt, size=X.shape[0])
    for i in np.where(n_jumps > 0)[0]:
        k = int(n_jumps[i])
        m_choice = rng.choice(mults, size=k, p=pm)
        ang = rng.random(k) * 2 * np.pi
        jx, jy = np.sum(sigma_L * m_choice * np.cos(ang)), np.sum(sigma_L * m_choice * np.sin(ang))
        X_new[i,0] += jx
        X_new[i,1] += jy
    return X_new

def logpi_mueller(x, y, eps):
    return -V_mueller(x, y) / (eps ** 2)

def grad_logpi_mueller(x, y, eps):
    dVx, dVy = gradV_mueller(x, y)
    scale = -1.0 / (eps ** 2)
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
    # Stage 1: local MALA move (gradient-informed)
    X_mid, _ = step_mala(X, dt, eps, rng)

    # Stage 2: symmetric Lévy jump MH move (global)
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
    dx, dy = gx[1]-gx[0], gy[1]-gy[0]
    return np.stack([gx[x_idx] + (rng.random(N)-0.5)*dx, gy[y_idx] + (rng.random(N)-0.5)*dy], axis=1)

# ============================================================
# 5. Main Execution
# ============================================================

def run_simulation(seed, eps, dt, T, N, gx, gy, dx_grid, dy_grid, lam, sigma_L, mults, pm, pi, bx, by, Sx, Sy, mala_dt=None):
    rng = np.random.default_rng(seed)
    ref_samples = sample_from_pi(rng, pi, gx, gy, N=2000)

    X_diff = np.zeros((N, 2)) + np.array([1.0, 0.0]) + rng.standard_normal((N,2))*0.1
    X_levy = X_diff.copy()
    X_mala = X_diff.copy()
    X_malevy = X_diff.copy()

    # History tracking
    metrics = {
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
    check_interval = steps // 20

    for i in range(steps+1):
        if i % check_interval == 0:
            t = i * dt
            metrics['t'].append(t)

            # 1. Wasserstein Distance (W2)
            idx_d = rng.choice(N, 1000)
            idx_l = rng.choice(N, 1000)
            idx_m = rng.choice(N, 1000)
            idx_ml = rng.choice(N, 1000)
            idx_r = rng.choice(2000, 1000)
            M_d = ot.dist(X_diff[idx_d], ref_samples[idx_r], metric='sqeuclidean')
            M_l = ot.dist(X_levy[idx_l], ref_samples[idx_r], metric='sqeuclidean')
            M_m = ot.dist(X_mala[idx_m], ref_samples[idx_r], metric='sqeuclidean')
            M_ml = ot.dist(X_malevy[idx_ml], ref_samples[idx_r], metric='sqeuclidean')
            metrics['w2_d'].append(np.sqrt(ot.sinkhorn2([], [], M_d, reg=0.1)))
            metrics['w2_l'].append(np.sqrt(ot.sinkhorn2([], [], M_l, reg=0.1)))
            metrics['w2_m'].append(np.sqrt(ot.sinkhorn2([], [], M_m, reg=0.1)))
            metrics['w2_ml'].append(np.sqrt(ot.sinkhorn2([], [], M_ml, reg=0.1)))

            # 2. Grid-based Errors (MAE and L2)
            d_dens = density_on_grid(X_diff, gx, gy)
            l_dens = density_on_grid(X_levy, gx, gy)
            m_dens = density_on_grid(X_mala, gx, gy)
            ml_dens = density_on_grid(X_malevy, gx, gy)
            _, mae_d, l2_d = compute_errors(d_dens, pi, dx_grid, dy_grid)
            _, mae_l, l2_l = compute_errors(l_dens, pi, dx_grid, dy_grid)
            _, mae_m, l2_m = compute_errors(m_dens, pi, dx_grid, dy_grid)
            _, mae_ml, l2_ml = compute_errors(ml_dens, pi, dx_grid, dy_grid)
            metrics['mae_d'].append(mae_d); metrics['mae_l'].append(mae_l)
            metrics['l2_d'].append(l2_d); metrics['l2_l'].append(l2_l)
            metrics['mae_m'].append(mae_m); metrics['l2_m'].append(l2_m)
            metrics['mae_ml'].append(mae_ml); metrics['l2_ml'].append(l2_ml)

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
    return metrics, X_diff, X_levy, X_mala, X_malevy, acc_rate, acc_rate_ml

def aggregate_histories(histories, keys):
    stacked = {k: np.stack([np.array(h[k]) for h in histories], axis=0) for k in keys}
    mean = {k: stacked[k].mean(axis=0) for k in keys}
    std = {k: stacked[k].std(axis=0) for k in keys}
    return mean, std

def main():
    # Setup
    eps, dt, T, N = 0.7, 2e-4, 2.0, 3000
    gx, gy = np.linspace(-1.8, 1.8, 200), np.linspace(-1.2, 2.2, 200)
    dx_grid, dy_grid = gx[1]-gx[0], gy[1]-gy[0]
    lam, sigma_L, mults, pm = 8.0, 1.2, [0.8, 1.2], [0.5, 0.5]
    num_seeds = 5
    seeds = list(range(num_seeds))
    
    print("Precomputing Pi and Score...")
    pi, bx, by, Sx, Sy = precompute_pi_b_S(eps, gx, gy, lam, sigma_L, mults, pm)

    print(f"Simulating {num_seeds} seeds...")
    histories = []
    first_final = None
    acc_rates = []
    acc_rates_ml = []
    for seed in seeds:
        metrics, X_diff, X_levy, X_mala, X_malevy, acc_rate, acc_rate_ml = run_simulation(
            seed, eps, dt, T, N, gx, gy, dx_grid, dy_grid, lam, sigma_L, mults, pm, pi, bx, by, Sx, Sy
        )
        histories.append(metrics)
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
    # 6. Enhanced Plotting
    # ============================================================
    out = "mueller_analysis"
    
    # Plot 1: Combined Error Convergence
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].errorbar(t, mean['w2_d'], yerr=std['w2_d'], fmt='b--', label='Diff', capsize=2, alpha=0.9)
    ax[0].errorbar(t, mean['w2_l'], yerr=std['w2_l'], fmt='r-', label='Levy', capsize=2, alpha=0.9)
    ax[0].errorbar(t, mean['w2_m'], yerr=std['w2_m'], fmt='g-', label='MALA', capsize=2, alpha=0.9)
    ax[0].errorbar(t, mean['w2_ml'], yerr=std['w2_ml'], fmt='k-', label='MALA-Levy', capsize=2, alpha=0.9)
    ax[0].set_title('Wasserstein-2 Distance'); ax[0].legend()
    
    ax[1].errorbar(t, mean['mae_d'], yerr=std['mae_d'], fmt='b--', label='Diff', capsize=2, alpha=0.9)
    ax[1].errorbar(t, mean['mae_l'], yerr=std['mae_l'], fmt='r-', label='Levy', capsize=2, alpha=0.9)
    ax[1].errorbar(t, mean['mae_m'], yerr=std['mae_m'], fmt='g-', label='MALA', capsize=2, alpha=0.9)
    ax[1].errorbar(t, mean['mae_ml'], yerr=std['mae_ml'], fmt='k-', label='MALA-Levy', capsize=2, alpha=0.9)
    ax[1].set_title('Mean Absolute Error ($L^1$)')
    ax[1].legend()
    
    ax[2].errorbar(t, mean['l2_d'], yerr=std['l2_d'], fmt='b--', label='Diff', capsize=2, alpha=0.9)
    ax[2].errorbar(t, mean['l2_l'], yerr=std['l2_l'], fmt='r-', label='Levy', capsize=2, alpha=0.9)
    ax[2].errorbar(t, mean['l2_m'], yerr=std['l2_m'], fmt='g-', label='MALA', capsize=2, alpha=0.9)
    ax[2].errorbar(t, mean['l2_ml'], yerr=std['l2_ml'], fmt='k-', label='MALA-Levy', capsize=2, alpha=0.9)
    ax[2].set_title('Root Mean Square Error ($L^2$)')
    ax[2].legend()
    plt.tight_layout(); plt.savefig(f"{out}_convergence.png", dpi=200)
    plt.close()

    # Plot 2: Final Spatial Error Map
    X_diff, X_levy, X_mala, X_malevy = first_final
    d_dens = density_on_grid(X_diff, gx, gy)
    l_dens = density_on_grid(X_levy, gx, gy)
    m_dens = density_on_grid(X_mala, gx, gy)
    ml_dens = density_on_grid(X_malevy, gx, gy)
    err_d, _, _ = compute_errors(d_dens, pi, dx_grid, dy_grid)
    err_l, _, _ = compute_errors(l_dens, pi, dx_grid, dy_grid)
    err_m, _, _ = compute_errors(m_dens, pi, dx_grid, dy_grid)
    err_ml, _, _ = compute_errors(ml_dens, pi, dx_grid, dy_grid)
    
    vmax = max(np.max(err_d), np.max(err_l), np.max(err_m), np.max(err_ml))
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    im1 = ax[0].imshow(err_d, origin='lower', extent=[gx[0],gx[-1],gy[0],gy[-1]], cmap='hot', vmin=0, vmax=vmax)
    ax[0].set_title("Spatial Error: Diffusion"); fig.colorbar(im1, ax=ax[0])
    im2 = ax[1].imshow(err_l, origin='lower', extent=[gx[0],gx[-1],gy[0],gy[-1]], cmap='hot', vmin=0, vmax=vmax)
    ax[1].set_title("Spatial Error: Lévy PF"); fig.colorbar(im2, ax=ax[1])
    im3 = ax[2].imshow(err_m, origin='lower', extent=[gx[0],gx[-1],gy[0],gy[-1]], cmap='hot', vmin=0, vmax=vmax)
    ax[2].set_title("Spatial Error: MALA"); fig.colorbar(im3, ax=ax[2])
    im4 = ax[3].imshow(err_ml, origin='lower', extent=[gx[0],gx[-1],gy[0],gy[-1]], cmap='hot', vmin=0, vmax=vmax)
    ax[3].set_title("Spatial Error: MALA-Levy"); fig.colorbar(im4, ax=ax[3])
    plt.savefig(f"{out}_spatial_err.png", dpi=200)
    plt.close()

    avg_acc = float(np.mean(acc_rates)) if acc_rates else 0.0
    print("Complete. Generated convergence and spatial error plots.")
    print(f"MALA mean acceptance rate: {avg_acc:.3f}")
    avg_acc_ml = float(np.mean(acc_rates_ml)) if acc_rates_ml else 0.0
    print(f"MALA-Levy mean acceptance rate: {avg_acc_ml:.3f}")

if __name__ == "__main__":
    main()
