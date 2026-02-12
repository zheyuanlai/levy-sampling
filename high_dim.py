#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

try:
    import ot  # POT for Wasserstein
    HAS_POT = True
except ImportError:
    HAS_POT = False
    print("Warning: POT not installed. W2 metric will be 0.0. (pip install POT)")

# ============================================================
# 1. 10D Separable Double Well Potential
# ============================================================
# V(x) = sum( (x_i^2 - 1)^2 )
# Invariant measure factorizes: pi(x) = prod( pi_1d(x_i) )

DIM = 10

def V_high_dim(X):
    return np.sum((X**2 - 1.0)**2, axis=1)

def gradV_high_dim(X):
    return 4.0 * X * (X**2 - 1.0)

def score_high_dim(X, eps):
    # Score of target pi(x) ∝ exp(-V/eps^2)
    return -gradV_high_dim(X) / (eps**2)

# ============================================================
# 2. Reference Sampler (Ground Truth)
# ============================================================
# Since dimensions are independent, we can sample 1D marginals 
# using Rejection Sampling and stack them.

def get_true_marginal_pdf(x_grid, eps):
    # pi_1d(x) propto exp( - (x^2-1)^2 / eps^2 )
    V = (x_grid**2 - 1.0)**2
    log_pi = -V / (eps**2)
    log_pi -= log_pi.max()
    pi = np.exp(log_pi)
    pi /= np.trapz(pi, x_grid)
    return pi

def sample_true_1d(n_samples, eps, rng):
    # Rejection sampling for 1D double well
    # Proposal: Uniform [-2.5, 2.5] (covers the bulk)
    samples = []
    while len(samples) < n_samples:
        batch_size = (n_samples - len(samples)) * 3
        prop = rng.uniform(-2.5, 2.5, batch_size)
        
        # Target unnormalized density
        log_target = -(prop**2 - 1.0)**2 / (eps**2)
        # We can just accept with prob exp(log_target) since max(log_target)=0
        accept_prob = np.exp(log_target)
        
        u = rng.random(batch_size)
        accepted = prop[u < accept_prob]
        samples.extend(accepted)
    
    return np.array(samples[:n_samples])

def sample_true_high_d(N, dim, eps, rng):
    # Stack independent 1D samples
    X = np.zeros((N, dim))
    for d in range(dim):
        X[:, d] = sample_true_1d(N, eps, rng)
    return X

# ============================================================
# 3. Metrics (W2, L1, L2)
# ============================================================

def compute_wasserstein(X, Y_ref, rng):
    """Compute W2 between X and Y_ref (subsampled for speed)"""
    if not HAS_POT: return 0.0
    
    # Subsample to 500 points to keep Sinkhorn fast
    n_sub = 500
    idx_x = rng.choice(X.shape[0], n_sub, replace=False)
    idx_y = rng.choice(Y_ref.shape[0], n_sub, replace=False)
    
    X_sub = X[idx_x]
    Y_sub = Y_ref[idx_y]
    
    # Cost matrix: Squared Euclidean distance
    M = ot.dist(X_sub, Y_sub, metric='sqeuclidean')
    
    # Sinkhorn W2 (regularized)
    # sqrt because sinkhorn2 returns squared W2
    val = ot.sinkhorn2([], [], M, reg=0.1, numItermax=2000)
    return np.sqrt(val)

def compute_marginal_errors(X, dim_idx, true_x, true_pdf):
    """Compute L1/L2 error on the marginal distribution of dimension `dim_idx`"""
    # 1. Estimate empirical density of X[:, dim_idx]
    # Use same bins as true_x grid
    hist, edges = np.histogram(X[:, dim_idx], bins=true_x, density=True)
    
    # true_pdf is evaluated at centers
    centers = (edges[:-1] + edges[1:]) / 2
    # Interpolate true_pdf to centers (or assume true_x was dense enough)
    # Let's assumes true_x closely matches the histogram bins for simplicity,
    # or just interpolate true PDF onto centers.
    ref_vals = np.interp(centers, true_x, true_pdf)
    
    diff = np.abs(hist - ref_vals)
    dx = edges[1] - edges[0]
    
    mae = np.sum(diff) * dx          # L1 = Integral |p - p_ref|
    rmse = np.sqrt(np.sum(diff**2) * dx) # L2 = Sqrt( Integral (p - p_ref)^2 )
    
    return mae, rmse

# ============================================================
# 4. Simulation Kernels
# ============================================================

def step_diff(X, dt, eps, rng):
    # Score-based drift: b = (eps^2 / 2) * grad log pi, pi ∝ exp(-V/eps^2)
    score = score_high_dim(X, eps)
    drift = 0.5 * (eps**2) * score
    
    # Euler-Maruyama: dX = b dt + eps dW
    noise = eps * np.sqrt(dt) * rng.standard_normal(X.shape)
    
    return X + drift * dt + noise

def step_levy(X, dt, eps, rng, lam, sigma_L):
    # 1. Diffusion Step (Score-based drift)
    score = score_high_dim(X, eps)
    drift = 0.5 * (eps**2) * score
    noise = eps * np.sqrt(dt) * rng.standard_normal(X.shape)
    X_new = X + drift * dt + noise
    
    # 2. Jump Step
    # Jump rate lambda
    n_samples = X.shape[0]
    # Poisson approximation for small dt
    jump_mask = rng.random(n_samples) < (lam * dt)
    n_jumps = np.sum(jump_mask)
    
    if n_jumps > 0:
        # Isotropic jumps in 10D
        direction = rng.standard_normal((n_jumps, DIM))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        
        # Jump size: Fixed large jumps to cross barrier (width ~ 2)
        # 2.5 is enough to jump from -1 to > 1
        jumps = direction * sigma_L
        
        X_new[jump_mask] += jumps
        
    return X_new

# ============================================================
# 5. Main Execution
# ============================================================

def run_simulation(seed, eps, dt, T, N, dim, lam, sigma_L, X_ref, grid_x, pdf_true):
    rng = np.random.default_rng(seed)

    # Initialization: Trapped in Left Well (-1)
    X_diff = -1.0 + 0.1 * rng.standard_normal((N, dim))
    X_levy = -1.0 + 0.1 * rng.standard_normal((N, dim))

    steps = int(T / dt)
    check_interval = steps // 40  # Check 40 times total

    history = {
        't': [],
        'w2_d': [], 'w2_l': [],
        'l1_d': [], 'l1_l': [],
        'l2_d': [], 'l2_l': []
    }

    for i in range(steps + 1):
        if i % check_interval == 0:
            t = i * dt
            history['t'].append(t)

            # 1. W2 Metric (Full High-D)
            w2_d = compute_wasserstein(X_diff, X_ref, rng)
            w2_l = compute_wasserstein(X_levy, X_ref, rng)

            # 2. Marginal Metrics (Dim 0)
            l1_d, l2_d = compute_marginal_errors(X_diff, 0, grid_x, pdf_true)
            l1_l, l2_l = compute_marginal_errors(X_levy, 0, grid_x, pdf_true)

            history['w2_d'].append(w2_d); history['w2_l'].append(w2_l)
            history['l1_d'].append(l1_d); history['l1_l'].append(l1_l)
            history['l2_d'].append(l2_d); history['l2_l'].append(l2_l)

            print(f"[seed {seed}] t={t:.1f} | W2(D):{w2_d:.2f} W2(L):{w2_l:.2f} | L1(D):{l1_d:.2f} L1(L):{l1_l:.2f}")

        # Steps
        X_diff = step_diff(X_diff, dt, eps, rng)
        X_levy = step_levy(X_levy, dt, eps, rng, lam, sigma_L)

    return history, X_diff, X_levy

def aggregate_histories(histories, keys):
    stacked = {k: np.stack([np.array(h[k]) for h in histories], axis=0) for k in keys}
    mean = {k: stacked[k].mean(axis=0) for k in keys}
    std = {k: stacked[k].std(axis=0) for k in keys}
    return mean, std

def main():
    eps = 0.75
    dt = 0.005
    T = 20.0
    N = 20000
    DIM = 10
    num_seeds = 5
    seeds = list(range(num_seeds))
    
    # Levy Params
    lam = 0.8       # Frequent enough jumps
    sigma_L = 2.5   # Distance between wells is 2.0 (-1 to 1). 2.5 ensures crossing.

    rng_ref = np.random.default_rng(12345)
    print("Generating Reference Samples (Ground Truth)...")
    X_ref = sample_true_high_d(2000, DIM, eps, rng_ref)
    
    # Precompute 1D Marginal PDF for Grid Metrics
    grid_x = np.linspace(-3, 3, 100)
    pdf_true = get_true_marginal_pdf(grid_x, eps)

    print(f"Simulating {num_seeds} seeds for 10D System (eps={eps} -> High Barrier)...")
    histories = []
    first_final = None
    for seed in seeds:
        history, X_diff, X_levy = run_simulation(
            seed, eps, dt, T, N, DIM, lam, sigma_L, X_ref, grid_x, pdf_true
        )
        histories.append(history)
        if first_final is None:
            first_final = (X_diff, X_levy)

    t = np.array(histories[0]['t'])
    keys = ['w2_d', 'w2_l', 'l1_d', 'l1_l', 'l2_d', 'l2_l']
    mean, std = aggregate_histories(histories, keys)

    # ============================================================
    # 6. Plotting
    # ============================================================
    out_prefix = "high_dim"
    
    # Plot 1: Metrics Convergence
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # W2
    axes[0].errorbar(t, mean['w2_d'], yerr=std['w2_d'], fmt='b--', label='Diffusion', capsize=2, alpha=0.9)
    axes[0].errorbar(t, mean['w2_l'], yerr=std['w2_l'], fmt='r-', label='Lévy PF', capsize=2, alpha=0.9)
    axes[0].set_title(f"Wasserstein-2 (Full {DIM}D)")
    axes[0].set_xlabel("Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # L1
    axes[1].errorbar(t, mean['l1_d'], yerr=std['l1_d'], fmt='b--', label='Diffusion', capsize=2, alpha=0.9)
    axes[1].errorbar(t, mean['l1_l'], yerr=std['l1_l'], fmt='r-', label='Lévy PF', capsize=2, alpha=0.9)
    axes[1].set_title("Marginal L1 Error (MAE)")
    axes[1].set_xlabel("Time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # L2
    axes[2].errorbar(t, mean['l2_d'], yerr=std['l2_d'], fmt='b--', label='Diffusion', capsize=2, alpha=0.9)
    axes[2].errorbar(t, mean['l2_l'], yerr=std['l2_l'], fmt='r-', label='Lévy PF', capsize=2, alpha=0.9)
    axes[2].set_title("Marginal L2 Error (RMSE)")
    axes[2].set_xlabel("Time")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_metrics.png", dpi=200)
    plt.close()
    
    # Plot 2: Final Marginal Density Comparison
    X_diff, X_levy = first_final
    plt.figure(figsize=(8, 6))
    plt.hist(X_diff[:, 0], bins=50, density=True, alpha=0.5, color='blue', label='Diffusion (Trapped)')
    plt.hist(X_levy[:, 0], bins=50, density=True, alpha=0.5, color='red', label='Lévy PF (Mixed)')
    plt.plot(grid_x, pdf_true, 'k--', linewidth=2, label='True Marginal')
    plt.title(f"Final Marginal Density (Dim 0) @ T={T}")
    plt.xlabel("x_0")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{out_prefix}_density.png", dpi=200)
    plt.close()
    
    print(f"Done. Saved {out_prefix}_metrics.png and {out_prefix}_density.png")

if __name__ == "__main__":
    main()
