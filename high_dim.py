#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from flmc_utils import step_flmc_nd


EXPO_CLIP = 40.0
POT_CLIP = 12.0
STATE_CLIP = 25.0


# ============================================================
# 1. 10D Separable Double-Well Potential (paper Sec. 4.6)
# ============================================================
# V(x) = sum_i (x_i^2 - 1)^2
# Diffusion baseline: dX_t = -∇V(X_t) dt + sigma dB_t
# Target invariant: pi(x) ∝ exp( -2 V(x) / sigma^2 )


def V_high_dim(X):
    X = np.asarray(X, dtype=float)
    Xc = np.clip(X, -POT_CLIP, POT_CLIP)
    return np.sum((Xc * Xc - 1.0) ** 2, axis=1)


def gradV_high_dim(X):
    X = np.asarray(X, dtype=float)
    Xc = np.clip(X, -POT_CLIP, POT_CLIP)
    return 4.0 * Xc * (Xc * Xc - 1.0)


def sanitize_state(X):
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=STATE_CLIP, neginf=-STATE_CLIP)
    return np.clip(X, -STATE_CLIP, STATE_CLIP)


def tamed_drift_increment(drift, dt):
    norm = np.linalg.norm(drift, axis=1, keepdims=True)
    return (dt * drift) / (1.0 + dt * norm)


# ============================================================
# 2. Reference Sampler (exact product target)
# ============================================================


def sample_true_1d(n_samples, sigma, rng):
    """Rejection sampler for 1D target pi_1d(x) ∝ exp(-2 (x^2-1)^2 / sigma^2)."""
    samples = []
    while len(samples) < n_samples:
        batch_size = max((n_samples - len(samples)) * 3, 64)
        prop = rng.uniform(-2.5, 2.5, batch_size)
        log_target = -2.0 * (prop**2 - 1.0) ** 2 / (sigma**2)
        accept_prob = np.exp(np.clip(log_target, -EXPO_CLIP, 0.0))
        accepted = prop[rng.random(batch_size) < accept_prob]
        samples.extend(accepted.tolist())
    return np.asarray(samples[:n_samples], dtype=float)


def sample_true_high_d(n_samples, dim, sigma, rng):
    """Exact sampling from the dD product target by stacking independent 1D draws."""
    X = np.zeros((n_samples, dim), dtype=float)
    for d in range(dim):
        X[:, d] = sample_true_1d(n_samples, sigma, rng)
    return X


# ============================================================
# 3. High-dimensional Metrics
# ============================================================


def _wasserstein2_1d_exact(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size <= 1 or y.size <= 1:
        return 0.0

    x = np.sort(x)
    y = np.sort(y)
    n = max(x.size, y.size)
    q = (np.arange(n, dtype=float) + 0.5) / n
    qx = (np.arange(x.size, dtype=float) + 0.5) / x.size
    qy = (np.arange(y.size, dtype=float) + 0.5) / y.size
    xq = np.interp(q, qx, x)
    yq = np.interp(q, qy, y)
    return float(np.sqrt(np.mean((xq - yq) ** 2)))


def compute_wasserstein(X, Y_ref, rng, n_proj=64, max_points=700):
    """
    Sliced W2 in full d dimensions.
    This avoids external OT dependencies and remains fully high-dimensional.
    """
    X = np.asarray(X, dtype=float)
    Y_ref = np.asarray(Y_ref, dtype=float)
    if X.ndim != 2 or Y_ref.ndim != 2:
        return 0.0
    if X.shape[0] <= 1 or Y_ref.shape[0] <= 1 or X.shape[1] != Y_ref.shape[1]:
        return 0.0

    m = min(max_points, X.shape[0], Y_ref.shape[0])
    if X.shape[0] > m:
        X = X[rng.choice(X.shape[0], size=m, replace=False)]
    if Y_ref.shape[0] > m:
        Y_ref = Y_ref[rng.choice(Y_ref.shape[0], size=m, replace=False)]

    dim = X.shape[1]
    dirs = rng.standard_normal((n_proj, dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12

    vals_sq = []
    for u in dirs:
        px = X @ u
        py = Y_ref @ u
        w1d = _wasserstein2_1d_exact(px, py)
        vals_sq.append(w1d * w1d)
    return float(np.sqrt(np.mean(vals_sq)))


def compute_orthant_errors(X):
    """
    High-dimensional L1/L2 errors on orthant occupancy.
    For this symmetric target, each orthant has true mass 1 / 2^d.
    """
    X = sanitize_state(X)
    n_samples, dim = X.shape
    n_orthants = 1 << dim
    bits = (X >= 0.0).astype(np.uint64)
    weights = (1 << np.arange(dim, dtype=np.uint64))
    codes = (bits * weights).sum(axis=1).astype(np.int64)
    hist = np.bincount(codes, minlength=n_orthants).astype(float) / float(n_samples)

    true_mass = 1.0 / n_orthants
    diff = hist - true_mass
    l1 = np.sum(np.abs(diff))
    l2 = np.sqrt(np.sum(diff**2))
    return float(l1), float(l2)


# ============================================================
# 4. Jump Law + Stationary Lévy-score Integral
# ============================================================


def sample_jump_vectors(rng, n_events, dim, sigma_L, multipliers, pm):
    """Sample jump vectors Ai ~ nu_J in R^d."""
    if n_events <= 0:
        return np.zeros((0, dim), dtype=float)

    mult_choice = rng.choice(multipliers, size=n_events, p=pm)
    dirs = rng.standard_normal((n_events, dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    radii = sigma_L * mult_choice
    return dirs * radii[:, None]


def levy_score_integral_stationary(
    X,
    sigma,
    lam,
    sigma_L,
    multipliers,
    pm,
    rng,
    n_dir_score=8,
    n_theta=7,
    expo_clip=EXPO_CLIP,
    s_clip=None,
):
    """
    Approximate I(x) = lam * int_0^1 int r * exp(-2(V(x-theta r)-V(x))/sigma^2) nu_J(dr) dtheta
    using:
      - trapezoidal quadrature in theta
      - Monte Carlo over isotropic directions with antithetic +/- pairing.

    The manuscript drift is: -gradV(x) - I(x).
    """
    X = sanitize_state(X)
    n_samples, dim = X.shape
    if n_samples == 0:
        return np.zeros_like(X)

    X_eval = np.clip(X, -POT_CLIP, POT_CLIP)
    V0 = V_high_dim(X_eval)

    theta = np.linspace(0.0, 1.0, n_theta)
    w_theta = np.ones_like(theta)
    w_theta[0] = 0.5
    w_theta[-1] = 0.5
    w_theta /= np.sum(w_theta)

    dirs = rng.standard_normal((n_dir_score, dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12

    integral = np.zeros_like(X)
    for mult, p_mult in zip(multipliers, pm):
        radius = sigma_L * mult
        for u in dirs:
            r = radius * u
            acc = np.zeros_like(X)
            for th, wth in zip(theta, w_theta):
                Xm = np.clip(X_eval - th * r, -POT_CLIP, POT_CLIP)
                Xp = np.clip(X_eval + th * r, -POT_CLIP, POT_CLIP)

                Vm = V_high_dim(Xm)
                Vp = V_high_dim(Xp)

                dlog_m = -2.0 * (Vm - V0) / (sigma**2)
                dlog_p = -2.0 * (Vp - V0) / (sigma**2)
                dlog_m = np.nan_to_num(dlog_m, nan=-expo_clip, posinf=expo_clip, neginf=-expo_clip)
                dlog_p = np.nan_to_num(dlog_p, nan=-expo_clip, posinf=expo_clip, neginf=-expo_clip)
                rm = np.exp(np.clip(dlog_m, -expo_clip, expo_clip))
                rp = np.exp(np.clip(dlog_p, -expo_clip, expo_clip))

                # Symmetric pair (+/-r): 0.5 * r * (ratio_minus - ratio_plus)
                term = 0.5 * (rm - rp)
                acc += wth * term[:, None] * r[None, :]

            integral += (p_mult / n_dir_score) * acc

    integral *= lam
    if s_clip is not None:
        integral = np.clip(integral, -s_clip, s_clip)
    return integral


# ============================================================
# 5. Simulation Kernels
# ============================================================


def step_diff(X, dt, sigma, rng):
    """Paper diffusion baseline (ULA): dX = -gradV dt + sigma dB."""
    X = sanitize_state(X)
    drift = -gradV_high_dim(X)
    noise = sigma * np.sqrt(dt) * rng.standard_normal(X.shape)
    X_new = X + tamed_drift_increment(drift, dt) + noise
    return sanitize_state(X_new)


def logpi_high_dim(X, sigma):
    """Log target density: log p_∞(x) = -2V(x)/σ² + const."""
    V = V_high_dim(X)
    return -2.0 * V / (sigma ** 2)


def grad_logpi_high_dim(X, sigma):
    """Gradient of log target: ∇ log p_∞(x) = -2∇V(x)/σ²."""
    return -2.0 * gradV_high_dim(X) / (sigma ** 2)


def step_mala(X, dt, sigma, rng):
    """
    MALA step for high-dimensional target p_∞(x) ∝ exp(-2V(x)/σ²).

    Proposal: Y = X + 0.5 * dt * grad_logpi(X) + √dt * Z
    Accept with MH ratio based on target density and proposal kernels.
    """
    X = sanitize_state(X)
    n_samples, dim = X.shape

    # Forward proposal
    grad_x = grad_logpi_high_dim(X, sigma)
    mean_fwd = X + 0.5 * dt * grad_x
    proposal = mean_fwd + np.sqrt(dt) * rng.standard_normal(X.shape)
    proposal = sanitize_state(proposal)

    # Log target densities
    logp_x = logpi_high_dim(X, sigma)
    logp_y = logpi_high_dim(proposal, sigma)

    # Backward proposal kernel
    grad_y = grad_logpi_high_dim(proposal, sigma)
    mean_bwd = proposal + 0.5 * dt * grad_y

    # Log proposal densities (Gaussian kernels)
    diff_fwd = proposal - mean_fwd
    diff_bwd = X - mean_bwd
    log_q_fwd = -np.sum(diff_fwd ** 2, axis=1) / (2.0 * dt)
    log_q_bwd = -np.sum(diff_bwd ** 2, axis=1) / (2.0 * dt)

    # MH acceptance
    log_alpha = (logp_y + log_q_bwd) - (logp_x + log_q_fwd)
    log_u = np.log(rng.random(n_samples))
    accept = log_u < log_alpha

    X_new = X.copy()
    X_new[accept] = proposal[accept]

    acc_rate = float(accept.mean())
    return sanitize_state(X_new), acc_rate


def step_flmc(X, dt, sigma, alpha, rng):
    """
    FLMC step for high-dimensional potential.

    FLMC: dX = -c_alpha * ∇V dt + sigma * dt^(1/alpha) * Z
    where Z is GENUINELY ISOTROPIC alpha-stable noise (no score correction).

    IMPORTANT: The potential V(x) is treated as a general high-dimensional function,
    NOT exploiting separability even when V(x) = Σᵢ Vᵢ(xᵢ). This ensures fair
    comparison with ULA/MALA/LSBMC, which also treat V as non-separable.

    The isotropic noise structure Z = R * U uses:
        - U ~ Uniform(S^(d-1)): random direction on unit sphere
        - R ~ S_alpha^(1/alpha): alpha-stable radial component

    This matches the rotational invariance of LSBMC's isotropic jumps.
    The sigma scaling ensures FLMC targets p_∞ ∝ exp(-2V/sigma²), matching ULA/MALA/LSBMC.
    """
    X = sanitize_state(X)
    X_new = step_flmc_nd(
        x=X,
        dt=dt,
        alpha=alpha,
        sigma=sigma,
        gradV_fn=gradV_high_dim,
        rng=rng,
        clip_bounds=(-STATE_CLIP, STATE_CLIP)
    )
    return sanitize_state(X_new)


def step_levy(
    X,
    dt,
    sigma,
    rng,
    lam,
    sigma_L,
    multipliers,
    pm,
    n_dir_score=8,
    n_theta=7,
):
    """Algorithm 1 style step with Lévy-score-corrected drift + compound-Poisson jumps."""
    X = sanitize_state(X)
    drift_base = -gradV_high_dim(X)
    score_int = levy_score_integral_stationary(
        X,
        sigma=sigma,
        lam=lam,
        sigma_L=sigma_L,
        multipliers=multipliers,
        pm=pm,
        rng=rng,
        n_dir_score=n_dir_score,
        n_theta=n_theta,
    )
    drift = drift_base - score_int
    noise = sigma * np.sqrt(dt) * rng.standard_normal(X.shape)
    X_new = X + tamed_drift_increment(drift, dt) + noise

    n_samples, dim = X.shape
    jump_counts = rng.poisson(lam * dt, size=n_samples)
    total_jumps = int(np.sum(jump_counts))
    if total_jumps > 0:
        rows = np.repeat(np.arange(n_samples), jump_counts)
        jumps = sample_jump_vectors(
            rng,
            n_events=total_jumps,
            dim=dim,
            sigma_L=sigma_L,
            multipliers=multipliers,
            pm=pm,
        )
        np.add.at(X_new, rows, jumps)

    return sanitize_state(X_new)


# ============================================================
# 6. Run + Aggregate + Plot
# ============================================================


def run_simulation(
    seed,
    sigma,
    dt,
    T,
    N,
    dim,
    lam,
    sigma_L,
    multipliers,
    pm,
    X_ref,
    n_dir_score,
    n_theta,
    alpha,
    mala_dt=None,
):
    rng = np.random.default_rng(seed)

    # Left-well initialization in all coordinates.
    X_diff = -1.0 + 0.1 * rng.standard_normal((N, dim))
    X_mala = X_diff.copy()
    X_flmc = X_diff.copy()
    X_levy = X_diff.copy()

    if mala_dt is None:
        mala_dt = dt

    steps = int(T / dt)
    check_interval = max(1, steps // 40)

    history = {
        "t": [],
        "w2_d": [], "w2_m": [], "w2_f": [], "w2_l": [],
        "l1_d": [], "l1_m": [], "l1_f": [], "l1_l": [],
        "l2_d": [], "l2_m": [], "l2_f": [], "l2_l": []
    }

    acc_sum = 0.0
    acc_count = 0

    for i in range(steps + 1):
        if i % check_interval == 0:
            t = i * dt
            history["t"].append(t)

            w2_d = compute_wasserstein(X_diff, X_ref, rng)
            w2_m = compute_wasserstein(X_mala, X_ref, rng)
            w2_f = compute_wasserstein(X_flmc, X_ref, rng)
            w2_l = compute_wasserstein(X_levy, X_ref, rng)

            l1_d, l2_d = compute_orthant_errors(X_diff)
            l1_m, l2_m = compute_orthant_errors(X_mala)
            l1_f, l2_f = compute_orthant_errors(X_flmc)
            l1_l, l2_l = compute_orthant_errors(X_levy)

            history["w2_d"].append(w2_d); history["w2_m"].append(w2_m)
            history["w2_f"].append(w2_f); history["w2_l"].append(w2_l)
            history["l1_d"].append(l1_d); history["l1_m"].append(l1_m)
            history["l1_f"].append(l1_f); history["l1_l"].append(l1_l)
            history["l2_d"].append(l2_d); history["l2_m"].append(l2_m)
            history["l2_f"].append(l2_f); history["l2_l"].append(l2_l)

            print(
                f"[seed {seed}] t={t:.2f} | "
                f"W2: D:{w2_d:.3f} M:{w2_m:.3f} F:{w2_f:.3f} L:{w2_l:.3f} | "
                f"Orth-L1: D:{l1_d:.3f} M:{l1_m:.3f} F:{l1_f:.3f} L:{l1_l:.3f}"
            )

        X_diff = step_diff(X_diff, dt, sigma, rng)
        X_mala, acc = step_mala(X_mala, mala_dt, sigma, rng)
        acc_sum += acc
        acc_count += 1
        X_flmc = step_flmc(X_flmc, dt, sigma, alpha, rng)
        X_levy = step_levy(
            X_levy,
            dt,
            sigma,
            rng,
            lam=lam,
            sigma_L=sigma_L,
            multipliers=multipliers,
            pm=pm,
            n_dir_score=n_dir_score,
            n_theta=n_theta,
        )

    acc_rate = acc_sum / max(acc_count, 1)
    return history, acc_rate


def aggregate_histories(histories, keys):
    stacked = {k: np.stack([np.asarray(h[k], dtype=float) for h in histories], axis=0) for k in keys}
    mean = {k: stacked[k].mean(axis=0) for k in keys}
    std = {k: stacked[k].std(axis=0) for k in keys}
    return mean, std


def main():
    sigma = 0.75
    dt = 0.005
    T = 10.0
    N = 20000
    dim = 10
    num_seeds = 3
    seeds = list(range(num_seeds))

    # Compound-Poisson jump law nu_J: isotropic direction + discrete radial multipliers.
    lam = 0.8
    sigma_L = 1.0
    multipliers = np.asarray([1.0, 1.8, 2.6], dtype=float)
    pm = np.asarray([0.70, 0.22, 0.08], dtype=float)
    pm /= pm.sum()

    # Numerical quadrature controls for LSBMC score.
    n_dir_score = 8
    n_theta = 7

    # FLMC parameter
    alpha = 1.75

    rng_ref = np.random.default_rng(12345)
    print("Generating reference samples from pi(x) ∝ exp(-2V/sigma^2) ...")
    X_ref = sample_true_high_d(2500, dim, sigma, rng_ref)

    print(f"Running {num_seeds} seeds for {dim}D model with ULA / MALA / FLMC / LSBMC ...")
    histories = []
    acc_rates = []
    for seed in seeds:
        h, acc_rate = run_simulation(
            seed=seed,
            sigma=sigma,
            dt=dt,
            T=T,
            N=N,
            dim=dim,
            lam=lam,
            sigma_L=sigma_L,
            multipliers=multipliers,
            pm=pm,
            X_ref=X_ref,
            n_dir_score=n_dir_score,
            n_theta=n_theta,
            alpha=alpha,
        )
        histories.append(h)
        acc_rates.append(acc_rate)

    t = np.asarray(histories[0]["t"], dtype=float)
    keys = [
        "w2_d", "w2_m", "w2_f", "w2_l",
        "l1_d", "l1_m", "l1_f", "l1_l",
        "l2_d", "l2_m", "l2_f", "l2_l"
    ]
    mean, std = aggregate_histories(histories, keys)

    out_prefix = "high_dim"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Sliced W2
    axes[0].errorbar(t, mean["w2_d"], yerr=std["w2_d"], fmt="-", color="C0", label="ULA", capsize=2, alpha=0.9)
    axes[0].errorbar(t, mean["w2_m"], yerr=std["w2_m"], fmt="-", color="C2", label="MALA", capsize=2, alpha=0.9)
    axes[0].errorbar(t, mean["w2_f"], yerr=std["w2_f"], fmt="-", color="tab:orange", label="FLMC", capsize=2, alpha=0.9)
    axes[0].errorbar(t, mean["w2_l"], yerr=std["w2_l"], fmt="-", color="C3", label="LSBMC", capsize=2, alpha=0.9)
    axes[0].set_title(f"Sliced W2")
    axes[0].set_xlabel("Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Orthant L1
    axes[1].errorbar(t, mean["l1_d"], yerr=std["l1_d"], fmt="-", color="C0", label="ULA", capsize=2, alpha=0.9)
    axes[1].errorbar(t, mean["l1_m"], yerr=std["l1_m"], fmt="-", color="C2", label="MALA", capsize=2, alpha=0.9)
    axes[1].errorbar(t, mean["l1_f"], yerr=std["l1_f"], fmt="-", color="tab:orange", label="FLMC", capsize=2, alpha=0.9)
    axes[1].errorbar(t, mean["l1_l"], yerr=std["l1_l"], fmt="-", color="C3", label="LSBMC", capsize=2, alpha=0.9)
    axes[1].set_title("Orthant L1 Error")
    axes[1].set_xlabel("Time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Orthant L2
    axes[2].errorbar(t, mean["l2_d"], yerr=std["l2_d"], fmt="-", color="C0", label="ULA", capsize=2, alpha=0.9)
    axes[2].errorbar(t, mean["l2_m"], yerr=std["l2_m"], fmt="-", color="C2", label="MALA", capsize=2, alpha=0.9)
    axes[2].errorbar(t, mean["l2_f"], yerr=std["l2_f"], fmt="-", color="tab:orange", label="FLMC", capsize=2, alpha=0.9)
    axes[2].errorbar(t, mean["l2_l"], yerr=std["l2_l"], fmt="-", color="C3", label="LSBMC", capsize=2, alpha=0.9)
    axes[2].set_title("Orthant L2 Error")
    axes[2].set_xlabel("Time")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_metrics.png", dpi=200)
    plt.savefig(f"{out_prefix}_metrics.pdf")
    plt.close()

    avg_acc = float(np.mean(acc_rates)) if acc_rates else 0.0
    print(f"Done. Saved {out_prefix}_metrics.png and .pdf")
    print(f"MALA mean acceptance rate: {avg_acc:.3f}")


if __name__ == "__main__":
    main()
