#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from high_dim_output.benchmark_metrics import (
    BENCHMARK_METHODS,
    compute_benchmark_metrics,
    get_default_benchmark_config,
    hypercube_mode_descriptors,
    init_benchmark_history,
    make_metric_rng,
    plot_benchmark_metrics_figure,
    save_benchmark_metadata_json,
    save_benchmark_metrics_csv,
)
from flmc_utils import flmc_self_check as shared_flmc_self_check, step_flmc_nd


EXPO_CLIP = 40.0
POT_CLIP = 12.0
STATE_CLIP = 25.0
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


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


def gradU_high_dim(X, sigma):
    """
    Paper-faithful FLMC potential for the 10D benchmark.

    Target:
        pi(x) ∝ exp(-2 V(x) / sigma^2)
    Therefore:
        U(x) = -log pi(x) = 2 V(x) / sigma^2 + const
        gradU(x) = (2 / sigma^2) gradV(x)
    and the FLMC drift is
        -c_alpha * gradU(x) = -(2 c_alpha / sigma^2) gradV(x).
    """
    return -grad_logpi_high_dim(X, sigma)


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
    FLMC step for the 10D benchmark following Simsekli et al., Section 3.3.

    We write the benchmark target as
        U(x) = 2 V(x) / sigma^2,
        pi(x) ∝ exp(-U(x)).
    The paper's simplified multidimensional FLA update is
        X_{n+1} = X_n - dt * c_alpha * ∇U(X_n) + dt^(1/alpha) * ξ_n
                = X_n - dt * (2 c_alpha / sigma^2) * ∇V(X_n) + dt^(1/alpha) * ξ_n,
    where ξ_n has iid coordinate-wise SαS(1) entries.

    IMPORTANT: The potential V(x) is treated as a general high-dimensional function,
    NOT exploiting separability even when V(x) = Σᵢ Vᵢ(xᵢ). This ensures fair
    comparison with ULA/MALA/LSBMC, which also treat V as non-separable.

    The multidimensional FLMC here follows the paper's independent-components
    setting. We intentionally do NOT use isotropic stable noise, because
    Section 3.3 does not cover isotropic stable processes.
    """
    X = sanitize_state(X)
    X_new = step_flmc_nd(
        x=X,
        dt=dt,
        alpha=alpha,
        gradU_fn=lambda arr: gradU_high_dim(arr, sigma),
        rng=rng,
        clip_bounds=(-STATE_CLIP, STATE_CLIP),
    )
    return sanitize_state(X_new)


def flmc_self_check(
    alpha: float = 2.0,
    dt: float = 1.0e-2,
    sigma: float = 1.3,
    dim: int = 10,
    n_samples: int = 20000,
    seed: int = 0,
    variance_rtol: float = 0.10,
):
    """
    Lightweight self-check for the paper-specific high-dimensional FLMC path.

    Verifies:
      1. For alpha = 2, dt^(1/2) * S2S(1) has variance 2 * dt per coordinate.
      2. The drift used by FLMC equals -(2 c_alpha / sigma^2) * gradV_high_dim(X).
    """
    X_probe = np.array(
        [
            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -0.75, 0.25, 0.9],
            [0.8, -1.2, 1.1, -0.4, 0.3, -0.9, 1.4, -1.6, 0.6, -0.2],
        ],
        dtype=float,
    )
    result = shared_flmc_self_check(
        gradU_fn=lambda arr: gradU_high_dim(arr, sigma),
        probe_points=X_probe,
        alpha=alpha,
        dt=dt,
        noise_dim=dim,
        n_noise_samples=n_samples,
        seed=seed,
        variance_rtol=variance_rtol,
    )
    result["sigma"] = float(sigma)
    return result


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
    benchmark_config=None,
    benchmark_mode_descriptors=None,
    return_benchmark_history=False,
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
    benchmark_metric_names = [
        "benchmark_sinkhorn_ot_cost",
        "benchmark_sinkhorn_divergence",
        "benchmark_mmd_squared",
        "benchmark_mmd",
    ]
    if benchmark_mode_descriptors is not None and benchmark_config is not None and bool(benchmark_config.get("emc_enabled", True)):
        benchmark_metric_names.append("benchmark_emc")
    benchmark_history = None
    benchmark_eval_steps = set()
    if benchmark_config is not None:
        benchmark_history = init_benchmark_history(benchmark_metric_names)
        benchmark_eval_steps = build_benchmark_eval_steps(
            steps,
            benchmark_config.get("benchmark_num_checkpoints", 7),
        )

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

            if benchmark_history is not None and i in benchmark_eval_steps:
                benchmark_history["t"].append(t)
                method_samples = {
                    "ula": X_diff,
                    "mala": X_mala,
                    "flmc": X_flmc,
                    "lsbmc": X_levy,
                }
                for method_idx, (slug, _, _) in enumerate(BENCHMARK_METHODS):
                    metric_rng = make_metric_rng(benchmark_config["metric_seed"], seed, i, method_idx)
                    values = compute_benchmark_metrics(
                        method_samples[slug],
                        X_ref,
                        benchmark_config,
                        metric_rng,
                        metric_prefix="benchmark",
                        mode_descriptors=benchmark_mode_descriptors,
                        context=f"high_dim seed={seed} t={t:.4f} method={slug}",
                    )
                    for metric_name in benchmark_metric_names:
                        benchmark_history[f"{metric_name}_{slug}"].append(
                            values.get(metric_name, float("nan"))
                        )

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
    if return_benchmark_history:
        return history, benchmark_history, acc_rate
    return history, acc_rate


def aggregate_histories(histories, keys):
    stacked = {k: np.stack([np.asarray(h[k], dtype=float) for h in histories], axis=0) for k in keys}
    mean = {k: stacked[k].mean(axis=0) for k in keys}
    std = {k: stacked[k].std(axis=0) for k in keys}
    return mean, std


def build_benchmark_eval_steps(steps, num_checkpoints):
    num_checkpoints = max(2, int(num_checkpoints))
    return set(np.unique(np.linspace(0, steps, num_checkpoints, dtype=int)).tolist())


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
    benchmark_config = get_default_benchmark_config({
        "benchmark_num_checkpoints": 21,
        "sinkhorn_epsilon": 1.0,
        "emc_enabled": True,
        "emc_max_modes": 2048,
    })
    benchmark_mode_descriptors = None
    emc_metadata = {
        "emc_applicable": True,
        "emc_enabled": False,
        "emc_reason": "Skipped by default for the high-dimensional target.",
    }
    mode_count = 1 << dim
    if bool(benchmark_config.get("emc_enabled", True)):
        if mode_count <= int(benchmark_config["emc_max_modes"]):
            benchmark_mode_descriptors = hypercube_mode_descriptors(dim)
            emc_metadata = {
                "emc_applicable": True,
                "emc_enabled": True,
                "mode_count": int(mode_count),
            }
        else:
            emc_metadata = {
                "emc_applicable": True,
                "emc_enabled": False,
                "mode_count": int(mode_count),
                "emc_reason": "Mode count exceeds emc_max_modes.",
            }

    rng_ref = np.random.default_rng(12345)
    print("Generating reference samples from pi(x) ∝ exp(-2V/sigma^2) ...")
    X_ref = sample_true_high_d(2500, dim, sigma, rng_ref)

    print(f"Running {num_seeds} seeds for {dim}D model with ULA / MALA / FLMC / LSBMC ...")
    histories = []
    benchmark_histories = []
    acc_rates = []
    for seed in seeds:
        h, benchmark_history, acc_rate = run_simulation(
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
            benchmark_config=benchmark_config,
            benchmark_mode_descriptors=benchmark_mode_descriptors,
            return_benchmark_history=True,
        )
        histories.append(h)
        benchmark_histories.append(benchmark_history)
        acc_rates.append(acc_rate)

    t = np.asarray(histories[0]["t"], dtype=float)
    keys = [
        "w2_d", "w2_m", "w2_f", "w2_l",
        "l1_d", "l1_m", "l1_f", "l1_l",
        "l2_d", "l2_m", "l2_f", "l2_l"
    ]
    mean, std = aggregate_histories(histories, keys)
    benchmark_metric_names = [
        "benchmark_sinkhorn_ot_cost",
        "benchmark_sinkhorn_divergence",
        "benchmark_mmd_squared",
        "benchmark_mmd",
    ]
    if benchmark_mode_descriptors is not None and bool(benchmark_config.get("emc_enabled", True)):
        benchmark_metric_names.append("benchmark_emc")
    benchmark_keys = [
        f"{metric_name}_{slug}"
        for metric_name in benchmark_metric_names
        for slug, _, _ in BENCHMARK_METHODS
    ]
    benchmark_mean, benchmark_std = aggregate_histories(benchmark_histories, benchmark_keys)
    benchmark_t = np.asarray(benchmark_histories[0]["t"], dtype=float)

    out_dir = os.path.join(THIS_DIR, "high_dim_output")
    os.makedirs(out_dir, exist_ok=True)
    out_prefix = os.path.join(out_dir, "high_dim")
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
    benchmark_base = os.path.join(out_dir, "benchmark_metrics_high_dim")
    benchmark_plot_metrics = ["benchmark_sinkhorn_divergence", "benchmark_mmd"]
    if benchmark_mode_descriptors is not None and bool(benchmark_config.get("emc_enabled", True)):
        benchmark_plot_metrics.append("benchmark_emc")
    plot_benchmark_metrics_figure(
        benchmark_t,
        benchmark_mean,
        benchmark_std,
        benchmark_plot_metrics,
        benchmark_base,
        "High-Dimensional Target: Benchmark Metrics",
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
        os.path.join(out_dir, "metrics_benchmark_high_dim.json"),
        "high_dim",
        benchmark_config,
        benchmark_metric_names,
        mode_descriptors=benchmark_mode_descriptors,
        extra_metadata={
            **emc_metadata,
            "benchmark_num_checkpoints": int(benchmark_t.size),
        },
    )

    avg_acc = float(np.mean(acc_rates)) if acc_rates else 0.0
    print(f"Done. Saved {out_prefix}_metrics.png and .pdf")
    print(f"Saved benchmark metrics to: {benchmark_base}.png/.pdf/.csv")
    print(f"MALA mean acceptance rate: {avg_acc:.3f}")


if __name__ == "__main__":
    main()
