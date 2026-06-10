#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D Mixture-of-40-Gaussians (MoG40) Benchmark

Convention: Global convention with sigma = sqrt(2).
    V(x)     = -log( sum_k exp(-||x - mu_k||^2 / 2) )
    Target   = exp(-2V / sigma^2) = exp(-V) = (1/K) sum_k N(x; mu_k, I_2)  (exact MoG40)

Mode centers: 40 points drawn from Uniform[-40, 40]^2 with seed=0 (fixed for reproducibility).
Component std: s = 1.

Samplers compared: ULA / MALA / FLMC / LSBMC
Metrics:
  - Sinkhorn divergence (vs exact reference)
  - MMD (vs exact reference)
  - Hard-assignment EMC: exp(H) / 40
  - Mode coverage: fraction of 40 modes hit
  - One combined 2x2 mode-occupancy figure (final step, last seed)
  - Single-panel scatter figures for the true reference and each sampler
"""

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from high_dim_output.benchmark_metrics import (
    BENCHMARK_METHODS,
    compute_benchmark_metrics,
    get_default_benchmark_config,
    init_benchmark_history,
    make_metric_rng,
    plot_benchmark_metrics_figure,
    save_figure_both,
    save_benchmark_metadata_json,
    save_benchmark_metrics_csv,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from flmc_utils import step_flmc_nd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPO_CLIP = 60.0   # clip log-ratio in score computation
S_CLIP    = 80.0   # clip score field magnitude
STATE_CLIP = 65.0  # hard state bound (modes go to ±40, buffer of 25)
K_MODES   = 40
SIGMA_COMP = 1.0   # component standard deviation

# ---------------------------------------------------------------------------
# Mode centres — deterministic, seed = 0, stored at module load
# ---------------------------------------------------------------------------
_rng_modes = np.random.default_rng(0)
MU_K = _rng_modes.uniform(-40.0, 40.0, size=(K_MODES, 2))  # (40, 2)


# ---------------------------------------------------------------------------
# Potential and gradient
# ---------------------------------------------------------------------------

def V_mog40(X):
    """
    MoG40 potential (log-sum-exp, numerically stable).

    V(x) = -log( sum_k exp(-||x - mu_k||^2 / (2 s^2)) )

    Args:
        X: (N, 2) array of particle positions
    Returns:
        V: (N,) array
    """
    X = np.asarray(X, dtype=float)
    diffs   = X[:, None, :] - MU_K[None, :, :]   # (N, K, 2)
    d2      = np.sum(diffs ** 2, axis=-1)          # (N, K)
    logw    = -0.5 * d2 / (SIGMA_COMP ** 2)        # (N, K)
    lw_max  = np.max(logw, axis=1, keepdims=True)
    log_sum = lw_max[:, 0] + np.log(
        np.sum(np.exp(logw - lw_max), axis=1) + 1e-300
    )
    return -log_sum  # (N,)


def gradV_mog40(X):
    """
    Gradient of MoG40 potential.

    gradV(x) = sum_k w_k(x) * (x - mu_k) / s^2
    where w_k(x) are softmax weights.

    Args:
        X: (N, 2) array
    Returns:
        grad: (N, 2) array
    """
    X = np.asarray(X, dtype=float)
    diffs   = X[:, None, :] - MU_K[None, :, :]   # (N, K, 2)
    d2      = np.sum(diffs ** 2, axis=-1)          # (N, K)
    logw    = -0.5 * d2 / (SIGMA_COMP ** 2)
    lw_max  = np.max(logw, axis=1, keepdims=True)
    w       = np.exp(logw - lw_max)               # unnormalized (N, K)
    w      /= np.sum(w, axis=1, keepdims=True) + 1e-300
    grad    = np.einsum("nk,nkd->nd", w, diffs) / (SIGMA_COMP ** 2)
    return grad  # (N, 2)


# ---------------------------------------------------------------------------
# Reference sampler (exact)
# ---------------------------------------------------------------------------

def sample_mog40_exact(rng, n_samples):
    """
    Exact sampling from MoG40.
      k ~ Uniform{0, ..., K-1}
      x | k ~ N(mu_k, s^2 I_2)

    Args:
        rng: NumPy random generator
        n_samples: int
    Returns:
        (n_samples, 2)
    """
    k       = rng.integers(0, K_MODES, size=n_samples)
    centers = MU_K[k]                                    # (N, 2)
    noise   = SIGMA_COMP * rng.standard_normal((n_samples, 2))
    return centers + noise


# ---------------------------------------------------------------------------
# Log-target and its gradient (for MALA)
# ---------------------------------------------------------------------------

def logpi_mog40(X, sigma):
    """log p_inf(x) = -2 V(x) / sigma^2."""
    return -2.0 * V_mog40(X) / (sigma ** 2)


def grad_logpi_mog40(X, sigma):
    """grad log p_inf(x) = -2 gradV(x) / sigma^2."""
    return -2.0 * gradV_mog40(X) / (sigma ** 2)


def gradU_mog40(X, sigma):
    """
    Paper-faithful FLMC mapping for the MoG40 benchmark.

    Target:
        pi(x) ∝ exp(-2 V(x) / sigma^2)
    Therefore:
        U(x) = -log pi(x) = 2 V(x) / sigma^2 + const
        gradU(x) = (2 / sigma^2) gradV(x)
    and the FLMC drift is
        -c_alpha * gradU(x) = -(2 c_alpha / sigma^2) gradV(x).
    """
    return -grad_logpi_mog40(X, sigma)


# ---------------------------------------------------------------------------
# State sanitisation
# ---------------------------------------------------------------------------

def _sanitize(X):
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=STATE_CLIP, neginf=-STATE_CLIP)
    return np.clip(X, -STATE_CLIP, STATE_CLIP)


def _tamed(drift, dt):
    norm = np.linalg.norm(drift, axis=1, keepdims=True)
    return (dt * drift) / (1.0 + dt * norm)


# ---------------------------------------------------------------------------
# Sampler steps
# ---------------------------------------------------------------------------

def step_ula_mog40(X, dt, sigma, rng):
    """Overdamped Langevin (ULA): dX = -gradV dt + sigma dB."""
    X  = _sanitize(X)
    dr = _tamed(-gradV_mog40(X), dt)
    return _sanitize(X + dr + sigma * math.sqrt(dt) * rng.standard_normal(X.shape))


def step_mala_mog40(X, dt, sigma, rng):
    """
    MALA for MoG40.
    Proposal: Y = X + 0.5 dt grad_logpi(X) + sqrt(dt) Z
    Returns (X_new, acceptance_rate).
    """
    X  = _sanitize(X)
    N  = X.shape[0]

    gl_x     = grad_logpi_mog40(X, sigma)
    mean_fwd = X + 0.5 * dt * gl_x
    prop     = mean_fwd + math.sqrt(dt) * rng.standard_normal(X.shape)
    prop     = _sanitize(prop)

    lp_x = logpi_mog40(X, sigma)
    lp_y = logpi_mog40(prop, sigma)

    gl_y     = grad_logpi_mog40(prop, sigma)
    mean_bwd = prop + 0.5 * dt * gl_y

    lq_fwd = -np.sum((prop - mean_fwd) ** 2, axis=1) / (2.0 * dt)
    lq_bwd = -np.sum((X    - mean_bwd) ** 2, axis=1) / (2.0 * dt)

    log_alpha = (lp_y + lq_bwd) - (lp_x + lq_fwd)
    accept    = np.log(rng.random(N) + 1e-300) < log_alpha

    X_new          = X.copy()
    X_new[accept]  = prop[accept]
    return _sanitize(X_new), float(accept.mean())


def step_flmc_mog40(X, dt, sigma, alpha, rng):
    """Paper-faithful FLMC for MoG40 with iid coordinate-wise SαS(1) noise."""
    X = _sanitize(X)
    return _sanitize(
        step_flmc_nd(
            x=X,
            dt=dt,
            alpha=alpha,
            gradU_fn=lambda arr: gradU_mog40(arr, sigma),
            rng=rng,
            clip_bounds=(-STATE_CLIP, STATE_CLIP),
        )
    )


# ---------------------------------------------------------------------------
# LSB-MC score and step
# ---------------------------------------------------------------------------

def _levy_score_mog40(X, sigma, lam, sigma_L, multipliers, pm, rng,
                      n_dir=4, n_theta=5):
    """
    Stationary Levy-score correction for MoG40 (on-the-fly, 2D isotropic).

    I(x) = lam * int_0^1 int r exp(-2(V(x-θr)-V(x))/σ²) ν(dr) dθ

    Uses random isotropic 2D directions with antithetic (+/-) pairing.
    Trapezoidal quadrature in θ.
    """
    X  = _sanitize(X)
    V0 = V_mog40(X)   # (N,)

    theta   = np.linspace(0.0, 1.0, n_theta)
    w_theta = np.ones(n_theta)
    w_theta[0] = 0.5; w_theta[-1] = 0.5
    w_theta /= w_theta.sum()

    # Random 2D unit directions
    raw  = rng.standard_normal((n_dir, 2))
    dirs = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12)

    integral = np.zeros_like(X)
    for mult, p_mult in zip(multipliers, pm):
        radius = sigma_L * mult
        for u in dirs:
            r   = radius * u          # 2D jump vector
            acc = np.zeros_like(X)
            for th, wth in zip(theta, w_theta):
                Xm = np.clip(X - th * r, -STATE_CLIP, STATE_CLIP)
                Xp = np.clip(X + th * r, -STATE_CLIP, STATE_CLIP)
                Vm = V_mog40(Xm)
                Vp = V_mog40(Xp)
                dlm = np.clip(-2.0 * (Vm - V0) / (sigma ** 2), -EXPO_CLIP, EXPO_CLIP)
                dlp = np.clip(-2.0 * (Vp - V0) / (sigma ** 2), -EXPO_CLIP, EXPO_CLIP)
                term = 0.5 * (np.exp(dlm) - np.exp(dlp))  # (N,)
                acc += wth * term[:, None] * r[None, :]    # (N, 2)
            integral += (p_mult / n_dir) * acc

    integral *= lam
    return np.clip(integral, -S_CLIP, S_CLIP)


def _sample_jumps_2d(rng, n_events, sigma_L, multipliers, pm):
    if n_events <= 0:
        return np.zeros((0, 2), dtype=float)
    mc   = rng.choice(len(multipliers), size=n_events, p=pm)
    dirs = rng.standard_normal((n_events, 2))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    return dirs * (sigma_L * multipliers[mc])[:, None]


def step_lsbmc_mog40(X, dt, sigma, rng, lam, sigma_L, multipliers, pm,
                     n_dir=4, n_theta=5):
    """
    LSB-MC step for MoG40.
    dZ = (-gradV + S_L^s) dt + sigma dB + dL  (compound Poisson)
    """
    X      = _sanitize(X)
    drift  = -gradV_mog40(X) - _levy_score_mog40(
        X, sigma, lam, sigma_L, multipliers, pm, rng, n_dir, n_theta)
    noise  = sigma * math.sqrt(dt) * rng.standard_normal(X.shape)
    X_new  = X + _tamed(drift, dt) + noise

    jump_counts = rng.poisson(lam * dt, size=X.shape[0])
    total_jumps = int(jump_counts.sum())
    if total_jumps > 0:
        rows  = np.repeat(np.arange(X.shape[0]), jump_counts)
        jumps = _sample_jumps_2d(rng, total_jumps, sigma_L, multipliers, pm)
        np.add.at(X_new, rows, jumps)

    return _sanitize(X_new)


# ---------------------------------------------------------------------------
# MoG40-specific metrics
# ---------------------------------------------------------------------------

def compute_hard_emc_mog40(X):
    """
    Hard nearest-mean EMC for MoG40.

    Assigns each sample to its nearest mode centre.
    EMC     = exp(H) / K   where H = -sum_k p_k log p_k
    Coverage = #{k : n_k > 0} / K

    Returns: (emc, coverage, counts_array)
    """
    X    = np.asarray(X, dtype=float)
    d2   = np.sum((X[:, None, :] - MU_K[None, :, :]) ** 2, axis=-1)  # (N, K)
    idx  = np.argmin(d2, axis=1)                                        # (N,)
    cnts = np.bincount(idx, minlength=K_MODES).astype(float)
    p    = cnts / X.shape[0]
    mask = p > 0.0
    H    = -np.sum(p[mask] * np.log(p[mask] + 1e-300))
    emc  = math.exp(H) / K_MODES
    cov  = float(np.sum(cnts > 0)) / K_MODES
    return float(emc), float(cov), cnts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_benchmark_steps(steps, num_checkpoints):
    num_checkpoints = max(2, int(num_checkpoints))
    return set(np.unique(np.linspace(0, steps, num_checkpoints, dtype=int)).tolist())


def _aggregate(histories, keys):
    stacked = {
        k: np.stack([np.asarray(h[k], dtype=float) for h in histories], axis=0)
        for k in keys
    }
    return (
        {k: stacked[k].mean(axis=0) for k in keys},
        {k: stacked[k].std(axis=0)  for k in keys},
    )


# ---------------------------------------------------------------------------
# Single-seed simulation
# ---------------------------------------------------------------------------

def run_seed(seed, sigma, dt, T, N, lam, sigma_L, multipliers, pm,
             X_ref, n_dir, n_theta, alpha, benchmark_config):
    rng = np.random.default_rng(seed)

    # All particles start near mode 0 (challenge: discover the other 39)
    X_init  = MU_K[0] + 0.5 * rng.standard_normal((N, 2))
    X_ula   = X_init.copy()
    X_mala  = X_init.copy()
    X_flmc  = X_init.copy()
    X_lsbmc = X_init.copy()

    steps          = int(T / dt)
    check_every    = max(1, steps // 40)
    bm_steps       = _build_benchmark_steps(steps, benchmark_config.get("benchmark_num_checkpoints", 9))

    bm_metric_names = [
        "benchmark_sinkhorn_ot_cost",
        "benchmark_sinkhorn_divergence",
        "benchmark_mmd_squared",
        "benchmark_mmd",
        "benchmark_hard_emc",
    ]
    emc_keys = ["benchmark_hard_emc", "benchmark_coverage"]

    history = {"t": []}
    for name in emc_keys:
        for slug, _, _ in BENCHMARK_METHODS:
            history[f"{name}_{slug}"] = []

    bm_history = init_benchmark_history(bm_metric_names)

    acc_sum = 0.0; acc_n = 0

    for i in range(steps + 1):
        t = i * dt

        if i % check_every == 0:
            history["t"].append(t)
            samp = {"ula": X_ula, "mala": X_mala, "flmc": X_flmc, "lsbmc": X_lsbmc}
            for slug, _, _ in BENCHMARK_METHODS:
                emc, cov, _ = compute_hard_emc_mog40(samp[slug])
                history[f"benchmark_hard_emc_{slug}"].append(emc)
                history[f"benchmark_coverage_{slug}"].append(cov)

        if i in bm_steps:
            bm_history["t"].append(t)
            samp = {"ula": X_ula, "mala": X_mala, "flmc": X_flmc, "lsbmc": X_lsbmc}
            for midx, (slug, _, _) in enumerate(BENCHMARK_METHODS):
                mrng   = make_metric_rng(benchmark_config["metric_seed"], seed, i, midx)
                values = compute_benchmark_metrics(
                    samp[slug], X_ref, benchmark_config, mrng,
                    metric_prefix="benchmark", mode_descriptors=None,
                    context=f"mog40 seed={seed} t={t:.3f} {slug}",
                )
                for name in bm_metric_names:
                    if name == "benchmark_hard_emc":
                        emc, _, _ = compute_hard_emc_mog40(samp[slug])
                        bm_history[f"{name}_{slug}"].append(emc)
                    else:
                        bm_history[f"{name}_{slug}"].append(values.get(name, float("nan")))

        if i % check_every == 0:
            emc_l = history["benchmark_hard_emc_lsbmc"][-1]
            cov_l = history["benchmark_coverage_lsbmc"][-1]
            print(f"  [seed {seed}] t={t:.2f}  EMC(LSBMC)={emc_l:.3f}  Cov(LSBMC)={cov_l:.2f}")

        if i < steps:
            X_ula              = step_ula_mog40(X_ula, dt, sigma, rng)
            X_mala, acc        = step_mala_mog40(X_mala, dt, sigma, rng)
            acc_sum           += acc; acc_n += 1
            X_flmc             = step_flmc_mog40(X_flmc, dt, sigma, alpha, rng)
            X_lsbmc            = step_lsbmc_mog40(
                X_lsbmc, dt, sigma, rng, lam, sigma_L, multipliers, pm, n_dir, n_theta)

    acc_rate = acc_sum / max(acc_n, 1)
    final    = {"ula": X_ula, "mala": X_mala, "flmc": X_flmc, "lsbmc": X_lsbmc}
    return history, bm_history, acc_rate, final


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_occupancy_grid(final_by_method, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (slug, label, color) in zip(axes, BENCHMARK_METHODS):
        _, _, counts = compute_hard_emc_mog40(final_by_method[slug])
        probs = counts / max(float(np.sum(counts)), 1.0)
        ax.bar(np.arange(K_MODES), probs, color=color, alpha=0.8, width=0.8)
        ax.axhline(1.0 / K_MODES, color="k", linestyle="--", linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel("Mode Index")
        ax.set_ylabel("Fraction")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("MoG40: Mode Occupancy (Final Step, Last Seed)")
    fig.tight_layout()
    save_figure_both(fig, os.path.join(out_dir, "mode_occupancy_mog40"))


def _plot_single_scatter(X_samp, title, slug, color, out_dir):
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter(X_samp[:, 0], X_samp[:, 1], s=2, c=color, alpha=0.5, rasterized=True)
    ax.scatter(
        MU_K[:, 0],
        MU_K[:, 1],
        marker="x",
        c="red",
        s=30,
        zorder=5,
        linewidths=1.2,
        label="Modes",
    )
    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 45)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(title)
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure_both(fig, os.path.join(out_dir, slug))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sigma     = math.sqrt(2.0)   # target = exp(-2V/2) = exp(-V) = MoG40 exactly
    dt        = 0.005
    T         = 20.0
    N         = 3000
    num_seeds = 3
    seeds     = list(range(num_seeds))

    lam         = 2.0
    sigma_L     = 3.0
    multipliers = np.array([1.0, 3.0, 6.0], dtype=float)
    pm          = np.array([0.50, 0.35, 0.15], dtype=float)
    pm         /= pm.sum()

    n_dir   = 4
    n_theta = 5
    alpha   = 1.5

    # Large epsilon because inter-mode squared distances are O(100-2500)
    bm_config = get_default_benchmark_config({
        "benchmark_num_checkpoints": 9,
        "sinkhorn_epsilon":          50.0,
        "sinkhorn_subsample_size":   512,
        "mmd_subsample_size":        512,
        "emc_enabled":               False,   # using hard EMC instead
    })

    rng_ref = np.random.default_rng(12345)
    print("Generating MoG40 reference samples ...")
    X_ref = sample_mog40_exact(rng_ref, 3000)

    out_dir = os.path.join(THIS_DIR, "mog40_output")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Running {num_seeds} seeds — ULA / MALA / FLMC / LSBMC ...")
    histories     = []
    bm_histories  = []
    acc_rates     = []
    final_samples = {slug: [] for slug, _, _ in BENCHMARK_METHODS}

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        h, bh, acc, final = run_seed(
            seed=seed, sigma=sigma, dt=dt, T=T, N=N,
            lam=lam, sigma_L=sigma_L, multipliers=multipliers, pm=pm,
            X_ref=X_ref, n_dir=n_dir, n_theta=n_theta, alpha=alpha,
            benchmark_config=bm_config,
        )
        histories.append(h)
        bm_histories.append(bh)
        acc_rates.append(acc)
        for slug, _, _ in BENCHMARK_METHODS:
            final_samples[slug].append(final[slug])

    t  = np.asarray(histories[0]["t"], dtype=float)
    bt = np.asarray(bm_histories[0]["t"], dtype=float)

    bm_metric_names = [
        "benchmark_sinkhorn_ot_cost",
        "benchmark_sinkhorn_divergence",
        "benchmark_mmd_squared",
        "benchmark_mmd",
        "benchmark_hard_emc",
    ]
    bm_keys = [f"{nm}_{slug}" for nm in bm_metric_names for slug, _, _ in BENCHMARK_METHODS]
    bm_mean, bm_std = _aggregate(bm_histories, bm_keys)

    bm_base = os.path.join(out_dir, "benchmark_metrics_mog40")
    plot_benchmark_metrics_figure(
        bt, bm_mean, bm_std,
        ["benchmark_sinkhorn_divergence", "benchmark_mmd", "benchmark_hard_emc"],
        bm_base,
        "MoG40: Benchmark Metrics",
        metric_labels={
            "benchmark_sinkhorn_divergence": "Sinkhorn",
            "benchmark_mmd": "MMD",
            "benchmark_hard_emc": "Hard EMC",
        },
    )
    save_benchmark_metrics_csv(f"{bm_base}.csv", bt, bm_mean, bm_std, bm_metric_names)
    save_benchmark_metadata_json(
        os.path.join(out_dir, "metrics_benchmark_mog40.json"),
        "mog40", bm_config, bm_metric_names,
        mode_descriptors=MU_K,
        extra_metadata={
            "k_modes":            K_MODES,
            "sigma_component":    SIGMA_COMP,
            "mode_source":        "seed=0, Uniform[-40,40]^2",
            "emc_type":           "hard_assignment_nearest_mean",
            "emc_formula":        "exp(H)/40, H = -sum p_k log p_k",
            "sigma_noise":        float(sigma),
            "dt":                 dt,
            "T":                  T,
            "N":                  N,
            "benchmark_num_checkpoints": int(bt.size),
        },
    )

    final_last_seed = {slug: final_samples[slug][-1] for slug, _, _ in BENCHMARK_METHODS}
    _plot_occupancy_grid(final_last_seed, out_dir)
    _plot_single_scatter(X_ref, "MoG40: True Scatter", "true_scatter", "gray", out_dir)
    for slug, label, color in BENCHMARK_METHODS:
        _plot_single_scatter(
            final_last_seed[slug],
            f"MoG40: {label} Scatter",
            f"{slug}_scatter",
            color,
            out_dir,
        )

    avg_acc = float(np.mean(acc_rates)) if acc_rates else 0.0
    print(f"\nDone. Output saved to {out_dir}")
    print(f"MALA mean acceptance rate: {avg_acc:.3f}")
    print(f"Mode centres file: stored in MU_K (seed=0).")


if __name__ == "__main__":
    main()
