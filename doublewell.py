#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ginzburg-Landau Double-Well Potential (1D)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from high_dim_output.benchmark_metrics import (
    BENCHMARK_METHODS,
    clip_samples_to_box,
    compute_benchmark_metrics,
    get_default_benchmark_config,
    init_benchmark_history,
    make_metric_rng,
    plot_benchmark_metrics_figure,
    save_benchmark_metadata_json,
    save_benchmark_metrics_csv,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from flmc_utils import step_flmc_1d


# Clipping constants for numerical stability
EXPO_CLIP = 60.0
S_CLIP = 80.0


# ============================================================
# Potential and Target Density
# ============================================================

def V_doublewell(x: np.ndarray) -> np.ndarray:
    """
    Ginzburg-Landau double-well potential.
    V(x) = (1/4)x^4 - (1/2)x^2

    Two wells at x = ±1, barrier at x = 0.
    """
    x = np.asarray(x, dtype=float)
    return 0.25 * x**4 - 0.5 * x**2


def gradV_doublewell(x: np.ndarray) -> np.ndarray:
    """Gradient of double-well potential: ∇V(x) = x^3 - x."""
    x = np.asarray(x, dtype=float)
    return x**3 - x


def logpi_doublewell(x: np.ndarray, sigma: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return -2.0 * V_doublewell(x) / (sigma ** 2)


def grad_logpi_doublewell(x: np.ndarray, sigma: float) -> np.ndarray:
    return -2.0 * gradV_doublewell(x) / (sigma ** 2)


def gradU_doublewell(x: np.ndarray, sigma: float) -> np.ndarray:
    """
    Paper-faithful FLMC mapping for the double-well benchmark.

    Target:
        pi(x) ∝ exp(-2 V(x) / sigma^2)
    Therefore:
        U(x) = -log pi(x) = 2 V(x) / sigma^2 + const
        gradU(x) = (2 / sigma^2) gradV(x)
    and the FLMC drift is
        -c_alpha * gradU(x) = -(2 c_alpha / sigma^2) gradV(x).
    """
    return -grad_logpi_doublewell(x, sigma)


def compute_target_density_1d(gx: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute target Boltzmann density on grid.

    Target: p_∞(x) ∝ exp(-2V(x)/σ²)

    Args:
        gx: Grid points (1D array)
        sigma: Noise scale

    Returns:
        Normalized density on grid
    """
    V = V_doublewell(gx)
    logpi = -2.0 * V / (sigma ** 2)
    logpi -= np.max(logpi)
    pi = np.exp(np.clip(logpi, -EXPO_CLIP, 0.0))

    dx = gx[1] - gx[0]
    pi /= (np.sum(pi) * dx + 1e-300)

    return pi


# ============================================================
# LSB-MC: Levy Score Precomputation (1D)
# ============================================================

def precompute_levy_score_1d(
    gx: np.ndarray,
    sigma: float,
    lam: float,
    sigma_L: float,
    multipliers: list,
    pm: list,
    n_theta: int = 23
) -> np.ndarray:
    """
    Precompute stationary Levy score on 1D grid for LSB-MC.

    Score formula (manuscript Eq. 27):
        S_L^s(x) = -lam * ∫₀¹ ∫ r * exp{-2(V(x-θr)-V(x))/σ²} ν(dr) dθ

    For 1D, the jump direction is ±1 (left or right), and we integrate over
    both directions and theta.

    Args:
        gx: Grid points
        sigma: Noise scale (target is exp(-2V/σ²))
        lam: Jump intensity
        sigma_L: Jump magnitude scale
        multipliers: Jump size multipliers
        pm: Probability mass for each multiplier
        n_theta: Number of theta integration points

    Returns:
        Score field S(x) on grid
    """
    V = V_doublewell(gx)
    S = np.zeros_like(gx)

    thetas = np.linspace(0.05, 1.0, n_theta)
    directions = np.array([1.0, -1.0])

    pm = np.array(pm, dtype=float)
    pm /= np.sum(pm)

    for k, prob_k in enumerate(pm):
        z_mag = sigma_L * multipliers[k]

        for direction in directions:
            r = z_mag * direction  # can be negative

            acc = np.zeros_like(gx)
            for th in thetas:
                V_minus = V_doublewell(gx - th * r)
                V_plus = V_doublewell(gx + th * r)

                dlog_minus = -2.0 * (V_minus - V) / (sigma ** 2)
                dlog_plus = -2.0 * (V_plus - V) / (sigma ** 2)

                ratio_minus = np.exp(np.clip(dlog_minus, -EXPO_CLIP, EXPO_CLIP))
                ratio_plus = np.exp(np.clip(dlog_plus, -EXPO_CLIP, EXPO_CLIP))

                term = 0.5 * (ratio_minus - ratio_plus)
                acc += term * r

            S += (prob_k / len(directions)) * (acc / len(thetas))

    S = lam * np.clip(S, -S_CLIP, S_CLIP)

    return S


# ============================================================
# Samplers
# ============================================================

def step_diffusion_1d(
    x: np.ndarray,
    dt: float,
    sigma: float,
    rng: np.random.Generator,
    clip_bounds: tuple = None
) -> np.ndarray:
    """
    One step of overdamped Langevin diffusion.

    SDE: dX = -∇V(X) dt + σ dB
         = (X - X^3) dt + σ dB

    Args:
        x: Current positions (N,)
        dt: Time step
        sigma: Noise scale
        rng: Random generator
        clip_bounds: Optional (min, max) clipping

    Returns:
        Updated positions (N,)
    """
    drift = -gradV_doublewell(x)
    x_new = x + dt * drift / (1.0 + dt * np.abs(drift)) + sigma * np.sqrt(dt) * rng.standard_normal(x.shape)

    if clip_bounds is not None:
        x_new = np.clip(x_new, clip_bounds[0], clip_bounds[1])

    return x_new


def step_mala_1d(
    x: np.ndarray,
    dt: float,
    sigma: float,
    rng: np.random.Generator,
    clip_bounds: tuple = None
) -> tuple[np.ndarray, float]:
    """
    One step of 1D MALA targeting p(x) ∝ exp(-2V(x)/sigma^2).
    """
    grad = grad_logpi_doublewell(x, sigma)
    mean = x + 0.5 * dt * grad
    proposal = mean + np.sqrt(dt) * rng.standard_normal(x.shape)

    if clip_bounds is not None:
        proposal = np.clip(proposal, clip_bounds[0], clip_bounds[1])

    logp_x = logpi_doublewell(x, sigma)
    logp_y = logpi_doublewell(proposal, sigma)

    grad_y = grad_logpi_doublewell(proposal, sigma)
    mean_y = proposal + 0.5 * dt * grad_y

    log_q_y_given_x = -((proposal - mean) ** 2) / (2.0 * dt)
    log_q_x_given_y = -((x - mean_y) ** 2) / (2.0 * dt)
    log_alpha = (logp_y + log_q_x_given_y) - (logp_x + log_q_y_given_x)

    accept = np.log(rng.random(x.shape[0])) < log_alpha
    x_new = x.copy()
    x_new[accept] = proposal[accept]
    return x_new, float(np.mean(accept))


def step_lsbmc_1d(
    x: np.ndarray,
    dt: float,
    sigma: float,
    gx: np.ndarray,
    drift_grid: np.ndarray,
    score_grid: np.ndarray,
    rng: np.random.Generator,
    lam: float,
    sigma_L: float,
    multipliers: list,
    pm: list,
    clip_bounds: tuple = None
) -> np.ndarray:
    """
    One step of LSB-MC (Levy-Score-Based Monte Carlo) for 1D.

    SDE: dZ = (-∇V + S_L^s) dt + σ dB + dL

    where dL is compound Poisson jump noise.

    Args:
        x: Current positions (N,)
        dt: Time step
        sigma: Noise scale
        gx: Grid points for interpolation
        drift_grid: Precomputed drift -∇V on grid
        score_grid: Precomputed Levy score on grid
        rng: Random generator
        lam: Jump intensity
        sigma_L: Jump magnitude
        multipliers: Jump size multipliers
        pm: Probability mass
        clip_bounds: Optional clipping

    Returns:
        Updated positions (N,)
    """
    drift = np.interp(np.clip(x, gx[0], gx[-1]), gx, drift_grid)
    score = np.interp(np.clip(x, gx[0], gx[-1]), gx, score_grid)

    # Total drift: -∇V + score
    total_drift = drift - score

    # Tamed Euler step
    x_new = x + dt * total_drift / (1.0 + dt * np.abs(total_drift))
    x_new += sigma * np.sqrt(dt) * rng.standard_normal(x.shape)

    # Compound Poisson jumps
    n_jumps = rng.poisson(lam * dt, size=x.shape[0])
    idx = np.where(n_jumps > 0)[0]

    if len(idx) > 0:
        pm_norm = np.array(pm, dtype=float)
        pm_norm /= np.sum(pm_norm)

        for i in idx:
            k = n_jumps[i]
            m_choice = rng.choice(multipliers, size=k, p=pm_norm)
            directions = rng.choice([-1.0, 1.0], size=k)
            jump = np.sum(sigma_L * m_choice * directions)
            x_new[i] += jump

    if clip_bounds is not None:
        x_new = np.clip(x_new, clip_bounds[0], clip_bounds[1])

    return x_new


# ============================================================
# Density Estimation
# ============================================================

def density_on_grid_1d(samples: np.ndarray, gx: np.ndarray, do_smooth: bool = True, sigma: float = 0.06) -> np.ndarray:
    """
    Estimate density from samples on 1D grid using histogram + optional smoothing.

    Args:
        samples: Particle positions (N,)
        gx: Grid points
        do_smooth: Whether to apply Gaussian smoothing
        sigma: Smoothing kernel width

    Returns:
        Normalized density on grid
    """
    dx = gx[1] - gx[0]
    bins = np.concatenate([gx - dx / 2.0, [gx[-1] + dx / 2.0]])

    hist, _ = np.histogram(samples, bins=bins)
    dens = hist.astype(float) / (samples.size * dx + 1e-12)

    if do_smooth:
        kernel_sigma = max(sigma, 1.2 * dx)
        m = int(np.ceil(4.0 * kernel_sigma / dx))
        xs = np.arange(-m, m + 1) * dx
        ker = np.exp(-0.5 * (xs / kernel_sigma) ** 2)
        ker = ker / (np.sum(ker) * dx + 1e-300)
        dens = np.convolve(dens, ker, mode="same")

    dens = np.maximum(dens, 1e-300)
    dens /= (np.sum(dens) * dx + 1e-300)

    return dens


# ============================================================
# Metrics
# ============================================================

def compute_l1_l2(dens: np.ndarray, pi: np.ndarray, dx: float) -> tuple:
    """
    Compute L1 and L2 errors between empirical and target densities.

    Args:
        dens: Empirical density
        pi: Target density
        dx: Grid spacing

    Returns:
        (L1, L2) errors
    """
    abs_err = np.abs(dens - pi)
    l1 = float(np.sum(abs_err) * dx)
    l2 = float(np.sqrt(np.sum((dens - pi) ** 2) * dx))
    return l1, l2


def wasserstein2_1d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute exact 1D Wasserstein-2 distance via quantile matching.

    Args:
        x, y: Two sets of samples

    Returns:
        W2 distance
    """
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


def compute_bias(samples: np.ndarray, true_mean: float) -> float:
    """
    Compute average bias for observable = mean(x).

    Bias = |E[observable(samples)] - true_observable|

    Args:
        samples: Current particle positions
        true_mean: True equilibrium mean

    Returns:
        Absolute bias
    """
    return float(np.abs(np.mean(samples) - true_mean))


def sample_from_target_1d(rng: np.random.Generator, pi: np.ndarray, gx: np.ndarray, n: int) -> np.ndarray:
    """
    Sample from target density on grid (for W2 computation).

    Args:
        rng: Random generator
        pi: Target density on grid
        gx: Grid points
        n: Number of samples

    Returns:
        Samples from target (n,)
    """
    dx = gx[1] - gx[0]
    w = np.asarray(pi, dtype=float) * dx
    w /= np.sum(w)

    idx = rng.choice(gx.size, size=n, p=w)
    jitter = (rng.random(n) - 0.5) * dx

    return gx[idx] + jitter


# ============================================================
# Experiment Runner
# ============================================================

def run_doublewell_experiment(
    sigma: float = 0.35,
    dt: float = 1e-3,
    T: float = 40.0,
    N: int = 8000,
    alpha: float = 1.5,
    lam: float = 1.0,
    sigma_L: float = 1.6,
    multipliers: list = None,
    pm: list = None,
    init_mean: float = -1.2,
    init_std: float = 0.08,
    seed: int = 42,
    output_dir: str = None,
    mala_dt: float = None,
    benchmark_config: dict = None,
):
    """
    Run double-well experiment comparing ULA vs MALA vs FLMC vs LSBMC.

    Args:
        sigma: Noise scale (target is exp(-2V/sigma^2))
        dt: Time step
        T: Final time
        N: Number of particles
        alpha: FLMC tail index (1 < alpha <= 2)
        lam: LSB-MC jump intensity
        sigma_L: LSB-MC jump magnitude
        multipliers: LSB-MC jump size multipliers
        pm: LSB-MC probability mass
        init_mean: Initial mean position
        init_std: Initial standard deviation
        seed: Random seed
        output_dir: Where to save outputs
        mala_dt: Optional MALA proposal step size
    """
    if multipliers is None:
        multipliers = [1.0, 1.8, 2.6]
    if pm is None:
        pm = [0.70, 0.22, 0.08]
    if benchmark_config is None:
        benchmark_config = get_default_benchmark_config({
            "benchmark_num_repeats": 4,
            "sinkhorn_method": "sinkhorn_stabilized",
            "sinkhorn_epsilon": 0.1,
            "sinkhorn_max_iter": 300,
            "sinkhorn_tol": 1e-5,
            "sinkhorn_subsample_size": 512,
            "mmd_subsample_size": 768,
            "emc_enabled": True,
            "emc_tau": 0.5,
        })

    print("=" * 70)
    print("GINZBURG-LANDAU DOUBLE-WELL (1D)")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  Potential: V(x) = x^4/4 - x^2/2")
    print(f"  Noise scale σ = {sigma}")
    print(f"  Time step dt = {dt}")
    print(f"  Final time T = {T}")
    print(f"  Particles N = {N}")
    print(f"  FLMC alpha = {alpha}")
    print(f"  LSBMC: λ={lam}, σ_L={sigma_L}, mults={multipliers}, pm={pm}")
    print(f"  Init: mean={init_mean}, std={init_std}")
    print()

    rng = np.random.default_rng(seed)
    gx = np.linspace(-8.0, 8.0, 2401)
    dx = gx[1] - gx[0]
    clip_bounds = (-10.0, 10.0)

    pi = compute_target_density_1d(gx, sigma)
    true_mean = np.sum(gx * pi) * dx  # ≈ 0 by symmetry

    print(f"Target density:")
    print(f"  Grid: [{gx[0]:.1f}, {gx[-1]:.1f}], dx={dx:.4f}")
    print(f"  True mean: {true_mean:.6f}")
    print()

    # Precompute LSB-MC score
    print("Precomputing LSB-MC Levy score...")
    drift_grid = -gradV_doublewell(gx)  # -∇V
    score_grid = precompute_levy_score_1d(gx, sigma, lam, sigma_L, multipliers, pm, n_theta=23)
    print(f"  Score range: [{np.min(score_grid):.4f}, {np.max(score_grid):.4f}]")
    print()

    x0 = init_mean + init_std * rng.standard_normal(N)
    x_diff = np.clip(x0.copy(), clip_bounds[0], clip_bounds[1])
    x_mala = np.clip(x0.copy(), clip_bounds[0], clip_bounds[1])
    x_flmc = np.clip(x0.copy(), clip_bounds[0], clip_bounds[1])
    x_lsb = np.clip(x0.copy(), clip_bounds[0], clip_bounds[1])

    if mala_dt is None:
        mala_dt = dt

    # Reference samples for W2
    n_ref = 3000
    ref_samples = sample_from_target_1d(rng, pi, gx, n_ref)
    benchmark_mode_descriptors = np.array([[-1.0], [1.0]], dtype=float)

    steps = int(round(T / dt))
    check_every = max(1, steps // 25)  # 25 checkpoints

    history = {
        "t": [],
        "w2_diff": [], "l1_diff": [], "l2_diff": [], "bias_diff": [],
        "w2_mala": [], "l1_mala": [], "l2_mala": [], "bias_mala": [],
        "w2_flmc": [], "l1_flmc": [], "l2_flmc": [], "bias_flmc": [],
        "w2_lsb": [], "l1_lsb": [], "l2_lsb": [], "bias_lsb": []
    }
    benchmark_metric_names = [
        "benchmark_sinkhorn_ot_cost",
        "benchmark_sinkhorn_divergence",
        "benchmark_mmd_squared",
        "benchmark_mmd",
        "benchmark_emc",
    ]
    benchmark_history = init_benchmark_history(benchmark_metric_names)
    acc_mala_sum = 0.0
    acc_mala_count = 0

    print(f"Running {steps} steps (checkpoints every {check_every} steps)...")
    print()

    for i in range(steps + 1):
        if i % check_every == 0 or i == steps:
            t = i * dt

            w2_d = wasserstein2_1d(x_diff, ref_samples)
            w2_m = wasserstein2_1d(x_mala, ref_samples)
            w2_f = wasserstein2_1d(x_flmc, ref_samples)
            w2_l = wasserstein2_1d(x_lsb, ref_samples)

            dens_d = density_on_grid_1d(x_diff, gx, do_smooth=True)
            dens_m = density_on_grid_1d(x_mala, gx, do_smooth=True)
            dens_f = density_on_grid_1d(x_flmc, gx, do_smooth=True)
            dens_l = density_on_grid_1d(x_lsb, gx, do_smooth=True)

            l1_d, l2_d = compute_l1_l2(dens_d, pi, dx)
            l1_m, l2_m = compute_l1_l2(dens_m, pi, dx)
            l1_f, l2_f = compute_l1_l2(dens_f, pi, dx)
            l1_l, l2_l = compute_l1_l2(dens_l, pi, dx)

            bias_d = compute_bias(x_diff, true_mean)
            bias_m = compute_bias(x_mala, true_mean)
            bias_f = compute_bias(x_flmc, true_mean)
            bias_l = compute_bias(x_lsb, true_mean)

            history["t"].append(t)
            history["w2_diff"].append(w2_d)
            history["l1_diff"].append(l1_d)
            history["l2_diff"].append(l2_d)
            history["bias_diff"].append(bias_d)
            history["w2_mala"].append(w2_m)
            history["l1_mala"].append(l1_m)
            history["l2_mala"].append(l2_m)
            history["bias_mala"].append(bias_m)
            history["w2_flmc"].append(w2_f)
            history["l1_flmc"].append(l1_f)
            history["l2_flmc"].append(l2_f)
            history["bias_flmc"].append(bias_f)
            history["w2_lsb"].append(w2_l)
            history["l1_lsb"].append(l1_l)
            history["l2_lsb"].append(l2_l)
            history["bias_lsb"].append(bias_l)

            benchmark_history["t"].append(t)
            method_samples = {
                "ula": clip_samples_to_box(x_diff, gx[0], gx[-1]),
                "mala": clip_samples_to_box(x_mala, gx[0], gx[-1]),
                "flmc": clip_samples_to_box(x_flmc, gx[0], gx[-1]),
                "lsbmc": clip_samples_to_box(x_lsb, gx[0], gx[-1]),
            }
            for method_idx, (slug, _, _) in enumerate(BENCHMARK_METHODS):
                metric_rng = make_metric_rng(benchmark_config["metric_seed"], seed, i, method_idx)
                values = compute_benchmark_metrics(
                    method_samples[slug],
                    ref_samples,
                    benchmark_config,
                    metric_rng,
                    metric_prefix="benchmark",
                    mode_descriptors=benchmark_mode_descriptors,
                    context=f"doublewell seed={seed} t={t:.4f} method={slug}",
                )
                for metric_name in benchmark_metric_names:
                    benchmark_history[f"{metric_name}_{slug}"].append(
                        values.get(metric_name, float("nan"))
                    )

            print(
                f"t={t:6.2f} | W2: ula={w2_d:.4f}, mala={w2_m:.4f}, flmc={w2_f:.4f}, lsbmc={w2_l:.4f} | "
                f"Bias: ula={bias_d:.4f}, mala={bias_m:.4f}, flmc={bias_f:.4f}, lsbmc={bias_l:.4f}"
            )

        if i == steps:
            break

        x_diff = step_diffusion_1d(x_diff, dt, sigma, rng, clip_bounds)
        x_mala, acc_m = step_mala_1d(x_mala, mala_dt, sigma, rng, clip_bounds)
        x_flmc = step_flmc_1d(
            x_flmc,
            dt,
            alpha,
            lambda arr: gradU_doublewell(arr, sigma),
            rng,
            clip_bounds,
        )
        x_lsb = step_lsbmc_1d(x_lsb, dt, sigma, gx, drift_grid, score_grid, rng, lam, sigma_L, multipliers, pm, clip_bounds)
        acc_mala_sum += acc_m
        acc_mala_count += 1

    print()
    print("Simulation complete!")
    print()

    # Final comparison
    print("Final metrics (t={:.1f}):".format(T))
    print(f"  ULA:       W2={history['w2_diff'][-1]:.4f}, L1={history['l1_diff'][-1]:.4f}, L2={history['l2_diff'][-1]:.4f}, Bias={history['bias_diff'][-1]:.4f}")
    print(f"  MALA:      W2={history['w2_mala'][-1]:.4f}, L1={history['l1_mala'][-1]:.4f}, L2={history['l2_mala'][-1]:.4f}, Bias={history['bias_mala'][-1]:.4f}")
    print(f"  FLMC:      W2={history['w2_flmc'][-1]:.4f}, L1={history['l1_flmc'][-1]:.4f}, L2={history['l2_flmc'][-1]:.4f}, Bias={history['bias_flmc'][-1]:.4f}")
    print(f"  LSBMC:     W2={history['w2_lsb'][-1]:.4f}, L1={history['l1_lsb'][-1]:.4f}, L2={history['l2_lsb'][-1]:.4f}, Bias={history['bias_lsb'][-1]:.4f}")
    print(f"  MALA mean acceptance rate: {acc_mala_sum / max(acc_mala_count, 1):.4f}")
    print()

    # Plot results
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plot_results(history, gx, pi, x_diff, x_mala, x_flmc, x_lsb, output_dir)
        benchmark_keys = [
            f"{metric_name}_{slug}"
            for metric_name in benchmark_metric_names
            for slug, _, _ in BENCHMARK_METHODS
        ]
        benchmark_mean = {
            key: np.asarray(benchmark_history[key], dtype=float)
            for key in benchmark_keys
        }
        benchmark_std = {
            key: np.zeros_like(benchmark_mean[key])
            for key in benchmark_keys
        }
        t_benchmark = np.asarray(benchmark_history["t"], dtype=float)
        benchmark_base = os.path.join(output_dir, "benchmark_metrics_doublewell")
        plot_benchmark_metrics_figure(
            t_benchmark,
            benchmark_mean,
            benchmark_std,
            ["benchmark_sinkhorn_divergence", "benchmark_mmd", "benchmark_emc"],
            benchmark_base,
            "Double-Well: Benchmark Metrics",
            metric_labels={
                "benchmark_sinkhorn_divergence": "Sinkhorn",
                "benchmark_mmd": "MMD",
                "benchmark_emc": "EMC",
            },
        )
        save_benchmark_metrics_csv(
            f"{benchmark_base}.csv",
            t_benchmark,
            benchmark_mean,
            benchmark_std,
            benchmark_metric_names,
        )
        save_benchmark_metadata_json(
            os.path.join(output_dir, "metrics_benchmark_doublewell.json"),
            "doublewell",
            benchmark_config,
            benchmark_metric_names,
            mode_descriptors=benchmark_mode_descriptors,
            extra_metadata={
                "benchmark_reference_size": int(ref_samples.shape[0]),
                "mode_descriptor_source": "xi_1=-1, xi_2=+1",
            },
        )
        print(f"Figures saved to: {output_dir}")
        print(f"Benchmark metrics saved to: {benchmark_base}.png/.pdf/.csv")

    return history


# ============================================================
# Plotting
# ============================================================

def plot_results(history, gx, pi, x_diff_final, x_mala_final, x_flmc_final, x_lsb_final, output_dir):
    """Generate all plots for double-well experiment."""

    def save_figure_both(fig, out_base):
        fig.savefig(f"{out_base}.png")
        fig.savefig(f"{out_base}.pdf")
        plt.close(fig)

    # Set academic style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    t = np.array(history["t"])

    # ========================================
    # Figure 1: Convergence Metrics (W2, L1, L2)
    # ========================================
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    axes[0].plot(t, history["w2_diff"], label="ULA", color="C0", linewidth=1.5)
    axes[0].plot(t, history["w2_mala"], label="MALA", color="C1", linewidth=1.5)
    axes[0].plot(t, history["w2_flmc"], label="FLMC", color="C2", linewidth=1.5)
    axes[0].plot(t, history["w2_lsb"], label="LSBMC", color="C3", linewidth=1.5)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("W2 Distance")
    axes[0].set_title("Wasserstein-2 Distance")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, history["l1_diff"], label="ULA", color="C0", linewidth=1.5)
    axes[1].plot(t, history["l1_mala"], label="MALA", color="C1", linewidth=1.5)
    axes[1].plot(t, history["l1_flmc"], label="FLMC", color="C2", linewidth=1.5)
    axes[1].plot(t, history["l1_lsb"], label="LSBMC", color="C3", linewidth=1.5)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("L1 Error")
    axes[1].set_title("L1 / MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, history["l2_diff"], label="ULA", color="C0", linewidth=1.5)
    axes[2].plot(t, history["l2_mala"], label="MALA", color="C1", linewidth=1.5)
    axes[2].plot(t, history["l2_flmc"], label="FLMC", color="C2", linewidth=1.5)
    axes[2].plot(t, history["l2_lsb"], label="LSBMC", color="C3", linewidth=1.5)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("L2 Error")
    axes[2].set_title("L2 / RMSE")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure_both(fig, os.path.join(output_dir, "doublewell_metrics"))

    # ========================================
    # Figure 2: Average Bias
    # ========================================
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(t, history["bias_diff"], label="ULA", color="C0", linewidth=1.5)
    ax.plot(t, history["bias_mala"], label="MALA", color="C1", linewidth=1.5)
    ax.plot(t, history["bias_flmc"], label="FLMC", color="C2", linewidth=1.5)
    ax.plot(t, history["bias_lsb"], label="LSBMC", color="C3", linewidth=1.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Average Bias")
    ax.set_title("Absolute Bias of the Sample Mean")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure_both(fig, os.path.join(output_dir, "doublewell_bias"))

    # ========================================
    # Figure 3: Final Density Comparison
    # ========================================
    dens_diff_final = density_on_grid_1d(x_diff_final, gx, do_smooth=True)
    dens_mala_final = density_on_grid_1d(x_mala_final, gx, do_smooth=True)
    dens_flmc_final = density_on_grid_1d(x_flmc_final, gx, do_smooth=True)
    dens_lsb_final = density_on_grid_1d(x_lsb_final, gx, do_smooth=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gx, pi, label="True Density", color="black", linewidth=2.0, linestyle="--")
    ax.plot(gx, dens_diff_final, label="ULA", color="C0", linewidth=1.8)
    ax.plot(gx, dens_mala_final, label="MALA", color="C1", linewidth=1.8)
    ax.plot(gx, dens_flmc_final, label="FLMC", color="C2", linewidth=1.8)
    ax.plot(gx, dens_lsb_final, label="LSBMC", color="C3", linewidth=1.8)
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title("Final Density Comparison (Double-Well)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure_both(fig, os.path.join(output_dir, "doublewell_final_density"))


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    output_dir = os.path.join(THIS_DIR, "doublewell_output")
    run_doublewell_experiment(output_dir=output_dir)
