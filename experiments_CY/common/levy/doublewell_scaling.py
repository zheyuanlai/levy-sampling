"""Production low-temperature simulation study for the double-well example."""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .mixing_time_models import (
    compare_model_families,
    fit_diagnostics,
    model_curve,
    select_temperature_subset,
)
from .one_d_density import (
    DensityDiagnosticConfig,
    binned_gaussian_kde_on_grid_with_diagnostics,
    central_interval_from_cdf,
    density_metric_bundle,
    target_local_scott_bandwidth,
    trapz_weights,
)
from .plot_style import (
    REFERENCE_STYLES,
    apply_plot_style,
    method_color,
    method_marker,
    panel_label,
)


DOUBLEWELL_BARRIER = 0.25
PRODUCTION_EPSILON_GRID = (
    0.35,
    0.30,
    0.26,
    0.22,
    0.20,
    0.18,
    0.16,
    0.14,
    0.125,
    0.11,
    0.10,
    0.09,
    0.08,
    0.07,
    0.06,
    0.05,
)
METHODS = ("Langevin", "LSC-CP")
METRICS = ("truncated_KDE_chi2", "well_TV", "W1", "CDF_sup")
SUBSETS = ("all", "exclude_largest_2", "lowest_10", "lowest_8", "uncensored_only")


@dataclass(frozen=True)
class DoubleWellScalingConfig:
    epsilon_values: tuple[float, ...]
    n_particles: int
    n_seeds: int
    dt: float
    record_dt: float
    persistence_records: int
    grid_n: int
    score_grid_n: int
    theta_n: int
    quadrature_r: int
    jump_center: float
    jump_half_width: float
    jump_intensity: float
    bias_floor_replicates: int
    bootstrap_replicates: int
    summary_bootstrap_replicates: int
    n_jobs: int
    global_seed: int
    clip_left: float = -6.0
    clip_right: float = 6.0


def config_for_profile(profile: str = "production") -> DoubleWellScalingConfig:
    profile = str(profile).lower()
    if profile == "smoke":
        return DoubleWellScalingConfig(
            epsilon_values=(0.30, 0.22, 0.16, 0.11, 0.08),
            n_particles=180,
            n_seeds=2,
            dt=0.01,
            record_dt=0.20,
            persistence_records=2,
            grid_n=260,
            score_grid_n=280,
            theta_n=5,
            quadrature_r=3,
            jump_center=2.0,
            jump_half_width=0.22,
            jump_intensity=1.0,
            bias_floor_replicates=4,
            bootstrap_replicates=8,
            summary_bootstrap_replicates=20,
            n_jobs=1,
            global_seed=20261712,
        )
    if profile not in {"production", "paper", "paperlite"}:
        raise ValueError(f"unknown Phase17L profile: {profile}")
    return DoubleWellScalingConfig(
        epsilon_values=PRODUCTION_EPSILON_GRID,
        n_particles=2400,
        n_seeds=6,
        dt=0.003,
        record_dt=0.24,
        persistence_records=4,
        grid_n=900,
        score_grid_n=1000,
        theta_n=16,
        quadrature_r=7,
        jump_center=2.0,
        jump_half_width=0.22,
        jump_intensity=1.0,
        bias_floor_replicates=24,
        bootstrap_replicates=300,
        summary_bootstrap_replicates=500,
        n_jobs=6,
        global_seed=20261712,
    )


def potential(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    return 0.25 * values**4 - 0.5 * values**2


def potential_gradient(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    return values**3 - values


def _normalize_density(grid: np.ndarray, log_density: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = trapz_weights(grid)
    shifted = np.asarray(log_density, dtype=float) - float(np.max(log_density))
    density = np.exp(shifted)
    density /= float(np.sum(density * weights))
    return density, weights


def _cdf_from_density(density: np.ndarray, weights: np.ndarray) -> np.ndarray:
    cdf = np.cumsum(np.asarray(density, dtype=float) * np.asarray(weights, dtype=float))
    cdf /= float(cdf[-1])
    cdf[-1] = 1.0
    return cdf


def _sample_target(
    grid: np.ndarray,
    density: np.ndarray,
    weights: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    probability = density * weights
    probability /= float(np.sum(probability))
    return grid[rng.choice(len(grid), size=int(size), p=probability)]


def _empirical_cdf(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(samples, dtype=float))
    return np.searchsorted(ordered, grid, side="right") / float(len(ordered))


def _w1_from_cdf(samples: np.ndarray, grid: np.ndarray, target_cdf: np.ndarray) -> float:
    empirical = _empirical_cdf(samples, grid)
    return float(np.trapz(np.abs(empirical - target_cdf), grid))


def _well_tv(samples: np.ndarray, target_left_mass: float) -> float:
    return float(abs(np.mean(np.asarray(samples) < 0.0) - target_left_mass))


def _target_context(epsilon: float, config: DoubleWellScalingConfig) -> dict[str, object]:
    grid = np.linspace(config.clip_left, config.clip_right, config.grid_n)
    density, weights = _normalize_density(grid, -potential(grid) / float(epsilon))
    cdf = _cdf_from_density(density, weights)
    target_left = float(np.sum(density[grid < 0.0] * weights[grid < 0.0]))
    bandwidth = target_local_scott_bandwidth(grid, density, config.n_particles)
    chi_left, chi_right, omitted = central_interval_from_cdf(grid, cdf, mass=0.98)
    mask = (grid >= chi_left) & (grid <= chi_right)
    density_config = DensityDiagnosticConfig(
        bandwidth=float(bandwidth),
        chi_interval=(float(chi_left), float(chi_right)),
        chi_omitted_mass=float(omitted),
        chi_target_min=float(np.min(density[mask])),
    )
    return {
        "epsilon": float(epsilon),
        "grid": grid,
        "density": density,
        "weights": weights,
        "cdf": cdf,
        "target_left_mass": target_left,
        "bandwidth": float(bandwidth),
        "density_config": density_config,
    }


def _metric_values(samples: np.ndarray, context: dict[str, object]) -> dict[str, float]:
    grid = np.asarray(context["grid"], dtype=float)
    density = np.asarray(context["density"], dtype=float)
    cdf = np.asarray(context["cdf"], dtype=float)
    bandwidth = float(context["bandwidth"])
    density_config = context["density_config"]
    kde, _ = binned_gaussian_kde_on_grid_with_diagnostics(samples, grid, bandwidth)
    bundle = density_metric_bundle(kde, density, grid, density_config, samples=samples)
    empirical_cdf = _empirical_cdf(samples, grid)
    return {
        "truncated_KDE_chi2": float(bundle["truncated_KDE_chi2"]),
        "well_TV": _well_tv(samples, float(context["target_left_mass"])),
        "W1": float(np.trapz(np.abs(empirical_cdf - cdf), grid)),
        "CDF_sup": float(np.max(np.abs(empirical_cdf - cdf))),
    }


def _bias_floor_and_thresholds(
    context: dict[str, object],
    config: DoubleWellScalingConfig,
    epsilon_index: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    rng = np.random.default_rng(config.global_seed + 10000 + 101 * epsilon_index)
    rows: list[dict[str, object]] = []
    for replicate in range(config.bias_floor_replicates):
        samples = _sample_target(
            np.asarray(context["grid"]),
            np.asarray(context["density"]),
            np.asarray(context["weights"]),
            config.n_particles,
            rng,
        )
        values = _metric_values(samples, context)
        for metric, value in values.items():
            rows.append(
                {
                    "epsilon": float(context["epsilon"]),
                    "replicate": replicate,
                    "metric": metric,
                    "bias_floor_value": float(value),
                    "N_particles": config.n_particles,
                    "KDE_bandwidth": float(context["bandwidth"]),
                }
            )
    floors = pd.DataFrame(rows)
    base_threshold = {
        "truncated_KDE_chi2": 0.05,
        "well_TV": 0.075,
        "W1": 0.08,
        "CDF_sup": 0.06,
    }
    threshold_rows: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = floors.loc[floors["metric"] == metric, "bias_floor_value"].to_numpy(dtype=float)
        floor_max = float(np.max(values))
        threshold = float(max(base_threshold[metric], 2.5 * floor_max))
        threshold_rows[metric] = {
            "bias_floor_mean": float(np.mean(values)),
            "bias_floor_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "bias_floor_q95": float(np.quantile(values, 0.95)),
            "bias_floor_max": floor_max,
            "threshold": threshold,
            "threshold_above_bias_floor": bool(threshold > floor_max),
            "threshold_rule": f"max({base_threshold[metric]:g}, 2.5 * maximum target-sampling floor)",
        }
    return floors, threshold_rows


def _jump_atoms(config: DoubleWellScalingConfig) -> tuple[np.ndarray, np.ndarray]:
    z, w = np.polynomial.legendre.leggauss(config.quadrature_r)
    positive = config.jump_center + config.jump_half_width * z
    negative = -config.jump_center + config.jump_half_width * z
    locations = np.concatenate([positive, negative])
    weights = np.concatenate([0.25 * w, 0.25 * w])
    return locations, weights / float(np.sum(weights))


def _levy_score_grid(
    epsilon: float,
    config: DoubleWellScalingConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    grid = np.linspace(config.clip_left, config.clip_right, config.score_grid_n)
    theta_z, theta_w = np.polynomial.legendre.leggauss(config.theta_n)
    theta = 0.5 * (theta_z + 1.0)
    theta_w = 0.5 * theta_w
    jumps, jump_w = _jump_atoms(config)
    logp_grid = -potential(grid) / float(epsilon)
    score_sum = np.zeros_like(grid)
    clipped = 0
    total = 0
    max_abs_log_ratio = 0.0
    overflow_guard = 690.0
    for jump, probability in zip(jumps, jump_w):
        inner = np.zeros_like(grid)
        for theta_value, theta_probability in zip(theta, theta_w):
            log_ratio = -potential(grid - theta_value * jump) / float(epsilon) - logp_grid
            clipped += int(np.count_nonzero(np.abs(log_ratio) > overflow_guard))
            total += int(log_ratio.size)
            max_abs_log_ratio = max(max_abs_log_ratio, float(np.max(np.abs(log_ratio))))
            ratio = np.exp(np.clip(log_ratio, -overflow_guard, overflow_guard))
            inner += float(theta_probability) * float(jump) * ratio
        score_sum += float(probability) * inner
    score = -float(config.jump_intensity) * score_sum
    if not np.all(np.isfinite(score)):
        raise FloatingPointError("nonfinite Levy-score grid")
    return grid, score, {
        "overflow_guard_log_ratio": overflow_guard,
        "overflow_guard_fraction": float(clipped / max(total, 1)),
        "max_abs_log_ratio": max_abs_log_ratio,
        "max_abs_score": float(np.max(np.abs(score))),
    }


def _apply_jumps(
    samples: np.ndarray,
    rng: np.random.Generator,
    dt: float,
    config: DoubleWellScalingConfig,
) -> np.ndarray:
    counts = rng.poisson(config.jump_intensity * dt, size=len(samples))
    total = int(np.sum(counts))
    if total == 0:
        return samples
    owners = np.repeat(np.arange(len(samples)), counts)
    signs = rng.choice(np.array([-1.0, 1.0]), size=total)
    magnitudes = rng.uniform(
        config.jump_center - config.jump_half_width,
        config.jump_center + config.jump_half_width,
        size=total,
    )
    increments = np.bincount(owners, weights=signs * magnitudes, minlength=len(samples))
    return samples + increments


def _maximum_horizon(method: str, epsilon: float) -> float:
    if method == "Langevin":
        return float(min(900.0, max(40.0, 8.0 * math.exp(DOUBLEWELL_BARRIER / epsilon))))
    if method == "LSC-CP":
        return float(min(120.0, max(30.0, 12.0 * epsilon ** (-0.5))))
    raise ValueError(method)


def _simulate_one(
    *,
    method: str,
    epsilon: float,
    epsilon_index: int,
    seed_id: int,
    initial_samples: np.ndarray,
    context: dict[str, object],
    thresholds: dict[str, dict[str, float]],
    score_grid: np.ndarray,
    score_values: np.ndarray,
    config: DoubleWellScalingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng_seed = config.global_seed + 100000 + 10000 * epsilon_index + 100 * seed_id
    if method == "LSC-CP":
        rng_seed += 1
    rng = np.random.default_rng(rng_seed)
    samples = np.asarray(initial_samples, dtype=float).copy()
    record_steps = max(1, int(round(config.record_dt / config.dt)))
    horizon = _maximum_horizon(method, epsilon)
    max_steps = int(math.ceil(horizon / config.dt))
    consecutive = {metric: 0 for metric in METRICS}
    first_crossing = {metric: np.nan for metric in METRICS}
    mixing_time = {metric: np.nan for metric in METRICS}
    below_count = {metric: 0 for metric in METRICS}
    raw_rows: list[dict[str, object]] = []
    clip_count = 0
    clip_sum = 0.0
    clip_max = 0.0
    final_step = 0

    for step in range(max_steps + 1):
        if step % record_steps == 0 or step == max_steps:
            current_time = float(step * config.dt)
            values = _metric_values(samples, context)
            for metric, value in values.items():
                threshold = float(thresholds[metric]["threshold"])
                below = bool(value <= threshold)
                below_count[metric] += int(below)
                if below and not np.isfinite(first_crossing[metric]):
                    first_crossing[metric] = current_time
                consecutive[metric] = consecutive[metric] + 1 if below else 0
                if (
                    consecutive[metric] >= config.persistence_records
                    and not np.isfinite(mixing_time[metric])
                ):
                    mixing_time[metric] = current_time - (
                        config.persistence_records - 1
                    ) * record_steps * config.dt
                raw_rows.append(
                    {
                        "epsilon": float(epsilon),
                        "method": method,
                        "seed": seed_id,
                        "rng_seed": rng_seed,
                        "metric": metric,
                        "time": current_time,
                        "metric_value": float(value),
                        "threshold": threshold,
                        "bias_floor": float(thresholds[metric]["bias_floor_max"]),
                        "reached": below,
                        "persistent_reached": bool(np.isfinite(mixing_time[metric])),
                        "N_particles": config.n_particles,
                        "dt": config.dt,
                        "planned_T_final": horizon,
                    }
                )
            if all(np.isfinite(mixing_time[metric]) for metric in METRICS):
                final_step = step
                break
        if step == max_steps:
            final_step = step
            break
        drift = -potential_gradient(samples)
        if method == "Langevin":
            samples = (
                samples
                + config.dt * drift
                + math.sqrt(2.0 * epsilon * config.dt) * rng.standard_normal(samples.shape)
            )
        elif method == "LSC-CP":
            correction = np.interp(
                np.clip(samples, score_grid[0], score_grid[-1]),
                score_grid,
                score_values,
            )
            total_drift = drift + correction
            increment = config.dt * total_drift / (1.0 + config.dt * np.abs(total_drift))
            samples = (
                samples
                + increment
                + math.sqrt(2.0 * epsilon * config.dt) * rng.standard_normal(samples.shape)
            )
            samples = _apply_jumps(samples, rng, config.dt, config)
        else:
            raise ValueError(method)
        clipped = np.clip(samples, config.clip_left, config.clip_right)
        fraction = float(np.mean(clipped != samples))
        samples = clipped
        clip_sum += fraction
        clip_max = max(clip_max, fraction)
        clip_count += 1

    final_time = float(final_step * config.dt)
    by_seed_rows: list[dict[str, object]] = []
    raw = pd.DataFrame(raw_rows)
    for metric in METRICS:
        metric_rows = raw[raw["metric"] == metric]
        persistent = bool(np.isfinite(mixing_time[metric]))
        final_metric = float(metric_rows.iloc[-1]["metric_value"])
        by_seed_rows.append(
            {
                "epsilon": float(epsilon),
                "method": method,
                "seed": seed_id,
                "rng_seed": rng_seed,
                "metric": metric,
                "threshold": float(thresholds[metric]["threshold"]),
                "bias_floor": float(thresholds[metric]["bias_floor_max"]),
                "first_crossing_time": float(first_crossing[metric]),
                "persistent_reached": persistent,
                "censored": not persistent,
                "mixing_time": float(mixing_time[metric]),
                "final_metric_value": final_metric,
                "n_recorded_points": int(len(metric_rows)),
                "n_recorded_points_below_threshold": int(below_count[metric]),
                "persistence_records": config.persistence_records,
                "N_particles": config.n_particles,
                "dt": config.dt,
                "nsteps": int(final_step),
                "T_final": final_time,
                "planned_T_final": horizon,
                "clip_fraction_mean": float(clip_sum / max(clip_count, 1)),
                "clip_fraction_max": clip_max,
            }
        )
    by_seed = pd.DataFrame(by_seed_rows)
    mixing_lookup = by_seed.set_index("metric")
    raw["censored"] = raw["metric"].map(mixing_lookup["censored"]).astype(bool)
    raw["mixing_time"] = raw["metric"].map(mixing_lookup["mixing_time"])
    raw["nsteps"] = int(final_step)
    raw["T_final"] = final_time
    return raw, by_seed


def _kaplan_meier_median(times: np.ndarray, events: np.ndarray) -> float:
    order = np.argsort(times)
    times = np.asarray(times, dtype=float)[order]
    events = np.asarray(events, dtype=bool)[order]
    survival = 1.0
    for value in np.unique(times):
        at_risk = int(np.count_nonzero(times >= value))
        deaths = int(np.count_nonzero((times == value) & events))
        if at_risk > 0 and deaths > 0:
            survival *= 1.0 - deaths / at_risk
        if survival <= 0.5:
            return float(value)
    return np.nan


def _summarize_mixing_times(
    by_seed: pd.DataFrame,
    config: DoubleWellScalingConfig,
) -> pd.DataFrame:
    rng = np.random.default_rng(config.global_seed + 700000)
    rows: list[dict[str, object]] = []
    for keys, group in by_seed.groupby(["epsilon", "method", "metric"], sort=True):
        epsilon, method, metric = keys
        censored = group["censored"].astype(bool).to_numpy()
        exact = group["mixing_time"].to_numpy(dtype=float)
        censor_time = group["T_final"].to_numpy(dtype=float)
        observed = np.where(censored, censor_time, exact)
        events = ~censored
        km_median = _kaplan_meier_median(observed, events)
        bootstrap = []
        for _ in range(config.summary_bootstrap_replicates):
            ids = rng.integers(0, len(group), size=len(group))
            value = _kaplan_meier_median(observed[ids], events[ids])
            if np.isfinite(value):
                bootstrap.append(value)
        ci = np.quantile(bootstrap, [0.025, 0.975]) if bootstrap else [np.nan, np.nan]
        reached = exact[events]
        rows.append(
            {
                "epsilon": float(epsilon),
                "method": method,
                "metric": metric,
                "n_seeds": int(len(group)),
                "n_reached": int(np.count_nonzero(events)),
                "n_censored": int(np.count_nonzero(censored)),
                "all_reached": bool(np.all(events)),
                "km_median_mixing_time": km_median,
                "median_reached_mixing_time": float(np.median(reached)) if reached.size else np.nan,
                "mean_reached_mixing_time": float(np.mean(reached)) if reached.size else np.nan,
                "q25_reached_mixing_time": float(np.quantile(reached, 0.25)) if reached.size else np.nan,
                "q75_reached_mixing_time": float(np.quantile(reached, 0.75)) if reached.size else np.nan,
                "bootstrap_median_ci_low": float(ci[0]),
                "bootstrap_median_ci_high": float(ci[1]),
                "threshold": float(group["threshold"].iloc[0]),
                "bias_floor": float(group["bias_floor"].iloc[0]),
                "maximum_censoring_time": float(np.max(censor_time)),
            }
        )
    return pd.DataFrame(rows)


def _build_fit_tables(
    by_seed: pd.DataFrame,
    config: DoubleWellScalingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    metric_index = {metric: index for index, metric in enumerate(METRICS)}
    method_index = {method: index for index, method in enumerate(METHODS)}
    subset_index = {subset: index for index, subset in enumerate(SUBSETS)}
    for method in METHODS:
        for metric in METRICS:
            base = by_seed[(by_seed["method"] == method) & (by_seed["metric"] == metric)].copy()
            for subset in SUBSETS:
                selected = select_temperature_subset(base, subset)
                subset_fits = []
                for family_index, family in enumerate(("arrhenius", "polynomial")):
                    seed = (
                        config.global_seed
                        + 800000
                        + 10000 * method_index[method]
                        + 1000 * metric_index[metric]
                        + 100 * subset_index[subset]
                        + family_index
                    )
                    diagnostics = fit_diagnostics(
                        selected,
                        family,
                        n_bootstrap=config.bootstrap_replicates,
                        bootstrap_seed=seed,
                    )
                    row = {
                        "method": method,
                        "metric": metric,
                        "epsilon_subset": subset,
                        **diagnostics,
                    }
                    subset_fits.append(row)
                    fit_rows.append(row)
                comparison = compare_model_families(subset_fits)
                selection_rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        "epsilon_subset": subset,
                        **comparison,
                    }
                )
    fits = pd.DataFrame(fit_rows)
    selections = pd.DataFrame(selection_rows)
    selected_lookup = selections.set_index(["method", "metric", "epsilon_subset"])[
        "selected_family"
    ]
    fits["selected_fit"] = [
        row.model_family
        == selected_lookup.loc[(row.method, row.metric, row.epsilon_subset)]
        for row in fits.itertuples()
    ]
    return fits, selections


def _acceptance_summary(
    fits: pd.DataFrame,
    selections: pd.DataFrame,
) -> pd.DataFrame:
    temperature_subsets = ("all", "exclude_largest_2", "lowest_10", "lowest_8")
    local_sel = selections[
        (selections["method"] == "Langevin")
        & (selections["metric"] == "truncated_KDE_chi2")
    ].set_index("epsilon_subset")
    local_fit = fits[
        (fits["method"] == "Langevin")
        & (fits["metric"] == "truncated_KDE_chi2")
        & (fits["model_family"] == "arrhenius")
    ].set_index("epsilon_subset")
    preferred_required = all(
        local_sel.loc[subset, "selected_family"] == "arrhenius"
        for subset in ("all", "exclude_largest_2")
    ) and any(
        local_sel.loc[subset, "selected_family"] == "arrhenius"
        for subset in ("lowest_10", "lowest_8")
    )
    barrier = float(local_fit.loc["all", "Delta_hat"])
    ci_low = float(local_fit.loc["all", "bootstrap_ci_low"])
    ci_high = float(local_fit.loc["all", "bootstrap_ci_high"])
    barrier_close = abs(barrier - DOUBLEWELL_BARRIER) <= 0.05 or (
        ci_low <= DOUBLEWELL_BARRIER <= ci_high
    )
    local_barriers = local_fit.loc[list(temperature_subsets), "Delta_hat"].to_numpy(dtype=float)
    barrier_stable = float(np.nanmax(local_barriers) - np.nanmin(local_barriers)) <= 0.08
    local_supported = bool(preferred_required and barrier_close and barrier_stable)

    lsc_sel = selections[
        (selections["method"] == "LSC-CP")
        & (selections["metric"] == "truncated_KDE_chi2")
    ].set_index("epsilon_subset")
    lsc_fit = fits[
        (fits["method"] == "LSC-CP")
        & (fits["metric"] == "truncated_KDE_chi2")
        & (fits["model_family"] == "polynomial")
    ].set_index("epsilon_subset")
    polynomial_wins = int(
        sum(lsc_sel.loc[subset, "selected_family"] == "polynomial" for subset in temperature_subsets)
    )
    exponents = lsc_fit.loc[list(temperature_subsets), "alpha_hat"].to_numpy(dtype=float)
    exponent_range = float(np.nanmax(exponents) - np.nanmin(exponents))
    exponent_limit = float(max(0.5, 0.75 * np.nanmedian(exponents)))
    exponent_stable = bool(np.all(np.isfinite(exponents)) and np.all(exponents > 0.0) and exponent_range <= exponent_limit)
    arrhenius_quarter_selected = any(
        lsc_sel.loc[subset, "selected_family"] == "arrhenius"
        and abs(
            float(
                fits[
                    (fits["method"] == "LSC-CP")
                    & (fits["metric"] == "truncated_KDE_chi2")
                    & (fits["epsilon_subset"] == subset)
                    & (fits["model_family"] == "arrhenius")
                ]["Delta_hat"].iloc[0]
            )
            - DOUBLEWELL_BARRIER
        )
        <= 0.05
        for subset in temperature_subsets
    )
    lsc_supported = bool(polynomial_wins >= 3 and exponent_stable and not arrhenius_quarter_selected)
    return pd.DataFrame(
        [
            {
                "claim": "Langevin Arrhenius barrier near one quarter",
                "supported": local_supported,
                "primary_metric": "truncated_KDE_chi2",
                "estimate": barrier,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "selection_count": int(
                    sum(local_sel.loc[s, "selected_family"] == "arrhenius" for s in temperature_subsets)
                ),
                "selection_total": len(temperature_subsets),
                "stability_range": float(np.nanmax(local_barriers) - np.nanmin(local_barriers)),
                "criterion": "pre-registered Langevin acceptance rule",
            },
            {
                "claim": "LSC-CP polynomial-like growth",
                "supported": lsc_supported,
                "primary_metric": "truncated_KDE_chi2",
                "estimate": float(lsc_fit.loc["all", "alpha_hat"]),
                "bootstrap_ci_low": float(lsc_fit.loc["all", "bootstrap_ci_low"]),
                "bootstrap_ci_high": float(lsc_fit.loc["all", "bootstrap_ci_high"]),
                "selection_count": polynomial_wins,
                "selection_total": len(temperature_subsets),
                "stability_range": exponent_range,
                "criterion": "pre-registered LSC-CP acceptance rule",
            },
        ]
    )


def _fixed_log_amplitude(epsilon: np.ndarray, mixing_time: np.ndarray, shape: np.ndarray) -> float:
    mask = (
        np.isfinite(epsilon)
        & np.isfinite(mixing_time)
        & np.isfinite(shape)
        & (mixing_time > 0.0)
        & (shape > 0.0)
    )
    return float(np.exp(np.mean(np.log(mixing_time[mask]) - np.log(shape[mask]))))


def generate_phase17l_figure(
    table_dir: Path | str,
    figure_dir: Path | str,
) -> tuple[Path, pd.DataFrame]:
    table_dir = Path(table_dir)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(table_dir / "doublewell_phase17l_mixing_times_summary.csv")
    fits = pd.read_csv(table_dir / "doublewell_phase17l_fit_stability.csv")
    selections = pd.read_csv(table_dir / "doublewell_phase17l_model_selection_summary.csv")
    apply_plot_style(plt)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4))
    axes = axes.ravel()
    eps_grid = np.linspace(float(summary["epsilon"].min()), float(summary["epsilon"].max()), 400)
    theory_rows: list[dict[str, object]] = []

    primary = summary[summary["metric"] == "truncated_KDE_chi2"].copy()
    for method in METHODS:
        sub = primary[primary["method"] == method].sort_values("epsilon")
        epsilon = sub["epsilon"].to_numpy(dtype=float)
        center = sub["km_median_mixing_time"].to_numpy(dtype=float)
        lo = sub["bootstrap_median_ci_low"].to_numpy(dtype=float)
        hi = sub["bootstrap_median_ci_high"].to_numpy(dtype=float)
        yerr = np.vstack([np.maximum(center - lo, 0.0), np.maximum(hi - center, 0.0)])
        exact = np.isfinite(center)
        censored = sub["n_censored"].to_numpy(dtype=int) > 0
        censor_bound = sub["maximum_censoring_time"].to_numpy(dtype=float)
        color = method_color(method)
        marker = method_marker(method)
        for ax in axes[:2]:
            x_values = epsilon if ax is axes[1] else 1.0 / epsilon
            ax.errorbar(
                x_values[exact],
                center[exact],
                yerr=yerr[:, exact],
                fmt=marker,
                color=color,
                mfc=color,
                mec=color,
                ms=5.0,
                capsize=2.0,
                linestyle="none",
                label=f"{method} simulation",
            )
            if np.any(censored):
                lower_error = 0.12 * censor_bound[censored]
                ax.errorbar(
                    x_values[censored],
                    censor_bound[censored],
                    yerr=lower_error,
                    lolims=True,
                    fmt=marker,
                    color=color,
                    mfc="white",
                    mec=color,
                    ms=5.5,
                    capsize=2.0,
                    linestyle="none",
                    label=f"{method} right-censored lower bound",
                )
        all_fit = fits[
            (fits["method"] == method)
            & (fits["metric"] == "truncated_KDE_chi2")
            & (fits["epsilon_subset"] == "all")
        ]
        selected_family = selections[
            (selections["method"] == method)
            & (selections["metric"] == "truncated_KDE_chi2")
            & (selections["epsilon_subset"] == "all")
        ]["selected_family"].iloc[0]
        selected = all_fit[all_fit["model_family"] == selected_family].iloc[0]
        empirical = model_curve(
            eps_grid,
            float(selected["C_hat"]),
            float(selected["parameter_hat"]),
            selected_family,
        )
        for ax in axes[:2]:
            x_values = eps_grid if ax is axes[1] else 1.0 / eps_grid
            ax.plot(
                x_values,
                empirical,
                color=color,
                label=f"{method} empirical {selected_family}",
                **REFERENCE_STYLES["empirical_fit"],
            )
        if method == "Langevin":
            shape_data = np.exp(DOUBLEWELL_BARRIER / epsilon)
            shape_grid = np.exp(DOUBLEWELL_BARRIER / eps_grid)
            reference = r"$C_L\exp(0.25/\varepsilon)$"
            family = "fixed Arrhenius barrier"
        else:
            shape_data = epsilon ** (-0.5)
            shape_grid = eps_grid ** (-0.5)
            reference = r"$C_S\varepsilon^{-1/2}$"
            family = "fixed graph-channel inverse scale"
        amplitude = _fixed_log_amplitude(epsilon, center, shape_data)
        theory = amplitude * shape_grid
        for ax in axes[:2]:
            x_values = eps_grid if ax is axes[1] else 1.0 / eps_grid
            ax.plot(
                x_values,
                theory,
                color=color,
                label=f"{method} theory {reference}",
                **REFERENCE_STYLES["theory"],
            )
        theory_rows.append(
            {
                "method": method,
                "metric": "truncated_KDE_chi2",
                "reference_family": family,
                "fixed_parameter": DOUBLEWELL_BARRIER if method == "Langevin" else 0.5,
                "amplitude": amplitude,
                "amplitude_rule": "least-squares log amplitude over Kaplan-Meier median simulation mixing times",
            }
        )

    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$1/\varepsilon$")
    axes[0].set_ylabel("persistent chi-square mixing time")
    axes[0].set_title("simulation mixing time on Arrhenius coordinates")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].invert_xaxis()
    axes[1].set_xlabel(r"$\varepsilon$")
    axes[1].set_ylabel("persistent chi-square mixing time")
    axes[1].set_title("same simulation evidence on log-log axes")

    secondary = summary[summary["metric"] == "well_TV"].copy()
    for method in METHODS:
        sub = secondary[secondary["method"] == method].sort_values("epsilon")
        epsilon = sub["epsilon"].to_numpy(dtype=float)
        center = sub["km_median_mixing_time"].to_numpy(dtype=float)
        lo = sub["bootstrap_median_ci_low"].to_numpy(dtype=float)
        hi = sub["bootstrap_median_ci_high"].to_numpy(dtype=float)
        axes[2].errorbar(
            1.0 / epsilon,
            center,
            yerr=np.vstack([np.maximum(center - lo, 0.0), np.maximum(hi - center, 0.0)]),
            fmt=method_marker(method),
            color=method_color(method),
            ms=5.0,
            capsize=2.0,
            linestyle="none",
            label=method,
        )
        selection = selections[
            (selections["method"] == method)
            & (selections["metric"] == "well_TV")
            & (selections["epsilon_subset"] == "all")
        ]["selected_family"].iloc[0]
        row = fits[
            (fits["method"] == method)
            & (fits["metric"] == "well_TV")
            & (fits["epsilon_subset"] == "all")
            & (fits["model_family"] == selection)
        ].iloc[0]
        axes[2].plot(
            1.0 / eps_grid,
            model_curve(eps_grid, row["C_hat"], row["parameter_hat"], selection),
            color=method_color(method),
            label=f"{method} empirical {selection}",
            **REFERENCE_STYLES["empirical_fit"],
        )
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"$1/\varepsilon$")
    axes[2].set_ylabel("persistent well-TV mixing time")
    axes[2].set_title("independent inter-well communication metric")

    pivot = secondary.pivot(
        index="epsilon", columns="method", values="km_median_mixing_time"
    ).dropna()
    improvement = pivot["Langevin"] / pivot["LSC-CP"]
    axes[3].semilogy(
        1.0 / improvement.index.to_numpy(dtype=float),
        improvement.to_numpy(dtype=float),
        marker="o",
        color="#4A5568",
        lw=1.8,
    )
    axes[3].axhline(1.0, color="black", lw=1.0, ls="--")
    axes[3].set_xlabel(r"$1/\varepsilon$")
    axes[3].set_ylabel("Langevin / LSC-CP well-TV time")
    axes[3].set_title("well-TV communication improvement factor")

    for index, ax in enumerate(axes):
        panel_label(ax, "abcd"[index])
        ax.grid(alpha=0.22)
    axes[0].legend(fontsize=6.2, ncol=2)
    axes[1].legend(fontsize=6.2, ncol=2)
    axes[2].legend(fontsize=7.0, ncol=2)
    fig.tight_layout()
    output = figure_dir / "doublewell_theory_epsilon_chi2_sweep.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    theory = pd.DataFrame(theory_rows)
    theory.to_csv(table_dir / "doublewell_phase17l_theory_references.csv", index=False)
    legacy_theory = theory.copy()
    legacy_theory["quantity"] = "chi2_mixing_time"
    legacy_theory["reference_family"] = legacy_theory["method"].map(
        {
            "Langevin": "fixed_arrhenius",
            "LSC-CP": "fixed_theorem_inverse_channel",
        }
    )
    legacy_theory["shape"] = legacy_theory["method"].map(
        {
            "Langevin": "exp(DeltaV / eps)",
            "LSC-CP": "eps^(-1/2)",
        }
    )
    legacy_theory["DeltaV"] = np.where(
        legacy_theory["method"] == "Langevin", DOUBLEWELL_BARRIER, np.nan
    )
    legacy_theory["n"] = primary.groupby("method")["epsilon"].nunique().reindex(
        legacy_theory["method"]
    ).to_numpy()
    legacy_theory[
        [
            "method",
            "quantity",
            "reference_family",
            "shape",
            "DeltaV",
            "amplitude_rule",
            "amplitude",
            "n",
        ]
    ].to_csv(table_dir / "doublewell_chi2_theory_references.csv", index=False)

    all_candidates = fits[
        (fits["metric"] == "truncated_KDE_chi2") & (fits["epsilon_subset"] == "all")
    ].copy()
    all_candidates["quantity"] = "chi2_mixing_time"
    all_candidates["fit_family"] = all_candidates["model_family"]
    all_candidates["prefactor"] = all_candidates["C_hat"]
    all_candidates["barrier"] = all_candidates["Delta_hat"]
    all_candidates["exponent"] = all_candidates["alpha_hat"]
    all_candidates["r2_log"] = all_candidates["log_R2"]
    all_candidates["selected_for_display"] = all_candidates["selected_fit"]
    all_candidates["selection_rule"] = (
        "two-of-three majority across AIC, BIC, and leave-one-epsilon-out likelihood"
    )
    all_candidates[
        [
            "method",
            "quantity",
            "fit_family",
            "n_points",
            "prefactor",
            "barrier",
            "exponent",
            "r2_log",
            "AIC",
            "BIC",
            "leave_one_out_error",
            "selected_for_display",
            "selection_rule",
        ]
    ].to_csv(table_dir / "doublewell_chi2_empirical_fit_candidates.csv", index=False)

    design_rows = []
    selected_map = (
        all_candidates[all_candidates["selected_for_display"]]
        .set_index("method")["fit_family"]
        .to_dict()
    )
    for method in METHODS:
        design_rows.extend(
            [
                {
                    "panel": "a-b",
                    "method": method,
                    "series_role": "data",
                    "family": "persistent simulation mixing times",
                    "color": method_color(method),
                    "linestyle": "none",
                    "marker": method_marker(method),
                },
                {
                    "panel": "a-b",
                    "method": method,
                    "series_role": "theory_reference",
                    "family": (
                        "fixed_arrhenius"
                        if method == "Langevin"
                        else "fixed_theorem_inverse_channel"
                    ),
                    "color": method_color(method),
                    "linestyle": REFERENCE_STYLES["theory"]["linestyle"],
                    "marker": "",
                },
                {
                    "panel": "a-b",
                    "method": method,
                    "series_role": "empirical_fit",
                    "family": selected_map[method],
                    "color": method_color(method),
                    "linestyle": REFERENCE_STYLES["empirical_fit"]["linestyle"],
                    "marker": "",
                },
            ]
        )
    pd.DataFrame(design_rows).to_csv(
        table_dir / "doublewell_chi2_six_series_design.csv", index=False
    )
    return output, theory


def run_phase17l_doublewell_study(
    table_dir: Path | str,
    figure_dir: Path | str,
    *,
    profile: str = "production",
) -> dict[str, object]:
    table_dir = Path(table_dir)
    figure_dir = Path(figure_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    config = config_for_profile(profile)
    start = time.perf_counter()
    raw_parts: list[pd.DataFrame] = []
    by_seed_parts: list[pd.DataFrame] = []
    floor_parts: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []

    for epsilon_index, epsilon in enumerate(config.epsilon_values):
        epsilon_start = time.perf_counter()
        context = _target_context(epsilon, config)
        floor_frame, thresholds = _bias_floor_and_thresholds(context, config, epsilon_index)
        floor_parts.append(floor_frame)
        for metric, values in thresholds.items():
            threshold_rows.append(
                {
                    "epsilon": float(epsilon),
                    "metric": metric,
                    "bias_floor_replicates": config.bias_floor_replicates,
                    "N_particles": config.n_particles,
                    "KDE_bandwidth": float(context["bandwidth"]),
                    **values,
                }
            )
        score_grid, score_values, score_diagnostics = _levy_score_grid(epsilon, config)
        score_rows.append({"epsilon": float(epsilon), **score_diagnostics})
        initial_clouds = []
        for seed_id in range(config.n_seeds):
            init_rng = np.random.default_rng(
                config.global_seed + 50000 + 1000 * epsilon_index + seed_id
            )
            cloud = -1.0 + 0.075 * init_rng.standard_normal(config.n_particles)
            initial_clouds.append(np.clip(cloud, config.clip_left, config.clip_right))
        tasks = []
        with ThreadPoolExecutor(max_workers=config.n_jobs) as executor:
            for seed_id, cloud in enumerate(initial_clouds):
                for method in METHODS:
                    tasks.append(
                        executor.submit(
                            _simulate_one,
                            method=method,
                            epsilon=epsilon,
                            epsilon_index=epsilon_index,
                            seed_id=seed_id,
                            initial_samples=cloud,
                            context=context,
                            thresholds=thresholds,
                            score_grid=score_grid,
                            score_values=score_values,
                            config=config,
                        )
                    )
            for task in as_completed(tasks):
                raw, by_seed = task.result()
                raw_parts.append(raw)
                by_seed_parts.append(by_seed)
        elapsed = time.perf_counter() - epsilon_start
        epsilon_seed = pd.concat(by_seed_parts, ignore_index=True)
        epsilon_seed = epsilon_seed[np.isclose(epsilon_seed["epsilon"], epsilon)]
        primary = epsilon_seed[epsilon_seed["metric"] == "truncated_KDE_chi2"]
        print(
            f"Phase17L epsilon={epsilon:g}: {elapsed:.1f}s, "
            f"chi2 censored={int(primary['censored'].sum())}/{len(primary)}"
        )

    raw = pd.concat(raw_parts, ignore_index=True)
    by_seed = pd.concat(by_seed_parts, ignore_index=True)
    floors = pd.concat(floor_parts, ignore_index=True)
    thresholds = pd.DataFrame(threshold_rows)
    score_diagnostics = pd.DataFrame(score_rows)
    summary = _summarize_mixing_times(by_seed, config)
    fits, selections = _build_fit_tables(by_seed, config)
    acceptance = _acceptance_summary(fits, selections)
    censoring = (
        by_seed.groupby(["method", "metric"], as_index=False)
        .agg(
            n_points=("censored", "size"),
            n_censored=("censored", "sum"),
            censoring_fraction=("censored", "mean"),
            minimum_epsilon=("epsilon", "min"),
            maximum_epsilon=("epsilon", "max"),
        )
    )

    raw.to_csv(table_dir / "doublewell_phase17l_simulation_sweep_raw.csv", index=False)
    by_seed.to_csv(table_dir / "doublewell_phase17l_mixing_times_by_seed.csv", index=False)
    summary.to_csv(table_dir / "doublewell_phase17l_mixing_times_summary.csv", index=False)
    thresholds.to_csv(table_dir / "doublewell_phase17l_thresholds_bias_floor.csv", index=False)
    floors.to_csv(table_dir / "doublewell_phase17l_bias_floor_replicates.csv", index=False)
    censoring.to_csv(table_dir / "doublewell_phase17l_censoring_summary.csv", index=False)
    fits[fits["epsilon_subset"] == "all"].to_csv(
        table_dir / "doublewell_phase17l_fit_candidates.csv", index=False
    )
    fits.to_csv(table_dir / "doublewell_phase17l_fit_stability.csv", index=False)
    selections.to_csv(table_dir / "doublewell_phase17l_model_selection_summary.csv", index=False)
    acceptance.to_csv(table_dir / "doublewell_phase17l_scaling_acceptance_summary.csv", index=False)
    score_diagnostics.to_csv(
        table_dir / "doublewell_phase17l_score_diagnostics.csv", index=False
    )
    pd.DataFrame([asdict(config)]).to_csv(
        table_dir / "doublewell_phase17l_simulation_config.csv", index=False
    )
    figure_path, theory = generate_phase17l_figure(table_dir, figure_dir)
    elapsed_total = time.perf_counter() - start
    return {
        "config": config,
        "raw": raw,
        "by_seed": by_seed,
        "summary": summary,
        "thresholds": thresholds,
        "censoring": censoring,
        "fits": fits,
        "selections": selections,
        "acceptance": acceptance,
        "theory": theory,
        "figure_path": figure_path,
        "runtime_seconds": elapsed_total,
    }
