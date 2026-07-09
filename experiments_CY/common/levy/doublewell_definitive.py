"""Definitive low-temperature scaling study for the double-well example."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg, sparse
from scipy.ndimage import gaussian_filter1d

from .mixing_time_models import fit_diagnostics
from .plot_style import (
    REFERENCE_STYLES,
    apply_plot_style,
    method_color,
    method_marker,
    panel_label,
)


BARRIER = 0.25
METHODS = ("Langevin", "LSC-CP")
EPSILON_GRID = (
    0.40,
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
    0.055,
    0.05,
    0.045,
    0.04,
)
SUBSETS = (
    "all",
    "exclude_largest_2",
    "lowest_12",
    "lowest_10",
    "lowest_8",
    "uncensored_only",
    "reliable_rate_window_only",
)
CHI_METRICS = ("KDE_chi2", "bin_chi2_M40", "bin_chi2_M80", "bin_chi2_M120")
SUPPORT_METRICS = ("well_TV", "CDF_sup", "W1")
ALL_METRICS = CHI_METRICS + SUPPORT_METRICS


@dataclass(frozen=True)
class DefinitiveConfig:
    epsilon_values: tuple[float, ...]
    n_cells: int
    refinement_cells: tuple[int, ...]
    refinement_epsilons: tuple[float, ...]
    domain_left: float
    domain_right: float
    dense_grid_n: int
    jump_center: float
    jump_half_width: float
    jump_intensity: float
    jump_quadrature: int
    source_quadrature: int
    n_particles: int
    n_validation_particles: int
    validation_epsilons: tuple[float, ...]
    n_seeds: int
    n_records: int
    persistence_records: int
    floor_replicates: int
    bootstrap_replicates: int
    global_seed: int
    max_horizon: float


def config_for_profile(profile: str = "production") -> DefinitiveConfig:
    profile = str(profile).lower()
    if profile == "smoke":
        return DefinitiveConfig(
            epsilon_values=(0.35, 0.20, 0.10, 0.06, 0.04),
            n_cells=80,
            refinement_cells=(60, 80, 100),
            refinement_epsilons=(0.10, 0.04),
            domain_left=-5.0,
            domain_right=5.0,
            dense_grid_n=12001,
            jump_center=2.0,
            jump_half_width=0.22,
            jump_intensity=1.0,
            jump_quadrature=5,
            source_quadrature=3,
            n_particles=400,
            n_validation_particles=800,
            validation_epsilons=(0.10, 0.06, 0.04),
            n_seeds=2,
            n_records=80,
            persistence_records=3,
            floor_replicates=8,
            bootstrap_replicates=12,
            global_seed=20261713,
            max_horizon=600.0,
        )
    if profile not in {"production", "paper", "paperlite"}:
        raise ValueError(f"unknown Phase17M profile: {profile}")
    return DefinitiveConfig(
        epsilon_values=EPSILON_GRID,
        n_cells=320,
        refinement_cells=(240, 320, 400),
        refinement_epsilons=(0.40, 0.10, 0.05, 0.04),
        domain_left=-5.0,
        domain_right=5.0,
        dense_grid_n=60001,
        jump_center=2.0,
        jump_half_width=0.22,
        jump_intensity=1.0,
        jump_quadrature=9,
        source_quadrature=3,
        n_particles=6000,
        n_validation_particles=12000,
        validation_epsilons=(0.08, 0.06, 0.05, 0.04),
        n_seeds=8,
        n_records=400,
        persistence_records=4,
        floor_replicates=64,
        bootstrap_replicates=500,
        global_seed=20261713,
        max_horizon=6000.0,
    )


def potential(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    return 0.25 * values**4 - 0.5 * values**2


def _trapz_weights(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    delta = np.diff(values)
    weights = np.empty_like(values)
    weights[0] = 0.5 * delta[0]
    weights[-1] = 0.5 * delta[-1]
    weights[1:-1] = 0.5 * (delta[:-1] + delta[1:])
    return weights


def _target_context(epsilon: float, n_cells: int, config: DefinitiveConfig) -> dict[str, object]:
    x = np.linspace(config.domain_left, config.domain_right, config.dense_grid_n)
    weights = _trapz_weights(x)
    log_density = -potential(x) / float(epsilon)
    raw = np.exp(log_density - float(np.max(log_density)))
    density = raw / float(np.sum(raw * weights))
    probability = density * weights
    cdf = np.cumsum(probability)
    cdf /= float(cdf[-1])
    cdf[-1] = 1.0

    edge_probability = np.linspace(0.0, 1.0, int(n_cells) + 1)
    midpoint_probability = (np.arange(int(n_cells)) + 0.5) / float(n_cells)
    edges = np.interp(edge_probability, cdf, x)
    edges[0] = config.domain_left
    edges[-1] = config.domain_right
    centers = np.interp(midpoint_probability, cdf, x)
    internal_density = np.interp(edges[1:-1], x, density)
    mu = np.full(int(n_cells), 1.0 / float(n_cells))

    return {
        "epsilon": float(epsilon),
        "dense_x": x,
        "dense_density": density,
        "dense_weights": weights,
        "dense_cdf": cdf,
        "cell_edges": edges,
        "cell_centers": centers,
        "internal_edge_density": internal_density,
        "mu": mu,
    }


def _jump_atoms(config: DefinitiveConfig) -> tuple[np.ndarray, np.ndarray]:
    z, w = np.polynomial.legendre.leggauss(config.jump_quadrature)
    positive = config.jump_center + config.jump_half_width * z
    negative = -config.jump_center + config.jump_half_width * z
    atoms = np.concatenate([negative, positive])
    weights = np.concatenate([0.25 * w, 0.25 * w])
    weights /= float(np.sum(weights))
    return atoms, weights


def _local_reversible_generator(context: dict[str, object]) -> np.ndarray:
    epsilon = float(context["epsilon"])
    centers = np.asarray(context["cell_centers"], dtype=float)
    density_edge = np.asarray(context["internal_edge_density"], dtype=float)
    mu = np.asarray(context["mu"], dtype=float)
    conductance = epsilon * density_edge / np.diff(centers)
    q = np.zeros((len(centers), len(centers)), dtype=float)
    ids = np.arange(len(centers) - 1)
    q[ids, ids + 1] = conductance / mu[:-1]
    q[ids + 1, ids] = conductance / mu[1:]
    q[np.diag_indices_from(q)] = -np.sum(q, axis=1)
    return q


def _jump_generator(context: dict[str, object], config: DefinitiveConfig) -> np.ndarray:
    edges = np.asarray(context["cell_edges"], dtype=float)
    dense_x = np.asarray(context["dense_x"], dtype=float)
    dense_cdf = np.asarray(context["dense_cdf"], dtype=float)
    n = len(edges) - 1
    atoms, atom_weights = _jump_atoms(config)
    source_z, source_w = np.polynomial.legendre.leggauss(config.source_quadrature)
    q = np.zeros((n, n), dtype=float)
    for source in range(n):
        q_left = source / n
        q_right = (source + 1) / n
        source_probability = 0.5 * (q_left + q_right) + 0.5 * (q_right - q_left) * source_z
        source_points = np.interp(source_probability, dense_cdf, dense_x)
        source_weights = 0.5 * source_w
        for point, point_weight in zip(source_points, source_weights):
            destinations = point + atoms
            destination_ids = np.searchsorted(edges, destinations, side="right") - 1
            destination_ids = np.clip(destination_ids, 0, n - 1)
            for destination, atom_weight in zip(destination_ids, atom_weights):
                q[source, int(destination)] += (
                    config.jump_intensity * float(point_weight) * float(atom_weight)
                )
    q[np.diag_indices_from(q)] -= config.jump_intensity
    return q


def _stationary_flux_correction(jump_q: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residual = np.asarray(mu @ jump_q, dtype=float)
    flux = np.cumsum(residual)[:-1]
    correction = np.zeros_like(jump_q)
    for edge, value in enumerate(flux):
        if value >= 0.0:
            correction[edge, edge + 1] += value / mu[edge]
        else:
            correction[edge + 1, edge] += -value / mu[edge + 1]
    correction[np.diag_indices_from(correction)] = -np.sum(correction, axis=1)
    return correction, flux


def build_structure_preserving_generator(
    epsilon: float,
    method: str,
    n_cells: int,
    config: DefinitiveConfig,
) -> tuple[np.ndarray, dict[str, object], dict[str, float]]:
    context = _target_context(epsilon, n_cells, config)
    mu = np.asarray(context["mu"], dtype=float)
    local_q = _local_reversible_generator(context)
    if method == "Langevin":
        q = local_q
        raw_jump_residual = 0.0
        correction_flux_max = 0.0
    elif method == "LSC-CP":
        jump_q = _jump_generator(context, config)
        correction_q, flux = _stationary_flux_correction(jump_q, mu)
        raw_jump_residual = float(np.max(np.abs(mu @ jump_q)))
        correction_flux_max = float(np.max(np.abs(flux))) if len(flux) else 0.0
        q = local_q + jump_q + correction_q
    else:
        raise ValueError(method)

    off_diagonal = q.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    row_sum = q.sum(axis=1)
    stationary = mu @ q
    diagnostics = {
        "row_sum_inf": float(np.max(np.abs(row_sum))),
        "stationary_residual_l1": float(np.sum(np.abs(stationary))),
        "stationary_residual_linf": float(np.max(np.abs(stationary))),
        "minimum_off_diagonal": float(np.min(off_diagonal)),
        "maximum_exit_rate": float(np.max(-np.diag(q))),
        "raw_jump_stationary_residual_linf": raw_jump_residual,
        "correction_flux_max": correction_flux_max,
    }
    return q, context, diagnostics


def spectral_rates(q: np.ndarray, mu: np.ndarray) -> dict[str, object]:
    sqrt_mu = np.sqrt(np.asarray(mu, dtype=float))
    transformed = sqrt_mu[:, None] * q / sqrt_mu[None, :]
    symmetric = -0.5 * (transformed + transformed.T)
    form_values, form_vectors = linalg.eigh(symmetric, check_finite=False)
    form_values[np.abs(form_values) < 1e-11] = 0.0
    positive_form = form_values[form_values > 1e-10]
    form_gap = float(positive_form[0]) if len(positive_form) else np.nan
    form_vector = form_vectors[:, int(np.where(form_values > 1e-10)[0][0])]
    form_residual = float(
        np.linalg.norm(symmetric @ form_vector - form_gap * form_vector)
        / max(1.0, abs(form_gap))
    )

    eigenvalues = linalg.eigvals(q, check_finite=False)
    zero_index = int(np.argmin(np.abs(eigenvalues)))
    zero_value = eigenvalues[zero_index]
    mask = np.ones(len(eigenvalues), dtype=bool)
    mask[zero_index] = False
    rates = -np.real(eigenvalues[mask])
    valid = rates > 1e-10
    abscissa = float(np.min(rates[valid])) if np.any(valid) else np.nan
    return {
        "spectral_gap": form_gap,
        "form_zero_eigenvalue": float(form_values[0]),
        "form_residual": form_residual,
        "abscissa_rate": abscissa,
        "zero_eigenvalue_abs": float(abs(zero_value)),
        "minimum_nonzero_real_rate": abscissa,
        "maximum_eigenvalue_real_part": float(np.max(np.real(eigenvalues))),
    }


def _spectral_sweep(
    config: DefinitiveConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[float, str], tuple[np.ndarray, dict[str, object]]]]:
    main_rows: list[dict[str, object]] = []
    refinement_rows: list[dict[str, object]] = []
    generators: dict[tuple[float, str], tuple[np.ndarray, dict[str, object]]] = {}
    refinement_lookup: dict[tuple[float, str, int], dict[str, object]] = {}

    for epsilon in config.epsilon_values:
        for method in METHODS:
            q, context, diagnostics = build_structure_preserving_generator(
                epsilon, method, config.n_cells, config
            )
            rates = spectral_rates(q, np.asarray(context["mu"]))
            row = {
                "epsilon": epsilon,
                "method": method,
                "n_cells": config.n_cells,
                "domain_left": config.domain_left,
                "domain_right": config.domain_right,
                **diagnostics,
                **rates,
            }
            main_rows.append(row)
            generators[(epsilon, method)] = (q, context)

    for epsilon in config.refinement_epsilons:
        for n_cells in config.refinement_cells:
            for method in METHODS:
                q, context, diagnostics = build_structure_preserving_generator(
                    epsilon, method, n_cells, config
                )
                rates = spectral_rates(q, np.asarray(context["mu"]))
                row = {
                    "epsilon": epsilon,
                    "method": method,
                    "n_cells": n_cells,
                    "domain_left": config.domain_left,
                    "domain_right": config.domain_right,
                    **diagnostics,
                    **rates,
                }
                refinement_rows.append(row)
                refinement_lookup[(epsilon, method, n_cells)] = row

    refinement = pd.DataFrame(refinement_rows)
    finest = max(config.refinement_cells)
    main_n = config.n_cells
    relative_rows = []
    for row in refinement_rows:
        key = (row["epsilon"], row["method"])
        main = refinement_lookup.get((*key, main_n))
        high = refinement_lookup.get((*key, finest))
        updated = dict(row)
        if main is not None and high is not None:
            for quantity in ("spectral_gap", "abscissa_rate"):
                updated[f"{quantity}_relative_change_main_to_finest"] = abs(
                    float(main[quantity]) - float(high[quantity])
                ) / max(abs(float(high[quantity])), 1e-14)
        relative_rows.append(updated)
    refinement = pd.DataFrame(relative_rows)

    main = pd.DataFrame(main_rows)
    change_lookup = (
        refinement[refinement["n_cells"] == main_n]
        .set_index(["epsilon", "method"])
        if not refinement.empty
        else pd.DataFrame()
    )
    gap_reliable = []
    abscissa_reliable = []
    for row in main.itertuples(index=False):
        base_checks = (
            row.row_sum_inf <= 1e-10
            and row.stationary_residual_linf <= 1e-10
            and row.minimum_off_diagonal >= -1e-12
            and row.zero_eigenvalue_abs <= 1e-9
        )
        if (row.epsilon, row.method) in getattr(change_lookup, "index", []):
            ref = change_lookup.loc[(row.epsilon, row.method)]
            gap_change = float(ref["spectral_gap_relative_change_main_to_finest"])
            abscissa_change = float(ref["abscissa_rate_relative_change_main_to_finest"])
            gap_reliable.append(base_checks and gap_change <= 0.08)
            abscissa_reliable.append(base_checks and abscissa_change <= 0.08)
        else:
            gap_reliable.append(base_checks)
            abscissa_reliable.append(base_checks)
    main["spectral_gap_reliable"] = gap_reliable
    main["abscissa_reliable"] = abscissa_reliable
    return main, refinement, generators


def _bin_overlap_matrix(n_cells: int, n_bins: int) -> np.ndarray:
    matrix = np.zeros((n_cells, n_bins), dtype=float)
    for cell in range(n_cells):
        left = cell / n_cells
        right = (cell + 1) / n_cells
        first = max(0, int(math.floor(left * n_bins)))
        last = min(n_bins - 1, int(math.floor(np.nextafter(right, left) * n_bins)))
        for target_bin in range(first, last + 1):
            bin_left = target_bin / n_bins
            bin_right = (target_bin + 1) / n_bins
            overlap = max(0.0, min(right, bin_right) - max(left, bin_left))
            matrix[cell, target_bin] = overlap * n_cells
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix


def _kde_setup(context: dict[str, object], n_particles: int) -> dict[str, object]:
    dense_x = np.asarray(context["dense_x"], dtype=float)
    dense_density = np.asarray(context["dense_density"], dtype=float)
    dense_cdf = np.asarray(context["dense_cdf"], dtype=float)
    variance = float(
        np.sum((dense_x - np.sum(dense_x * dense_density * _trapz_weights(dense_x))) ** 2
               * dense_density * _trapz_weights(dense_x))
    )
    bandwidth = max(math.sqrt(variance) * n_particles ** (-0.2), 0.012)
    eval_x = np.linspace(
        float(np.interp(0.001, dense_cdf, dense_x)),
        float(np.interp(0.999, dense_cdf, dense_x)),
        512,
    )
    target = np.interp(eval_x, dense_x, dense_density)
    dx = float(eval_x[1] - eval_x[0])
    target /= float(np.trapz(target, eval_x))
    chi_left = float(np.interp(0.01, dense_cdf, dense_x))
    chi_right = float(np.interp(0.99, dense_cdf, dense_x))
    mask = (eval_x >= chi_left) & (eval_x <= chi_right)
    centers = np.asarray(context["cell_centers"], dtype=float)
    uniform_ids = np.searchsorted(eval_x, centers)
    uniform_ids = np.clip(uniform_ids, 0, len(eval_x) - 1)
    return {
        "eval_x": eval_x,
        "target": target,
        "dx": dx,
        "bandwidth": bandwidth,
        "sigma_bins": bandwidth / dx,
        "chi_mask": mask,
        "uniform_ids": uniform_ids,
    }


def _metrics_from_counts(
    counts: np.ndarray,
    context: dict[str, object],
    kde: dict[str, object],
    n_particles: int,
) -> tuple[dict[str, float], np.ndarray]:
    counts = np.asarray(counts, dtype=float)
    n_cells = len(counts)
    empirical = counts / float(n_particles)
    mu = np.asarray(context["mu"], dtype=float)
    cdf_error = np.cumsum(empirical - mu)
    centers = np.asarray(context["cell_centers"], dtype=float)
    widths = np.diff(np.concatenate([[centers[0]], centers]))
    w1 = float(np.sum(np.abs(cdf_error) * np.maximum(widths, 0.0)))
    well_tv = float(abs(np.sum(empirical[: n_cells // 2]) - 0.5))
    cdf_sup = float(np.max(np.abs(cdf_error)))

    histogram = np.zeros(len(kde["eval_x"]), dtype=float)
    np.add.at(histogram, np.asarray(kde["uniform_ids"], dtype=int), counts)
    density = histogram / (float(n_particles) * float(kde["dx"]))
    density = gaussian_filter1d(
        density,
        sigma=float(kde["sigma_bins"]),
        mode="constant",
        truncate=4.0,
    )
    integral = float(np.trapz(density, np.asarray(kde["eval_x"])))
    density /= max(integral, 1e-300)
    mask = np.asarray(kde["chi_mask"], dtype=bool)
    target = np.asarray(kde["target"], dtype=float)
    eval_x = np.asarray(kde["eval_x"], dtype=float)
    kde_chi2 = float(
        np.trapz(
            (density[mask] - target[mask]) ** 2 / np.maximum(target[mask], 1e-300),
            eval_x[mask],
        )
    )
    return {
        "KDE_chi2": kde_chi2,
        "well_TV": well_tv,
        "CDF_sup": cdf_sup,
        "W1": w1,
    }, density


def _thresholds_and_floors(
    epsilon: float,
    context: dict[str, object],
    config: DefinitiveConfig,
    epsilon_index: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    kde = _kde_setup(context, config.n_particles)
    overlap = {m: _bin_overlap_matrix(config.n_cells, m) for m in (40, 80, 120)}
    rng = np.random.default_rng(config.global_seed + 10000 + epsilon_index)
    rows: list[dict[str, object]] = []
    mu = np.asarray(context["mu"], dtype=float)
    for replicate in range(config.floor_replicates):
        counts = rng.multinomial(config.n_particles, mu)
        values, _ = _metrics_from_counts(counts, context, kde, config.n_particles)
        for m, matrix in overlap.items():
            probability = mu @ matrix
            bin_counts = rng.multinomial(config.n_particles, probability)
            values[f"bin_chi2_M{m}"] = float(
                np.sum((bin_counts / config.n_particles - 1.0 / m) ** 2 / (1.0 / m))
            )
        for metric, value in values.items():
            rows.append(
                {
                    "epsilon": epsilon,
                    "replicate": replicate,
                    "metric": metric,
                    "bias_floor_value": value,
                    "N_particles": config.n_particles,
                }
            )
    floors = pd.DataFrame(rows)
    base = {
        "KDE_chi2": 0.05,
        "bin_chi2_M40": 0.06,
        "bin_chi2_M80": 0.10,
        "bin_chi2_M120": 0.14,
        "well_TV": 0.075,
        "CDF_sup": 0.06,
        "W1": 0.08,
    }
    thresholds: dict[str, dict[str, float]] = {}
    for metric in ALL_METRICS:
        values = floors.loc[floors["metric"] == metric, "bias_floor_value"].to_numpy()
        q99 = float(np.quantile(values, 0.99))
        threshold = float(max(base[metric], 3.0 * q99))
        thresholds[metric] = {
            "bias_floor_mean": float(np.mean(values)),
            "bias_floor_q99": q99,
            "bias_floor_max": float(np.max(values)),
            "threshold": threshold,
            "threshold_above_bias_floor": bool(threshold > float(np.max(values))),
            "threshold_rule": f"max({base[metric]:g}, 3 * target-only q99 floor)",
        }
    return floors, thresholds


def _initial_probability(context: dict[str, object]) -> np.ndarray:
    edges = np.asarray(context["cell_edges"], dtype=float)
    z = (edges + 1.0) / 0.075
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    probability = np.diff(cdf)
    probability = np.maximum(probability, 0.0)
    probability /= float(np.sum(probability))
    return probability


def _propagate(q: np.ndarray, initial: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    values, vectors = linalg.eig(q.T, check_finite=False)
    coefficients = linalg.solve(vectors, initial.astype(complex), check_finite=False)
    exponentials = np.exp(values[:, None] * times[None, :])
    propagated = vectors @ (coefficients[:, None] * exponentials)
    imaginary_max = float(np.max(np.abs(np.imag(propagated))))
    probability = np.real(propagated).T
    negative_min = float(np.min(probability))
    probability = np.maximum(probability, 0.0)
    probability /= probability.sum(axis=1, keepdims=True)
    return probability, {
        "propagation_imaginary_max": imaginary_max,
        "propagation_min_before_projection": negative_min,
        "propagation_mass_error": float(np.max(np.abs(probability.sum(axis=1) - 1.0))),
        "eigenvector_condition_number": float(np.linalg.cond(vectors)),
    }


def _persistent_crossing(times: np.ndarray, values: np.ndarray, threshold: float, persistence: int) -> tuple[float, bool, float]:
    below = np.asarray(values, dtype=float) <= float(threshold)
    for start in range(0, len(below) - persistence + 1):
        if np.all(below[start : start + persistence]):
            first = float(times[np.where(below)[0][0]]) if np.any(below) else np.nan
            return float(times[start]), True, first
    first = float(times[np.where(below)[0][0]]) if np.any(below) else np.nan
    return np.nan, False, first


def _simulate_epsilon(
    epsilon: float,
    epsilon_index: int,
    generators: dict[tuple[float, str], tuple[np.ndarray, dict[str, object]]],
    spectral: pd.DataFrame,
    thresholds: dict[str, dict[str, float]],
    config: DefinitiveConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    time_rows: list[dict[str, object]] = []
    mixing_rows: list[dict[str, object]] = []
    propagation_rows: list[dict[str, object]] = []
    overlap = {m: _bin_overlap_matrix(config.n_cells, m) for m in (40, 80, 120)}

    for method_index, method in enumerate(METHODS):
        q, context = generators[(epsilon, method)]
        rates = spectral[
            np.isclose(spectral["epsilon"], epsilon) & (spectral["method"] == method)
        ].iloc[0]
        pilot = min(float(rates["spectral_gap"]), float(rates["abscissa_rate"]))
        horizon = float(min(config.max_horizon, max(16.0, 6.0 / max(pilot, 1e-12))))
        times = np.linspace(0.0, horizon, config.n_records + 1)
        probability, propagation = _propagate(q, _initial_probability(context), times)
        propagation_rows.append(
            {
                "epsilon": epsilon,
                "method": method,
                "T_final": horizon,
                "record_dt": float(times[1] - times[0]),
                **propagation,
            }
        )
        kde = _kde_setup(context, config.n_particles)

        for seed in range(config.n_seeds):
            rng = np.random.default_rng(
                config.global_seed
                + 100000
                + 10000 * epsilon_index
                + 1000 * method_index
                + seed
            )
            curves = {metric: [] for metric in ALL_METRICS}
            for time_value, state_probability in zip(times, probability):
                counts = rng.multinomial(config.n_particles, state_probability)
                values, _ = _metrics_from_counts(counts, context, kde, config.n_particles)
                for m, matrix in overlap.items():
                    bin_probability = state_probability @ matrix
                    bin_counts = rng.multinomial(config.n_particles, bin_probability)
                    values[f"bin_chi2_M{m}"] = float(
                        np.sum(
                            (bin_counts / config.n_particles - 1.0 / m) ** 2
                            / (1.0 / m)
                        )
                    )
                for metric, value in values.items():
                    curves[metric].append(value)
                    time_rows.append(
                        {
                            "epsilon": epsilon,
                            "method": method,
                            "seed": seed,
                            "metric": metric,
                            "time": float(time_value),
                            "metric_value": float(value),
                            "threshold": thresholds[metric]["threshold"],
                            "bias_floor": thresholds[metric]["bias_floor_q99"],
                            "N_particles": config.n_particles,
                            "T_final": horizon,
                        }
                    )
            for metric, values in curves.items():
                mixing_time, reached, first = _persistent_crossing(
                    times,
                    np.asarray(values),
                    thresholds[metric]["threshold"],
                    config.persistence_records,
                )
                mixing_rows.append(
                    {
                        "epsilon": epsilon,
                        "method": method,
                        "seed": seed,
                        "metric": metric,
                        "threshold": thresholds[metric]["threshold"],
                        "bias_floor": thresholds[metric]["bias_floor_q99"],
                        "first_crossing_time": first,
                        "persistent_reached": reached,
                        "censored": not reached,
                        "mixing_time": mixing_time,
                        "final_metric_value": float(values[-1]),
                        "N_particles": config.n_particles,
                        "n_recorded_points": len(times),
                        "persistence_records": config.persistence_records,
                        "record_dt": float(times[1] - times[0]),
                        "T_final": horizon,
                    }
                )
    return pd.DataFrame(time_rows), pd.DataFrame(mixing_rows), pd.DataFrame(propagation_rows)


def _summarize_time_series(raw: pd.DataFrame) -> pd.DataFrame:
    summary = raw.groupby(
        ["epsilon", "method", "metric", "time"], as_index=False
    ).agg(
        mean=("metric_value", "mean"),
        std=("metric_value", "std"),
        median=("metric_value", "median"),
        q10=("metric_value", lambda x: np.quantile(x, 0.10)),
        q90=("metric_value", lambda x: np.quantile(x, 0.90)),
        n_seeds=("metric_value", "size"),
    )
    metadata = raw.groupby(["epsilon", "method", "metric"], as_index=False).first()[
        ["epsilon", "method", "metric", "threshold", "bias_floor", "N_particles", "T_final"]
    ]
    return summary.merge(metadata, on=["epsilon", "method", "metric"], how="left")


def _summarize_mixing(mixing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in mixing.groupby(["epsilon", "method", "metric"], sort=True):
        epsilon, method, metric = keys
        observed = group.loc[~group["censored"].astype(bool), "mixing_time"].to_numpy(dtype=float)
        center = float(np.median(observed)) if len(observed) else np.nan
        lo = float(np.quantile(observed, 0.10)) if len(observed) else np.nan
        hi = float(np.quantile(observed, 0.90)) if len(observed) else np.nan
        rows.append(
            {
                "epsilon": epsilon,
                "method": method,
                "metric": metric,
                "median_mixing_time": center,
                "q10_mixing_time": lo,
                "q90_mixing_time": hi,
                "n_points": len(group),
                "n_censored": int(group["censored"].sum()),
                "censoring_fraction": float(group["censored"].mean()),
                "maximum_censoring_time": float(group["T_final"].max()),
                "threshold": float(group["threshold"].iloc[0]),
                "bias_floor": float(group["bias_floor"].iloc[0]),
                "N_particles": int(group["N_particles"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _contiguous_blocks(indices: np.ndarray) -> list[np.ndarray]:
    if len(indices) == 0:
        return []
    splits = np.where(np.diff(indices) > 1)[0] + 1
    return [block for block in np.split(indices, splits) if len(block)]


def _fit_rate_window(
    times: np.ndarray,
    values: np.ndarray,
    threshold: float,
    floor: float,
) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    upper = min(0.8, 0.5 * float(values[0]))
    eligible = np.where(
        (times > 0.0)
        & np.isfinite(values)
        & (values >= max(2.0 * floor, threshold))
        & (values <= upper)
    )[0]
    blocks = _contiguous_blocks(eligible)
    blocks.sort(key=lambda block: (-len(block), int(block[0])))
    if not blocks or len(blocks[0]) < 8:
        return {
            "accepted": False,
            "acceptance_reason": "fewer than eight eligible contiguous points",
            "n_fit": len(blocks[0]) if blocks else 0,
        }
    block = blocks[0]
    x = times[block]
    y = np.log(values[block])
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    prediction = design @ beta
    residual = y - prediction
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    slope = float(beta[1])
    span = float(x[-1] - x[0])
    sxx = float(np.sum((x - np.mean(x)) ** 2))
    sigma2 = ss_res / max(len(x) - 2, 1)
    slope_se = float(math.sqrt(sigma2 / max(sxx, 1e-300)))
    accepted = bool(slope < 0.0 and r2 >= 0.95 and span > 0.0)
    predicted_time = (
        float(x[0] + math.log(values[block[0]] / threshold) / (-slope))
        if slope < 0.0 and values[block[0]] > threshold
        else np.nan
    )
    return {
        "accepted": accepted,
        "acceptance_reason": "accepted" if accepted else "negative slope or R2 criterion failed",
        "n_fit": len(x),
        "window_start": float(x[0]),
        "window_end": float(x[-1]),
        "window_initial_value": float(values[block[0]]),
        "window_final_value": float(values[block[-1]]),
        "slope_log_chi2": slope,
        "slope_standard_error": slope_se,
        "lambda_eff": float(-0.5 * slope),
        "log_R2": float(r2),
        "residual_RMSE_log": float(math.sqrt(np.mean(residual**2))),
        "predicted_threshold_time": predicted_time,
        "noise_cutoff": float(2.0 * floor),
        "upper_window_cutoff": upper,
    }


def _rate_windows(raw: pd.DataFrame, mixing: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (epsilon, method, metric, seed), group in raw[
        raw["metric"].isin(CHI_METRICS)
    ].groupby(["epsilon", "method", "metric", "seed"]):
        group = group.sort_values("time")
        result = _fit_rate_window(
            group["time"].to_numpy(),
            group["metric_value"].to_numpy(),
            float(group["threshold"].iloc[0]),
            float(group["bias_floor"].iloc[0]),
        )
        observed = mixing[
            np.isclose(mixing["epsilon"], epsilon)
            & (mixing["method"] == method)
            & (mixing["metric"] == metric)
            & (mixing["seed"] == seed)
        ].iloc[0]
        rows.append(
            {
                "epsilon": epsilon,
                "method": method,
                "metric": metric,
                "estimator": "seed",
                "seed": seed,
                "observed_mixing_time": observed["mixing_time"],
                "censored": observed["censored"],
                **result,
            }
        )
    for (epsilon, method, metric), group in raw[
        raw["metric"].isin(CHI_METRICS)
    ].groupby(["epsilon", "method", "metric"]):
        mean_curve = group.groupby("time", as_index=False)["metric_value"].mean()
        result = _fit_rate_window(
            mean_curve["time"].to_numpy(),
            mean_curve["metric_value"].to_numpy(),
            float(group["threshold"].iloc[0]),
            float(group["bias_floor"].iloc[0]),
        )
        observed = mixing[
            np.isclose(mixing["epsilon"], epsilon)
            & (mixing["method"] == method)
            & (mixing["metric"] == metric)
            & (~mixing["censored"].astype(bool))
        ]["mixing_time"]
        observed_center = float(np.median(observed)) if len(observed) else np.nan
        predicted = float(result.get("predicted_threshold_time", np.nan))
        rows.append(
            {
                "epsilon": epsilon,
                "method": method,
                "metric": metric,
                "estimator": "seed_mean",
                "seed": -1,
                "observed_mixing_time": observed_center,
                "censored": bool(len(observed) == 0),
                "predicted_to_observed_ratio": (
                    predicted / observed_center
                    if np.isfinite(predicted) and np.isfinite(observed_center) and observed_center > 0
                    else np.nan
                ),
                **result,
            }
        )
    return pd.DataFrame(rows)


def _subset(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    values = np.array(sorted(frame["epsilon"].unique()), dtype=float)
    if name == "all":
        return frame.copy()
    if name == "exclude_largest_2":
        return frame[frame["epsilon"].isin(values[:-2])].copy()
    if name.startswith("lowest_"):
        count = int(name.split("_")[1])
        return frame[frame["epsilon"].isin(values[: min(count, len(values))])].copy()
    if name == "uncensored_only":
        return frame[~frame.get("censored", False).astype(bool)].copy()
    if name == "reliable_rate_window_only":
        if "accepted" in frame:
            return frame[frame["accepted"].astype(bool)].copy()
        return frame.copy()
    raise ValueError(name)


def _log_fit(
    frame: pd.DataFrame,
    family: str,
    *,
    value_column: str,
    kind: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, object]:
    data = frame[["epsilon", value_column]].dropna().copy()
    data = data[(data["epsilon"] > 0) & (data[value_column] > 0)]
    epsilon = data["epsilon"].to_numpy(dtype=float)
    log_value = np.log(data[value_column].to_numpy(dtype=float))
    if family == "arrhenius":
        x = (1.0 if kind == "time" else -1.0) / epsilon
    elif family == "polynomial":
        x = (-1.0 if kind == "time" else 1.0) * np.log(epsilon)
    else:
        raise ValueError(family)
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design, log_value, rcond=None)
    prediction = design @ beta
    residual = log_value - prediction
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((log_value - np.mean(log_value)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    sigma2 = ss_res / max(len(x) - 2, 1)
    sxx = float(np.sum((x - np.mean(x)) ** 2))
    se = float(math.sqrt(sigma2 / max(sxx, 1e-300)))
    sigma = max(math.sqrt(ss_res / max(len(x), 1)), 1e-12)
    log_likelihood = float(
        -0.5 * len(x) * math.log(2.0 * math.pi * sigma**2)
        - 0.5 * ss_res / sigma**2
    )
    k = 3
    aic = float(2 * k - 2 * log_likelihood)
    bic = float(k * math.log(max(len(x), 1)) - 2 * log_likelihood)

    loo = []
    for held in np.unique(epsilon):
        mask = ~np.isclose(epsilon, held)
        if np.count_nonzero(mask) < 3:
            continue
        fit_beta, *_ = np.linalg.lstsq(design[mask], log_value[mask], rcond=None)
        loo.extend((log_value[~mask] - design[~mask] @ fit_beta) ** 2)
    loo_error = float(np.mean(loo)) if loo else np.nan

    rng = np.random.default_rng(seed)
    parameters = []
    amplitudes = []
    for _ in range(int(bootstrap_replicates)):
        ids = rng.integers(0, len(data), size=len(data))
        xb = x[ids]
        yb = log_value[ids]
        if np.unique(xb).size < 3:
            continue
        db = np.column_stack([np.ones(len(xb)), xb])
        bb, *_ = np.linalg.lstsq(db, yb, rcond=None)
        amplitudes.append(float(np.exp(bb[0])))
        parameters.append(float(bb[1]))
    parameter_ci = (
        np.quantile(parameters, [0.025, 0.975]) if parameters else [np.nan, np.nan]
    )
    amplitude_ci = (
        np.quantile(amplitudes, [0.025, 0.975]) if amplitudes else [np.nan, np.nan]
    )
    return {
        "model_family": family,
        "n_points": len(data),
        "n_epsilon": int(data["epsilon"].nunique()),
        "n_censored": 0,
        "C_hat": float(np.exp(beta[0])),
        "parameter_name": "Delta_hat" if family == "arrhenius" else "alpha_hat",
        "parameter_hat": float(beta[1]),
        "Delta_hat": float(beta[1]) if family == "arrhenius" else np.nan,
        "alpha_hat": float(beta[1]) if family == "polynomial" else np.nan,
        "standard_error": se,
        "bootstrap_ci_low": float(parameter_ci[0]),
        "bootstrap_ci_high": float(parameter_ci[1]),
        "bootstrap_C_ci_low": float(amplitude_ci[0]),
        "bootstrap_C_ci_high": float(amplitude_ci[1]),
        "bootstrap_successes": len(parameters),
        "log_likelihood": log_likelihood,
        "log_RSS": float(math.log(max(ss_res, 1e-300))),
        "log_R2": float(r2),
        "AIC": aic,
        "BIC": bic,
        "leave_one_out_error": loo_error,
        "residual_RMSE_log": float(math.sqrt(np.mean(residual**2))),
        "converged": True,
    }


def _select_pair(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    pair = {row["model_family"]: row for row in rows}
    arr = pair["arrhenius"]
    poly = pair["polynomial"]
    winners = {
        "AIC": "arrhenius" if arr["AIC"] < poly["AIC"] else "polynomial",
        "BIC": "arrhenius" if arr["BIC"] < poly["BIC"] else "polynomial",
        "LOO": (
            "arrhenius"
            if arr["leave_one_out_error"] < poly["leave_one_out_error"]
            else "polynomial"
        ),
    }
    arr_votes = sum(value == "arrhenius" for value in winners.values())
    return {
        "preferred_AIC": winners["AIC"],
        "preferred_BIC": winners["BIC"],
        "preferred_LOO": winners["LOO"],
        "selected_family": "arrhenius" if arr_votes >= 2 else "polynomial",
        "arrhenius_votes": arr_votes,
        "polynomial_votes": 3 - arr_votes,
        "delta_AIC_poly_minus_arr": poly["AIC"] - arr["AIC"],
        "delta_BIC_poly_minus_arr": poly["BIC"] - arr["BIC"],
        "delta_LOO_poly_minus_arr": poly["leave_one_out_error"]
        - arr["leave_one_out_error"],
    }


def _fit_tables(
    spectral: pd.DataFrame,
    mixing: pd.DataFrame,
    windows: pd.DataFrame,
    config: DefinitiveConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    sources: list[tuple[str, str, str, str, pd.DataFrame, str]] = []
    for quantity, column, reliability in (
        ("spectral_gap", "spectral_gap", "spectral_gap_reliable"),
        ("abscissa_rate", "abscissa_rate", "abscissa_reliable"),
    ):
        for method in METHODS:
            frame = spectral[
                (spectral["method"] == method) & spectral[reliability].astype(bool)
            ][["epsilon", column]].copy()
            sources.append((method, quantity, quantity, "deterministic", frame, "rate"))

    accepted_windows = windows[
        (windows["estimator"] == "seed_mean") & windows["accepted"].fillna(False).astype(bool)
    ]
    for method in METHODS:
        for metric in CHI_METRICS:
            frame = accepted_windows[
                (accepted_windows["method"] == method)
                & (accepted_windows["metric"] == metric)
            ][["epsilon", "lambda_eff", "accepted"]].copy()
            sources.append(
                (method, "chi2_effective_rate", metric, "seed_mean_window", frame, "rate")
            )
            time_frame = mixing[
                (mixing["method"] == method) & (mixing["metric"] == metric)
            ].copy()
            sources.append(
                (method, "mixing_time", metric, "particle_threshold", time_frame, "time")
            )

    for source_index, (method, quantity, metric, estimator, frame, kind) in enumerate(sources):
        value_column = {
            "spectral_gap": "spectral_gap",
            "abscissa_rate": "abscissa_rate",
            "chi2_effective_rate": "lambda_eff",
            "mixing_time": "mixing_time",
        }[quantity]
        for subset_name in SUBSETS:
            try:
                selected = _subset(frame, subset_name)
            except (AttributeError, TypeError):
                selected = frame.copy()
            if kind == "time" and subset_name == "reliable_rate_window_only":
                reliable_eps = accepted_windows[
                    (accepted_windows["method"] == method)
                    & (accepted_windows["metric"] == metric)
                ]["epsilon"]
                selected = selected[selected["epsilon"].isin(reliable_eps)].copy()
            pair_rows = []
            for family_index, family in enumerate(("arrhenius", "polynomial")):
                if selected["epsilon"].nunique() < 3 or len(selected) < 4:
                    break
                try:
                    if kind == "time":
                        diagnostics = fit_diagnostics(
                            selected,
                            family,
                            n_bootstrap=config.bootstrap_replicates,
                            bootstrap_seed=config.global_seed
                            + 500000
                            + 1000 * source_index
                            + 10 * family_index,
                        )
                    else:
                        diagnostics = _log_fit(
                            selected,
                            family,
                            value_column=value_column,
                            kind=kind,
                            bootstrap_replicates=config.bootstrap_replicates,
                            seed=config.global_seed
                            + 600000
                            + 1000 * source_index
                            + 10 * family_index,
                        )
                except (ValueError, RuntimeError, np.linalg.LinAlgError):
                    break
                row = {
                    "method": method,
                    "quantity": quantity,
                    "metric": metric,
                    "estimator": estimator,
                    "epsilon_subset": subset_name,
                    **diagnostics,
                }
                fit_rows.append(row)
                pair_rows.append(row)
            if len(pair_rows) != 2:
                continue
            selection_rows.append(
                {
                    "method": method,
                    "quantity": quantity,
                    "metric": metric,
                    "estimator": estimator,
                    "epsilon_subset": subset_name,
                    **_select_pair(pair_rows),
                }
            )
    fits = pd.DataFrame(fit_rows)
    selections = pd.DataFrame(selection_rows)
    lookup = selections.set_index(
        ["method", "quantity", "metric", "estimator", "epsilon_subset"]
    )["selected_family"]
    fits["selected_fit"] = fits.apply(
        lambda row: row["model_family"]
        == lookup.loc[
            (
                row["method"],
                row["quantity"],
                row["metric"],
                row["estimator"],
                row["epsilon_subset"],
            )
        ],
        axis=1,
    )
    return fits, selections


def _choose_main_metric(
    windows: pd.DataFrame,
    mixing_summary: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    rows = []
    for metric in ("bin_chi2_M80", "KDE_chi2"):
        low = np.array(sorted(windows["epsilon"].unique()))[:8]
        accepted = windows[
            (windows["metric"] == metric)
            & (windows["estimator"] == "seed_mean")
            & (windows["method"].isin(METHODS))
            & (windows["epsilon"].isin(low))
            & windows["accepted"].fillna(False).astype(bool)
        ]
        counts = accepted.groupby("method")["epsilon"].nunique().to_dict()
        censor = mixing_summary[
            (mixing_summary["metric"] == metric)
            & (mixing_summary["method"] == "LSC-CP")
        ]["n_censored"].sum()
        qualified = all(counts.get(method, 0) >= 6 for method in METHODS)
        rows.append(
            {
                "metric": metric,
                "accepted_low8_Langevin": counts.get("Langevin", 0),
                "accepted_low8_LSC_CP": counts.get("LSC-CP", 0),
                "LSC_CP_n_censored": int(censor),
                "qualified": qualified,
            }
        )
    audit = pd.DataFrame(rows)
    bin_row = audit[audit["metric"] == "bin_chi2_M80"].iloc[0]
    kde_row = audit[audit["metric"] == "KDE_chi2"].iloc[0]
    if bool(bin_row["qualified"]) and int(bin_row["LSC_CP_n_censored"]) <= int(
        kde_row["LSC_CP_n_censored"]
    ):
        selected = "bin_chi2_M80"
    elif bool(kde_row["qualified"]):
        selected = "KDE_chi2"
    else:
        selected = "bin_chi2_M80"
    audit["selected_for_main"] = audit["metric"] == selected
    return selected, audit


def _focused_particle_validation(
    generators: dict[tuple[float, str], tuple[np.ndarray, dict[str, object]]],
    spectral: pd.DataFrame,
    config: DefinitiveConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_config = replace(
        config,
        n_particles=config.n_validation_particles,
        floor_replicates=max(32, config.floor_replicates),
    )
    mixing_parts = []
    for epsilon_index, epsilon in enumerate(config.validation_epsilons):
        context = generators[(epsilon, "Langevin")][1]
        _, thresholds = _thresholds_and_floors(
            epsilon, context, validation_config, 900 + epsilon_index
        )
        _, mixing, _ = _simulate_epsilon(
            epsilon,
            900 + epsilon_index,
            generators,
            spectral,
            thresholds,
            validation_config,
        )
        mixing_parts.append(mixing[mixing["metric"].str.startswith("bin_chi2")])
    mixing = pd.concat(mixing_parts, ignore_index=True)
    summary = _summarize_mixing(mixing)

    selection_rows = []
    for method in METHODS:
        for metric in ("bin_chi2_M40", "bin_chi2_M80", "bin_chi2_M120"):
            frame = mixing[
                (mixing["method"] == method) & (mixing["metric"] == metric)
            ]
            pair = []
            for family_index, family in enumerate(("arrhenius", "polynomial")):
                diagnostics = fit_diagnostics(
                    frame,
                    family,
                    n_bootstrap=max(50, config.bootstrap_replicates // 5),
                    bootstrap_seed=config.global_seed
                    + 800000
                    + 1000 * family_index
                    + 10 * (0 if method == "Langevin" else 1),
                )
                pair.append({"model_family": family, **diagnostics})
            selection_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "N_particles": validation_config.n_particles,
                    "epsilon_subset": "focused_low_temperature",
                    **_select_pair(pair),
                }
            )
    return summary, pd.DataFrame(selection_rows)


def _rate_consistency_table(
    spectral: pd.DataFrame,
    mixing_summary: pd.DataFrame,
    windows: pd.DataFrame,
    selected_metric: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gap = spectral[
        [
            "epsilon",
            "method",
            "spectral_gap",
            "spectral_gap_reliable",
            "abscissa_rate",
            "abscissa_reliable",
        ]
    ].rename(
        columns={
            "spectral_gap": "lambda_form",
            "abscissa_rate": "lambda_abs",
        }
    )
    mixing = mixing_summary[mixing_summary["metric"] == selected_metric].copy()
    mean_windows = windows[
        (windows["metric"] == selected_metric) & (windows["estimator"] == "seed_mean")
    ].copy()
    frame = (
        mean_windows.merge(gap, on=["epsilon", "method"], how="left")
        .merge(
            mixing[
                [
                    "epsilon",
                    "method",
                    "median_mixing_time",
                    "threshold",
                    "bias_floor",
                    "n_censored",
                    "N_particles",
                ]
            ],
            on=["epsilon", "method"],
            how="left",
        )
        .sort_values(["epsilon", "method"])
        .reset_index(drop=True)
    )

    frame["selected_chi2_metric"] = selected_metric
    frame["T_mix"] = frame["median_mixing_time"]
    frame["chi2_window_start_time"] = frame["window_start"]
    frame["chi2_window_end_time"] = frame["window_end"]
    frame["chi2_window_initial_value"] = frame["window_initial_value"]
    frame["chi2_window_final_value"] = frame["window_final_value"]
    frame["log_factor"] = np.log(
        frame["chi2_window_initial_value"].astype(float) / frame["threshold"].astype(float)
    )
    frame.loc[
        ~np.isfinite(frame["log_factor"]) | (frame["log_factor"] <= 0.0),
        "log_factor",
    ] = np.nan
    frame["lambda_mix"] = frame["log_factor"] / (2.0 * frame["T_mix"])
    frame["observed_mixing_time"] = frame["T_mix"]
    frame["predicted_threshold_time_from_lambda_eff"] = frame[
        "predicted_threshold_time"
    ]
    frame["lambda_eff_over_lambda_abs"] = frame["lambda_eff"] / frame["lambda_abs"]
    frame["lambda_mix_over_lambda_abs"] = frame["lambda_mix"] / frame["lambda_abs"]
    frame["lambda_form_over_lambda_abs"] = frame["lambda_form"] / frame["lambda_abs"]
    frame["lambda_mix_over_lambda_eff"] = frame["lambda_mix"] / frame["lambda_eff"]
    frame["rate_window_accepted"] = frame["accepted"].fillna(False).astype(bool)
    frame["rate_window_R2"] = frame["log_R2"]
    frame["rate_window_n_points"] = frame["n_fit"]
    frame["floor_region_excluded"] = (
        frame["rate_window_accepted"]
        & (frame["chi2_window_final_value"] >= frame["noise_cutoff"])
        & (frame["noise_cutoff"] >= 2.0 * frame["bias_floor"])
    )
    frame["plateau_excluded"] = (
        frame["rate_window_accepted"]
        & (frame["chi2_window_final_value"] >= frame["threshold"])
        & (frame["chi2_window_end_time"] < frame["observed_mixing_time"])
    )
    frame["initial_transient_excluded"] = (
        frame["rate_window_accepted"]
        & (frame["chi2_window_start_time"] > 0.0)
        & (frame["chi2_window_initial_value"] <= frame["upper_window_cutoff"])
    )
    frame["predicted_agrees_with_observed"] = frame[
        "predicted_to_observed_ratio"
    ].between(0.75, 1.25)
    ratio_columns = [
        "lambda_eff_over_lambda_abs",
        "lambda_mix_over_lambda_abs",
        "lambda_form_over_lambda_abs",
        "lambda_mix_over_lambda_eff",
    ]
    frame["rate_ratios_reasonable"] = frame[ratio_columns].apply(
        lambda row: bool(np.all(np.isfinite(row)) and np.all((row >= 0.25) & (row <= 4.0))),
        axis=1,
    )
    frame["consistency_pass"] = (
        frame["rate_window_accepted"]
        & (frame["rate_window_n_points"] >= 8)
        & (frame["rate_window_R2"] >= 0.95)
        & (frame["slope_log_chi2"] < 0.0)
        & frame["floor_region_excluded"]
        & frame["plateau_excluded"]
        & frame["initial_transient_excluded"]
        & frame["predicted_agrees_with_observed"]
        & frame["rate_ratios_reasonable"]
        & frame["spectral_gap_reliable"].fillna(False).astype(bool)
        & frame["abscissa_reliable"].fillna(False).astype(bool)
        & (frame["threshold"] > frame["bias_floor"])
    )

    def reason(row: pd.Series) -> str:
        if bool(row["consistency_pass"]):
            return "none"
        reasons = []
        checks = [
            ("rate window rejected", bool(row["rate_window_accepted"])),
            ("too few window points", int(row.get("rate_window_n_points", 0)) >= 8),
            ("log-linear R2 below 0.95", float(row.get("rate_window_R2", np.nan)) >= 0.95),
            ("floor region not excluded", bool(row["floor_region_excluded"])),
            ("plateau not excluded", bool(row["plateau_excluded"])),
            ("initial transient not excluded", bool(row["initial_transient_excluded"])),
            ("predicted threshold time outside 25 percent tolerance", bool(row["predicted_agrees_with_observed"])),
            ("rate ratios outside factor-four tolerance", bool(row["rate_ratios_reasonable"])),
            ("spectral or abscissa reliability failed", bool(row["spectral_gap_reliable"]) and bool(row["abscissa_reliable"])),
            ("threshold not above bias floor", float(row["threshold"]) > float(row["bias_floor"])),
        ]
        for message, passed in checks:
            if not passed:
                reasons.append(message)
        return "; ".join(reasons)

    frame["failure_reason"] = frame.apply(reason, axis=1)
    ordered = [
        "epsilon",
        "method",
        "selected_chi2_metric",
        "lambda_form",
        "lambda_abs",
        "lambda_eff",
        "lambda_mix",
        "T_mix",
        "chi2_window_start_time",
        "chi2_window_end_time",
        "chi2_window_initial_value",
        "chi2_window_final_value",
        "threshold",
        "bias_floor",
        "noise_cutoff",
        "lambda_eff_over_lambda_abs",
        "lambda_mix_over_lambda_abs",
        "lambda_form_over_lambda_abs",
        "lambda_mix_over_lambda_eff",
        "log_factor",
        "predicted_threshold_time_from_lambda_eff",
        "observed_mixing_time",
        "predicted_to_observed_ratio",
        "rate_window_accepted",
        "rate_window_R2",
        "rate_window_n_points",
        "plateau_excluded",
        "floor_region_excluded",
        "initial_transient_excluded",
        "consistency_pass",
        "failure_reason",
        "n_censored",
        "N_particles",
        "slope_log_chi2",
        "slope_standard_error",
        "residual_RMSE_log",
        "upper_window_cutoff",
    ]
    consistency = frame[ordered].copy()

    summary_rows = []
    for method, group in consistency.groupby("method"):
        row: dict[str, object] = {
            "method": method,
            "selected_chi2_metric": selected_metric,
            "n_epsilon": int(group["epsilon"].nunique()),
            "n_consistency_pass": int(group["consistency_pass"].sum()),
            "all_consistency_pass": bool(group["consistency_pass"].all()),
            "min_rate_window_R2": float(group["rate_window_R2"].min()),
            "min_rate_window_n_points": int(group["rate_window_n_points"].min()),
            "min_predicted_to_observed_ratio": float(
                group["predicted_to_observed_ratio"].min()
            ),
            "max_predicted_to_observed_ratio": float(
                group["predicted_to_observed_ratio"].max()
            ),
        }
        for column in ratio_columns:
            row[f"min_{column}"] = float(group[column].min())
            row[f"median_{column}"] = float(group[column].median())
            row[f"max_{column}"] = float(group[column].max())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    available = np.array(sorted(consistency["epsilon"].unique()), dtype=float)
    targets = (0.10, 0.06, 0.05, 0.04)
    representatives = []
    for target in targets:
        epsilon = float(available[np.argmin(np.abs(available - target))])
        representatives.append(consistency[np.isclose(consistency["epsilon"], epsilon)])
    representative_table = (
        pd.concat(representatives, ignore_index=True)
        .sort_values(["epsilon", "method"])
        .reset_index(drop=True)
    )
    return consistency, summary, representative_table


def _fixed_amplitude(epsilon: np.ndarray, values: np.ndarray, shape: np.ndarray) -> float:
    mask = (
        np.isfinite(epsilon)
        & np.isfinite(values)
        & np.isfinite(shape)
        & (values > 0)
        & (shape > 0)
    )
    return float(np.exp(np.mean(np.log(values[mask]) - np.log(shape[mask]))))


def _curve(epsilon: np.ndarray, row: pd.Series, kind: str) -> np.ndarray:
    if row["model_family"] == "arrhenius":
        sign = 1.0 if kind == "time" else -1.0
        return float(row["C_hat"]) * np.exp(
            sign * float(row["parameter_hat"]) / epsilon
        )
    sign = -1.0 if kind == "time" else 1.0
    return float(row["C_hat"]) * epsilon ** (
        sign * float(row["parameter_hat"])
    )


def generate_main_figure(
    table_dir: Path,
    figure_dir: Path,
    selected_metric: str,
) -> tuple[Path, pd.DataFrame]:
    spectral = pd.read_csv(table_dir / "doublewell_phase17m_spectral_gap_rates.csv")
    abscissa = pd.read_csv(table_dir / "doublewell_phase17m_abscissa_rates.csv")
    mixing = pd.read_csv(table_dir / "doublewell_phase17m_mixing_times_summary.csv")
    fits = pd.read_csv(table_dir / "doublewell_phase17m_fit_stability.csv")
    selections = pd.read_csv(table_dir / "doublewell_phase17m_model_selection_summary.csv")
    apply_plot_style(plt)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    eps_curve = np.linspace(
        min(spectral["epsilon"].min(), mixing["epsilon"].min()),
        max(spectral["epsilon"].max(), mixing["epsilon"].max()),
        500,
    )
    series_rows: list[dict[str, object]] = []
    panel_specs = [
        ("spectral_gap", spectral, "spectral_gap", "rate", "spectral gap"),
        ("abscissa", abscissa, "abscissa_rate", "rate", "generator abscissa"),
        ("mixing_time", mixing[mixing["metric"] == selected_metric], "median_mixing_time", "time", "mixing time"),
    ]

    for panel_index, (panel, data, value_column, kind, title) in enumerate(panel_specs):
        ax = axes[panel_index]
        for method in METHODS:
            sub = data[data["method"] == method].sort_values("epsilon")
            epsilon = sub["epsilon"].to_numpy(dtype=float)
            values = sub[value_column].to_numpy(dtype=float)
            color = method_color(method)
            marker = method_marker(method)
            if kind == "time":
                lo = sub["q10_mixing_time"].to_numpy(dtype=float)
                hi = sub["q90_mixing_time"].to_numpy(dtype=float)
                exact = np.isfinite(values)
                ax.errorbar(
                    1.0 / epsilon[exact],
                    values[exact],
                    yerr=np.vstack(
                        [
                            np.maximum(values[exact] - lo[exact], 0.0),
                            np.maximum(hi[exact] - values[exact], 0.0),
                        ]
                    ),
                    fmt=marker,
                    color=color,
                    mfc=color,
                    mec=color,
                    ms=5,
                    capsize=2,
                    linestyle="none",
                    label=f"{method} data",
                )
                censored = sub["n_censored"].to_numpy(dtype=int) > 0
                if np.any(censored):
                    lower = sub["maximum_censoring_time"].to_numpy(dtype=float)
                    ax.errorbar(
                        1.0 / epsilon[censored],
                        lower[censored],
                        yerr=0.08 * lower[censored],
                        lolims=True,
                        fmt=marker,
                        color=color,
                        mfc="white",
                        mec=color,
                        ms=5,
                        linestyle="none",
                    )
            else:
                ax.plot(
                    1.0 / epsilon,
                    values,
                    linestyle="none",
                    marker=marker,
                    color=color,
                    ms=5,
                    label=f"{method} data",
                )
            series_rows.append(
                {
                    "panel": panel,
                    "method": method,
                    "series_role": "data",
                    "quantity": value_column,
                    "family": "computed data" if kind == "rate" else "particle simulation",
                    "color": color,
                    "linestyle": "none",
                    "marker": marker,
                }
            )

            quantity = value_column if kind == "rate" else "mixing_time"
            metric = value_column if kind == "rate" else selected_metric
            estimator = "deterministic" if kind == "rate" else "particle_threshold"
            selection = selections[
                (selections["method"] == method)
                & (selections["quantity"] == quantity)
                & (selections["metric"] == metric)
                & (selections["estimator"] == estimator)
                & (selections["epsilon_subset"] == "all")
            ].iloc[0]
            selected = fits[
                (fits["method"] == method)
                & (fits["quantity"] == quantity)
                & (fits["metric"] == metric)
                & (fits["estimator"] == estimator)
                & (fits["epsilon_subset"] == "all")
                & (fits["model_family"] == selection["selected_family"])
            ].iloc[0]
            ax.plot(
                1.0 / eps_curve,
                _curve(eps_curve, selected, kind),
                color=color,
                label=f"{method} empirical {selection['selected_family']}",
                **REFERENCE_STYLES["empirical_fit"],
            )
            series_rows.append(
                {
                    "panel": panel,
                    "method": method,
                    "series_role": "empirical_fit",
                    "quantity": value_column,
                    "family": selection["selected_family"],
                    "color": color,
                    "linestyle": REFERENCE_STYLES["empirical_fit"]["linestyle"],
                    "marker": "",
                }
            )

            if method == "Langevin":
                shape_data = np.exp((-1.0 if kind == "rate" else 1.0) * BARRIER / epsilon)
                shape_curve = np.exp(
                    (-1.0 if kind == "rate" else 1.0) * BARRIER / eps_curve
                )
                family = "fixed Arrhenius DeltaV=1/4"
            else:
                shape_data = epsilon ** (0.5 if kind == "rate" else -0.5)
                shape_curve = eps_curve ** (0.5 if kind == "rate" else -0.5)
                family = "fixed conservative graph-channel exponent 1/2"
            amplitude = _fixed_amplitude(epsilon, values, shape_data)
            ax.plot(
                1.0 / eps_curve,
                amplitude * shape_curve,
                color=color,
                label=f"{method} theory reference",
                **REFERENCE_STYLES["theory"],
            )
            series_rows.append(
                {
                    "panel": panel,
                    "method": method,
                    "series_role": "theory_reference",
                    "quantity": value_column,
                    "family": family,
                    "fixed_parameter": BARRIER if method == "Langevin" else 0.5,
                    "amplitude": amplitude,
                    "amplitude_rule": "least-squares log amplitude over valid displayed data",
                    "color": color,
                    "linestyle": REFERENCE_STYLES["theory"]["linestyle"],
                    "marker": "",
                }
            )
        ax.set_yscale("log")
        ax.set_xlabel(r"$1/\varepsilon$")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.22)
        panel_label(ax, "abc"[panel_index])
        ax.legend(fontsize=5.8, ncol=1)
    fig.tight_layout()
    output = figure_dir / "doublewell_theory_epsilon_chi2_sweep.pdf"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    series = pd.DataFrame(series_rows)
    series.to_csv(table_dir / "doublewell_phase17m_main_figure_series.csv", index=False)

    mixing_series = series[series["panel"] == "mixing_time"].copy()
    mixing_series[
        ["method", "series_role", "family", "color", "linestyle", "marker"]
    ].to_csv(table_dir / "doublewell_chi2_six_series_design.csv", index=False)

    theory = mixing_series[mixing_series["series_role"] == "theory_reference"].copy()
    theory["quantity"] = "chi2_mixing_time"
    theory["reference_family"] = theory["method"].map(
        {
            "Langevin": "fixed_arrhenius",
            "LSC-CP": "fixed_theorem_inverse_channel",
        }
    )
    theory["shape"] = theory["method"].map(
        {"Langevin": "exp(DeltaV / eps)", "LSC-CP": "eps^(-1/2)"}
    )
    theory["DeltaV"] = np.where(theory["method"] == "Langevin", BARRIER, np.nan)
    theory["n"] = mixing[mixing["metric"] == selected_metric].groupby("method")[
        "epsilon"
    ].nunique().reindex(theory["method"]).to_numpy()
    theory[
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

    candidates = fits[
        (fits["quantity"] == "mixing_time")
        & (fits["metric"] == selected_metric)
        & (fits["epsilon_subset"] == "all")
    ].copy()
    candidates["quantity"] = "chi2_mixing_time"
    candidates["fit_family"] = candidates["model_family"]
    candidates["prefactor"] = candidates["C_hat"]
    candidates["barrier"] = candidates["Delta_hat"]
    candidates["exponent"] = candidates["alpha_hat"]
    candidates["r2_log"] = candidates["log_R2"]
    candidates["selected_for_display"] = candidates["selected_fit"]
    candidates["selection_rule"] = (
        "two-of-three majority across AIC, BIC, and leave-one-epsilon-out likelihood"
    )
    candidates[
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
    return output, series


def _validation_figure(
    table_dir: Path,
    appendix_dir: Path,
    selected_metric: str,
) -> tuple[Path, Path, Path]:
    summary = pd.read_csv(table_dir / "doublewell_phase17m_chi2_time_series_summary.csv")
    windows = pd.read_csv(table_dir / "doublewell_phase17m_chi2_rate_windows.csv")
    consistency = pd.read_csv(table_dir / "doublewell_phase17m_rate_consistency.csv")
    refinement = pd.read_csv(table_dir / "doublewell_phase17m_grid_refinement.csv")
    apply_plot_style(plt)

    available = np.array(sorted(summary["epsilon"].unique()), dtype=float)
    representative = [
        float(available[np.argmin(np.abs(available - target))])
        for target in (0.10, 0.06, 0.05, 0.04)
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16.5, 7.4), sharey=False)
    for column, epsilon in enumerate(representative):
        for row, method in enumerate(METHODS):
            ax = axes[row, column]
            sub = summary[
                np.isclose(summary["epsilon"], epsilon)
                & (summary["method"] == method)
                & (summary["metric"] == selected_metric)
            ].sort_values("time")
            ax.semilogy(
                sub["time"],
                sub["mean"],
                color=method_color(method),
                label=f"{method} mean",
            )
            ax.fill_between(
                sub["time"],
                np.maximum(sub["q10"], 1e-12),
                np.maximum(sub["q90"], 1e-12),
                color=method_color(method),
                alpha=0.15,
            )
            c = consistency[
                np.isclose(consistency["epsilon"], epsilon)
                & (consistency["method"] == method)
            ].iloc[0]
            window = windows[
                np.isclose(windows["epsilon"], epsilon)
                & (windows["method"] == method)
                & (windows["metric"] == selected_metric)
                & (windows["estimator"] == "seed_mean")
            ]
            if len(window) and bool(window.iloc[0]["accepted"]):
                w = window.iloc[0]
                t = np.linspace(w["window_start"], w["window_end"], 100)
                y0 = w["window_initial_value"]
                fit = y0 * np.exp(w["slope_log_chi2"] * (t - w["window_start"]))
                ax.axvspan(
                    float(w["window_start"]),
                    float(w["window_end"]),
                    color=method_color(method),
                    alpha=0.08,
                    label="fit window",
                )
                ax.plot(t, fit, color="black", ls="--", lw=1.2, label="fit")
            noise_cutoff = float(c["noise_cutoff"])
            threshold = float(c["threshold"])
            bias_floor = float(c["bias_floor"])
            ax.axhspan(1e-12, noise_cutoff, color="#777777", alpha=0.10, label="floor region")
            ax.axhline(threshold, color="#555555", ls=":", lw=1.1, label="threshold")
            ax.axhline(bias_floor, color="#999999", ls="-.", lw=0.9, label="bias floor")
            ax.axhline(noise_cutoff, color="#777777", ls="--", lw=0.9, label="noise cutoff")
            ax.axvline(
                float(c["predicted_threshold_time_from_lambda_eff"]),
                color="#111111",
                ls="--",
                lw=0.9,
                label="predicted crossing",
            )
            ax.axvline(
                float(c["observed_mixing_time"]),
                color="#111111",
                ls=":",
                lw=0.9,
                label="observed crossing",
            )
            if float(c["observed_mixing_time"]) < float(sub["time"].max()):
                ax.axvspan(
                    float(c["observed_mixing_time"]),
                    float(sub["time"].max()),
                    color="#AAAAAA",
                    alpha=0.06,
                    label="post-threshold plateau",
                )
            ax.set_title(f"{method}, epsilon={epsilon:g}")
            ax.set_xlabel("time")
            ax.set_ylabel(selected_metric)
            ax.legend(fontsize=5.2)
    fig.tight_layout()
    decay_path = appendix_dir / "doublewell_phase17m_chi2_rate_consistency_windows.pdf"
    fig.savefig(decay_path, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=False)
    rate_specs = [
        ("lambda_form", "form gap", "-", "o"),
        ("lambda_abs", "abscissa", "--", "s"),
        ("lambda_eff", "chi-square window", ":", "^"),
        ("lambda_mix", "threshold implied", "-.", "D"),
    ]
    for ax, method in zip(axes, METHODS):
        sub = consistency[consistency["method"] == method].sort_values("epsilon")
        for column, label, linestyle, marker in rate_specs:
            ax.plot(
                sub["epsilon"],
                sub[column],
                marker=marker,
                linestyle=linestyle,
                lw=1.4,
                ms=4.5,
                label=label,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(method)
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel("rate")
        ax.grid(alpha=0.22)
        ax.legend(fontsize=7)
    fig.tight_layout()
    rate_path = appendix_dir / "doublewell_phase17m_rate_consistency_rates.pdf"
    fig.savefig(rate_path, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for method in METHODS:
        sub = refinement[refinement["method"] == method]
        for epsilon in sorted(sub["epsilon"].unique()):
            current = sub[np.isclose(sub["epsilon"], epsilon)].sort_values("n_cells")
            axes[0].plot(
                current["n_cells"],
                current["spectral_gap"],
                marker=method_marker(method),
                color=method_color(method),
                alpha=0.5 + 0.5 * (epsilon == min(sub["epsilon"].unique())),
                label=f"{method}, eps={epsilon:g}",
            )
            axes[1].plot(
                current["n_cells"],
                current["abscissa_rate"],
                marker=method_marker(method),
                color=method_color(method),
                alpha=0.5 + 0.5 * (epsilon == min(sub["epsilon"].unique())),
                label=f"{method}, eps={epsilon:g}",
            )
    axes[0].set_title("spectral-gap refinement")
    axes[1].set_title("abscissa refinement")
    for ax in axes:
        ax.set_xlabel("finite-volume cells")
        ax.set_ylabel("rate")
        ax.set_yscale("log")
        ax.legend(fontsize=5.5, ncol=2)
    fig.tight_layout()
    refinement_path = appendix_dir / "doublewell_phase17m_spectral_refinement.pdf"
    fig.savefig(refinement_path, bbox_inches="tight")
    plt.close(fig)
    return decay_path, rate_path, refinement_path


def _acceptance_summary(
    spectral: pd.DataFrame,
    mixing_summary: pd.DataFrame,
    windows: pd.DataFrame,
    fits: pd.DataFrame,
    selections: pd.DataFrame,
    selected_metric: str,
    validation_selection: pd.DataFrame,
    consistency: pd.DataFrame,
) -> pd.DataFrame:
    temperature_subsets = [
        "all",
        "exclude_largest_2",
        "lowest_12",
        "lowest_10",
        "lowest_8",
    ]

    def selection_count(method: str, quantity: str, metric: str, family: str) -> int:
        sub = selections[
            (selections["method"] == method)
            & (selections["quantity"] == quantity)
            & (selections["metric"] == metric)
            & (selections["epsilon_subset"].isin(temperature_subsets))
        ]
        return int(np.sum(sub["selected_family"] == family))

    langevin_fit = fits[
        (fits["method"] == "Langevin")
        & (fits["quantity"] == "mixing_time")
        & (fits["metric"] == selected_metric)
        & (fits["epsilon_subset"] == "lowest_8")
        & (fits["model_family"] == "arrhenius")
    ].iloc[0]
    lsc_fit = fits[
        (fits["method"] == "LSC-CP")
        & (fits["quantity"] == "mixing_time")
        & (fits["metric"] == selected_metric)
        & (fits["epsilon_subset"] == "all")
        & (fits["model_family"] == "polynomial")
    ].iloc[0]
    accepted_low = windows[
        (windows["method"] == "LSC-CP")
        & (windows["metric"] == selected_metric)
        & (windows["estimator"] == "seed_mean")
        & windows["accepted"].fillna(False).astype(bool)
        & (windows["epsilon"].isin(sorted(windows["epsilon"].unique())[:8]))
    ]["epsilon"].nunique()
    lsc_censored = int(
        mixing_summary[
            (mixing_summary["method"] == "LSC-CP")
            & (mixing_summary["metric"] == selected_metric)
        ]["n_censored"].sum()
    )
    lsc_bin_robustness = all(
        selection_count("LSC-CP", "mixing_time", metric, "polynomial") >= 3
        for metric in ("bin_chi2_M40", "bin_chi2_M80", "bin_chi2_M120")
    )
    lsc_validation = validation_selection[validation_selection["method"] == "LSC-CP"]
    weak_order_one_reversal = (
        (lsc_validation["selected_family"] == "arrhenius")
        & (lsc_validation["delta_AIC_poly_minus_arr"].abs() <= 1.0)
        & (lsc_validation["delta_BIC_poly_minus_arr"].abs() <= 1.0)
    )
    validation_robustness = bool(
        ((lsc_validation["selected_family"] == "polynomial") | weak_order_one_reversal).all()
    )
    consistency_by_method = (
        consistency.groupby("method")["consistency_pass"].all().reindex(METHODS).fillna(False)
    )
    langevin_barriers = fits[
        (fits["method"] == "Langevin")
        & (fits["epsilon_subset"].isin(["all", "lowest_12", "lowest_10", "lowest_8"]))
        & (fits["model_family"] == "arrhenius")
        & (fits["quantity"].isin(
            ["spectral_gap", "abscissa_rate", "chi2_effective_rate", "mixing_time"]
        ))
        & (
            (fits["metric"].isin(["spectral_gap", "abscissa_rate"]))
            | (fits["metric"] == selected_metric)
        )
    ]["Delta_hat"].dropna()
    langevin_supported = (
        selection_count("Langevin", "spectral_gap", "spectral_gap", "arrhenius") >= 4
        and selection_count("Langevin", "abscissa_rate", "abscissa_rate", "arrhenius") >= 4
        and selection_count("Langevin", "chi2_effective_rate", selected_metric, "arrhenius") >= 4
        and selection_count("Langevin", "mixing_time", selected_metric, "arrhenius") >= 4
        and len(langevin_barriers) > 0
        and bool(((langevin_barriers >= 0.20) & (langevin_barriers <= 0.30)).all())
        and bool(consistency_by_method.loc["Langevin"])
    )
    lsc_supported = (
        selection_count("LSC-CP", "spectral_gap", "spectral_gap", "polynomial") >= 4
        and selection_count("LSC-CP", "abscissa_rate", "abscissa_rate", "polynomial") >= 4
        and selection_count("LSC-CP", "mixing_time", selected_metric, "polynomial") >= 4
        and selection_count("LSC-CP", "chi2_effective_rate", selected_metric, "polynomial") >= 3
        and accepted_low >= 6
        and lsc_censored <= 2
        and lsc_bin_robustness
        and validation_robustness
        and bool(consistency_by_method.loc["LSC-CP"])
    )
    return pd.DataFrame(
        [
            {
                "claim": "Langevin Arrhenius gap, abscissa, rate, and mixing time",
                "supported": langevin_supported,
                "selected_metric": selected_metric,
                "estimate": float(langevin_fit["Delta_hat"]),
                "bootstrap_ci_low": float(langevin_fit["bootstrap_ci_low"]),
                "bootstrap_ci_high": float(langevin_fit["bootstrap_ci_high"]),
                "rate_consistency_passed": bool(consistency_by_method.loc["Langevin"]),
                "selection_summary": "stable family required on four of five subsets",
            },
            {
                "claim": "LSC-CP non-Arrhenius graph-channel rate and mixing time",
                "supported": lsc_supported,
                "selected_metric": selected_metric,
                "estimate": float(lsc_fit["alpha_hat"]),
                "bootstrap_ci_low": float(lsc_fit["bootstrap_ci_low"]),
                "bootstrap_ci_high": float(lsc_fit["bootstrap_ci_high"]),
                "rate_consistency_passed": bool(consistency_by_method.loc["LSC-CP"]),
                "accepted_low8_rate_windows": int(accepted_low),
                "n_censored": lsc_censored,
                "bin_count_robustness": lsc_bin_robustness,
                "focused_particle_robustness": validation_robustness,
                "selection_summary": "polynomial family on the production grid with order-one low-temperature flattening allowed",
            },
            {
                "claim": "strong cross-diagnostic scaling separation",
                "supported": bool(langevin_supported and lsc_supported),
                "selected_metric": selected_metric,
                "rate_consistency_passed": bool(consistency["consistency_pass"].all()),
                "selection_summary": "both method-specific acceptance rules and rate-consistency checks",
            },
        ]
    )


def run_phase17m_definitive_study(
    table_dir: Path | str,
    main_figure_dir: Path | str,
    appendix_figure_dir: Path | str,
    *,
    profile: str = "production",
) -> dict[str, object]:
    table_dir = Path(table_dir)
    main_figure_dir = Path(main_figure_dir)
    appendix_figure_dir = Path(appendix_figure_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    main_figure_dir.mkdir(parents=True, exist_ok=True)
    appendix_figure_dir.mkdir(parents=True, exist_ok=True)
    config = config_for_profile(profile)
    started = time.perf_counter()

    spectral, refinement, generators = _spectral_sweep(config)
    spectral[
        [
            "epsilon",
            "method",
            "n_cells",
            "spectral_gap",
            "spectral_gap_reliable",
            "form_zero_eigenvalue",
            "form_residual",
            "row_sum_inf",
            "stationary_residual_l1",
            "stationary_residual_linf",
            "minimum_off_diagonal",
        ]
    ].to_csv(table_dir / "doublewell_phase17m_spectral_gap_rates.csv", index=False)
    spectral[
        [
            "epsilon",
            "method",
            "n_cells",
            "abscissa_rate",
            "abscissa_reliable",
            "zero_eigenvalue_abs",
            "maximum_eigenvalue_real_part",
            "row_sum_inf",
            "stationary_residual_l1",
            "stationary_residual_linf",
            "minimum_off_diagonal",
            "maximum_exit_rate",
            "raw_jump_stationary_residual_linf",
            "correction_flux_max",
        ]
    ].to_csv(table_dir / "doublewell_phase17m_abscissa_rates.csv", index=False)
    refinement.to_csv(table_dir / "doublewell_phase17m_grid_refinement.csv", index=False)

    raw_parts = []
    mixing_parts = []
    floor_parts = []
    threshold_rows = []
    propagation_parts = []
    for epsilon_index, epsilon in enumerate(config.epsilon_values):
        epsilon_start = time.perf_counter()
        context = generators[(epsilon, "Langevin")][1]
        floors, thresholds = _thresholds_and_floors(
            epsilon, context, config, epsilon_index
        )
        floor_parts.append(floors)
        for metric, values in thresholds.items():
            threshold_rows.append(
                {
                    "epsilon": epsilon,
                    "metric": metric,
                    "bias_floor_replicates": config.floor_replicates,
                    "N_particles": config.n_particles,
                    **values,
                }
            )
        raw, mixing, propagation = _simulate_epsilon(
            epsilon,
            epsilon_index,
            generators,
            spectral,
            thresholds,
            config,
        )
        raw_parts.append(raw)
        mixing_parts.append(mixing)
        propagation_parts.append(propagation)
        primary = mixing[mixing["metric"] == "bin_chi2_M80"]
        print(
            f"Phase17M epsilon={epsilon:g}: {time.perf_counter() - epsilon_start:.1f}s, "
            f"M80 censored={int(primary['censored'].sum())}/{len(primary)}"
        )

    raw = pd.concat(raw_parts, ignore_index=True)
    mixing = pd.concat(mixing_parts, ignore_index=True)
    floors = pd.concat(floor_parts, ignore_index=True)
    thresholds = pd.DataFrame(threshold_rows)
    propagation = pd.concat(propagation_parts, ignore_index=True)
    time_summary = _summarize_time_series(raw)
    mixing_summary = _summarize_mixing(mixing)
    windows = _rate_windows(raw, mixing)
    selected_metric, estimator_audit = _choose_main_metric(windows, mixing_summary)
    fits, selections = _fit_tables(spectral, mixing, windows, config)
    validation_summary, validation_selection = _focused_particle_validation(
        generators, spectral, config
    )
    consistency, consistency_summary, window_representatives = _rate_consistency_table(
        spectral,
        mixing_summary,
        windows,
        selected_metric,
    )
    acceptance = _acceptance_summary(
        spectral,
        mixing_summary,
        windows,
        fits,
        selections,
        selected_metric,
        validation_selection,
        consistency,
    )

    time_summary.to_csv(
        table_dir / "doublewell_phase17m_chi2_time_series_summary.csv", index=False
    )
    raw[raw["metric"].isin(CHI_METRICS)].to_csv(
        table_dir / "doublewell_phase17m_chi2_time_series_by_seed.csv", index=False
    )
    windows.to_csv(
        table_dir / "doublewell_phase17m_chi2_rate_windows.csv", index=False
    )
    consistency.to_csv(
        table_dir / "doublewell_phase17m_rate_consistency.csv", index=False
    )
    consistency_summary.to_csv(
        table_dir / "doublewell_phase17m_rate_consistency_summary.csv", index=False
    )
    window_representatives.to_csv(
        table_dir / "doublewell_phase17m_chi2_window_representatives.csv",
        index=False,
    )
    mixing.to_csv(
        table_dir / "doublewell_phase17m_mixing_times_by_seed.csv", index=False
    )
    mixing_summary.to_csv(
        table_dir / "doublewell_phase17m_mixing_times_summary.csv", index=False
    )
    thresholds.to_csv(
        table_dir / "doublewell_phase17m_thresholds_bias_floor.csv", index=False
    )
    floors.to_csv(
        table_dir / "doublewell_phase17m_bias_floor_replicates.csv", index=False
    )
    fits[fits["epsilon_subset"] == "all"].to_csv(
        table_dir / "doublewell_phase17m_fit_candidates.csv", index=False
    )
    fits.to_csv(table_dir / "doublewell_phase17m_fit_stability.csv", index=False)
    selections.to_csv(
        table_dir / "doublewell_phase17m_model_selection_summary.csv", index=False
    )
    acceptance.to_csv(
        table_dir / "doublewell_phase17m_acceptance_summary.csv", index=False
    )
    estimator_audit.to_csv(
        table_dir / "doublewell_phase17m_estimator_selection.csv", index=False
    )
    propagation.to_csv(
        table_dir / "doublewell_phase17m_propagation_validation.csv", index=False
    )
    validation_summary.to_csv(
        table_dir / "doublewell_phase17m_particle_validation.csv", index=False
    )
    validation_selection.to_csv(
        table_dir / "doublewell_phase17m_particle_validation_model_selection.csv",
        index=False,
    )
    pd.DataFrame([asdict(config)]).to_csv(
        table_dir / "doublewell_phase17m_simulation_config.csv", index=False
    )

    main_figure, series = generate_main_figure(
        table_dir, main_figure_dir, selected_metric
    )
    appendix_figures = _validation_figure(
        table_dir, appendix_figure_dir, selected_metric
    )
    elapsed = time.perf_counter() - started
    return {
        "config": config,
        "spectral": spectral,
        "refinement": refinement,
        "time_summary": time_summary,
        "mixing": mixing,
        "mixing_summary": mixing_summary,
        "windows": windows,
        "rate_consistency": consistency,
        "rate_consistency_summary": consistency_summary,
        "window_representatives": window_representatives,
        "fits": fits,
        "selections": selections,
        "acceptance": acceptance,
        "particle_validation": validation_summary,
        "particle_validation_selection": validation_selection,
        "selected_metric": selected_metric,
        "main_figure": main_figure,
        "main_figure_series": series,
        "appendix_figures": appendix_figures,
        "runtime_seconds": elapsed,
    }
