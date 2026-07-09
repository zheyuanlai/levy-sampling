"""Censored scaling models for seed-level simulation mixing times."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass(frozen=True)
class CensoredFit:
    intercept: float
    slope: float
    log_sigma: float
    log_likelihood: float
    converged: bool

    @property
    def sigma(self) -> float:
        return float(np.exp(self.log_sigma))

    @property
    def amplitude(self) -> float:
        return float(np.exp(self.intercept))


def model_covariate(epsilon: np.ndarray, family: str) -> np.ndarray:
    eps = np.asarray(epsilon, dtype=float)
    if family == "arrhenius":
        return 1.0 / eps
    if family == "polynomial":
        return -np.log(eps)
    raise ValueError(f"unknown model family: {family}")


def model_curve(epsilon: np.ndarray, amplitude: float, parameter: float, family: str) -> np.ndarray:
    eps = np.asarray(epsilon, dtype=float)
    if family == "arrhenius":
        return float(amplitude) * np.exp(float(parameter) / eps)
    if family == "polynomial":
        return float(amplitude) * eps ** (-float(parameter))
    raise ValueError(f"unknown model family: {family}")


def _negative_log_likelihood(
    params: np.ndarray,
    x: np.ndarray,
    log_time: np.ndarray,
    event: np.ndarray,
) -> float:
    intercept, slope, log_sigma = np.asarray(params, dtype=float)
    sigma = float(np.exp(log_sigma))
    if not np.isfinite(sigma) or sigma <= 0.0:
        return np.inf
    mu = intercept + slope * x
    z = (log_time - mu) / sigma
    ll = np.empty_like(z)
    ll[event] = norm.logpdf(z[event]) - log_sigma
    ll[~event] = norm.logsf(z[~event])
    if not np.all(np.isfinite(ll)):
        return np.inf
    return float(-np.sum(ll))


def _prepare_observations(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"epsilon", "mixing_time", "censored", "T_final"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    eps = frame["epsilon"].to_numpy(dtype=float)
    censored = frame["censored"].astype(bool).to_numpy()
    exact = frame["mixing_time"].to_numpy(dtype=float)
    censor_time = frame["T_final"].to_numpy(dtype=float)
    observed = np.where(censored, censor_time, exact)
    valid = np.isfinite(eps) & (eps > 0.0) & np.isfinite(observed) & (observed > 0.0)
    return eps[valid], np.log(observed[valid]), (~censored[valid])


def _initial_parameters(x: np.ndarray, log_time: np.ndarray, event: np.ndarray) -> np.ndarray:
    fit_mask = event if np.count_nonzero(event) >= 2 else np.ones_like(event, dtype=bool)
    design = np.column_stack([np.ones(np.count_nonzero(fit_mask)), x[fit_mask]])
    beta, *_ = np.linalg.lstsq(design, log_time[fit_mask], rcond=None)
    residual = log_time[fit_mask] - design @ beta
    sigma = float(np.std(residual, ddof=1)) if residual.size > 2 else 0.25
    sigma = max(sigma, 0.08)
    return np.array([float(beta[0]), float(beta[1]), float(np.log(sigma))])


def fit_censored_model(frame: pd.DataFrame, family: str) -> CensoredFit:
    eps, log_time, event = _prepare_observations(frame)
    if eps.size < 4 or np.unique(eps).size < 3:
        raise ValueError("at least four observations and three epsilon values are required")
    x = model_covariate(eps, family)
    start = _initial_parameters(x, log_time, event)
    result = minimize(
        _negative_log_likelihood,
        start,
        args=(x, log_time, event),
        method="L-BFGS-B",
        bounds=[(None, None), (None, None), (-6.0, 4.0)],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    params = np.asarray(result.x, dtype=float)
    return CensoredFit(
        intercept=float(params[0]),
        slope=float(params[1]),
        log_sigma=float(params[2]),
        log_likelihood=float(-_negative_log_likelihood(params, x, log_time, event)),
        converged=bool(result.success and np.all(np.isfinite(params))),
    )


def _numerical_hessian(fun, point: np.ndarray) -> np.ndarray:
    p = np.asarray(point, dtype=float)
    n = p.size
    steps = 2e-4 * (1.0 + np.abs(p))
    hessian = np.zeros((n, n), dtype=float)
    f0 = float(fun(p))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = steps[i]
        hessian[i, i] = (fun(p + ei) - 2.0 * f0 + fun(p - ei)) / (steps[i] ** 2)
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = steps[j]
            value = (
                fun(p + ei + ej)
                - fun(p + ei - ej)
                - fun(p - ei + ej)
                + fun(p - ei - ej)
            ) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian


def parameter_standard_error(frame: pd.DataFrame, family: str, fit: CensoredFit) -> float:
    eps, log_time, event = _prepare_observations(frame)
    x = model_covariate(eps, family)
    point = np.array([fit.intercept, fit.slope, fit.log_sigma], dtype=float)
    try:
        hessian = _numerical_hessian(
            lambda p: _negative_log_likelihood(p, x, log_time, event),
            point,
        )
        covariance = np.linalg.pinv(hessian)
        variance = float(covariance[1, 1])
        return float(np.sqrt(variance)) if variance >= 0.0 and np.isfinite(variance) else np.nan
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return np.nan


def leave_one_epsilon_out_error(frame: pd.DataFrame, family: str) -> float:
    scores: list[float] = []
    counts: list[int] = []
    for epsilon in sorted(frame["epsilon"].unique()):
        train = frame[~np.isclose(frame["epsilon"], epsilon)].copy()
        test = frame[np.isclose(frame["epsilon"], epsilon)].copy()
        try:
            fit = fit_censored_model(train, family)
        except (ValueError, RuntimeError):
            continue
        eps, log_time, event = _prepare_observations(test)
        x = model_covariate(eps, family)
        params = np.array([fit.intercept, fit.slope, fit.log_sigma], dtype=float)
        score = _negative_log_likelihood(params, x, log_time, event)
        if np.isfinite(score):
            scores.append(float(score))
            counts.append(int(len(log_time)))
    total = int(np.sum(counts))
    return float(np.sum(scores) / total) if total > 0 else np.nan


def stratified_bootstrap_parameters(
    frame: pd.DataFrame,
    family: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    amplitudes: list[float] = []
    parameters: list[float] = []
    groups = [group.reset_index(drop=True) for _, group in frame.groupby("epsilon", sort=True)]
    for _ in range(int(n_bootstrap)):
        sampled = []
        for group in groups:
            ids = rng.integers(0, len(group), size=len(group))
            sampled.append(group.iloc[ids])
        boot = pd.concat(sampled, ignore_index=True)
        try:
            fit = fit_censored_model(boot, family)
        except (ValueError, RuntimeError):
            continue
        if fit.converged and np.isfinite(fit.amplitude) and np.isfinite(fit.slope):
            amplitudes.append(fit.amplitude)
            parameters.append(fit.slope)
    return np.asarray(amplitudes, dtype=float), np.asarray(parameters, dtype=float)


def fit_diagnostics(
    frame: pd.DataFrame,
    family: str,
    *,
    n_bootstrap: int = 300,
    bootstrap_seed: int = 20261712,
) -> dict[str, object]:
    fit = fit_censored_model(frame, family)
    eps, log_time, event = _prepare_observations(frame)
    x = model_covariate(eps, family)
    prediction = fit.intercept + fit.slope * x
    residual = log_time[event] - prediction[event]
    ss_res = float(np.sum(residual**2))
    centered = log_time[event] - float(np.mean(log_time[event])) if np.any(event) else np.array([])
    ss_tot = float(np.sum(centered**2))
    log_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan
    rmse = float(np.sqrt(np.mean(residual**2))) if residual.size else np.nan
    n = int(len(log_time))
    k = 3
    aic = float(2 * k - 2.0 * fit.log_likelihood)
    bic = float(k * np.log(max(n, 1)) - 2.0 * fit.log_likelihood)
    se = parameter_standard_error(frame, family, fit)
    amplitudes, parameters = stratified_bootstrap_parameters(
        frame,
        family,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )
    amp_ci = np.quantile(amplitudes, [0.025, 0.975]) if amplitudes.size else [np.nan, np.nan]
    par_ci = np.quantile(parameters, [0.025, 0.975]) if parameters.size else [np.nan, np.nan]
    parameter_name = "Delta_hat" if family == "arrhenius" else "alpha_hat"
    return {
        "model_family": family,
        "n_points": n,
        "n_epsilon": int(np.unique(eps).size),
        "n_censored": int(np.count_nonzero(~event)),
        "C_hat": fit.amplitude,
        "parameter_name": parameter_name,
        "parameter_hat": fit.slope,
        "Delta_hat": fit.slope if family == "arrhenius" else np.nan,
        "alpha_hat": fit.slope if family == "polynomial" else np.nan,
        "standard_error": se,
        "bootstrap_CI": f"[{par_ci[0]:.8g}, {par_ci[1]:.8g}]",
        "bootstrap_ci_low": float(par_ci[0]),
        "bootstrap_ci_high": float(par_ci[1]),
        "bootstrap_C_CI": f"[{amp_ci[0]:.8g}, {amp_ci[1]:.8g}]",
        "bootstrap_C_ci_low": float(amp_ci[0]),
        "bootstrap_C_ci_high": float(amp_ci[1]),
        "bootstrap_successes": int(parameters.size),
        "log_likelihood": fit.log_likelihood,
        "log_R2": log_r2,
        "AIC": aic,
        "BIC": bic,
        "leave_one_out_error": leave_one_epsilon_out_error(frame, family),
        "residual_RMSE_log": rmse,
        "sigma_log_time": fit.sigma,
        "converged": fit.converged,
    }


def select_temperature_subset(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    values = np.array(sorted(frame["epsilon"].unique()), dtype=float)
    if subset == "all":
        return frame.copy()
    if subset == "exclude_largest_2":
        keep = values[:-2] if values.size > 2 else values
        return frame[frame["epsilon"].isin(keep)].copy()
    if subset == "lowest_10":
        keep = values[: min(10, values.size)]
        return frame[frame["epsilon"].isin(keep)].copy()
    if subset == "lowest_8":
        keep = values[: min(8, values.size)]
        return frame[frame["epsilon"].isin(keep)].copy()
    if subset == "uncensored_only":
        return frame[~frame["censored"].astype(bool)].copy()
    raise ValueError(f"unknown epsilon subset: {subset}")


def compare_model_families(fits: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = {str(row["model_family"]): row for row in fits}
    arr = rows["arrhenius"]
    poly = rows["polynomial"]
    winners = {
        "AIC": "arrhenius" if float(arr["AIC"]) < float(poly["AIC"]) else "polynomial",
        "BIC": "arrhenius" if float(arr["BIC"]) < float(poly["BIC"]) else "polynomial",
        "LOO": (
            "arrhenius"
            if float(arr["leave_one_out_error"]) < float(poly["leave_one_out_error"])
            else "polynomial"
        ),
    }
    arr_votes = int(sum(value == "arrhenius" for value in winners.values()))
    selected = "arrhenius" if arr_votes >= 2 else "polynomial"
    return {
        "preferred_AIC": winners["AIC"],
        "preferred_BIC": winners["BIC"],
        "preferred_LOO": winners["LOO"],
        "arrhenius_votes": arr_votes,
        "polynomial_votes": 3 - arr_votes,
        "selected_family": selected,
        "delta_AIC_poly_minus_arr": float(poly["AIC"]) - float(arr["AIC"]),
        "delta_BIC_poly_minus_arr": float(poly["BIC"]) - float(arr["BIC"]),
        "delta_LOO_poly_minus_arr": (
            float(poly["leave_one_out_error"]) - float(arr["leave_one_out_error"])
        ),
    }
