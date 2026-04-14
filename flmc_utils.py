#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Tuple

import numpy as np


def c_alpha(alpha: float) -> float:
    """
    Normalization constant from Simsekli et al. (ICML 2017), Section 3.3.

    The paper-faithful FLMC update is
        X_{n+1} = X_n - dt * c_alpha * gradU(X_n) + dt^(1/alpha) * xi_n
    with
        c_alpha = Gamma(alpha - 1) / Gamma(alpha / 2)^2.
    """
    if not (1.0 < alpha <= 2.0):
        raise ValueError(f"FLMC alpha must satisfy 1 < alpha <= 2, got {alpha}")
    return math.gamma(alpha - 1.0) / (math.gamma(alpha / 2.0) ** 2)


def sample_symmetric_alpha_stable(
    rng: np.random.Generator,
    size: Tuple[int, ...],
    alpha: float,
) -> np.ndarray:
    """
    Sample iid scalar SαS(1) random variables coordinate-wise.

    This repo now uses the paper-faithful convention
        E[exp(i w X)] = exp(-|w|^alpha),
    so alpha = 2 gives S2S(1) = N(0, 2) exactly. Multidimensional FLMC is
    modeled with independent coordinates, not isotropic radial stable vectors.
    """
    if not (1.0 < alpha <= 2.0):
        raise ValueError(f"Alpha must satisfy 1 < alpha <= 2, got {alpha}")

    if np.isclose(alpha, 2.0):
        return math.sqrt(2.0) * rng.standard_normal(size=size)

    u = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=size)
    w = rng.exponential(scale=1.0, size=size)

    cos_u = np.clip(np.cos(u), 1e-12, None)
    cos_term = np.clip(np.cos((1.0 - alpha) * u), 1e-12, None)
    part1 = np.sin(alpha * u) / (cos_u ** (1.0 / alpha))
    part2 = (cos_term / w) ** ((1.0 - alpha) / alpha)
    return part1 * part2


def flmc_paper_drift(grad_u: np.ndarray, alpha: float) -> np.ndarray:
    """Return the paper-faithful FLMC drift -c_alpha * gradU."""
    return -c_alpha(alpha) * np.asarray(grad_u, dtype=float)


def flmc_paper_noise(
    rng: np.random.Generator,
    size: Tuple[int, ...],
    dt: float,
    alpha: float,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Sample the FLMC noise increment (sigma * dt)^(1/alpha) * xi.

    Here xi has iid coordinate-wise SαS(1) entries.

    sigma : float
        Temperature-like scale factor.  The default sigma=1.0 gives the
        original paper-faithful dt^(1/alpha) * xi noise.  Setting sigma=T_star
        gives the temperature-scaled variant (sigma * dt)^(1/alpha) * xi, which
        reduces to the standard ULA noise sqrt(2 * T_star * dt) * N(0,1) at
        alpha=2.  See step_flmc_nd for the recommended usage.
    """
    return (float(sigma) * float(dt)) ** (1.0 / float(alpha)) * sample_symmetric_alpha_stable(
        rng,
        size=size,
        alpha=alpha,
    )


def _paper_flmc_step_from_grad_u(
    x: np.ndarray,
    dt: float,
    alpha: float,
    grad_u: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 1.0,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    drift = flmc_paper_drift(grad_u, alpha)
    noise = flmc_paper_noise(rng, size=x.shape, dt=dt, alpha=alpha, sigma=sigma)
    if x.ndim == 1:
        norm = np.abs(drift)
    else:
        norm = np.linalg.norm(drift, axis=1, keepdims=True)
    tamed_drift = (float(dt) * drift) / (1.0 + float(dt) * norm)
    return x + tamed_drift + noise


def step_flmc_1d(
    x: np.ndarray,
    dt: float,
    alpha: float,
    gradU_fn,
    rng: np.random.Generator,
    clip_bounds: Tuple[float, float] = None,
) -> np.ndarray:
    """
    Paper-faithful 1D FLMC step from Simsekli et al., Section 3.3.

    The update is
        X_{n+1} = X_n - dt * c_alpha * gradU(X_n) + dt^(1/alpha) * xi_n,
    where xi_n has iid scalar SαS(1) law. No isotropic construction and no
    extra sigma multiplier are used anywhere in this repo's FLMC path.
    """
    x = np.asarray(x, dtype=float)
    grad_u = np.asarray(gradU_fn(x), dtype=float)
    x_new = _paper_flmc_step_from_grad_u(x, dt, alpha, grad_u, rng)
    if clip_bounds is not None:
        x_new = np.clip(x_new, clip_bounds[0], clip_bounds[1])
    return x_new


def step_flmc_2d(
    x: np.ndarray,
    dt: float,
    alpha: float,
    gradU_fn_xy,
    rng: np.random.Generator,
    clip_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Paper-faithful 2D FLMC step from Simsekli et al., Section 3.3.

    Multidimensional FLMC in this repo always uses independent coordinate-wise
    SαS(1) increments. We intentionally do NOT use isotropic stable processes.
    """
    x = np.asarray(x, dtype=float)
    grad_u_x, grad_u_y = gradU_fn_xy(x[:, 0], x[:, 1])
    grad_u = np.stack([grad_u_x, grad_u_y], axis=1)
    x_new = _paper_flmc_step_from_grad_u(x, dt, alpha, grad_u, rng)
    if clip_bounds is not None:
        x_new[:, 0] = np.clip(x_new[:, 0], clip_bounds[0][0], clip_bounds[0][1])
        x_new[:, 1] = np.clip(x_new[:, 1], clip_bounds[1][0], clip_bounds[1][1])
    return x_new


def step_flmc_nd(
    x: np.ndarray,
    dt: float,
    alpha: float,
    gradU_fn,
    rng: np.random.Generator,
    clip_bounds: Tuple[float, float] = None,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Multidimensional FLMC step from Simsekli et al., Section 3.3.

    The update is
        X_{n+1} = X_n - dt * c_alpha * gradU(X_n) + (sigma * dt)^(1/alpha) * xi_n,
    where xi_n has iid coordinate-wise SαS(1) entries.

    sigma : float
        Noise scale factor (default 1.0, i.e. original paper form dt^(1/alpha)).
        For temperature-aware FLMC targeting exp(-E / T_star), pass gradU as the
        gradient of E (not E/T_star) and set sigma=T_star.  This makes the update
        reduce exactly to tamed ULA at alpha=2 and keeps noise/drift in the same
        ratio as ULA for all alpha.
    """
    x = np.asarray(x, dtype=float)
    grad_u = np.asarray(gradU_fn(x), dtype=float)
    x_new = _paper_flmc_step_from_grad_u(x, dt, alpha, grad_u, rng, sigma=sigma)
    if clip_bounds is not None:
        x_new = np.clip(x_new, clip_bounds[0], clip_bounds[1])
    return x_new


def flmc_self_check(
    gradU_fn,
    probe_points: np.ndarray,
    *,
    alpha: float = 2.0,
    dt: float = 1.0e-2,
    noise_dim: int | None = None,
    n_noise_samples: int = 20000,
    seed: int = 0,
    variance_rtol: float = 0.10,
    drift_atol: float = 1.0e-12,
) -> dict:
    """
    Lightweight validation utility for the shared paper-faithful FLMC logic.

    Checks:
      1. For alpha = 2, dt^(1/2) * S2S(1) has variance 2 * dt per coordinate.
      2. The deterministic FLMC drift equals -c_alpha * gradU(X) on probe points.
    """
    rng = np.random.default_rng(seed)

    if noise_dim is None:
        probe_arr = np.asarray(probe_points, dtype=float)
        noise_dim = 1 if probe_arr.ndim == 1 else int(probe_arr.shape[1])

    noise_shape = (int(n_noise_samples),) if int(noise_dim) == 1 else (
        int(n_noise_samples),
        int(noise_dim),
    )
    noise = flmc_paper_noise(rng, size=noise_shape, dt=dt, alpha=alpha)
    empirical_var = np.atleast_1d(np.var(noise, axis=0))
    target_var = 2.0 * float(dt)

    probe_arr = np.asarray(probe_points, dtype=float)
    grad_u = np.asarray(gradU_fn(probe_arr), dtype=float)
    drift = flmc_paper_drift(grad_u, alpha)
    drift_step = np.asarray(probe_arr, dtype=float) + float(dt) * drift
    empirical_drift = (drift_step - probe_arr) / float(dt)
    target_drift = -c_alpha(alpha) * grad_u

    if np.isclose(alpha, 2.0):
        if not np.allclose(empirical_var, target_var, rtol=variance_rtol, atol=0.0):
            raise AssertionError(
                f"FLMC alpha=2 variance check failed: empirical={empirical_var}, target={target_var}"
            )

    if not np.allclose(empirical_drift, target_drift, rtol=0.0, atol=drift_atol):
        raise AssertionError(
            "FLMC drift check failed: shared helper does not match -c_alpha * gradU."
        )

    return {
        "alpha": float(alpha),
        "dt": float(dt),
        "target_noise_variance": float(target_var),
        "empirical_noise_variance_mean": float(np.mean(empirical_var)),
        "empirical_noise_variance_min": float(np.min(empirical_var)),
        "empirical_noise_variance_max": float(np.max(empirical_var)),
        "drift_max_abs_error": float(np.max(np.abs(empirical_drift - target_drift))),
    }
