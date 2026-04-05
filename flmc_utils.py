#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLMC (Fractional Langevin Monte Carlo) Utilities

This module provides core utilities for FLMC sampling, which uses alpha-stable
Levy noise instead of compound Poisson jumps. FLMC is distinct from LSB-MC:
- FLMC: uses alpha-stable noise, NO score correction
- LSB-MC: uses compound Poisson jumps WITH score correction

Reference: Deleted FLA.py (commit 843946a), adapted for current V-notation convention.
"""

import math
from typing import Tuple
import numpy as np


def c_alpha(alpha: float) -> float:
    """
    Normalization constant for FLMC with alpha-stable noise.

    For alpha-stable Levy process with tail index alpha in (1, 2],
    this constant appears in the drift term of the FLMC discretization.

    Formula: c_alpha = Gamma(alpha - 1) / Gamma(alpha/2)^2

    Args:
        alpha: Tail index, must satisfy 1 < alpha <= 2

    Returns:
        Normalization constant

    Raises:
        ValueError: If alpha not in (1, 2]
    """
    if not (1.0 < alpha <= 2.0):
        raise ValueError(f"FLMC alpha must satisfy 1 < alpha <= 2, got {alpha}")
    return math.gamma(alpha - 1.0) / (math.gamma(alpha / 2.0) ** 2)


def sample_symmetric_alpha_stable(
    rng: np.random.Generator,
    size: Tuple[int, ...],
    alpha: float
) -> np.ndarray:
    """
    Sample from symmetric alpha-stable distribution S(alpha, beta=0, scale=1, loc=0).

    Uses Chambers-Mallows-Stuck algorithm for generating alpha-stable random variables.
    The characteristic function is exp(-|t|^alpha).

    Args:
        rng: NumPy random generator
        size: Shape of output array
        alpha: Tail index, must satisfy 1 < alpha <= 2

    Returns:
        Array of alpha-stable samples with shape `size`

    Reference:
        Chambers, J. M., Mallows, C. L., & Stuck, B. W. (1976).
        A method for simulating stable random variables.
        Journal of the American Statistical Association, 71(354), 340-344.
    """
    if not (1.0 < alpha <= 2.0):
        raise ValueError(f"Alpha must satisfy 1 < alpha <= 2, got {alpha}")

    # Uniform in (-pi/2, pi/2)
    U = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=size)

    # Special case: alpha = 1 gives Cauchy distribution
    if np.isclose(alpha, 1.0):
        return np.tan(U)

    # Exponential(1)
    W = rng.exponential(scale=1.0, size=size)

    # Chambers-Mallows-Stuck formula
    cos_u = np.clip(np.cos(U), 1e-12, None)
    cos_term = np.clip(np.cos((1.0 - alpha) * U), 1e-12, None)

    part1 = np.sin(alpha * U) / (cos_u ** (1.0 / alpha))
    part2 = (cos_term / W) ** ((1.0 - alpha) / alpha)

    return part1 * part2


def step_flmc_1d(
    x: np.ndarray,
    dt: float,
    alpha: float,
    sigma: float,
    V_fn,
    gradV_fn,
    rng: np.random.Generator,
    clip_bounds: Tuple[float, float] = None
) -> np.ndarray:
    """
    One step of FLMC (Fractional Langevin MC) for 1D potential V.

    FLMC discretization (following manuscript global convention):
        X_{n+1} = X_n - c_alpha * dt * gradV(X_n) + dt^(1/alpha) * Z

    where:
        - Target density: p ∝ exp(-2V/sigma^2)
        - Z ~ symmetric alpha-stable(alpha, 0, 1, 0)
        - c_alpha = Gamma(alpha-1) / Gamma(alpha/2)^2

    NOTE: FLMC does NOT use score correction (unlike LSB-MC).

    Args:
        x: Current positions (N,)
        dt: Time step
        alpha: Tail index in (1, 2]
        sigma: Noise scale (determines target density exp(-2V/sigma^2))
        V_fn: Potential function V(x)
        gradV_fn: Gradient function gradV(x)
        rng: Random generator
        clip_bounds: Optional (min, max) bounds for clipping

    Returns:
        Updated positions (N,)
    """
    # Compute drift: -c_alpha * gradV
    # For target exp(-2V/sigma^2), the drift in the SDE is -gradV
    # FLMC uses c_alpha as a rescaling factor
    c_val = c_alpha(alpha)
    grad = gradV_fn(x)
    drift = -c_val * grad

    # Alpha-stable noise
    stable_noise = sample_symmetric_alpha_stable(rng, size=x.shape, alpha=alpha)

    # Update with taming for stability
    x_new = x + dt * drift / (1.0 + dt * np.abs(drift)) + (dt ** (1.0 / alpha)) * stable_noise

    # Clip to bounds if provided
    if clip_bounds is not None:
        x_new = np.clip(x_new, clip_bounds[0], clip_bounds[1])

    return x_new


def step_flmc_2d(
    x: np.ndarray,
    dt: float,
    alpha: float,
    sigma: float,
    V_fn_xy,
    gradV_fn_xy,
    rng: np.random.Generator,
    clip_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = None
) -> np.ndarray:
    """
    One step of FLMC for 2D potential V(x, y).

    FLMC discretization:
        X_{n+1} = X_n - c_alpha * dt * gradV(X_n) + dt^(1/alpha) * Z

    where Z is 2D isotropic alpha-stable noise.

    Args:
        x: Current positions (N, 2)
        dt: Time step
        alpha: Tail index in (1, 2]
        sigma: Noise scale
        V_fn_xy: Potential V(x, y)
        gradV_fn_xy: Gradient (gradV_x, gradV_y)
        rng: Random generator
        clip_bounds: Optional ((xmin, xmax), (ymin, ymax))

    Returns:
        Updated positions (N, 2)
    """
    c_val = c_alpha(alpha)

    # Compute gradient
    gradV_x, gradV_y = gradV_fn_xy(x[:, 0], x[:, 1])
    grad = np.stack([gradV_x, gradV_y], axis=1)
    drift = -c_val * grad

    # 2D isotropic alpha-stable noise
    stable_noise = sample_symmetric_alpha_stable(rng, size=x.shape, alpha=alpha)

    # Update with taming
    norm = np.linalg.norm(drift, axis=1, keepdims=True)
    x_new = x + (dt * drift) / (1.0 + dt * norm) + (dt ** (1.0 / alpha)) * stable_noise

    # Clip to bounds if provided
    if clip_bounds is not None:
        x_new[:, 0] = np.clip(x_new[:, 0], clip_bounds[0][0], clip_bounds[0][1])
        x_new[:, 1] = np.clip(x_new[:, 1], clip_bounds[1][0], clip_bounds[1][1])

    return x_new
