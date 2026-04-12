#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

    WARNING: This generates COORDINATEWISE INDEPENDENT alpha-stable samples.
    For genuinely isotropic high-dimensional alpha-stable vectors, use
    sample_isotropic_alpha_stable_vector() instead.

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


def sample_isotropic_alpha_stable_vector(
    rng: np.random.Generator,
    n_samples: int,
    dim: int,
    alpha: float
) -> np.ndarray:
    """
    Sample isotropic alpha-stable vectors in R^d.

    For genuinely isotropic (rotationally invariant) alpha-stable vectors,
    we use the representation:
        Z = R * U
    where:
        - U ~ Uniform(S^(d-1)) is a uniformly distributed direction
        - R ~ S_alpha^(1/alpha) is a radial component

    This ensures the distribution is invariant under rotations, matching
    the isotropic structure used in LSBMC jumps.

    Args:
        rng: NumPy random generator
        n_samples: Number of vectors to generate
        dim: Dimension of each vector
        alpha: Tail index in (1, 2]

    Returns:
        Array of shape (n_samples, dim) with isotropic alpha-stable vectors

    Reference:
        Nolan, J. P. (2020). Univariate Stable Distributions.
        Springer Series in Operations Research and Financial Engineering.
    """
    if not (1.0 < alpha <= 2.0):
        raise ValueError(f"Alpha must satisfy 1 < alpha <= 2, got {alpha}")

    # Generate uniform directions on unit sphere S^(d-1)
    directions = rng.standard_normal((n_samples, dim))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / (norms + 1e-12)

    # Generate radial components
    # For isotropic alpha-stable in R^d, radial part is S_alpha^(1/alpha)
    radii_raw = sample_symmetric_alpha_stable(rng, size=(n_samples,), alpha=alpha)
    radii = np.abs(radii_raw) ** (1.0 / alpha)

    # Combine direction and radius
    return directions * radii[:, None]


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
        X_{n+1} = X_n - c_alpha * dt * gradV(X_n) + sigma * dt^(1/alpha) * Z

    where:
        - Target density: p ∝ exp(-2V/sigma^2)
        - Z ~ symmetric alpha-stable(alpha, 0, 1, 0)
        - c_alpha = Gamma(alpha-1) / Gamma(alpha/2)^2

    NOTE: FLMC does NOT use score correction (unlike LSB-MC).
    The sigma scaling ensures temperature matching with ULA/MALA/LSBMC.

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
    # NOTE: sigma scaling ensures FLMC targets same distribution as ULA/MALA/LSBMC
    x_new = x + dt * drift / (1.0 + dt * np.abs(drift)) + sigma * (dt ** (1.0 / alpha)) * stable_noise

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
        X_{n+1} = X_n - c_alpha * dt * gradV(X_n) + sigma * dt^(1/alpha) * Z

    where Z is 2D isotropic alpha-stable noise.

    The sigma factor ensures FLMC targets p_∞ ∝ exp(-2V/sigma²), matching ULA/MALA/LSBMC.

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
    # NOTE: sigma scaling ensures FLMC targets same distribution as ULA/MALA/LSBMC
    norm = np.linalg.norm(drift, axis=1, keepdims=True)
    x_new = x + (dt * drift) / (1.0 + dt * norm) + sigma * (dt ** (1.0 / alpha)) * stable_noise

    # Clip to bounds if provided
    if clip_bounds is not None:
        x_new[:, 0] = np.clip(x_new[:, 0], clip_bounds[0][0], clip_bounds[0][1])
        x_new[:, 1] = np.clip(x_new[:, 1], clip_bounds[1][0], clip_bounds[1][1])

    return x_new


def step_flmc_nd(
    x: np.ndarray,
    dt: float,
    alpha: float,
    sigma: float,
    gradV_fn,
    rng: np.random.Generator,
    clip_bounds: Tuple[float, float] = None,
    noise_cap: float | None = None,
) -> np.ndarray:
    """
    One step of FLMC for high-dimensional potential V(x).

    FLMC discretization:
        X_{n+1} = X_n - c_alpha * dt * gradV(X_n) + sigma * dt^(1/alpha) * Z

    where Z is GENUINELY ISOTROPIC alpha-stable noise in R^d (not coordinatewise).

    The isotropic structure uses Z = R * U where:
        - U ~ Uniform(S^(d-1)) is a random direction
        - R ~ S_alpha^(1/alpha) is a radial component

    This ensures rotational invariance, matching the structure of LSBMC jumps.
    The sigma factor ensures FLMC targets p_∞ ∝ exp(-2V/sigma²), matching ULA/MALA/LSBMC.

    IMPORTANT — noise_cap for singular potentials:
    Alpha-stable noise with alpha < 2 has heavy tails: individual samples can be
    O(10^2)–O(10^4) with non-negligible probability. For singular potentials such as
    Lennard-Jones (LJ ~ r^{-12}), an uncapped noise step can push atom pairs to
    near-zero separations, producing astronomically large energies whose descriptors
    lie far outside any reference cloud. This permanently inflates Sinkhorn/MMD even
    after the subsequent tamed drift attempts recovery.  Set noise_cap to a value
    comparable to the LSBMC jump_cap (e.g. 2.0 for LJ7(2d)) to avoid this.

    Args:
        x: Current positions (N, d)
        dt: Time step
        alpha: Tail index in (1, 2]
        sigma: Noise scale
        gradV_fn: Gradient function gradV(X) returning (N, d) array
        rng: Random generator
        clip_bounds: Optional (min, max) bounds for clipping all coordinates
        noise_cap: Optional cap on the Euclidean norm of the noise vector
                   (applied before adding to state). Required for singular
                   potentials like LJ to prevent atom-overlap blowups.

    Returns:
        Updated positions (N, d)
    """
    c_val = c_alpha(alpha)
    n_samples, dim = x.shape

    # Compute gradient
    grad = gradV_fn(x)
    drift = -c_val * grad

    # Genuinely isotropic alpha-stable noise in R^d
    # Uses direction × radius decomposition to ensure rotational invariance
    stable_noise = sample_isotropic_alpha_stable_vector(rng, n_samples, dim, alpha)
    noise_vec = sigma * (dt ** (1.0 / alpha)) * stable_noise

    # Cap noise norm for singular potentials (e.g. LJ).
    # This truncates the heavy tail of the alpha-stable distribution in the
    # direction sense only; the noise retains its heavy-tailed character for
    # moderate-amplitude samples.
    if noise_cap is not None:
        noise_norm = np.linalg.norm(noise_vec, axis=1, keepdims=True)
        noise_vec = noise_vec * np.minimum(1.0, float(noise_cap) / (noise_norm + 1e-12))

    # Update with taming on drift
    # NOTE: sigma scaling ensures FLMC targets same distribution as ULA/MALA/LSBMC
    norm = np.linalg.norm(drift, axis=1, keepdims=True)
    x_new = x + (dt * drift) / (1.0 + dt * norm) + noise_vec

    # Clip to bounds if provided
    if clip_bounds is not None:
        x_new = np.clip(x_new, clip_bounds[0], clip_bounds[1])

    return x_new

