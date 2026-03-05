#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import ot  # POT

    HAS_POT = True
except Exception:
    HAS_POT = False


EXPO_CLIP = 60.0
LOGR_CLIP = 35.0
R2_EPS = 1e-12


# ============================================================
# Utilities
# ============================================================


def _normalize_probs(pm: Sequence[float]) -> np.ndarray:
    p = np.asarray(pm, dtype=float)
    p = np.maximum(p, 0.0)
    s = p.sum()
    if s <= 0:
        raise ValueError("Probability mass list pm must have positive sum.")
    return p / s


def _clip_1d(x: np.ndarray, bounds: Optional[Tuple[float, float]]) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    if bounds is None:
        return x
    return np.clip(x, bounds[0], bounds[1])


def _clip_2d(
    x: np.ndarray, bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    if bounds is None:
        return x
    x[:, 0] = np.clip(x[:, 0], bounds[0][0], bounds[0][1])
    x[:, 1] = np.clip(x[:, 1], bounds[1][0], bounds[1][1])
    return x


def c_alpha(alpha: float) -> float:
    if not (1.0 < alpha <= 2.0):
        raise ValueError("FLA alpha must satisfy 1 < alpha <= 2.")
    return math.gamma(alpha - 1.0) / (math.gamma(alpha / 2.0) ** 2)


def sample_symmetric_alpha_stable(
    rng: np.random.Generator, size: Tuple[int, ...], alpha: float
) -> np.ndarray:
    """
    Chambers-Mallows-Stuck sampler for S(alpha, beta=0, scale=1, loc=0).
    Characteristic function: exp(-|t|^alpha).
    """
    U = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=size)
    if np.isclose(alpha, 1.0):
        return np.tan(U)

    W = rng.exponential(scale=1.0, size=size)
    cos_u = np.clip(np.cos(U), 1e-12, None)
    cos_term = np.clip(np.cos((1.0 - alpha) * U), 1e-12, None)
    part1 = np.sin(alpha * U) / (cos_u ** (1.0 / alpha))
    part2 = (cos_term / W) ** ((1.0 - alpha) / alpha)
    return part1 * part2


def _tamed_increment_1d(drift: np.ndarray, dt: float) -> np.ndarray:
    return dt * drift / (1.0 + dt * np.abs(drift))


def _tamed_increment_2d(drift: np.ndarray, dt: float) -> np.ndarray:
    norm = np.linalg.norm(drift, axis=1, keepdims=True)
    return (dt * drift) / (1.0 + dt * norm)


def interp1d_values(x: np.ndarray, gx: np.ndarray, f: np.ndarray) -> np.ndarray:
    return np.interp(np.clip(x, gx[0], gx[-1]), gx, f)


def bilinear_interp(x: np.ndarray, y: np.ndarray, gx: np.ndarray, gy: np.ndarray, f: np.ndarray) -> np.ndarray:
    x = np.clip(x, gx[0], gx[-1])
    y = np.clip(y, gy[0], gy[-1])

    dx = gx[1] - gx[0]
    dy = gy[1] - gy[0]

    ix = np.clip(np.searchsorted(gx, x, side="right") - 1, 0, gx.size - 2)
    iy = np.clip(np.searchsorted(gy, y, side="right") - 1, 0, gy.size - 2)
    x1 = gx[ix]
    y1 = gy[iy]
    wx = (x - x1) / (dx + 1e-12)
    wy = (y - y1) / (dy + 1e-12)

    return (
        (1.0 - wx) * (1.0 - wy) * f[iy, ix]
        + wx * (1.0 - wy) * f[iy, ix + 1]
        + (1.0 - wx) * wy * f[iy + 1, ix]
        + wx * wy * f[iy + 1, ix + 1]
    )


def gaussian_kernel_1d(sigma: float, dx: float, truncate: float = 4.0) -> np.ndarray:
    m = int(np.ceil(truncate * sigma / dx))
    xs = np.arange(-m, m + 1) * dx
    ker = np.exp(-0.5 * (xs / sigma) ** 2)
    return ker / (np.sum(ker) * dx + 1e-300)


def smooth2d_separable(p: np.ndarray, ker_x: np.ndarray, ker_y: np.ndarray) -> np.ndarray:
    tmp = np.array([np.convolve(row, ker_x, mode="same") for row in p])
    out = np.array([np.convolve(tmp[:, j], ker_y, mode="same") for j in range(p.shape[1])]).T
    return out


def density_on_grid_1d(samples: np.ndarray, gx: np.ndarray, do_smooth: bool = True, sigma: float = 0.06) -> np.ndarray:
    dx = gx[1] - gx[0]
    bins = np.concatenate([gx - dx / 2.0, [gx[-1] + dx / 2.0]])
    hist, _ = np.histogram(samples, bins=bins)
    dens = hist.astype(float) / (samples.size * dx + 1e-12)
    if do_smooth:
        ker = gaussian_kernel_1d(sigma=max(sigma, 1.2 * dx), dx=dx)
        dens = np.convolve(dens, ker, mode="same")
    dens = np.maximum(dens, 1e-300)
    return dens / (np.sum(dens) * dx + 1e-300)


def density_on_grid_2d(
    samples: np.ndarray, gx: np.ndarray, gy: np.ndarray, do_smooth: bool = True, sigma: float = 0.08
) -> np.ndarray:
    dx = gx[1] - gx[0]
    dy = gy[1] - gy[0]
    bins_x = np.concatenate([gx - dx / 2.0, [gx[-1] + dx / 2.0]])
    bins_y = np.concatenate([gy - dy / 2.0, [gy[-1] + dy / 2.0]])
    h, _, _ = np.histogram2d(samples[:, 1], samples[:, 0], bins=[bins_y, bins_x])
    dens = h / (samples.shape[0] * dx * dy + 1e-12)
    if do_smooth:
        ker = gaussian_kernel_1d(sigma=max(sigma, 1.2 * dx), dx=dx)
        dens = smooth2d_separable(dens, ker, ker)
    dens = np.maximum(dens, 1e-300)
    return dens / (np.sum(dens) * dx * dy + 1e-300)


def compute_errors_1d(dens: np.ndarray, pi: np.ndarray, dx: float) -> Tuple[np.ndarray, float, float]:
    abs_err = np.abs(dens - pi)
    l1 = float(np.sum(abs_err) * dx)
    l2 = float(np.sqrt(np.sum((dens - pi) ** 2) * dx))
    return abs_err, l1, l2


def compute_errors_2d(dens: np.ndarray, pi: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, float, float]:
    abs_err = np.abs(dens - pi)
    l1 = float(np.sum(abs_err) * dx * dy)
    l2 = float(np.sqrt(np.sum((dens - pi) ** 2) * dx * dy))
    return abs_err, l1, l2


def sample_from_pi_grid_1d(rng: np.random.Generator, pi: np.ndarray, gx: np.ndarray, n: int) -> np.ndarray:
    dx = gx[1] - gx[0]
    w = np.asarray(pi, dtype=float) * dx
    w /= np.sum(w)
    idx = rng.choice(gx.size, size=n, p=w)
    jitter = (rng.random(n) - 0.5) * dx
    return gx[idx] + jitter


def sample_from_pi_grid_2d(rng: np.random.Generator, pi: np.ndarray, gx: np.ndarray, gy: np.ndarray, n: int) -> np.ndarray:
    dx = gx[1] - gx[0]
    dy = gy[1] - gy[0]
    w = (pi * dx * dy).ravel()
    w /= np.sum(w)
    idx = rng.choice(w.size, size=n, p=w)
    iy, ix = np.unravel_index(idx, pi.shape)
    xs = gx[ix] + (rng.random(n) - 0.5) * dx
    ys = gy[iy] + (rng.random(n) - 0.5) * dy
    return np.stack([xs, ys], axis=1)


def wasserstein2_1d_exact(x: np.ndarray, y: np.ndarray) -> float:
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


def sliced_w2_2d(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_proj: int = 64, m: int = 600) -> float:
    if x.shape[0] <= 1 or y.shape[0] <= 1:
        return 0.0
    m_use = min(m, x.shape[0], y.shape[0])
    a = x[rng.choice(x.shape[0], size=m_use, replace=False)]
    b = y[rng.choice(y.shape[0], size=m_use, replace=False)]
    dirs = rng.standard_normal((n_proj, 2))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    vals = []
    for u in dirs:
        vals.append(wasserstein2_1d_exact(a @ u, b @ u) ** 2)
    return float(np.sqrt(np.mean(vals)))


def wasserstein2_2d(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, m: int = 700) -> float:
    if x.shape[0] <= 1 or y.shape[0] <= 1:
        return 0.0
    m_use = min(m, x.shape[0], y.shape[0])
    ax = x[rng.choice(x.shape[0], size=m_use, replace=False)]
    by = y[rng.choice(y.shape[0], size=m_use, replace=False)]
    if HAS_POT:
        cost = ot.dist(ax, by, metric="sqeuclidean")
        w = np.full(m_use, 1.0 / m_use)
        return float(np.sqrt(ot.sinkhorn2(w, w, cost, reg=0.05, numItermax=5000)))
    return sliced_w2_2d(ax, by, rng, n_proj=64, m=m_use)


def aggregate_histories(histories: Sequence[Dict[str, Sequence[float]]], keys: Sequence[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    stacked = {k: np.stack([np.asarray(h[k], dtype=float) for h in histories], axis=0) for k in keys}
    mean = {k: stacked[k].mean(axis=0) for k in keys}
    std = {k: stacked[k].std(axis=0) for k in keys}
    return mean, std


# ============================================================
# Potentials (U form, where drift is -grad U)
# ============================================================


def U_doublewell(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 0.25 * x**4 - 0.5 * x**2


def gradU_doublewell(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x**3 - x


def U_ring_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r2 = x * x + y * y
    r2s = r2 + R2_EPS
    return (1.0 - r2) ** 2 + (y * y) / r2s


def gradU_ring_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r2 = x * x + y * y
    r2s = r2 + R2_EPS
    r4s = r2s * r2s
    dUx = 4.0 * (r2 - 1.0) * x - 2.0 * x * (y * y) / r4s
    dUy = 4.0 * (r2 - 1.0) * y + 2.0 * y * (x * x) / r4s
    return dUx, dUy


def U_fourwell_xy(x: np.ndarray, y: np.ndarray, a: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return 0.5 * ((x * x - a * a) ** 2 + (y * y - a * a) ** 2)


def gradU_fourwell_xy(x: np.ndarray, y: np.ndarray, a: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dUx = 2.0 * x * (x * x - a * a)
    dUy = 2.0 * y * (y * y - a * a)
    return dUx, dUy


MUELLER_SCALE = 0.05
MUELLER_PARAMS = np.array(
    [
        [-200.0, -1.0, 0.0, -10.0, 1.0, 0.0],
        [-200.0, -1.0, 0.0, -10.0, 0.0, 0.5],
        [-200.0, -6.5, 11.0, -6.5, -0.5, 1.5],
        [-200.0, -3.0, 0.0, -3.0, -0.8, -0.5],
    ],
    dtype=float,
)


def V_mueller_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(x)
    for p in MUELLER_PARAMS:
        A, a, b, c, x0, y0 = p
        dx = x - x0
        dy = y - y0
        out += (A * MUELLER_SCALE) * np.exp(a * dx**2 + b * dx * dy + c * dy**2)
    return out


def gradV_mueller_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dVx = np.zeros_like(x)
    dVy = np.zeros_like(y)
    for p in MUELLER_PARAMS:
        A, a, b, c, x0, y0 = p
        dx = x - x0
        dy = y - y0
        arg = a * dx**2 + b * dx * dy + c * dy**2
        exp_term = (A * MUELLER_SCALE) * np.exp(arg)
        dVx += exp_term * (2.0 * a * dx + b * dy)
        dVy += exp_term * (b * dx + 2.0 * c * dy)
    return dVx, dVy


def U_mueller_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.5 * V_mueller_xy(x, y)


def gradU_mueller_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dVx, dVy = gradV_mueller_xy(x, y)
    return 0.5 * dVx, 0.5 * dVy


# ============================================================
# LSB-MC (oracle score + jump) precompute
# ============================================================


def precompute_lsb_fields_1d(
    gx: np.ndarray,
    u_fn: Callable[[np.ndarray], np.ndarray],
    grad_u_fn: Callable[[np.ndarray], np.ndarray],
    beta: float,
    lam: float,
    sigma_L: float,
    multipliers: np.ndarray,
    pm: np.ndarray,
    n_theta: int = 17,
    s_clip: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u0 = u_fn(gx)
    logpi = -beta * u0
    logpi -= np.max(logpi)
    pi = np.exp(np.clip(logpi, -EXPO_CLIP, 0.0))
    dx = gx[1] - gx[0]
    pi /= np.sum(pi) * dx + 1e-300

    b = -grad_u_fn(gx)
    s = np.zeros_like(gx)
    thetas = np.linspace(0.05, 1.0, n_theta)

    for mult, prob in zip(multipliers, pm):
        z = sigma_L * mult
        acc = np.zeros_like(gx)
        for th in thetas:
            up = u_fn(gx + th * z)
            um = u_fn(gx - th * z)
            rp = np.exp(np.clip(-beta * (up - u0), -LOGR_CLIP, LOGR_CLIP))
            rm = np.exp(np.clip(-beta * (um - u0), -LOGR_CLIP, LOGR_CLIP))
            acc += 0.5 * (rm - rp) * z
        s += prob * (acc / n_theta)

    s = lam * np.clip(s, -s_clip, s_clip)
    return pi, b, s


def precompute_lsb_fields_2d(
    gx: np.ndarray,
    gy: np.ndarray,
    u_fn_xy: Callable[[np.ndarray, np.ndarray], np.ndarray],
    grad_u_fn_xy: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    beta: float,
    lam: float,
    sigma_L: float,
    multipliers: np.ndarray,
    pm: np.ndarray,
    n_theta: int = 17,
    n_dirs: int = 16,
    s_clip: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xg, yg = np.meshgrid(gx, gy, indexing="xy")
    u0 = u_fn_xy(xg, yg)

    logpi = -beta * u0
    logpi -= np.max(logpi)
    pi = np.exp(np.clip(logpi, -EXPO_CLIP, 0.0))
    dx = gx[1] - gx[0]
    dy = gy[1] - gy[0]
    pi /= np.sum(pi) * dx * dy + 1e-300

    dUx, dUy = grad_u_fn_xy(xg, yg)
    bx, by = -dUx, -dUy

    sx = np.zeros_like(pi)
    sy = np.zeros_like(pi)
    thetas = np.linspace(0.05, 1.0, n_theta)
    ang = np.linspace(0.0, 2.0 * np.pi, n_dirs, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)

    for mult, prob in zip(multipliers, pm):
        z_mag = sigma_L * mult
        for u in dirs:
            zx = z_mag * u[0]
            zy = z_mag * u[1]
            accx = np.zeros_like(pi)
            accy = np.zeros_like(pi)
            for th in thetas:
                up = u_fn_xy(xg + th * zx, yg + th * zy)
                um = u_fn_xy(xg - th * zx, yg - th * zy)
                rp = np.exp(np.clip(-beta * (up - u0), -LOGR_CLIP, LOGR_CLIP))
                rm = np.exp(np.clip(-beta * (um - u0), -LOGR_CLIP, LOGR_CLIP))
                term = 0.5 * (rm - rp)
                accx += term * zx
                accy += term * zy
            sx += (prob / n_dirs) * (accx / n_theta)
            sy += (prob / n_dirs) * (accy / n_theta)

    sx = lam * np.clip(sx, -s_clip, s_clip)
    sy = lam * np.clip(sy, -s_clip, s_clip)
    return pi, bx, by, sx, sy


# ============================================================
# Sampler steps
# ============================================================


def step_diff_1d(
    x: np.ndarray,
    dt: float,
    sigma: float,
    gx: np.ndarray,
    b_grid: np.ndarray,
    rng: np.random.Generator,
    clip_bounds: Optional[Tuple[float, float]],
) -> np.ndarray:
    b = interp1d_values(x, gx, b_grid)
    x_new = x + _tamed_increment_1d(b, dt) + sigma * np.sqrt(dt) * rng.standard_normal(x.shape)
    return _clip_1d(x_new, clip_bounds)


def step_lsb_1d(
    x: np.ndarray,
    dt: float,
    sigma: float,
    gx: np.ndarray,
    b_grid: np.ndarray,
    s_grid: np.ndarray,
    rng: np.random.Generator,
    lam: float,
    sigma_L: float,
    multipliers: np.ndarray,
    pm: np.ndarray,
    clip_bounds: Optional[Tuple[float, float]],
) -> np.ndarray:
    b = interp1d_values(x, gx, b_grid)
    s = interp1d_values(x, gx, s_grid)
    drift = b - s
    x_new = x + _tamed_increment_1d(drift, dt) + sigma * np.sqrt(dt) * rng.standard_normal(x.shape)

    n = x.shape[0]
    jump_counts = rng.poisson(lam * dt, size=n)
    total = int(np.sum(jump_counts))
    if total > 0:
        rows = np.repeat(np.arange(n), jump_counts)
        mags = sigma_L * rng.choice(multipliers, size=total, p=pm)
        signs = rng.choice(np.array([-1.0, 1.0]), size=total)
        np.add.at(x_new, rows, mags * signs)

    return _clip_1d(x_new, clip_bounds)


def step_fla_1d(
    x: np.ndarray,
    dt: float,
    alpha: float,
    c_alpha_val: float,
    beta: float,
    grad_u_fn: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    clip_bounds: Optional[Tuple[float, float]],
) -> np.ndarray:
    grad_target = beta * grad_u_fn(x)
    stable_noise = sample_symmetric_alpha_stable(rng, size=x.shape, alpha=alpha)
    x_new = x - dt * c_alpha_val * grad_target + (dt ** (1.0 / alpha)) * stable_noise
    return _clip_1d(x_new, clip_bounds)


def step_diff_2d(
    x: np.ndarray,
    dt: float,
    sigma: float,
    gx: np.ndarray,
    gy: np.ndarray,
    bx_grid: np.ndarray,
    by_grid: np.ndarray,
    rng: np.random.Generator,
    clip_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> np.ndarray:
    bx = bilinear_interp(x[:, 0], x[:, 1], gx, gy, bx_grid)
    by = bilinear_interp(x[:, 0], x[:, 1], gx, gy, by_grid)
    drift = np.stack([bx, by], axis=1)
    x_new = x + _tamed_increment_2d(drift, dt) + sigma * np.sqrt(dt) * rng.standard_normal(x.shape)
    return _clip_2d(x_new, clip_bounds)


def step_lsb_2d(
    x: np.ndarray,
    dt: float,
    sigma: float,
    gx: np.ndarray,
    gy: np.ndarray,
    bx_grid: np.ndarray,
    by_grid: np.ndarray,
    sx_grid: np.ndarray,
    sy_grid: np.ndarray,
    rng: np.random.Generator,
    lam: float,
    sigma_L: float,
    multipliers: np.ndarray,
    pm: np.ndarray,
    clip_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> np.ndarray:
    bx = bilinear_interp(x[:, 0], x[:, 1], gx, gy, bx_grid)
    by = bilinear_interp(x[:, 0], x[:, 1], gx, gy, by_grid)
    sx = bilinear_interp(x[:, 0], x[:, 1], gx, gy, sx_grid)
    sy = bilinear_interp(x[:, 0], x[:, 1], gx, gy, sy_grid)
    drift = np.stack([bx - sx, by - sy], axis=1)
    x_new = x + _tamed_increment_2d(drift, dt) + sigma * np.sqrt(dt) * rng.standard_normal(x.shape)

    n = x.shape[0]
    jump_counts = rng.poisson(lam * dt, size=n)
    total = int(np.sum(jump_counts))
    if total > 0:
        rows = np.repeat(np.arange(n), jump_counts)
        mags = sigma_L * rng.choice(multipliers, size=total, p=pm)
        ang = rng.random(total) * 2.0 * np.pi
        jumps = np.stack([mags * np.cos(ang), mags * np.sin(ang)], axis=1)
        np.add.at(x_new, rows, jumps)

    return _clip_2d(x_new, clip_bounds)


def step_fla_2d(
    x: np.ndarray,
    dt: float,
    alpha: float,
    c_alpha_val: float,
    beta: float,
    grad_u_fn_xy: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    clip_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> np.ndarray:
    gx, gy = grad_u_fn_xy(x[:, 0], x[:, 1])
    grad_target = beta * np.stack([gx, gy], axis=1)
    stable_noise = sample_symmetric_alpha_stable(rng, size=x.shape, alpha=alpha)
    x_new = x - dt * c_alpha_val * grad_target + (dt ** (1.0 / alpha)) * stable_noise
    return _clip_2d(x_new, clip_bounds)


# ============================================================
# Benchmark configs
# ============================================================


@dataclass
class Benchmark1DConfig:
    name: str
    out_prefix: str
    sigma: float
    dt: float
    T: float
    N: int
    gx: np.ndarray
    lam: float
    sigma_L: float
    multipliers: np.ndarray
    pm: np.ndarray
    alpha: float
    init_mean: float
    init_std: float
    clip_bounds: Optional[Tuple[float, float]]
    n_ref: int
    u_fn: Callable[[np.ndarray], np.ndarray]
    grad_u_fn: Callable[[np.ndarray], np.ndarray]
    n_theta_score: int = 17
    s_clip: float = 80.0


@dataclass
class Benchmark2DConfig:
    name: str
    out_prefix: str
    sigma: float
    dt: float
    T: float
    N: int
    gx: np.ndarray
    gy: np.ndarray
    lam: float
    sigma_L: float
    multipliers: np.ndarray
    pm: np.ndarray
    alpha: float
    init_mean: Tuple[float, float]
    init_std: float
    clip_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
    n_ref: int
    u_fn_xy: Callable[[np.ndarray, np.ndarray], np.ndarray]
    grad_u_fn_xy: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    n_theta_score: int = 17
    n_dir_score: int = 16
    s_clip: float = 80.0


# ============================================================
# Run + plot (1D)
# ============================================================


def run_benchmark_1d(cfg: Benchmark1DConfig, seeds: Sequence[int]) -> None:
    pm = _normalize_probs(cfg.pm)
    multipliers = np.asarray(cfg.multipliers, dtype=float)
    beta = 2.0 / (cfg.sigma**2)
    c_val = c_alpha(cfg.alpha)

    dx = cfg.gx[1] - cfg.gx[0]
    pi, b_grid, s_grid = precompute_lsb_fields_1d(
        gx=cfg.gx,
        u_fn=cfg.u_fn,
        grad_u_fn=cfg.grad_u_fn,
        beta=beta,
        lam=cfg.lam,
        sigma_L=cfg.sigma_L,
        multipliers=multipliers,
        pm=pm,
        n_theta=cfg.n_theta_score,
        s_clip=cfg.s_clip,
    )

    histories = []
    first_final = None
    steps = int(round(cfg.T / cfg.dt))
    check = max(1, steps // 25)

    print(f"[{cfg.name}] steps={steps}, N={cfg.N}, alpha={cfg.alpha:.3f}, c_alpha={c_val:.5f}")

    for seed in seeds:
        rng = np.random.default_rng(seed)
        ref = sample_from_pi_grid_1d(rng, pi, cfg.gx, cfg.n_ref)

        x0 = cfg.init_mean + cfg.init_std * rng.standard_normal(cfg.N)
        x_diff = _clip_1d(x0.copy(), cfg.clip_bounds)
        x_fla = _clip_1d(x0.copy(), cfg.clip_bounds)
        x_lsb = _clip_1d(x0.copy(), cfg.clip_bounds)

        hist = {
            "t": [],
            "w2_d": [],
            "w2_f": [],
            "w2_l": [],
            "l1_d": [],
            "l1_f": [],
            "l1_l": [],
            "l2_d": [],
            "l2_f": [],
            "l2_l": [],
        }

        for i in range(steps + 1):
            if i % check == 0 or i == steps:
                t = i * cfg.dt
                hist["t"].append(t)

                hist["w2_d"].append(wasserstein2_1d_exact(x_diff, ref))
                hist["w2_f"].append(wasserstein2_1d_exact(x_fla, ref))
                hist["w2_l"].append(wasserstein2_1d_exact(x_lsb, ref))

                d_dens = density_on_grid_1d(x_diff, cfg.gx)
                f_dens = density_on_grid_1d(x_fla, cfg.gx)
                l_dens = density_on_grid_1d(x_lsb, cfg.gx)

                _, l1_d, l2_d = compute_errors_1d(d_dens, pi, dx)
                _, l1_f, l2_f = compute_errors_1d(f_dens, pi, dx)
                _, l1_l, l2_l = compute_errors_1d(l_dens, pi, dx)

                hist["l1_d"].append(l1_d)
                hist["l1_f"].append(l1_f)
                hist["l1_l"].append(l1_l)
                hist["l2_d"].append(l2_d)
                hist["l2_f"].append(l2_f)
                hist["l2_l"].append(l2_l)

            if i == steps:
                break

            x_diff = step_diff_1d(
                x_diff,
                dt=cfg.dt,
                sigma=cfg.sigma,
                gx=cfg.gx,
                b_grid=b_grid,
                rng=rng,
                clip_bounds=cfg.clip_bounds,
            )
            x_fla = step_fla_1d(
                x_fla,
                dt=cfg.dt,
                alpha=cfg.alpha,
                c_alpha_val=c_val,
                beta=beta,
                grad_u_fn=cfg.grad_u_fn,
                rng=rng,
                clip_bounds=cfg.clip_bounds,
            )
            x_lsb = step_lsb_1d(
                x_lsb,
                dt=cfg.dt,
                sigma=cfg.sigma,
                gx=cfg.gx,
                b_grid=b_grid,
                s_grid=s_grid,
                rng=rng,
                lam=cfg.lam,
                sigma_L=cfg.sigma_L,
                multipliers=multipliers,
                pm=pm,
                clip_bounds=cfg.clip_bounds,
            )

        histories.append(hist)
        if first_final is None:
            first_final = (x_diff.copy(), x_fla.copy(), x_lsb.copy())

    t = np.asarray(histories[0]["t"], dtype=float)
    keys = ["w2_d", "w2_f", "w2_l", "l1_d", "l1_f", "l1_l", "l2_d", "l2_f", "l2_l"]
    mean, std = aggregate_histories(histories, keys)
    plot_metrics_three_way(t, mean, std, cfg.out_prefix, cfg.name)

    x_diff, x_fla, x_lsb = first_final
    plot_density_final_1d(cfg.gx, pi, x_diff, x_fla, x_lsb, cfg.out_prefix, cfg.name)

    print(
        f"[{cfg.name}] final mean metrics: "
        f"W2(D/F/L)=({mean['w2_d'][-1]:.4f}/{mean['w2_f'][-1]:.4f}/{mean['w2_l'][-1]:.4f}), "
        f"L1(D/F/L)=({mean['l1_d'][-1]:.4f}/{mean['l1_f'][-1]:.4f}/{mean['l1_l'][-1]:.4f})"
    )


# ============================================================
# Run + plot (2D)
# ============================================================


def run_benchmark_2d(cfg: Benchmark2DConfig, seeds: Sequence[int]) -> None:
    pm = _normalize_probs(cfg.pm)
    multipliers = np.asarray(cfg.multipliers, dtype=float)
    beta = 2.0 / (cfg.sigma**2)
    c_val = c_alpha(cfg.alpha)

    dx = cfg.gx[1] - cfg.gx[0]
    dy = cfg.gy[1] - cfg.gy[0]
    pi, bx, by, sx, sy = precompute_lsb_fields_2d(
        gx=cfg.gx,
        gy=cfg.gy,
        u_fn_xy=cfg.u_fn_xy,
        grad_u_fn_xy=cfg.grad_u_fn_xy,
        beta=beta,
        lam=cfg.lam,
        sigma_L=cfg.sigma_L,
        multipliers=multipliers,
        pm=pm,
        n_theta=cfg.n_theta_score,
        n_dirs=cfg.n_dir_score,
        s_clip=cfg.s_clip,
    )

    histories = []
    first_final = None
    steps = int(round(cfg.T / cfg.dt))
    check = max(1, steps // 25)

    print(f"[{cfg.name}] steps={steps}, N={cfg.N}, alpha={cfg.alpha:.3f}, c_alpha={c_val:.5f}")

    for seed in seeds:
        rng = np.random.default_rng(seed)
        ref = sample_from_pi_grid_2d(rng, pi, cfg.gx, cfg.gy, cfg.n_ref)

        x0 = np.asarray(cfg.init_mean, dtype=float)[None, :] + cfg.init_std * rng.standard_normal((cfg.N, 2))
        x_diff = _clip_2d(x0.copy(), cfg.clip_bounds)
        x_fla = _clip_2d(x0.copy(), cfg.clip_bounds)
        x_lsb = _clip_2d(x0.copy(), cfg.clip_bounds)

        hist = {
            "t": [],
            "w2_d": [],
            "w2_f": [],
            "w2_l": [],
            "l1_d": [],
            "l1_f": [],
            "l1_l": [],
            "l2_d": [],
            "l2_f": [],
            "l2_l": [],
        }

        for i in range(steps + 1):
            if i % check == 0 or i == steps:
                t = i * cfg.dt
                hist["t"].append(t)

                hist["w2_d"].append(wasserstein2_2d(x_diff, ref, rng))
                hist["w2_f"].append(wasserstein2_2d(x_fla, ref, rng))
                hist["w2_l"].append(wasserstein2_2d(x_lsb, ref, rng))

                d_dens = density_on_grid_2d(x_diff, cfg.gx, cfg.gy)
                f_dens = density_on_grid_2d(x_fla, cfg.gx, cfg.gy)
                l_dens = density_on_grid_2d(x_lsb, cfg.gx, cfg.gy)

                _, l1_d, l2_d = compute_errors_2d(d_dens, pi, dx, dy)
                _, l1_f, l2_f = compute_errors_2d(f_dens, pi, dx, dy)
                _, l1_l, l2_l = compute_errors_2d(l_dens, pi, dx, dy)

                hist["l1_d"].append(l1_d)
                hist["l1_f"].append(l1_f)
                hist["l1_l"].append(l1_l)
                hist["l2_d"].append(l2_d)
                hist["l2_f"].append(l2_f)
                hist["l2_l"].append(l2_l)

            if i == steps:
                break

            x_diff = step_diff_2d(
                x_diff,
                dt=cfg.dt,
                sigma=cfg.sigma,
                gx=cfg.gx,
                gy=cfg.gy,
                bx_grid=bx,
                by_grid=by,
                rng=rng,
                clip_bounds=cfg.clip_bounds,
            )
            x_fla = step_fla_2d(
                x_fla,
                dt=cfg.dt,
                alpha=cfg.alpha,
                c_alpha_val=c_val,
                beta=beta,
                grad_u_fn_xy=cfg.grad_u_fn_xy,
                rng=rng,
                clip_bounds=cfg.clip_bounds,
            )
            x_lsb = step_lsb_2d(
                x_lsb,
                dt=cfg.dt,
                sigma=cfg.sigma,
                gx=cfg.gx,
                gy=cfg.gy,
                bx_grid=bx,
                by_grid=by,
                sx_grid=sx,
                sy_grid=sy,
                rng=rng,
                lam=cfg.lam,
                sigma_L=cfg.sigma_L,
                multipliers=multipliers,
                pm=pm,
                clip_bounds=cfg.clip_bounds,
            )

        histories.append(hist)
        if first_final is None:
            first_final = (x_diff.copy(), x_fla.copy(), x_lsb.copy())

    t = np.asarray(histories[0]["t"], dtype=float)
    keys = ["w2_d", "w2_f", "w2_l", "l1_d", "l1_f", "l1_l", "l2_d", "l2_f", "l2_l"]
    mean, std = aggregate_histories(histories, keys)
    plot_metrics_three_way(t, mean, std, cfg.out_prefix, cfg.name)

    x_diff, x_fla, x_lsb = first_final
    plot_spatial_error_2d(cfg.gx, cfg.gy, pi, x_diff, x_fla, x_lsb, cfg.out_prefix, cfg.name)

    print(
        f"[{cfg.name}] final mean metrics: "
        f"W2(D/F/L)=({mean['w2_d'][-1]:.4f}/{mean['w2_f'][-1]:.4f}/{mean['w2_l'][-1]:.4f}), "
        f"L1(D/F/L)=({mean['l1_d'][-1]:.4f}/{mean['l1_f'][-1]:.4f}/{mean['l1_l'][-1]:.4f})"
    )


# ============================================================
# Plot helpers
# ============================================================


def plot_metrics_three_way(
    t: np.ndarray, mean: Dict[str, np.ndarray], std: Dict[str, np.ndarray], out_prefix: str, title: str
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].errorbar(t, mean["w2_d"], yerr=std["w2_d"], fmt="b--", label="Diffusion", alpha=0.9, capsize=2)
    axes[0].errorbar(t, mean["w2_f"], yerr=std["w2_f"], fmt="g-", label="FLA", alpha=0.9, capsize=2)
    axes[0].errorbar(t, mean["w2_l"], yerr=std["w2_l"], fmt="r-", label="LSB-MC", alpha=0.9, capsize=2)
    axes[0].set_title("Wasserstein-2")
    axes[0].set_xlabel("Time")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].errorbar(t, mean["l1_d"], yerr=std["l1_d"], fmt="b--", label="Diffusion", alpha=0.9, capsize=2)
    axes[1].errorbar(t, mean["l1_f"], yerr=std["l1_f"], fmt="g-", label="FLA", alpha=0.9, capsize=2)
    axes[1].errorbar(t, mean["l1_l"], yerr=std["l1_l"], fmt="r-", label="LSB-MC", alpha=0.9, capsize=2)
    axes[1].set_title("L1 Error")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].errorbar(t, mean["l2_d"], yerr=std["l2_d"], fmt="b--", label="Diffusion", alpha=0.9, capsize=2)
    axes[2].errorbar(t, mean["l2_f"], yerr=std["l2_f"], fmt="g-", label="FLA", alpha=0.9, capsize=2)
    axes[2].errorbar(t, mean["l2_l"], yerr=std["l2_l"], fmt="r-", label="LSB-MC", alpha=0.9, capsize=2)
    axes[2].set_title("L2 Error")
    axes[2].set_xlabel("Time")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_metrics.png", dpi=200)
    plt.close()


def plot_density_final_1d(
    gx: np.ndarray,
    pi: np.ndarray,
    x_diff: np.ndarray,
    x_fla: np.ndarray,
    x_lsb: np.ndarray,
    out_prefix: str,
    title: str,
) -> None:
    dens_d = density_on_grid_1d(x_diff, gx)
    dens_f = density_on_grid_1d(x_fla, gx)
    dens_l = density_on_grid_1d(x_lsb, gx)

    plt.figure(figsize=(8, 5))
    plt.plot(gx, pi, "k--", lw=2.0, label="Target")
    plt.plot(gx, dens_d, "b-", lw=1.8, label="Diffusion")
    plt.plot(gx, dens_f, "g-", lw=1.8, label="FLA")
    plt.plot(gx, dens_l, "r-", lw=1.8, label="LSB-MC")
    plt.title(f"{title}: Final Density")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_final_density.png", dpi=220)
    plt.close()


def plot_spatial_error_2d(
    gx: np.ndarray,
    gy: np.ndarray,
    pi: np.ndarray,
    x_diff: np.ndarray,
    x_fla: np.ndarray,
    x_lsb: np.ndarray,
    out_prefix: str,
    title: str,
) -> None:
    dens_d = density_on_grid_2d(x_diff, gx, gy)
    dens_f = density_on_grid_2d(x_fla, gx, gy)
    dens_l = density_on_grid_2d(x_lsb, gx, gy)
    err_d = np.abs(dens_d - pi)
    err_f = np.abs(dens_f - pi)
    err_l = np.abs(dens_l - pi)
    vmax = max(np.max(err_d), np.max(err_f), np.max(err_l))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    extent = [gx[0], gx[-1], gy[0], gy[-1]]

    im0 = axes[0].imshow(err_d, origin="lower", extent=extent, cmap="hot", vmin=0, vmax=vmax)
    axes[0].set_title("Diffusion |dens - pi|")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(err_f, origin="lower", extent=extent, cmap="hot", vmin=0, vmax=vmax)
    axes[1].set_title("FLA |dens - pi|")
    axes[1].set_xlabel("x")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(err_l, origin="lower", extent=extent, cmap="hot", vmin=0, vmax=vmax)
    axes[2].set_title("LSB-MC |dens - pi|")
    axes[2].set_xlabel("x")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"{title}: Final Spatial Error", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_spatial_error.png", dpi=220)
    plt.close()


# ============================================================
# Main: all 4 requested examples
# ============================================================


def main() -> None:
    # Shared FLA tail index. Common choice in Fractional Langevin sampling: alpha in (1, 2).
    alpha = 1.50
    seeds = [0, 1, 2]
    output_dir = os.path.join(os.path.dirname(__file__), "fla_figures")
    os.makedirs(output_dir, exist_ok=True)

    # 1) Ginzburg-Landau double-well (1D)
    # dX = (X - X^3) dt + eps dB
    # => U(x) = 1/4 x^4 - 1/2 x^2, sigma = eps
    cfg_doublewell = Benchmark1DConfig(
        name="Ginzburg-Landau Double-Well (1D)",
        out_prefix=os.path.join(output_dir, "fla_doublewell"),
        sigma=0.35,
        dt=1e-3,
        T=40.0,
        N=8000,
        gx=np.linspace(-8.0, 8.0, 2401),
        lam=1.0,
        sigma_L=1.6,
        multipliers=np.array([1.0, 1.8, 2.6], dtype=float),
        pm=np.array([0.70, 0.22, 0.08], dtype=float),
        alpha=alpha,
        init_mean=-1.2,
        init_std=0.08,
        clip_bounds=(-10.0, 10.0),
        n_ref=3000,
        u_fn=U_doublewell,
        grad_u_fn=gradU_doublewell,
        n_theta_score=23,
        s_clip=80.0,
    )

    # 2) Ring potential (2D)
    # dX = -dVx dt + sqrt(eps) dB1, dY = -dVy dt + sqrt(eps) dB2
    # => U = V, sigma^2 = eps
    cfg_ring = Benchmark2DConfig(
        name="Ring Potential (2D)",
        out_prefix=os.path.join(output_dir, "fla_ring"),
        sigma=np.sqrt(0.35),
        dt=0.0015,
        T=40.0,
        N=5000,
        gx=np.linspace(-2.2, 2.2, 220),
        gy=np.linspace(-2.2, 2.2, 220),
        lam=1.6,
        sigma_L=1.25,
        multipliers=np.array([1.0, 1.7, 2.4], dtype=float),
        pm=np.array([0.70, 0.22, 0.08], dtype=float),
        alpha=alpha,
        init_mean=(-1.0, 0.0),
        init_std=0.05,
        clip_bounds=((-3.0, 3.0), (-3.0, 3.0)),
        n_ref=2500,
        u_fn_xy=U_ring_xy,
        grad_u_fn_xy=gradU_ring_xy,
        n_theta_score=17,
        n_dir_score=16,
        s_clip=50.0,
    )

    # 3) Symmetric four-well potential (2D)
    # dX = -0.5 dVx dt + eps dB1, dY = -0.5 dVy dt + eps dB2
    # => U = 0.5 * V, sigma = eps
    cfg_fourwell = Benchmark2DConfig(
        name="Four-Well Potential (2D)",
        out_prefix=os.path.join(output_dir, "fla_fourwell"),
        sigma=0.50,
        dt=0.005,
        T=15.0,
        N=5000,
        gx=np.linspace(-1.8, 1.8, 200),
        gy=np.linspace(-1.8, 1.8, 200),
        lam=1.2,
        sigma_L=1.0,
        multipliers=np.array([1.0, 2.0], dtype=float),
        pm=np.array([0.85, 0.15], dtype=float),
        alpha=alpha,
        init_mean=(1.0, 1.0),
        init_std=0.10,
        clip_bounds=((-2.5, 2.5), (-2.5, 2.5)),
        n_ref=2500,
        u_fn_xy=U_fourwell_xy,
        grad_u_fn_xy=gradU_fourwell_xy,
        n_theta_score=17,
        n_dir_score=12,
        s_clip=80.0,
    )

    # 4) Muller-Brown potential (2D)
    # dX = -0.5 dVx dt + eps dB1, dY = -0.5 dVy dt + eps dB2
    # => U = 0.5 * V, sigma = eps
    cfg_mueller = Benchmark2DConfig(
        name="Muller-Brown Potential (2D)",
        out_prefix=os.path.join(output_dir, "fla_mueller"),
        sigma=0.70,
        dt=2e-4,
        T=2.0,
        N=3000,
        gx=np.linspace(-1.8, 1.8, 200),
        gy=np.linspace(-1.2, 2.2, 200),
        lam=8.0,
        sigma_L=1.2,
        multipliers=np.array([0.8, 1.2], dtype=float),
        pm=np.array([0.5, 0.5], dtype=float),
        alpha=alpha,
        init_mean=(1.0, 0.0),
        init_std=0.10,
        clip_bounds=((-2.2, 2.2), (-1.5, 2.5)),
        n_ref=2200,
        u_fn_xy=U_mueller_xy,
        grad_u_fn_xy=gradU_mueller_xy,
        n_theta_score=20,
        n_dir_score=16,
        s_clip=100.0,
    )

    print("Running 4 benchmarks: Diffusion vs FLA vs LSB-MC")
    if not HAS_POT:
        print("POT not found: falling back to sliced-W2 in 2D.")

    run_benchmark_1d(cfg_doublewell, seeds)
    run_benchmark_2d(cfg_ring, seeds)
    run_benchmark_2d(cfg_fourwell, seeds)
    run_benchmark_2d(cfg_mueller, seeds)

    print(f"Done. Generated all figures under: {output_dir}")


if __name__ == "__main__":
    main()
