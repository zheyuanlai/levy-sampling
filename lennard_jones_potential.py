#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
import warnings

HAS_POT = None
_OT_MODULE = None


EXPO_CLIP = 80.0
LOGR_CLIP = 40.0
S_CLIP = 120.0
R2_EPS = 1e-12


# ============================================================
# 1. Lennard-Jones model (paper-aligned core)
# ============================================================

def V_pair(r, lj_eps=1.0, sigma=1.0):
    """Lennard-Jones pair potential 4*eps*((sigma/r)^12-(sigma/r)^6)."""
    r = np.asarray(r)
    r_safe = np.maximum(r, np.sqrt(R2_EPS))
    sr = sigma / r_safe
    sr6 = sr**6
    sr12 = sr6**2
    return 4.0 * lj_eps * (sr12 - sr6)


def V_lj_cluster(R, lj_eps=1.0, sigma=1.0, r_soft=0.0):
    """
    Full N-particle LJ cluster potential.

    R shape:
      - (N, d), or
      - (B, N, d) for batched evaluations.
    """
    arr = np.asarray(R, dtype=float)
    squeeze = False
    if arr.ndim == 2:
        arr = arr[None, ...]
        squeeze = True
    if arr.ndim != 3:
        raise ValueError("R must have shape (N,d) or (B,N,d).")

    _, n_particles, _ = arr.shape
    v = np.zeros(arr.shape[0], dtype=float)
    soft2 = float(r_soft) ** 2

    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dr = arr[:, i, :] - arr[:, j, :]
            r2 = np.sum(dr * dr, axis=1) + soft2 + R2_EPS
            inv_r2 = (sigma * sigma) / r2
            inv_r6 = inv_r2**3
            inv_r12 = inv_r6**2
            v += 4.0 * lj_eps * (inv_r12 - inv_r6)

    if squeeze:
        return float(v[0])
    return v


def gradV_lj_cluster(R, lj_eps=1.0, sigma=1.0, r_soft=0.0):
    """
    Gradient of full N-particle LJ cluster potential.

    Returns array with same shape as R.
    """
    arr = np.asarray(R, dtype=float)
    squeeze = False
    if arr.ndim == 2:
        arr = arr[None, ...]
        squeeze = True
    if arr.ndim != 3:
        raise ValueError("R must have shape (N,d) or (B,N,d).")

    _, n_particles, _ = arr.shape
    grad = np.zeros_like(arr)
    soft2 = float(r_soft) ** 2

    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dr = arr[:, i, :] - arr[:, j, :]
            r2 = np.sum(dr * dr, axis=1) + soft2 + R2_EPS
            inv_r2 = (sigma * sigma) / r2
            inv_r6 = inv_r2**3
            inv_r12 = inv_r6**2
            coef = 24.0 * lj_eps * (inv_r6 - 2.0 * inv_r12) / r2
            force_ij = coef[:, None] * dr
            grad[:, i, :] += force_ij
            grad[:, j, :] -= force_ij

    if squeeze:
        return grad[0]
    return grad


def step_langevin_cluster(R, dt, eps, lj_eps=1.0, sigma=1.0, r_soft=0.0, rng=None):
    """
    Euler-Maruyama step for the paper SDE on (R^d)^N:
      dR_t = -0.5 * grad V(R_t) dt + eps dB_t
    """
    if rng is None:
        rng = np.random.default_rng()
    arr = np.asarray(R, dtype=float)
    grad = gradV_lj_cluster(arr, lj_eps=lj_eps, sigma=sigma, r_soft=r_soft)
    noise = eps * np.sqrt(dt) * rng.standard_normal(arr.shape)
    return arr - 0.5 * dt * grad + noise


# ============================================================
# 2. Reduced 2D potential used by existing plotting scripts
# ============================================================

def V_lj_xy(x, y, lj_eps=1.0, sigma=1.0, r_soft=0.20):
    x = np.asarray(x)
    y = np.asarray(y)
    r2 = x * x + y * y + float(r_soft) ** 2 + R2_EPS
    inv_r2 = (sigma * sigma) / r2
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    return 4.0 * lj_eps * (inv_r12 - inv_r6)


def gradV_lj_xy(x, y, lj_eps=1.0, sigma=1.0, r_soft=0.20):
    x = np.asarray(x)
    y = np.asarray(y)
    r2 = x * x + y * y + float(r_soft) ** 2 + R2_EPS
    inv_r2 = (sigma * sigma) / r2
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    coef = 24.0 * lj_eps * (inv_r6 - 2.0 * inv_r12) / r2
    return coef * x, coef * y


def logpi_lj_xy(x, y, eps, lj_eps=1.0, sigma=1.0, r_soft=0.20):
    return -V_lj_xy(x, y, lj_eps=lj_eps, sigma=sigma, r_soft=r_soft) / (eps**2)


def grad_logpi_lj_xy(x, y, eps, lj_eps=1.0, sigma=1.0, r_soft=0.20):
    gx, gy = gradV_lj_xy(x, y, lj_eps=lj_eps, sigma=sigma, r_soft=r_soft)
    return -gx / (eps**2), -gy / (eps**2)


# ============================================================
# 3. Utilities
# ============================================================

def bilinear_interp(x, y, gx, gy, F):
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
        (1.0 - wx) * (1.0 - wy) * F[iy, ix]
        + wx * (1.0 - wy) * F[iy, ix + 1]
        + (1.0 - wx) * wy * F[iy + 1, ix]
        + wx * wy * F[iy + 1, ix + 1]
    )


def gaussian_kernel_1d(sigma, dx, truncate=4.0):
    m = int(np.ceil(truncate * sigma / dx))
    xs = np.arange(-m, m + 1) * dx
    ker = np.exp(-0.5 * (xs / sigma) ** 2)
    return ker / (np.sum(ker) * dx + 1e-300)


def smooth2d_separable(P, ker_x, ker_y):
    tmp = np.array([np.convolve(row, ker_x, mode="same") for row in P])
    return np.array(
        [np.convolve(tmp[:, j], ker_y, mode="same") for j in range(P.shape[1])]
    ).T


def density_on_grid(samples, gx, gy, do_smooth=True, smoothing_sigma=0.07):
    dx = gx[1] - gx[0]
    dy = gy[1] - gy[0]
    bins_x = np.concatenate([gx - dx / 2.0, [gx[-1] + dx / 2.0]])
    bins_y = np.concatenate([gy - dy / 2.0, [gy[-1] + dy / 2.0]])
    H, _, _ = np.histogram2d(samples[:, 1], samples[:, 0], bins=[bins_y, bins_x])
    P = H / (samples.shape[0] * dx * dy + 1e-12)
    if do_smooth:
        ker = gaussian_kernel_1d(smoothing_sigma, dx)
        P = smooth2d_separable(P, ker, ker)
    return np.maximum(P, 1e-300) / (P.sum() * dx * dy + 1e-300)


def compute_errors(dens, pi, dx, dy):
    abs_err = np.abs(dens - pi)
    mae = np.sum(abs_err) * dx * dy
    l2_err = np.sqrt(np.sum((dens - pi) ** 2) * dx * dy)
    return abs_err, mae, l2_err


def sample_from_pi_grid(rng, pi, gx, gy, n_samples):
    dx = gx[1] - gx[0]
    dy = gy[1] - gy[0]
    w = (pi * dx * dy).ravel()
    w = w / (w.sum() + 1e-300)
    idx = rng.choice(w.size, size=n_samples, p=w)
    iy, ix = np.unravel_index(idx, pi.shape)
    return np.stack(
        [
            gx[ix] + (rng.random(n_samples) - 0.5) * dx,
            gy[iy] + (rng.random(n_samples) - 0.5) * dy,
        ],
        axis=1,
    )


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


def _sanitize_point_cloud(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        return np.zeros((0, 1), dtype=float)
    keep = np.all(np.isfinite(X), axis=1)
    return X[keep]


def _sliced_wasserstein2(X, Y, rng, n_proj=96, max_points=None):
    X = _sanitize_point_cloud(X)
    Y = _sanitize_point_cloud(Y)
    if X.shape[0] <= 1 or Y.shape[0] <= 1:
        return 0.0

    if max_points is not None:
        m_eff = min(int(max_points), X.shape[0], Y.shape[0])
        if X.shape[0] > m_eff:
            X = X[rng.choice(X.shape[0], m_eff, replace=False)]
        if Y.shape[0] > m_eff:
            Y = Y[rng.choice(Y.shape[0], m_eff, replace=False)]

    dim = X.shape[1]
    dirs = rng.standard_normal((int(n_proj), dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    w2_sq = np.empty(int(n_proj), dtype=float)
    for k in range(int(n_proj)):
        px = X @ dirs[k]
        py = Y @ dirs[k]
        w = _wasserstein2_1d_exact(px, py)
        w2_sq[k] = w * w
    return float(np.sqrt(np.mean(w2_sq)))


def wasserstein2(X, Y, rng, m=None):
    X = _sanitize_point_cloud(X)
    Y = _sanitize_point_cloud(Y)
    if X.shape[0] <= 1 or Y.shape[0] <= 1:
        return 0.0

    # For 1D descriptors (e.g., N=2 pair distance), use exact empirical W2.
    if X.shape[1] == 1 and Y.shape[1] == 1:
        return _wasserstein2_1d_exact(X[:, 0], Y[:, 0])

    # For multi-D descriptors, use sliced W2 for stable, deterministic tracking.
    return _sliced_wasserstein2(X, Y, rng, n_proj=96, max_points=m)


# ============================================================
# 4. Precompute invariant density, drift and Levy score
# ============================================================

def precompute_pi_b_S(
    eps,
    gx,
    gy,
    lj_eps,
    sigma,
    r_soft,
    lam,
    sigma_L,
    mults,
    pm,
    num_dirs=16,
    jump_mu=0.0,
    jump_kappa=0.0,
):
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    V = V_lj_xy(X, Y, lj_eps=lj_eps, sigma=sigma, r_soft=r_soft)
    logpi = -V / (eps**2)
    logpi -= np.max(logpi)
    pi = np.exp(np.clip(logpi, -EXPO_CLIP, 0.0))
    pi /= np.sum(pi) * (gx[1] - gx[0]) * (gy[1] - gy[0]) + 1e-300

    dVx, dVy = gradV_lj_xy(X, Y, lj_eps=lj_eps, sigma=sigma, r_soft=r_soft)
    bx, by = -0.5 * dVx, -0.5 * dVy

    angles = np.linspace(0.0, 2.0 * np.pi, num_dirs, endpoint=False)
    if jump_kappa > 1e-12:
        w = np.exp(
            np.clip(jump_kappa * np.cos(angles - jump_mu), -LOGR_CLIP, LOGR_CLIP)
        )
        dir_weights = w / (np.sum(w) + 1e-300)
    else:
        dir_weights = np.ones_like(angles) / len(angles)
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    Sx = np.zeros_like(pi)
    Sy = np.zeros_like(pi)
    thetas = np.linspace(0.05, 1.0, 17)

    for k, pk in enumerate(pm):
        z_mag = sigma_L * mults[k]
        for d_idx, u in enumerate(dirs):
            zx = z_mag * u[0]
            zy = z_mag * u[1]
            accx = np.zeros_like(pi)
            accy = np.zeros_like(pi)
            for th in thetas:
                Vp = V_lj_xy(
                    X + th * zx,
                    Y + th * zy,
                    lj_eps=lj_eps,
                    sigma=sigma,
                    r_soft=r_soft,
                )
                Vm = V_lj_xy(
                    X - th * zx,
                    Y - th * zy,
                    lj_eps=lj_eps,
                    sigma=sigma,
                    r_soft=r_soft,
                )
                rp = np.exp(np.clip(-(Vp - V) / (eps**2), -LOGR_CLIP, LOGR_CLIP))
                rm = np.exp(np.clip(-(Vm - V) / (eps**2), -LOGR_CLIP, LOGR_CLIP))
                term = 0.5 * (rm - rp)
                accx += term * zx
                accy += term * zy
            Sx += pk * dir_weights[d_idx] * (accx / len(thetas))
            Sy += pk * dir_weights[d_idx] * (accy / len(thetas))

    return (
        pi,
        bx,
        by,
        lam * np.clip(Sx, -S_CLIP, S_CLIP),
        lam * np.clip(Sy, -S_CLIP, S_CLIP),
    )


# ============================================================
# 5. Simulation kernels
# ============================================================

def step_diff(X, dt, eps, gx, gy, bx_g, by_g, rng):
    bx = bilinear_interp(X[:, 0], X[:, 1], gx, gy, bx_g)
    by = bilinear_interp(X[:, 0], X[:, 1], gx, gy, by_g)
    norm = np.sqrt(bx * bx + by * by) + 1e-8
    drift = dt * np.stack([bx, by], axis=1) / (1.0 + dt * norm)[:, None]
    noise = eps * np.sqrt(dt) * rng.standard_normal(X.shape)
    return X + drift + noise


def _sample_jump_angles(rng, size, jump_mu, jump_kappa):
    if jump_kappa > 1e-12:
        return rng.vonmises(jump_mu, jump_kappa, size=size)
    return rng.random(size) * 2.0 * np.pi


def step_levy(
    X,
    dt,
    eps,
    gx,
    gy,
    bx_g,
    by_g,
    Sx_g,
    Sy_g,
    rng,
    lam,
    sigma_L,
    mults,
    pm,
    jump_mu=0.0,
    jump_kappa=0.0,
    jump_cap=2.0,
):
    bx = bilinear_interp(X[:, 0], X[:, 1], gx, gy, bx_g)
    by = bilinear_interp(X[:, 0], X[:, 1], gx, gy, by_g)
    sx = bilinear_interp(X[:, 0], X[:, 1], gx, gy, Sx_g)
    sy = bilinear_interp(X[:, 0], X[:, 1], gx, gy, Sy_g)
    dx = bx - sx
    dy = by - sy
    norm = np.sqrt(dx * dx + dy * dy) + 1e-8
    X_new = X + dt * np.stack([dx, dy], axis=1) / (1.0 + dt * norm)[:, None]
    X_new += eps * np.sqrt(dt) * rng.standard_normal(X.shape)

    n_jumps = rng.poisson(lam * dt, size=X.shape[0])
    idx = np.where(n_jumps > 0)[0]
    if idx.size > 0:
        for i in idx:
            k = int(n_jumps[i])
            mags = sigma_L * rng.choice(mults, size=k, p=pm)
            ang = _sample_jump_angles(rng, size=k, jump_mu=jump_mu, jump_kappa=jump_kappa)
            jx = np.sum(mags * np.cos(ang))
            jy = np.sum(mags * np.sin(ang))
            if jump_cap is not None:
                mag = np.sqrt(jx * jx + jy * jy) + 1e-12
                if mag > jump_cap:
                    scale = jump_cap / mag
                    jx *= scale
                    jy *= scale
            X_new[i, 0] += jx
            X_new[i, 1] += jy
    return X_new


def step_mala(X, dt, eps, lj_eps, sigma, r_soft, rng):
    x = X[:, 0]
    y = X[:, 1]
    gx, gy = grad_logpi_lj_xy(
        x, y, eps, lj_eps=lj_eps, sigma=sigma, r_soft=r_soft
    )
    grad = np.stack([gx, gy], axis=1)

    mean = X + 0.5 * dt * grad
    proposal = mean + np.sqrt(dt) * rng.standard_normal(X.shape)

    logp_x = logpi_lj_xy(x, y, eps, lj_eps=lj_eps, sigma=sigma, r_soft=r_soft)
    logp_y = logpi_lj_xy(
        proposal[:, 0],
        proposal[:, 1],
        eps,
        lj_eps=lj_eps,
        sigma=sigma,
        r_soft=r_soft,
    )

    gyx, gyy = grad_logpi_lj_xy(
        proposal[:, 0],
        proposal[:, 1],
        eps,
        lj_eps=lj_eps,
        sigma=sigma,
        r_soft=r_soft,
    )
    grad_y = np.stack([gyx, gyy], axis=1)
    mean_y = proposal + 0.5 * dt * grad_y

    log_q_y_given_x = -np.sum((proposal - mean) ** 2, axis=1) / (2.0 * dt)
    log_q_x_given_y = -np.sum((X - mean_y) ** 2, axis=1) / (2.0 * dt)
    log_alpha = (logp_y + log_q_x_given_y) - (logp_x + log_q_y_given_x)

    accept = np.log(rng.random(X.shape[0])) < log_alpha
    X_new = X.copy()
    X_new[accept] = proposal[accept]
    return X_new, float(accept.mean())


def step_malevy(
    X,
    dt,
    eps,
    lj_eps,
    sigma,
    r_soft,
    rng,
    lam,
    sigma_L,
    mults,
    pm,
    jump_mu=0.0,
    jump_kappa=0.0,
    jump_cap=2.0,
):
    X_mid, _ = step_mala(X, dt, eps, lj_eps, sigma, r_soft, rng)
    proposal = X_mid.copy()
    n_jumps = rng.poisson(lam * dt, size=X.shape[0])
    idx = np.where(n_jumps > 0)[0]
    if idx.size > 0:
        for i in idx:
            k = int(n_jumps[i])
            mags = sigma_L * rng.choice(mults, size=k, p=pm)
            ang = _sample_jump_angles(rng, size=k, jump_mu=jump_mu, jump_kappa=jump_kappa)
            jx = np.sum(mags * np.cos(ang))
            jy = np.sum(mags * np.sin(ang))
            if jump_cap is not None:
                mag = np.sqrt(jx * jx + jy * jy) + 1e-12
                if mag > jump_cap:
                    scale = jump_cap / mag
                    jx *= scale
                    jy *= scale
            proposal[i, 0] += jx
            proposal[i, 1] += jy

    logp_x = logpi_lj_xy(
        X_mid[:, 0], X_mid[:, 1], eps, lj_eps=lj_eps, sigma=sigma, r_soft=r_soft
    )
    logp_y = logpi_lj_xy(
        proposal[:, 0], proposal[:, 1], eps, lj_eps=lj_eps, sigma=sigma, r_soft=r_soft
    )
    log_alpha = logp_y - logp_x
    accept = np.log(rng.random(X.shape[0])) < log_alpha
    X_new = X_mid.copy()
    X_new[accept] = proposal[accept]
    return X_new, float(accept.mean())


# ============================================================
# 6. High-dimensional LJ-cluster helpers (default: N=7, d=3)
# ============================================================

DEFAULT_LJ_N_PARTICLES = 7
DEFAULT_LJ_SPATIAL_DIM = 3

LJ7_N_PARTICLES = DEFAULT_LJ_N_PARTICLES
LJ7_SPATIAL_DIM = DEFAULT_LJ_SPATIAL_DIM
LJ7_DIM = LJ7_N_PARTICLES * LJ7_SPATIAL_DIM
PAIR_I, PAIR_J = np.triu_indices(LJ7_N_PARTICLES, k=1)


def configure_lj7_geometry(
    n_particles=DEFAULT_LJ_N_PARTICLES,
    spatial_dim=DEFAULT_LJ_SPATIAL_DIM,
):
    """
    Configure flattened-cluster helpers to work on (R^d)^N.
    """
    global LJ7_N_PARTICLES, LJ7_SPATIAL_DIM, LJ7_DIM, PAIR_I, PAIR_J
    n = int(n_particles)
    d = int(spatial_dim)
    if n < 2:
        raise ValueError("n_particles must be >= 2.")
    if d < 1:
        raise ValueError("spatial_dim must be >= 1.")
    LJ7_N_PARTICLES = n
    LJ7_SPATIAL_DIM = d
    LJ7_DIM = n * d
    PAIR_I, PAIR_J = np.triu_indices(n, k=1)


def _to_lj7_cluster(X):
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == LJ7_DIM:
        return arr.reshape(arr.shape[0], LJ7_N_PARTICLES, LJ7_SPATIAL_DIM)
    if (
        arr.ndim == 3
        and arr.shape[1] == LJ7_N_PARTICLES
        and arr.shape[2] == LJ7_SPATIAL_DIM
    ):
        return arr
    raise ValueError(
        f"Expected shape (batch, {LJ7_DIM}) or "
        f"(batch, {LJ7_N_PARTICLES}, {LJ7_SPATIAL_DIM})."
    )


def _from_lj7_cluster(R):
    arr = np.asarray(R, dtype=float)
    if (
        arr.ndim != 3
        or arr.shape[1] != LJ7_N_PARTICLES
        or arr.shape[2] != LJ7_SPATIAL_DIM
    ):
        raise ValueError(
            f"Expected shape (batch, {LJ7_N_PARTICLES}, {LJ7_SPATIAL_DIM})."
        )
    return arr.reshape(arr.shape[0], LJ7_DIM)


def _random_rotation_matrix(rng, dim):
    if dim == 1:
        return np.array([[-1.0 if rng.random() < 0.5 else 1.0]])
    mat = rng.standard_normal((dim, dim))
    q, r = np.linalg.qr(mat)
    sgn = np.sign(np.diag(r))
    sgn[sgn == 0.0] = 1.0
    q = q * sgn
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return q


def _apply_random_rotations(R, rng):
    for i in range(R.shape[0]):
        q = _random_rotation_matrix(rng, LJ7_SPATIAL_DIM)
        R[i] = R[i] @ q.T


def _deterministic_unit_vectors(n, dim):
    rng = np.random.default_rng(31 + 17 * n + dim)
    vec = rng.standard_normal((n, dim))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
    return vec


def canonical_lj7_configuration(sigma=1.0):
    """
    Canonical near-equilibrium template.
    - N=2: dimer at distance r0.
    - N=3, d>=2: equilateral triangle.
    - N=7, d>=3: pentagonal bipyramid.
    - Otherwise: one center + deterministic shell.
    """
    r0 = (2.0 ** (1.0 / 6.0)) * sigma
    n = LJ7_N_PARTICLES
    d = LJ7_SPATIAL_DIM
    R = np.zeros((n, d), dtype=float)

    if n == 2:
        R[0, 0] = -0.5 * r0
        R[1, 0] = 0.5 * r0
        return R

    if n == 3 and d >= 2:
        R[0, :2] = [0.0, 0.0]
        R[1, :2] = [r0, 0.0]
        R[2, :2] = [0.5 * r0, 0.5 * np.sqrt(3.0) * r0]
        R -= np.mean(R, axis=0, keepdims=True)
        return R

    if n == 7 and d >= 3:
        # Pentagonal bipyramid with nearest-neighbor distance ~r0.
        a = r0 / (2.0 * np.sin(np.pi / 5.0))
        h2 = max(r0 * r0 - a * a, 1e-12)
        h = np.sqrt(h2)
        for k in range(5):
            ang = 2.0 * np.pi * (k / 5.0)
            R[k, 0] = a * np.cos(ang)
            R[k, 1] = a * np.sin(ang)
        R[5, 2] = h
        R[6, 2] = -h
        R -= np.mean(R, axis=0, keepdims=True)
        return R

    if n > 1:
        dirs = _deterministic_unit_vectors(n - 1, d)
        R[1:, :] = r0 * dirs
    R -= np.mean(R, axis=0, keepdims=True)
    return R


def init_lj7_ensemble(n_samples, sigma, rng, noise_scale=0.15, random_rotate=True):
    base = canonical_lj7_configuration(sigma=sigma)
    R = np.repeat(base[None, :, :], n_samples, axis=0)
    R += noise_scale * rng.standard_normal(
        (n_samples, LJ7_N_PARTICLES, LJ7_SPATIAL_DIM)
    )

    if random_rotate:
        _apply_random_rotations(R, rng)

    # Remove translation mode by fixing center of mass at 0.
    R -= np.mean(R, axis=1, keepdims=True)
    return _from_lj7_cluster(R)


def init_lj7_out_of_equilibrium(
    n_samples,
    sigma,
    rng,
    scale=1.8,
    noise_scale=0.05,
    random_rotate=True,
):
    """
    Deliberately off-target initialization: expanded shell.
    """
    base = canonical_lj7_configuration(sigma=sigma * scale)
    R = np.repeat(base[None, :, :], n_samples, axis=0)
    R += noise_scale * rng.standard_normal(
        (n_samples, LJ7_N_PARTICLES, LJ7_SPATIAL_DIM)
    )

    if random_rotate:
        _apply_random_rotations(R, rng)

    R -= np.mean(R, axis=1, keepdims=True)
    return _from_lj7_cluster(R)


def remove_center_of_mass_flat(X):
    R = _to_lj7_cluster(X).copy()
    R -= np.mean(R, axis=1, keepdims=True)
    return _from_lj7_cluster(R)


def clip_flat_state(X, box_L):
    return np.clip(X, -box_L, box_L)


def reflect_flat_state(X, box_L):
    """
    Reflective box map into [-box_L, box_L] per coordinate.
    This avoids boundary mass pile-up induced by hard clipping.
    """
    if box_L is None:
        return np.asarray(X, dtype=float)
    L = float(box_L)
    if L <= 0.0:
        return np.asarray(X, dtype=float)
    z = (np.asarray(X, dtype=float) + L) % (4.0 * L)
    return np.where(z <= 2.0 * L, z, 4.0 * L - z) - L


def apply_box_constraint_flat(X, box_L, mode="reflect"):
    if mode == "clip":
        return clip_flat_state(X, box_L)
    return reflect_flat_state(X, box_L)


def pair_distance_descriptor_flat(X):
    """
    Symmetry-invariant descriptor for cluster configurations:
    sorted pairwise distances (dimension C(N,2)).
    """
    R = _to_lj7_cluster(X)
    dR = R[:, PAIR_I, :] - R[:, PAIR_J, :]
    dists = np.linalg.norm(dR, axis=2)
    return np.sort(dists, axis=1)


def build_pair_distance_reference_histogram(
    D_ref, n_bins=120, q_low=1e-3, q_high=1.0 - 1e-3
):
    vals = np.asarray(D_ref, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("D_ref has no finite values for histogram construction.")
    lo, hi = np.quantile(vals, [q_low, q_high])
    span = max(hi - lo, 1e-6)
    edges = np.linspace(lo - 0.1 * span, hi + 0.1 * span, n_bins + 1)
    widths = np.diff(edges)
    pdf_ref = _safe_hist_pdf(vals, edges)
    return edges, widths, pdf_ref


def _safe_hist_pdf(vals, edges):
    vals = np.asarray(vals, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    widths = np.diff(edges)
    if vals.size == 0:
        return np.zeros_like(widths)

    lo = float(edges[0])
    hi = float(edges[-1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(widths)

    # Keep all mass inside [lo, hi] to avoid empty in-range histograms.
    vals = np.clip(vals, lo + 1e-12, hi - 1e-12)
    counts, _ = np.histogram(vals, bins=edges, density=False)
    total = float(np.sum(counts))
    if total <= 0.0:
        return np.zeros_like(widths)
    return counts.astype(float) / (total * widths + 1e-300)


def compute_pair_distance_hist_errors(D, edges, widths, pdf_ref):
    vals = np.asarray(D, dtype=float).reshape(-1)
    pdf = _safe_hist_pdf(vals, edges)
    diff = pdf - pdf_ref
    l1 = np.sum(np.abs(diff) * widths)
    l2 = np.sqrt(np.sum(diff**2 * widths))
    return float(l1), float(l2)


def V_lj7_flat(X, lj_eps=1.0, sigma=1.0, r_soft=0.0):
    R = _to_lj7_cluster(X)
    return V_lj_cluster(R, lj_eps=lj_eps, sigma=sigma, r_soft=r_soft)


def gradV_lj7_flat(X, lj_eps=1.0, sigma=1.0, r_soft=0.0):
    R = _to_lj7_cluster(X)
    G = gradV_lj_cluster(R, lj_eps=lj_eps, sigma=sigma, r_soft=r_soft)
    return _from_lj7_cluster(G)


def step_diff_lj7(
    X, dt, eps, rng, lj_eps, sigma, r_soft, box_L, box_mode="clip"
):
    grad = gradV_lj7_flat(X, lj_eps=lj_eps, sigma=sigma, r_soft=r_soft)
    drift = -0.5 * grad
    norm = np.linalg.norm(drift, axis=1) + 1e-8
    X_new = X + dt * drift / (1.0 + dt * norm)[:, None]
    X_new += eps * np.sqrt(dt) * rng.standard_normal(X.shape)
    X_new = remove_center_of_mass_flat(X_new)
    return apply_box_constraint_flat(X_new, box_L, mode=box_mode)


def _sample_isotropic_dirs(rng, n, dim):
    vec = rng.standard_normal((n, dim))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
    return vec


def step_levy_lj7(
    X,
    dt,
    eps,
    rng,
    lj_eps,
    sigma,
    r_soft,
    lam,
    sigma_L,
    mults,
    pm,
    box_L,
    jump_cap=4.0,
    use_jump_mh=True,
    return_stats=False,
    local_jump_particles=1,
    box_mode="clip",
):
    X_new = step_diff_lj7(
        X,
        dt,
        eps,
        rng,
        lj_eps=lj_eps,
        sigma=sigma,
        r_soft=r_soft,
        box_L=box_L,
        box_mode=box_mode,
    )

    n_jumps = rng.poisson(lam * dt, size=X.shape[0])
    idx = np.where(n_jumps > 0)[0]
    if idx.size == 0:
        if return_stats:
            return X_new, 0, 0
        return X_new

    X_prop = X_new[idx].copy()
    for k_row, i in enumerate(idx):
        k = int(n_jumps[i])
        mags = sigma_L * rng.choice(mults, size=k, p=pm)
        delta = np.zeros((LJ7_N_PARTICLES, LJ7_SPATIAL_DIM), dtype=float)
        n_local = min(LJ7_N_PARTICLES, max(1, int(local_jump_particles)))
        dirs = _sample_isotropic_dirs(rng, k, LJ7_SPATIAL_DIM)
        for _ in range(k):
            mag = float(mags[_])
            ids = rng.choice(LJ7_N_PARTICLES, size=n_local, replace=False)
            # Split one jump event across a few particles for higher MH acceptance.
            step_mag = mag / np.sqrt(float(n_local))
            delta[ids, :] += step_mag * dirs[_]
        jump_vec = delta.reshape(-1)
        if jump_cap is not None:
            jn = np.linalg.norm(jump_vec) + 1e-12
            if jn > jump_cap:
                jump_vec *= jump_cap / jn
        X_prop[k_row] += jump_vec

    X_prop = remove_center_of_mass_flat(X_prop)
    X_prop = apply_box_constraint_flat(X_prop, box_L, mode=box_mode)

    if use_jump_mh:
        e_cur = V_lj7_flat(X_new[idx], lj_eps=lj_eps, sigma=sigma, r_soft=r_soft)
        e_prop = V_lj7_flat(X_prop, lj_eps=lj_eps, sigma=sigma, r_soft=r_soft)
        log_alpha = -(e_prop - e_cur) / (eps**2)
        accept = np.log(rng.random(idx.size)) < log_alpha
        n_accept = int(np.sum(accept))
        if np.any(accept):
            X_new[idx[accept]] = X_prop[accept]
    else:
        X_new[idx] = X_prop
        n_accept = int(idx.size)

    if return_stats:
        return X_new, int(idx.size), n_accept
    return X_new


def sample_reference_lj7(
    rng,
    n_ref,
    eps,
    lj_eps,
    sigma,
    r_soft,
    lam,
    sigma_L,
    mults,
    pm,
    dt_ref=0.0005,
    warmup_steps=20000,
    box_L=4.5,
    jump_cap=4.0,
    local_jump_particles=1,
    use_jump_mh=True,
    box_mode="clip",
):
    X_ref = init_lj7_ensemble(
        n_ref, sigma=sigma, rng=rng, noise_scale=0.22, random_rotate=True
    )
    for _ in range(warmup_steps):
        X_ref = step_levy_lj7(
            X_ref,
            dt_ref,
            eps,
            rng,
            lj_eps=lj_eps,
            sigma=sigma,
            r_soft=r_soft,
            lam=lam,
            sigma_L=sigma_L,
            mults=mults,
            pm=pm,
            box_L=box_L,
            jump_cap=jump_cap,
            local_jump_particles=local_jump_particles,
            use_jump_mh=use_jump_mh,
            box_mode=box_mode,
        )
    return X_ref


def run_lj7_highdim_simulation(
    seed,
    eps,
    dt,
    T,
    N,
    lj_eps,
    sigma,
    r_soft,
    lam,
    sigma_L,
    mults,
    pm,
    box_L,
    X_ref,
    D_ref,
    pd_edges,
    pd_widths,
    pd_pdf_ref,
    jump_cap=4.0,
    init_mode="expanded",
    init_scale=1.8,
    init_noise=0.05,
    return_jump_stats=False,
    local_jump_particles=1,
    use_jump_mh=True,
    box_mode="clip",
    jump_anneal_boost=0.0,
    jump_anneal_tau=2.5,
    jump_sigma_anneal=0.35,
    jump_cap_anneal=0.25,
):
    rng = np.random.default_rng(seed)
    if init_mode == "expanded":
        X_diff = init_lj7_out_of_equilibrium(
            N,
            sigma=sigma,
            rng=rng,
            scale=init_scale,
            noise_scale=init_noise,
            random_rotate=True,
        )
    else:
        X_diff = init_lj7_ensemble(
            N, sigma=sigma, rng=rng, noise_scale=0.18, random_rotate=True
        )
    X_levy = X_diff.copy()

    history = {
        "t": [],
        "w2_d": [],
        "w2_l": [],
        "l1_d": [],
        "l1_l": [],
        "l2_d": [],
        "l2_l": [],
    }

    steps = int(T / dt)
    eval_steps = _build_eval_steps(steps)
    jump_event_total = 0
    jump_accept_total = 0
    n_particle_updates = 0

    for i in range(steps + 1):
        if i in eval_steps:
            t = i * dt
            history["t"].append(t)

            D_diff = pair_distance_descriptor_flat(X_diff)
            D_levy = pair_distance_descriptor_flat(X_levy)
            history["w2_d"].append(wasserstein2(D_diff, D_ref, rng, m=None))
            history["w2_l"].append(wasserstein2(D_levy, D_ref, rng, m=None))

            l1d, l2d = compute_pair_distance_hist_errors(
                D_diff, pd_edges, pd_widths, pd_pdf_ref
            )
            l1l, l2l = compute_pair_distance_hist_errors(
                D_levy, pd_edges, pd_widths, pd_pdf_ref
            )
            history["l1_d"].append(l1d)
            history["l1_l"].append(l1l)
            history["l2_d"].append(l2d)
            history["l2_l"].append(l2l)

        X_diff = step_diff_lj7(
            X_diff,
            dt,
            eps,
            rng,
            lj_eps=lj_eps,
            sigma=sigma,
            r_soft=r_soft,
            box_L=box_L,
            box_mode=box_mode,
        )
        t_cur = i * dt
        if jump_anneal_boost > 0.0 and jump_anneal_tau > 0.0:
            a = np.exp(-t_cur / jump_anneal_tau)
            lam_t = lam * (1.0 + jump_anneal_boost * a)
            sigma_t = sigma_L * (1.0 + jump_sigma_anneal * jump_anneal_boost * a)
            if jump_cap is None:
                jump_cap_t = None
            else:
                jump_cap_t = jump_cap * (1.0 + jump_cap_anneal * jump_anneal_boost * a)
        else:
            lam_t = lam
            sigma_t = sigma_L
            jump_cap_t = jump_cap
        X_levy, n_event, n_accept = step_levy_lj7(
            X_levy,
            dt,
            eps,
            rng,
            lj_eps=lj_eps,
            sigma=sigma,
            r_soft=r_soft,
            lam=lam_t,
            sigma_L=sigma_t,
            mults=mults,
            pm=pm,
            box_L=box_L,
            jump_cap=jump_cap_t,
            local_jump_particles=local_jump_particles,
            use_jump_mh=use_jump_mh,
            box_mode=box_mode,
            return_stats=True,
        )
        jump_event_total += n_event
        jump_accept_total += n_accept
        n_particle_updates += X_levy.shape[0]

    if return_jump_stats:
        stats = {
            "event_rate": jump_event_total / max(n_particle_updates, 1),
            "accept_rate_given_event": jump_accept_total / max(jump_event_total, 1),
            "accepted_event_rate": jump_accept_total / max(n_particle_updates, 1),
        }
        return history, X_diff, X_levy, stats
    return history, X_diff, X_levy


def aggregate_histories(histories, keys):
    stacked = {k: np.stack([np.array(h[k]) for h in histories], axis=0) for k in keys}
    mean = {k: stacked[k].mean(axis=0) for k in keys}
    std = {k: stacked[k].std(axis=0) for k in keys}
    return mean, std


def _build_eval_steps(steps, n_linear=45, n_early=28, early_frac=0.18):
    if steps <= 2:
        return {0, int(steps)}
    linear = np.linspace(0, steps, int(n_linear), dtype=int)
    early_max = max(2, int(np.ceil(early_frac * steps)))
    early = np.rint(np.geomspace(1.0, float(early_max), int(n_early))).astype(int)
    points = np.unique(np.concatenate(([0, steps], linear, early)))
    return set(int(v) for v in points.tolist())


def plot_highdim_errors_over_time(t, mean, std, out_prefix):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].errorbar(
        t, mean["w2_d"], yerr=std["w2_d"], fmt="b--", label="Diffusion", alpha=0.9, capsize=2
    )
    axes[0].errorbar(
        t, mean["w2_l"], yerr=std["w2_l"], fmt="r-", label="LSB-MC", alpha=0.9, capsize=2
    )
    axes[0].set_title("Sliced W2")
    axes[0].set_xlabel("Time")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].errorbar(
        t, mean["l1_d"], yerr=std["l1_d"], fmt="b--", label="Diffusion", alpha=0.9, capsize=2
    )
    axes[1].errorbar(
        t, mean["l1_l"], yerr=std["l1_l"], fmt="r-", label="LSB-MC", alpha=0.9, capsize=2
    )
    axes[1].set_title("Marginal L1 Error")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].errorbar(
        t, mean["l2_d"], yerr=std["l2_d"], fmt="b--", label="Diffusion", alpha=0.9, capsize=2
    )
    axes[2].errorbar(
        t, mean["l2_l"], yerr=std["l2_l"], fmt="r-", label="LSB-MC", alpha=0.9, capsize=2
    )
    axes[2].set_title("Marginal L2 Error")
    axes[2].set_xlabel("Time")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_metrics.png", dpi=200)
    plt.close()


def _parse_args():
    p = argparse.ArgumentParser(
        description="LJ cluster diffusion vs Levy simulation on (R^d)^N."
    )
    p.add_argument(
        "--n-particles",
        type=int,
        default=3,
        help="Cluster size N. Start with 2 or 3 before trying 7.",
    )
    p.add_argument(
        "--spatial-dim",
        type=int,
        default=3,
        help="Spatial dimension d for each particle coordinate.",
    )
    p.add_argument(
        "--lj-eps",
        type=float,
        default=2.0,
        help="Lennard-Jones energy scale epsilon_LJ.",
    )
    p.add_argument(
        "--noise-ratio",
        type=float,
        default=0.05,
        help="Set eps = noise_ratio * epsilon_LJ (try 0.10 or 0.05).",
    )
    p.add_argument("--dt", type=float, default=0.0015, help="Time step.")
    p.add_argument("--T", type=float, default=12.0, help="Final time horizon.")
    p.add_argument(
        "--n-samples",
        type=int,
        default=2200,
        help="Number of particles in Monte Carlo ensemble.",
    )
    p.add_argument("--n-ref", type=int, default=2500, help="Reference sample size.")
    p.add_argument(
        "--warmup-steps", type=int, default=24000, help="Reference warm-up steps."
    )
    p.add_argument("--num-seeds", type=int, default=5, help="Number of random seeds.")
    p.add_argument(
        "--lam",
        type=float,
        default=None,
        help="Jump intensity. Default adapts to n-particles.",
    )
    p.add_argument(
        "--sigma-L",
        type=float,
        default=None,
        help="Base jump magnitude. Default adapts to n-particles.",
    )
    p.add_argument(
        "--jump-cap",
        type=float,
        default=None,
        help="Cap on flattened jump norm. Default adapts to n-particles.",
    )
    p.add_argument(
        "--local-jump-particles",
        type=int,
        default=None,
        help="How many particles share each jump event. Default adapts to n-particles.",
    )
    p.add_argument(
        "--init-mode",
        type=str,
        default="expanded",
        choices=["expanded", "equilibrium"],
        help="Initial ensemble mode.",
    )
    p.add_argument(
        "--init-scale",
        type=float,
        default=None,
        help="Expansion factor for expanded init. Default adapts to n-particles.",
    )
    p.add_argument(
        "--init-noise",
        type=float,
        default=None,
        help="Initialization perturbation scale. Default adapts to n-particles.",
    )
    p.add_argument(
        "--jump-anneal-boost",
        type=float,
        default=None,
        help="Early-time jump-rate boost. Default adapts to n-particles.",
    )
    p.add_argument(
        "--jump-anneal-tau",
        type=float,
        default=None,
        help="Decay timescale for jump annealing. Default adapts to n-particles.",
    )
    p.add_argument(
        "--jump-sigma-anneal",
        type=float,
        default=None,
        help="Relative jump-size anneal factor. Default adapts to n-particles.",
    )
    p.add_argument(
        "--jump-cap-anneal",
        type=float,
        default=None,
        help="Relative jump-cap anneal factor. Default adapts to n-particles.",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    configure_lj7_geometry(args.n_particles, args.spatial_dim)

    lj_eps, sigma, r_soft = float(args.lj_eps), 1.0, 0.20
    eps = float(args.noise_ratio) * lj_eps
    if args.noise_ratio > 0.20:
        warnings.warn(
            "noise_ratio > 0.2 may overly flatten low LJ barriers; "
            "consider 0.10 or 0.05.",
            RuntimeWarning,
        )
    dt = float(args.dt)
    T = float(args.T)
    N = int(args.n_samples)
    box_L = 4.5

    if args.lam is None:
        lam = 8.0 if LJ7_N_PARTICLES <= 3 else 20.0
    else:
        lam = float(args.lam)
    if args.sigma_L is None:
        sigma_L = 0.12 if LJ7_N_PARTICLES <= 3 else 0.20
    else:
        sigma_L = float(args.sigma_L)
    mults = [1.0, 2.0, 3.5]
    pm = [0.78, 0.18, 0.04]
    if args.jump_cap is None:
        jump_cap = 0.60 if LJ7_N_PARTICLES <= 3 else 1.10
    else:
        jump_cap = float(args.jump_cap)
    if args.local_jump_particles is None:
        local_jump_particles = 2 if LJ7_N_PARTICLES <= 3 else 1
    else:
        local_jump_particles = max(1, int(args.local_jump_particles))
    use_jump_mh = True
    box_mode = "clip"
    if args.jump_anneal_boost is None:
        jump_anneal_boost = 2.0 if LJ7_N_PARTICLES <= 3 else 0.0
    else:
        jump_anneal_boost = float(args.jump_anneal_boost)
    if args.jump_anneal_tau is None:
        jump_anneal_tau = 1.5 if LJ7_N_PARTICLES <= 3 else 2.2
    else:
        jump_anneal_tau = float(args.jump_anneal_tau)
    if args.jump_sigma_anneal is None:
        jump_sigma_anneal = 0.45 if LJ7_N_PARTICLES <= 3 else 0.35
    else:
        jump_sigma_anneal = float(args.jump_sigma_anneal)
    if args.jump_cap_anneal is None:
        jump_cap_anneal = 0.35 if LJ7_N_PARTICLES <= 3 else 0.25
    else:
        jump_cap_anneal = float(args.jump_cap_anneal)
    init_mode = args.init_mode
    if args.init_scale is None:
        init_scale = 2.6 if LJ7_N_PARTICLES <= 3 else 1.9
    else:
        init_scale = float(args.init_scale)
    if args.init_noise is None:
        init_noise = 0.05 if LJ7_N_PARTICLES <= 3 else 0.06
    else:
        init_noise = float(args.init_noise)

    num_seeds = int(args.num_seeds)
    seeds = list(range(num_seeds))

    print(
        f"Configured LJ cluster: N={LJ7_N_PARTICLES}, d={LJ7_SPATIAL_DIM}, "
        f"flat_dim={LJ7_DIM}"
    )
    print(
        f"Using epsilon_LJ={lj_eps:.3f}, eps={eps:.3f} "
        f"(noise_ratio={args.noise_ratio:.3f})"
    )
    print(
        f"Levy jump params: lam={lam:.3f}, sigma_L={sigma_L:.3f}, "
        f"jump_cap={jump_cap:.3f}"
    )
    print(
        f"Init/anneal: mode={init_mode}, init_scale={init_scale:.3f}, "
        f"init_noise={init_noise:.3f}, local_jump_particles={local_jump_particles}, "
        f"jump_anneal_boost={jump_anneal_boost:.3f}, tau={jump_anneal_tau:.3f}"
    )
    print(f"Generating reference samples in R^{LJ7_DIM}...")

    rng_ref = np.random.default_rng(2026)
    X_ref = sample_reference_lj7(
        rng_ref,
        n_ref=int(args.n_ref),
        eps=eps,
        lj_eps=lj_eps,
        sigma=sigma,
        r_soft=r_soft,
        lam=lam,
        sigma_L=sigma_L,
        mults=mults,
        pm=pm,
        dt_ref=0.0005,
        warmup_steps=int(args.warmup_steps),
        box_L=box_L,
        jump_cap=jump_cap,
        local_jump_particles=local_jump_particles,
        use_jump_mh=use_jump_mh,
        box_mode=box_mode,
    )
    D_ref = pair_distance_descriptor_flat(X_ref)
    pd_edges, pd_widths, pd_pdf_ref = build_pair_distance_reference_histogram(
        D_ref, n_bins=120
    )

    print(f"Running {num_seeds} seeds (Diffusion vs Levy PF)...")
    histories = []
    jump_stats = []
    for seed in seeds:
        history, _, _, stats = run_lj7_highdim_simulation(
            seed=seed,
            eps=eps,
            dt=dt,
            T=T,
            N=N,
            lj_eps=lj_eps,
            sigma=sigma,
            r_soft=r_soft,
            lam=lam,
            sigma_L=sigma_L,
            mults=mults,
            pm=pm,
            box_L=box_L,
            X_ref=X_ref,
            D_ref=D_ref,
            pd_edges=pd_edges,
            pd_widths=pd_widths,
            pd_pdf_ref=pd_pdf_ref,
            jump_cap=jump_cap,
            init_mode=init_mode,
            init_scale=init_scale,
            init_noise=init_noise,
            return_jump_stats=True,
            local_jump_particles=local_jump_particles,
            use_jump_mh=use_jump_mh,
            box_mode=box_mode,
            jump_anneal_boost=jump_anneal_boost,
            jump_anneal_tau=jump_anneal_tau,
            jump_sigma_anneal=jump_sigma_anneal,
            jump_cap_anneal=jump_cap_anneal,
        )
        histories.append(history)
        jump_stats.append(stats)

    t = np.array(histories[0]["t"])
    keys = ["w2_d", "w2_l", "l1_d", "l1_l", "l2_d", "l2_l"]
    mean, std = aggregate_histories(histories, keys)

    out_prefix = f"lennard_jones_n{LJ7_N_PARTICLES}_d{LJ7_SPATIAL_DIM}_highdim"
    plot_highdim_errors_over_time(t, mean, std, out_prefix)

    print(f"Done. Saved: {out_prefix}_metrics.png")
    if jump_stats:
        ev = float(np.mean([s["event_rate"] for s in jump_stats]))
        acc = float(np.mean([s["accept_rate_given_event"] for s in jump_stats]))
        aev = float(np.mean([s["accepted_event_rate"] for s in jump_stats]))
        print(f"Levy jump event rate per particle-step: {ev:.4f}")
        print(f"Levy jump accept rate given event: {acc:.4f}")
        print(f"Levy accepted event rate per particle-step: {aev:.4f}")


if __name__ == "__main__":
    main()

# python mae_l2/lennard_jones_potential.py --n-particles 3 --spatial-dim 3 --T 12 --num-seeds 10