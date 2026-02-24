#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import ot  # POT
except Exception as e:
    raise RuntimeError("POT (ot) is required: pip install POT")

# ============================================================
# 1. Lennard-Jones Potential (2D, softened core)
# ============================================================

EXPO_CLIP = 60.0
LOGR_CLIP = 30.0
S_CLIP = 80.0


def V_lennard_jones(x, y, lj_eps=1.0, sigma=1.0, r_soft=0.20):
    x, y = np.asarray(x), np.asarray(y)
    r2 = x * x + y * y + r_soft * r_soft
    inv_r2 = (sigma * sigma) / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    return 4.0 * lj_eps * (inv_r12 - inv_r6)


def gradV_lennard_jones(x, y, lj_eps=1.0, sigma=1.0, r_soft=0.20):
    x, y = np.asarray(x), np.asarray(y)
    r2 = x * x + y * y + r_soft * r_soft
    inv_r2 = (sigma * sigma) / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    pref = 24.0 * lj_eps * (inv_r6 - 2.0 * inv_r12) / r2
    dVx = pref * x
    dVy = pref * y
    return dVx, dVy


# ============================================================
# 2. Utilities & Density Estimation
# ============================================================

def bilinear_interp(x, y, gx, gy, F):
    x, y = np.clip(x, gx[0], gx[-1]), np.clip(y, gy[0], gy[-1])
    dx, dy = gx[1] - gx[0], gy[1] - gy[0]
    ix = np.clip(np.searchsorted(gx, x, side="right") - 1, 0, gx.size - 2)
    iy = np.clip(np.searchsorted(gy, y, side="right") - 1, 0, gy.size - 2)
    x1, y1 = gx[ix], gy[iy]
    wx, wy = (x - x1) / (dx + 1e-12), (y - y1) / (dy + 1e-12)
    return (1 - wx) * (1 - wy) * F[iy, ix] + wx * (1 - wy) * F[iy, ix + 1] + (1 - wx) * wy * F[iy + 1, ix] + wx * wy * F[iy + 1, ix + 1]


def gaussian_kernel_1d(sigma, dx):
    m = int(np.ceil(4.0 * sigma / dx))
    xs = np.arange(-m, m + 1) * dx
    ker = np.exp(-0.5 * (xs / sigma) ** 2)
    return ker / (np.sum(ker) * dx + 1e-300)


def smooth2d_separable(P, ker_x, ker_y):
    tmp = np.array([np.convolve(row, ker_x, mode="same") for row in P])
    return np.array([np.convolve(tmp[:, j], ker_y, mode="same") for j in range(P.shape[1])]).T


def clip_to_domain(samples, gx, gy):
    out = samples.copy()
    out[:, 0] = np.clip(out[:, 0], gx[0], gx[-1])
    out[:, 1] = np.clip(out[:, 1], gy[0], gy[-1])
    return out


def outside_fraction(samples, gx, gy):
    out_x = (samples[:, 0] < gx[0]) | (samples[:, 0] > gx[-1])
    out_y = (samples[:, 1] < gy[0]) | (samples[:, 1] > gy[-1])
    return float(np.mean(out_x | out_y))


def density_on_grid(samples, gx, gy, do_smooth=True):
    samples = clip_to_domain(samples, gx, gy)
    dx, dy = gx[1] - gx[0], gy[1] - gy[0]
    bins_x = np.concatenate([gx - dx / 2, [gx[-1] + dx / 2]])
    bins_y = np.concatenate([gy - dy / 2, [gy[-1] + dy / 2]])
    H, _, _ = np.histogram2d(samples[:, 1], samples[:, 0], bins=[bins_y, bins_x])
    P = H / (samples.shape[0] * dx * dy + 1e-12)
    if do_smooth:
        ker = gaussian_kernel_1d(0.08, dx)
        P = smooth2d_separable(P, ker, ker)
    return np.maximum(P, 1e-300) / (P.sum() * dx * dy)


# ============================================================
# 3. Error Metrics
# ============================================================

def compute_errors(dens, pi, dx, dy):
    abs_err_map = np.abs(dens - pi)
    mae = np.sum(abs_err_map) * dx * dy
    l2_err = np.sqrt(np.sum((dens - pi) ** 2) * dx * dy)
    return abs_err_map, mae, l2_err


def build_direction_quadrature(num_dirs=12, jump_mu=0.0, jump_kappa=0.0):
    angs = np.linspace(0.0, 2.0 * np.pi, num_dirs, endpoint=False)
    dirs = np.stack([np.cos(angs), np.sin(angs)], axis=1)
    if jump_kappa <= 0.0:
        w = np.full(num_dirs, 1.0 / num_dirs)
    else:
        logits = jump_kappa * np.cos(angs - jump_mu)
        logits = logits - np.max(logits)
        w = np.exp(logits)
        w = w / np.sum(w)
    return dirs, w


# ============================================================
# 4. Oracle Score & Simulation Steps
# ============================================================

def wasserstein2_stable(A, B, reg_base=0.08):
    nA = A.shape[0]
    nB = B.shape[0]
    a = np.full(nA, 1.0 / nA)
    b = np.full(nB, 1.0 / nB)

    M = ot.dist(A, B, metric="sqeuclidean")
    scale = np.percentile(M, 90.0)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = np.mean(M) + 1e-12
    M_scaled = M / (scale + 1e-12)

    # Log-domain Sinkhorn with adaptive regularization.
    for reg in (reg_base, 2.0 * reg_base, 4.0 * reg_base, 8.0 * reg_base):
        try:
            val = ot.sinkhorn2(
                a,
                b,
                M_scaled,
                reg=reg,
                method="sinkhorn_log",
                numItermax=5000,
                stopThr=1e-8,
                warn=False,
            )
            if np.isfinite(val) and val >= 0.0:
                return float(np.sqrt(val * scale))
        except Exception:
            pass

    # Fallback: exact OT on clipped cost matrix.
    M_clip = np.minimum(M, np.percentile(M, 99.5))
    val = ot.emd2(a, b, M_clip)
    return float(np.sqrt(max(val, 0.0)))


def precompute_pi_b_S(
    noise_eps,
    gx,
    gy,
    lj_eps,
    sigma,
    r_soft,
    lam,
    sigma_L,
    mults,
    pm,
    num_dirs=12,
    jump_mu=0.0,
    jump_kappa=0.0,
):
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    V = V_lennard_jones(X, Y, lj_eps=lj_eps, sigma=sigma, r_soft=r_soft)
    logpi = -V / (noise_eps ** 2)
    logpi -= np.max(logpi)
    pi = np.exp(np.clip(logpi, -EXPO_CLIP, 0.0))
    pi /= (np.sum(pi) * (gx[1] - gx[0]) * (gy[1] - gy[0]))

    dVx, dVy = gradV_lennard_jones(X, Y, lj_eps=lj_eps, sigma=sigma, r_soft=r_soft)
    bx, by = -0.5 * dVx, -0.5 * dVy

    Sx, Sy = np.zeros_like(pi), np.zeros_like(pi)
    thetas = np.linspace(0.05, 1.0, 15)
    dirs, dir_w = build_direction_quadrature(num_dirs=num_dirs, jump_mu=jump_mu, jump_kappa=jump_kappa)

    for k, pk in enumerate(pm):
        z_mag = sigma_L * mults[k]
        for j, u in enumerate(dirs):
            zx, zy = z_mag * u[0], z_mag * u[1]
            tx, ty = 0, 0
            for th in thetas:
                Vp = V_lennard_jones(X + th * zx, Y + th * zy, lj_eps=lj_eps, sigma=sigma, r_soft=r_soft)
                Vm = V_lennard_jones(X - th * zx, Y - th * zy, lj_eps=lj_eps, sigma=sigma, r_soft=r_soft)
                rp = np.exp(np.clip(-(Vp - V) / (noise_eps ** 2), -LOGR_CLIP, LOGR_CLIP))
                rm = np.exp(np.clip(-(Vm - V) / (noise_eps ** 2), -LOGR_CLIP, LOGR_CLIP))
                tx += 0.5 * (rm - rp) * zx
                ty += 0.5 * (rm - rp) * zy
            Sx += (pk * dir_w[j]) * (tx / len(thetas))
            Sy += (pk * dir_w[j]) * (ty / len(thetas))

    return pi, bx, by, lam * np.clip(Sx, -S_CLIP, S_CLIP), lam * np.clip(Sy, -S_CLIP, S_CLIP)


def step_diff(X, dt, noise_eps, gx, gy, bx_g, by_g, rng):
    bx = bilinear_interp(X[:, 0], X[:, 1], gx, gy, bx_g)
    by = bilinear_interp(X[:, 0], X[:, 1], gx, gy, by_g)
    norm = np.sqrt(bx ** 2 + by ** 2) + 1e-8
    drift = dt * np.stack([bx, by], axis=1) / (1.0 + dt * norm)[:, None]
    return X + drift + noise_eps * np.sqrt(dt) * rng.standard_normal(X.shape)


def step_levy(
    X,
    dt,
    noise_eps,
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
    jump_cap=2.5,
):
    bx = bilinear_interp(X[:, 0], X[:, 1], gx, gy, bx_g)
    by = bilinear_interp(X[:, 0], X[:, 1], gx, gy, by_g)
    sx = bilinear_interp(X[:, 0], X[:, 1], gx, gy, Sx_g)
    sy = bilinear_interp(X[:, 0], X[:, 1], gx, gy, Sy_g)
    dx, dy = bx - sx, by - sy

    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-8
    X_new = X + dt * np.stack([dx, dy], axis=1) / (1.0 + dt * norm)[:, None]
    X_new += noise_eps * np.sqrt(dt) * rng.standard_normal(X.shape)

    n_jumps = rng.poisson(lam * dt, size=X.shape[0])
    for i in np.where(n_jumps > 0)[0]:
        k = int(n_jumps[i])
        m_choice = rng.choice(mults, size=k, p=pm)
        if jump_kappa <= 0.0:
            ang = rng.random(k) * 2.0 * np.pi
        else:
            ang = rng.vonmises(mu=jump_mu, kappa=jump_kappa, size=k)
        jx = np.sum(sigma_L * m_choice * np.cos(ang))
        jy = np.sum(sigma_L * m_choice * np.sin(ang))
        if jump_cap is not None and jump_cap > 0.0:
            mag = np.sqrt(jx * jx + jy * jy) + 1e-12
            scale = min(1.0, jump_cap / mag)
            jx *= scale
            jy *= scale
        X_new[i, 0] += jx
        X_new[i, 1] += jy
    return X_new


def logpi_lennard_jones_xy(x, y, noise_eps, lj_eps, sigma, r_soft):
    return -V_lennard_jones(x, y, lj_eps=lj_eps, sigma=sigma, r_soft=r_soft) / (noise_eps ** 2)


def grad_logpi_lennard_jones_xy(x, y, noise_eps, lj_eps, sigma, r_soft):
    dVx, dVy = gradV_lennard_jones(x, y, lj_eps=lj_eps, sigma=sigma, r_soft=r_soft)
    scale = -1.0 / (noise_eps ** 2)
    return scale * dVx, scale * dVy


def step_mala(X, dt, noise_eps, lj_eps, sigma, r_soft, rng):
    x = X[:, 0]
    y = X[:, 1]
    gx, gy = grad_logpi_lennard_jones_xy(x, y, noise_eps, lj_eps, sigma, r_soft)
    grad = np.stack([gx, gy], axis=1)

    mean = X + 0.5 * dt * grad
    proposal = mean + np.sqrt(dt) * rng.standard_normal(X.shape)

    logp_x = logpi_lennard_jones_xy(x, y, noise_eps, lj_eps, sigma, r_soft)
    logp_y = logpi_lennard_jones_xy(proposal[:, 0], proposal[:, 1], noise_eps, lj_eps, sigma, r_soft)

    gyx, gyy = grad_logpi_lennard_jones_xy(proposal[:, 0], proposal[:, 1], noise_eps, lj_eps, sigma, r_soft)
    grad_y = np.stack([gyx, gyy], axis=1)
    mean_y = proposal + 0.5 * dt * grad_y

    log_q_y_given_x = -np.sum((proposal - mean) ** 2, axis=1) / (2.0 * dt)
    log_q_x_given_y = -np.sum((X - mean_y) ** 2, axis=1) / (2.0 * dt)
    log_alpha = (logp_y + log_q_x_given_y) - (logp_x + log_q_y_given_x)

    accept = np.log(rng.random(X.shape[0])) < log_alpha
    X_new = X.copy()
    X_new[accept] = proposal[accept]
    return X_new, float(accept.mean())


def sample_from_pi(rng, pi, gx, gy, N):
    flat_pi = pi.ravel() / pi.sum()
    idx = rng.choice(flat_pi.size, size=N, p=flat_pi)
    y_idx, x_idx = np.unravel_index(idx, pi.shape)
    dx, dy = gx[1] - gx[0], gy[1] - gy[0]
    return np.stack([gx[x_idx] + (rng.random(N) - 0.5) * dx, gy[y_idx] + (rng.random(N) - 0.5) * dy], axis=1)


# ============================================================
# 5. Main Simulation
# ============================================================

def run_simulation(
    seed,
    noise_eps,
    dt,
    T,
    N,
    gx,
    gy,
    dx,
    dy,
    lam,
    sigma_L,
    mults,
    pm,
    pi,
    bx,
    by,
    Sx,
    Sy,
    sigma,
    jump_mu=0.0,
    jump_kappa=0.0,
    jump_cap=2.5,
):
    rng = np.random.default_rng(seed)
    ref = sample_from_pi(rng, pi, gx, gy, N=2000)

    init = np.array([2.5 * sigma, 0.0])
    X_diff = init + 0.08 * rng.standard_normal((N, 2))
    X_levy = X_diff.copy()

    history = {
        "t": [],
        "w2_d": [], "w2_l": [],
        "mae_d": [], "mae_l": [],
        "l2_d": [], "l2_l": [],
        "out_d": [], "out_l": [],
    }

    steps = int(T / dt)
    check = max(steps // 20, 1)

    for i in range(steps + 1):
        if i % check == 0 or i == steps:
            t = i * dt
            history["t"].append(t)

            sub = min(700, N, ref.shape[0])
            idx_ref = rng.choice(ref.shape[0], sub, replace=False)
            ref_sub = ref[idx_ref]
            A_d = clip_to_domain(X_diff[rng.choice(N, sub, replace=False)], gx, gy)
            A_l = clip_to_domain(X_levy[rng.choice(N, sub, replace=False)], gx, gy)
            history["w2_d"].append(wasserstein2_stable(A_d, ref_sub))
            history["w2_l"].append(wasserstein2_stable(A_l, ref_sub))
            history["out_d"].append(outside_fraction(X_diff, gx, gy))
            history["out_l"].append(outside_fraction(X_levy, gx, gy))

            d_dens = density_on_grid(X_diff, gx, gy)
            l_dens = density_on_grid(X_levy, gx, gy)

            _, md, l2d = compute_errors(d_dens, pi, dx, dy)
            _, ml, l2l = compute_errors(l_dens, pi, dx, dy)
            history["mae_d"].append(md)
            history["mae_l"].append(ml)
            history["l2_d"].append(l2d)
            history["l2_l"].append(l2l)

        X_diff = step_diff(X_diff, dt, noise_eps, gx, gy, bx, by, rng)
        X_levy = step_levy(
            X_levy,
            dt,
            noise_eps,
            gx,
            gy,
            bx,
            by,
            Sx,
            Sy,
            rng,
            lam,
            sigma_L,
            mults,
            pm,
            jump_mu=jump_mu,
            jump_kappa=jump_kappa,
            jump_cap=jump_cap,
        )
        # MALA baseline is intentionally disabled in this script.

    return history, X_diff, X_levy


def aggregate_histories(histories, keys):
    stacked = {k: np.stack([np.array(h[k]) for h in histories], axis=0) for k in keys}
    mean = {k: stacked[k].mean(axis=0) for k in keys}
    std = {k: stacked[k].std(axis=0) for k in keys}
    return mean, std


def main():
    # Params
    lj_eps, sigma, r_soft = 1.0, 1.0, 0.20
    noise_eps, dt, T, N = 0.50, 0.002, 20.0, 5000
    gx, gy = np.linspace(-3.2, 3.2, 220), np.linspace(-3.2, 3.2, 220)
    dx, dy = gx[1] - gx[0], gy[1] - gy[0]
    # Levy jump hyper-parameters:
    # strength: lam, sigma_L, mults, pm
    # direction: jump_mu (radian), jump_kappa (0 => isotropic), num_dirs
    # safety: jump_cap (max single jump magnitude)
    lam, sigma_L, mults, pm = 1.0, 0.65, [1.0, 1.6, 2.2], [0.78, 0.18, 0.04]
    num_dirs = 16
    jump_mu = 0.0
    jump_kappa = 0.0
    jump_cap = 2.0
    num_seeds = 5
    seeds = list(range(num_seeds))

    print("Precomputing fields...")
    pi, bx, by, Sx, Sy = precompute_pi_b_S(
        noise_eps,
        gx,
        gy,
        lj_eps,
        sigma,
        r_soft,
        lam,
        sigma_L,
        mults,
        pm,
        num_dirs=num_dirs,
        jump_mu=jump_mu,
        jump_kappa=jump_kappa,
    )

    print(f"Starting simulations over {num_seeds} seeds...")
    print(
        f"Levy params | lam={lam:.3f}, sigma_L={sigma_L:.3f}, "
        f"jump_mu={jump_mu:.3f}, jump_kappa={jump_kappa:.3f}, jump_cap={jump_cap:.3f}"
    )
    histories = []
    first_final = None
    for seed in seeds:
        history, X_diff, X_levy = run_simulation(
            seed,
            noise_eps,
            dt,
            T,
            N,
            gx,
            gy,
            dx,
            dy,
            lam,
            sigma_L,
            mults,
            pm,
            pi,
            bx,
            by,
            Sx,
            Sy,
            sigma,
            jump_mu=jump_mu,
            jump_kappa=jump_kappa,
            jump_cap=jump_cap,
        )
        histories.append(history)
        if first_final is None:
            first_final = (X_diff, X_levy)

    t = np.array(histories[0]["t"])
    keys = ["w2_d", "w2_l", "mae_d", "mae_l", "l2_d", "l2_l"]
    mean, std = aggregate_histories(histories, keys)

    # ============================================================
    # 6. Visualization
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].errorbar(t, mean["w2_d"], yerr=std["w2_d"], fmt="b--", label="Diffusion", capsize=2, alpha=0.9)
    axes[0].errorbar(t, mean["w2_l"], yerr=std["w2_l"], fmt="r-", label="Levy", capsize=2, alpha=0.9)
    axes[0].set_title("W2 Distance")
    axes[0].legend()

    axes[1].errorbar(t, mean["mae_d"], yerr=std["mae_d"], fmt="b--", label="Diffusion", capsize=2, alpha=0.9)
    axes[1].errorbar(t, mean["mae_l"], yerr=std["mae_l"], fmt="r-", label="Levy", capsize=2, alpha=0.9)
    axes[1].set_title("MAE ($L^1$) Error")
    axes[1].legend()

    axes[2].errorbar(t, mean["l2_d"], yerr=std["l2_d"], fmt="b--", label="Diffusion", capsize=2, alpha=0.9)
    axes[2].errorbar(t, mean["l2_l"], yerr=std["l2_l"], fmt="r-", label="Levy", capsize=2, alpha=0.9)
    axes[2].set_title("$L^2$ (RMSE) Error")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig("lennard_jones_errors.png", dpi=200)
    plt.close()

    X_diff, X_levy = first_final
    d_dens = density_on_grid(X_diff, gx, gy)
    l_dens = density_on_grid(X_levy, gx, gy)
    err_d, _, _ = compute_errors(d_dens, pi, dx, dy)
    err_l, _, _ = compute_errors(l_dens, pi, dx, dy)
    vmax = max(np.max(err_d), np.max(err_l))

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    im1 = ax[0].imshow(err_d, origin="lower", extent=[gx[0], gx[-1], gy[0], gy[-1]], cmap="inferno", vmin=0, vmax=vmax)
    ax[0].set_title("Diffusion Spatial Error")
    fig.colorbar(im1, ax=ax[0])
    im2 = ax[1].imshow(err_l, origin="lower", extent=[gx[0], gx[-1], gy[0], gy[-1]], cmap="inferno", vmin=0, vmax=vmax)
    ax[1].set_title("Levy Spatial Error")
    fig.colorbar(im2, ax=ax[1])
    plt.tight_layout()
    plt.savefig("lennard_jones_spatial_error.png", dpi=200)
    plt.close()

    print("Done. Check lennard_jones_errors.png and lennard_jones_spatial_error.png")
    final_out_d = float(np.mean([h["out_d"][-1] for h in histories]))
    final_out_l = float(np.mean([h["out_l"][-1] for h in histories]))
    print(f"Final outside-domain fraction (Diffusion/Levy): {final_out_d:.3f} / {final_out_l:.3f}")


if __name__ == "__main__":
    main()
