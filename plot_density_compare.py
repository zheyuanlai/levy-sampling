#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Allow imports from mae_l2 directory
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

import fourwells as fw
import mueller as mu
import ring as rg

def set_academic_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.linewidth": 1.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

def save_density_image(dens, gx, gy, title, fname, norm, init_point=None):
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    extent = [gx[0], gx[-1], gy[0], gy[-1]]
    im = ax.imshow(
        dens,
        origin="lower",
        extent=extent,
        cmap="viridis",
        norm=norm,
        aspect="equal",
    )
    if init_point is not None:
        ax.scatter(
            init_point[0],
            init_point[1],
            marker="*",
            s=120,
            c="red",
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
        )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Density")
    plt.savefig(fname)
    plt.close()

def simulate_fourwells(seed=42):
    a, eps, dt, T, N = 1.0, 0.5, 0.005, 15.0, 5000
    gx, gy = np.linspace(-1.8, 1.8, 200), np.linspace(-1.8, 1.8, 200)
    lam, sigma_L, mults, pm = 1.2, 1.0, [1.0, 2.0], [0.85, 0.15]
    rng = np.random.default_rng(seed)

    pi, bx, by, Sx, Sy = fw.precompute_pi_b_S(eps, gx, gy, a, lam, sigma_L, mults, pm)

    init = np.array([a, a])
    X_diff = init + 0.1 * rng.standard_normal((N, 2))
    X_levy = X_diff.copy()
    X_mala = X_diff.copy()
    X_malevy = X_diff.copy()

    steps = int(T / dt)
    for _ in range(steps):
        X_diff = fw.step_diff(X_diff, dt, eps, gx, gy, bx, by, rng)
        X_levy = fw.step_levy(X_levy, dt, eps, gx, gy, bx, by, Sx, Sy, rng, lam, sigma_L, mults, pm)
        X_mala, _ = fw.step_mala(X_mala, dt, eps, a, rng)
        X_malevy, _ = fw.step_malevy(X_malevy, dt, eps, a, rng, lam, sigma_L, mults, pm)

    dens_d = fw.density_on_grid(X_diff, gx, gy)
    dens_l = fw.density_on_grid(X_levy, gx, gy)
    dens_m = fw.density_on_grid(X_mala, gx, gy)
    dens_ml = fw.density_on_grid(X_malevy, gx, gy)
    return gx, gy, pi, dens_d, dens_l, dens_m, dens_ml, init

def simulate_mueller(seed=42):
    eps, dt, T, N = 0.7, 2e-4, 2.0, 3000
    gx, gy = np.linspace(-1.8, 1.8, 200), np.linspace(-1.2, 2.2, 200)
    lam, sigma_L, mults, pm = 8.0, 1.2, [0.8, 1.2], [0.5, 0.5]
    rng = np.random.default_rng(seed)

    pi, bx, by, Sx, Sy = mu.precompute_pi_b_S(eps, gx, gy, lam, sigma_L, mults, pm)

    init = np.array([1.0, 0.0])
    X_diff = np.zeros((N, 2)) + init + rng.standard_normal((N, 2)) * 0.1
    X_levy = X_diff.copy()
    X_mala = X_diff.copy()
    X_malevy = X_diff.copy()

    steps = int(T / dt)
    for _ in range(steps):
        X_diff = mu.step_diff(X_diff, dt, eps, gx, gy, bx, by, rng)
        X_levy = mu.step_levy(X_levy, dt, eps, gx, gy, bx, by, Sx, Sy, rng, lam, sigma_L, mults, pm)
        X_mala, _ = mu.step_mala(X_mala, dt, eps, rng)
        X_malevy, _ = mu.step_malevy(X_malevy, dt, eps, rng, lam, sigma_L, mults, pm)

    dens_d = mu.density_on_grid(X_diff, gx, gy)
    dens_l = mu.density_on_grid(X_levy, gx, gy)
    dens_m = mu.density_on_grid(X_mala, gx, gy)
    dens_ml = mu.density_on_grid(X_malevy, gx, gy)
    return gx, gy, pi, dens_d, dens_l, dens_m, dens_ml, init

def simulate_ring(seed=42):
    eps, dt, T, N = 0.35, 0.0015, 40.0, 5000
    gx, gy = np.linspace(-2.2, 2.2, 240), np.linspace(-2.2, 2.2, 240)
    lam, sigma_L = 1.6, 1.25
    mults, pm = [1.0, 1.7, 2.4], [0.70, 0.22, 0.08]
    rng = np.random.default_rng(seed)

    pi, bx, by, Sx, Sy = rg.precompute_pi_drift_score_on_grid(eps, gx, gy, lam, sigma_L, mults, pm)

    init = np.array([-1.0, 0.0])
    X_diff = init + 0.05 * rng.standard_normal((N, 2))
    X_levy = X_diff.copy()
    X_mala = X_diff.copy()
    X_malevy = X_diff.copy()

    steps = int(T / dt)
    for _ in range(steps):
        X_diff = rg.step_diff(X_diff, dt, eps, gx, gy, bx, by, rng)
        X_levy = rg.step_levy(X_levy, dt, eps, gx, gy, bx, by, Sx, Sy, rng, lam, sigma_L, mults, pm)
        X_mala, _ = rg.step_mala(X_mala, dt, eps, rng)
        X_malevy, _ = rg.step_malevy(X_malevy, dt, eps, rng, lam, sigma_L, mults, pm)

    dens_d = rg.density_on_grid(X_diff, gx, gy)
    dens_l = rg.density_on_grid(X_levy, gx, gy)
    dens_m = rg.density_on_grid(X_mala, gx, gy)
    dens_ml = rg.density_on_grid(X_malevy, gx, gy)
    return gx, gy, pi, dens_d, dens_l, dens_m, dens_ml, init

def _shared_norm(arrs, use_log=True, gamma=0.7):
    vmax = max(a.max() for a in arrs)
    if use_log:
        min_pos = min(a[a > 0].min() for a in arrs)
        vmin = max(min_pos, vmax * 1e-6)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin = 0.0
        norm = mcolors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
    return vmin, vmax, norm

def generate_all(out_dir, use_log=False):
    set_academic_style()

    os.makedirs(out_dir, exist_ok=True)

    # Four wells
    gx, gy, pi, dens_d, dens_l, dens_m, dens_ml, init = simulate_fourwells()
    _, _, norm = _shared_norm([pi, dens_d, dens_l, dens_m, dens_ml], use_log=use_log, gamma=0.5)
    save_density_image(pi, gx, gy, "Four-Well: True Invariant Density", os.path.join(out_dir, "fourwell_true_density.png"), norm)
    save_density_image(dens_d, gx, gy, "Four-Well: Diffusion Density", os.path.join(out_dir, "fourwell_diffusion_density.png"), norm, init_point=init)
    save_density_image(dens_l, gx, gy, "Four-Well: Lévy Density", os.path.join(out_dir, "fourwell_levy_density.png"), norm, init_point=init)
    save_density_image(dens_m, gx, gy, "Four-Well: MALA Density", os.path.join(out_dir, "fourwell_mala_density.png"), norm, init_point=init)
    save_density_image(dens_ml, gx, gy, "Four-Well: MALA-Levy Density", os.path.join(out_dir, "fourwell_malevy_density.png"), norm, init_point=init)

    # Mueller
    gx, gy, pi, dens_d, dens_l, dens_m, dens_ml, init = simulate_mueller()
    _, _, norm = _shared_norm([pi, dens_d, dens_l, dens_m, dens_ml], use_log=use_log, gamma=0.5)
    save_density_image(pi, gx, gy, "Müller-Brown: True Invariant Density", os.path.join(out_dir, "mueller_true_density.png"), norm)
    save_density_image(dens_d, gx, gy, "Müller-Brown: Diffusion Density", os.path.join(out_dir, "mueller_diffusion_density.png"), norm, init_point=init)
    save_density_image(dens_l, gx, gy, "Müller-Brown: Lévy Density", os.path.join(out_dir, "mueller_levy_density.png"), norm, init_point=init)
    save_density_image(dens_m, gx, gy, "Müller-Brown: MALA Density", os.path.join(out_dir, "mueller_mala_density.png"), norm, init_point=init)
    save_density_image(dens_ml, gx, gy, "Müller-Brown: MALA-Levy Density", os.path.join(out_dir, "mueller_malevy_density.png"), norm, init_point=init)

    # Ring
    gx, gy, pi, dens_d, dens_l, dens_m, dens_ml, init = simulate_ring()
    _, _, norm = _shared_norm([pi, dens_d, dens_l, dens_m, dens_ml], use_log=use_log, gamma=0.5)
    save_density_image(pi, gx, gy, "Ring: True Invariant Density", os.path.join(out_dir, "ring_true_density.png"), norm)
    save_density_image(dens_d, gx, gy, "Ring: Diffusion Density", os.path.join(out_dir, "ring_diffusion_density.png"), norm, init_point=init)
    save_density_image(dens_l, gx, gy, "Ring: Lévy Density", os.path.join(out_dir, "ring_levy_density.png"), norm, init_point=init)
    save_density_image(dens_m, gx, gy, "Ring: MALA Density", os.path.join(out_dir, "ring_mala_density.png"), norm, init_point=init)
    save_density_image(dens_ml, gx, gy, "Ring: MALA-Levy Density", os.path.join(out_dir, "ring_malevy_density.png"), norm, init_point=init)

if __name__ == "__main__":
    output_dir = os.path.join(THIS_DIR, "density_compare")
    generate_all(output_dir, use_log=False)
    print(f"Saved density comparisons to: {output_dir}")
