#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

def save_error_image(err, gx, gy, title, fname, norm):
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    extent = [gx[0], gx[-1], gy[0], gy[-1]]
    im = ax.imshow(
        err,
        origin="lower",
        extent=extent,
        cmap="inferno",
        norm=norm,
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Absolute Error")
    plt.savefig(fname)
    plt.close()

def _shared_err_norm(errs, gamma=0.8):
    vmax = max(e.max() for e in errs)
    if vmax <= 0:
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    else:
        norm = mcolors.PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax)
    return norm

def simulate_fourwells(seed=42):
    a, eps, dt, T, N = 1.0, 0.5, 0.005, 15.0, 5000
    gx, gy = np.linspace(-1.8, 1.8, 200), np.linspace(-1.8, 1.8, 200)
    lam, sigma_L, mults, pm = 1.2, 1.0, [1.0, 2.0], [0.85, 0.15]
    rng = np.random.default_rng(seed)

    pi, bx, by, Sx, Sy = fw.precompute_pi_b_S(eps, gx, gy, a, lam, sigma_L, mults, pm)
    X_diff = np.array([a, a]) + 0.1 * rng.standard_normal((N, 2))
    X_levy = X_diff.copy()

    steps = int(T / dt)
    for _ in range(steps):
        X_diff = fw.step_diff(X_diff, dt, eps, gx, gy, bx, by, rng)
        X_levy = fw.step_levy(X_levy, dt, eps, gx, gy, bx, by, Sx, Sy, rng, lam, sigma_L, mults, pm)

    dens_d = fw.density_on_grid(X_diff, gx, gy)
    dens_l = fw.density_on_grid(X_levy, gx, gy)
    return gx, gy, pi, dens_d, dens_l

def simulate_mueller(seed=42):
    eps, dt, T, N = 0.7, 2e-4, 2.0, 3000
    gx, gy = np.linspace(-1.8, 1.8, 200), np.linspace(-1.2, 2.2, 200)
    lam, sigma_L, mults, pm = 8.0, 1.2, [0.8, 1.2], [0.5, 0.5]
    rng = np.random.default_rng(seed)

    pi, bx, by, Sx, Sy = mu.precompute_pi_b_S(eps, gx, gy, lam, sigma_L, mults, pm)
    X_diff = np.zeros((N, 2)) + np.array([1.0, 0.0]) + rng.standard_normal((N, 2)) * 0.1
    X_levy = X_diff.copy()

    steps = int(T / dt)
    for _ in range(steps):
        X_diff = mu.step_diff(X_diff, dt, eps, gx, gy, bx, by, rng)
        X_levy = mu.step_levy(X_levy, dt, eps, gx, gy, bx, by, Sx, Sy, rng, lam, sigma_L, mults, pm)

    dens_d = mu.density_on_grid(X_diff, gx, gy)
    dens_l = mu.density_on_grid(X_levy, gx, gy)
    return gx, gy, pi, dens_d, dens_l

def simulate_ring(seed=42):
    eps, dt, T, N = 0.35, 0.0015, 40.0, 5000
    gx, gy = np.linspace(-2.2, 2.2, 240), np.linspace(-2.2, 2.2, 240)
    lam, sigma_L = 1.6, 1.25
    mults, pm = [1.0, 1.7, 2.4], [0.70, 0.22, 0.08]
    rng = np.random.default_rng(seed)

    pi, bx, by, Sx, Sy = rg.precompute_pi_drift_score_on_grid(eps, gx, gy, lam, sigma_L, mults, pm)
    X_diff = np.array([-1.0, 0.0]) + 0.05 * rng.standard_normal((N, 2))
    X_levy = X_diff.copy()

    steps = int(T / dt)
    for _ in range(steps):
        X_diff = rg.step_diff(X_diff, dt, eps, gx, gy, bx, by, rng)
        X_levy = rg.step_levy(X_levy, dt, eps, gx, gy, bx, by, Sx, Sy, rng, lam, sigma_L, mults, pm)

    dens_d = rg.density_on_grid(X_diff, gx, gy)
    dens_l = rg.density_on_grid(X_levy, gx, gy)
    return gx, gy, pi, dens_d, dens_l

def generate_all(out_dir):
    set_academic_style()
    os.makedirs(out_dir, exist_ok=True)

    # Four wells
    gx, gy, pi, dens_d, dens_l = simulate_fourwells()
    err_d = np.abs(dens_d - pi)
    err_l = np.abs(dens_l - pi)
    norm = _shared_err_norm([err_d, err_l], gamma=0.8)
    save_error_image(err_d, gx, gy, "Four-Well: |Diffusion − True|", os.path.join(out_dir, "fourwell_abs_err_diffusion.png"), norm)
    save_error_image(err_l, gx, gy, "Four-Well: |Lévy − True|", os.path.join(out_dir, "fourwell_abs_err_levy.png"), norm)

    # Mueller
    gx, gy, pi, dens_d, dens_l = simulate_mueller()
    err_d = np.abs(dens_d - pi)
    err_l = np.abs(dens_l - pi)
    norm = _shared_err_norm([err_d, err_l], gamma=0.8)
    save_error_image(err_d, gx, gy, "Müller-Brown: |Diffusion − True|", os.path.join(out_dir, "mueller_abs_err_diffusion.png"), norm)
    save_error_image(err_l, gx, gy, "Müller-Brown: |Lévy − True|", os.path.join(out_dir, "mueller_abs_err_levy.png"), norm)

    # Ring
    gx, gy, pi, dens_d, dens_l = simulate_ring()
    err_d = np.abs(dens_d - pi)
    err_l = np.abs(dens_l - pi)
    norm = _shared_err_norm([err_d, err_l], gamma=0.8)
    save_error_image(err_d, gx, gy, "Ring: |Diffusion − True|", os.path.join(out_dir, "ring_abs_err_diffusion.png"), norm)
    save_error_image(err_l, gx, gy, "Ring: |Lévy − True|", os.path.join(out_dir, "ring_abs_err_levy.png"), norm)

if __name__ == "__main__":
    output_dir = os.path.join(THIS_DIR, "abs_error")
    generate_all(output_dir)
    print(f"Saved absolute error plots to: {output_dir}")
