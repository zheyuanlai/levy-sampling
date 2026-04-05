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
import lennard_jones_potential as lj

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
    X_flmc = X_diff.copy()
    X_mala = X_diff.copy()
    X_malevy = X_diff.copy()
    alpha = 1.5

    steps = int(T / dt)
    for _ in range(steps):
        X_diff = fw.step_diff(X_diff, dt, eps, gx, gy, bx, by, rng)
        X_levy = fw.step_levy(X_levy, dt, eps, gx, gy, bx, by, Sx, Sy, rng, lam, sigma_L, mults, pm)
        X_flmc = fw.step_flmc_2d(
            X_flmc,
            dt,
            alpha,
            eps,
            lambda x, y: fw.V_fourwell_flmc(x, y, a=a),
            lambda x, y: fw.gradV_fourwell_flmc(x, y, a=a),
            rng,
        )
        X_mala, _ = fw.step_mala(X_mala, dt, eps, a, rng)
        X_malevy, _ = fw.step_malevy(X_malevy, dt, eps, a, rng, lam, sigma_L, mults, pm)

    dens_d = fw.density_on_grid(X_diff, gx, gy)
    dens_l = fw.density_on_grid(X_levy, gx, gy)
    dens_f = fw.density_on_grid(X_flmc, gx, gy)
    dens_m = fw.density_on_grid(X_mala, gx, gy)
    dens_ml = fw.density_on_grid(X_malevy, gx, gy)
    return gx, gy, pi, dens_d, dens_l, dens_f, dens_m, dens_ml

def simulate_mueller(seed=42):
    eps, dt, T, N = 0.7, 2e-4, 2.0, 3000
    gx, gy = np.linspace(-1.8, 1.8, 200), np.linspace(-1.2, 2.2, 200)
    lam, sigma_L, mults, pm = 8.0, 1.2, [0.8, 1.2], [0.5, 0.5]
    rng = np.random.default_rng(seed)

    pi, bx, by, Sx, Sy = mu.precompute_pi_b_S(eps, gx, gy, lam, sigma_L, mults, pm)
    X_diff = np.zeros((N, 2)) + np.array([1.0, 0.0]) + rng.standard_normal((N, 2)) * 0.1
    X_levy = X_diff.copy()
    X_flmc = X_diff.copy()
    X_mala = X_diff.copy()
    X_malevy = X_diff.copy()
    alpha = 1.5

    steps = int(T / dt)
    for _ in range(steps):
        X_diff = mu.step_diff(X_diff, dt, eps, gx, gy, bx, by, rng)
        X_levy = mu.step_levy(X_levy, dt, eps, gx, gy, bx, by, Sx, Sy, rng, lam, sigma_L, mults, pm)
        X_flmc = mu.step_flmc_2d(X_flmc, dt, alpha, eps, mu.V_mueller_flmc, mu.gradV_mueller_flmc, rng)
        X_mala, _ = mu.step_mala(X_mala, dt, eps, rng)
        X_malevy, _ = mu.step_malevy(X_malevy, dt, eps, rng, lam, sigma_L, mults, pm)

    dens_d = mu.density_on_grid(X_diff, gx, gy)
    dens_l = mu.density_on_grid(X_levy, gx, gy)
    dens_f = mu.density_on_grid(X_flmc, gx, gy)
    dens_m = mu.density_on_grid(X_mala, gx, gy)
    dens_ml = mu.density_on_grid(X_malevy, gx, gy)
    return gx, gy, pi, dens_d, dens_l, dens_f, dens_m, dens_ml

def simulate_ring(seed=42):
    eps, dt, T, N = 0.35, 0.0015, 40.0, 5000
    gx, gy = np.linspace(-2.2, 2.2, 240), np.linspace(-2.2, 2.2, 240)
    lam, sigma_L = 1.6, 1.25
    mults, pm = [1.0, 1.7, 2.4], [0.70, 0.22, 0.08]
    rng = np.random.default_rng(seed)

    pi, bx, by, Sx, Sy = rg.precompute_pi_drift_score_on_grid(eps, gx, gy, lam, sigma_L, mults, pm)
    X_diff = np.array([-1.0, 0.0]) + 0.05 * rng.standard_normal((N, 2))
    X_levy = X_diff.copy()
    X_flmc = X_diff.copy()
    X_mala = X_diff.copy()
    X_malevy = X_diff.copy()
    alpha = 1.5

    steps = int(T / dt)
    for _ in range(steps):
        X_diff = rg.step_diff(X_diff, dt, eps, gx, gy, bx, by, rng)
        X_levy = rg.step_levy(X_levy, dt, eps, gx, gy, bx, by, Sx, Sy, rng, lam, sigma_L, mults, pm)
        X_flmc = rg.step_flmc_2d(X_flmc, dt, alpha, eps, rg.V_ring, rg.gradV_ring, rng)
        X_mala, _ = rg.step_mala(X_mala, dt, eps, rng)
        X_malevy, _ = rg.step_malevy(X_malevy, dt, eps, rng, lam, sigma_L, mults, pm)

    dens_d = rg.density_on_grid(X_diff, gx, gy)
    dens_l = rg.density_on_grid(X_levy, gx, gy)
    dens_f = rg.density_on_grid(X_flmc, gx, gy)
    dens_m = rg.density_on_grid(X_mala, gx, gy)
    dens_ml = rg.density_on_grid(X_malevy, gx, gy)
    return gx, gy, pi, dens_d, dens_l, dens_f, dens_m, dens_ml


def simulate_lennard(seed=42):
    lj_eps, sigma, r_soft = 1.0, 1.0, 0.20
    noise_eps, dt, T, N = 0.50, 0.002, 20.0, 5000
    gx, gy = np.linspace(-3.2, 3.2, 220), np.linspace(-3.2, 3.2, 220)
    lam, sigma_L, mults, pm = 1.0, 0.65, [1.0, 1.6, 2.2], [0.78, 0.18, 0.04]
    num_dirs = 16
    jump_mu = 0.0
    jump_kappa = 0.0
    jump_cap = 2.0
    rng = np.random.default_rng(seed)

    pi, bx, by, Sx, Sy = lj.precompute_pi_b_S(
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

    X_diff = np.array([2.5 * sigma, 0.0]) + 0.08 * rng.standard_normal((N, 2))
    X_levy = X_diff.copy()

    steps = int(T / dt)
    for _ in range(steps):
        X_diff = lj.step_diff(X_diff, dt, noise_eps, gx, gy, bx, by, rng)
        X_levy = lj.step_levy(
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

    dens_d = lj.density_on_grid(X_diff, gx, gy)
    dens_l = lj.density_on_grid(X_levy, gx, gy)
    return gx, gy, pi, dens_d, dens_l

def generate_all(out_dir):
    set_academic_style()
    os.makedirs(out_dir, exist_ok=True)

    # Four wells
    gx, gy, pi, dens_d, dens_l, dens_f, dens_m, dens_ml = simulate_fourwells()
    err_d = np.abs(dens_d - pi)
    err_l = np.abs(dens_l - pi)
    err_f = np.abs(dens_f - pi)
    err_m = np.abs(dens_m - pi)
    err_ml = np.abs(dens_ml - pi)
    norm = _shared_err_norm([err_d, err_l, err_f, err_m, err_ml], gamma=0.8)
    save_error_image(err_d, gx, gy, "Four-Well: |Diffusion − True|", os.path.join(out_dir, "fourwell_abs_err_diffusion.png"), norm)
    save_error_image(err_l, gx, gy, "Four-Well: |LSB-MC − True|", os.path.join(out_dir, "fourwell_abs_err_levy.png"), norm)
    save_error_image(err_f, gx, gy, "Four-Well: |FLMC − True|", os.path.join(out_dir, "fourwell_abs_err_flmc.png"), norm)
    save_error_image(err_m, gx, gy, "Four-Well: |MALA − True|", os.path.join(out_dir, "fourwell_abs_err_mala.png"), norm)
    save_error_image(err_ml, gx, gy, "Four-Well: |MALA-Levy − True|", os.path.join(out_dir, "fourwell_abs_err_malevy.png"), norm)

    # Mueller
    gx, gy, pi, dens_d, dens_l, dens_f, dens_m, dens_ml = simulate_mueller()
    err_d = np.abs(dens_d - pi)
    err_l = np.abs(dens_l - pi)
    err_f = np.abs(dens_f - pi)
    err_m = np.abs(dens_m - pi)
    err_ml = np.abs(dens_ml - pi)
    norm = _shared_err_norm([err_d, err_l, err_f, err_m, err_ml], gamma=0.8)
    save_error_image(err_d, gx, gy, "Müller-Brown: |Diffusion − True|", os.path.join(out_dir, "mueller_abs_err_diffusion.png"), norm)
    save_error_image(err_l, gx, gy, "Müller-Brown: |LSB-MC − True|", os.path.join(out_dir, "mueller_abs_err_levy.png"), norm)
    save_error_image(err_f, gx, gy, "Müller-Brown: |FLMC − True|", os.path.join(out_dir, "mueller_abs_err_flmc.png"), norm)
    save_error_image(err_m, gx, gy, "Müller-Brown: |MALA − True|", os.path.join(out_dir, "mueller_abs_err_mala.png"), norm)
    save_error_image(err_ml, gx, gy, "Müller-Brown: |MALA-Levy − True|", os.path.join(out_dir, "mueller_abs_err_malevy.png"), norm)

    # Ring
    gx, gy, pi, dens_d, dens_l, dens_f, dens_m, dens_ml = simulate_ring()
    err_d = np.abs(dens_d - pi)
    err_l = np.abs(dens_l - pi)
    err_f = np.abs(dens_f - pi)
    err_m = np.abs(dens_m - pi)
    err_ml = np.abs(dens_ml - pi)
    norm = _shared_err_norm([err_d, err_l, err_f, err_m, err_ml], gamma=0.8)
    save_error_image(err_d, gx, gy, "Ring: |Diffusion − True|", os.path.join(out_dir, "ring_abs_err_diffusion.png"), norm)
    save_error_image(err_l, gx, gy, "Ring: |LSB-MC − True|", os.path.join(out_dir, "ring_abs_err_levy.png"), norm)
    save_error_image(err_f, gx, gy, "Ring: |FLMC − True|", os.path.join(out_dir, "ring_abs_err_flmc.png"), norm)
    save_error_image(err_m, gx, gy, "Ring: |MALA − True|", os.path.join(out_dir, "ring_abs_err_mala.png"), norm)
    save_error_image(err_ml, gx, gy, "Ring: |MALA-Levy − True|", os.path.join(out_dir, "ring_abs_err_malevy.png"), norm)

    # Lennard-Jones
    gx, gy, pi, dens_d, dens_l = simulate_lennard()
    err_d = np.abs(dens_d - pi)
    err_l = np.abs(dens_l - pi)
    norm = _shared_err_norm([err_d, err_l], gamma=0.8)
    save_error_image(err_d, gx, gy, "Lennard-Jones: |Diffusion − True|", os.path.join(out_dir, "lennard_jones_abs_err_diffusion.png"), norm)
    save_error_image(err_l, gx, gy, "Lennard-Jones: |Lévy − True|", os.path.join(out_dir, "lennard_jones_abs_err_levy.png"), norm)

if __name__ == "__main__":
    output_dir = os.path.join(THIS_DIR, "abs_error")
    generate_all(output_dir)
    print(f"Saved absolute error plots to: {output_dir}")
