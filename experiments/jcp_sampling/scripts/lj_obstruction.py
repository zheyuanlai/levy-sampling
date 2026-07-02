"""E9 Lennard-Jones obstruction demonstration (mechanism / documented limitation).

A FIXED Cartesian jump bank is incompatible with a cluster's continuous rotational symmetry: a
jump aligned in the catalogue frame is productive only while the cluster keeps that orientation,
and rotational diffusion destroys its productivity. A rotationally-augmented gated bank (rotated
copies of the jump, gated by landing energy) restores productivity. This is a scope-limitation
result, not an equilibrium efficiency claim. Writes fig_lj_obstruction.{png,pdf} and
lj_obstruction.json. Run AFTER make_figures (which clears the figures directory).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.jcp_sampling.core.potentials import LennardJones2D


def rot(theta):
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)


def apply_rot(xflat, R, n):
    return (xflat.reshape(-1, n, 2) @ R.T).reshape(xflat.shape)


def kabsch(P, Q):
    U, _, Vt = torch.linalg.svd(P.T @ Q)
    d = torch.sign(torch.linalg.det(Vt.T @ U.T))
    return U @ torch.diag(torch.tensor([1.0, d])) @ Vt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-atoms", type=int, default=7)
    ap.add_argument("--fig-dir", default="reports/jcp_sampling_report/figures")
    ap.add_argument("--results-root", default="results/jcp_sampling")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    N = args.n_atoms
    lj = LennardJones2D(n_atoms=N)
    rng = np.random.default_rng(args.seed)

    xs = torch.tensor(rng.normal(0, 1.1, size=(600, 2 * N)), dtype=torch.float32)
    xq = lj.remove_com(lj.quench(xs, n_steps=2000, lr=1e-3))
    Eq = lj.potential(xq)
    order = torch.argsort(Eq)
    glob = lj.remove_com(xq[order[0]]).reshape(N, 2)
    E_glob = float(Eq[order[0]])
    dg = lj.descriptor(xq[order[0]])
    target, E_iso = None, None
    for idx in order.tolist():
        if float((lj.descriptor(xq[idx]) - dg).norm()) > 0.3 and float(Eq[idx]) < 0:
            target = lj.remove_com(xq[idx]).reshape(N, 2); E_iso = float(Eq[idx]); break
    v = ((target @ kabsch(target, glob)) - glob).reshape(-1)

    thetas = np.linspace(0, 2 * np.pi, 73)
    phi_grid = np.linspace(0, 2 * np.pi, 361)
    E_fixed, E_gated = [], []
    thresh = E_iso + 1.0
    prod_fixed = prod_gated = 0
    for th in thetas:
        g_th = apply_rot(glob.reshape(-1), rot(th), N)
        e_fix = float(lj.potential((g_th + v).reshape(1, -1)))
        cand = torch.stack([g_th + apply_rot(v, rot(phi), N) for phi in phi_grid])
        e_gat = float(lj.potential(cand).min())
        E_fixed.append(min(e_fix, 50)); E_gated.append(e_gat)
        prod_fixed += (e_fix < thresh); prod_gated += (e_gat < thresh)
    prod_fixed /= len(thetas); prod_gated /= len(thetas)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(np.degrees(thetas), E_fixed, color="#E4572E", label=f"fixed bank ({prod_fixed:.0%} productive)")
    ax.plot(np.degrees(thetas), E_gated, color="#4C78A8", label=f"rotation-augmented gated ({prod_gated:.0%})")
    ax.axhline(E_iso, ls="--", color="k", alpha=0.5, lw=0.8, label=f"target isomer E={E_iso:.2f}")
    ax.set_xlabel("cluster rotation angle (deg)"); ax.set_ylabel("post-jump energy")
    ax.set_ylim(E_iso - 1.5, 12)
    ax.set_title(f"LJ$_{N}$ obstruction: fixed jumps break under rotation")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig_lj_obstruction.{ext}", bbox_inches="tight", dpi=180)
    plt.close(fig)
    out = {"global_energy": E_glob, "isomer_energy": E_iso, "jump_norm": float(v.norm()),
           "productivity_fixed": prod_fixed, "productivity_gated": prod_gated, "n_atoms": N}
    (Path(args.results_root) / "lj_obstruction.json").write_text(json.dumps(out, indent=2))
    print(f"LJ{N}: global={E_glob:.3f} isomer={E_iso:.3f} "
          f"productivity fixed={prod_fixed:.2%} gated={prod_gated:.2%}")


if __name__ == "__main__":
    main()
