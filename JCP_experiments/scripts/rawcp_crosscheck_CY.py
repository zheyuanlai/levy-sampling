"""P0 addendum: reconcile our raw-CP forensic with the collaborator's generator.

The collaborator (`experiments_CY/common/levy/doublewell_definitive.py`) is a
DISCRETIZED-GENERATOR / master-equation method on a GIBBS-ADAPTIVE grid (cells
equi-probable under the target). That grid is ideal for the methods they
propagate (`Langevin`, `LSC-CP`, both targeting Gibbs), but it CANNOT represent
the raw-CP biased stationary law: raw CP injects O(10%) mass into the
low-Gibbs-density tails / barrier (beyond ~+-1.3 at beta=16), where the
equi-probable-cell construction places ZERO cells at any resolution. Refining
n_cells does NOT help -- W1(CY raw-CP, true) plateaus at ~0.10.

Our uniform-grid forensic (`src/stationary.py`) and our particle SDE resolve the
true biased law correctly. This script overlays the two and shows the plateau.

Requires the sibling checkout `experiments_CY/` next to this repo.
Usage:  python scripts/rawcp_crosscheck_CY.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_JCP = os.path.dirname(_HERE)
_CY = os.path.join(os.path.dirname(_JCP), "experiments_CY", "common")
if not os.path.isdir(_CY):
    sys.exit(f"experiments_CY not found at {_CY}; skipping cross-check")
sys.path.insert(0, _CY)
sys.path.insert(0, _JCP)

from levy.doublewell_definitive import (          # collaborator's generator code
    _target_context, _local_reversible_generator, _jump_generator,
    _jump_atoms, config_for_profile)
from src.stationary import (rawcp_stationary_density, gibbs_density,
                            w1_between_cdfs)

EPS = 0.0625                                       # diffusion coeff; beta = 1/eps
cfg = config_for_profile("smoke")                  # center +-2, half-width 0.22, lam 1
LO, HI = cfg.domain_left, cfg.domain_right
V = lambda x: 0.25 * x**4 - 0.5 * x**2            # = (1/4)(x^2-1)^2 - 1/4
dV = lambda x: x**3 - x
atoms, wts = _jump_atoms(cfg)


def cy_rawcp_cdf(n_cells):
    ctx = _target_context(EPS, n_cells, cfg)
    Q = _local_reversible_generator(ctx) + _jump_generator(ctx, cfg)  # raw-CP CTMC
    w, Vv = np.linalg.eig(Q.T)
    pi = np.abs(np.real(Vv[:, int(np.argmin(np.abs(w)))]))
    pi /= pi.sum()
    edges = np.asarray(ctx["cell_edges"])
    n_tail = int(np.sum(np.abs(np.asarray(ctx["cell_centers"])) > 1.5))
    return edges[1:], np.cumsum(pi), n_tail


xg, _, cdf_ours = rawcp_stationary_density(dV, 1.0 / EPS, cfg.jump_intensity,
                                           atoms, wts, LO, HI, 4001)
_, _, cdf_gibbs = gibbs_density(V, 1.0 / EPS, LO, HI, 4001)

print("W1(collaborator raw-CP, our fine-grid prediction) vs n_cells:")
for nc in (80, 160, 320, 640, 1280):
    ex, cdf_cy, n_tail = cy_rawcp_cdf(nc)
    w1 = w1_between_cdfs(xg, np.interp(xg, ex, cdf_cy, left=0, right=1), cdf_ours)
    print(f"  n_cells={nc:4d}  tail cells(|x|>1.5)={n_tail:3d}  W1={w1:.4f}")
print(f"W1(our raw-CP, Gibbs) = {w1_between_cdfs(xg, cdf_ours, cdf_gibbs):.4f}  (the true bias)")

# figure at the production n_cells
ex, cdf_cy, _ = cy_rawcp_cdf(80)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5.0, 3.6))
ax.plot(xg, cdf_gibbs, color="#888", lw=2.4, label=r"Gibbs $\pi$")
ax.plot(xg, cdf_ours, color="#D55E00", lw=1.8, ls="--",
        label="raw-CP true (ours, uniform grid)")
ax.plot(ex, cdf_cy, color="#000", lw=1.1, marker="o", ms=3, markevery=4,
        markerfacecolor="w", label="raw-CP on Gibbs-adaptive grid (CY)")
ax.set_xlim(-3, 3); ax.set_xlabel("x"); ax.set_ylabel("CDF")
ax.set_title("raw-CP: Gibbs-adaptive grid truncates the tail bias mass", fontsize=8)
ax.legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout()
figdir = os.path.join(_JCP, "figures", "double_well")
os.makedirs(figdir, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(figdir, f"rawcp_crosscheck_CY.{ext}"), dpi=200,
                bbox_inches="tight")
print("saved:", os.path.join(figdir, "rawcp_crosscheck_CY.{png,pdf}"))
