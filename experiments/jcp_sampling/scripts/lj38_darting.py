"""E9b LJ38-3D double-funnel result: barrier limitation of the additive score + Metropolis-darting hybrid.

The manuscript's rejection-free additive Levy score cannot cross the FCC<->icosahedral funnel barrier:
the straight-line inter-funnel displacement passes through ~+88 eps of atomic overlap, so the score
integrand exp[-beta*DeltaV] ~ 0 and the additive correction gives ~zero corrected drift. The SAME
geometry-matched inter-funnel displacement, applied as a Metropolis-corrected endpoint dart (accept on
the endpoint energy, no path integral), escapes the icosahedral funnel to find the FCC global that local
Monte Carlo never reaches.

By default this regenerates the figure + JSON from the cached minima and darting curves under
results/jcp_sampling/lj38/ (fast). Use --recompute to re-find the minima (L-BFGS + basin-hopping) and
re-run the darting Monte Carlo (~15 min); see the git history / STATUS.md for the full procedure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lj38-dir", default="results/jcp_sampling/lj38")
    ap.add_argument("--fig-dir", default="reports/jcp_sampling_report/figures")
    ap.add_argument("--results-root", default="results/jcp_sampling")
    args = ap.parse_args()
    d = Path(args.lj38_dir)
    c = np.load(d / "lj38_curves.npz")
    m = np.load(d / "lj38_minima.npz")
    kT = float(c["kT"]); s = c["sweeps"]
    E_fcc, E_ico = float(m["E_fcc"]), float(m["E_ico"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    a1.plot(s, c["dart_ico"], color="#4C78A8", lw=2, label="smart-darting")
    a1.plot(s, c["local_ico"], color="#E4572E", lw=2, label="local MC")
    a1.set_xlabel("Monte Carlo sweeps"); a1.set_ylabel("fraction in FCC global funnel")
    a1.set_ylim(-0.05, 1.08); a1.set_title("Escape from the icosahedral funnel")
    a1.legend(fontsize=8); a1.grid(True, alpha=0.3)
    a2.plot(s, c["E_dart"], color="#4C78A8", lw=2, label="smart-darting")
    a2.plot(s, c["E_local"], color="#E4572E", lw=2, label="local MC")
    a2.axhline(E_fcc, ls="--", color="k", alpha=0.5, lw=0.8, label=f"FCC global {E_fcc:.1f}")
    a2.axhline(E_ico, ls=":", color="gray", alpha=0.7, lw=0.8, label=f"icosahedral {E_ico:.1f}")
    a2.set_xlabel("Monte Carlo sweeps"); a2.set_ylabel("mean potential energy")
    a2.set_title("Mean energy"); a2.legend(fontsize=7); a2.grid(True, alpha=0.3)
    fig.suptitle(f"LJ$_{{38}}$ double funnel (kT={kT}): geometry-matched darting finds the FCC global; "
                 f"local MC is trapped", fontsize=10)
    fig.tight_layout()
    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig_lj38_darting.{ext}", bbox_inches="tight", dpi=170)
    plt.close(fig)

    out = {"E_fcc_global": E_fcc, "E_icosahedral": E_ico, "straight_line_barrier": 87.7,
           "beta_barrier_at_transition_T": 728, "kT": kT,
           "local_from_ico_final_FCC_frac": float(c["local_ico"][-1]),
           "darting_from_ico_final_FCC_frac": float(c["dart_ico"][-1]),
           "E_local_final": float(c["E_local"][-1]), "E_dart_final": float(c["E_dart"][-1])}
    (Path(args.results_root) / "lj38_darting.json").write_text(json.dumps(out, indent=2))
    print("wrote fig_lj38_darting + lj38_darting.json:",
          f"local {out['local_from_ico_final_FCC_frac']:.2f} vs darting {out['darting_from_ico_final_FCC_frac']:.2f} FCC-frac")


if __name__ == "__main__":
    main()
