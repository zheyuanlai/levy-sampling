"""Estimator-agreement appendix figure from PRODUCTION data: exact ShellScore
vs the single-atom RA estimator on E1 + E2 (both are in those experiments'
matrices; 24 seeds), W2 and MMD vs t. CSV-only -- no GPU.

Usage:  python scripts/plot_consistency.py
Writes: figures/double_well/consistency_exact_vs_ra.{png,pdf}
"""
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_JCP = os.path.dirname(_HERE)
sys.path.insert(0, _JCP)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.plotting import apply_style, _series, blend_toward_white, _running_mean

STY = {"LSC-CP": dict(color="#000000", ls="-", label="exact score"),
       "LSC-CP-RA": dict(color="#7030A0", ls=(0, (4, 2)), label="single-atom RA")}
PANELS = [("double_well", "W2", "E1 double well"),
          ("double_well", "MMD", "E1 double well"),
          ("mog40", "W2", "E2 MoG-40"),
          ("mog40", "MMD", "E2 MoG-40")]


def main() -> int:
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 2.8))
    for ax, (exp, met, ttl) in zip(axes, PANELS):
        path = os.path.join(_JCP, "results", exp, "metrics_timeseries.csv")
        rows = list(csv.DictReader(open(path)))
        for m, st in STY.items():
            x, y, sd = _series(rows, m, met)
            y = _running_mean(y, 9)
            sd = _running_mean(sd, 9)
            ax.fill_between(x, y - sd, y + sd,
                            color=blend_toward_white(st["color"]), lw=0)
            ax.plot(x, y, color=st["color"], ls=st["ls"], label=st["label"],
                    lw=1.4)
        ax.set_yscale("log")
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(met)
        ax.set_title(ttl, fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02),
               frameon=False)
    fig.tight_layout()
    out = os.path.join(_JCP, "figures", "double_well", "consistency_exact_vs_ra")
    fig.savefig(out + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print("wrote", out + ".{png,pdf}  (production CSVs, CSV-only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
