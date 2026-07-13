"""Regenerate all figures for an experiment from its metrics_timeseries.csv,
using the fixed + stationary-running-average plotting. No re-sampling needed --
the fixes are plotting-only. Recomputes bias floors (for the dashed floor line)
by rebuilding the experiment's make_metrics.

Usage:  JCP_EXTRA_GPUS=0 JCP_GPU=0 python scripts/replot_figures.py <experiment>
        <experiment> in {double_well, mog40, mb3well_10d, coupled_phi4}
"""
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_JCP = os.path.dirname(_HERE)
sys.path.insert(0, _JCP)
from src.gpu_guard import select_gpu
select_gpu(os.environ.get("JCP_GPU", "4"))
import torch
torch.set_default_dtype(torch.float64)

from src.experiments import (build_e1, build_e2, build_e3, build_e4, make_metrics)
from src.plotting import metric_single, metric_grid

_BUILD = {"double_well": build_e1, "mog40": build_e2,
          "mb3well_10d": build_e3, "coupled_phi4": build_e4}
_SINGLE = ("W2", "TV", "TV_density", "MMD", "e_F", "basin_rel_max", "KSD",
           "W1_cdf", "CDF_sup", "pdf_L1", "KDE_chi2", "W2_10d")
DEV = "cuda"


def main() -> int:
    name = sys.argv[1]
    csv_path = os.path.join(_JCP, "results", name, "metrics_timeseries.csv")
    if not os.path.exists(csv_path):
        print(f"no CSV for {name} at {csv_path}; skip", file=sys.stderr)
        return 1
    rows = list(csv.DictReader(open(csv_path)))

    kw = {}
    if name in ("mb3well_10d", "coupled_phi4"):
        cache = os.path.join(_JCP, "results", name, "basin_map.npz")
        if os.path.exists(cache):
            kw["basin_cache"] = cache
    exp = _BUILD[name](device=DEV, **kw)
    _, floors, _ = make_metrics(exp, exp.cfg.n_particles, device=DEV)
    emc = exp.emc_target

    figdir = os.path.join(_JCP, "figures", name)
    present = set().union(*[set(r) for r in rows])
    in_data = {r["method"] for r in rows}
    # single raw-CP baseline: full-law CP if present, else the atomic CP-RA
    # relabelled "Raw-CP" (raw CP uses no Levy score, so there is no "RA" raw-CP;
    # CP and CP-RA differ only in negligible >=2-jumps-per-step events).
    raw = "CP" if "CP" in in_data else "CP-RA"
    label_overrides = {} if raw == "CP" else {"CP-RA": "Raw-CP"}
    order = ["ULA", "MALA", "FLA", "BAOAB", "PT", raw, "LSC-CP", "LSC-CP-RA"]
    methods = [m for m in order if m in in_data]
    single = [m for m in _SINGLE if m in present]
    for m in single:
        for axis in ("t", "nfe", "wallclock"):
            metric_single(rows, m, os.path.join(figdir, f"{name}_{m}_{axis}"),
                          xaxis=axis, floors=floors, methods=methods,
                          emc_target=emc, show=False, label_overrides=label_overrides)
    metric_grid(rows, os.path.join(figdir, f"{name}_metrics"),
                metrics=("W2", "MMD", "EMC"), floors=floors, emc_target=emc,
                methods=methods, show=False, label_overrides=label_overrides)
    print(f"replotted {name}: {len(single)} metrics x 3 axes + grid  "
          f"(smoothed, floors from rebuilt make_metrics)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
