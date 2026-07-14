"""Regenerate all figures for an experiment from results/<experiment>/ ALONE:
metrics_timeseries.csv (data) + manifest.json (bias_floors, emc_target, plot
policy). No GPU, no experiment rebuild -- anyone with the repo's results/
directory can reproduce every figure.

Usage:  python scripts/replot_figures.py <experiment>
        <experiment> in {double_well, mog40, mb3well_10d, coupled_phi4}

Fallback: for old result dirs whose manifest.json is missing or predates the
emc_target/plot fields, the experiment is rebuilt on GPU to recover the floors
(set JCP_GPU; this needs the jcp-exp env + a free GPU).
"""
import csv
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_JCP = os.path.dirname(_HERE)
sys.path.insert(0, _JCP)

_SINGLE = ("W2", "TV", "TV_density", "MMD", "e_F", "basin_rel_max", "KSD",
           "W1_cdf", "CDF_sup", "pdf_L1", "KDE_chi2", "W2_10d")


def _plot_policy(in_data: set) -> tuple[list, dict]:
    """Derive the plotted-method policy from the methods present in the CSV
    (mirrors the notebook figures cell): one raw-CP line, one LSC-CP line
    (the practical estimator: MA > single-atom RA > exact), simple labels."""
    lsc = next((m for m in ("LSC-CP-MA", "LSC-CP-RA", "LSC-CP")
                if m in in_data), None)
    raw = "CP" if "CP" in in_data else ("CP-RA" if "CP-RA" in in_data else None)
    methods = [m for m in ("ULA", "MALA", "FLA", "BAOAB", "PT") if m in in_data]
    lov = {}
    if raw:
        methods.append(raw)
        lov[raw] = "Raw-CP"
    if lsc:
        methods.append(lsc)
        lov[lsc] = "LSC-CP"
    return methods, lov


def _floors_from_gpu(name: str):
    """Legacy fallback: rebuild the experiment to recover floors/emc_target
    (old result dirs without a full manifest)."""
    from src.gpu_guard import select_gpu
    select_gpu(os.environ.get("JCP_GPU", "4"))
    import torch
    torch.set_default_dtype(torch.float64)
    from src.experiments import (build_e1, build_e2, build_e3, build_e4,
                                 make_metrics)
    build = {"double_well": build_e1, "mog40": build_e2,
             "mb3well_10d": build_e3, "coupled_phi4": build_e4}[name]
    kw = {}
    cache = os.path.join(_JCP, "results", name, "basin_map.npz")
    if name in ("mb3well_10d", "coupled_phi4") and os.path.exists(cache):
        kw["basin_cache"] = cache
    exp = build(device="cuda", **kw)
    _, floors, _ = make_metrics(exp, exp.cfg.n_particles, device="cuda")
    return floors, exp.emc_target


def main() -> int:
    name = sys.argv[1]
    res = os.path.join(_JCP, "results", name)
    csv_path = os.path.join(res, "metrics_timeseries.csv")
    if not os.path.exists(csv_path):
        print(f"no CSV for {name} at {csv_path}; skip", file=sys.stderr)
        return 1
    rows = list(csv.DictReader(open(csv_path)))

    man_path = os.path.join(res, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}
    floors = man.get("bias_floors")
    emc = man.get("emc_target")
    if floors is None or emc is None:
        print("manifest.json missing/incomplete -> GPU fallback for floors")
        floors, emc = _floors_from_gpu(name)

    in_data = {r["method"] for r in rows}
    plot = man.get("plot") or {}
    methods = [m for m in plot.get("methods", []) if m in in_data]
    lov = plot.get("label_overrides") or {}
    if not methods:
        methods, lov = _plot_policy(in_data)

    from src.plotting import metric_single, metric_grid  # matplotlib only
    figdir = os.path.join(_JCP, "figures", name)
    present = set().union(*[set(r) for r in rows])
    single = [m for m in _SINGLE if m in present]
    for m in single:
        for axis in ("t", "nfe", "wallclock"):
            metric_single(rows, m, os.path.join(figdir, f"{name}_{m}_{axis}"),
                          xaxis=axis, floors=floors, methods=methods,
                          emc_target=emc, show=False, label_overrides=lov)
    metric_grid(rows, os.path.join(figdir, f"{name}_metrics"),
                metrics=("W2", "MMD", "EMC"), floors=floors, emc_target=emc,
                methods=methods, show=False, label_overrides=lov)
    print(f"replotted {name}: {len(single)} metrics x 3 axes + grid  "
          f"(methods={methods}, CSV+manifest only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
