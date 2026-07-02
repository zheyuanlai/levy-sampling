from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    try:
        x = float(row.get(key, "nan"))
    except Exception:
        return math.nan
    return x if math.isfinite(x) else math.nan


def label_for(row):
    bank = row.get("bank_name", "")
    method = row.get("method", "")
    if bank and bank != "none":
        return f"{method}\n{bank.replace('_', ' ')}"
    return method


def save_fig(fig, out_dir, name):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out / f"{name}.{ext}"; fig.savefig(p, bbox_inches="tight", dpi=180); paths.append(p)
    plt.close(fig); return paths


def metric_bar(rows, metric: str, out_dir, name: str, title: str, ylabel: str | None = None):
    col = f"{metric}_mean"
    vals, labels = [], []
    for r in rows:
        v = fnum(r, col)
        if math.isfinite(v):
            vals.append(v); labels.append(label_for(r))
    if not vals: return []
    fig, ax = plt.subplots(figsize=(max(6.0, 0.62 * len(labels)), 4.2))
    ax.bar(range(len(labels)), vals, color="#4C78A8")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(ylabel or metric.replace("_", " "))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); return save_fig(fig, out_dir, name)


def manywell_line(rows, metric: str, out_dir, name: str, title: str, ylabel: str):
    sub = [r for r in rows if r.get("experiment_name") == "manywell_scaling"]
    series: dict[str, list[tuple[float, float]]] = {}
    for r in sub:
        d = fnum(r, "dimension_mean")
        y = fnum(r, f"{metric}_mean")
        if not (math.isfinite(d) and math.isfinite(y)):
            continue
        series.setdefault(label_for(r).replace("\n", "/"), []).append((d, y))
    if not series:
        return []
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for lab, pts in sorted(series.items()):
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=lab)
    ax.set_xlabel("dimension")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log", base=2)
    if metric in {"ess_per_sec", "count_mode_kl", "block_marginal_kl"}:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); return save_fig(fig, out_dir, name)


METHOD_LABEL = {"ULA": "local Langevin", "RawCP": "raw CP", "LSBMC": "LSC-CP"}
METHOD_COLOR = {"ULA": "#888888", "RawCP": "#E4572E", "LSBMC": "#4C78A8",
                "MALA": "#54A24B", "BAOAB": "#B279A2", "HMC": "#EECA3B", "PT": "#72B7B2"}


def grouped_metric_bar(rows, metric, out_dir, name, title, ylabel,
                       methods=("ULA", "RawCP", "LSBMC"), by_bank=True):
    """Bar chart of a metric grouped by (bank, method) for the manuscript spine methods."""
    col = f"{metric}_mean"; se_col = f"{metric}_se"
    banks = []
    for r in rows:
        b = r.get("bank_name", "")
        if b and b not in banks and (by_bank or b == "none"):
            banks.append(b)
    if not by_bank:
        banks = ["none"]
    data = {}
    for r in rows:
        m = r.get("method"); b = r.get("bank_name", "")
        if m in methods:
            data[(b, m)] = (fnum(r, col), fnum(r, se_col))
    # x groups = banks that any jump method uses; local (ULA/none) shown once per group
    jump_banks = [b for b in banks if b != "none"] or ["none"]
    fig, ax = plt.subplots(figsize=(max(5.5, 1.6 * len(jump_banks)), 4.2))
    width = 0.8 / max(1, len(methods)); x0 = range(len(jump_banks))
    any_bar = False
    for mi, m in enumerate(methods):
        ys, es, xs = [], [], []
        for gi, b in enumerate(jump_banks):
            key = (b, m) if m in ("RawCP", "LSBMC") else ("none", m)
            v, e = data.get(key, (math.nan, math.nan))
            if math.isfinite(v):
                ys.append(v); es.append(e if math.isfinite(e) else 0.0); xs.append(gi + mi * width)
        if ys:
            any_bar = True
            ax.bar(xs, ys, width=width, yerr=es, capsize=3,
                   color=METHOD_COLOR.get(m, "#4C78A8"), label=METHOD_LABEL.get(m, m))
    if not any_bar:
        plt.close(fig); return []
    ax.set_xticks([i + 0.4 - width / 2 for i in x0], [b.replace("_", " ") for b in jump_banks], fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); return save_fig(fig, out_dir, name)


def timestep_bias_line(rows, out_dir, name, metric="cdf_sup_error",
                       methods=("ULA", "RawCP", "LSBMC")):
    """CDF-sup (residual, equilibrium init) vs dt across the double_well_timestep_* experiments."""
    series = {m: [] for m in methods}
    for r in rows:
        exp = r.get("experiment_name", "")
        if not exp.startswith("double_well_timestep_dt"):
            continue
        dt = fnum(r, "dt_mean"); y = fnum(r, f"{metric}_mean")
        m = r.get("method")
        if m in series and math.isfinite(dt) and math.isfinite(y):
            series[m].append((dt, y))
    if not any(series.values()):
        return []
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for m in methods:
        pts = sorted(series[m])
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                    color=METHOD_COLOR.get(m), label=METHOD_LABEL.get(m, m))
    ax.set_xscale("log"); ax.set_xlabel("timestep $h$"); ax.set_ylabel("residual CDF-sup at equilibrium")
    ax.set_title("Timestep bias: raw CP defect is $h$-independent")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); return save_fig(fig, out_dir, name)


def scale_intensity_heatmap(rows, out_dir, name, experiment, metric="cdf_sup_error", title=""):
    """Heatmap of a metric over jump scale c (rows) x intensity lambda (cols) for LSC-CP."""
    import numpy as np
    cs, lams, cell = set(), set(), {}
    for r in rows:
        if r.get("experiment_name") != experiment:
            continue
        bn = r.get("bank_name", "")
        try:
            c = float(bn.split("_l")[0][1:]); lam = float(bn.split("_l")[1])
        except Exception:
            continue
        v = fnum(r, f"{metric}_mean")
        if math.isfinite(v):
            cs.add(c); lams.add(lam); cell[(c, lam)] = v
    if not cell:
        return []
    cs = sorted(cs); lams = sorted(lams)
    M = np.array([[cell.get((c, lam), math.nan) for lam in lams] for c in cs])
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis_r")
    ax.set_xticks(range(len(lams)), lams); ax.set_yticks(range(len(cs)), cs)
    ax.set_xlabel("intensity $\\lambda$"); ax.set_ylabel("jump scale $c$ (length $cD$)")
    ax.set_title(title or experiment); fig.colorbar(im, ax=ax, label=metric.replace("_", " "))
    fig.tight_layout(); return save_fig(fig, out_dir, name)


def tv_over_time(run_dirs, out_dir, name, title, methods=("ULA", "RawCP", "LSBMC"), bank_filter=None):
    """Basin-TV vs time from per-run timeseries.csv (the manuscript's time-evolution figure)."""
    import numpy as np
    series = {}
    for rd in run_dirs:
        ts = Path(rd) / "timeseries.csv"
        if not ts.exists():
            continue
        for r in read_csv(ts):
            m = r.get("method"); b = r.get("bank_name", "")
            if m not in methods:
                continue
            if bank_filter is not None and m in ("RawCP", "LSBMC") and b != bank_filter:
                continue
            t = fnum(r, "time"); tv = fnum(r, "basin_tv")
            if math.isfinite(t) and math.isfinite(tv):
                series.setdefault(m, {}).setdefault(t, []).append(tv)
    if not series:
        return []
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for m in methods:
        if m not in series:
            continue
        ts_sorted = sorted(series[m])
        ax.plot(ts_sorted, [np.mean(series[m][t]) for t in ts_sorted],
                color=METHOD_COLOR.get(m), label=METHOD_LABEL.get(m, m))
    ax.axhline(0.1, ls="--", color="k", alpha=0.4, lw=0.8)
    ax.set_xlabel("time"); ax.set_ylabel("basin-TV to target"); ax.set_title(title)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); return save_fig(fig, out_dir, name)


def _run_dirs_for(raw_rows, experiment):
    return sorted({r.get("run_dir", "") for r in raw_rows
                   if r.get("experiment_name") == experiment and r.get("run_dir")})


def generate_manuscript_figures(summary_csv: str, report_fig_dir: str = "reports/jcp_sampling_report/figures",
                                results_root: str | None = None):
    rows = read_csv(summary_csv)
    out = Path(report_fig_dir); out.mkdir(parents=True, exist_ok=True)
    for ext in ("*.png", "*.pdf"):
        for existing in out.glob(ext):
            existing.unlink()
    fig_paths = []
    # --- manuscript examples: local / raw CP / LSC-CP spine ---
    dw = [r for r in rows if r.get("experiment_name") == "double_well_reproduction"]
    fig_paths += grouped_metric_bar(dw, "cdf_sup_error", out, "fig_dw_cdf_sup",
                                    "Double well: target fidelity", "CDF-sup error to Gibbs")
    fig_paths += grouped_metric_bar(dw, "density_l1_error", out, "fig_dw_density_l1",
                                    "Double well: density error", "density $L^1$ error")
    tw = [r for r in rows if r.get("experiment_name") == "triple_well_support"]
    fig_paths += grouped_metric_bar(tw, "basin_population_error", out, "fig_tw_mode_tv",
                                    "Triple well: mode-TV by support", "mode-TV to target")
    fig_paths += grouped_metric_bar(tw, "observable_bias_middle_mass", out, "fig_tw_middle_mass",
                                    "Triple well: middle-mass error", "middle-mass error")
    mb = [r for r in rows if r.get("experiment_name") == "muller10d_basin_communication"]
    fig_paths += grouped_metric_bar(mb, "basin_tv_final", out, "fig_mb10_basin_tv",
                                    "10D Muller--Brown: basin-TV (lifted vs random)", "basin-TV to target")
    # --- ablations ---
    fig_paths += timestep_bias_line(rows, out, "fig_timestep_bias")
    fig_paths += scale_intensity_heatmap(rows, out, "fig_scale_intensity_equil",
                                         "double_well_scale_intensity", "cdf_sup_error",
                                         "Scale$\\times$intensity: target preservation (equilibrium init)")
    fig_paths += scale_intensity_heatmap(rows, out, "fig_scale_intensity_relax",
                                         "double_well_scale_intensity_relax", "cdf_sup_error",
                                         "Scale$\\times$intensity: relaxation speed (ridge at $c\\approx1$)")
    inv = [r for r in rows if r.get("experiment_name") in ("triple_well_invariance", "muller10d_invariance")]
    fig_paths += grouped_metric_bar(inv, "basin_tv_final", out, "fig_invariance",
                                    "Start-at-equilibrium invariance", "terminal basin-TV")
    al = [r for r in rows if r.get("experiment_name") == "alanine_ramachandran"]
    fig_paths += grouped_metric_bar(al, "basin_tv_final", out, "fig_alanine_basin_tv",
                                    "Alanine dipeptide: basin-TV (wrapped vs random)", "basin-TV to target",
                                    methods=("ULA", "PT", "RawCP", "LSBMC"))
    # --- time-evolution (needs per-run timeseries; run_dir lives in all_raw_metrics.csv) ---
    if results_root:
        raw_path = Path(results_root) / "aggregate" / "all_raw_metrics.csv"
        raw_rows = read_csv(raw_path) if raw_path.exists() else []
        for exp, bank, nm, ttl in [
            ("double_well_reproduction", "double_well_shell", "fig_dw_tv_time", "Double well: TV relaxation"),
            ("triple_well_support", "overlong", "fig_tw_tv_time", "Triple well: TV relaxation (overlong)"),
            ("muller10d_basin_communication", "minima_complete_graph", "fig_mb10_tv_time", "10D MB: TV relaxation")]:
            fig_paths += tv_over_time(_run_dirs_for(raw_rows, exp), out, nm, ttl, bank_filter=bank)
    # --- legacy / additional-landscape figures ---
    specs = [
        ("double_well_scale", "basin_kl", "fig_double_well_basin_kl", "Double well: basin KL", "basin KL"),
        ("double_well_scale", "ess_per_sec", "fig_double_well_ess_per_sec", "Double well: ESS/sec", "ESS/sec"),
        ("four_well_graph", "basin_kl", "fig_four_well_basin_kl", "Four well: basin KL", "basin KL"),
        ("four_well_graph", "basin_population_error", "fig_four_well_population_error", "Four well: basin population error", "population error"),
        ("muller_brown_free_energy", "free_energy_rmse", "fig_muller_brown_free_energy_rmse", "Muller--Brown: free-energy RMSE", "free-energy RMSE"),
        ("muller_brown_free_energy", "basin_kl", "fig_muller_brown_basin_kl", "Muller--Brown: basin KL", "basin KL"),
    ]
    for exp, metric, name, title, ylabel in specs:
        sub = [r for r in rows if r.get("experiment_name") == exp]
        fig_paths.extend(metric_bar(sub, metric, out, name, title, ylabel))
    fig_paths.extend(manywell_line(rows, "ess_per_sec", out, "fig_manywell_ess_per_sec", "ManyWell scaling: ESS/sec", "ESS/sec"))
    fig_paths.extend(manywell_line(rows, "block_marginal_kl", out, "fig_manywell_block_kl", "ManyWell scaling: block marginal KL", "block marginal KL"))
    fig_paths.extend(manywell_line(rows, "count_mode_kl", out, "fig_manywell_count_kl", "ManyWell scaling: count-mode KL", "count-mode KL"))
    return fig_paths
