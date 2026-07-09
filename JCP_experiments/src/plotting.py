"""Manuscript-ready figures for JCP.

Rules baked in here:
* one figure per metric, all seven methods on it; two x-axes per metric
  (t = n dt, and wall-clock) -> 10 figures per experiment;
* every figure saved separately in .pdf, .png (600 dpi) and .eps;
* EPS does not support transparency -> NO alpha anywhere; the +-1 s.d. band
  is a pre-blended solid RGB (line colour blended 70% toward white);
* colour-blind-safe palette + distinct linestyle + distinct marker per
  method, fixed once in METHOD_STYLE and used everywhere;
* log y for W2, TV, MMD, EJS; linear for EMC; dashed horizontal line for
  the bias floor (W2, MMD, TV, EJS) and the target value (EMC);
* serif/STIX fonts, single column 3.375 in, fonttype 42, captions emitted
  as .txt next to the figure files.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import METHOD_LABELS, METHODS

# Okabe-Ito colour-blind-safe palette
METHOD_STYLE: dict[str, dict] = {
    "ULA":    dict(color="#0072B2", ls="-",  marker="o"),
    "MALA":   dict(color="#56B4E9", ls="--", marker="s"),
    "FLA":    dict(color="#009E73", ls="-.", marker="^"),
    "BAOAB":  dict(color="#E69F00", ls=":",  marker="v"),
    "PT":     dict(color="#CC79A7", ls=(0, (3, 1, 1, 1, 1, 1)), marker="D"),
    "CP":     dict(color="#D55E00", ls=(0, (5, 2)), marker="X"),
    "LSC-CP": dict(color="#000000", ls="-",  marker="*"),
}

LOG_METRICS = {"W2", "TV", "MMD", "EJS", "TV_density", "W2_10d"}

WALLCLOCK_CAPTION_NOTE = (
    "Since $\\Delta t$ is shared and per-step cost is essentially constant, "
    "the wall-clock panel is the time panel rescaled horizontally per method "
    "by its cost per step; its content is the cost ratio between methods."
)

METRIC_LABEL = {
    "W2": r"$W_2$", "TV": "occupancy TV", "MMD": "MMD", "EMC": "EMC",
    "EJS": "EJS", "TV_density": "density TV", "W2_10d": r"sliced $W_2$ (10D)",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (3.375, 2.4),           # single column, 86 mm
        "figure.dpi": 110,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.4,
        "grid.color": "#cccccc",                  # light grey, NO alpha (EPS)
        "grid.alpha": 1.0,
        "legend.frameon": True,
        "legend.framealpha": 1.0,                 # solid frame, NO alpha
        "legend.edgecolor": "#cccccc",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def blend_toward_white(color: str, keep: float = 0.30) -> tuple:
    """Pre-blended solid band colour: `keep` of the line colour, rest white."""
    rgb = matplotlib.colors.to_rgb(color)
    return tuple(keep * c + (1.0 - keep) * 1.0 for c in rgb)


def _series(rows, method, ykey, xkey):
    """(x_mean, y_mean, y_std) across seeds at each checkpoint step."""
    by_step: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        if r["method"] != method or ykey not in r or r[ykey] == "":
            continue
        e = by_step.setdefault(r["step"], {"x": [], "y": []})
        e["x"].append(float(r[xkey]))
        e["y"].append(float(r[ykey]))
    steps = sorted(by_step)
    x = np.array([np.mean(by_step[s]["x"]) for s in steps])
    y = np.array([np.mean(by_step[s]["y"]) for s in steps])
    sd = np.array([np.std(by_step[s]["y"], ddof=1) if len(by_step[s]["y"]) > 1 else 0.0
                   for s in steps])
    return x, y, sd


def plot_metric(rows: list[dict], metric: str, xaxis: str, out_base: str,
                floor: float | None = None, target: float | None = None,
                methods=METHODS, caption_extra: str = "") -> str:
    """One metric, all methods; xaxis in {'t', 'wallclock_s'}. Saves
    .pdf/.png/.eps + a caption .txt; returns the caption."""
    fig, ax = plt.subplots()
    logy = metric in LOG_METRICS
    for method in methods:
        x, y, sd = _series(rows, method, metric, xaxis)
        if len(x) == 0:
            continue
        st = METHOD_STYLE[method]
        if logy:
            y = np.maximum(y, 1e-300)
        band_lo = np.maximum(y - sd, 1e-300) if logy else y - sd
        ax.fill_between(x, band_lo, y + sd,
                        color=blend_toward_white(st["color"]), lw=0, zorder=1)
        ax.plot(x, y, color=st["color"], ls=st["ls"], marker=st["marker"],
                markevery=max(1, len(x) // 9), markerfacecolor="white",
                markeredgecolor=st["color"], markeredgewidth=0.6,
                label=METHOD_LABELS[method], zorder=3)
    if logy:
        ax.set_yscale("log")
    if floor is not None and floor > 0:
        ax.axhline(floor, color="#666666", ls=(0, (2, 2)), lw=0.8, zorder=2)
    if target is not None:
        ax.axhline(target, color="#666666", ls=(0, (2, 2)), lw=0.8, zorder=2)
    ax.set_xlabel(r"$t = n\,\Delta t$" if xaxis == "t" else "wall-clock time (s)")
    if xaxis == "wallclock_s":
        ax.set_xscale("log")
    ax.set_ylabel(METRIC_LABEL.get(metric, metric))
    ax.legend(ncol=2, handlelength=1.6, columnspacing=0.8, borderpad=0.3,
              labelspacing=0.25, handletextpad=0.5, loc="best")
    caption = _caption(metric, xaxis, floor, target, caption_extra)
    save_fig(fig, out_base)
    with open(out_base + ".txt", "w") as f:
        f.write(caption + "\n")
    plt.close(fig)
    return caption


def _caption(metric: str, xaxis: str, floor, target, extra: str) -> str:
    name = METRIC_LABEL.get(metric, metric)
    parts = [f"{name} versus "
             + ("dynamics time $t=n\\Delta t$" if xaxis == "t" else "wall-clock time")
             + " for all seven methods (mean over 5 seeds; band $\\pm$1 s.d.,"
             " drawn as a pre-blended solid fill)."]
    if floor is not None and floor > 0:
        parts.append("Dashed line: finite-sample bias floor (two independent"
                     " reference samples of the same size, 20 replicates).")
    if target is not None:
        parts.append(f"Dashed line: target value {target:.4g}.")
    if xaxis == "wallclock_s":
        parts.append(WALLCLOCK_CAPTION_NOTE)
    if extra:
        parts.append(extra)
    return " ".join(parts)


def save_fig(fig, out_base: str) -> None:
    """Save each figure separately (never a grid) in pdf, png(600), eps."""
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    fig.savefig(out_base + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out_base + ".eps", bbox_inches="tight")


def make_all_figures(rows: list[dict], out_dir: str, floors: dict,
                     emc_target: float, metrics=("W2", "TV", "MMD", "EMC", "EJS"),
                     methods=METHODS) -> list[str]:
    """10 figures per experiment: {metric} x {time, wallclock}."""
    apply_style()
    written = []
    for metric in metrics:
        floor = floors.get(metric, {}).get("mean")
        target = emc_target if metric == "EMC" else None
        fl = None if metric == "EMC" else floor
        for xaxis, tag in (("t", "vs_time"), ("wallclock_s", "vs_wallclock")):
            base = os.path.join(out_dir, f"{metric}_{tag}")
            plot_metric(rows, metric, xaxis, base, floor=fl, target=target,
                        methods=methods)
            written.append(base)
    return written
