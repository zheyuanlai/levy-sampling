"""Figures: one grid per experiment, metric vs t = n*dt only.

Conventions (per project owner):
* primary metrics W2, MMD, EMC in a single-row grid; linear y axes;
* simple legend labels (ULA, MALA, FLA, BAOAB, PT, Raw-CP, LSC-CP) placed
  OUTSIDE the axes so they never cover curves;
* saved as .png (600 dpi) and .pdf only, and displayed inline.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import METHODS

# Okabe-Ito colour-blind-safe palette; distinct linestyle + marker per method
METHOD_STYLE: dict[str, dict] = {
    "ULA":    dict(color="#0072B2", ls="-",  marker="o"),
    "MALA":   dict(color="#56B4E9", ls="--", marker="s"),
    "FLA":    dict(color="#009E73", ls="-.", marker="^"),
    "BAOAB":  dict(color="#E69F00", ls=":",  marker="v"),
    "PT":     dict(color="#CC79A7", ls=(0, (3, 1, 1, 1, 1, 1)), marker="D"),
    "CP":     dict(color="#D55E00", ls=(0, (5, 2)), marker="X"),
    "LSC-CP": dict(color="#000000", ls="-",  marker="*"),
}

SIMPLE_LABELS: dict[str, str] = {
    "ULA": "ULA", "MALA": "MALA", "FLA": "FLA", "BAOAB": "BAOAB",
    "PT": "PT", "CP": "Raw-CP", "LSC-CP": "LSC-CP",
}

METRIC_LABEL = {
    "W2": r"$W_2$", "MMD": "MMD", "EMC": "EMC", "TV": "TV",
    "TV_density": "density TV", "W2_10d": r"sliced $W_2$ (10D)",
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.4,
        "grid.color": "#cccccc",
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def blend_toward_white(color: str, keep: float = 0.30) -> tuple:
    rgb = matplotlib.colors.to_rgb(color)
    return tuple(keep * c + (1.0 - keep) * 1.0 for c in rgb)


def _series(rows, method, ykey):
    by_step: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        if r["method"] != method or ykey not in r or r[ykey] == "":
            continue
        e = by_step.setdefault(r["step"], {"x": [], "y": []})
        e["x"].append(float(r["t"]))
        e["y"].append(float(r[ykey]))
    steps = sorted(by_step)
    x = np.array([np.mean(by_step[s]["x"]) for s in steps])
    y = np.array([np.mean(by_step[s]["y"]) for s in steps])
    sd = np.array([np.std(by_step[s]["y"], ddof=1) if len(by_step[s]["y"]) > 1 else 0.0
                   for s in steps])
    return x, y, sd


def _display_inline(fig) -> None:
    """Show the figure in notebook output under the Agg backend: plt.show()
    is a no-op there, and display(fig) only emits a text repr unless the
    matplotlib-inline backend is active, so render PNG bytes explicitly."""
    try:
        import io
        from IPython.display import Image, display
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
        display(Image(data=buf.getvalue()))
    except Exception:
        plt.show()


def cdf_comparison(samples: dict, true_x, true_cdf, out_base: str,
                   methods=METHODS, xlabel: str = r"$x$",
                   max_points: int = 4000, show: bool = True):
    """1D empirical CDF of each method's terminal sample vs the true CDF,
    all on a single plot. `samples`: method -> 1D array; true CDF on a dense
    grid. Saved as .png/.pdf; legend outside the axes."""
    apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(np.asarray(true_x), np.asarray(true_cdf), color="#888888",
            lw=2.4, ls="-", label="true", zorder=1)
    for method in methods:
        if method not in samples:
            continue
        xs = np.sort(np.asarray(samples[method]).reshape(-1))
        cdf = np.arange(1, xs.size + 1) / xs.size
        if xs.size > max_points:                      # thin for plotting only
            idx = np.linspace(0, xs.size - 1, max_points).astype(int)
            xs, cdf = xs[idx], cdf[idx]
        st = METHOD_STYLE[method]
        ax.plot(xs, cdf, color=st["color"], ls=st["ls"], lw=1.1,
                marker=st["marker"], markevery=max(1, xs.size // 10),
                markerfacecolor="white", markeredgecolor=st["color"],
                markeredgewidth=0.6, markersize=3.5,
                label=SIMPLE_LABELS[method], zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_ylim(-0.02, 1.02)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 1.005), frameon=False,
               handlelength=1.9, columnspacing=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    if show:
        _display_inline(fig)
    return fig


def metric_grid(rows: list[dict], out_base: str,
                metrics=("W2", "MMD", "EMC"), floors: dict | None = None,
                emc_target: float = 1.0, methods=METHODS,
                figsize_per_panel=(3.4, 2.6), show: bool = True):
    """One row of panels (metric vs t), shared legend above the grid.
    Saves out_base + .png/.pdf and returns the figure (also shown inline)."""
    apply_style()
    floors = floors or {}
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel[0] * n,
                                            figsize_per_panel[1]))
    if n == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        for method in methods:
            x, y, sd = _series(rows, method, metric)
            if len(x) == 0:
                continue
            st = METHOD_STYLE[method]
            ax.fill_between(x, y - sd, y + sd,
                            color=blend_toward_white(st["color"]), lw=0, zorder=1)
            ax.plot(x, y, color=st["color"], ls=st["ls"], marker=st["marker"],
                    markevery=max(1, len(x) // 8), markerfacecolor="white",
                    markeredgecolor=st["color"], markeredgewidth=0.6,
                    label=SIMPLE_LABELS[method], zorder=3)
        if metric == "EMC":
            ax.axhline(emc_target, color="#666666", ls=(0, (2, 2)), lw=0.8, zorder=2)
            ax.set_ylim(-0.02, 1.05)
        else:
            fl = floors.get(metric, {}).get("mean")
            if fl:
                ax.axhline(fl, color="#666666", ls=(0, (2, 2)), lw=0.8, zorder=2)
            ax.set_ylim(bottom=0.0)
        ax.set_xlabel(r"$t = n\,\Delta t$")
        ax.set_ylabel(METRIC_LABEL.get(metric, metric))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), loc="lower center",
               bbox_to_anchor=(0.5, 1.005), frameon=False,
               handlelength=1.9, columnspacing=1.1)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    if show:
        _display_inline(fig)
    return fig
