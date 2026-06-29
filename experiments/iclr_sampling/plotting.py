"""Publication-quality plotting for the ICLR sampling suite.

Every figure is written as both ``.png`` and ``.pdf``.  Uses ``paper.mplstyle``
if present, otherwise sane defaults.  All functions are defensive against NaN
(inapplicable metrics) so they can be reused across targets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_COLORS = {
    "ULA": "steelblue", "MALA": "darkorange", "FLMC": "seagreen",
    "LSBMC": "firebrick", "PT": "purple", "HMC": "saddlebrown",
    "ULD": "teal",
}
METHOD_MARKERS = {
    "ULA": "o", "MALA": "s", "FLMC": "^", "LSBMC": "D",
    "PT": "v", "HMC": "P", "ULD": "X",
}


def use_paper_style(style_path: str = "paper.mplstyle"):
    try:
        plt.style.use(style_path)
    except Exception:
        matplotlib.rcParams.update({"figure.dpi": 120, "font.size": 11,
                                    "axes.grid": True, "grid.alpha": 0.3})


def _color(m):
    return METHOD_COLORS.get(m, None)


def _marker(m):
    return METHOD_MARKERS.get(m, "o")


def save_fig(fig, out_dir, name):
    out_dir = Path(out_dir)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        paths.append(p)
    plt.close(fig)
    return paths


def scaling_plot(summary: pd.DataFrame, x_col: str, metrics: List[str],
                 metric_labels: Dict[str, str], methods: List[str],
                 out_dir, name: str, title: str,
                 log_metrics: Optional[List[str]] = None,
                 x_label: Optional[str] = None):
    """Plot mean +/- SE of each metric vs a scaling variable, one line/method."""
    log_metrics = log_metrics or []
    metrics = [m for m in metrics if f"{m}_mean" in summary.columns]
    n = len(metrics)
    if n == 0:
        return None
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.2 * nrow),
                             squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, metric in zip(axes.ravel(), metrics):
        for m in methods:
            sub = summary[summary["method"] == m].sort_values(x_col)
            if sub.empty or f"{metric}_mean" not in sub:
                continue
            y = sub[f"{metric}_mean"].to_numpy()
            ye = sub[f"{metric}_se"].to_numpy()
            x = sub[x_col].to_numpy()
            if np.all(np.isnan(y)):
                continue
            ax.errorbar(x, y, yerr=ye, label=m, color=_color(m), marker=_marker(m),
                        lw=1.7, ms=5, capsize=3)
        ax.set_xlabel(x_label or x_col)
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_title(metric_labels.get(metric, metric))
        if metric in log_metrics:
            ax.set_yscale("log")
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return save_fig(fig, out_dir, name)


def mode_weight_bars(weights_by_method: Dict[str, np.ndarray],
                     target_weight: float, out_dir, name: str, title: str,
                     methods: Optional[List[str]] = None):
    """Bar plot of empirical per-mode probabilities vs the uniform target."""
    methods = methods or list(weights_by_method.keys())
    methods = [m for m in methods if m in weights_by_method]
    n = len(methods)
    ncol = min(n, 3)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow),
                             squeeze=False, sharey=True)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, m in zip(axes.ravel(), methods):
        w = weights_by_method[m]
        ax.bar(np.arange(len(w)), w, color=_color(m), alpha=0.85, width=0.8)
        ax.axhline(target_weight, color="k", ls="--", lw=1.2, label="target (uniform)")
        ax.set_title(m); ax.set_xlabel("mode index")
        ax.legend(fontsize=8)
    axes.ravel()[0].set_ylabel("empirical probability")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return save_fig(fig, out_dir, name)


def compute_quality_plot(summary: pd.DataFrame, compute_col: str,
                         quality_metric: str, quality_label: str,
                         methods: List[str], out_dir, name: str, title: str,
                         group_col: str = "dimension",
                         compute_label: Optional[str] = None):
    """Quality metric vs compute (wall-clock or gradient evals), per method."""
    if f"{quality_metric}_mean" not in summary.columns or compute_col not in summary.columns:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for m in methods:
        sub = summary[summary["method"] == m].sort_values(group_col)
        if sub.empty:
            continue
        x = sub[compute_col].to_numpy()
        y = sub[f"{quality_metric}_mean"].to_numpy()
        if np.all(np.isnan(y)):
            continue
        ax.plot(x, y, label=m, color=_color(m), marker=_marker(m), lw=1.5, ms=5)
    ax.set_xlabel(compute_label or compute_col)
    ax.set_ylabel(quality_label)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_fig(fig, out_dir, name)


def acceptance_plot(summary: pd.DataFrame, x_col: str, methods: List[str],
                    out_dir, name: str, title: str, x_label: Optional[str] = None):
    """Acceptance-rate diagnostics for MH samplers (and PT swap rate)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    plotted = False
    for m in methods:
        sub = summary[summary["method"] == m].sort_values(x_col)
        if sub.empty or "acceptance_rate_mean" not in sub:
            continue
        y = sub["acceptance_rate_mean"].to_numpy()
        if np.all(np.isnan(y)):
            continue
        ax.plot(sub[x_col].to_numpy(), y, label=f"{m} (MH acc)", color=_color(m),
                marker=_marker(m), lw=1.5, ms=5)
        plotted = True
    # PT swap acceptance
    sub = summary[summary["method"] == "PT"].sort_values(x_col)
    if not sub.empty and "swap_acceptance_rate_mean" in sub:
        y = sub["swap_acceptance_rate_mean"].to_numpy()
        if not np.all(np.isnan(y)):
            ax.plot(sub[x_col].to_numpy(), y, label="PT (swap acc)", color="purple",
                    ls="--", marker="v", lw=1.5, ms=5)
            plotted = True
    if not plotted:
        plt.close(fig); return None
    ax.set_xlabel(x_label or x_col); ax.set_ylabel("acceptance rate")
    ax.set_ylim(0, 1); ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout()
    return save_fig(fig, out_dir, name)


def convergence_plot(curves_by_config: Dict, metric: str, metric_label: str,
                     methods: List[str], out_dir, name: str, title: str):
    """Convergence of a cheap progress metric over time for representative configs."""
    configs = list(curves_by_config.keys())
    n = len(configs)
    ncol = min(n, 4)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.4 * nrow),
                             squeeze=False, sharey=True)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, cfg_name in zip(axes.ravel(), configs):
        curves = curves_by_config[cfg_name]
        for m in methods:
            if m not in curves:
                continue
            t = np.asarray(curves[m]["t"])
            y = np.asarray(curves[m][metric])
            if np.all(np.isnan(y)):
                continue
            ax.plot(t, y, label=m, color=_color(m), lw=1.4)
        ax.set_title(str(cfg_name)); ax.set_xlabel("time $t$")
        ax.legend(fontsize=7)
    axes.ravel()[0].set_ylabel(metric_label)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return save_fig(fig, out_dir, name)


def mog_scatter_panels(finals: Dict[str, np.ndarray], mu: np.ndarray,
                       ref: np.ndarray, out_dir, name: str, title: str,
                       methods: List[str], lim: float = 45.0):
    """Scatter of final samples per method with mode centres overlaid."""
    cols = ["reference"] + [m for m in methods if m in finals]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.8),
                             squeeze=False)
    for ax, name_c in zip(axes[0], cols):
        if name_c == "reference":
            pts = ref
            col = "0.5"
        else:
            pts = finals[name_c]
            col = _color(name_c)
        idx = np.random.default_rng(0).choice(pts.shape[0],
                                              min(2000, pts.shape[0]), replace=False)
        ax.scatter(pts[idx, 0], pts[idx, 1], s=3, c=col, alpha=0.5, rasterized=True)
        ax.scatter(mu[:, 0], mu[:, 1], marker="x", c="red", s=20, lw=1.0, zorder=5)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(name_c); ax.set_xlabel("$x_1$")
    axes[0, 0].set_ylabel("$x_2$")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return save_fig(fig, out_dir, name)
