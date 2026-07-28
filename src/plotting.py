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
    # Every experiment now plots TWO LSC arms, so they must be visually
    # distinct: BLACK is always the exact deterministic-quadrature score, PURPLE
    # is always the realised-displacement estimator (single-atom RA on E1/E2,
    # atom-stratified MA on E3/E4). Before both arms shared a figure, LSC-CP-MA
    # was styled identically to LSC-CP and the two curves would now coincide.
    "LSC-CP": dict(color="#000000", ls="-",  marker="*"),
    # realised-displacement estimator variants
    "CP-RA":     dict(color="#D55E00", ls=(0, (1, 1)), marker="P"),
    "LSC-CP-RA": dict(color="#7030A0", ls="-",  marker="*"),
    "LSC-CP-MA": dict(color="#7030A0", ls=(0, (4, 1, 1, 1)), marker="p"),
}

SIMPLE_LABELS: dict[str, str] = {
    "ULA": "ULA", "MALA": "MALA", "FLA": "FLA", "BAOAB": "BAOAB",
    "PT": "PT", "CP": "Raw-CP", "LSC-CP": "LSC-CP",
    # MA is the A-atom generalisation of RA, so it reads as "LSC-CP-RA"; each
    # experiment appends its atom count via label_overrides, e.g. E3 -> (4).
    "CP-RA": "Raw-CP (RA)", "LSC-CP-RA": "LSC-CP-RA", "LSC-CP-MA": "LSC-CP-RA",
}

METRIC_LABEL = {
    "W2": r"$W_2$", "MMD": "MMD", "EMC": "EMC", "TV": "TV",
    "TV_density": "density TV", "W2_10d": r"sliced $W_2$ (10D)",
    "EJS": "EJS",
    "FES_RMSE_kBT": r"FES RMSE  [$k_BT$]",
    "FES_outside_mass": "FES outside-grid mass",
    "basin_KL_target": (
        r"$D_{\mathrm{KL}}(p^\star_{\mathrm{basin}}"
        r"\Vert\,\hat p_{\mathrm{basin}})$"
    ),
    "e_F": r"FES RMSE  [$k_BT$]",  # legacy alias
    "basin_rel_max": "max basin rel. mass err", "basin_L1": r"basin $L_1$",
    "V_mean_err": r"$|\langle V\rangle-\langle V\rangle_\pi|$",
    "V_var_err": r"$|\mathrm{Var}(V)-\mathrm{Var}_\pi(V)|$",
    "E_overlap_deficit": "energy overlap deficit", "KSD": "KSD",
    "W1_cdf": r"$W_1$  ($\int|\hat F-F^\star|$)", "CDF_sup": "CDF sup (KS)",
    "cdf_L2": r"CDF $L_2$", "pdf_L1": r"pdf $L_1$", "pdf_L2": r"pdf $L_2$",
    "KDE_chi2": r"KDE $\chi^2$", "well_TV": "well TV",
    "bin_chi2_M40": r"bin $\chi^2$ (M=40)", "bin_chi2_M80": r"bin $\chi^2$ (M=80)",
    "bin_chi2_M120": r"bin $\chi^2$ (M=120)",
}

X_AXIS = {
    "t": ("t", r"$t=n\,\Delta t$"),
    "nfe": ("nfe", "NFE"),
    "wallclock": ("wallclock_s", "wall-clock (s)"),
}

# Terminal-cost ratio (slowest / fastest method) above which a cost axis is
# drawn on a log scale. Measured spreads on the delivered runs reach 104x
# (E2 wall-clock) and 32x (E1 NFE), so 10x is comfortably below the real cases.
LOGX_SPREAD = 10.0


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


def _running_mean(y: np.ndarray, w: int) -> np.ndarray:
    """Centered running mean, window w.

    LEFT edge uses symmetric-TRUNCATED windows (radius i at index i), so the
    first point is returned EXACTLY: all methods share the identical n=0
    ensemble, and an asymmetric partial window there would average in each
    method's own early transient -- fast converters get dragged down, slow
    ones don't, and the shared start visibly splits apart in the figures.
    The same asymmetry bias applies throughout the steep (non-stationary)
    early descent, which the growing symmetric window avoids.

    RIGHT edge keeps edge-normalized partial windows: the tail is stationary,
    where a partial-window mean is unbiased -- and it is exactly where the
    checkpoint-to-checkpoint Monte-Carlo jitter needs suppressing."""
    if w is None or w <= 1 or len(y) < 2:
        return y
    w = min(int(w), len(y))
    k = np.ones(w)
    num = np.convolve(y, k, mode="same")
    den = np.convolve(np.ones_like(y), k, mode="same")
    out = num / den
    for i in range(min(w // 2, len(y))):
        out[i] = y[: 2 * i + 1].mean()
    return out


def _series(rows, method, ykey):
    by_step: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        if r["method"] != method or ykey not in r or r[ykey] == "":
            continue
        # coerce step to int: rows may come from CSV (strings) -- a lexicographic
        # sort of string steps would connect points out of order
        e = by_step.setdefault(int(float(r["step"])), {"x": [], "y": []})
        e["x"].append(float(r["t"]))
        e["y"].append(float(r[ykey]))
    steps = sorted(by_step)
    x = np.array([np.mean(by_step[s]["x"]) for s in steps])
    y = np.array([np.mean(by_step[s]["y"]) for s in steps])
    sd = np.array([np.std(by_step[s]["y"], ddof=1) if len(by_step[s]["y"]) > 1 else 0.0
                   for s in steps])
    return x, y, sd


def _series_x(rows, method, ykey, xkey):
    """Per-checkpoint (mean, sd) of ykey vs a chosen x column (t / nfe /
    wallclock_s), aggregated over seeds. Includes the n=0 (step 0) frame."""
    by_step: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        if r["method"] != method or ykey not in r or r[ykey] == "":
            continue
        if r.get(xkey, "") == "":
            continue
        e = by_step.setdefault(int(float(r["step"])), {"x": [], "y": []})
        e["x"].append(float(r[xkey]))
        e["y"].append(float(r[ykey]))
    steps = sorted(by_step)
    x = np.array([np.mean(by_step[s]["x"]) for s in steps])
    y = np.array([np.mean(by_step[s]["y"]) for s in steps])
    sd = np.array([np.std(by_step[s]["y"], ddof=1) if len(by_step[s]["y"]) > 1 else 0.0
                   for s in steps])
    return x, y, sd


def metric_single(rows: list[dict], metric: str, out_base: str,
                  xaxis: str = "t", logy: bool = True, floors: dict | None = None,
                  emc_target: float = 1.0, methods=METHODS,
                  figsize=(4.4, 3.2), show: bool = True, smooth: int = 9,
                  label_overrides: dict | None = None,
                  xmax_mode: str | None = "baselines",
                  logx: str | bool = "auto"):
    """One metric, one figure (saved individually as png+pdf). Global log-y
    (except EMC, which is in [0,1]). `xaxis` in {'t','nfe','wallclock'}.

    `logx="auto"` puts the cost axes (nfe / wallclock) on a log scale whenever
    the terminal cost spans more than LOGX_SPREAD across methods, which is the
    normal case once the exact LSC-CP arm is plotted alongside a local baseline.
    The t-axis stays linear (shared physical time, and x=0 is meaningful).
    Pass True/False to force it. On a log x-axis the x=0 frame is dropped.

    `smooth` (default 9) applies a centered stationary running-average to each
    method's seed-mean curve and its band, suppressing the checkpoint-to-
    checkpoint Monte-Carlo jitter of the noisier estimators (W2/MMD/KSD) without
    biasing the plateau level. The raw per-frame values remain in the CSV; set
    smooth=1 to plot them unsmoothed.

    `xmax_mode="baselines"` (default; None disables): on the cost axes
    (nfe / wallclock) the scored LSC-CP method spends ~10-30x more per step
    than any baseline, so its curve would stretch the x-range and squeeze
    every other method against the y-axis. Truncate the x-axis at the largest
    terminal x among the plotted NON-LSC methods (usually PT) -- the LSC-CP
    curve is clipped there, past its equilibration plateau. The t-axis is
    never truncated (shared physical time)."""
    apply_style()
    floors = floors or {}
    lov = label_overrides or {}
    xkey, xlabel = X_AXIS[xaxis]
    fig, ax = plt.subplots(figsize=figsize)

    series = {}
    for method in methods:
        x, y, sd = _series_x(rows, method, metric, xkey)
        if len(x) == 0:
            continue
        series[method] = (x, _running_mean(y, smooth), _running_mean(sd, smooth))
    plotted = bool(series)
    xmax_by_method = {m: float(v[0].max()) for m, v in series.items()}

    # Cost axes: the exact LSC-CP arm can cost orders of magnitude more per step
    # than a local baseline, so a linear axis squeezes every baseline onto the
    # y-axis. Truncating at the slowest baseline (the old behaviour) hid the
    # LSC-CP tail instead. Use a log x-axis whenever the terminal-cost spread is
    # wide; it shows every curve in full.
    use_logx = bool(logx) if isinstance(logx, bool) else False
    if logx == "auto" and xaxis in ("nfe", "wallclock") and len(xmax_by_method) > 1:
        finite = [v for v in xmax_by_method.values() if v > 0]
        use_logx = bool(finite) and (max(finite) / min(finite)) > LOGX_SPREAD

    for method, (x, y, sd) in series.items():
        if use_logx:
            # x = 0 is the shared pre-run origin and has no place on a log axis
            keep = x > 0
            x, y, sd = x[keep], y[keep], sd[keep]
            if len(x) == 0:
                continue
        # a method relabeled to a canonical name adopts that name's style, so
        # the exact arm is black and the realised-displacement arm purple in
        # every figure regardless of which estimator produced the curve
        style_key = lov.get(method) if lov.get(method) in METHOD_STYLE else method
        st = METHOD_STYLE.get(style_key, dict(color="#444444", ls="-", marker="."))
        ax.fill_between(x, y - sd, y + sd,
                        color=blend_toward_white(st["color"]), lw=0, zorder=1)
        ax.plot(x, y, color=st["color"], ls=st["ls"], marker=st["marker"],
                markevery=max(1, len(x) // 8), markerfacecolor="white",
                markeredgecolor=st["color"], markeredgewidth=0.6,
                label=lov.get(method, SIMPLE_LABELS.get(method, method)), zorder=3)
    if use_logx:
        ax.set_xscale("log")
    elif (xmax_mode == "baselines" and xaxis in ("nfe", "wallclock")
            and len(xmax_by_method) > 1):
        base_max = max((v for m, v in xmax_by_method.items()
                        if not m.startswith("LSC-CP")), default=0.0)
        if base_max > 0.0 and max(xmax_by_method.values()) > base_max:
            # set BOTH limits: with only `right`, matplotlib keeps the auto
            # left margin computed for the untruncated LSC range (negative
            # and huge). All curves share the n=0 start at x=0.
            ax.set_xlim(-0.02 * base_max, 1.02 * base_max)
    if metric == "EMC":
        ax.axhline(emc_target, color="#666666", ls=(0, (2, 2)), lw=0.8, zorder=2)
        ax.set_ylim(-0.02, 1.05)
    elif metric == "FES_outside_mass":
        # A probability diagnostic belongs on a bounded linear scale; zeros
        # are scientifically meaningful and disappear on a log axis.
        ax.set_ylim(-0.02, 1.02)
    else:
        if logy:
            ax.set_yscale("log")
        fl = floors.get(metric, {}).get("mean")
        if fl:
            ax.axhline(fl, color="#666666", ls=(0, (2, 2)), lw=0.8, zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(METRIC_LABEL.get(metric, metric))
    if plotted:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, ncol=min(4, len(labels)), loc="lower center",
                   bbox_to_anchor=(0.5, 1.005), frameon=False,
                   handlelength=1.9, columnspacing=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    if show:
        _display_inline(fig)
    plt.close(fig)
    return fig


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
    plt.close(fig)
    return fig


def metric_grid(rows: list[dict], out_base: str,
                metrics=("W2", "MMD", "EMC"), floors: dict | None = None,
                emc_target: float = 1.0, methods=METHODS,
                figsize_per_panel=(3.4, 2.6), show: bool = True, smooth: int = 9,
                label_overrides: dict | None = None):
    """One row of panels (metric vs t), shared legend above the grid.
    Saves out_base + .png/.pdf and returns the figure (also shown inline).
    `smooth` applies the same stationary running-average as `metric_single`."""
    apply_style()
    floors = floors or {}
    lov = label_overrides or {}
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
            y = _running_mean(y, smooth)
            sd = _running_mean(sd, smooth)
            style_key = (lov.get(method)
                         if lov.get(method) in METHOD_STYLE else method)
            st = METHOD_STYLE.get(style_key, dict(color="#444444", ls="-", marker="."))
            ax.fill_between(x, y - sd, y + sd,
                            color=blend_toward_white(st["color"]), lw=0, zorder=1)
            ax.plot(x, y, color=st["color"], ls=st["ls"], marker=st["marker"],
                    markevery=max(1, len(x) // 8), markerfacecolor="white",
                    markeredgecolor=st["color"], markeredgewidth=0.6,
                    label=lov.get(method, SIMPLE_LABELS.get(method, method)), zorder=3)
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
    plt.close(fig)
    return fig


# ======================================================== CSV-only sample figures
REFERENCE_KEY = "reference"


def load_positions_csv(path: str) -> dict[str, np.ndarray]:
    """method -> (n, k) array, as written by runner.write_positions_csv.

    This is the only input the density-overlay and free-energy figures need, so
    they regenerate from the run's CSVs with no in-memory state.
    """
    import csv as _csv
    blocks: dict[str, list] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = _csv.DictReader(handle)
        cvs = [c for c in reader.fieldnames or [] if c.startswith("cv")]
        if not cvs:
            raise ValueError(f"{path} has no cv columns")
        for row in reader:
            vals = [row[c] for c in cvs]
            blocks.setdefault(row["method"], []).append(
                [float(v) for v in vals if v != ""])
    return {m: np.asarray(v, dtype=float) for m, v in blocks.items()}


def _shared_extent(positions: dict, pad: float = 0.04):
    ref = positions[REFERENCE_KEY]
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    return lo - pad * span, hi + pad * span


def density_overlay(positions: dict, method: str, out_base: str, *,
                    bins: int = 110, figsize=(4.0, 3.4), show: bool = True,
                    label_overrides: dict | None = None):
    """One method's terminal sample against the reference -- one method, one
    figure (no grid). 1-D draws both densities as step histograms; 2-D draws the
    reference as filled contours with the sample scattered on top, so a sampler
    that misses a mode or leaks off-support is visible directly."""
    apply_style()
    lov = label_overrides or {}
    name = lov.get(method, SIMPLE_LABELS.get(method, method))
    ref, emp = positions[REFERENCE_KEY], positions[method]
    lo, hi = _shared_extent(positions)
    fig, ax = plt.subplots(figsize=figsize)
    if ref.shape[1] == 1:
        edges = np.linspace(lo[0], hi[0], bins + 1)
        for data, color, lab, lw in ((ref, "#666666", "reference", 1.0),
                                     (emp, "#000000", name, 1.4)):
            h, _ = np.histogram(data[:, 0], bins=edges, density=True)
            ax.step(0.5 * (edges[1:] + edges[:-1]), h, where="mid",
                    color=color, lw=lw, label=lab)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel("density")
        ax.legend(frameon=False, fontsize=8)
    else:
        hr, xe, ye = np.histogram2d(
            ref[:, 0], ref[:, 1], bins=bins,
            range=[[lo[0], hi[0]], [lo[1], hi[1]]], density=True)
        ax.contourf(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), hr.T,
                    levels=10, cmap="Greys", zorder=1)
        ax.scatter(emp[:, 0], emp[:, 1], s=1.4, lw=0, alpha=0.28,
                   color="#D55E00", zorder=2, rasterized=True)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_xlabel(r"$s_1$")
        ax.set_ylabel(r"$s_2$")
        ax.set_title(f"{name}  vs reference (shaded)", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    if show:
        _display_inline(fig)
    plt.close(fig)
    return fig


def _fes_from_samples(data, edges, beta, floor_counts=0.5):
    """beta*F = -log p, shifted so min = 0. Empty cells get a pseudocount so an
    unvisited basin has a finite (large) free energy rather than +inf."""
    if len(edges) == 1:
        counts, _ = np.histogram(data[:, 0], bins=edges[0])
    else:
        counts, _, _ = np.histogram2d(data[:, 0], data[:, 1], bins=edges)
    p = (counts + floor_counts) / (counts + floor_counts).sum()
    f = -np.log(p) / float(beta)
    return f - np.nanmin(f)


def fes_profile_1d(positions: dict, out_base: str, *, beta: float,
                   methods=None, bins: int = 60, figsize=(4.6, 3.3),
                   show: bool = True, label_overrides: dict | None = None):
    """True free-energy profile plus the profile each sampler produces."""
    apply_style()
    lov = label_overrides or {}
    lo, hi = _shared_extent(positions)
    edges = [np.linspace(lo[0], hi[0], bins + 1)]
    centres = 0.5 * (edges[0][1:] + edges[0][:-1])
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(centres, _fes_from_samples(positions[REFERENCE_KEY], edges, beta),
            color="#666666", lw=2.2, ls="-", label="true FES", zorder=2)
    for method in (methods if methods is not None else
                   [m for m in positions if m != REFERENCE_KEY]):
        if method not in positions:
            continue
        st = METHOD_STYLE.get(method, dict(color="#444444", ls="-", marker="."))
        ax.plot(centres, _fes_from_samples(positions[method], edges, beta),
                color=st["color"], ls=st["ls"], lw=1.1, zorder=3,
                label=lov.get(method, SIMPLE_LABELS.get(method, method)))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$F(x)\ [k_BT]$")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=min(4, len(labels)), loc="lower center",
               bbox_to_anchor=(0.5, 1.005), frameon=False, handlelength=1.9,
               columnspacing=1.0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    if show:
        _display_inline(fig)
    plt.close(fig)
    return fig


def fes_ceiling(positions: dict, *, beta: float, bins: int = 90,
                percentile: float = 99.0) -> float:
    """Shared colour ceiling for a set of 2-D free-energy maps.

    Taken from the REFERENCE surface so every method's figure is drawn on the
    same scale and the maps are visually comparable; a per-figure autoscale
    would make a sampler that missed a basin look like it matched the target.
    """
    lo, hi = _shared_extent(positions)
    edges = [np.linspace(lo[j], hi[j], bins + 1) for j in range(2)]
    return float(np.nanpercentile(
        _fes_from_samples(positions[REFERENCE_KEY], edges, beta), percentile))


def fes_map_2d(positions: dict, method: str, out_base: str, *, beta: float,
               bins: int = 90, fmax: float | None = None, figsize=(4.2, 3.4),
               show: bool = True, label_overrides: dict | None = None):
    """One free-energy surface per figure. `method` may be REFERENCE_KEY to draw
    the true surface. `fmax` clips the colour range so every method shares one
    scale -- pass the same value for all figures of an experiment."""
    apply_style()
    lov = label_overrides or {}
    name = ("true FES" if method == REFERENCE_KEY
            else lov.get(method, SIMPLE_LABELS.get(method, method)))
    lo, hi = _shared_extent(positions)
    edges = [np.linspace(lo[0], hi[0], bins + 1),
             np.linspace(lo[1], hi[1], bins + 1)]
    f = _fes_from_samples(positions[method], edges, beta)
    fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(edges[0], edges[1], f.T, cmap="viridis",
                         vmin=0.0, vmax=fmax, shading="auto", rasterized=True)
    fig.colorbar(mesh, ax=ax, label=r"$F\ [k_BT]$")
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    ax.set_title(name, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    if show:
        _display_inline(fig)
    plt.close(fig)
    return fig
