"""Regenerate the manuscript metric figures from the frozen in-repo results.

* E1--E4 use the manuscript DISPLAY matrix ``REPORT_METHODS``, a subset of the
  internal method matrix in ``src.manuscript``. Nothing is deleted from the
  result files; the omitted arms are simply not drawn.
* BAOAB is displayed as ULD (underdamped Langevin dynamics).
* Only W2, MMD, basin TV, and worst-basin ESS are plotted. The ``W2`` column is
  an exact 1D W_2 in E1 and a fixed-projection sliced W_2 in E2--E4, and each
  figure is labeled accordingly.
* Every metric is exported individually for physical-time, NFE, and wall-clock
  views.
* Every experiment/axis also receives one 2-by-2 combined figure.
* Outputs are separated by format into ``figures/{png,tiff,svg,pdf}``.

The first three metrics are checkpoint time series.  Worst-basin ESS is only
available as a post-settling stationarity statistic, so it is shown as a bar
comparison: raw ESS in the t view, ESS per million NFE in the NFE view, and
ESS per second in the wall-clock view. The script never fabricates an ESS
time series.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from typing import Iterable

# Matplotlib needs a writable cache in the managed desktop environment.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/jcp-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
JCP_ROOT = HERE.parent
DEFAULT_RESULTS_DIR = JCP_ROOT / "results"
DEFAULT_FIGURES_DIR = JCP_ROOT / "figures"
if str(JCP_ROOT) not in sys.path:
    sys.path.insert(0, str(JCP_ROOT))
from src.manuscript import (  # noqa: E402
    EXPERIMENTS as RELEASE_EXPERIMENTS,
    FIGURE_FORMATS,
    RESOURCE_AXES,
)

AXES = RESOURCE_AXES
METRICS = ("W2", "MMD", "TV", "worst_basin_ESS")
TIME_SERIES_METRICS = METRICS[:3]


@dataclass(frozen=True)
class ExperimentSpec:
    display_name: str
    methods: tuple[str, ...]
    labels: dict[str, str]


# Manuscript DISPLAY matrix. The internal method matrix in src/manuscript.py
# and the result CSV files are unchanged; this selects which arms each figure
# draws.
#
# Every competing method an experiment runs is drawn. The only omissions are
# MALA, which tracks ULA to within ~0.5% on every metric and would plot as a
# second line under it, and Raw-CP outside E1, where it is not a comparator but
# the geometric-bias diagnostic E1 exists to isolate.
#
# FLA in particular is always drawn. It is the UNCORRECTED nonlocal baseline,
# so it is the natural comparator for a corrected nonlocal method, and it is
# competitive where it matters: on E1 it beats LSC-CP on W2 (0.096 vs 0.111,
# z = 3.0), MMD and basin TV, and on E4 it is the nearest non-LSC method
# (0.194 against PT's 0.364). PT is drawn on E1 for the same reason -- it sits
# essentially on the W2 bias floor there (0.064 against a floor of 0.061).
# Dropping either would remove exactly the curves a reader most needs in order
# to judge the method, and both are recoverable from the CSV files anyway.
REPORT_METHODS: dict[str, tuple[str, ...]] = {
    "double_well": ("ULA", "BAOAB", "PT", "FLA", "CP", "LSC-CP", "LSC-CP-RA"),
    "mog40": ("ULA", "BAOAB", "PT", "FLA", "LSC-CP", "LSC-CP-RA"),
    "mb3well_10d": ("ULA", "BAOAB", "PT", "FLA", "LSC-CP", "LSC-CP-MA"),
    "coupled_phi4": ("ULA", "BAOAB", "PT", "FLA", "LSC-CP", "LSC-CP-MA"),
}

# Only the 1D example computes an exact W_2; the rest compute a fixed
# projection sliced W_2 into the same CSV column.
EXACT_W2 = {"double_well"}


def _report_methods(key: str, available: tuple[str, ...]) -> tuple[str, ...]:
    """Display subset for `key`, ordered as in the internal method matrix."""
    selected = REPORT_METHODS.get(key)
    if selected is None:
        return available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(
            f"{key}: display matrix requests methods outside the release "
            f"matrix: {unknown}"
        )
    return tuple(method for method in available if method in set(selected))


EXPERIMENTS: dict[str, ExperimentSpec] = {
    key: ExperimentSpec(
        display_name=f"{spec.number}: {spec.title}",
        methods=_report_methods(key, spec.methods),
        labels=spec.display_labels,
    )
    for key, spec in RELEASE_EXPERIMENTS.items()
}


# figures4papers semantic palette and print-safe redundant encodings.
METHOD_STYLE: dict[str, dict] = {
    "ULA": {
        "color": "#767676", "linestyle": "-", "marker": "o", "hatch": "..",
    },
    "BAOAB": {
        "color": "#42949E", "linestyle": "--", "marker": "s", "hatch": "//",
    },
    "FLA": {
        "color": "#009E73", "linestyle": ":", "marker": "^", "hatch": "oo",
    },
    "PT": {
        "color": "#9A4D8E", "linestyle": "-.", "marker": "D", "hatch": "\\\\",
    },
    "CP": {
        "color": "#B64342", "linestyle": (0, (5, 2)), "marker": "X", "hatch": "xx",
    },
    "LSC-CP": {
        "color": "#0F4D92", "linestyle": "-", "marker": "*", "hatch": "",
    },
    "LSC-CP-RA": {
        "color": "#3775BA", "linestyle": (0, (3, 1, 1, 1)),
        "marker": "P", "hatch": "++",
    },
    "LSC-CP-MA": {
        "color": "#3775BA", "linestyle": (0, (3, 1, 1, 1)),
        "marker": "P", "hatch": "++",
    },
}

METRIC_LABELS = {
    "W2": r"$W_2$",
    "MMD": "MMD",
    "TV": "Basin TV",
    "worst_basin_ESS": "Worst-basin ESS",
}


def metric_label(metric: str, *, exact_w2: bool = True) -> str:
    """Dimension-aware display label: exact W_2 in E1, sliced W_2 elsewhere."""
    if metric == "W2" and not exact_w2:
        return r"$\mathrm{SW}_2$"
    return METRIC_LABELS[metric]


X_AXIS = {
    "t": ("t", r"$t=n\,\Delta t$"),
    "nfe": ("nfe", "NFE"),
    "wallclock": ("wallclock_s", "Wall-clock time (s)"),
}

COST_AXES = ("nfe", "wallclock")

# Publication export formats come from src.manuscript (the release's single
# source of truth); raster ones get RASTER_DPI, vector ones stay vector.
RASTER_FORMATS: frozenset[str] = frozenset({"png", "tiff"})
RASTER_DPI = 600

ESS_AXIS = {
    "t": ("worst_basin_ess", "Worst-basin ESS", 1.0),
    "nfe": (
        "worst_basin_ess_per_nfe",
        r"Worst-basin ESS per $10^6$ NFE",
        1.0e6,
    ),
    "wallclock": (
        "worst_basin_ess_per_second",
        r"Worst-basin ESS s$^{-1}$",
        1.0,
    ),
}


def apply_publication_style() -> None:
    """Apply the figures4papers publication conventions."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 2.0,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "lines.linewidth": 2.3,
        "lines.markersize": 6,
        "legend.frameon": False,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    })


def _blend_with_white(color: str, keep: float = 0.22) -> tuple[float, ...]:
    rgb = np.asarray(matplotlib.colors.to_rgb(color))
    return tuple(keep * rgb + (1.0 - keep) * np.ones(3))


def _running_mean(values: np.ndarray, window: int = 9) -> np.ndarray:
    """Centered, edge-normalized running mean that preserves the shared start."""
    if window <= 1 or len(values) < 2:
        return values.copy()
    window = min(int(window), len(values))
    kernel = np.ones(window)
    out = (
        np.convolve(values, kernel, mode="same")
        / np.convolve(np.ones_like(values), kernel, mode="same")
    )
    for i in range(min(window // 2, len(values))):
        out[i] = values[: 2 * i + 1].mean()
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def _load_floors(experiment_dir: Path) -> dict:
    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    floors = manifest.get("bias_floors")
    if not isinstance(floors, dict):
        raise ValueError(f"{manifest_path} has no bias_floors mapping")
    return floors


def _series(
    rows: Iterable[dict[str, str]],
    method: str,
    metric: str,
    xkey: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_step: dict[int, dict[str, list[float]]] = {}
    for row in rows:
        if row.get("method") != method:
            continue
        if row.get(metric, "") == "" or row.get(xkey, "") == "":
            continue
        step = int(float(row["step"]))
        entry = by_step.setdefault(step, {"x": [], "y": []})
        entry["x"].append(float(row[xkey]))
        entry["y"].append(float(row[metric]))
    steps = sorted(by_step)
    x = np.asarray([np.mean(by_step[s]["x"]) for s in steps], dtype=float)
    y = np.asarray([np.mean(by_step[s]["y"]) for s in steps], dtype=float)
    sd = np.asarray([
        np.std(by_step[s]["y"], ddof=1) if len(by_step[s]["y"]) > 1 else 0.0
        for s in steps
    ])
    return x, y, sd


def _stationarity_summary(
    path: Path,
    methods: Iterable[str],
) -> dict[str, dict[str, float]]:
    rows = _read_csv(path)
    selected: dict[str, dict[str, float]] = {}
    wanted = set(methods)
    for row in rows:
        method = row.get("method")
        if method not in wanted or method in selected:
            continue
        gradient = float(row.get("gradient_evals") or 0.0)
        potential = float(row.get("potential_evals") or 0.0)
        score = float(row.get("score_quadrature_evals") or 0.0)
        total_nfe = gradient + potential + score
        ess = float(row.get("worst_basin_ess") or 0.0)
        # Every method's stationarity trace is produced by the same run, on the
        # one declared GPU, so ESS/s is an apples-to-apples comparison.
        ess_per_second = float(row.get("worst_basin_ess_per_second") or 0.0)
        selected[method] = {
            "worst_basin_ess": ess,
            "worst_basin_ess_per_nfe": ess / total_nfe if total_nfe else math.nan,
            "worst_basin_ess_per_second": ess_per_second,
        }
    return selected


def _plot_time_series(
    ax,
    rows: list[dict[str, str]],
    spec: ExperimentSpec,
    metric: str,
    axis: str,
    floors: dict,
    *,
    smooth: int = 1,
    exact_w2: bool = True,
) -> None:
    xkey, xlabel = X_AXIS[axis]
    all_terminal_x: list[float] = []
    # Publication panels show the raw checkpoint means (smooth=1). The band is
    # a display element on a log axis: a non-positive mean-minus-SD edge is
    # otherwise clipped to the float minimum and then owns the whole y range,
    # so it is clipped at a positive data-scale display floor instead.
    series = {}
    for method in spec.methods:
        x, y, sd = _series(rows, method, metric, xkey)
        if not len(x):
            continue
        y = _running_mean(y, smooth)
        sd = _running_mean(sd, smooth)
        if axis in COST_AXES:
            keep = x > 0
            x, y, sd = x[keep], y[keep], sd[keep]
        if not len(x):
            continue
        series[method] = (x, y, sd)
    positive = [float(v) for _, y, _ in series.values() for v in y
                if np.isfinite(v) and v > 0.0]
    band_floor = 0.5 * min(positive) if positive else np.finfo(float).tiny

    for method, (x, y, sd) in series.items():
        style = METHOD_STYLE[method]
        all_terminal_x.append(float(x[-1]))
        lower = np.maximum(y - sd, band_floor)
        upper = np.maximum(y + sd, band_floor)
        ax.fill_between(
            x,
            lower,
            upper,
            color=_blend_with_white(style["color"]),
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            y,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=max(1, len(x) // 8),
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=1.0,
            label=spec.labels[method],
            zorder=3,
        )

    if axis in COST_AXES and all_terminal_x:
        terminal = [value for value in all_terminal_x if value > 0]
        if terminal and max(terminal) / min(terminal) >= 10:
            ax.set_xscale("log")
    ax.set_yscale("log")
    floor = (floors.get(metric) or {}).get("mean")
    if floor is not None and float(floor) > 0:
        ax.axhline(
            float(floor),
            color="#4D4D4D",
            linestyle=(0, (2, 2)),
            linewidth=1.3,
            zorder=2,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric_label(metric, exact_w2=exact_w2))


def _plot_ess(
    ax,
    summary: dict[str, dict[str, float]],
    spec: ExperimentSpec,
    axis: str,
) -> None:
    key, ylabel, scale = ESS_AXIS[axis]
    positions = np.arange(len(spec.methods), dtype=float)
    finite_values: list[float] = []
    for xpos, method in zip(positions, spec.methods):
        style = METHOD_STYLE[method]
        value = (summary.get(method) or {}).get(key, math.nan)
        if math.isfinite(value):
            plotted = value * scale
            finite_values.append(plotted)
            ax.bar(
                xpos,
                plotted,
                width=0.72,
                color=style["color"],
                edgecolor="#272727",
                linewidth=1.2,
                hatch=style["hatch"],
                zorder=2,
            )
            if plotted == 0:
                ax.text(
                    xpos,
                    0,
                    "0",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="#272727",
                )
        else:
            ax.text(
                xpos,
                0,
                "n/a",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#767676",
            )
    ax.set_xticks(positions, [spec.labels[m] for m in spec.methods], rotation=30)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Method")
    top = max(finite_values, default=1.0)
    ax.set_ylim(0, top * 1.16 if top > 0 else 1.0)


def _legend_handles(spec: ExperimentSpec) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_STYLE[method]["color"],
            linestyle=METHOD_STYLE[method]["linestyle"],
            marker=METHOD_STYLE[method]["marker"],
            markerfacecolor="white",
            markeredgewidth=1.0,
            linewidth=2.3,
            label=spec.labels[method],
        )
        for method in spec.methods
    ]


def _save_figure(
    fig,
    basename: str,
    format_dirs: dict[str, Path],
) -> tuple[Path, ...]:
    """Write one figure in every publication format, one directory per format.

    Raster formats are written at ``RASTER_DPI``; TIFF additionally uses
    lossless LZW compression, which is what journal production systems ask for.
    SVG and PDF stay vector with embedded (type-42) fonts."""
    written: list[Path] = []
    for extension, directory in format_dirs.items():
        path = directory / f"{basename}.{extension}"
        kwargs: dict = {"bbox_inches": "tight", "facecolor": "white"}
        if extension in RASTER_FORMATS:
            kwargs["dpi"] = RASTER_DPI
        if extension == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        written.append(path)
    plt.close(fig)
    return tuple(written)


def _individual_figure(
    experiment: str,
    spec: ExperimentSpec,
    rows: list[dict[str, str]],
    floors: dict,
    summary: dict[str, dict[str, float]],
    metric: str,
    axis: str,
    format_dirs: dict[str, Path],
    *,
    exact_w2: bool = True,
) -> tuple[Path, ...]:
    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    if metric in TIME_SERIES_METRICS:
        _plot_time_series(ax, rows, spec, metric, axis, floors,
                          exact_w2=exact_w2)
        fig.legend(
            handles=_legend_handles(spec),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.035),
            ncol=min(3, len(spec.methods)),
            columnspacing=1.1,
            handlelength=2.5,
        )
        fig.tight_layout(pad=1.0, rect=(0, 0, 1, 0.88))
    else:
        _plot_ess(ax, summary, spec, axis)
        fig.tight_layout(pad=1.2)
    return _save_figure(fig, f"{experiment}_{metric}_{axis}", format_dirs)


def _combined_figure(
    experiment: str,
    spec: ExperimentSpec,
    rows: list[dict[str, str]],
    floors: dict,
    summary: dict[str, dict[str, float]],
    axis: str,
    format_dirs: dict[str, Path],
    *,
    exact_w2: bool = True,
) -> tuple[Path, ...]:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    flat = axes.ravel()
    for panel, metric in zip(flat[:3], TIME_SERIES_METRICS):
        _plot_time_series(panel, rows, spec, metric, axis, floors,
                          exact_w2=exact_w2)
    _plot_ess(flat[3], summary, spec, axis)

    for label, panel in zip(("a", "b", "c", "d"), flat):
        panel.text(
            -0.16,
            1.04,
            f"({label})",
            transform=panel.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
        )

    fig.legend(
        handles=_legend_handles(spec),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(spec.methods),
        columnspacing=1.2,
        handlelength=2.5,
    )
    fig.tight_layout(pad=1.0, rect=(0, 0, 1, 0.93))
    return _save_figure(fig, f"{experiment}_combined_{axis}", format_dirs)


def regenerate(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    *,
    clean: bool = True,
) -> dict:
    """Regenerate the complete manuscript figure matrix."""
    results_dir = results_dir.expanduser().resolve()
    figures_dir = figures_dir.expanduser().resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(results_dir)
    protected = {
        Path("/").resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
        JCP_ROOT.resolve(),
        results_dir,
    }
    if figures_dir in protected or figures_dir.parent == figures_dir:
        raise ValueError(f"refusing unsafe figures directory: {figures_dir}")

    if clean and figures_dir.exists():
        shutil.rmtree(figures_dir)
    format_dirs = {extension: figures_dir / extension
                   for extension in FIGURE_FORMATS}
    for directory in format_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    apply_publication_style()
    outputs: list[Path] = []
    per_experiment: dict[str, dict] = {}
    for experiment, spec in EXPERIMENTS.items():
        experiment_dir = results_dir / experiment
        rows = _read_csv(experiment_dir / "metrics_timeseries.csv")
        floors = _load_floors(experiment_dir)
        in_data = {row["method"] for row in rows}
        missing_methods = sorted(set(spec.methods) - in_data)
        if missing_methods:
            raise ValueError(
                f"{experiment} is missing requested methods: {missing_methods}"
            )
        summary = _stationarity_summary(
            experiment_dir / "stationarity" / "all_methods_summary.csv",
            spec.methods,
        )

        exact_w2 = experiment in EXACT_W2
        for axis in AXES:
            for metric in METRICS:
                outputs.extend(_individual_figure(
                    experiment,
                    spec,
                    rows,
                    floors,
                    summary,
                    metric,
                    axis,
                    format_dirs,
                    exact_w2=exact_w2,
                ))
            outputs.extend(_combined_figure(
                experiment,
                spec,
                rows,
                floors,
                summary,
                axis,
                format_dirs,
                exact_w2=exact_w2,
            ))
        per_experiment[experiment] = {
            "methods": list(spec.methods),
            "labels": {method: spec.labels[method] for method in spec.methods},
            "stationarity_methods": sorted(summary),
        }

    expected_per_format = len(EXPERIMENTS) * len(AXES) * (len(METRICS) + 1)
    # Validate the files generated by this function, not every file already in
    # the shared manuscript directories. Density/scatter figures intentionally
    # coexist there when ``--no-clean`` is used.
    by_format = {
        extension: sorted(path for path in outputs
                          if path.suffix == f".{extension}")
        for extension in FIGURE_FORMATS
    }
    for extension, paths in by_format.items():
        if len(paths) != expected_per_format:
            raise RuntimeError(
                f"expected {expected_per_format} {extension.upper()} figures, "
                f"found {len(paths)}"
            )
    return {
        "results_dir": str(results_dir),
        "figures_dir": str(figures_dir),
        "formats": list(FIGURE_FORMATS),
        "count_per_format": expected_per_format,
        "counts": {extension: len(paths)
                   for extension, paths in by_format.items()},
        "png_count": len(by_format["png"]),
        "pdf_count": len(by_format["pdf"]),
        "per_experiment": per_experiment,
        "outputs": [str(path) for path in outputs],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="directory containing E1--E4 result folders",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="output directory; one subdirectory per format is created here",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="do not remove the existing figures directory before writing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = regenerate(
        args.results_dir,
        args.figures_dir,
        clean=not args.no_clean,
    )
    counts = ", ".join(f"{count} {extension.upper()}"
                       for extension, count in result["counts"].items())
    print(f"Regenerated manuscript figures: {counts} in "
          f"{result['figures_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
