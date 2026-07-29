"""Regenerate the manuscript metric figures from the frozen in-repo results.

* E1--E4 use the manuscript method matrix agreed for the JCP paper.
* BAOAB is displayed as ULD (underdamped Langevin dynamics).
* Only W2, MMD, basin TV, and worst-basin ESS are plotted.
* Every metric is exported individually for physical-time and NFE views.
* Every experiment/axis also receives one 2-by-2 combined figure.
* PNG and PDF outputs are separated into ``figures/png`` and ``figures/pdf``.

The first three metrics are checkpoint time series.  Worst-basin ESS is only
available as a post-settling stationarity statistic, so it is shown as a bar
comparison: raw ESS in the t view and ESS per million NFE in the NFE view.
The script never fabricates an ESS time series. Wall-clock plots are withheld
until all methods have one common hardware and batching protocol.
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


EXPERIMENTS: dict[str, ExperimentSpec] = {
    key: ExperimentSpec(
        display_name=f"{spec.number}: {spec.title}",
        methods=spec.methods,
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

X_AXIS = {
    "t": ("t", r"$t=n\,\Delta t$"),
    "nfe": ("nfe", "NFE"),
}

ESS_AXIS = {
    "t": ("worst_basin_ess", "Worst-basin ESS", 1.0),
    "nfe": (
        "worst_basin_ess_per_nfe",
        r"Worst-basin ESS per $10^6$ NFE",
        1.0e6,
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
        # E1 Raw-CP stationarity was explicitly added on local CPU, whereas
        # the frozen stationarity runs for the other methods used the original
        # production hardware.  Its raw ESS and NFE-normalized ESS are valid,
        # but a cross-hardware ESS/s bar would not be an apples-to-apples
        # comparison.
        ess_per_second = (
            math.nan
            if method == "CP"
            else float(row.get("worst_basin_ess_per_second") or 0.0)
        )
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
    smooth: int = 9,
) -> None:
    xkey, xlabel = X_AXIS[axis]
    all_terminal_x: list[float] = []
    for method in spec.methods:
        x, y, sd = _series(rows, method, metric, xkey)
        if not len(x):
            continue
        style = METHOD_STYLE[method]
        y = _running_mean(y, smooth)
        sd = _running_mean(sd, smooth)
        if axis == "nfe":
            keep = x > 0
            x, y, sd = x[keep], y[keep], sd[keep]
        if not len(x):
            continue
        all_terminal_x.append(float(x[-1]))
        lower = np.maximum(y - sd, np.finfo(float).tiny)
        upper = np.maximum(y + sd, np.finfo(float).tiny)
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

    if axis == "nfe" and all_terminal_x:
        positive = [value for value in all_terminal_x if value > 0]
        if positive and max(positive) / min(positive) >= 10:
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
    ax.set_ylabel(METRIC_LABELS[metric])


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
    png_dir: Path,
    pdf_dir: Path,
) -> tuple[Path, Path]:
    png_path = png_dir / f"{basename}.png"
    pdf_path = pdf_dir / f"{basename}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def _individual_figure(
    experiment: str,
    spec: ExperimentSpec,
    rows: list[dict[str, str]],
    floors: dict,
    summary: dict[str, dict[str, float]],
    metric: str,
    axis: str,
    png_dir: Path,
    pdf_dir: Path,
) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    if metric in TIME_SERIES_METRICS:
        _plot_time_series(ax, rows, spec, metric, axis, floors)
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
    return _save_figure(
        fig,
        f"{experiment}_{metric}_{axis}",
        png_dir,
        pdf_dir,
    )


def _combined_figure(
    experiment: str,
    spec: ExperimentSpec,
    rows: list[dict[str, str]],
    floors: dict,
    summary: dict[str, dict[str, float]],
    axis: str,
    png_dir: Path,
    pdf_dir: Path,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    flat = axes.ravel()
    for panel, metric in zip(flat[:3], TIME_SERIES_METRICS):
        _plot_time_series(panel, rows, spec, metric, axis, floors)
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
    return _save_figure(
        fig,
        f"{experiment}_combined_{axis}",
        png_dir,
        pdf_dir,
    )


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
    png_dir = figures_dir / "png"
    pdf_dir = figures_dir / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

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
                    png_dir,
                    pdf_dir,
                ))
            outputs.extend(_combined_figure(
                experiment,
                spec,
                rows,
                floors,
                summary,
                axis,
                png_dir,
                pdf_dir,
            ))
        per_experiment[experiment] = {
            "methods": list(spec.methods),
            "labels": spec.labels,
            "stationarity_methods": sorted(summary),
        }

    expected_per_format = len(EXPERIMENTS) * len(AXES) * (len(METRICS) + 1)
    # Validate the files generated by this function, not every file already in
    # the shared manuscript directories. Density/scatter figures intentionally
    # coexist there when ``--no-clean`` is used.
    png_outputs = sorted(path for path in outputs if path.suffix == ".png")
    pdf_outputs = sorted(path for path in outputs if path.suffix == ".pdf")
    if len(png_outputs) != expected_per_format:
        raise RuntimeError(
            f"expected {expected_per_format} PNGs, found {len(png_outputs)}"
        )
    if len(pdf_outputs) != expected_per_format:
        raise RuntimeError(
            f"expected {expected_per_format} PDFs, found {len(pdf_outputs)}"
        )
    return {
        "results_dir": str(results_dir),
        "figures_dir": str(figures_dir),
        "png_count": len(png_outputs),
        "pdf_count": len(pdf_outputs),
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
        help="output directory; PNG/PDF subdirectories are created here",
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
    print(
        "Regenerated manuscript figures: "
        f"{result['png_count']} PNG + {result['pdf_count']} PDF in "
        f"{result['figures_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
