"""Regenerate figures from one immutable JCP experiment artifact directory.

The command is intentionally path-explicit and CPU-only.  It never searches a
legacy ``results/<experiment>`` tree and never rebuilds an experiment on a GPU.
The input directory must contain ``metrics_timeseries.csv`` and ``manifest.json``
from a completed notebook run; output is written to a new directory.

Example::

    python scripts/replot_figures.py \
      --experiment double_well \
      --artifacts-dir ../results/jcp_sampling/<run-id>/double_well/artifacts \
      --output-dir /tmp/double-well-replot
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
_JCP = _HERE.parent
sys.path.insert(0, str(_JCP))

_EXPERIMENTS = ("double_well", "mog40", "mb3well_10d", "coupled_phi4",
                "alanine_dipeptide")
_SINGLE = (
    "W2", "TV", "TV_density", "MMD",
    "FES_RMSE_kBT", "FES_outside_mass", "basin_KL_target",
    "e_F",  # legacy metric column, suppressed when FES_RMSE_kBT is present
    "basin_rel_max", "KSD", "W1_cdf", "CDF_sup", "pdf_L1", "KDE_chi2",
    "W2_10d",
)


def _plot_policy(experiment: str,
                 in_data: set[str]) -> tuple[list[str], dict[str, str]]:
    """Use exact LSC in low dimension and paired MA in E3/E4.

    Random-atomic LSC is a secondary estimator and is never silently relabeled
    as the preferred method; it is shown explicitly only as a compatibility
    fallback when the experiment's preferred estimator is absent.
    """
    low_dim_exact = experiment in ("double_well", "mog40")
    preferred = "LSC-CP" if low_dim_exact else "LSC-CP-MA"
    lsc = preferred if preferred in in_data else (
        "LSC-CP-RA" if "LSC-CP-RA" in in_data else None)
    raw = "CP" if "CP" in in_data else ("CP-RA" if "CP-RA" in in_data else None)
    methods = [m for m in ("ULA", "MALA", "FLA", "BAOAB", "PT")
               if m in in_data]
    label_overrides: dict[str, str] = {}
    if raw:
        methods.append(raw)
        label_overrides[raw] = "Raw-CP"
    if lsc:
        methods.append(lsc)
        label_overrides[lsc] = (
            "LSC-CP-RA (secondary)" if lsc == "LSC-CP-RA" else "LSC-CP")
    return methods, label_overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=_EXPERIMENTS)
    parser.add_argument(
        "--artifacts-dir", required=True, type=Path,
        help="immutable .../<run-id>/<experiment>/artifacts directory",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="new directory for regenerated PNG/PDF figures (must not exist)",
    )
    return parser


def replot(experiment: str, artifacts_dir: Path, output_dir: Path) -> dict:
    """Replot from CSV+manifest only and return a small result manifest."""
    artifacts_dir = artifacts_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    csv_path = artifacts_dir / "metrics_timeseries.csv"
    manifest_path = artifacts_dir / "manifest.json"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing metrics CSV: {csv_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")

    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest_experiment = manifest.get("experiment")
    if manifest_experiment != experiment:
        raise ValueError(
            f"manifest experiment {manifest_experiment!r} does not match "
            f"--experiment {experiment!r}"
        )
    floors = manifest.get("bias_floors")
    emc = manifest.get("emc_target")
    if not isinstance(floors, dict) or emc is None:
        raise ValueError(
            "manifest must contain bias_floors and emc_target; GPU fallback "
            "and legacy path reconstruction are intentionally disabled"
        )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"metrics CSV is empty: {csv_path}")
    if any("method" not in row for row in rows):
        raise ValueError("metrics CSV has no method column")

    in_data = {row["method"] for row in rows}
    plot = manifest.get("plot") or {}
    methods = [method for method in plot.get("methods", []) if method in in_data]
    label_overrides = plot.get("label_overrides") or {}
    if not methods:
        methods, label_overrides = _plot_policy(experiment, in_data)
    if not methods:
        raise ValueError("no plottable methods found in metrics CSV")

    # A replot is a new derived artifact, not an in-place mutation of the
    # immutable production directory.
    output_dir.mkdir(parents=True, exist_ok=False)
    from src.plotting import metric_grid, metric_single  # matplotlib only

    present = set().union(*(set(row) for row in rows))
    single = [
        metric for metric in _SINGLE
        if metric in present
        and not (metric == "e_F" and "FES_RMSE_kBT" in present)
    ]
    for metric in single:
        for axis in ("t", "nfe", "wallclock"):
            metric_single(
                rows, metric, str(output_dir / f"{experiment}_{metric}_{axis}"),
                xaxis=axis, floors=floors, methods=methods,
                emc_target=float(emc), show=False,
                label_overrides=label_overrides,
            )
    metric_grid(
        rows, str(output_dir / f"{experiment}_metrics"),
        metrics=("W2", "MMD", "EMC"), floors=floors,
        emc_target=float(emc), methods=methods, show=False,
        label_overrides=label_overrides,
    )
    return {
        "experiment": experiment,
        "artifacts_dir": str(artifacts_dir),
        "output_dir": str(output_dir),
        "single_metric_count": len(single),
        "methods": methods,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = replot(args.experiment, args.artifacts_dir, args.output_dir)
    print(
        f"replotted {result['experiment']}: "
        f"{result['single_metric_count']} metrics x 3 axes + grid into "
        f"{result['output_dir']} (methods={result['methods']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
