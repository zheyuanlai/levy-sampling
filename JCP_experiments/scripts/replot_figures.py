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


_RA_LABEL = {"mb3well_10d": "LSC-CP-RA (4)", "coupled_phi4": "LSC-CP-RA (8)"}


def _plot_policy(experiment: str,
                 in_data: set[str]) -> tuple[list[str], dict[str, str]]:
    """Plot BOTH LSC arms: the exact deterministic-quadrature score and the
    realised-displacement estimator (single-atom RA on E1/E2, atom-stratified MA
    on E3/E4, labelled with its atom count). Must stay in sync with the notebook
    generator's CELL_FIGURES policy.
    """
    raw = "CP" if "CP" in in_data else ("CP-RA" if "CP-RA" in in_data else None)
    methods = [m for m in ("ULA", "MALA", "FLA", "BAOAB", "PT")
               if m in in_data]
    label_overrides: dict[str, str] = {}
    if raw:
        methods.append(raw)
        label_overrides[raw] = "Raw-CP"
    if "LSC-CP" in in_data:
        methods.append("LSC-CP")
        label_overrides["LSC-CP"] = "LSC-CP"
    for ra in ("LSC-CP-RA", "LSC-CP-MA"):
        if ra in in_data:
            methods.append(ra)
            label_overrides[ra] = _RA_LABEL.get(experiment, "LSC-CP-RA")
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
    parser.add_argument(
        "--use-current-policy", action="store_true",
        help=("ignore the manifest's recorded plot policy and apply the current "
              "one. Needed for runs that predate the two-LSC-arm policy: their "
              "CSVs contain both arms but their manifest names only one."),
    )
    return parser


def replot(experiment: str, artifacts_dir: Path, output_dir: Path, *,
           use_current_policy: bool = False) -> dict:
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
    if use_current_policy:
        methods, label_overrides = _plot_policy(experiment, in_data)
    if not methods:
        methods, label_overrides = _plot_policy(experiment, in_data)
    # A run predating the two-arm policy recorded only one LSC arm in its
    # manifest, but its CSVs hold both. Say so rather than silently dropping one.
    missing = sorted({m for m in in_data if m.startswith("LSC-CP")} - set(methods))
    if missing:
        print(f"note: {missing} present in the CSV but absent from the plot "
              "policy; pass --use-current-policy to plot both LSC arms")
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
    # Sample-space figures come from positions.csv alone. Older artifact
    # directories predate that file; skip rather than fail, and say so.
    from src.plotting import (load_positions_csv, density_overlay, fes_ceiling,
                              fes_profile_1d, fes_map_2d, REFERENCE_KEY)
    positions_path = artifacts_dir / "positions.csv"
    sample_figures = 0
    if not positions_path.is_file():
        print(f"note: {positions_path.name} absent -- skipping density/FES "
              "figures (artifacts predate sample persistence)")
    else:
        beta = ((manifest.get("config") or {}).get("beta"))
        if beta is None:
            raise ValueError("manifest.config.beta is required for FES figures")
        pos = load_positions_csv(str(positions_path))
        if REFERENCE_KEY not in pos:
            raise ValueError(f"{positions_path} has no '{REFERENCE_KEY}' block")
        plot_pos = [m for m in methods if m in pos]
        for method in plot_pos:
            density_overlay(pos, method,
                            str(output_dir / f"{experiment}_density_{method}"),
                            label_overrides=label_overrides, show=False)
            sample_figures += 1
        if pos[REFERENCE_KEY].shape[1] == 1:
            fes_profile_1d(pos, str(output_dir / f"{experiment}_FES_profile"),
                           beta=float(beta), methods=plot_pos,
                           label_overrides=label_overrides, show=False)
            sample_figures += 1
        else:
            fmax = fes_ceiling(pos, beta=float(beta))
            for method in [REFERENCE_KEY] + plot_pos:
                fes_map_2d(pos, method,
                           str(output_dir / f"{experiment}_FES_{method}"),
                           beta=float(beta), fmax=fmax,
                           label_overrides=label_overrides, show=False)
                sample_figures += 1
    return {
        "experiment": experiment,
        "artifacts_dir": str(artifacts_dir),
        "output_dir": str(output_dir),
        "single_metric_count": len(single),
        "sample_figure_count": sample_figures,
        "methods": methods,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = replot(args.experiment, args.artifacts_dir, args.output_dir,
                    use_current_policy=args.use_current_policy)
    print(
        f"replotted {result['experiment']}: "
        f"{result['single_metric_count']} metrics x 3 axes + grid "
        f"+ {result['sample_figure_count']} sample-space figures into "
        f"{result['output_dir']} (methods={result['methods']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
