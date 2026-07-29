"""Validate the portable E1--E4 manuscript release without running a GPU job."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import nbformat
import yaml


HERE = Path(__file__).resolve().parent
JCP_ROOT = HERE.parent
if str(JCP_ROOT) not in sys.path:
    sys.path.insert(0, str(JCP_ROOT))

from src.manuscript import (  # noqa: E402
    EXPERIMENTS,
    METRICS,
    RESOURCE_AXES,
)


class ReleaseValidationError(RuntimeError):
    pass


def _require(path: Path, kind: str = "file") -> None:
    predicate = path.is_dir if kind == "directory" else path.is_file
    if not predicate():
        raise ReleaseValidationError(f"missing required {kind}: {path}")


def _csv_methods(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "method" not in reader.fieldnames:
            raise ReleaseValidationError(f"{path} has no method column")
        return {row["method"] for row in reader if row.get("method")}


def _reject_wallclock_keys(value, *, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if "wallclock" in normalized or normalized.endswith("_per_second"):
                raise ReleaseValidationError(
                    f"{path}: wall-clock result key remains: {key!r}"
                )
            _reject_wallclock_keys(child, path=path)
    elif isinstance(value, list):
        for child in value:
            _reject_wallclock_keys(child, path=path)


def _reject_wallclock_columns(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        columns = set(csv.DictReader(handle).fieldnames or ())
    forbidden = {
        column for column in columns
        if "wallclock" in column.lower() or column.endswith("_per_second")
    }
    if forbidden:
        raise ReleaseValidationError(
            f"{path}: wall-clock-derived columns remain: {sorted(forbidden)}"
        )


def _validate_config(root: Path, key: str) -> dict:
    spec = EXPERIMENTS[key]
    path = root / "configs" / spec.config
    _require(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReleaseValidationError(f"{path} is not a YAML mapping")
    if payload.get("experiment") != key:
        raise ReleaseValidationError(
            f"{path}: experiment={payload.get('experiment')!r}, expected {key!r}"
        )
    methods = tuple((payload.get("methods") or {}).keys())
    if methods != spec.methods:
        raise ReleaseValidationError(
            f"{path}: method order {methods} != release order {spec.methods}"
        )
    if tuple(payload.get("resource_axes") or ()) != RESOURCE_AXES:
        raise ReleaseValidationError(f"{path}: incorrect resource_axes")
    configured_metrics = tuple(payload.get("metrics") or ())
    expected_metrics = ("W2", "MMD", "basin_TV", "worst_basin_ESS")
    if configured_metrics != expected_metrics:
        raise ReleaseValidationError(
            f"{path}: metrics {configured_metrics} != {expected_metrics}"
        )
    return payload


def _validate_notebook(root: Path, key: str) -> dict:
    spec = EXPERIMENTS[key]
    path = root / "notebooks" / spec.notebook
    _require(path)
    notebook = nbformat.read(path, as_version=4)
    if not notebook.cells:
        raise ReleaseValidationError(f"{path} contains no cells")
    source = "\n".join(
        "".join(cell.get("source", "")) for cell in notebook.cells
    )
    if spec.number not in source and key not in source:
        raise ReleaseValidationError(
            f"{path} does not identify {spec.number}/{key}"
        )
    forbidden = ("/home/", "/Users/", "C:\\\\")
    hits = [token for token in forbidden if token in source]
    if hits:
        raise ReleaseValidationError(
            f"{path} contains non-portable absolute path token(s): {hits}"
        )
    return {
        "path": str(path.relative_to(root)),
        "cells": len(notebook.cells),
        "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
    }


def _validate_results(root: Path, key: str) -> dict:
    spec = EXPERIMENTS[key]
    directory = root / "results" / key
    _require(directory, "directory")
    required = (
        directory / "manifest.json",
        directory / "metrics_timeseries.csv",
        directory / "summary.csv",
        directory / "positions.csv",
    )
    for path in required:
        _require(path)

    manifest = json.loads(required[0].read_text(encoding="utf-8"))
    _reject_wallclock_keys(manifest, path=str(required[0]))
    for csv_path in required[1:]:
        _reject_wallclock_columns(csv_path)
    config = manifest.get("config") or {}
    for field in ("d", "N", "T", "dt", "beta", "seeds"):
        if field not in config:
            raise ReleaseValidationError(
                f"{required[0]}: config is missing {field!r}"
            )

    with required[1].open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        expected_columns = {
            "method", "seed", "t", "nfe", *METRICS[:3]
        }
        missing_columns = expected_columns - columns
        if missing_columns:
            raise ReleaseValidationError(
                f"{required[1]} is missing columns {sorted(missing_columns)}"
            )
        time_series_methods = {
            row["method"] for row in reader if row.get("method")
        }
    missing_methods = set(spec.methods) - time_series_methods
    if missing_methods:
        raise ReleaseValidationError(
            f"{required[1]} is missing release methods {sorted(missing_methods)}"
        )

    position_methods = _csv_methods(required[3])
    missing_positions = set(spec.methods) - position_methods
    if missing_positions:
        raise ReleaseValidationError(
            f"{required[3]} is missing release methods {sorted(missing_positions)}"
        )

    # FLA and E1 Raw-CP are non-target-preserving mixing diagnostics. Their ESS
    # is still required, but must be interpreted alongside distributional bias.
    ess_methods = set(spec.methods)
    stationarity_dir = directory / "stationarity"
    _require(stationarity_dir, "directory")
    _reject_wallclock_columns(stationarity_dir / "all_methods_summary.csv")
    for method in ess_methods:
        summary = stationarity_dir / f"{method}_summary.csv"
        _require(summary)
        _reject_wallclock_columns(summary)
        with summary.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "worst_basin_ess" not in set(reader.fieldnames or ()):
                raise ReleaseValidationError(
                    f"{summary} has no worst_basin_ess column"
                )
            first = next(reader, None)
            if first is None or not first.get("worst_basin_ess"):
                raise ReleaseValidationError(
                    f"{summary} has no worst-basin ESS value"
                )

    return {
        "directory": str(directory.relative_to(root)),
        "release_methods": list(spec.methods),
        "extra_frozen_methods": sorted(time_series_methods - set(spec.methods)),
        "ess_methods": sorted(ess_methods),
    }


def validate_release(
    root: Path = JCP_ROOT,
    *,
    check_results: bool = True,
    require_figures: bool = False,
) -> dict:
    root = root.resolve()
    legacy_root_entries = (
        "JCP_experiments",
        "archive",
        "archive 2",
        "doublewell.ipynb",
        "doublewell_output",
        "experiments",
        "experiments_CY",
        "experiment_note",
        "manywell.ipynb",
        "manywell_output",
        "mog40.ipynb",
        "mog40_output",
        "paper.mplstyle",
        "reports",
        "tests",
        "tests_cpu",
    )
    remaining_legacy = [
        name for name in legacy_root_entries if (root / name).exists()
    ]
    if remaining_legacy:
        raise ReleaseValidationError(
            "release must live at repository root; legacy root entries remain: "
            + ", ".join(remaining_legacy)
        )
    for unrelated in (
        root / "notebooks" / "05_alanine_dipeptide.ipynb",
        root / "results" / "alanine_dipeptide",
        root / "scripts" / "smoke_experiment.py",
        root / "src" / "e5_alanine",
    ):
        if unrelated.exists():
            raise ReleaseValidationError(
                f"unrelated non-E1--E4 content remains: {unrelated}"
            )
    for path in (
        root / "README.md",
        root / "environment.yml",
        root / "pyproject.toml",
        root / "src" / "manuscript.py",
        root / "notebooks" / "00_environment_check.ipynb",
        root / "notebooks" / "05_manuscript_plotting.ipynb",
    ):
        _require(path)

    report = {
        "status": "passed",
        "root": str(root),
        "experiments": {},
        "checked_results": check_results,
        "required_figures": require_figures,
    }
    for key in EXPERIMENTS:
        item = {
            "config": _validate_config(root, key),
            "notebook": _validate_notebook(root, key),
        }
        if check_results:
            item["results"] = _validate_results(root, key)
        report["experiments"][key] = item

    if require_figures:
        for extension in ("png", "pdf"):
            directory = root / "figures" / extension
            _require(directory, "directory")
            forbidden = sorted(directory.glob("*wallclock*"))
            if forbidden:
                raise ReleaseValidationError(
                    "wall-clock figures remain: "
                    + ", ".join(str(path) for path in forbidden)
                )
            for key in EXPERIMENTS:
                for axis in RESOURCE_AXES:
                    _require(directory / f"{key}_combined_{axis}.{extension}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=JCP_ROOT)
    parser.add_argument(
        "--skip-results",
        action="store_true",
        help="validate code/config/notebook structure without frozen results",
    )
    parser.add_argument(
        "--require-figures",
        action="store_true",
        help="also require all combined PNG/PDF manuscript figures",
    )
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_release(
        args.root,
        check_results=not args.skip_results,
        require_figures=args.require_figures,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(
        "Release validation PASSED: "
        f"{len(report['experiments'])} experiments; "
        f"results={'yes' if report['checked_results'] else 'no'}; "
        f"figures={'required' if report['required_figures'] else 'not required'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
