"""Derived index over completed run directories.

``catalog.csv`` is never maintained by a worker. It is rebuilt at any time by
scanning manifests, so a lost or stale catalog costs nothing and concurrent runs
never contend for it. Plot notebooks may use the catalog or scan the manifests
directly; both give the same answer.

Only runs that pass ``src.results.verify_run`` are admitted: manifest present,
``COMPLETE`` present, schema version understood, file hashes matching, and not
marked invalid.
"""
from __future__ import annotations

from pathlib import Path

from .results import (MANIFEST_NAME, RunPaths, read_manifest, slugify,
                      verify_run)

#: Columns the catalog always carries, in this order. Any additional manifest
#: scalars are appended alphabetically.
CORE_COLUMNS = (
    "experiment_id",
    "method",
    "variant_label",
    "run_id",
    "tame",
    "tame_cap",
    "dt",
    "particles",
    "n_seeds",
    "schema_version",
    "status",
    "target_hash",
    "reference_hash",
    "calibration_hash",
    "variant_hash",
    "rng_pair_group_hash",
    "fee_calibration_hash",
    "fee_cost_unit",
    "device_type",
    "dtype",
    "written_at_utc",
    "has_stationarity",
    "is_latest_for_variant",
    "run_directory",
)


def iter_run_directories(experiment_dir: Path):
    """Yield every candidate run directory under ``<experiment>/runs``."""
    runs_dir = Path(experiment_dir) / "runs"
    if not runs_dir.is_dir():
        return
    for method_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            if run_dir.name.startswith(".tmp-"):
                continue
            yield run_dir


def scan(experiment_dir: Path, *, check_hashes: bool = True,
         collect_rejections: bool = False):
    """Return ``(rows, rejections)`` for one experiment directory."""
    rows, rejections = [], []
    for run_dir in iter_run_directories(experiment_dir):
        admissible, reason = verify_run(run_dir, check_hashes=check_hashes)
        if not admissible:
            rejections.append({"run_directory": str(run_dir), "reason": reason})
            continue
        manifest = read_manifest(run_dir)
        rows.append(_row_from_manifest(manifest, run_dir))
    _mark_latest(rows)
    rows.sort(key=lambda row: (str(row.get("method")),
                               str(row.get("variant_label")),
                               str(row.get("written_at_utc"))))
    if collect_rejections:
        return rows, rejections
    return rows, rejections


def _row_from_manifest(manifest: dict, run_dir: Path) -> dict:
    row = {
        "experiment_id": manifest.get("experiment_id"),
        "method": manifest.get("method"),
        "variant_label": manifest.get("variant_label"),
        "run_id": manifest.get("run_id"),
        "schema_version": manifest.get("schema_version"),
        "status": manifest.get("status", "complete"),
        "target_hash": manifest.get("target_hash"),
        "reference_hash": manifest.get("reference_hash"),
        "calibration_hash": manifest.get("calibration_hash"),
        "variant_hash": manifest.get("variant_hash"),
        "rng_pair_group_hash": manifest.get("rng_pair_group_hash"),
        "fee_calibration_hash": manifest.get("fee_calibration_hash"),
        "fee_cost_unit": manifest.get("fee_cost_unit"),
        "written_at_utc": manifest.get("written_at_utc"),
        "run_directory": str(run_dir),
        "has_stationarity": bool(
            (run_dir / "stationarity.npz").is_file()),
    }
    parameters = manifest.get("parameters") or {}
    row["tame"] = parameters.get("tame", manifest.get("tame"))
    row["tame_cap"] = manifest.get("tame_cap")
    row["dt"] = manifest.get("dt")
    row["particles"] = manifest.get("particles")
    seeds = manifest.get("seeds") or []
    row["n_seeds"] = len(seeds) if isinstance(seeds, (list, tuple)) else None
    provenance = manifest.get("device_provenance") or {}
    row["device_type"] = provenance.get("device_type")
    row["dtype"] = provenance.get("dtype")
    # Method hyperparameters become columns so a plot config can select on them
    # without opening every manifest.
    for key, value in parameters.items():
        if key == "tame":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[f"param_{key}"] = value
    return row


def _mark_latest(rows: list[dict]) -> None:
    """Flag the newest admissible run for each (method, variant) pair."""
    newest: dict[tuple, str] = {}
    for row in rows:
        key = (row.get("method"), row.get("variant_hash"))
        stamp = str(row.get("written_at_utc") or "")
        if key not in newest or stamp > newest[key]:
            newest[key] = stamp
    for row in rows:
        key = (row.get("method"), row.get("variant_hash"))
        row["is_latest_for_variant"] = (
            str(row.get("written_at_utc") or "") == newest.get(key))


def catalog_columns(rows: list[dict]) -> list[str]:
    extras = sorted({key for row in rows for key in row}
                    - set(CORE_COLUMNS))
    return list(CORE_COLUMNS) + extras


def write_catalog(experiment_dir: Path, *, check_hashes: bool = True) -> dict:
    """Rebuild ``catalog.csv`` from the manifests under one experiment."""
    import csv

    experiment_dir = Path(experiment_dir)
    rows, rejections = scan(experiment_dir, check_hashes=check_hashes)
    path = experiment_dir / "catalog.csv"
    columns = catalog_columns(rows)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="",
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return {"catalog": str(path), "n_runs": len(rows),
            "n_rejected": len(rejections), "rejections": rejections}


def load_catalog(experiment_dir: Path) -> list[dict]:
    """Read ``catalog.csv``, rebuilding it from manifests when it is absent."""
    import csv

    path = Path(experiment_dir) / "catalog.csv"
    if not path.is_file():
        write_catalog(experiment_dir)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_runs(experiment_dir: Path, *, method: str | None = None,
                variant_label: str | None = None, tame: bool | None = None,
                latest_only: bool = True, from_manifests: bool = False,
                **parameter_filters) -> list[dict]:
    """Pick runs by method, variant, tame flag, and hyperparameter values.

    ``from_manifests=True`` bypasses the catalog and scans manifests directly,
    which is what a plot notebook does when it wants to be independent of any
    derived index.
    """
    if from_manifests:
        rows, _ = scan(experiment_dir)
    else:
        rows = load_catalog(experiment_dir)

    def as_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("true", "1", "yes")

    selected = []
    for row in rows:
        if method is not None and row.get("method") != method:
            continue
        if variant_label is not None and row.get("variant_label") != variant_label:
            continue
        if tame is not None and as_bool(row.get("tame")) != bool(tame):
            continue
        if latest_only and not as_bool(row.get("is_latest_for_variant")):
            continue
        if any(str(row.get(f"param_{key}")) != str(value)
               for key, value in parameter_filters.items()):
            continue
        selected.append(row)
    return selected


def rebuild_all(results_root: Path) -> list[dict]:
    """Rebuild the catalog for every experiment directory under a results root."""
    results_root = Path(results_root)
    reports = []
    for experiment_dir in sorted(p for p in results_root.iterdir()
                                 if p.is_dir() and (p / "runs").is_dir()):
        report = write_catalog(experiment_dir)
        report["experiment_dir"] = str(experiment_dir)
        reports.append(report)
    return reports
