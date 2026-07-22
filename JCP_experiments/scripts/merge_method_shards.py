"""Merge the per-method CSV shards of one experiment into a single artifact set.

E5's exact deterministic-quadrature arm (``LSC-CP``) costs ~25 h at the
production ensemble -- roughly eight times the paired multi-atom arm -- so the
eight-method matrix is run as two concurrent *method shards* on separate GPUs
rather than as one serial job. Each shard is a complete, gated production run of
its own methods: it passes the same unit tests, the same bounded dynamics smoke
(``scripts/smoke_experiment.py``, which permits a shard to narrow the matrix but
never to widen it), and writes the same artifacts.

What a shard is NOT is a complete experiment. This script is the step that makes
it one, and it is deliberately fail-closed:

  * the union of the shards' methods must equal the registered method matrix --
    a missing method aborts the merge rather than silently producing a summary
    with a hole in it;
  * no method may appear in two shards (an overlap would double-count seeds);
  * every shard must have finished with status ``ok``;
  * the shards must agree on git commit, seeds, and per-method protocol, since a
    merged CSV whose rows came from different code or different seed sets is not
    a comparison at all.

Provenance for the merged run records every source run-id, so the merged
artifact can always be traced back to the two gated runs that produced it.

Usage:
    python -m scripts.merge_method_shards \
        --experiment alanine_dipeptide \
        --shards RUNID-e5EXACT RUNID-e5REST \
        --out-run-id RUNID-e5MERGED
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
JCP_ROOT = HERE.parent
sys.path.insert(0, str(JCP_ROOT))

RESULTS_ROOT = JCP_ROOT.parent / "results" / "jcp_sampling"
# Row-level CSVs: concatenated across shards, keyed by the "method" column.
MERGED_CSVS = ("metrics_timeseries.csv", "summary.csv", "positions.csv")
# Shard-level agreement: these must be identical across shards or the merge is
# comparing runs that are not actually comparable.
PINNED_PLAN_KEYS = ("registered_methods", "experiments", "smoke_config")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _artifacts_dir(run_dir: Path, experiment: str) -> Path:
    return run_dir / experiment / "artifacts"


def _read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _methods_in(rows: list[dict]) -> set[str]:
    return {row["method"] for row in rows if row.get("method")}


def merge(experiment: str, shard_ids: list[str], out_run_id: str,
          results_root: Path = RESULTS_ROOT) -> dict:
    if len(shard_ids) < 2:
        raise ValueError("merging needs at least two shards")
    if len(set(shard_ids)) != len(shard_ids):
        raise ValueError(f"duplicate shard run-ids: {shard_ids}")

    plans, statuses = {}, {}
    for rid in shard_ids:
        run_dir = results_root / rid
        plan_path = run_dir / "launch_plan.json"
        status_path = run_dir / "status.json"
        if not plan_path.exists():
            raise FileNotFoundError(f"{rid}: no launch_plan.json at {plan_path}")
        if not status_path.exists():
            raise FileNotFoundError(f"{rid}: no status.json -- shard unfinished")
        plans[rid] = json.loads(plan_path.read_text(encoding="utf-8"))
        statuses[rid] = json.loads(status_path.read_text(encoding="utf-8"))
        if statuses[rid].get("status") != "ok":
            raise ValueError(
                f"{rid}: shard status is {statuses[rid].get('status')!r}, not 'ok'; "
                f"failure_phase={statuses[rid].get('failure_phase')!r}. "
                "A failed shard is never merged.")

    # --- the shards must describe the same experiment, code and protocol ----
    head = plans[shard_ids[0]]
    for rid in shard_ids[1:]:
        for key in PINNED_PLAN_KEYS:
            if plans[rid].get(key) != head.get(key):
                raise ValueError(
                    f"shards disagree on {key!r}: "
                    f"{shard_ids[0]}={head.get(key)!r} vs {rid}={plans[rid].get(key)!r}")
        commits = (head.get("git", {}).get("commit"),
                   plans[rid].get("git", {}).get("commit"))
        if commits[0] != commits[1]:
            raise ValueError(
                f"shards were run at different commits {commits}; a merged CSV "
                "whose rows came from different code is not a comparison")

    registered = head.get("registered_methods", {}).get(experiment)
    if not registered:
        raise ValueError(
            f"launch_plan.json records no registered_methods for {experiment!r}; "
            "rerun the shards with a launcher that records shard provenance")
    registered_set = set(registered.split(","))

    # --- union-of-shards coverage, the check the smoke gate delegates here --
    shard_methods: dict[str, set[str]] = {}
    for rid in shard_ids:
        declared = plans[rid].get("methods", {}).get(experiment)
        if not declared:
            raise ValueError(f"{rid}: launch_plan.json records no methods for "
                             f"{experiment!r}")
        shard_methods[rid] = set(declared.split(","))

    seen: dict[str, str] = {}
    for rid, methods in shard_methods.items():
        for method in methods:
            if method in seen:
                raise ValueError(
                    f"method {method!r} appears in both {seen[method]} and {rid}; "
                    "overlapping shards would double-count seeds")
            seen[method] = rid

    union = set().union(*shard_methods.values())
    missing = sorted(registered_set - union)
    extra = sorted(union - registered_set)
    if missing or extra:
        raise ValueError(
            "the union of shards must equal the registered method matrix; "
            f"missing={missing}, extra={extra}. A merged artifact is only "
            "emitted for complete coverage.")

    # --- merge the row-level CSVs ------------------------------------------
    out_dir = _artifacts_dir(results_root / out_run_id, experiment)
    out_dir.mkdir(parents=True, exist_ok=False)
    merged_counts: dict[str, dict] = {}

    for csv_name in MERGED_CSVS:
        fieldnames: list[str] | None = None
        rows_out: list[dict] = []
        per_shard: dict[str, int] = {}
        for rid in shard_ids:
            path = _artifacts_dir(results_root / rid, experiment) / csv_name
            if not path.exists():
                raise FileNotFoundError(f"{rid}: missing {csv_name}")
            names, rows = _read_csv(path)
            if fieldnames is None:
                fieldnames = names
            elif names != fieldnames:
                raise ValueError(
                    f"{csv_name}: shards disagree on columns; "
                    f"{shard_ids[0]}={fieldnames} vs {rid}={names}")
            # a shard's CSV must contain exactly the methods it declared
            present = _methods_in(rows)
            if present != shard_methods[rid]:
                raise ValueError(
                    f"{rid}/{csv_name}: rows cover {sorted(present)} but the "
                    f"shard declared {sorted(shard_methods[rid])}")
            rows_out.extend(rows)
            per_shard[rid] = len(rows)

        with (out_dir / csv_name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)
        merged_counts[csv_name] = {"rows": len(rows_out), "per_shard": per_shard}

    # --- non-row artifacts: copied per shard, never silently collapsed -----
    for rid in shard_ids:
        src = _artifacts_dir(results_root / rid, experiment)
        for item in sorted(src.iterdir()):
            if item.name in MERGED_CSVS:
                continue
            dest = out_dir / "per_shard" / rid / item.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    provenance = {
        "schema_version": 1,
        "kind": "merged_method_shards",
        "experiment": experiment,
        "run_id": out_run_id,
        "created_at_utc": _utc_now(),
        "source_run_ids": list(shard_ids),
        "shard_methods": {rid: sorted(methods)
                          for rid, methods in shard_methods.items()},
        "registered_methods": sorted(registered_set),
        "coverage_complete": True,
        "git": head.get("git"),
        "merged_csv_counts": merged_counts,
        "source_status": {rid: statuses[rid].get("status") for rid in shard_ids},
    }
    (out_dir / "merge_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--shards", nargs="+", required=True,
                        help="run-ids of the shards to merge")
    parser.add_argument("--out-run-id", required=True)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args(argv)
    provenance = merge(args.experiment, list(args.shards), args.out_run_id,
                       args.results_root)
    print(json.dumps(provenance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
