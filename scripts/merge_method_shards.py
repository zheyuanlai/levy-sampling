"""Merge the per-method CSV shards of one E1--E4 experiment.

Method sharding lets expensive methods run concurrently on separate GPUs.
Each shard is a complete production run of its own methods: it passes release
validation and writes the same artifacts.

What a shard is NOT is a complete experiment. This script is the step that makes
it one, and it is deliberately fail-closed:

  * the union of the shards' methods must equal the registered method matrix --
    a missing method aborts the merge rather than silently producing a summary
    with a hole in it;
  * no method may appear in two shards (an overlap would double-count seeds);
  * every shard must have finished with status ``success`` (a data-complete
    shard whose trailing non-data cell crashed is salvaged, see below);
  * the shards must agree on git commit, seeds, and per-method protocol, since a
    merged CSV whose rows came from different code or different seed sets is not
    a comparison at all.

Provenance for the merged run records every source run-id, so the merged
artifact can always be traced back to the two gated runs that produced it.

Usage:
    python -m scripts.merge_method_shards \
        --experiment mb3well_10d \
        --shards RUNID-e3A RUNID-e3B \
        --out-run-id RUNID-e3MERGED
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
JCP_ROOT = HERE.parent
sys.path.insert(0, str(JCP_ROOT))

RESULTS_ROOT = JCP_ROOT / "results" / "jcp_sampling"
# Row-level CSVs: concatenated across shards, keyed by the "method" column.
MERGED_CSVS = ("metrics_timeseries.csv", "summary.csv", "positions.csv")
# Blocks in the "method" column that belong to no shard. positions.csv carries
# the reference draw as a pseudo-method block, so every shard writes its own
# copy: concatenating blindly would triplicate it AND make each shard look like
# it covered a method it never ran. The copies must be identical -- they come
# from the same fixed reference seed -- and a divergence means the shards were
# not comparing against the same reference, which invalidates the merge.
SHARED_BLOCKS = frozenset({"reference"})
# Shard-level agreement: these must be identical across shards or the merge is
# comparing runs that are not actually comparable.
PINNED_PLAN_KEYS = ("registered_methods", "experiments")
# The numerical engine: every file whose bytes determine a method's produced
# rows (sampler dynamics, Levy score, jump law, energy, metrics, experiment
# wiring, run config). Two shards at different commits are mergeable iff these
# are byte-identical at both commits -- see the commit check in merge().
NUMERIC_ENGINE_FILES = (
    "src/samplers.py",
    "src/score.py",
    "src/jumps.py",
    "src/potentials.py",
    "src/metrics.py",
    "src/experiments.py",
    "src/config.py",
)


def _git_blob(commit: str, repo_path: str) -> bytes | None:
    """Bytes of ``repo_path`` at ``commit``; None if absent at that commit."""
    result = subprocess.run(
        ["git", "-C", str(JCP_ROOT), "show", f"{commit}:{repo_path}"],
        capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else None


def _numeric_engine_diff(commit_a: str, commit_b: str) -> set[str]:
    """Engine files whose bytes differ between the two commits (empty == same).

    A file missing at one commit but present at the other counts as differing.
    Raises if git cannot resolve a commit, since silently treating an
    unresolvable commit as 'identical' would defeat the check.
    """
    differing = set()
    for path in NUMERIC_ENGINE_FILES:
        a, b = _git_blob(commit_a, path), _git_blob(commit_b, path)
        if a is None and b is None:
            raise ValueError(
                f"{path} is absent at both {commit_a} and {commit_b}; cannot "
                "verify numerical-engine equality")
        if a != b:
            differing.add(path)
    return differing


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

    plans, statuses, salvaged = {}, {}, {}
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
        if statuses[rid].get("status") != "success":
            # The merge combines row-level data. A shard whose DATA is complete
            # is usable even if a trailing non-data cell crashed -- e.g. a shard
            # that ran every method's dynamics and wrote every row CSV, then died
            # in the manifest/provenance cell. So instead of trusting the coarse
            # launcher status flag, require the actual data artifacts the merge
            # concatenates to be present; a shard missing any of them is a real
            # partial and is refused. The salvage is recorded in provenance.
            art = _artifacts_dir(run_dir, experiment)
            missing = [name for name in MERGED_CSVS if not (art / name).exists()]
            if missing:
                raise ValueError(
                    f"{rid}: shard status is {statuses[rid].get('status')!r} "
                    f"(failure_phase={statuses[rid].get('failure_phase')!r}) AND "
                    f"row data is incomplete (missing {missing}). Not merged.")
            salvaged[rid] = statuses[rid].get("failure_phase")

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
            # Different commits are allowed ONLY when the numerical engine is
            # byte-identical across them. Every method's dynamics, score, jumps,
            # energy, metrics and run config live in NUMERIC_ENGINE_FILES; if
            # those are identical at both commits, the two shards computed their
            # rows with the same numbers and only differ in orchestration
            # (notebook cells, launcher, docs), so the merge is a real
            # comparison. This is what lets a shard survive a later
            # notebook/launcher fix without a full recompute. A difference in
            # ANY engine file falls through to the hard error.
            differing = _numeric_engine_diff(commits[0], commits[1])
            if differing:
                raise ValueError(
                    f"shards were run at different commits {commits} AND the "
                    f"numerical engine differs between them ({sorted(differing)}); "
                    "a merged CSV whose rows came from different numerics is not "
                    "a comparison")

    registered = head.get("registered_methods", {}).get(experiment)
    if not registered:
        raise ValueError(
            f"launch_plan.json records no registered_methods for {experiment!r}; "
            "rerun the shards with a launcher that records shard provenance")
    registered_set = set(registered.split(","))

    # --- union-of-shards coverage ------------------------------------------
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
    shared_rows: dict[str, list[dict]] = {}

    for csv_name in MERGED_CSVS:
        # Column sets legitimately DIFFER between shards: write_summary_csv emits
        # acceptance columns only when MALA/PT ran and jump columns only when a
        # CP-family method ran, so each shard's header covers its own method
        # families. The merge takes the ORDER-PRESERVING UNION of columns and
        # pads missing cells -- exactly the header a single full run produces --
        # rather than demanding identical headers. DictWriter fills absent keys
        # with "" (restval), and every row's keys are a subset of the union, so
        # no data is dropped.
        fieldnames: list[str] = []
        rows_out: list[dict] = []
        per_shard: dict[str, int] = {}
        for rid in shard_ids:
            path = _artifacts_dir(results_root / rid, experiment) / csv_name
            if not path.exists():
                raise FileNotFoundError(f"{rid}: missing {csv_name}")
            names, rows = _read_csv(path)
            for name in names:
                if name not in fieldnames:
                    fieldnames.append(name)
            # a shard's CSV must contain exactly the methods it declared, once
            # the shard-independent blocks are set aside
            shared = [r for r in rows if r.get("method") in SHARED_BLOCKS]
            owned = [r for r in rows if r.get("method") not in SHARED_BLOCKS]
            present = _methods_in(owned)
            if present != shard_methods[rid]:
                raise ValueError(
                    f"{rid}/{csv_name}: rows cover {sorted(present)} but the "
                    f"shard declared {sorted(shard_methods[rid])}")
            if shared:
                if rid == shard_ids[0]:
                    shared_rows[csv_name] = shared      # keep exactly one copy
                elif shared != shared_rows.get(csv_name):
                    raise ValueError(
                        f"{csv_name}: shard {rid} disagrees with {shard_ids[0]} "
                        f"on the {sorted(SHARED_BLOCKS)} block; the shards did "
                        "not compare against the same reference")
            rows_out.extend(owned)
            per_shard[rid] = len(owned)
        rows_out.extend(shared_rows.get(csv_name, []))

        with (out_dir / csv_name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows_out)
        merged_counts[csv_name] = {
            "rows": len(rows_out), "per_shard": per_shard,
            "shared_block_rows": len(shared_rows.get(csv_name, [])),
        }

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
        # Per-shard commits, and the basis on which shards at different commits
        # were judged comparable (numerical engine byte-identical). With a single
        # commit this is trivially satisfied; with several it is the audit trail.
        "shard_commits": {rid: plans[rid].get("git", {}).get("commit")
                          for rid in shard_ids},
        "numeric_engine_verified_identical": sorted(
            {plans[rid].get("git", {}).get("commit") for rid in shard_ids}) != [None],
        "numeric_engine_files": list(NUMERIC_ENGINE_FILES),
        # Shards accepted despite a non-ok launcher status because their row data
        # was complete (a trailing non-data cell crashed). rid -> failure_phase.
        "salvaged_shards": salvaged,
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
