#!/usr/bin/env python
"""Collect the frozen release's manifest and resolved-config copies.

    python scripts/collect_release_artifacts.py
    python scripts/collect_release_artifacts.py E3 --root .

``validate_release.py --release`` requires top-level ``manifests/`` and
``resolved_configs/`` directories holding byte-identical copies of the files
belonging to the selected default outcomes -- it compares SHA-256 hashes, not
filenames. This script produces exactly that collection, using the same
selection rule the validator applies, so the two cannot drift apart:

* only outcomes whose ``variant_hash`` matches a default variant,
* only ``complete`` or ``uncalibratable`` statuses,
* only outcomes whose ``resolved_config.yaml`` reproduces the committed default
  experiment YAML exactly, so a reduced or edited run cannot be collected,
* the most recently written outcome when several qualify.

Copying rather than moving is deliberate: the run directory keeps the original,
and the collection is a derived view that can be deleted and rebuilt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.catalog import scan  # noqa: E402
from src.config import (DEFAULT_RESULTS_ROOT, default_variants,  # noqa: E402
                        load_method_configs, load_registry, load_yaml)
from src.results import MANIFEST_NAME, stable_hash  # noqa: E402


def select_outcomes(root: Path, results_root: Path, experiment_ids):
    """The default outcome matrix, chosen the way the validator chooses it."""
    registry = load_registry(root / "configs")
    method_configs = load_method_configs(root / "configs")

    selected, missing = [], []
    for experiment_id in experiment_ids:
        entry = registry["experiments"][experiment_id]
        experiment_dir = results_root / f"{experiment_id}_{entry['slug']}"
        if not (experiment_dir / "runs").is_dir():
            missing.append(f"{experiment_id}: no runs directory")
            continue

        rows, _ = scan(experiment_dir)
        expected_config = load_yaml(root / entry["config"])
        expected = []
        for method in entry["methods"]:
            expected.extend(default_variants(
                registry, method_configs, experiment_id, method))

        for variant in expected:
            production = []
            for row in rows:
                if row.get("variant_hash") != variant.hash:
                    continue
                if row.get("status") not in ("complete", "uncalibratable"):
                    continue
                try:
                    resolved = load_yaml(
                        Path(row["run_directory"]) / "resolved_config.yaml")
                    saved_input = {key: value for key, value in resolved.items()
                                   if key != "resolved"}
                except Exception:                             # noqa: BLE001
                    continue
                if stable_hash(saved_input) == stable_hash(expected_config):
                    production.append(row)
            if not production:
                missing.append(f"{experiment_id}: {variant.label}")
                continue
            chosen = max(production,
                         key=lambda row: str(row.get("written_at_utc") or ""))
            selected.append({"experiment_id": experiment_id,
                             "variant_label": variant.label,
                             "variant_slug": variant.slug,
                             "status": chosen.get("status"),
                             "run_id": chosen.get("run_id"),
                             "run_directory": chosen["run_directory"]})
    return selected, missing


def collect(selected, manifests_dir: Path, configs_dir: Path,
            experiment_ids) -> tuple[list[dict], list[str]]:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    records, written = [], set()
    for outcome in selected:
        run_dir = Path(outcome["run_directory"])
        stem = (f"{outcome['experiment_id']}__{outcome['variant_slug']}"
                f"__{outcome['run_id']}")
        pairs = ((run_dir / MANIFEST_NAME, manifests_dir / f"{stem}.json"),
                 (run_dir / "resolved_config.yaml",
                  configs_dir / f"{stem}.yaml"))
        record = dict(outcome)
        for source, destination in pairs:
            if not source.is_file():
                record.setdefault("errors", []).append(f"missing {source}")
                continue
            # copyfile, not copy2: the content is what is hashed, and carrying
            # the source mtime over would only make the copy look stale.
            shutil.copyfile(source, destination)
            written.add(destination)
            record.setdefault("copied", []).append(str(destination))
        records.append(record)

    # A superseded run leaves its copy behind, and a stale copy in a collection
    # that is supposed to mirror the selected outcomes is misleading evidence.
    # Only files this invocation is responsible for are considered: an
    # experiment that was not collected keeps its copies untouched.
    prefixes = tuple(f"{experiment_id}__" for experiment_id in experiment_ids)
    pruned = []
    for directory in (manifests_dir, configs_dir):
        for path in sorted(directory.iterdir()):
            if (path.is_file() and path.name.startswith(prefixes)
                    and path not in written):
                path.unlink()
                pruned.append(str(path))
    return records, pruned


def main(argv=None) -> int:
    registry = load_registry()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="*",
                        default=sorted(registry["experiments"]),
                        help="experiment ids (default: all)")
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--results-root", type=Path,
                        default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    unknown = [name for name in args.experiments
               if name not in registry["experiments"]]
    if unknown:
        parser.error(f"unknown experiments {unknown}")

    selected, missing = select_outcomes(
        args.root, args.results_root, args.experiments)
    records, pruned = collect(selected,
                              args.root / "manifests",
                              args.root / "resolved_configs",
                              args.experiments)

    failed = [record for record in records if record.get("errors")]
    by_status: dict[str, int] = {}
    for record in records:
        by_status[record["status"]] = by_status.get(record["status"], 0) + 1
    print(f"collected {len(records)} outcome(s): "
          + ", ".join(f"{count} {status}"
                      for status, count in sorted(by_status.items())))
    if pruned:
        print(f"pruned {len(pruned)} stale copy/copies")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps({"collected": records, "missing": missing,
                        "pruned": pruned}, indent=2),
            encoding="utf-8")
    if missing:
        print(f"\n{len(missing)} default outcome(s) have no production run:")
        for item in missing:
            print(f"  {item}")
    if failed:
        print(f"\n{len(failed)} outcome(s) had unreadable files:")
        for record in failed:
            print(f"  {record['variant_label']}: {record['errors']}")
    return 1 if (missing or failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
