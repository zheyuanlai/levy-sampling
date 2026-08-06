#!/usr/bin/env python
"""Execute notebooks into ``executed_notebooks/`` for a frozen release.

    python scripts/execute_notebooks.py --plot-only
    python scripts/execute_notebooks.py E3

The source notebooks stay clean: outputs cleared, no execution counts, no
machine-specific paths. Executed copies with their outputs belong only to the
frozen release, and they are written here rather than back over the source.

``--plot-only`` runs just the plot notebooks, which is the usual case: the run
notebooks may take hours, while replotting from saved results is quick.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config import load_registry  # noqa: E402

OUTPUT_DIRECTORY = REPOSITORY_ROOT / "executed_notebooks"


def notebooks_for(registry: dict, experiments, *, run: bool, plot: bool):
    selected = []
    for experiment_id in experiments:
        entry = registry["experiments"][experiment_id]
        if run:
            selected.append((experiment_id, "run",
                             REPOSITORY_ROOT / entry["run_notebook"]))
        if plot:
            selected.append((experiment_id, "plot",
                             REPOSITORY_ROOT / entry["plot_notebook"]))
    return selected


def execute(path: Path, output_dir: Path, *, timeout: int) -> dict:
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=timeout,
                            kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}})
    started = time.monotonic()
    record = {"notebook": str(path.relative_to(REPOSITORY_ROOT))}
    try:
        client.execute()
    except Exception as error:                                # noqa: BLE001
        record.update({"status": "failed", "error_type": type(error).__name__,
                       "error_message": str(error)[:2000]})
    else:
        record["status"] = "success"
    record["elapsed_seconds"] = time.monotonic() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / path.name
    # Written even on failure: the executed copy with its traceback is the
    # evidence of what went wrong.
    nbformat.write(notebook, destination)
    record["executed"] = str(destination.relative_to(REPOSITORY_ROOT))
    return record


def main(argv=None) -> int:
    registry = load_registry()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="*",
                        default=sorted(registry["experiments"]),
                        help="experiment ids (default: all)")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--timeout", type=int, default=28_800)
    args = parser.parse_args(argv)

    unknown = [name for name in args.experiments
               if name not in registry["experiments"]]
    if unknown:
        parser.error(f"unknown experiments {unknown}")
    if args.plot_only and args.run_only:
        parser.error("--plot-only and --run-only are mutually exclusive")

    selected = notebooks_for(registry, args.experiments,
                             run=not args.plot_only,
                             plot=not args.run_only)
    records = []
    for experiment_id, kind, path in selected:
        print(f"executing {experiment_id} {kind}: {path.name}", flush=True)
        record = execute(path, args.output, timeout=args.timeout)
        record.update({"experiment_id": experiment_id, "kind": kind})
        records.append(record)
        print(f"  {record['status']} in {record['elapsed_seconds']:.1f}s",
              flush=True)

    # Merge with whatever is already recorded rather than replacing it. The
    # usual invocation is --plot-only, and a plain overwrite would delete the
    # run notebooks' records every time figures are refreshed -- which makes
    # the documented workflow unable to satisfy release validation, since that
    # requires a successful record for all eight notebooks. Re-executing a
    # notebook replaces its own entry, so a stale success can never survive a
    # later failure.
    report_path = args.output / "execution_report.json"
    merged: dict[str, dict] = {}
    try:
        previous = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        previous = []
    for record in previous if isinstance(previous, list) else []:
        name = record.get("notebook")
        if name:
            merged[name] = record
    for record in records:
        merged[record["notebook"]] = record
    report_path.write_text(
        json.dumps([merged[name] for name in sorted(merged)], indent=2),
        encoding="utf-8")
    failures = [record for record in records if record["status"] != "success"]
    if failures:
        print(f"\n{len(failures)} notebook(s) failed:")
        for record in failures:
            print(f"  {record['notebook']}: {record.get('error_type')}: "
                  f"{record.get('error_message', '')[:200]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
