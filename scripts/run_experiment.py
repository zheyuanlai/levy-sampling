#!/usr/bin/env python
"""Headless entry point: run an experiment's default variant matrix.

    python scripts/run_experiment.py E3
    python scripts/run_experiment.py E3 --methods FLA,LSC-CP-RA
    python scripts/run_experiment.py E1 --device cpu --stationarity

This does exactly what the run notebook does, for people who would rather not
open Jupyter. It is not a scheduler: there is no GPU allow-list, no pinned
device index, and no concurrency policy. ``--device auto`` picks CUDA when it is
there and CPU otherwise, and both are fully supported execution paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.catalog import write_catalog  # noqa: E402
from src.config import DEFAULT_RESULTS_ROOT, load_registry  # noqa: E402
from src.pipeline import load_experiment, run_variants_and_save  # noqa: E402


def main(argv=None) -> int:
    registry = load_registry()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=sorted(registry["experiments"]))
    parser.add_argument("--methods", default=None,
                        help="comma-separated subset of the experiment's methods")
    parser.add_argument("--device", default="auto",
                        help="auto (default), cpu, cuda, or cuda:N")
    parser.add_argument("--results-root", type=Path,
                        default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--stationarity", action="store_true",
                        help="also run the optional per-variant long-chain diagnostic")
    parser.add_argument("--refresh-calibration", action="store_true")
    parser.add_argument("--rebuild-reference", action="store_true")
    parser.add_argument("--report", type=Path, default=None,
                        help="write the per-variant report as JSON")
    args = parser.parse_args(argv)

    experiment = load_experiment(args.experiment, device=args.device,
                                 results_root=args.results_root)
    print(f"{experiment.key}: device={experiment.device}, "
          f"{experiment.particles} particles x {len(experiment.seeds)} seeds",
          flush=True)

    reference = experiment.ensure_reference(rebuild=args.rebuild_reference)
    print(f"reference: {reference.kind} ({experiment.reference_hash})",
          flush=True)
    fee = experiment.ensure_fee_calibration()
    print(f"FEE calibration: rho={fee.rho:.4g} ({fee.hash})", flush=True)

    enabled = registry["experiments"][args.experiment]["methods"]
    methods = (list(enabled) if args.methods is None
               else [name.strip() for name in args.methods.split(",")
                     if name.strip()])
    unknown = [name for name in methods if name not in enabled]
    if unknown:
        parser.error(f"{unknown} are not enabled for {args.experiment}; "
                     f"choose from {sorted(enabled)}")

    reports = {}
    for method in methods:
        print(f"\n=== {method} ===", flush=True)
        reports[method] = run_variants_and_save(
            experiment=experiment, method=method,
            run_stationarity=args.stationarity,
            refresh_calibration=args.refresh_calibration)

    catalog = write_catalog(experiment.paths.experiment_dir)
    print(f"\ncatalog: {catalog['n_runs']} run(s) indexed", flush=True)

    failures = [report for method in reports for report in reports[method]
                if report.get("status") == "failed"]
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(reports, indent=2, default=str),
                               encoding="utf-8")
    if failures:
        print(f"\n{len(failures)} variant(s) failed:", flush=True)
        for failure in failures:
            print(f"  {failure['variant_label']}: "
                  f"{failure.get('error_type')}: {failure.get('error_message')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
