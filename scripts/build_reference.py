#!/usr/bin/env python
"""Build (or rebuild) one experiment's reference and validate it.

    python scripts/build_reference.py E4
    python scripts/build_reference.py E4 --rebuild --device cuda

Exits nonzero when a reference fails its frozen acceptance gates. A failing
reference is never promoted: the validation record is written so the failure is
inspectable, but ``reference_validated`` stays unset and no run may cite it.

For E4 the escalation order on failure is fixed and is NOT "average the two
references" or "pick the nicer one": extend the parallel-tempering length and
temperature mixing, then improve the importance-sampling proposal, and only then
consider the optional annealed sequential Monte Carlo fallback.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.config import DEFAULT_RESULTS_ROOT, load_experiment, load_registry  # noqa: E402
from src.results import json_safe  # noqa: E402


def main(argv=None) -> int:
    registry = load_registry()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=sorted(registry["experiments"]))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)

    experiment = load_experiment(args.experiment, device=args.device,
                                 results_root=args.results_root)
    try:
        reference = experiment.ensure_reference(rebuild=args.rebuild)
    except Exception as error:                                # noqa: BLE001
        print(f"reference construction failed: {type(error).__name__}: {error}",
              file=sys.stderr)
        gates = getattr(error, "failed_gates", None)
        if gates:
            print(json.dumps(json_safe(gates), indent=2), file=sys.stderr)
        return 1

    description = reference.describe()
    print(f"{experiment.key}: {reference.kind}")
    print(f"reference_hash: {experiment.reference_hash}")
    print(f"directory: {experiment.paths.reference_dir}")
    validation = description.get("validation")
    if validation is not None:
        print(json.dumps(json_safe(validation), indent=2)[:4000])
        if validation.get("reference_validated") is False:
            print("reference did NOT pass its acceptance gates", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
