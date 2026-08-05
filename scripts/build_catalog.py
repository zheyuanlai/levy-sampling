#!/usr/bin/env python
"""Rebuild the derived run index by scanning manifests.

    python scripts/build_catalog.py results/E3_muller_brown/
    python scripts/build_catalog.py --all results/

The catalog is derived, never authoritative. Workers do not write it, so it can
be rebuilt at any time and a lost or stale copy costs nothing. Only runs that
carry a manifest and a COMPLETE marker, declare a schema version this code
understands, match their recorded file hashes, and are not marked invalid are
admitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.catalog import rebuild_all, write_catalog  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path,
                        help="an experiment directory, or a results root with --all")
    parser.add_argument("--all", action="store_true",
                        help="treat PATH as a results root and rebuild every experiment")
    parser.add_argument("--no-verify-hashes", action="store_true",
                        help="skip file-hash verification (faster, weaker)")
    parser.add_argument("--json", action="store_true",
                        help="print the report as JSON")
    args = parser.parse_args(argv)

    if not args.path.is_dir():
        parser.error(f"{args.path} is not a directory")

    if args.all:
        reports = rebuild_all(args.path)
    else:
        report = write_catalog(args.path,
                               check_hashes=not args.no_verify_hashes)
        report["experiment_dir"] = str(args.path)
        reports = [report]

    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        for report in reports:
            print(f"{report['experiment_dir']}: {report['n_runs']} run(s) "
                  f"indexed, {report['n_rejected']} rejected -> "
                  f"{report['catalog']}")
            for rejection in report["rejections"]:
                print(f"    rejected {rejection['run_directory']}: "
                      f"{rejection['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
