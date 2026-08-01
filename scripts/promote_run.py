"""Promote a gated production run into the frozen release under ``results/``.

The notebooks write immutable per-run artifacts to
``results/jcp_sampling/<run-id>/<experiment>/artifacts/``. The published
release is the flat ``results/<experiment>/`` tree that the manuscript
plotting, the validator, and the collaborator archive all read. This script is
the step between the two, and it is fail-closed:

  * the run's own status must be ``success`` for every promoted experiment;
  * every released file must exist in the run's artifacts;
  * the run manifest must show the single-GPU, no-co-tenant timing protocol,
    since the released wall-clock axis depends on it;
  * promotion is atomic per experiment -- the destination is replaced only
    after every source file has been read.

    python scripts/promote_run.py --run-id 20260801-wallclock-gpu0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.manuscript import EXPERIMENTS  # noqa: E402
from src.runner import MIRROR_CSV_NAMES  # noqa: E402
from scripts.validate_release import _require_single_gpu_timing  # noqa: E402

# The released per-experiment file set is defined once, by the in-notebook
# mirror in src/runner.py. Do not restate it here: an independent list silently
# drops whatever it forgets (it forgot E2's modes.csv, which the generated
# sample figures need).
RELEASED_FILES = MIRROR_CSV_NAMES
# Present in every experiment; the rest of RELEASED_FILES is experiment
# specific (only E2 has modes.csv).
REQUIRED_FILES = ("manifest.json", "metrics_timeseries.csv", "summary.csv",
                  "positions.csv")
RELEASED_DIRS = ("stationarity",)


class PromotionError(RuntimeError):
    pass


def promote(run_dir: Path, results_dir: Path, names) -> dict:
    report = {}
    staged: dict[str, Path] = {}
    staging = Path(tempfile.mkdtemp(prefix="jcp-promote-", dir=results_dir))
    try:
        for name in names:
            job = run_dir / name
            artifacts = job / "artifacts"
            status_path = job / "status.json"
            if not status_path.is_file():
                raise PromotionError(f"{name}: no status.json in {job}")
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") != "success":
                raise PromotionError(
                    f"{name}: run status is {status.get('status')!r}, refusing "
                    "to promote")

            manifest_path = artifacts / "manifest.json"
            if not manifest_path.is_file():
                raise PromotionError(f"{name}: missing {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            _require_single_gpu_timing(manifest, path=str(manifest_path))

            target = staging / name
            target.mkdir(parents=True)
            copied = []
            for filename in RELEASED_FILES:
                source = artifacts / filename
                if not source.is_file():
                    if filename in REQUIRED_FILES:
                        raise PromotionError(f"{name}: missing {source}")
                    continue
                shutil.copy2(source, target / filename)
                copied.append(filename)
            missing_required = [f for f in REQUIRED_FILES if f not in copied]
            if missing_required:
                raise PromotionError(
                    f"{name}: missing required files {missing_required}")
            for dirname in RELEASED_DIRS:
                source = artifacts / dirname
                if not source.is_dir():
                    raise PromotionError(f"{name}: missing {source}")
                csvs = sorted(source.glob("*.csv"))
                if not csvs:
                    raise PromotionError(f"{name}: no CSV files in {source}")
                (target / dirname).mkdir()
                for path in csvs:
                    shutil.copy2(path, target / dirname / path.name)
                    copied.append(f"{dirname}/{path.name}")
            staged[name] = target
            report[name] = {
                "run_status": status.get("status"),
                "elapsed_seconds": status.get("elapsed_seconds"),
                "files": copied,
            }

        # Every source read succeeded: swap the destinations in.
        for name, target in staged.items():
            destination = results_dir / name
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(target), str(destination))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--results-root", type=Path,
                        default=ROOT / "results" / "jcp_sampling")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    names = [n.strip() for n in args.experiments.split(",") if n.strip()]
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"unknown experiments: {unknown}")
    run_dir = args.results_root.resolve() / args.run_id
    if not run_dir.is_dir():
        raise SystemExit(f"no such run: {run_dir}")
    report = promote(run_dir, args.results_dir.resolve(), names)
    for name, item in report.items():
        print(f"promoted {name}: {len(item['files'])} files "
              f"({item['elapsed_seconds'] / 3600:.2f} h run)")
    print(f"source run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
