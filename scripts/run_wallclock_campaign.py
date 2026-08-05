"""Run the full E1--E4 campaign serially on one dedicated GPU.

This is the wall-clock-safe driver. It executes exactly what
``launch_production.py`` executes per job -- ``notebooks/run_notebook.py`` with
the launcher's one-visible-GPU child environment -- but strictly one experiment
at a time, so no two timed samplers ever share a device.

It exists because ``launch_production.py`` gates on
``validate_release.py --require-figures`` before starting, and that gate cannot
pass while the frozen results predate a change to the release contract (here:
the reinstated wall-clock columns and figures). Use the launcher for ordinary
reruns; use this driver to bootstrap a new contract.

    python scripts/run_wallclock_campaign.py --gpu 0 --run-id my-run

Pass ``--gpu cpu`` on a host with no CUDA device: the child then sees no GPU
at all and src/device.py resolves the run to CPU. Serial execution still
holds, but CPU wall-clock is not comparable to the published GPU numbers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.manuscript import EXPERIMENTS  # noqa: E402

ORDER = ("double_well", "mog40", "mb3well_10d", "coupled_phi4")

# The run matrix is wider than the manuscript DISPLAY matrix in src.manuscript
# -- it also carries MALA everywhere and Raw-CP in E2--E4, which stay in the
# CSV files as provenance -- but it is not simply "everything in the released
# CSVs".
#
# The realised-displacement arm differs by experiment, and only one of them is
# run per experiment:
#   E1/E2 (continuous jump laws): the genuine single-atom estimator LSC-CP-RA.
#   E3/E4 (atom banks): the atom-stratified estimator LSC-CP-MA, which IS the
#     arm the manuscript plots as "LSC-CP-RA (4)" and "LSC-CP-RA (8)" -- see
#     display_labels in src/manuscript.py. Single-atom LSC-CP-RA is NOT run
#     there. Stray LSC-CP-RA columns in the released E3/E4 CSV files are
#     leftovers from an exploratory comparison arm and are not reproduced here;
#     the released E4 manifest's own method_info confirms the gated run had
#     exactly the eight methods below.
_PROVENANCE_METHODS = ("ULA", "MALA", "FLA", "BAOAB", "PT", "CP", "LSC-CP")
RUN_MATRIX: dict[str, tuple[str, ...]] = {
    key: _PROVENANCE_METHODS + (spec.realised_arm,)
    for key, spec in EXPERIMENTS.items()
}


# Declared fail-closed threshold overrides, per experiment. Every declared
# threshold in the notebooks is env-overridable and every override is recorded
# in the run's immutable source config, resolved config, and manifest, so a
# relaxed gate is never silent. None is needed: with the realised arms above,
# E1--E4 all pass their declared thresholds unchanged.
ENV_OVERRIDES: dict[str, dict[str, str]] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0",
                        help="single physical GPU index, or 'cpu' for no GPU")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiments", default=",".join(ORDER))
    parser.add_argument("--results-root", type=Path,
                        default=ROOT / "results" / "jcp_sampling")
    parser.add_argument("--cell-timeout", type=int, default=43_200,
                        help="per-cell timeout handed to nbclient")
    parser.add_argument("--resume", action="store_true",
                        help="add experiments to an existing run directory; "
                             "an experiment that already has a job directory "
                             "is still refused, never overwritten")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    names = [n.strip() for n in args.experiments.split(",") if n.strip()]
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"unknown experiments: {unknown}")

    run_dir = args.results_root.resolve() / args.run_id
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    plan_name = "campaign_plan.json"
    if args.resume:
        existing = sorted(run_dir.glob("campaign_plan*.json"))
        plan_name = f"campaign_plan.resume{len(existing)}.json"
    (run_dir / plan_name).write_text(json.dumps({
        "run_id": args.run_id,
        "gpu": args.gpu,
        "resume": bool(args.resume),
        "experiments": names,
        "methods": {name: list(RUN_MATRIX[name]) for name in names},
        "threshold_overrides": {name: ENV_OVERRIDES[name]
                                for name in names if name in ENV_OVERRIDES},
        "serial": True,
        "created_at_utc": _utc_now(),
        "driver": "scripts/run_wallclock_campaign.py",
        "timing_protocol": (
            "one dedicated GPU, one experiment at a time, one visible device "
            "per child, identical batched ensemble per method"),
    }, indent=2) + "\n", encoding="utf-8")

    overall = 0
    for name in names:
        spec = EXPERIMENTS[name]
        job_dir = run_dir / name
        job_dir.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ)
        # "cpu" claims no device: hide every GPU instead of pinning one, and
        # leave the gpu_guard allow-list untouched (there is nothing to allow).
        on_cpu = args.gpu.strip().lower() == "cpu"
        env["CUDA_VISIBLE_DEVICES"] = "" if on_cpu else args.gpu
        env["JCP_GPU"] = args.gpu
        env["JCP_EXTRA_GPUS"] = "" if on_cpu else args.gpu
        env["JCP_RUN_ID"] = args.run_id
        env["JCP_RESULTS_ROOT"] = str(args.results_root.resolve())
        env["JCP_METHODS"] = ",".join(RUN_MATRIX[name])
        env["PYTHONUNBUFFERED"] = "1"
        overrides = ENV_OVERRIDES.get(name, {})
        env.update(overrides)
        if overrides:
            print(f"  {name} threshold overrides: {overrides}", flush=True)

        command = [
            sys.executable, str(ROOT / "notebooks" / "run_notebook.py"),
            str(ROOT / "notebooks" / spec.notebook),
            "--output-notebook", str(job_dir / "executed_notebook.ipynb"),
            "--status-path", str(job_dir / "notebook_status.json"),
            "--timeout", str(args.cell_timeout),
        ]
        started = time.monotonic()
        print(f"[{_utc_now()}] START {spec.number} {name} on GPU {args.gpu}",
              flush=True)
        with (job_dir / "stdout.log").open("w", encoding="utf-8") as out, \
                (job_dir / "stderr.log").open("w", encoding="utf-8") as err:
            code = subprocess.run(command, cwd=ROOT, env=env,
                                  stdout=out, stderr=err).returncode
        elapsed = time.monotonic() - started
        status = {
            "experiment": name, "gpu": args.gpu, "returncode": code,
            "status": "success" if code == 0 else "failed",
            "elapsed_seconds": elapsed, "finished_at_utc": _utc_now(),
        }
        (job_dir / "status.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(f"[{_utc_now()}] {status['status'].upper()} {name} "
              f"in {elapsed / 3600:.2f} h (rc={code})", flush=True)
        if code != 0:
            overall = 1
            print(f"stopping: {name} failed; see {job_dir / 'stderr.log'}",
                  flush=True)
            break

    (run_dir / "status.json").write_text(json.dumps({
        "run_id": args.run_id,
        "status": "failed" if overall else "success",
        "finished_at_utc": _utc_now(),
    }, indent=2) + "\n", encoding="utf-8")
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
