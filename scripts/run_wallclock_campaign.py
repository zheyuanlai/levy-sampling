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

# The run matrix is the FROZEN RELEASE matrix, which is wider than the
# manuscript display matrix in src.manuscript: the released CSV files also
# carry MALA everywhere, Raw-CP in E2--E4, and the genuine single-atom
# LSC-CP-RA arm alongside the multi-atom LSC-CP-MA arm in E3/E4. Rerunning
# only the display matrix would silently drop those columns from results/, so
# the renewed results reproduce the released matrix method-for-method.
RUN_MATRIX: dict[str, tuple[str, ...]] = {
    "double_well": ("ULA", "MALA", "FLA", "BAOAB", "PT", "CP",
                    "LSC-CP", "LSC-CP-RA"),
    "mog40": ("ULA", "MALA", "FLA", "BAOAB", "PT", "CP",
              "LSC-CP", "LSC-CP-RA"),
    "mb3well_10d": ("ULA", "MALA", "FLA", "BAOAB", "PT", "CP",
                    "LSC-CP", "LSC-CP-RA", "LSC-CP-MA"),
    "coupled_phi4": ("ULA", "MALA", "FLA", "BAOAB", "PT", "CP",
                     "LSC-CP", "LSC-CP-RA", "LSC-CP-MA"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0", help="single physical GPU index")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiments", default=",".join(ORDER))
    parser.add_argument("--results-root", type=Path,
                        default=ROOT / "results" / "jcp_sampling")
    parser.add_argument("--cell-timeout", type=int, default=43_200,
                        help="per-cell timeout handed to nbclient")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    names = [n.strip() for n in args.experiments.split(",") if n.strip()]
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"unknown experiments: {unknown}")

    run_dir = args.results_root.resolve() / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "campaign_plan.json").write_text(json.dumps({
        "run_id": args.run_id,
        "gpu": args.gpu,
        "experiments": names,
        "methods": {name: list(RUN_MATRIX[name]) for name in names},
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
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        env["JCP_GPU"] = args.gpu
        env["JCP_EXTRA_GPUS"] = args.gpu
        env["JCP_RUN_ID"] = args.run_id
        env["JCP_RESULTS_ROOT"] = str(args.results_root.resolve())
        env["JCP_METHODS"] = ",".join(RUN_MATRIX[name])
        env["PYTHONUNBUFFERED"] = "1"

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
