"""Bounded, provenance-preserving launcher for the four JCP experiments."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import time
from typing import Iterable

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "jcp_sampling"
HARD_MAX_CONCURRENT = 2

DUAL = "ULA,MALA,FLA,BAOAB,PT,CP,LSC-CP,LSC-CP-RA"
PAIRED_MA = "ULA,MALA,FLA,BAOAB,PT,CP,LSC-CP-MA"
EXPERIMENTS = {
    "double_well": ("01_double_well.ipynb", DUAL),
    "mog40": ("02_mog40.ipynb", DUAL),
    "mb3well_10d": ("03_mb3well_10d.ipynb", PAIRED_MA),
    "coupled_phi4": ("04_coupled_phi4.ipynb", PAIRED_MA),
}
SMOKE_CONFIG = {
    "particles": 64,
    "steps": 64,
    "q_theta": 4,
    "q_rho": 2,
    "pt_replicas": 4,
    # Coarse smoke-only reference construction. Production builder defaults
    # remain 600/400 basin grids, 40k flow steps, a 2400^2 E3 grid, and 200k
    # E4 SNIS proposals.
    # 96 keeps the smoke basin-map cell size after the E4 domain widened to
    # +-4 (jump-reachable order-parameter coverage).
    "basin_n_grid": 96,
    "basin_flow_steps": 800,
    "basin_mass_n_quad": 96,
    "reference_grid_size": 128,
    "snis_proposals": 2_048,
    # Fail-closed numerical diagnostics. These thresholds are part of every
    # smoke plan/config rather than hidden launcher constants.
    "max_score_clip_fraction": 0.01,
    "max_state_box_clip_fraction": 0.01,
    # Conditional on applied CP jumps, artificial boundary clipping must be
    # absent in smoke. Production uses the same fail-closed zero default.
    "max_jump_boundary_clip_fraction": 0.0,
    # Basin maps clamp lookups outside their construction grid; expose and gate
    # that mass rather than silently treating clamped labels as exact.
    "max_basin_map_outside_fraction": 0.0,
    "max_jump_cap_hits": 0,
    "min_mala_acceptance": 0.01,
    "min_pt_swap_acceptance": 0.01,
}

FULL_PREFLIGHT_ARTIFACT_NAMES = (
    "original_config.yaml",
    "resolved_preflight_config.json",
    "certificate_result.json",
)
FULL_SUCCESS_ARTIFACT_NAMES = FULL_PREFLIGHT_ARTIFACT_NAMES + (
    "resolved_config.json",
    "metrics_timeseries.csv",
    "summary.csv",
    "manifest.json",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-p{os.getpid()}"


def _write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")


def _git_text(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPOSITORY_ROOT, capture_output=True,
            text=True, check=False, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_provenance() -> dict:
    status = _git_text(["status", "--porcelain"])
    return {
        "commit": _git_text(["rev-parse", "HEAD"]) or "unknown",
        "branch": _git_text(["rev-parse", "--abbrev-ref", "HEAD"]) or "unknown",
        "dirty": None if status is None else bool(status),
        "status_porcelain": status if status is not None else "unknown",
    }


def parse_gpus(value: str) -> tuple[str, ...]:
    gpus = tuple(part.strip() for part in value.split(",") if part.strip())
    if not gpus:
        raise argparse.ArgumentTypeError("--gpus must contain at least one index")
    if len(set(gpus)) != len(gpus):
        raise argparse.ArgumentTypeError("--gpus indices must be unique")
    if any(not gpu.isdigit() for gpu in gpus):
        raise argparse.ArgumentTypeError("--gpus must be comma-separated integer indices")
    return gpus


def parse_experiments(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = [name for name in names if name not in EXPERIMENTS]
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"unknown/empty experiments {unknown}; choose from {tuple(EXPERIMENTS)}"
        )
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("experiment names must be unique")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpus", type=parse_gpus,
        default=parse_gpus(os.environ.get("JCP_GPU", "4")),
        help="physical GPU indices, e.g. --gpus 0,1",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=1,
        help="concurrent experiment processes (hard maximum: 2)",
    )
    parser.add_argument(
        "--experiments", type=parse_experiments,
        default=tuple(EXPERIMENTS),
        help="comma-separated subset of double_well,mog40,mb3well_10d,coupled_phi4",
    )
    parser.add_argument("--run-id", default=_default_run_id())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--no-regen", action="store_true",
                        default=os.environ.get("JCP_REGEN", "1") == "0")
    parser.add_argument("--skip-tests", action="store_true",
                        help="skip the required unit-test preflight")
    parser.add_argument("--dry-run", action="store_true",
                        help="write the launch plan without starting subprocesses")
    parser.add_argument(
        "--smoke-only", action="store_true",
        help=("run required regeneration/tests and all selected bounded smokes, "
              "then stop before certificates/full notebooks"))
    parser.add_argument("--notebook-timeout", type=int, default=28_800,
                        help="nbclient per-cell timeout")
    parser.add_argument("--wall-timeout", type=int, default=43_200,
                        help="whole notebook subprocess timeout")
    parser.add_argument("--smoke-timeout", type=int, default=3_600,
                        help="whole per-experiment dynamics-smoke timeout")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not 1 <= args.max_concurrent <= HARD_MAX_CONCURRENT:
        parser.error(f"--max-concurrent must be between 1 and {HARD_MAX_CONCURRENT}")
    if (args.notebook_timeout < 1 or args.wall_timeout < 1
            or args.smoke_timeout < 1):
        parser.error("timeouts must be positive")
    if args.skip_tests and not args.dry_run:
        parser.error("--skip-tests is allowed only with --dry-run; production tests are mandatory")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id):
        parser.error("--run-id may contain only letters, digits, dot, underscore, and hyphen")


def job_environment(base: dict[str, str], *, gpu: str,
                    selected_gpus: Iterable[str], run_id: str,
                    methods: str, results_root: Path) -> dict[str, str]:
    """Return the one-physical-GPU child environment."""
    env = dict(base)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["JCP_GPU"] = gpu
    existing = {item.strip() for item in env.get("JCP_EXTRA_GPUS", "").split(",")
                if item.strip()}
    # Supplying --gpus is an explicit opt-in for indices outside the historical
    # 4-7 allow-list. Include all requested devices so gpu_guard accepts each
    # one while every child still sees exactly one CUDA_VISIBLE_DEVICES entry.
    existing.update(str(item) for item in selected_gpus)
    env["JCP_EXTRA_GPUS"] = ",".join(sorted(existing, key=int))
    env["JCP_RUN_ID"] = run_id
    env["JCP_RESULTS_ROOT"] = str(results_root.resolve())
    env["JCP_METHODS"] = methods
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _initialize_logs(stdout_path: Path, stderr_path: Path, label: str) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=False)
    with stdout_path.open("x", encoding="utf-8") as handle:
        handle.write(f"[{_utc_now()}] start {label}\n")
    with stderr_path.open("x", encoding="utf-8") as handle:
        handle.write(f"[{_utc_now()}] start {label}\n")


def _terminate_process_group(process: subprocess.Popen, grace_seconds: float = 5.0) -> None:
    """Terminate a timed-out launcher and every inherited kernel/GPU child."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_logged(command: list[str], *, cwd: Path, env: dict[str, str],
               stdout_path: Path, stderr_path: Path, timeout: int,
               phase: str) -> int:
    """Append streams and kill the full child process group on timeout."""
    marker = f"\n[{_utc_now()}] phase={phase} command={command!r}\n"
    with stdout_path.open("a", encoding="utf-8") as stdout_handle, \
            stderr_path.open("a", encoding="utf-8") as stderr_handle:
        stdout_handle.write(marker)
        stdout_handle.flush()
        stderr_handle.write(marker)
        stderr_handle.flush()
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=stdout_handle,
            stderr=stderr_handle, start_new_session=True,
        )
        try:
            return int(process.wait(timeout=timeout))
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            raise


def _smoke_plan(name: str, gpu: str, run_id: str, job_dir: Path) -> dict:
    _, methods = EXPERIMENTS[name]
    return {
        "schema_version": 1,
        "kind": "per_experiment_dynamics_smoke",
        "experiment": name,
        "gpu": gpu,
        "methods": methods.split(","),
        "run_id": run_id,
        "config": dict(SMOKE_CONFIG),
        "expected_artifacts_directory": str(job_dir / "artifacts"),
    }


def run_smoke_job(name: str, gpu: str, run_dir: Path,
                  args: argparse.Namespace) -> dict:
    """Run one real, bounded model/method smoke before any full notebook."""
    _, methods = EXPERIMENTS[name]
    job_dir = run_dir / "smoke" / name
    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"
    _initialize_logs(stdout_path, stderr_path, f"smoke:{name}")
    env = job_environment(
        os.environ, gpu=gpu, selected_gpus=args.gpus,
        run_id=args.run_id, methods=methods, results_root=args.output_root,
    )
    plan = _smoke_plan(name, gpu, args.run_id, job_dir)
    _write_json_exclusive(job_dir / "smoke_plan.json", plan)
    status = {**plan, "status": "failed", "started_at_utc": _utc_now()}
    started = time.monotonic()
    artifacts = job_dir / "artifacts"
    try:
        command = [
            sys.executable, str(HERE / "scripts" / "smoke_experiment.py"),
            "--experiment", name,
            "--methods", methods,
            "--output-dir", str(artifacts),
            "--particles", str(SMOKE_CONFIG["particles"]),
            "--steps", str(SMOKE_CONFIG["steps"]),
            "--q-theta", str(SMOKE_CONFIG["q_theta"]),
            "--q-rho", str(SMOKE_CONFIG["q_rho"]),
            "--pt-replicas", str(SMOKE_CONFIG["pt_replicas"]),
            "--basin-n-grid", str(SMOKE_CONFIG["basin_n_grid"]),
            "--basin-flow-steps", str(SMOKE_CONFIG["basin_flow_steps"]),
            "--basin-mass-n-quad", str(SMOKE_CONFIG["basin_mass_n_quad"]),
            "--reference-grid-size", str(SMOKE_CONFIG["reference_grid_size"]),
            "--snis-proposals", str(SMOKE_CONFIG["snis_proposals"]),
            "--max-score-clip-fraction",
            str(SMOKE_CONFIG["max_score_clip_fraction"]),
            "--max-state-box-clip-fraction",
            str(SMOKE_CONFIG["max_state_box_clip_fraction"]),
            "--max-jump-boundary-clip-fraction",
            str(SMOKE_CONFIG["max_jump_boundary_clip_fraction"]),
            "--max-basin-map-outside-fraction",
            str(SMOKE_CONFIG["max_basin_map_outside_fraction"]),
            "--max-jump-cap-hits", str(SMOKE_CONFIG["max_jump_cap_hits"]),
            "--min-mala-acceptance",
            str(SMOKE_CONFIG["min_mala_acceptance"]),
            "--min-pt-swap-acceptance",
            str(SMOKE_CONFIG["min_pt_swap_acceptance"]),
        ]
        code = run_logged(
            command, cwd=HERE, env=env, stdout_path=stdout_path,
            stderr_path=stderr_path, timeout=args.smoke_timeout,
            phase="dynamics_smoke",
        )
        status["returncode"] = code
        if code != 0:
            status["failure_phase"] = "dynamics_smoke"
            return status
        required = (
            "original_config.yaml", "resolved_config.json",
            "smoke_metrics.csv", "smoke_manifest.json",
        )
        missing = [name for name in required if not (artifacts / name).is_file()]
        if missing:
            status.update({
                "failure_phase": "smoke_artifacts",
                "missing_artifacts": missing,
                "returncode": 1,
            })
            return status
        status.update({"status": "success", "returncode": 0})
        return status
    except subprocess.TimeoutExpired as exc:
        status.update({
            "failure_phase": "smoke_timeout", "returncode": None,
            "error_type": type(exc).__name__, "error_message": str(exc),
        })
        return status
    except BaseException as exc:
        status.update({
            "failure_phase": "smoke_launcher", "returncode": None,
            "error_type": type(exc).__name__, "error_message": str(exc),
        })
        return status
    finally:
        status["finished_at_utc"] = _utc_now()
        status["elapsed_seconds"] = time.monotonic() - started
        _write_json_exclusive(job_dir / "status.json", status)


def _job_plan(name: str, gpu: str, run_id: str, job_dir: Path,
              notebook_timeout: int) -> dict:
    notebook, methods = EXPERIMENTS[name]
    return {
        "experiment": name,
        "gpu": gpu,
        "methods": methods.split(","),
        "run_id": run_id,
        "source_notebook": str(HERE / "notebooks" / notebook),
        "executed_notebook": str(job_dir / "executed_notebook.ipynb"),
        "notebook_status": str(job_dir / "notebook_status.json"),
        "notebook_timeout_seconds": notebook_timeout,
        "expected_results_directory": str(job_dir / "artifacts"),
        "expected_figures_directory": str(job_dir / "artifacts" / "figures"),
        "certificate_gate_execution": (
            "inside notebook after resolved_preflight_config.json; "
            "certificate_result.json is written before the pass assertion"),
        "expected_preflight_artifacts": [
            str(job_dir / "artifacts" / name)
            for name in FULL_PREFLIGHT_ARTIFACT_NAMES],
        "expected_success_artifacts": [
            str(job_dir / "artifacts" / name)
            for name in FULL_SUCCESS_ARTIFACT_NAMES],
    }


def _record_full_artifact_state(status: dict, artifacts: Path) -> None:
    """Record preserved notebook artifacts without changing failure phase."""
    names = tuple(dict.fromkeys(
        FULL_SUCCESS_ARTIFACT_NAMES
        + ("stationarity/all_methods_summary.csv",)))
    presence = {name: (artifacts / name).is_file() for name in names}
    status["artifact_presence_after_notebook"] = presence
    status["preserved_artifacts"] = [
        str(artifacts / name) for name, present in presence.items() if present]
    if presence.get("resolved_config.json"):
        stage = "final_resolved"
    elif presence.get("certificate_result.json"):
        stage = "certificate_measured"
    elif presence.get("resolved_preflight_config.json"):
        stage = "model_cache_resolved"
    elif presence.get("original_config.yaml"):
        stage = "source_request_recorded"
    else:
        stage = "no_notebook_artifact"
    status["last_preserved_artifact_stage"] = stage
    certificate_path = artifacts / "certificate_result.json"
    if certificate_path.is_file():
        try:
            payload = json.loads(certificate_path.read_text(encoding="utf-8"))
            status["certificate_result_summary"] = {
                key: payload.get(key) for key in (
                    "passed", "max_residual", "tolerance", "settings")}
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            status["certificate_result_read_error"] = (
                f"{type(exc).__name__}: {exc}")


def run_experiment_job(name: str, gpu: str, run_dir: Path,
                       args: argparse.Namespace) -> dict:
    notebook, methods = EXPERIMENTS[name]
    job_dir = run_dir / name
    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"
    _initialize_logs(stdout_path, stderr_path, name)
    env = job_environment(
        os.environ, gpu=gpu, selected_gpus=args.gpus,
        run_id=args.run_id, methods=methods,
        results_root=args.output_root,
    )
    plan = _job_plan(name, gpu, args.run_id, job_dir, args.notebook_timeout)
    _write_json_exclusive(job_dir / "run_plan.json", plan)
    started_at = _utc_now()
    started = time.monotonic()
    status = {**plan, "status": "failed", "started_at_utc": started_at}
    try:
        # The only certificate gate is inside the notebook, after immutable
        # source and resolved model/cache provenance have been written.
        notebook_command = [
            sys.executable, str(HERE / "notebooks" / "run_notebook.py"),
            str(HERE / "notebooks" / notebook),
            "--output-notebook", str(job_dir / "executed_notebook.ipynb"),
            "--status-path", str(job_dir / "notebook_status.json"),
            "--timeout", str(args.notebook_timeout),
        ]
        notebook_code = run_logged(
            notebook_command, cwd=HERE, env=env, stdout_path=stdout_path,
            stderr_path=stderr_path, timeout=args.wall_timeout,
            phase="notebook",
        )
        status["notebook_returncode"] = notebook_code
        artifacts = job_dir / "artifacts"
        _record_full_artifact_state(status, artifacts)
        if notebook_code != 0:
            status.update({"failure_phase": "notebook", "returncode": notebook_code})
            return status
        required_files = (
            job_dir / "executed_notebook.ipynb",
            job_dir / "notebook_status.json",
            *(artifacts / name for name in FULL_SUCCESS_ARTIFACT_NAMES),
            artifacts / "stationarity" / "all_methods_summary.csv",
        )
        missing = [str(path) for path in required_files if not path.is_file()]
        figure_dir = artifacts / "figures"
        figure_pngs = list(figure_dir.glob("*.png")) if figure_dir.is_dir() else []
        figure_pdfs = list(figure_dir.glob("*.pdf")) if figure_dir.is_dir() else []
        if not figure_pngs or not figure_pdfs:
            missing.append(str(figure_dir / "<at-least-one-png-and-pdf>"))
        status["required_artifacts_checked"] = [str(path) for path in required_files]
        status["figure_png_count"] = len(figure_pngs)
        status["figure_pdf_count"] = len(figure_pdfs)
        if missing:
            status.update({
                "failure_phase": "required_artifacts",
                "missing_artifacts": missing,
                "returncode": 1,
            })
            return status
        status.update({"status": "success", "returncode": 0})
        return status
    except subprocess.TimeoutExpired as exc:
        status.update({
            "failure_phase": "timeout",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "returncode": None,
        })
        return status
    except BaseException as exc:
        status.update({
            "failure_phase": "launcher",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "returncode": None,
        })
        return status
    finally:
        status["finished_at_utc"] = _utc_now()
        status["elapsed_seconds"] = time.monotonic() - started
        _write_json_exclusive(job_dir / "status.json", status)


def _run_preflight(label: str, command: list[str], run_dir: Path,
                   env: dict[str, str], timeout: int) -> dict:
    directory = run_dir / label
    stdout_path = directory / "stdout.log"
    stderr_path = directory / "stderr.log"
    _initialize_logs(stdout_path, stderr_path, label)
    started = time.monotonic()
    started_at = _utc_now()
    try:
        code = run_logged(
            command, cwd=HERE, env=env, stdout_path=stdout_path,
            stderr_path=stderr_path, timeout=timeout, phase=label,
        )
        status = {"status": "success" if code == 0 else "failed", "returncode": code}
    except subprocess.TimeoutExpired as exc:
        status = {"status": "failed", "returncode": None,
                  "error_type": type(exc).__name__, "error_message": str(exc)}
    status.update({"started_at_utc": started_at, "finished_at_utc": _utc_now(),
                   "elapsed_seconds": time.monotonic() - started,
                   "command": command})
    _write_json_exclusive(directory / "status.json", status)
    return status


def _run_bounded_jobs(names: Iterable[str], gpus: tuple[str, ...],
                      max_concurrent: int, runner, *, stage: str) -> list[dict]:
    """Run a bounded stage with one physical GPU assigned to each child."""
    names = tuple(names)
    gpu_pool: queue.Queue[str] = queue.Queue()
    for gpu in gpus:
        gpu_pool.put(gpu)

    def scheduled(name: str) -> dict:
        gpu = gpu_pool.get()
        try:
            return runner(name, gpu)
        finally:
            gpu_pool.put(gpu)

    effective = min(max_concurrent, len(gpus))
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=effective) as executor:
        futures = {executor.submit(scheduled, name): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
            except BaseException as exc:
                result = {
                    "experiment": name,
                    "gpu": "unknown",
                    "status": "failed",
                    "failure_phase": f"{stage}_launcher",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            results.append(result)
            print(
                f"{stage} {result['experiment']}: {result['status']} "
                f"on GPU {result.get('gpu', 'unknown')}", flush=True,
            )
    order = {name: index for index, name in enumerate(names)}
    return sorted(results, key=lambda item: order[item["experiment"]])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    run_dir = args.output_root.resolve() / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    plan = {
        "schema_version": 1,
        "run_id": args.run_id,
        "results_root": str(args.output_root.resolve()),
        "created_at_utc": _utc_now(),
        "gpus": list(args.gpus),
        "max_concurrent_requested": args.max_concurrent,
        "max_concurrent_effective": min(args.max_concurrent, len(args.gpus)),
        "hard_max_concurrent": HARD_MAX_CONCURRENT,
        "experiments": list(args.experiments),
        "regenerate_notebooks": not args.no_regen,
        "run_tests": not args.skip_tests,
        "smoke_required_before_full": True,
        "full_certificate_gate": "in_notebook_after_resolved_preflight_config",
        "redundant_launcher_certificate_gate": False,
        "full_success_artifacts": list(FULL_SUCCESS_ARTIFACT_NAMES),
        "smoke_only": bool(args.smoke_only),
        "smoke_config": dict(SMOKE_CONFIG),
        "dry_run": args.dry_run,
        "git": git_provenance(),
    }
    _write_json_exclusive(run_dir / "launch_plan.json", plan)

    if args.dry_run:
        for index, name in enumerate(args.experiments):
            job_dir = run_dir / name
            job_dir.mkdir(parents=True, exist_ok=False)
            gpu = args.gpus[index % len(args.gpus)]
            payload = {**_job_plan(name, gpu, args.run_id, job_dir,
                                   args.notebook_timeout),
                       "required_smoke": _smoke_plan(
                           name, gpu, args.run_id, run_dir / "smoke" / name),
                       "status": "dry_run"}
            _write_json_exclusive(job_dir / "status.json", payload)
        _write_json_exclusive(run_dir / "status.json", {
            "status": "dry_run", "run_id": args.run_id,
            "finished_at_utc": _utc_now(),
        })
        print(f"dry-run launch plan: {run_dir}")
        return 0

    preflight_env = job_environment(
        os.environ, gpu=args.gpus[0], selected_gpus=args.gpus,
        run_id=args.run_id, methods=DUAL,
        results_root=args.output_root,
    )
    if not args.no_regen:
        status = _run_preflight(
            "notebook_regeneration",
            [sys.executable, str(HERE / "notebooks" / "build_notebooks.py")],
            run_dir, preflight_env, timeout=600,
        )
        if status["status"] != "success":
            _write_json_exclusive(run_dir / "status.json", {
                "status": "failed", "failure_phase": "notebook_regeneration",
                "run_id": args.run_id, "finished_at_utc": _utc_now(),
            })
            return 1
    if not args.skip_tests:
        status = _run_preflight(
            "unit_tests", [sys.executable, "-m", "pytest", "-q",
                           "tests", "tests_cpu"],
            run_dir, preflight_env, timeout=1_800,
        )
        if status["status"] != "success":
            _write_json_exclusive(run_dir / "status.json", {
                "status": "failed", "failure_phase": "unit_tests",
                "run_id": args.run_id, "finished_at_utc": _utc_now(),
            })
            return 1

    # Every selected model must complete a real small dynamics run with its
    # exact production method matrix.  This entire stage completes before a
    # single certificate/full notebook process is submitted.
    smoke_results = _run_bounded_jobs(
        args.experiments, args.gpus, args.max_concurrent,
        lambda name, gpu: run_smoke_job(name, gpu, run_dir, args),
        stage="smoke",
    )
    smoke_failed = [result for result in smoke_results
                    if result["status"] != "success"]
    if smoke_failed:
        _write_json_exclusive(run_dir / "status.json", {
            "status": "failed",
            "failure_phase": "smoke",
            "run_id": args.run_id,
            "finished_at_utc": _utc_now(),
            "smoke_results": smoke_results,
            "failed_smoke_experiments": [
                item["experiment"] for item in smoke_failed],
            "full_experiments_started": False,
        })
        return 1

    if args.smoke_only:
        _write_json_exclusive(run_dir / "status.json", {
            "status": "success",
            "completion_scope": "smoke_only",
            "run_id": args.run_id,
            "finished_at_utc": _utc_now(),
            "smoke_results": smoke_results,
            "full_experiments_started": False,
        })
        print(f"smoke-only gate passed: {run_dir}")
        return 0

    results = _run_bounded_jobs(
        args.experiments, args.gpus, args.max_concurrent,
        lambda name, gpu: run_experiment_job(name, gpu, run_dir, args),
        stage="full",
    )
    failed = [result for result in results if result["status"] != "success"]
    batch_status = {
        "status": "failed" if failed else "success",
        "run_id": args.run_id,
        "finished_at_utc": _utc_now(),
        "smoke_results": smoke_results,
        "experiments": results,
        "failed_experiments": [item["experiment"] for item in failed],
        "full_experiments_started": True,
    }
    _write_json_exclusive(run_dir / "status.json", batch_status)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
