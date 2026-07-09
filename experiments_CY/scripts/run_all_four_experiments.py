"""Execute the four canonical notebooks sequentially and record progress."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RELEASE = ROOT / "manuscript_clean_active" / "numerics" / "four_experiment_release"
NOTEBOOK_DIR = RELEASE / "notebooks"
EXECUTED_DIR = RELEASE / "executed_notebooks"
LOG_DIR = RELEASE / "logs"
PROGRESS = LOG_DIR / "NOTEBOOK_RUN_PROGRESS.md"
RUN_LABEL = os.environ.get("LEVY_RUN_LABEL", "Canonical release")

NOTEBOOKS = [
    "01_double_well.ipynb",
    "02_triple_well.ipynb",
    "03_muller_brown_10d.ipynb",
    "04_coupled_phi4_gl.ipynb",
]
PROGRESS_COLUMNS = ["notebook", "start", "end", "runtime", "status", "tables", "figures", "executed", "error"]


def fmt_duration(seconds: float) -> str:
    minutes, sec = divmod(float(seconds), 60.0)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:d}:{minutes:02d}:{sec:04.1f}"


def write_progress(rows: list[dict[str, str]]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {RUN_LABEL} notebook run progress",
        "",
        "| notebook | start time | end time | runtime | status | tables generated | figures generated | executed notebook path | error |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {notebook} | {start} | {end} | {runtime} | {status} | {tables} | {figures} | {executed} | {error} |".format(
                **row
            )
        )
    PROGRESS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_progress() -> list[dict[str, str]]:
    if not PROGRESS.exists():
        return []
    rows = []
    for line in PROGRESS.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("| ") or line.startswith("|---") or line.startswith("| notebook "):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != len(PROGRESS_COLUMNS):
            continue
        rows.append(dict(zip(PROGRESS_COLUMNS, parts)))
    return rows


def count_outputs() -> tuple[int, int]:
    table_count = len(list((RELEASE / "tables").rglob("*.csv")))
    fig_count = len(list((ROOT / "manuscript_clean_active" / "figures" / "four_experiment_release").rglob("*.*")))
    return table_count, fig_count


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Usage: python run_all_four_experiments.py [01_double_well.ipynb ...]")
        return 0
    notebooks = [Path(arg).name for arg in sys.argv[1:]] if len(sys.argv) > 1 else NOTEBOOKS
    invalid = [name for name in notebooks if name not in NOTEBOOKS]
    if invalid:
        print(f"Unknown release notebook(s): {', '.join(invalid)}")
        return 2
    EXECUTED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LEVY_PROFILE"] = env.get("LEVY_PROFILE", "paperlite")
    runtime_dir = LOG_DIR / "jupyter_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    env["JUPYTER_RUNTIME_DIR"] = str(runtime_dir)
    env["JUPYTER_ALLOW_INSECURE_WRITES"] = "1"
    env["MPLBACKEND"] = "Agg"
    rows: list[dict[str, str]] = read_progress() if len(sys.argv) > 1 else []
    rows = [row for row in rows if row.get("notebook") in NOTEBOOKS]
    rows = [row for row in rows if row.get("notebook") not in notebooks]
    write_progress(rows)
    for notebook in notebooks:
        start = datetime.now().isoformat(timespec="seconds")
        t0 = time.time()
        out_name = notebook
        log_prefix = "canonical_" + Path(notebook).stem
        log_path = LOG_DIR / f"{log_prefix}_nbconvert.log"
        cmd = [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(NOTEBOOK_DIR / notebook),
            "--output-dir",
            str(EXECUTED_DIR),
            "--output",
            out_name,
            "--ExecutePreprocessor.timeout=7200",
            "--ExecutePreprocessor.kernel_name=python3",
        ]
        status = "running"
        row = {
            "notebook": notebook,
            "start": start,
            "end": "",
            "runtime": "",
            "status": status,
            "tables": "",
            "figures": "",
            "executed": (EXECUTED_DIR / out_name).relative_to(ROOT).as_posix(),
            "error": "",
        }
        rows.append(row)
        write_progress(rows)
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        elapsed = time.time() - t0
        end = datetime.now().isoformat(timespec="seconds")
        tables, figures = count_outputs()
        row.update(
            {
                "end": end,
                "runtime": fmt_duration(elapsed),
                "status": "success" if proc.returncode == 0 else "failed",
                "tables": str(tables),
                "figures": str(figures),
                "error": "" if proc.returncode == 0 else log_path.relative_to(ROOT).as_posix(),
            }
        )
        write_progress(rows)
        if proc.returncode != 0:
            return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
