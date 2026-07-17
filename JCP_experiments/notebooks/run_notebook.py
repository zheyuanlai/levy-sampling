"""Execute one production notebook while preserving partial output and status.

The production launcher writes the executed notebook to an immutable run
folder. Direct legacy invocation still executes in place:

    JCP_GPU=4 python run_notebook.py 01_double_well.ipynb
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

import nbformat
from nbclient import NotebookClient


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.write("\n")


def _write_notebook(nb, source: Path, destination: Path | None) -> Path:
    """Write output exclusively, except for explicit legacy in-place mode."""
    output = source if destination is None else destination
    output.parent.mkdir(parents=True, exist_ok=True)
    if destination is None or output.resolve() == source.resolve():
        nbformat.write(nb, output)
    else:
        with output.open("x", encoding="utf-8") as handle:
            nbformat.write(nb, handle)
    return output


def execute_notebook(path: str | os.PathLike, *,
                     output_notebook: str | os.PathLike | None = None,
                     status_path: str | os.PathLike | None = None,
                     timeout: int = 28_800) -> dict:
    """Execute ``path`` and return a success manifest; failures are re-raised.

    When ``output_notebook`` is provided, both it and ``status_path`` use
    exclusive creation. The partially executed notebook is retained even when
    execution raises.
    """
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = Path(output_notebook).resolve() if output_notebook else None
    status_output = Path(status_path).resolve() if status_path else None
    for candidate in (destination, status_output):
        if candidate is not None and candidate.exists():
            raise FileExistsError(candidate)

    notebook = nbformat.read(source, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(source.parent)}},
    )
    started_at = _utc_now()
    started = time.monotonic()
    error: BaseException | None = None
    error_traceback = None
    try:
        client.execute()
    except BaseException as exc:
        error = exc
        error_traceback = traceback.format_exc()
    finally:
        written = _write_notebook(notebook, source, destination)

    elapsed = time.monotonic() - started
    status = {
        "status": "success" if error is None else "failed",
        "source_notebook": str(source),
        "executed_notebook": str(written),
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "elapsed_seconds": elapsed,
        "timeout_seconds": timeout,
        "run_id": os.environ.get("JCP_RUN_ID", ""),
        "jcp_gpu": os.environ.get("JCP_GPU", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    if error is not None:
        status.update({
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": error_traceback,
        })
    if status_output is not None:
        _write_json_exclusive(status_output, status)
    if error is not None:
        raise error
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", help="source .ipynb")
    parser.add_argument("--output-notebook",
                        help="immutable executed/partial notebook destination")
    parser.add_argument("--status-path", help="exclusive success/failure JSON path")
    parser.add_argument("--timeout", type=int, default=28_800,
                        help="per-cell timeout in seconds (default: 28800)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout < 1:
        raise SystemExit("--timeout must be positive")
    status = execute_notebook(
        args.notebook,
        output_notebook=args.output_notebook,
        status_path=args.status_path,
        timeout=args.timeout,
    )
    print(f"executed {args.notebook} in {status['elapsed_seconds']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
