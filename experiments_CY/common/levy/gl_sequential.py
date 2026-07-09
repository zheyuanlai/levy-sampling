"""Sequential GL paperlite bookkeeping helpers."""

from __future__ import annotations

import re
from pathlib import Path


PROGRESS_COLUMNS = (
    "Method",
    "Start time",
    "End time",
    "Runtime",
    "Status",
    "Output CSV",
    "Output figure",
    "Log",
)


def safe_method_slug(method: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(method).strip().lower()).strip("_")
    if not slug:
        raise ValueError("method name produced an empty slug")
    return slug


def gl_method_output_paths(project_root, method: str, profile: str = "paperlite") -> dict[str, Path]:
    root = Path(project_root)
    slug = safe_method_slug(method)
    profile_slug = safe_method_slug(profile)
    release_root = root / "manuscript_clean_active" / "numerics" / "four_experiment_release"
    return {
        "csv": root
        / "manuscript_clean_active"
        / "numerics"
        / "four_experiment_release"
        / "tables"
        / "04_coupled_phi4_gl"
        / "per_method"
        / f"gl_method_{slug}_{profile_slug}.csv",
        "figure": root
        / "manuscript_clean_active"
        / "figures"
        / "four_experiment_release"
        / "diagnostics"
        / f"gl_method_{slug}_{profile_slug}.pdf",
        "log": release_root / "logs" / f"gl_method_{slug}_{profile_slug}.log",
    }


def gl_progress_path(project_root) -> Path:
    return (
        Path(project_root)
        / "manuscript_clean_active"
        / "numerics"
        / "four_experiment_release"
        / "logs"
        / "GL_METHOD_PROGRESS.md"
    )


def write_gl_progress(progress_path, rows: list[dict[str, object]]) -> Path:
    path = Path(progress_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GL paperlite progress",
        "",
        "| Method | Start time | End time | Runtime | Status | Output CSV | Output figure | Log |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        values = [str(row.get(col, "")) for col in PROGRESS_COLUMNS]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
