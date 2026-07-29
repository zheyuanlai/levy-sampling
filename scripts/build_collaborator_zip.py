"""Build a portable, checksum-indexed E1--E4 collaborator review archive."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile


HERE = Path(__file__).resolve().parent
JCP_ROOT = HERE.parent
if str(JCP_ROOT) not in sys.path:
    sys.path.insert(0, str(JCP_ROOT))

from scripts.validate_release import validate_release  # noqa: E402


INCLUDE = (
    "README.md",
    "environment.yml",
    "pyproject.toml",
    ".gitignore",
    "configs",
    "src",
    "notebooks/00_environment_check.ipynb",
    "notebooks/01_double_well.ipynb",
    "notebooks/02_mog40.ipynb",
    "notebooks/03_mb3well_10d.ipynb",
    "notebooks/04_coupled_phi4.ipynb",
    "notebooks/05_manuscript_plotting.ipynb",
    "notebooks/build_notebooks.py",
    "notebooks/run_notebook.py",
    "scripts/validate_release.py",
    "scripts/replot_manuscript_figures.py",
    "scripts/replot_generated_samples.py",
    "scripts/merge_method_shards.py",
    "scripts/build_collaborator_zip.py",
    "launch_production.py",
    "run_production.sh",
    "results/double_well",
    "results/mog40",
    "results/mb3well_10d",
    "results/coupled_phi4",
    "figures/png",
    "figures/pdf",
)

EXCLUDED_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
    ".DS_Store",
    "__MACOSX",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".npz"}


def _ignored(path: Path) -> bool:
    return (
        any(part in EXCLUDED_NAMES for part in path.parts)
        or path.suffix in EXCLUDED_SUFFIXES
    )


def _copy_entry(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if _ignored(relative) or not path.is_file():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def build_archive(output: Path, *, root: Path = JCP_ROOT) -> Path:
    root = root.resolve()
    validate_release(root, check_results=True, require_figures=True)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {output}")

    with tempfile.TemporaryDirectory(prefix="jcp-release-") as temp:
        package = Path(temp) / "JCP_levy_sampler_code"
        package.mkdir()
        for entry in INCLUDE:
            source = root / entry
            _copy_entry(source, package / entry)
        _write_checksums(package)
        # Validate the staged copy itself; this catches accidental dependencies
        # on files that were present in the research tree but omitted here.
        validate_release(package, check_results=True, require_figures=True)
        with zipfile.ZipFile(
            output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    archive.write(
                        path,
                        Path(package.name) / path.relative_to(package),
                    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=JCP_ROOT / "dist" / "JCP_levy_sampler_code.zip",
    )
    args = parser.parse_args(argv)
    path = build_archive(args.output)
    print(f"Built collaborator archive: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
