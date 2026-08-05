#!/usr/bin/env python
"""Package the source archive, or the frozen release archive.

    python scripts/build_release.py --source   dist/jcp-source.zip
    python scripts/build_release.py --frozen   dist/jcp-release.zip

The source archive is what someone needs to run everything from zero: code,
configuration, notebooks, scripts, tests, README, and the environment lock. It
deliberately excludes results, figures, and caches, and the resulting archive is
checked to be runnable without them.

The frozen archive additionally carries results, figures, resolved configs,
manifests, and executed notebooks. Only that archive is expected to be complete
in the release-validation sense.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zipfile

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

SOURCE_TREES = ("src", "configs", "notebooks", "scripts", "tests")
SOURCE_FILES = ("README.md", "environment.yml", "pyproject.toml", "AGENTS.md")
FROZEN_TREES = ("results", "figures", "resolved_configs", "manifests",
                "executed_notebooks")

EXCLUDED_DIRECTORIES = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache",
                        ".git"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp"}


def _iter_files(root: Path, relative: str):
    base = root / relative
    if base.is_file():
        yield base
        return
    if not base.is_dir():
        return
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        if EXCLUDED_DIRECTORIES & set(path.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES:
            continue
        yield path


def build(root: Path, output: Path, *, frozen: bool) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    trees = list(SOURCE_TREES) + list(SOURCE_FILES)
    if frozen:
        trees += list(FROZEN_TREES)
    written = 0
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in trees:
            for path in _iter_files(root, relative):
                archive.write(path, path.relative_to(root))
                written += 1
    return {"archive": str(output), "n_files": written,
            "kind": "frozen release" if frozen else "source"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", action="store_true")
    group.add_argument("--frozen", action="store_true")
    parser.add_argument("output", type=Path)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    args = parser.parse_args(argv)

    report = build(args.root, args.output, frozen=args.frozen)
    print(f"{report['kind']} archive: {report['archive']} "
          f"({report['n_files']} files)")
    if args.source:
        with zipfile.ZipFile(args.output) as archive:
            names = archive.namelist()
        leaked = [name for name in names
                  if name.split("/")[0] in ("results", "figures", "cache")]
        if leaked:
            print(f"source archive must not carry {sorted(set(n.split('/')[0] for n in leaked))}",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
