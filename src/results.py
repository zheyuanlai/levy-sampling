"""Atomic per-variant result directories.

Every variant writes its own run directory and nothing else. Workers never
append to or otherwise touch a shared index, so any number of variants can run
concurrently without coordinating.

The write protocol is:

1. write everything into ``runs/<method>/.tmp-<uuid>/``;
2. flush and close every file;
3. write ``manifest.json`` carrying the schema version, file hashes, and run
   status;
4. atomically rename the temporary directory to ``runs/<method>/<run-id>/``;
5. write ``COMPLETE``.

A reader therefore never sees a half-written run: a directory without
``COMPLETE`` is ignored by the catalog scanner, and a directory whose file
hashes disagree with its manifest is rejected outright.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import uuid

import numpy as np
import torch

SCHEMA_VERSION = 2
COMPLETE_MARKER = "COMPLETE"
MANIFEST_NAME = "manifest.json"
INVALID_MARKER = "INVALID"

#: Files a completed production run is expected to carry.
REQUIRED_ARTIFACTS = (
    "resolved_config.yaml",
    "metrics_timeseries.csv",
    "cost_timeseries.csv",
    "terminal_samples.npz",
    "diagnostics.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value):
    """Convert values to strict, portable JSON.

    Non-finite floats become the sentinel strings ``"inf"``, ``"-inf"``, and
    ``"nan"`` so a payload can be written with ``allow_nan=False`` and still
    record values that are legitimately infinite.
    """
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, (np.floating, np.integer)):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return None
    # Coerce subclasses (torch.TorchVersion, numpy.str_, IntEnum, ...) to the
    # exact builtin: YAML and JSON serializers dispatch on the concrete type and
    # refuse anything they do not recognise.
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, int):
        return int(value)
    return str(value)


def stable_hash(payload, *, digest_size: int = 16) -> str:
    """Content hash of a JSON-serialisable payload, stable across processes."""
    text = json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(text.encode("utf-8"),
                           digest_size=digest_size).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_provenance(repo_root: Path | None = None) -> dict:
    def git(*args) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=repo_root, capture_output=True, text=True,
                check=False, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD") or "unknown",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty": None if status is None else bool(status),
    }


def slugify(text: str) -> str:
    """Filesystem-safe token that still reads like the original label."""
    safe = []
    for character in str(text):
        if character.isalnum() or character in "-_.":
            safe.append(character)
        elif character in " /\\":
            safe.append("-")
        elif character == "=":
            safe.append("")
        else:
            safe.append("-")
    token = "".join(safe).strip("-")
    while "--" in token:
        token = token.replace("--", "-")
    return token or "variant"


@dataclass
class RunPaths:
    """Where one experiment's artifacts live."""

    root: Path
    experiment_key: str

    @property
    def experiment_dir(self) -> Path:
        return self.root / self.experiment_key

    @property
    def runs_dir(self) -> Path:
        return self.experiment_dir / "runs"

    @property
    def reference_dir(self) -> Path:
        return self.experiment_dir / "reference"

    @property
    def protocols_dir(self) -> Path:
        return self.experiment_dir / "protocols"

    @property
    def fee_cache_dir(self) -> Path:
        return self.experiment_dir / "fee_calibration"

    @property
    def catalog_path(self) -> Path:
        return self.experiment_dir / "catalog.csv"

    def method_dir(self, method: str) -> Path:
        return self.runs_dir / slugify(method)

    def ensure(self) -> "RunPaths":
        for directory in (self.experiment_dir, self.runs_dir,
                          self.reference_dir, self.protocols_dir,
                          self.fee_cache_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self


class RunWriter:
    """Context manager implementing the atomic write protocol.

    Use as::

        with RunWriter(paths, method="FLA", run_id=...) as writer:
            writer.write_text("resolved_config.yaml", ...)
            writer.write_csv("metrics_timeseries.csv", rows, columns)
            writer.write_npz("terminal_samples.npz", ...)
            writer.set_manifest(...)

    On a clean exit the directory is renamed into place and ``COMPLETE`` is
    written. On an exception the temporary directory is removed, so a failed run
    leaves no partial artifact behind for the catalog to trip over.
    """

    def __init__(self, paths: RunPaths, *, method: str, run_id: str) -> None:
        self.paths = paths
        self.method = method
        self.run_id = run_id
        self.method_dir = paths.method_dir(method)
        self.final_dir = self.method_dir / run_id
        self.temp_dir = self.method_dir / f".tmp-{uuid.uuid4().hex}"
        self._manifest: dict | None = None
        self._written: list[str] = []

    def __enter__(self) -> "RunWriter":
        if self.final_dir.exists():
            raise FileExistsError(
                f"run directory already exists: {self.final_dir}")
        self.temp_dir.mkdir(parents=True, exist_ok=False)
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            return False
        self._finalize()
        return False

    # -- writers -----------------------------------------------------------
    def path(self, name: str) -> Path:
        target = self.temp_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def _record(self, name: str) -> None:
        if name not in self._written:
            self._written.append(name)

    def write_text(self, name: str, text: str) -> None:
        path = self.path(name)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        self._record(name)

    def write_json(self, name: str, payload) -> None:
        self.write_text(name, json.dumps(json_safe(payload), indent=2,
                                         sort_keys=True, allow_nan=False) + "\n")

    def write_yaml(self, name: str, payload) -> None:
        import yaml

        self.write_text(name, yaml.safe_dump(json_safe(payload),
                                             sort_keys=True,
                                             default_flow_style=False))

    def write_csv(self, name: str, rows, columns=None) -> None:
        rows = list(rows)
        if columns is None:
            seen: list[str] = []
            for row in rows:
                for key in row:
                    if key not in seen:
                        seen.append(key)
            columns = seen
        path = self.path(name)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns),
                                    restval="", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        self._record(name)

    def write_npz(self, name: str, **arrays) -> None:
        path = self.path(name)
        converted = {}
        for key, value in arrays.items():
            if isinstance(value, torch.Tensor):
                converted[key] = value.detach().cpu().numpy()
            else:
                converted[key] = np.asarray(value)
        with path.open("wb") as handle:
            np.savez_compressed(handle, **converted)
            handle.flush()
            os.fsync(handle.fileno())
        self._record(name)

    # -- manifest and rename ----------------------------------------------
    def set_manifest(self, manifest: dict) -> None:
        self._manifest = dict(manifest)

    def _file_hashes(self) -> dict[str, str]:
        hashes = {}
        for path in sorted(self.temp_dir.rglob("*")):
            if path.is_file():
                relative = str(path.relative_to(self.temp_dir))
                if relative == MANIFEST_NAME:
                    continue
                hashes[relative] = sha256_file(path)
        return hashes

    def _finalize(self) -> None:
        if self._manifest is None:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            raise RuntimeError(
                "a run must call set_manifest() before the writer closes")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            **self._manifest,
            "written_at_utc": utc_now(),
            "files": self._file_hashes(),
        }
        manifest.setdefault("status", "complete")
        manifest.setdefault("result_directory",
                            str(self.final_dir.relative_to(
                                self.paths.root.parent)
                                if self.paths.root.parent in self.final_dir.parents
                                else self.final_dir))
        path = self.temp_dir / MANIFEST_NAME
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(manifest), indent=2,
                                    sort_keys=True, allow_nan=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.method_dir.mkdir(parents=True, exist_ok=True)
        os.replace(self.temp_dir, self.final_dir)
        marker = self.final_dir / COMPLETE_MARKER
        with marker.open("w", encoding="utf-8") as handle:
            handle.write(utc_now() + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def read_manifest(run_dir: Path) -> dict | None:
    path = Path(run_dir) / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def verify_run(run_dir: Path, *, check_hashes: bool = True) -> tuple[bool, str]:
    """Decide whether a run directory is admissible.

    A run counts only when it has a manifest, has ``COMPLETE``, declares a
    schema version this code understands, is not marked invalid, and has file
    contents matching the hashes its manifest recorded.
    """
    run_dir = Path(run_dir)
    if (run_dir / INVALID_MARKER).exists():
        return False, "marked invalid"
    manifest = read_manifest(run_dir)
    if manifest is None:
        return False, "missing or unreadable manifest.json"
    if not (run_dir / COMPLETE_MARKER).is_file():
        return False, "missing COMPLETE marker"
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        return False, (f"schema version {manifest.get('schema_version')} != "
                       f"{SCHEMA_VERSION}")
    if manifest.get("status") not in (None, "complete"):
        return False, f"status is {manifest.get('status')!r}"
    files = manifest.get("files") or {}
    if not files:
        return False, "manifest records no files"
    if check_hashes:
        for relative, expected in files.items():
            path = run_dir / relative
            if not path.is_file():
                return False, f"missing file {relative}"
            if sha256_file(path) != expected:
                return False, f"hash mismatch for {relative}"
    return True, "ok"


def mark_invalid(run_dir: Path, reason: str) -> None:
    """Retire a run without deleting it; the catalog scanner will skip it."""
    path = Path(run_dir) / INVALID_MARKER
    path.write_text(f"{utc_now()} {reason}\n", encoding="utf-8")
