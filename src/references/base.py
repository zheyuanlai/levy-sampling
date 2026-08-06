"""The reference contract: one frozen ground truth per experiment.

A reference is the object every metric is measured against. It is built once,
written to disk, and reused by every method and every seed, so nothing a
sampler does can move it. Three properties are load-bearing:

*Frozen.* Every random draw a reference makes -- sample banks, bootstrap
replicates, projection directions -- takes an explicit seed read from the
experiment YAML. Rebuilding on the same configuration reproduces the same
numbers bit for bit on the same device.

*Uncounted.* Reference construction runs inside ``target.no_count()``. It is
analysis infrastructure, not sampler work, and must never move the oracle
counters.

*Self-describing.* :meth:`Reference.describe` returns a JSON-safe record of
every construction parameter, bank size, seed, and validation summary. It is
hashed into the ``reference_hash`` written to every run manifest, so a figure
can always be traced to the exact ground truth it was scored against.

Reuse is decided by :attr:`Reference.provenance`, the sub-record of
``describe()`` that depends only on the configuration and the target. Values
that a build measures -- descriptor masses, bandwidths, validation numbers --
live outside it, because comparing them would require the build the cache is
supposed to avoid.
"""
from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile

import numpy as np
import torch

from ..results import json_safe, stable_hash

#: Every reference writes its ``describe()`` payload here.
REFERENCE_JSON = "reference.json"


# ================================================================== atomic I/O
def _replace_from_temporary(path: Path, write) -> None:
    """Write through a temporary file in the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp",
        delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_npz(path: str | Path, **arrays) -> None:
    """Write ``arrays`` to an uncompressed ``.npz`` atomically.

    Torch tensors are detached, moved to the host, and stored as numpy arrays;
    float tensors keep float64.
    """
    payload = {}
    for key, value in arrays.items():
        if isinstance(value, torch.Tensor):
            payload[key] = value.detach().cpu().numpy()
        else:
            payload[key] = np.asarray(value)
    _replace_from_temporary(Path(path),
                            lambda handle: np.savez(handle, **payload))


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Read every array of an ``.npz`` into a plain dict."""
    with np.load(str(path), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def write_json(path: str | Path, payload) -> None:
    """Write a JSON-safe payload atomically, with sorted keys and no NaNs."""
    text = json.dumps(json_safe(payload), indent=2, sort_keys=True,
                      allow_nan=False) + "\n"
    _replace_from_temporary(Path(path),
                            lambda handle: handle.write(text.encode("utf-8")))


def read_json(path: str | Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ==================================================================== helpers
def frozen_generator(device, seed: int) -> torch.Generator:
    """A generator on ``device`` seeded by an explicit frozen seed."""
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(int(seed))
    return generator


def as_tensor(array, device, dtype=torch.float64) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=dtype,
                           device=torch.device(device))


def check_positive_int(value, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def check_seed(value, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


def import_metrics(*names, retries: int = 1, delay: float = 60.0):
    """Import functions from ``src.metrics``, retrying once after a pause.

    ``src.metrics`` is the single home of the metric definitions; a reference
    freezes a bandwidth or a projection set by calling into it rather than by
    growing a private copy. The retry exists only so a build launched while
    that module is mid-edit waits instead of forking the definition.
    """
    import time

    last: Exception | None = None
    for attempt in range(int(retries) + 1):
        try:
            from .. import metrics

            return tuple(getattr(metrics, name) for name in names)
        except (ImportError, AttributeError) as error:
            last = error
            if attempt < int(retries):
                time.sleep(float(delay))
    raise ImportError(
        f"could not import {list(names)} from src.metrics: {last}") from last


def passed_check(name: str, value: float, tolerance: float, *,
                 statistic: str, extra: dict | None = None) -> dict:
    """One validation check: the measured value, its tolerance, and a verdict."""
    record = {
        "check": str(name),
        "statistic": str(statistic),
        "value": float(value),
        "tolerance": float(tolerance),
        "passed": bool(float(value) <= float(tolerance)),
    }
    if extra:
        record.update(extra)
    return record


# ================================================================== the class
class Reference:
    """A frozen ground truth for one experiment.

    Subclasses set :attr:`kind` and :attr:`experiment_id` and implement the four
    methods below. ``sample`` returns draws in SAMPLING coordinates -- the same
    coordinates a sampler's state lives in -- so a metric never has to guess
    which frame a reference bank is expressed in.
    """

    #: Construction method, e.g. ``"grid_inverse_cdf_1d"``.
    kind: str = ""
    #: Owning experiment, e.g. ``"E1"``.
    experiment_id: str = ""

    def describe(self) -> dict:
        """JSON-safe provenance; hashed into the run manifests."""
        raise NotImplementedError

    def save(self, directory: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, directory: Path, target, device) -> "Reference":
        raise NotImplementedError

    def assert_valid_for_use(self, directory: Path | None = None) -> None:
        """Reject a stored reference that is evidence of a failed build.

        Most reference types are valid by construction. References with
        explicit acceptance gates override this hook so the generic cache
        loader cannot accidentally promote a persisted negative result.
        """
        return None

    def sample(self, n: int, generator: torch.Generator) -> torch.Tensor:
        """``(n, d)`` draws in sampling coordinates."""
        raise NotImplementedError

    # -- identity ----------------------------------------------------------
    @property
    def provenance(self) -> dict:
        """The configuration-determined part of ``describe()``.

        Two references with equal provenance were asked to compute the same
        thing; a cached build may be reused for the other. Measured outputs are
        deliberately excluded, since checking them would need the build.
        """
        return dict(self.describe().get("provenance", {}))

    @property
    def provenance_hash(self) -> str:
        return stable_hash(self.provenance)

    @property
    def hash(self) -> str:
        return stable_hash(self.describe())

    def write_describe(self, directory: Path) -> None:
        write_json(Path(directory) / REFERENCE_JSON, self.describe())

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(experiment_id={self.experiment_id!r}, "
                f"kind={self.kind!r}, provenance_hash={self.provenance_hash})")


def stored_provenance_hash(directory: Path) -> str | None:
    """The provenance hash of a saved reference, or ``None`` when absent."""
    path = Path(directory) / REFERENCE_JSON
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    stored = payload.get("provenance_hash")
    if stored is not None:
        return str(stored)
    provenance = payload.get("provenance")
    return None if provenance is None else stable_hash(provenance)
