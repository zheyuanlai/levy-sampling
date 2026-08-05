"""Frozen ground truths, one per experiment, built once and reused everywhere.

``build_or_load`` is the only entry point a run needs. It resolves the
experiment to its reference class, compares the provenance the current
configuration would produce against the provenance of whatever is already on
disk, and either reuses the stored reference or builds and saves a new one. The
comparison is over configuration-determined values only -- bounds, grid sizes,
bank sizes, seeds, tolerances -- because deciding reuse from measured outputs
would require the very build the cache exists to avoid.

Building always runs inside ``target.no_count()``: a reference is analysis
infrastructure and must never move the oracle counters.
"""
from __future__ import annotations

from pathlib import Path
import warnings

from ..device import resolve_device
from ..results import stable_hash
from .base import (REFERENCE_JSON, Reference, load_npz, read_json, save_npz,
                   stored_provenance_hash, write_json)
from .e1 import DoubleWellReference
from .e2 import MoG40Reference
from .e3 import MullerBrownReference

__all__ = [
    "REFERENCE_JSON",
    "Reference",
    "DoubleWellReference",
    "MoG40Reference",
    "MullerBrownReference",
    "build_or_load",
    "reference_class",
    "load_npz",
    "read_json",
    "save_npz",
    "write_json",
]

_LOCAL_CLASSES = {
    "E1": DoubleWellReference,
    "E2": MoG40Reference,
    "E3": MullerBrownReference,
}


def reference_class(experiment_id: str):
    """The reference class owning ``experiment_id``.

    ``E4`` is imported lazily: that module is developed separately, and a
    missing or broken import must surface as an ``ImportError`` naming it
    rather than as a failure to import this package at all.
    """
    key = str(experiment_id).strip().upper()
    if key in _LOCAL_CLASSES:
        return _LOCAL_CLASSES[key]
    if key == "E4":
        try:
            from . import e4
        except ImportError as error:
            raise ImportError(
                "the E4 reference lives in src/references/e4.py, which is not "
                f"importable: {error}. That module is written separately; "
                "E1-E3 do not depend on it."
            ) from error
        declared = getattr(e4, "REFERENCE_CLASS", None)
        if declared is None:
            candidates = [value for value in vars(e4).values()
                          if isinstance(value, type)
                          and issubclass(value, Reference)
                          and value is not Reference]
            if len(candidates) != 1:
                raise ImportError(
                    "src/references/e4.py must expose exactly one Reference "
                    "subclass, or name it in a module-level REFERENCE_CLASS; "
                    f"found {sorted(cls.__name__ for cls in candidates)}")
            declared = candidates[0]
        return declared
    raise KeyError(
        f"no reference is defined for experiment {experiment_id!r}; known "
        f"experiments are E1, E2, E3, E4")


def _build(cls, config: dict, target, directory: Path, *, device,
           verbose: bool):
    builder = getattr(cls, "build", None)
    if builder is not None:
        return builder(config, target, directory, device=device,
                       verbose=verbose)
    module = __import__(cls.__module__, fromlist=["build_reference"])
    return module.build_reference(config, target, directory, device=device,
                                  verbose=verbose)


def build_or_load(experiment_id: str, config: dict, target, directory: Path, *,
                  device, rebuild: bool = False, verbose: bool = False
                  ) -> Reference:
    """Return the experiment's reference, loading it when the config matches.

    A stored reference is reused when its recorded provenance hash equals the
    hash of the provenance the current configuration would build. Anything else
    -- no stored reference, a changed bound, grid, bank size, seed, or
    tolerance, or ``rebuild=True`` -- triggers a fresh build, which is then
    saved into ``directory``.
    """
    directory = Path(directory)
    device = resolve_device(device)
    cls = reference_class(experiment_id)

    declared = (config.get("reference") or {}).get("method")
    if declared is not None and str(declared) != str(cls.kind):
        raise ValueError(
            f"{experiment_id} declares reference.method={declared!r} but "
            f"{cls.__name__} builds a {cls.kind!r} reference")

    if not rebuild and (directory / REFERENCE_JSON).is_file():
        payload = read_json(directory / REFERENCE_JSON)
        stored = stored_provenance_hash(directory)
        matches_identity = (str(payload.get("experiment_id")) == str(cls.experiment_id)
                            and str(payload.get("kind")) == str(cls.kind))
        if stored is not None and matches_identity:
            expected = _expected_provenance_hash(cls, config, target,
                                                 experiment_id)
            if expected is None or stored == expected:
                return cls.load(directory, target, device)

    directory.mkdir(parents=True, exist_ok=True)
    reference = _build(cls, config, target, directory, device=device,
                       verbose=verbose)
    reference.save(directory)
    return reference


def _expected_provenance_hash(cls, config: dict, target,
                              experiment_id: str) -> str | None:
    """Hash of the provenance the current configuration would build.

    ``None`` means the class does not declare a provenance, in which case a
    stored reference is reused on identity alone -- a weaker guarantee, so it
    is warned about rather than assumed.
    """
    provenance_for = getattr(cls, "provenance_for", None)
    if provenance_for is None:
        warnings.warn(
            f"{cls.__name__} does not implement provenance_for(config, target); "
            f"the stored {experiment_id} reference is being reused without "
            "checking it against the current configuration",
            RuntimeWarning, stacklevel=3)
        return None
    return stable_hash(provenance_for(config, target))
