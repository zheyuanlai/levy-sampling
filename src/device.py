"""Device policy: run on CUDA when it is visible, on CPU otherwise.

Every module default, script entry point, and notebook resolves its device
through this module, so the benchmark runs unchanged on a CPU-only host.  The
numerics are identical (float64 everywhere); only the wall-clock columns and
the RNG streams differ, because a ``torch.Generator`` seeded on CPU does not
reproduce the CUDA stream for the same seed.  CPU runs are therefore valid on
their own but are not bitwise comparable to a GPU run of the same seed.

``src.gpu_guard.select_gpu`` must still be called *before* this module is
imported when a specific GPU has to be pinned: importing here imports torch.
"""
from __future__ import annotations

import os

import torch


def cuda_available() -> bool:
    """True when this process has at least one usable CUDA device.

    A CPU-only torch build, a driver mismatch, or ``CUDA_VISIBLE_DEVICES=""``
    all land here as ``False`` rather than as an exception.
    """
    try:
        return bool(torch.cuda.is_available()) and torch.cuda.device_count() > 0
    except (AssertionError, RuntimeError):
        return False


def default_device() -> str:
    """``"cuda"`` when a GPU is visible, otherwise ``"cpu"``."""
    return "cuda" if cuda_available() else "cpu"


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Normalize a requested device, downgrading CUDA to CPU when absent.

    ``None`` means "whatever this host has".  An explicit ``"cuda"`` request on
    a CPU-only host is honoured as CPU rather than raising, so archived scripts
    and manifests that hard-code the device stay runnable.
    """
    if device is None:
        return torch.device(default_device())
    dev = torch.device(device)
    if dev.type == "cuda" and not cuda_available():
        return torch.device("cpu")
    return dev


def synchronize() -> None:
    """Flush queued CUDA work before a wall-clock read; a no-op on CPU."""
    if cuda_available():
        torch.cuda.synchronize()


def empty_cache() -> None:
    """Release cached CUDA blocks between phases; a no-op on CPU."""
    if cuda_available():
        torch.cuda.empty_cache()


def require_single_device() -> str:
    """Entry-point guard returning the device this run must use.

    One visible GPU -> ``"cuda"``; no GPU -> ``"cpu"``.  Several visible GPUs
    is an error: the run protocol pins exactly one device, and co-tenants on a
    shared node make unpinned wall-clock numbers meaningless.
    """
    if not cuda_available():
        # Say so rather than falling back silently: the usual cause on a GPU
        # host is gpu_guard pinning an index this machine does not have.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        detail = "" if visible is None else f" (CUDA_VISIBLE_DEVICES={visible!r})"
        print(f"[device] no CUDA device visible{detail}; running on CPU",
              flush=True)
        return "cpu"
    count = int(torch.cuda.device_count())
    if count != 1:
        raise RuntimeError(
            f"expected exactly one visible CUDA device, found {count}; pin one "
            "with JCP_GPU (see src/gpu_guard.py) or CUDA_VISIBLE_DEVICES")
    return "cuda"


# Snapshot for use as a signature default.  CUDA visibility is fixed for the
# lifetime of a process, so this cannot go stale mid-run; call default_device()
# instead wherever a test monkeypatches torch.cuda.
DEFAULT_DEVICE: str = default_device()
