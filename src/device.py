"""Device resolution: ``auto`` by default, CPU and CUDA both fully supported.

CPU and CUDA are equally official execution paths. There is no allow-list, no
pinned GPU index, and no environment variable that can forbid a run. Device
identity is recorded as provenance in every manifest, and FEE cost calibration
is per device and dtype, but nothing here decides whether the code is allowed
to execute.
"""
from __future__ import annotations

from importlib import metadata as importlib_metadata
import platform

import torch

#: The dtype the whole benchmark runs in.
DTYPE = torch.float64


def cuda_available() -> bool:
    """True when this process has at least one usable CUDA device."""
    try:
        return bool(torch.cuda.is_available()) and torch.cuda.device_count() > 0
    except (AssertionError, RuntimeError):
        return False


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve a device request.

    ``auto`` (the default) picks CUDA when it is available and CPU otherwise.
    ``cpu`` always resolves to CPU. An explicit CUDA request on a host without
    CUDA is an error: silently downgrading it would make a run's recorded device
    provenance disagree with what actually executed.
    """
    if device is None or (isinstance(device, str) and device.strip().lower() == "auto"):
        return torch.device("cuda" if cuda_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not cuda_available():
        raise RuntimeError(
            f"device {device!r} was requested explicitly but no CUDA device is "
            "available; use device='auto' to fall back to CPU")
    if resolved.type == "cuda" and resolved.index is not None:
        count = int(torch.cuda.device_count())
        if resolved.index >= count:
            raise RuntimeError(
                f"CUDA device index {resolved.index} is out of range "
                f"({count} device(s) visible)")
    return resolved


def synchronize(device: torch.device | None = None) -> None:
    """Flush queued CUDA work before a timing read; a no-op on CPU."""
    if device is None:
        if cuda_available():
            torch.cuda.synchronize()
        return
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def empty_cache() -> None:
    """Release cached CUDA blocks between phases; a no-op on CPU."""
    if cuda_available():
        torch.cuda.empty_cache()


def device_provenance(device: str | torch.device,
                      dtype: torch.dtype = DTYPE) -> dict:
    """Device/software provenance for manifests and FEE calibration records."""
    resolved = torch.device(device)
    device_index = None
    if resolved.type == "cuda":
        device_index = (torch.cuda.current_device() if resolved.index is None
                        else int(resolved.index))
    record = {
        "device_type": resolved.type,
        "device_index": device_index,
        "dtype": str(dtype).replace("torch.", ""),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
    }
    if resolved.type == "cuda":
        index = int(device_index)
        try:
            properties = torch.cuda.get_device_properties(index)
            record["gpu_name"] = properties.name
            record["gpu_total_memory_bytes"] = int(properties.total_memory)
            record["gpu_uuid"] = str(getattr(properties, "uuid", "")) or None
            record["gpu_capability"] = f"{properties.major}.{properties.minor}"
        except (AssertionError, AttributeError, RuntimeError):
            record["gpu_name"] = None
    for package in ("numpy", "scipy", "matplotlib"):
        try:
            record[f"{package}_version"] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            record[f"{package}_version"] = None
    return record


def software_version_key(device: str | torch.device,
                         dtype: torch.dtype = DTYPE) -> str:
    """Compact identity of the numerical stack, used inside the FEE hash."""
    provenance = device_provenance(device, dtype)
    parts = [
        provenance["device_type"],
        provenance.get("gpu_name") or provenance.get("cpu_model") or "unknown",
        provenance["dtype"],
        f"torch{provenance['torch_version']}",
        f"cuda{provenance['cuda_runtime_version']}",
        f"numpy{provenance.get('numpy_version')}",
    ]
    return "|".join(str(part) for part in parts)


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"
