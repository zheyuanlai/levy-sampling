# src/gpu_guard.py
"""GPU selection guard. MUST be imported (and select_gpu called) before torch.

DEFAULT allowed set is GPUs 4-7 (GPUs 0-3 default off-limits: another project).
A specific GPU can be explicitly opted-in for a run via the env var
JCP_EXTRA_GPUS (comma-separated), e.g. `JCP_EXTRA_GPUS=0` -- use ONLY when you
have verified that GPU is free and belongs to your group. The committed default
is unchanged (4-7 only).

`select_gpu("cpu")` asks for no GPU at all: it hides every device, so the run
falls through to the CPU path in src/device.py. It takes no GPU from anyone and
is therefore not subject to the allow-list.
"""
import os
import sys

ALLOWED = {"4", "5", "6", "7"}
# Sentinels meaning "claim no GPU"; a host with no CUDA at all needs no pinning.
CPU_ALIASES = {"cpu", "none", ""}


def _allowed() -> set[str]:
    extra = {g.strip() for g in os.environ.get("JCP_EXTRA_GPUS", "").split(",")
             if g.strip()}
    return ALLOWED | extra


def select_gpu(index: str | int) -> None:
    idx = str(index).strip()
    if idx.lower() in CPU_ALIASES:
        if "torch" in sys.modules:
            raise RuntimeError("select_gpu() must be called before importing torch.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return
    if idx not in _allowed():
        raise RuntimeError(
            f"GPU {idx} is forbidden. Allowed: {sorted(_allowed())} "
            "(default 4-7; opt-in others via JCP_EXTRA_GPUS). GPUs 0-3 are "
            "default off-limits."
        )
    if "torch" in sys.modules:
        raise RuntimeError("select_gpu() must be called before importing torch.")
    os.environ["CUDA_VISIBLE_DEVICES"] = idx
