# src/gpu_guard.py
"""GPU selection guard. MUST be imported (and select_gpu called) before torch.

GPUs 0-3 on this node belong to another project and must never be touched.
"""
import os
import sys

ALLOWED = {"4", "5", "6", "7"}


def select_gpu(index: str | int) -> None:
    idx = str(index)
    if idx not in ALLOWED:
        raise RuntimeError(
            f"GPU {idx} is forbidden. Allowed: {sorted(ALLOWED)}. "
            "GPUs 0-3 belong to another project."
        )
    if "torch" in sys.modules:
        raise RuntimeError("select_gpu() must be called before importing torch.")
    os.environ["CUDA_VISIBLE_DEVICES"] = idx
