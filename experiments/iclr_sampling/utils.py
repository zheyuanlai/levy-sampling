"""Shared utilities for the ICLR sampling experiment suite.

Conventions used throughout this suite
--------------------------------------
* Target density:  p_inf(x) ∝ exp(-2 V(x) / sigma^2).
* We fix sigma^2 = 2 for every target, so the density is exactly ∝ exp(-V(x)).
* The overdamped Langevin SDE  dX = -∇V dt + sigma dB  has p_inf as its
  stationary distribution.  Hence ``force(x) = -∇V(x)`` and the diffusion noise
  per step is ``sigma * sqrt(dt) * N(0, I)``.
* The LSB-MC dynamics are
      dZ = (-∇V(Z) + S_L(Z)) dt + sigma dB + dL,
  where ``L`` is a compound-Poisson process with a finite jump bank and
  ``S_L`` is the Lévy-score correction that preserves p_inf.

Only a single GPU is ever used.  Callers are expected to set
``CUDA_VISIBLE_DEVICES`` *before* importing torch (see ``select_gpu``), after
which everything lives on ``cuda:0``.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# GPU selection (must run before `import torch`)
# --------------------------------------------------------------------------- #
def select_gpu(preferred: Optional[int] = None) -> str:
    """Pin the process to a single GPU via ``CUDA_VISIBLE_DEVICES``.

    The shared H200 box requires that we use exactly one GPU.  Per the project
    convention we prefer GPU 4 (where the user's other jobs already live and
    there is spare capacity).  If ``CUDA_VISIBLE_DEVICES`` is already set we
    leave it untouched.  Returns the value that was set (as a string).
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"]
    gpu = preferred if preferred is not None else int(os.environ.get("ICLR_GPU", "4"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return str(gpu)


# --------------------------------------------------------------------------- #
# Reproducibility helpers
# --------------------------------------------------------------------------- #
def torch_generator(seed: int, device) -> "torch.Generator":  # noqa: F821
    import torch

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def set_global_seed(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Output-directory management (timestamped, never overwrites)
# --------------------------------------------------------------------------- #
def make_run_dir(root: str = "results/iclr_sampling", tag: str = "") -> Path:
    """Create a fresh timestamped output directory and its subfolders."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{tag}" if tag else stamp
    base = Path(root) / name
    for sub in ("configs", "figures", "logs", "report_artifacts"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def save_config(run_dir: Path, name: str, cfg: Dict[str, Any]) -> Path:
    path = Path(run_dir) / "configs" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonify(cfg), f, indent=2, sort_keys=True)
    return path


def _jsonify(obj):
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


# --------------------------------------------------------------------------- #
# Environment / hardware logging
# --------------------------------------------------------------------------- #
def environment_info() -> Dict[str, Any]:
    import torch

    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu_name"] = p.name
        info["gpu_total_gb"] = round(p.total_memory / 1e9, 1)
    try:
        info["numpy"] = np.__version__
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
        info["nvidia_smi"] = out.strip()
    except Exception:
        pass
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        info["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass
    return info


# --------------------------------------------------------------------------- #
# Numerical helpers (torch) — imported lazily so this module is import-safe.
# --------------------------------------------------------------------------- #
def tame(drift, dt: float, cap: float = 1.0):
    """Tamed Euler drift step, with a configurable maximum displacement ``cap``.

        step = dt*f / (1 + (dt/cap)*||f||)   (norm over the last axis)

    ``cap=1`` recovers the standard tamed step ``dt*f/(1+dt||f||)`` used by the
    reference notebooks (displacement bounded by 1).  A larger ``cap`` lets the
    drift take bigger steps when the force is large, which is useful for letting
    particles relax quickly after an (off-mode) compound-Poisson jump, more
    closely matching the continuous-time dynamics.  The diffusion term is
    unaffected, so the scheme is still stable for stiff potentials.
    """
    nrm = drift.norm(dim=-1, keepdim=True)
    return dt * drift / (1.0 + (dt / cap) * nrm)


def sanitize(x, clip: float):
    """Replace NaN/Inf with finite values and clip the state to [-clip, clip]."""
    import torch

    x = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
    return x.clamp(-clip, clip)


def randn_like_g(x, gen):
    import torch

    return torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)


def gauss_legendre_01(n: int, device, dtype):
    """Gauss-Legendre nodes/weights mapped to the interval [0, 1]."""
    import torch

    nodes, weights = np.polynomial.legendre.leggauss(int(n))
    nodes = 0.5 * (nodes + 1.0)        # [-1,1] -> [0,1]
    weights = 0.5 * weights
    return (torch.tensor(nodes, device=device, dtype=dtype),
            torch.tensor(weights, device=device, dtype=dtype))


def sample_symmetric_alpha_stable(shape, alpha: float, gen, device, dtype):
    """Symmetric alpha-stable S_alpha_S(1) samples (Chambers-Mallows-Stuck).

    Char. function ``E[exp(i t X)] = exp(-|t|^alpha)`` so alpha=2 gives N(0, 2).
    Matches ``archive/flmc_utils.py`` and both reference notebooks.
    """
    import math

    import torch

    if abs(alpha - 2.0) < 1e-9:
        return math.sqrt(2.0) * torch.randn(shape, generator=gen, device=device, dtype=dtype)
    V = (torch.rand(shape, generator=gen, device=device, dtype=dtype) - 0.5) * math.pi
    W = -torch.log(torch.rand(shape, generator=gen, device=device, dtype=dtype).clamp_min(1e-12))
    cosV = torch.cos(V).clamp_min(1e-12)
    cos1mV = torch.cos((1.0 - alpha) * V).clamp_min(1e-12)
    return (torch.sin(alpha * V) / cosV ** (1.0 / alpha)) * \
           (cos1mV / W) ** ((1.0 - alpha) / alpha)


def flmc_c_alpha(alpha: float) -> float:
    """FLMC drift normalisation  Gamma(alpha-1)/Gamma(alpha/2)^2  (Simsekli 2017)."""
    import math

    if not (1.0 < alpha <= 2.0):
        raise ValueError(f"FLMC alpha must satisfy 1 < alpha <= 2, got {alpha}")
    return math.gamma(alpha - 1.0) / (math.gamma(alpha / 2.0) ** 2)


@dataclass
class GradCounter:
    """Lightweight counter for potential/gradient evaluations (compute fairness).

    ``add`` is called with the number of *batch* gradient/potential evaluations;
    we report the cumulative count so methods with inner loops (HMC leapfrog,
    PT replicas, LSBMC quadrature) are charged fairly relative to plain ULA.
    """

    grad_evals: int = 0
    pot_evals: int = 0

    def add_grad(self, n: int = 1):
        self.grad_evals += int(n)

    def add_pot(self, n: int = 1):
        self.pot_evals += int(n)
