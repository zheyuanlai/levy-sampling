
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


def make_run_dir(root: str = "results/jcp_sampling", tag: str = "run") -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = Path(root) / f"{stamp}_{tag}"
    for sub in ["configs", "logs", "figures", "tables", "samples"]:
        (path / sub).mkdir(parents=True, exist_ok=False)
    return path


def jsonify(obj: Any):
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(jsonify(obj), indent=2, sort_keys=True))


def environment_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    try:
        import torch
        info.update({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "torch_cuda_version": torch.version.cuda})
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update({"gpu_name": p.name, "gpu_total_gb": round(p.total_memory / 1e9, 1)})
    except Exception as e:
        info["torch_error"] = str(e)
    try:
        info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        info["git_branch"] = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        info["git_status_short"] = subprocess.check_output(["git", "status", "--short"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception as e:
        info["git_error"] = str(e)
    return info


class Tee:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("a")
    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        self.f.write(msg + "\n"); self.f.flush()
    def close(self):
        self.f.close()
