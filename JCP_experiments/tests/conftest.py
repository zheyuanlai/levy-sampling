import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gpu_guard import select_gpu  # noqa: E402

if "torch" not in sys.modules:          # conftest may be re-imported by tests
    select_gpu(os.environ.get("JCP_GPU", "4"))

import torch  # noqa: E402

assert torch.cuda.device_count() == 1
torch.set_default_dtype(torch.float64)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
