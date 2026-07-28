"""LSC-CP JCP benchmark package.

Import order matters: call src.gpu_guard.select_gpu(...) BEFORE importing
torch (and therefore before importing any other submodule here).
"""
