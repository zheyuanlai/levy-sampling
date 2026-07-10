"""Execute a notebook in place with nbclient (used for the production runs).

Usage:  JCP_GPU=4 python run_notebook.py 01_double_well.ipynb
"""
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

path = sys.argv[1]
nb = nbformat.read(path, as_version=4)
client = NotebookClient(
    nb,
    timeout=28800,
    kernel_name="python3",
    resources={"metadata": {"path": os.path.dirname(os.path.abspath(path)) or "."}},
)
t0 = time.time()
try:
    client.execute()
finally:
    nbformat.write(nb, path)      # keep partial outputs on failure too
print(f"executed {path} in {time.time() - t0:.0f}s")
