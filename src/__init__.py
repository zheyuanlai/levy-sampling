"""LSC-CP benchmark: samplers, references, metrics, and the run/plot split.

Two entry points cover ordinary use::

    from src.pipeline import load_experiment, run_variants_and_save   # running
    from src.plotting import load_runs, curve_figure, save_figure      # plotting

There is no import-order requirement and nothing to call before importing
torch. The device is resolved at run time by ``src.device.resolve_device`` and
defaults to ``auto``: CUDA when it is available, CPU otherwise. Both are fully
supported execution paths, and the device is recorded as provenance rather than
being a precondition for running.
"""
