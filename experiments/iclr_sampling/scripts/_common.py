"""Shared harness for the experiment scripts.

Pins the process to a single GPU *before* importing torch, then provides helpers
to load a YAML config, set up a timestamped run directory with tee'd logging,
build targets, optionally tune MH step sizes, and drive a scaling experiment.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

# --- pin to one GPU before torch initialises CUDA --------------------------- #
from experiments.iclr_sampling.utils import select_gpu  # noqa: E402
_GPU = select_gpu()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from experiments.iclr_sampling import reporting, plotting  # noqa: E402
from experiments.iclr_sampling.experiment import (  # noqa: E402
    run_target_experiment, tune_step_size, ALL_SAMPLERS)
from experiments.iclr_sampling.samplers import MALA, FLMC  # noqa: E402
from experiments.iclr_sampling.baselines import HMC  # noqa: E402
from experiments.iclr_sampling.targets import (  # noqa: E402
    ManyWellTarget, MoGTarget, BayesGMMTarget)
from experiments.iclr_sampling.utils import (  # noqa: E402
    environment_info, make_run_dir, save_config)


class Tee:
    """Write log lines to stdout and a logfile simultaneously."""

    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_run(cfg: Dict, cfg_path: str, tag: str):
    run_dir = make_run_dir(tag=tag)
    log = Tee(run_dir / "logs" / "run.log")
    log(f"=== {tag} ===")
    log(f"GPU (CUDA_VISIBLE_DEVICES) = {_GPU}")
    env = environment_info()
    for k, v in env.items():
        log(f"  {k}: {v}")
    save_config(run_dir, "experiment_config", cfg)
    save_config(run_dir, "environment", env)
    # keep a copy of the source config file too
    try:
        import shutil
        shutil.copy(cfg_path, run_dir / "configs" / Path(cfg_path).name)
    except Exception:
        pass
    return run_dir, log, env


def build_target(target_name: str, var: str, value, target_cfg: Dict, device):
    cfg = dict(target_cfg or {})
    if target_name == "manywell":
        return ManyWellTarget(n_blocks=int(value), device=device, **cfg)
    if target_name == "mog":
        return MoGTarget(n_modes=int(value), device=device, **cfg)
    if target_name == "bayes_gmm":
        return BayesGMMTarget(K=int(value), device=device, **cfg)
    raise ValueError(f"unknown target {target_name}")


def maybe_tune(methods, method_cfgs, target, run_cfg, tune_cfg, device, log):
    """Optionally auto-tune MALA/PT step size and HMC step size via short pilots."""
    if not tune_cfg or not tune_cfg.get("enabled", False):
        return method_cfgs
    method_cfgs = {m: dict(method_cfgs.get(m, {})) for m in methods}
    if "MALA" in methods and "mala_dt" in tune_cfg:
        dt = tune_step_size(MALA, target, {}, tune_cfg["mala_dt"], device,
                            n_part=tune_cfg.get("n_part", 512), log_fn=log)
        method_cfgs["MALA"]["dt"] = dt
        if "PT" in methods and tune_cfg.get("share_mala_dt_with_pt", True):
            method_cfgs["PT"]["dt"] = dt
    if "HMC" in methods and "hmc_eps" in tune_cfg:
        base = {k: v for k, v in method_cfgs["HMC"].items() if k != "dt"}
        eps = tune_step_size(HMC, target, base, tune_cfg["hmc_eps"], device,
                             n_part=tune_cfg.get("n_part", 512),
                             target_acc=0.7, min_acc=0.55, log_fn=log)
        method_cfgs["HMC"]["dt"] = eps
    return method_cfgs


def run_scaling(cfg: Dict, run_dir: Path, log, device):
    """Loop over the scaling variable, run all methods, collect raw rows.

    Returns (raw_rows, curves_by_config, finals_by_config, refs_by_value,
    targets_by_value).
    """
    exp_name = cfg["experiment_name"]
    target_name = cfg["target"]
    var = cfg["scaling"]["var"]
    values = cfg["scaling"]["values"]
    methods = cfg["methods"]
    run_cfg = cfg["run"]
    method_cfgs = cfg.get("method_cfgs", {})
    target_cfg = cfg.get("target_cfg", {})
    tune_cfg = cfg.get("tune", {})

    raw_rows: List[Dict] = []
    curves_by_config: Dict = {}
    finals_by_config: Dict = {}
    refs: Dict = {}
    targets: Dict = {}

    for value in values:
        log(f"\n########## {target_name} {var}={value} ##########")
        t_build = time.time()
        target = build_target(target_name, var, value, target_cfg, device)
        log(f"  target built ({time.time()-t_build:.1f}s): {target.metadata()}")
        targets[value] = target

        # reference (cache to disk)
        n_ref = int(run_cfg.get("n_ref", 20000))
        ref = target.sample_reference(n_ref, run_cfg.get("ref_seed", 12345), device)
        np.savez(run_dir / "report_artifacts" / f"ref_{target.name}.npz",
                 ref=ref.detach().cpu().numpy())
        refs[value] = ref

        mcfgs = maybe_tune(methods, method_cfgs, target, run_cfg, tune_cfg, device, log)
        save_config(run_dir, f"method_cfgs_{target.name}", mcfgs)

        res = run_target_experiment(target, methods, mcfgs, run_cfg, device,
                                    log_fn=log, ref=ref)
        for row in res["rows"]:
            row["experiment_name"] = exp_name
            row["target_name"] = target_name
            row[var] = value
            row["n_blocks"] = getattr(target, "n_blocks", np.nan)
            row["n_modes"] = getattr(target, "n_modes", np.nan)
            row["notes"] = ""
            raw_rows.append(row)
        curves_by_config[f"{var}={value}"] = res["curves"]
        finals_by_config[value] = res["finals"]

    return raw_rows, curves_by_config, finals_by_config, refs, targets


def finalize(run_dir: Path, raw_rows, scaling_var: str, log):
    """Write raw + summary CSVs grouped by (scaling_var, method)."""
    df = reporting.write_raw_csv(raw_rows, run_dir / "raw_runs.csv")
    summary = reporting.summarize(df, [scaling_var, "method"],
                                  run_dir / "summary_by_config.csv")
    log(f"\nWrote raw_runs.csv ({len(df)} rows) and summary_by_config.csv")
    return df, summary


def safe(log, fn, *args, **kwargs):
    """Run a (plotting/table) call, logging and swallowing any error so that a
    single artifact failure never discards an expensive completed run."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        import traceback
        log(f"  [warn] artifact step {getattr(fn,'__name__',fn)} failed: {e}")
        log(traceback.format_exc())
        return None


def copy_to_report(run_dir: Path, fig_paths: List, report_fig_dir: str,
                   table_paths: List = None, report_table_dir: str = None):
    import shutil
    Path(report_fig_dir).mkdir(parents=True, exist_ok=True)
    for p in fig_paths:
        if p is None:
            continue
        for pp in (p if isinstance(p, (list, tuple)) else [p]):
            try:
                shutil.copy(pp, Path(report_fig_dir) / Path(pp).name)
            except Exception:
                pass
    if table_paths and report_table_dir:
        Path(report_table_dir).mkdir(parents=True, exist_ok=True)
        for p in table_paths:
            if p is None:
                continue
            try:
                shutil.copy(p, Path(report_table_dir) / Path(p).name)
            except Exception:
                pass
