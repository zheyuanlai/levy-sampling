"""Experiment driver: run a set of samplers on one target and collect results.

Seeds are realised as independent particle batches stacked on the leading
dimension and advanced under a common PRNG stream (the same vectorisation used
by the reference notebooks); the spread across these batches gives the error
bars.  Compute is recorded per vectorised run (wall-clock, gradient and
potential evaluations) and attached to every seed row for the raw CSV.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .baselines import BASELINE_REGISTRY
from .metrics import seed_metrics
from .samplers import SAMPLER_REGISTRY
from .utils import torch_generator

ALL_SAMPLERS = {**SAMPLER_REGISTRY, **BASELINE_REGISTRY}


def build_sampler(method: str, target, dt: float, kwargs: Dict):
    cls = ALL_SAMPLERS[method]
    return cls(target, dt, **kwargs)


def tune_step_size(sampler_cls, target, base_kwargs, candidates, device,
                   n_pilot: int = 150, n_part: int = 512, n_seeds: int = 1,
                   target_acc: float = 0.55, min_acc: float = 0.45,
                   seed: int = 9999, log_fn=print) -> float:
    """Short acceptance-driven pilot to pick a step size for MH samplers.

    Picks the largest candidate with mean acceptance >= ``min_acc``; otherwise
    the candidate closest to ``target_acc``.
    """
    results = {}
    for dt in candidates:
        s = sampler_cls(target, dt, **base_kwargs)
        g = torch_generator(seed, device)
        x = s.init_state(n_seeds, n_part, seed, device)
        accs = []
        for _ in range(n_pilot):
            x, diag = s.step(x, g)
            if "acc" in diag:
                accs.append(diag["acc"])
        results[dt] = float(np.mean(accs)) if accs else float("nan")
    ok = [dt for dt in candidates if results[dt] >= min_acc]
    best = max(ok) if ok else min(candidates, key=lambda dt: abs(results[dt] - target_acc))
    log_fn(f"    step-size tuning {sampler_cls.__name__}: "
           + ", ".join(f"{dt:g}->{results[dt]:.2f}" for dt in candidates)
           + f"  | chosen dt={best:g}")
    return best


def run_method(target, method: str, method_cfg: Dict, run_cfg: Dict, ref,
               device, log_fn=print) -> Dict:
    """Run a single sampler (vectorised over seeds) and return results."""
    n_seeds = len(run_cfg["seeds"])
    n_part = int(run_cfg["n_particles"])
    dt = float(method_cfg.get("dt", run_cfg["dt"]))
    # per-method step count allows a sampler to use a finer dt while running the
    # same *physical* time T = dt * n_steps (compute is reported via runtime/grads)
    n_steps = int(method_cfg.get("n_steps", run_cfg["n_steps"]))
    metric_every = int(method_cfg.get("metric_every",
                                      run_cfg.get("metric_every", max(1, n_steps // 20))))
    base_seed = int(run_cfg.get("base_seed", 0))

    kwargs = {k: v for k, v in method_cfg.items() if k != "dt"}
    sampler = build_sampler(method, target, dt, kwargs)

    # deterministic per-method RNG offset (hash() is salted per process)
    moff = (sum(ord(c) for c in method) % 97) + 1
    g = torch_generator(base_seed + 1000 * moff, device)
    x = sampler.init_state(n_seeds, n_part, base_seed, device)

    curve = {"t": [], "coverage": [], "emc": []}
    accs, swap_accs = [], []
    t0 = time.time()
    cheap_cfg = {"w2_sub": 0, "mmd_sub": 0}
    with torch.no_grad():
        for step in range(n_steps + 1):
            if step % metric_every == 0:
                fs = sampler.final_samples(x)
                X = fs.reshape(-1, target.dim)
                cg = torch_generator(base_seed, device)
                m = seed_metrics(target, X, None, cg, cheap_cfg)
                curve["t"].append(step * dt)
                curve["coverage"].append(m.get("mode_coverage", float("nan")))
                curve["emc"].append(m.get("emc", float("nan")))
            if step < n_steps:
                x, diag = sampler.step(x, g)
                if "acc" in diag:
                    accs.append(diag["acc"])
                if "swap_acc" in diag and np.isfinite(diag["swap_acc"]):
                    swap_accs.append(diag["swap_acc"])
    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime = time.time() - t0

    final = sampler.final_samples(x)             # (n_seeds, n_part, dim)
    acc_rate = float(np.mean(accs)) if accs else float("nan")
    swap_rate = float(np.mean(swap_accs)) if swap_accs else float("nan")

    # ---- full metrics per seed ---- #
    rows = []
    for si, seed in enumerate(run_cfg["seeds"]):
        cg = torch_generator(1234 + seed, device)
        m = seed_metrics(target, final[si], ref, cg, run_cfg)
        row = {
            "method": method, "seed": int(seed),
            "n_particles": n_part, "n_steps": n_steps, "dt": dt,
            "sigma": float(target.sigma), "dimension": int(target.dim),
            "jump_bank_size": int(getattr(target, "jump_bank_size", 0)),
            "runtime_sec": round(runtime, 3),
            "grad_evals": int(sampler.grad_evals),
            "pot_evals": int(sampler.pot_evals),
            "acceptance_rate": acc_rate,
            "swap_acceptance_rate": swap_rate,
        }
        row.update(m)
        rows.append(row)

    log_fn(f"  {method:5s}: runtime={runtime:6.1f}s grads={sampler.grad_evals:>7d} "
           f"acc={acc_rate if np.isfinite(acc_rate) else float('nan'):.3f} "
           f"final[seed0]: " + _fmt_final(rows[0]))

    return {"rows": rows, "curve": curve,
            "final_seed0": final[0].detach().cpu().numpy(),
            "runtime": runtime, "grad_evals": sampler.grad_evals,
            "pot_evals": sampler.pot_evals,
            "acc": acc_rate, "swap_acc": swap_rate}


def _fmt_final(row: Dict) -> str:
    parts = []
    for k in ["w2_or_sliced_w2", "mmd", "mode_coverage", "emc", "count_emc"]:
        if k in row and row[k] == row[k]:   # not NaN
            parts.append(f"{k.split('_')[0]}={row[k]:.3f}")
    return " ".join(parts)


def run_target_experiment(target, methods: List[str], method_cfgs: Dict,
                          run_cfg: Dict, device, log_fn=print,
                          ref: Optional[torch.Tensor] = None) -> Dict:
    """Run all ``methods`` on ``target`` and return aggregated results."""
    if ref is None:
        n_ref = int(run_cfg.get("n_ref", 20000))
        log_fn(f"  building reference sample (n={n_ref}) ...")
        ref = target.sample_reference(n_ref, run_cfg.get("ref_seed", 12345), device)
    out = {"rows": [], "curves": {}, "finals": {}, "compute": {}}
    for method in methods:
        mcfg = dict(method_cfgs.get(method, {}))
        res = run_method(target, method, mcfg, run_cfg, ref, device, log_fn)
        out["rows"].extend(res["rows"])
        out["curves"][method] = res["curve"]
        out["finals"][method] = res["final_seed0"]
        out["compute"][method] = {"runtime": res["runtime"],
                                  "grad_evals": res["grad_evals"],
                                  "pot_evals": res["pot_evals"],
                                  "acc": res["acc"], "swap_acc": res["swap_acc"]}
    out["ref"] = ref
    return out
