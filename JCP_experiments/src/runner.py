"""Checkpoint loop, defensible timing, CSV emission, dt refinement, and the
ULA first-passage barrier verification.

Timing policy (the wall-clock axis is a reported result):
* torch.cuda.synchronize() immediately before starting and stopping every
  timed region; the timer covers ONLY sampler work (metrics, reference
  sampling, plotting and I/O are outside it);
* 20 untimed warm-up steps on a throwaway sampler absorb allocator/JIT
  effects without touching the production chain;
* seeds run sequentially, never batched, so per-run wall-clock is
  meaningful; PT's step advances all K replicas, so its wall-clock includes
  every replica by construction.
"""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
import time

import numpy as np
import torch

CSV_BASE_COLUMNS = [
    "method", "seed", "step", "t", "wallclock_s",
    "W2", "TV", "TV_density", "MMD", "EMC", "EJS",
    "nonfinite_frac", "m_clip_fraction", "max_log_magnitude",
    "mala_accept", "pt_swap_accept", "jump_count_mean",
]


# ---------------------------------------------------------------- hardware
def hardware_manifest() -> dict:
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=False).stdout.strip()
    except Exception:
        sha = "unknown"
    return {
        "cpu": cpu,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "git_sha": sha,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


# ---------------------------------------------------------------- one run
def run_one(method: str, seed: int, sampler_factory, n_steps: int,
            steps_per_ck: int, dt: float, metrics_fn, potential,
            warmup: int = 20, quiet: bool = False) -> tuple[list[dict], dict]:
    """One (method, seed) production run. Returns (checkpoint rows, info)."""
    # warm-up on a throwaway sampler: warms allocator/JIT, chain untouched
    warm = sampler_factory(method, seed)
    for _ in range(warmup):
        warm.step()
    del warm
    torch.cuda.synchronize()

    sampler = sampler_factory(method, seed)
    torch.cuda.synchronize()
    nV0, ng0, nq0 = potential.n_V, potential.n_grad, potential.n_Vdelta

    rows: list[dict] = []
    wall = 0.0
    n_ck = n_steps // steps_per_ck
    step = 0
    for _ck in range(n_ck):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps_per_ck):
            sampler.step()
        torch.cuda.synchronize()
        wall += time.perf_counter() - t0
        step += steps_per_ck
        row = {"method": method, "seed": seed, "step": step,
               "t": step * dt, "wallclock_s": wall}
        row.update(metrics_fn(sampler.positions()))
        row.update(sampler.pop_diagnostics())
        rows.append(row)
    info = {
        "wallclock_total_s": wall,
        "V_evals_per_step": (potential.n_V - nV0) / n_steps,
        "grad_evals_per_step": (potential.n_grad - ng0) / n_steps,
        "score_quad_evals_per_step": (potential.n_Vdelta - nq0) / n_steps,
    }
    if not quiet:
        print(f"  {method} seed {seed}: {wall:.1f}s sampler wall-clock", flush=True)
    return rows, info


def run_experiment(methods, seeds, sampler_factory, n_steps: int,
                   steps_per_ck: int, dt: float, metrics_fn, potential,
                   warmup: int = 20) -> tuple[list[dict], dict]:
    """Sequential production loop. Returns (all rows, per-method info)."""
    all_rows: list[dict] = []
    method_info: dict[str, dict] = {}
    for method in methods:
        t_m = time.perf_counter()
        infos = []
        for seed in seeds:
            rows, info = run_one(method, seed, sampler_factory, n_steps,
                                 steps_per_ck, dt, metrics_fn, potential, warmup)
            all_rows.extend(rows)
            infos.append(info)
        method_info[method] = {
            "wallclock_mean_s": float(np.mean([i["wallclock_total_s"] for i in infos])),
            "wallclock_std_s": float(np.std([i["wallclock_total_s"] for i in infos], ddof=1))
            if len(infos) > 1 else 0.0,
            "V_evals_per_step": infos[0]["V_evals_per_step"],
            "grad_evals_per_step": infos[0]["grad_evals_per_step"],
            "score_quad_evals_per_step": infos[0]["score_quad_evals_per_step"],
        }
        print(f"{method}: done in {time.perf_counter() - t_m:.1f}s total", flush=True)
    return all_rows, method_info


# ---------------------------------------------------------------- CSV / IO
def write_timeseries_csv(rows: list[dict], path: str) -> list[str]:
    extras = sorted({k for r in rows for k in r} - set(CSV_BASE_COLUMNS))
    cols = CSV_BASE_COLUMNS + extras
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return cols


def _terminal_stats(rows, methods, seeds, metric_keys):
    out = {}
    for method in methods:
        out[method] = {}
        for key in metric_keys:
            vals = []
            for seed in seeds:
                r = [row for row in rows if row["method"] == method
                     and row["seed"] == seed and key in row]
                if r:
                    vals.append(r[-1][key])
            if vals:
                out[method][key] = (float(np.mean(vals)),
                                    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
    return out


def time_to_threshold(rows, method, seeds, floor: float, key: str = "TV",
                      persist: int = 5) -> float:
    """First t at which the seed-mean of `key` is <= 2x bias floor and stays
    there for `persist` consecutive checkpoints; inf if never."""
    by_step: dict[int, list[float]] = {}
    for row in rows:
        if row["method"] == method and key in row:
            by_step.setdefault(row["step"], []).append(row[key])
    steps = sorted(by_step)
    mean_curve = [float(np.mean(by_step[s])) for s in steps]
    thresh = 2.0 * floor
    run = 0
    for i, v in enumerate(mean_curve):
        run = run + 1 if v <= thresh else 0
        if run >= persist:
            first = steps[i - persist + 1]
            for row in rows:
                if row["method"] == method and row["step"] == first:
                    return row["t"]
    return math.inf


def write_summary_csv(rows, methods, seeds, metric_keys, method_info,
                      floors: dict, path: str) -> list[dict]:
    stats = _terminal_stats(rows, methods, seeds, metric_keys)
    diag_keys = ["mala_accept", "pt_swap_accept", "jump_count_mean",
                 "m_clip_fraction"]
    dstats = _terminal_stats(rows, methods, seeds, diag_keys)
    tv_floor = floors.get("TV", {}).get("mean", float("nan"))
    out_rows = []
    for method in methods:
        row = {"method": method}
        for key in metric_keys:
            if key in stats[method]:
                m, s = stats[method][key]
                row[f"{key}_mean"] = m
                row[f"{key}_std"] = s
        row["time_to_threshold_TV"] = time_to_threshold(rows, method, seeds, tv_floor)
        row.update({k: dstats[method][k][0] for k in diag_keys if k in dstats[method]})
        row.update(method_info.get(method, {}))
        out_rows.append(row)
    cols = sorted({k for r in out_rows for k in r}, key=lambda c: (c != "method", c))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    return out_rows


def write_manifest(path: str, **entries) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, torch.Tensor):
            return o.tolist()
        if isinstance(o, float) and math.isinf(o):
            return "inf"
        return str(o)

    with open(path, "w") as f:
        json.dump(entries, f, indent=2, default=_default)


# ------------------------------------------------------------ dt refinement
def refine_dt(run_terminal_fn, dt0: float, floors: dict, tol: float = 0.05,
              max_halvings: int = 4) -> tuple[float, list[dict]]:
    """Declared dt selection rule, applied uniformly: the largest dt on a
    dyadic grid at which EVERY method's terminal value of every metric is
    within `tol` of its dt/2 value. `run_terminal_fn(dt)` returns
    {method: {metric: value}}. Differences are measured relative to
    max(|value at dt/2|, bias floor): once a metric sits at its sampling
    floor, sub-floor differences are statistical noise, not bias.
    Returns (chosen dt, comparison table)."""
    table = []
    dt = dt0
    cache: dict[float, dict] = {}

    def get(d):
        if d not in cache:
            cache[d] = run_terminal_fn(d)
        return cache[d]

    for _ in range(max_halvings):
        vals, vals_half = get(dt), get(dt / 2.0)
        worst = 0.0
        worst_entry = None
        for method, metrics_d in vals.items():
            for metric, v in metrics_d.items():
                vh = vals_half[method][metric]
                denom = max(abs(vh), floors.get(metric, {}).get("mean", 0.0))
                rel = abs(v - vh) / denom if denom > 0 else 0.0
                if rel > worst:
                    worst, worst_entry = rel, (method, metric, v, vh)
        table.append({"dt": dt, "worst_rel_diff": worst,
                      "worst_case": worst_entry, "pass": worst <= tol})
        if worst <= tol:
            return dt, table
        dt = dt / 2.0
    return dt, table


# -------------------------------------------------- barrier verification
def ula_first_passage(pot, box, x0: torch.Tensor, in_basin_fn, dt: float,
                      n_steps: int, eps: float, gen: torch.Generator) -> dict:
    """Empirical ULA mean first-passage time out of the initial basin,
    censored-exponential MLE: tau_hat = (total time in basin) / (# exits).
    Compare with the Kramers estimate; do not trust Kramers alone."""
    from .samplers import tame
    x = x0.clone()
    n = x.shape[0]
    dev = x.device
    first_exit = torch.full((n,), math.inf, dtype=torch.float64, device=dev)
    noise = math.sqrt(2.0 * eps * dt)
    for s in range(n_steps):
        g = pot.grad(x)
        xi = torch.randn(x.shape, generator=gen, device=dev, dtype=x.dtype)
        x = box.clip(x + dt * tame(-g, dt) + noise * xi)
        outside = ~in_basin_fn(x)
        newly = outside & torch.isinf(first_exit)
        first_exit = torch.where(
            newly, torch.tensor((s + 1) * dt, dtype=torch.float64, device=dev),
            first_exit)
    T = n_steps * dt
    exited = torch.isfinite(first_exit)
    n_exit = int(exited.sum().item())
    total_time = float(torch.where(exited, first_exit,
                                   torch.tensor(T, device=dev)).sum().item())
    tau = total_time / n_exit if n_exit > 0 else math.inf
    return {"n_particles": n, "T": T, "n_exits": n_exit,
            "exit_fraction": n_exit / n, "mfpt_estimate": tau}
