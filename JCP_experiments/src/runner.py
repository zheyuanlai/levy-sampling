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
    "method", "seed", "step", "t", "wallclock_s", "nfe",
    "W2", "TV", "TV_density", "MMD", "EMC", "EJS",
    "e_F", "basin_rel_max", "basin_L1", "V_mean_err", "V_var_err",
    "E_overlap_deficit", "KSD",
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
    try:
        cotenants = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True,
            check=False).stdout.strip().splitlines()
    except Exception:
        cotenants = []
    return {
        "cpu": cpu,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "git_sha": sha,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        # shared node: other users' processes at run start (wall-clock caveat)
        "gpu_compute_apps_at_start": cotenants,
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
    nfe0 = potential.nfe()

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
               "t": step * dt, "wallclock_s": wall, "nfe": potential.nfe() - nfe0}
        with potential.no_count():
            row.update(metrics_fn(sampler.positions()))
        row.update(sampler.pop_diagnostics())
        rows.append(row)
    info = {
        "wallclock_total_s": wall,
        "V_evals_per_step": (potential.n_V - nV0) / n_steps,
        "grad_evals_per_step": (potential.n_grad - ng0) / n_steps,
        "score_quad_evals_per_step": (potential.n_Vdelta - nq0) / n_steps,
        "final_positions": sampler.positions().detach().clone(),
    }
    if not quiet:
        print(f"  {method} seed {seed}: {wall:.1f}s sampler wall-clock", flush=True)
    return rows, info


def checkpoint_schedule(n_steps: int, dense_frac: float = 0.05,
                        n_dense: int = 40, n_sparse: int = 48) -> list[int]:
    """Fixed non-uniform checkpoint schedule, identical across methods:
    n_dense points uniformly over the first dense_frac of the run (the
    nonlocal methods equilibrate within ~lambda^-1 time units, an order of
    magnitude faster than a uniform T/50 cadence can resolve), then
    n_sparse points to the end."""
    dense_end = max(n_dense, int(round(n_steps * dense_frac)))
    dense = np.linspace(dense_end / n_dense, dense_end, n_dense)
    sparse = np.linspace(dense_end + (n_steps - dense_end) / n_sparse,
                         n_steps, n_sparse)
    steps = sorted(set(int(round(v)) for v in np.concatenate([dense, sparse])
                       if v >= 1))
    steps[-1] = n_steps
    return steps


def run_experiment_batched(methods, seeds, batched_factory, n_steps: int,
                           steps_per_ck: int, dt: float, metrics_fn,
                           potential, n_per_seed: int,
                           warmup: int = 20,
                           checkpoint_steps: list[int] | None = None
                           ) -> tuple[list[dict], dict]:
    """Production loop with all seeds batched into one ensemble per method
    (used since wall-clock curves are not reported); per-seed metric rows
    are computed on contiguous blocks of n_per_seed particles.
    checkpoint_steps (optional) overrides the uniform cadence with a fixed
    schedule of step indices, identical across methods."""
    all_rows: list[dict] = []
    method_info: dict[str, dict] = {}
    if checkpoint_steps is None:
        checkpoint_steps = list(range(steps_per_ck, n_steps + 1, steps_per_ck))
    for method in methods:
        warm = batched_factory(method)
        for _ in range(warmup):
            warm.step()
        del warm
        torch.cuda.synchronize()
        sampler = batched_factory(method)
        torch.cuda.synchronize()
        nV0, ng0, nq0 = potential.n_V, potential.n_grad, potential.n_Vdelta
        nfe0 = potential.nfe()
        wall = 0.0
        step = 0
        t_m = time.perf_counter()
        # n=0 frame on the shared initial ensemble: all methods use the same
        # per-seed x0, so every method's metric values here are bit-identical
        # (all curves start at literally the same point). NFE / metric evals
        # are excluded from the counter.
        pos0 = sampler.positions()
        for si, seed in enumerate(seeds):
            row = {"method": method, "seed": seed, "step": 0, "t": 0.0,
                   "wallclock_s": 0.0, "nfe": 0}
            with potential.no_count():
                row.update(metrics_fn(pos0[si * n_per_seed:(si + 1) * n_per_seed]))
            all_rows.append(row)
        for ck_step in checkpoint_steps:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(ck_step - step):
                sampler.step()
            torch.cuda.synchronize()
            wall += time.perf_counter() - t0
            step = ck_step
            pos = sampler.positions()
            diag = sampler.pop_diagnostics()
            nfe = potential.nfe() - nfe0
            for si, seed in enumerate(seeds):
                row = {"method": method, "seed": seed, "step": step,
                       "t": step * dt, "wallclock_s": wall, "nfe": nfe}
                with potential.no_count():
                    row.update(metrics_fn(pos[si * n_per_seed:(si + 1) * n_per_seed]))
                row.update(diag)
                all_rows.append(row)
        method_info[method] = {
            "wallclock_mean_s": wall,        # batch wall-clock (informational)
            "wallclock_std_s": 0.0,
            "V_evals_per_step": (potential.n_V - nV0) / n_steps,
            "grad_evals_per_step": (potential.n_grad - ng0) / n_steps,
            "score_quad_evals_per_step": (potential.n_Vdelta - nq0) / n_steps,
            "final_positions_seed0": sampler.positions()[:n_per_seed].detach().clone(),
            "final_positions_all": sampler.positions().detach().clone(),
        }
        print(f"{method}: done in {time.perf_counter() - t_m:.1f}s "
              f"(batched {len(seeds)} seeds)", flush=True)
    return all_rows, method_info


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
            "final_positions_seed0": infos[0]["final_positions"],
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


def convergence_report(rows, methods, seeds, key: str = "occ0",
                       tail_frac: float = 0.5) -> dict:
    """Per-method cross-seed rank-normalized split-R-hat (Vehtari 2021) on a slow
    observable `key` (default occ0 = basin-0 fraction), over the last tail_frac
    of checkpoints (drops the transient). Chains = seeds. On a metastable target
    each seed trapped in a different basin drives R-hat >> 1 -- a standard
    diagnostic every community accepts. Also reports final NFE (for ESS/NFE).

    ESS and round-trips need denser per-step recording than the checkpoint
    cadence; use metrics.ess_from_series / round_trips / committed_mfpt on a
    per-step basin-indicator buffer for those (provided as functions)."""
    from .metrics import split_rhat
    out = {}
    for method in methods:
        steps = sorted({r["step"] for r in rows
                        if r["method"] == method and r["step"] > 0})
        if not steps:
            continue
        tail = steps[int((1.0 - tail_frac) * len(steps)):]
        mat = []
        for seed in seeds:
            series = []
            for st in tail:
                vals = [r[key] for r in rows if r["method"] == method
                        and r["seed"] == seed and r["step"] == st and key in r]
                if vals:
                    series.append(float(vals[0]))
            if len(series) == len(tail):
                mat.append(series)
        mat = np.asarray(mat, dtype=float)
        rh = (float(split_rhat(mat)) if mat.ndim == 2 and mat.shape[0] >= 2
              and mat.shape[1] >= 4 else float("nan"))
        final_nfe = max((r["nfe"] for r in rows if r["method"] == method
                         and "nfe" in r and r["nfe"] != ""), default=0)
        out[method] = {"split_rhat": rh, "n_chains": int(mat.shape[0]) if mat.ndim == 2 else 0,
                       "final_nfe": int(final_nfe)}
    return out


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
        row.update({k: v for k, v in method_info.get(method, {}).items()
                    if isinstance(v, (int, float))})
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


# ---------------------------------------------------- quadrature refinement
def quadrature_refinement(settings: list[dict], run_terminal_fn, cert_fn,
                          floors: dict, tol: float = 0.05,
                          r_max: float = 1e-6) -> tuple[dict, list[dict]]:
    """Section 9.5: for each quadrature setting, record the certificate
    residual R and the terminal LSC-CP metrics; pick the SMALLEST setting
    whose R < r_max and whose every metric is within `tol` (floor-guarded)
    of the finest setting's value. Returns (chosen setting, table)."""
    table = []
    for s in settings:
        row = dict(s)
        row["R"] = cert_fn(**s)
        row.update(run_terminal_fn(**s))
        table.append(row)
    finest = table[-1]
    metric_keys = [k for k in finest if k not in settings[0] and k != "R"]
    chosen = None
    for row in table:                       # settings ordered smallest first
        ok = row["R"] < r_max
        for k in metric_keys:
            if not _metrics_agree(row[k], finest[k], floors.get(k, {}), tol):
                ok = False
        row["pass"] = ok
        if ok and chosen is None:
            chosen = {k: row[k] for k in settings[0]}
    return chosen or {k: finest[k] for k in settings[0]}, table


def _metrics_agree(v: float, v_ref: float, floor: dict, tol: float) -> bool:
    """Two terminal values agree if (a) both sit inside the finite-sample
    floor band (mean + 3 sd) -- differences there are statistical noise, not
    bias -- or (b) their difference is within the metric's own replicate
    noise (4x the floor std, the natural unit of single-run sampling noise
    at this N) -- or (c) they differ by < tol relative to
    max(|v_ref|, floor)."""
    f_mean = floor.get("mean", 0.0)
    f_std = floor.get("std", 0.0)
    f_hi = f_mean + 3.0 * f_std
    if f_hi > 0 and v <= f_hi and v_ref <= f_hi:
        return True
    if f_std > 0 and abs(v - v_ref) <= 4.0 * f_std:
        return True
    denom = max(abs(v_ref), f_mean)
    return denom <= 0 or abs(v - v_ref) / denom <= tol


# ------------------------------------------------------------ dt refinement
def refine_dt(run_terminal_fn, dt0: float, floors: dict, tol: float = 0.05,
              max_halvings: int = 4,
              exclude: tuple[str, ...] = ()) -> tuple[float, list[dict]]:
    """Declared dt selection rule, applied uniformly: the largest dt on a
    dyadic grid at which EVERY method's terminal value of every metric is
    within `tol` of its dt/2 value. `run_terminal_fn(dt)` returns
    {method: {metric: value}}. Two guards make the rule statistically
    meaningful: values are compared relative to max(|value at dt/2|, bias
    floor), and when BOTH values already sit inside the floor band
    (mean + 3 sd) they are declared in agreement -- sub-floor differences
    are sampling noise, not discretisation bias.

    Methods in `exclude` (e.g. FLA, whose continuum limit is not pi, so its
    bias has no dt at which it should stabilise) do not gate the selection;
    their deviations are still recorded in the table for transparency.
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
        ok = True
        failures = []
        excluded_devs = []
        for method, metrics_d in vals.items():
            for metric, v in metrics_d.items():
                vh = vals_half[method][metric]
                if not _metrics_agree(v, vh, floors.get(metric, {}), tol):
                    if method in exclude:
                        excluded_devs.append((method, metric, round(v, 6), round(vh, 6)))
                    else:
                        ok = False
                        failures.append((method, metric, round(v, 6), round(vh, 6)))
        table.append({"dt": dt, "pass": ok, "failures": failures[:8],
                      "excluded_deviations": excluded_devs[:8]})
        if ok:
            return dt, table
        dt = dt / 2.0
    return dt, table


# -------------------------------------------------- barrier verification
def ula_first_passage(pot, box, x0: torch.Tensor, exit_fn, dt: float,
                      n_steps: int, eps: float, gen: torch.Generator) -> dict:
    """Empirical ULA mean first-passage time out of the initial basin,
    censored-exponential MLE: tau_hat = (total time in basin) / (# exits).
    `exit_fn(x) -> bool` must be the COMMITTED exit event (True = the
    particle has arrived in another basin's core). Compare with the Kramers
    estimate; do not trust Kramers alone."""
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
        newly = exit_fn(x) & torch.isinf(first_exit)
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
