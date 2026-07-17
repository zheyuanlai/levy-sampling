"""Checkpoint loop, defensible timing, CSV emission, dt refinement, and the
ULA first-passage barrier verification.

Timing policy (the wall-clock axis is a reported result):
* _cuda_synchronize() immediately before starting and stopping every
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
from importlib import metadata as importlib_metadata
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

MANUSCRIPT_METRIC_COLUMNS = (
    "FES_RMSE_kBT", "FES_outside_mass", "basin_KL_target",
    "basin_map_outside_mass", "nonfinite_count",
)

CUMULATIVE_DIAGNOSTIC_COLUMNS = (
    "state_box_clip_count_cumulative",
    "state_box_clip_fraction_cumulative",
    "outside_proposal_reject_count_cumulative",
    "outside_proposal_reject_fraction_cumulative",
    "nonfinite_proposal_count_cumulative",
    "nonfinite_proposal_fraction_cumulative",
    "score_clip_count_cumulative",
    "score_clip_fraction_cumulative",
    "mala_accept_count_cumulative",
    "mala_proposal_count_cumulative",
    "mala_accept_fraction_cumulative",
    "pt_swap_accept_count_cumulative",
    "pt_swap_proposal_count_cumulative",
    "pt_swap_accept_fraction_cumulative",
    "jump_count_cumulative",
    "jump_count_applied_cumulative",
    "jump_boundary_clip_count_cumulative",
    "jump_boundary_applied_count_cumulative",
    "jump_boundary_clip_fraction_per_applied_jump_cumulative",
    "jump_rate_per_particle_time_cumulative",
    "jump_applied_rate_per_particle_time_cumulative",
    "jump_cap_hit_count_cumulative",
    "jump_cap_hit_fraction_cumulative",
    "jump_cap_excess_count_cumulative",
)
MANIFEST_DIAGNOSTIC_COLUMNS = CUMULATIVE_DIAGNOSTIC_COLUMNS + (
    "jump_cap_k", "nonfinite_count", "nonfinite_frac",
)
CUMULATIVE_COUNT_COLUMNS = {
    key for key in MANIFEST_DIAGNOSTIC_COLUMNS
    if key.endswith("_count_cumulative")
} | {"jump_count_cumulative", "jump_count_applied_cumulative", "nonfinite_count"}

CSV_BASE_COLUMNS = [
    "method", "seed", "step", "t", "wallclock_s", "nfe",
    "W2", "TV", "TV_density", "MMD", "EMC", "EJS",
    "FES_RMSE_kBT", "FES_outside_mass", "basin_KL_target",
    "basin_map_outside_mass", "e_F", "basin_rel_max", "basin_L1", "V_mean_err", "V_var_err",
    "E_overlap_deficit", "KSD",
    "W1_cdf", "CDF_sup", "cdf_L2", "pdf_L1", "pdf_L2", "KDE_chi2",
    "bin_chi2_M40", "bin_chi2_M80", "bin_chi2_M120", "well_TV",
    "nonfinite_count", "nonfinite_frac",
    "state_box_clip_count_cumulative", "state_box_clip_fraction_cumulative",
    "outside_proposal_reject_count_cumulative",
    "outside_proposal_reject_fraction_cumulative",
    "nonfinite_proposal_count_cumulative",
    "nonfinite_proposal_fraction_cumulative",
    "score_clip_count_cumulative", "score_clip_fraction_cumulative",
    "mala_accept_count_cumulative", "mala_proposal_count_cumulative",
    "mala_accept_fraction_cumulative",
    "pt_swap_accept_count_cumulative", "pt_swap_proposal_count_cumulative",
    "pt_swap_accept_fraction_cumulative",
    "m_clip_fraction", "max_log_magnitude",
    "mala_accept", "pt_swap_accept", "jump_count_mean",
    "jump_count_cumulative", "jump_count_applied_cumulative",
    "jump_boundary_clip_count_cumulative",
    "jump_boundary_applied_count_cumulative",
    "jump_boundary_clip_fraction_per_applied_jump_cumulative",
    "jump_rate_per_particle_time_cumulative",
    "jump_applied_rate_per_particle_time_cumulative",
    "jump_cap_k", "jump_cap_hit_count_cumulative",
    "jump_cap_hit_fraction_cumulative", "jump_cap_excess_count_cumulative",
]


class RefinementError(RuntimeError):
    """A numerical refinement grid was exhausted without a certified choice.

    ``table`` preserves every attempted comparison for the run manifest, while
    ``next_candidate`` is only a suggestion for extending the grid.  It is not
    a certified setting and must not be used for production automatically.
    """

    def __init__(self, kind: str, table: list[dict],
                 next_candidate=None) -> None:
        self.kind = kind
        self.status = "failed"
        self.table = table
        self.next_candidate = next_candidate
        detail = f"{kind} refinement failed after {len(table)} comparison(s)"
        if next_candidate is not None:
            detail += f"; next unverified candidate is {next_candidate!r}"
        super().__init__(detail)


def _cuda_synchronize() -> None:
    """Synchronize CUDA timing when CUDA is available; otherwise a no-op."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _uniform_checkpoint_steps(n_steps: int, steps_per_ck: int) -> list[int]:
    """Uniform checkpoints including the final (possibly partial) interval."""
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if steps_per_ck < 1:
        raise ValueError("steps_per_ck must be positive")
    steps = list(range(steps_per_ck, n_steps + 1, steps_per_ck))
    if not steps or steps[-1] != n_steps:
        steps.append(n_steps)
    return steps


# ---------------------------------------------------------------- hardware
def _git_text(args: list[str]) -> str | None:
    """Return stripped git output, or ``None`` outside a usable work tree."""
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def hardware_manifest() -> dict:
    """Collect CPU-safe hardware, environment, and git provenance.

    This function is intentionally valid on hosts with a CPU-only torch build,
    no NVIDIA driver, no ``nvidia-smi``, or no enclosing git work tree.
    """
    cpu = platform.processor() or platform.machine() or "unknown"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    commit = _git_text(["rev-parse", "HEAD"])
    branch = _git_text(["rev-parse", "--abbrev-ref", "HEAD"])
    porcelain = _git_text(["status", "--porcelain"])

    cuda_available = bool(torch.cuda.is_available())
    gpu_count = 0
    gpu_name = None
    if cuda_available:
        try:
            gpu_count = int(torch.cuda.device_count())
            if gpu_count:
                gpu_name = torch.cuda.get_device_name(0)
        except (AssertionError, RuntimeError):
            cuda_available = False
            gpu_count = 0
            gpu_name = None

    python_packages = {}
    for package in ("numpy", "scipy", "pandas", "matplotlib", "nbformat"):
        try:
            python_packages[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            python_packages[package] = None

    cotenants: list[str] = []
    if cuda_available:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                 "--format=csv,noheader"],
                capture_output=True, text=True, check=False, timeout=5,
            )
            if result.returncode == 0:
                cotenants = [line for line in result.stdout.strip().splitlines() if line]
        except (OSError, subprocess.SubprocessError):
            pass

    return {
        "cpu": cpu,
        "gpu": gpu_name,
        "gpu_count_visible": gpu_count,
        "cuda_available": cuda_available,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "python_packages": python_packages,
        # Keep git_sha for readers of the previous schema; git_commit is the
        # canonical provenance field going forward.
        "git_commit": commit or "unknown",
        "git_sha": commit or "unknown",
        "git_branch": branch or "unknown",
        "git_dirty": None if porcelain is None else bool(porcelain),
        "git_status_porcelain": porcelain if porcelain is not None else "unknown",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "jcp_gpu": os.environ.get("JCP_GPU", ""),
        # Shared node: other processes at run start (wall-clock caveat).
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
    _cuda_synchronize()

    sampler = sampler_factory(method, seed)
    _cuda_synchronize()
    nV0, ng0, nq0 = potential.n_V, potential.n_grad, potential.n_Vdelta
    nfe0 = potential.nfe()

    rows: list[dict] = []
    wall = 0.0
    step = 0
    last_diagnostics: dict[str, float | int] = {}
    for ck_step in _uniform_checkpoint_steps(n_steps, steps_per_ck):
        _cuda_synchronize()
        t0 = time.perf_counter()
        for _ in range(ck_step - step):
            sampler.step()
        _cuda_synchronize()
        wall += time.perf_counter() - t0
        step = ck_step
        row = {"method": method, "seed": seed, "step": step,
               "t": step * dt, "wallclock_s": wall, "nfe": potential.nfe() - nfe0}
        with potential.no_count():
            row.update(metrics_fn(sampler.positions()))
        last_diagnostics = sampler.pop_diagnostics()
        row.update(last_diagnostics)
        rows.append(row)
    info = {
        "wallclock_total_s": wall,
        "V_evals_per_step": (potential.n_V - nV0) / n_steps,
        "grad_evals_per_step": (potential.n_grad - ng0) / n_steps,
        "score_quad_evals_per_step": (potential.n_Vdelta - nq0) / n_steps,
        "final_positions": sampler.positions().detach().clone(),
    }
    info.update({key: last_diagnostics[key] for key in MANIFEST_DIAGNOSTIC_COLUMNS
                 if key in last_diagnostics})
    if rows:
        info.update({key: rows[-1][key]
                     for key in ("nonfinite_count", "nonfinite_frac")
                     if key in rows[-1]})
    if not quiet:
        print(f"  {method} seed {seed}: {wall:.1f}s sampler wall-clock", flush=True)
    return rows, info


def checkpoint_schedule(n_steps: int, dense_frac: float = 0.05,
                        n_dense: int = 60, n_sparse: int = 160) -> list[int]:
    """Fixed non-uniform checkpoint schedule, identical across methods:
    n_dense points uniformly over the first dense_frac of the run (the
    nonlocal methods equilibrate within ~lambda^-1 time units, an order of
    magnitude faster than a uniform T/50 cadence can resolve), then
    n_sparse points to the end. (60+160 -- the earlier 40+48 left the
    post-equilibration 95% of the run with only ~one point per 2% of T,
    too sparse for the plotted curves and their running average.)"""
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if n_dense < 1 or n_sparse < 1:
        raise ValueError("n_dense and n_sparse must be positive")
    if not 0.0 <= dense_frac <= 1.0:
        raise ValueError("dense_frac must lie in [0, 1]")

    # For short smoke runs there may be fewer steps than requested points.  In
    # that case use each available integer step exactly once.  For production
    # runs this reduces to the declared dense-then-sparse schedule.
    dense_end = min(n_steps, max(n_dense, int(round(n_steps * dense_frac))))
    dense_count = min(n_dense, dense_end)
    dense = np.linspace(1, dense_end, dense_count)
    remaining = n_steps - dense_end
    sparse_count = min(n_sparse, remaining)
    sparse = (np.linspace(dense_end + 1, n_steps, sparse_count)
              if sparse_count else np.empty(0))
    steps = sorted(set(int(round(v)) for v in np.concatenate([dense, sparse])))
    if not steps or steps[0] < 1 or steps[-1] > n_steps:
        raise RuntimeError("internal checkpoint schedule error")
    if steps[-1] != n_steps:
        steps.append(n_steps)
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
        checkpoint_steps = _uniform_checkpoint_steps(n_steps, steps_per_ck)
    else:
        checkpoint_steps = list(checkpoint_steps)
        if (not checkpoint_steps
                or any(isinstance(s, bool) or not isinstance(s, (int, np.integer))
                       for s in checkpoint_steps)
                or any(s < 1 or s > n_steps for s in checkpoint_steps)
                or any(b <= a for a, b in zip(checkpoint_steps, checkpoint_steps[1:]))):
            raise ValueError(
                "checkpoint_steps must be strictly increasing integers in [1, n_steps]"
            )
        if checkpoint_steps[-1] != n_steps:
            checkpoint_steps.append(n_steps)
    for method in methods:
        warm = batched_factory(method)
        for _ in range(warmup):
            warm.step()
        del warm
        _cuda_synchronize()
        sampler = batched_factory(method)
        _cuda_synchronize()
        nV0, ng0, nq0 = potential.n_V, potential.n_grad, potential.n_Vdelta
        nfe0 = potential.nfe()
        wall = 0.0
        step = 0
        last_diagnostics: dict[str, float | int] = {}
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
            _cuda_synchronize()
            t0 = time.perf_counter()
            for _ in range(ck_step - step):
                sampler.step()
            _cuda_synchronize()
            wall += time.perf_counter() - t0
            step = ck_step
            pos = sampler.positions()
            diag = sampler.pop_diagnostics()
            last_diagnostics = diag
            # Cumulative counts describe the entire batched sampler, not any
            # individual seed block. Keep them once in method_info/manifest;
            # fractions/rates remain meaningful shared checkpoint diagnostics.
            row_diag = {key: value for key, value in diag.items()
                        if key not in CUMULATIVE_COUNT_COLUMNS}
            nfe = potential.nfe() - nfe0
            for si, seed in enumerate(seeds):
                row = {"method": method, "seed": seed, "step": step,
                       "t": step * dt, "wallclock_s": wall, "nfe": nfe}
                with potential.no_count():
                    row.update(metrics_fn(pos[si * n_per_seed:(si + 1) * n_per_seed]))
                row.update(row_diag)
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
        method_info[method].update({
            key: last_diagnostics[key] for key in MANIFEST_DIAGNOSTIC_COLUMNS
            if key in last_diagnostics
        })
        terminal_rows = [row for row in all_rows
                         if row["method"] == method and row["step"] == n_steps]
        if terminal_rows:
            method_info[method]["nonfinite_count"] = sum(
                int(row.get("nonfinite_count", 0)) for row in terminal_rows)
            fractions = [row["nonfinite_frac"] for row in terminal_rows
                         if "nonfinite_frac" in row]
            if fractions:
                method_info[method]["nonfinite_frac"] = float(np.mean(fractions))
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
        for key in MANIFEST_DIAGNOSTIC_COLUMNS:
            values = [info[key] for info in infos if key in info]
            if not values:
                continue
            if key in CUMULATIVE_COUNT_COLUMNS:
                method_info[method][key] = sum(values)
            elif key == "jump_cap_k":
                method_info[method][key] = values[0]
            else:
                method_info[method][key] = float(np.mean(values))
        print(f"{method}: done in {time.perf_counter() - t_m:.1f}s total", flush=True)
    return all_rows, method_info


# ---------------------------------------------------------------- CSV / IO
def write_timeseries_csv(rows: list[dict], path: str | os.PathLike,
                         *, overwrite: bool = False) -> list[str]:
    """Write checkpoint rows, refusing to replace an artifact by default."""
    extras = sorted({k for r in rows for k in r} - set(CSV_BASE_COLUMNS))
    cols = CSV_BASE_COLUMNS + extras
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with output.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, restval="")
        writer.writeheader()
        writer.writerows(rows)
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
    requested_seeds = tuple(seeds)
    seed_set = set(requested_seeds)
    if not requested_seeds:
        return math.inf
    by_step: dict[int, dict[int, float]] = {}
    time_by_step: dict[int, float] = {}
    for row in rows:
        if (row["method"] == method and row.get("seed") in seed_set
                and key in row):
            by_step.setdefault(row["step"], {})[row["seed"]] = row[key]
            time_by_step[row["step"]] = row["t"]
    # A seed mean is comparable across time only when every requested seed is
    # represented.  Ignore incomplete checkpoints rather than changing the
    # estimator's sample size silently.
    steps = sorted(s for s, values in by_step.items()
                   if seed_set.issubset(values))
    mean_curve = [float(np.mean([by_step[s][seed] for seed in requested_seeds]))
                  for s in steps]
    thresh = 2.0 * floor
    run = 0
    for i, v in enumerate(mean_curve):
        run = run + 1 if v <= thresh else 0
        if run >= persist:
            first = steps[i - persist + 1]
            return time_by_step[first]
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
                      floors: dict, path: str | os.PathLike, *,
                      overwrite: bool = False) -> list[dict]:
    # Keep legacy caller-selected summaries, but automatically carry the
    # manuscript FES/KL metrics whenever the time-series rows provide them.
    metric_keys = list(metric_keys)
    available = {key for row in rows for key in row}
    metric_keys.extend(
        key for key in MANUSCRIPT_METRIC_COLUMNS
        if key in available and key not in metric_keys
    )
    stats = _terminal_stats(rows, methods, seeds, metric_keys)
    diag_keys = [
        "mala_accept", "pt_swap_accept", "jump_count_mean", "m_clip_fraction",
        *CUMULATIVE_DIAGNOSTIC_COLUMNS, "jump_cap_k",
    ]
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
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with output.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, restval="")
        writer.writeheader()
        writer.writerows(out_rows)
    return out_rows


def _json_safe(value):
    """Convert scientific values to strict, portable JSON recursively."""
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, (np.floating, np.integer)):
        return _json_safe(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def write_manifest(path: str | os.PathLike, *, overwrite: bool = False,
                   **entries) -> None:
    """Serialize a strict JSON manifest, refusing overwrite by default."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Serialize before opening the destination so a serialization error cannot
    # leave behind a truncated artifact that blocks a corrected rerun. JSON's
    # non-standard NaN/Infinity literals are disabled; scientific sentinels are
    # explicit strings instead.
    payload = json.dumps(_json_safe(entries), indent=2, allow_nan=False)
    mode = "w" if overwrite else "x"
    with output.open(mode, encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")


# ---------------------------------------------------- quadrature refinement
def quadrature_refinement(settings: list[dict], run_terminal_fn, cert_fn,
                          floors: dict, tol: float = 0.05,
                          r_max: float = 1e-6) -> tuple[dict, list[dict]]:
    """Section 9.5: for each quadrature setting, record the certificate
    residual R and the terminal LSC-CP metrics; pick the SMALLEST setting
    whose R < r_max and whose every metric is within `tol` (floor-guarded)
    of the finest setting's value. Returns (chosen setting, table)."""
    if not settings:
        raise ValueError("settings must contain at least one quadrature configuration")
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
    if chosen is None:
        candidate = {k: finest[k] for k in settings[0]}
        raise RefinementError("quadrature", table, next_candidate=candidate)
    return chosen, table


def _metrics_agree(v: float, v_ref: float, floor: dict, tol: float,
                   atol: float = 1e-12) -> bool:
    """Two terminal values agree if (a) both sit inside the finite-sample
    floor band (mean + 3 sd) -- differences there are statistical noise, not
    bias -- or (b) their difference is within the metric's own replicate
    noise (4x the floor std, the natural unit of single-run sampling noise
    at this N) -- or (c) they differ by < tol relative to
    max(|v_ref|, floor)."""
    if not math.isfinite(v) or not math.isfinite(v_ref):
        return False
    f_mean = max(float(floor.get("mean", 0.0)), 0.0)
    f_std = max(float(floor.get("std", 0.0)), 0.0)
    f_hi = f_mean + 3.0 * f_std
    if f_hi > 0 and v <= f_hi and v_ref <= f_hi:
        return True
    difference = abs(v - v_ref)
    scale = max(abs(v), abs(v_ref), f_mean)
    # The absolute term covers a genuinely zero reference; unlike the old
    # ``denom <= 0`` shortcut it cannot approve an arbitrary discrepancy.
    return difference <= atol + 4.0 * f_std + tol * scale


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
    if dt0 <= 0:
        raise ValueError("dt0 must be positive")
    if max_halvings < 1:
        raise ValueError("max_halvings must be positive")
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
    raise RefinementError("timestep", table, next_candidate=dt)


# -------------------------------------------------- barrier verification
def _kaplan_meier_rmst(times: np.ndarray, events: np.ndarray,
                       horizon: float) -> float:
    """Kaplan-Meier restricted mean survival time on ``[0, horizon]``."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=bool)
    if times.ndim != 1 or events.shape != times.shape or times.size == 0:
        raise ValueError("times/events must be nonempty one-dimensional arrays")
    if horizon <= 0 or np.any(~np.isfinite(times)) or np.any(times < 0):
        raise ValueError("horizon and observed times must be finite and nonnegative")
    observed = np.minimum(times, horizon)
    event_times = np.unique(observed[events & (observed <= horizon)])
    survival = 1.0
    area = 0.0
    previous = 0.0
    for event_time in event_times:
        area += survival * (event_time - previous)
        at_risk = int(np.count_nonzero(observed >= event_time))
        event_count = int(np.count_nonzero(events & (observed == event_time)))
        if at_risk:
            survival *= 1.0 - event_count / at_risk
        previous = float(event_time)
    area += survival * (horizon - previous)
    return float(area)


def ula_first_passage(pot, box, x0: torch.Tensor, exit_fn, dt: float,
                      n_steps: int, eps: float, gen: torch.Generator) -> dict:
    """Committed-exit observations with explicit right-censoring estimators.

    ``exponential_waiting_time_mle`` is total exposure divided by event count,
    which is valid under a constant-hazard exponential waiting-time model. It
    is not a general empirical MFPT. ``kaplan_meier_rmst_at_horizon`` is the
    nonparametric restricted mean first-passage time through the run horizon.
    """
    from .samplers import tame
    x = x0.clone()
    n = x.shape[0]
    dev = x.device
    first_exit = torch.full((n,), math.inf, dtype=torch.float64, device=dev)
    noise = math.sqrt(2.0 * eps * dt)
    for step_index in range(n_steps):
        g = pot.grad(x)
        xi = torch.randn(x.shape, generator=gen, device=dev, dtype=x.dtype)
        x = box.clip(x + dt * tame(-g, dt) + noise * xi)
        newly = exit_fn(x) & torch.isinf(first_exit)
        first_exit = torch.where(
            newly,
            torch.tensor((step_index + 1) * dt, dtype=torch.float64, device=dev),
            first_exit,
        )
    horizon = n_steps * dt
    exited = torch.isfinite(first_exit)
    event_count = int(exited.sum().item())
    observed = torch.where(
        exited, first_exit,
        torch.tensor(horizon, dtype=torch.float64, device=dev),
    )
    total_exposure = float(observed.sum().item())
    exponential_mle = total_exposure / event_count if event_count else math.inf
    observed_np = observed.detach().cpu().numpy()
    events_np = exited.detach().cpu().numpy()
    rmst = _kaplan_meier_rmst(observed_np, events_np, horizon)
    censored_count = n - event_count
    return {
        "n_particles": n,
        "horizon": horizon,
        "T": horizon,  # legacy horizon field
        "event_count": event_count,
        "n_exits": event_count,  # legacy count field
        "censored_count": censored_count,
        "event_fraction": event_count / n,
        "exit_fraction": event_count / n,  # legacy fraction field
        "censoring_fraction": censored_count / n,
        "total_exposure_time": total_exposure,
        "exponential_waiting_time_mle": exponential_mle,
        "kaplan_meier_rmst_at_horizon": rmst,
        # Compatibility for generated notebooks; this alias should not be
        # described as an empirical MFPT in new text.
        "mfpt_estimate": exponential_mle,
        "mfpt_estimate_definition": "legacy alias of exponential_waiting_time_mle",
    }
