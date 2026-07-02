
from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .io_utils import Tee, environment_info, make_run_dir, write_json
from .jump_banks import FiniteJumpBank, build_jump_bank
from .metrics import compute_metric_bundle
from .potentials import build_potential
from .samplers import SAMPLERS, LevyScoreJumpDiffusion, torch_generator


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _as_float(v):
    try:
        x = float(v)
    except Exception:
        return None
    return x if np.isfinite(x) else None


def _summarize(rows: list[dict], group_cols: list[str]) -> list[dict]:
    groups = {}
    for r in rows:
        key = tuple(r.get(c, "") for c in group_cols)
        groups.setdefault(key, []).append(r)
    out = []
    for key, sub in sorted(groups.items()):
        rec = dict(zip(group_cols, key)); rec["n_rows"] = len(sub)
        numeric_keys = sorted({k for r in sub for k, v in r.items() if _as_float(v) is not None and k not in set(group_cols + ["seed"])})
        for m in numeric_keys:
            vals = [_as_float(r.get(m)) for r in sub]
            vals = np.array([v for v in vals if v is not None], dtype=float)
            if vals.size:
                rec[f"{m}_mean"] = float(vals.mean())
                rec[f"{m}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
                rec[f"{m}_se"] = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
            else:
                rec[f"{m}_mean"] = float("nan"); rec[f"{m}_std"] = float("nan"); rec[f"{m}_se"] = float("nan")
        out.append(rec)
    return out


def _build_sampler(method: str, potential, bank: FiniteJumpBank | None, cfg: dict, run_cfg: dict):
    mcfg = dict(cfg.get("method_cfgs", {}).get(method, {}))
    mcfg.pop("n_steps", None)  # consumed in _run_seed, not a sampler kwarg
    dt = float(mcfg.pop("dt", run_cfg.get("dt", 0.01)))
    if method == "LSBMC":
        return LevyScoreJumpDiffusion(potential, dt, bank=bank, use_score=True, **mcfg)
    if method == "RawCP":
        return LevyScoreJumpDiffusion(potential, dt, bank=bank, use_score=False, **mcfg)
    cls = SAMPLERS[method]
    return cls(potential, dt, **mcfg)


# Methods that carry a compound-Poisson jump bank (LSC-CP + its no-score control).
JUMP_METHODS = ("LSBMC", "RawCP")


def _run_seed(potential, method: str, bank, cfg: dict, run_cfg: dict, ref: torch.Tensor, seed: int, device, log):
    sampler = _build_sampler(method, potential, bank, cfg, run_cfg)
    n_particles = int(run_cfg.get("n_particles", 256))
    n_steps = int(cfg.get("method_cfgs", {}).get(method, {}).get("n_steps", run_cfg.get("n_steps", 100)))
    record_every = int(run_cfg.get("record_every", max(1, n_steps // 10)))
    # Stride at which we snapshot per-chain basin labels for the IAT/ESS time series.
    iat_points = int(run_cfg.get("iat_points", 2000))
    iat_stride = max(1, n_steps // max(1, iat_points))
    # Common-random-numbers: raw CP and LSC-CP share one RNG stream (identical Brownian noise and
    # jump realizations) so their comparison isolates the stationary drift correction.
    method_key = "JUMPDIFF" if method in JUMP_METHODS else method
    gen = torch_generator(seed + 1009 + sum(ord(c) for c in method_key), device)
    x = sampler.init_state(n_particles, seed, device)
    # Optional start-at-equilibrium init (E1 timestep-bias / E5 invariance): initialize the ensemble
    # from the Gibbs reference so residual drift over the horizon isolates the invariant-law defect
    # (raw CP) and the discretization bias (all methods) from the metastable equilibration transient.
    if str(run_cfg.get("init", "")).lower() in ("equilibrium", "reference"):
        x = potential.reference(n_particles, seed + 5000, device)
    tp = potential.target_basin_probs(device=device)
    tp_np = tp.detach().cpu().numpy() if tp is not None else None
    label_hist = []  # list of (N,) int arrays: basin label of every chain over time
    cv_series = []   # ensemble mean of the continuous slow CV at each recorded time
    lab0 = None
    ts_rows = []
    t0 = time.time()
    status = "ok"; failure = ""
    try:
        with torch.no_grad():
            for step in range(n_steps + 1):
                do_iat = (step % iat_stride == 0)
                do_ts = (step % record_every == 0)
                if do_iat or do_ts:
                    fs = sampler.final_samples(x)
                    labels = potential.basin_labels(fs).detach().cpu().numpy().reshape(-1).astype(np.int32)
                    if lab0 is None:
                        lab0 = labels
                    if do_iat:
                        label_hist.append(labels)
                        cv_series.append(float(potential.slow_cv(fs).mean().item()))
                    if do_ts:
                        escaped = float(np.mean(labels != lab0)) if labels.size else np.nan
                        btv = np.nan
                        if tp_np is not None and labels.size:
                            counts = np.bincount(np.clip(labels, 0, tp_np.size - 1), minlength=tp_np.size)[:tp_np.size].astype(float)
                            emp = counts / max(counts.sum(), 1.0)
                            btv = float(0.5 * np.abs(emp - tp_np).sum())
                        ts_rows.append({"seed": seed, "method": method, "bank_name": bank.name if bank else "none",
                                        "step": step, "time": step * sampler.dt,
                                        "escaped_fraction": escaped, "basin_tv": btv,
                                        "trace_mean_label": float(np.mean(labels)) if labels.size else np.nan})
                        if int((~torch.isfinite(fs)).sum().item()) > int(run_cfg.get("max_nonfinite", 0)):
                            raise FloatingPointError("nonfinite state threshold exceeded")
                if step < n_steps:
                    x, _ = sampler.step(x, gen)
    except Exception as e:
        status = "failed"; failure = repr(e); log(f"  [failed] {method}/{bank.name if bank else 'none'}/seed={seed}: {failure}")
    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime = time.time() - t0
    fs = sampler.final_samples(x).detach()
    diag = sampler.diag.as_dict()
    hist = np.stack(label_hist, axis=0) if label_hist else np.zeros((1, n_particles), dtype=np.int32)
    cv = np.asarray(cv_series, dtype=float) if cv_series else np.zeros(1)
    row = compute_metric_bundle(potential, fs, ref, hist, cv, diag, runtime, iat_stride=iat_stride, dt=sampler.dt)
    bank_meta = bank.metadata if bank else {}
    row.update({"seed": int(seed), "method": method, "bank_name": bank.name if bank else "none", "status": status,
                "failure_reason": failure, "target_name": potential.name, "dimension": potential.dim,
                "beta": potential.beta, "n_steps": n_steps, "dt": sampler.dt,
                "bank_intensity": float(bank.intensity) if bank else 0.0,
                "bank_scale": bank_meta.get("scale", ""), "bank_displacement": bank_meta.get("displacement", "")})
    return row, ts_rows, fs.detach().cpu().numpy()


def run_config(config_path: str | Path, output_root: str = "results/jcp_sampling", tag: str | None = None,
               device: str | None = None) -> Path:
    cfg = load_config(config_path)
    tag = tag or cfg.get("experiment_name", Path(config_path).stem)
    run_dir = make_run_dir(output_root, tag)
    log = Tee(run_dir / "logs" / "run.log")
    log(f"=== JCP run: {tag} ===")
    log(f"config: {config_path}")
    shutil.copy(config_path, run_dir / "configs" / Path(config_path).name)
    write_json(run_dir / "configs" / "resolved_config.json", cfg)
    env = environment_info(); write_json(run_dir / "environment.json", env)
    log(f"env: {env}")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    potential = build_potential({"kind": cfg["target"], "target_cfg": cfg.get("target_cfg", {})})
    log(f"target: {potential.metadata()}")
    ref_cfg = cfg.get("reference", {})
    ref = potential.reference(int(ref_cfg.get("n_ref", cfg.get("run", {}).get("n_ref", 2000))), int(ref_cfg.get("seed", 123)), dev)
    np.savez(run_dir / "samples" / "reference.npz", ref=ref.detach().cpu().numpy())

    banks = [build_jump_bank(b.get("kind"), potential, b) for b in cfg.get("jump_banks", [{"kind": "none", "intensity": 0.0}])]
    none_bank = FiniteJumpBank("none", torch.zeros(1, potential.dim), torch.ones(1), 0.0)
    rows, ts_all = [], []
    methods = cfg.get("methods", ["ULA", "LSBMC"])
    seeds = cfg.get("run", {}).get("seeds", [0])
    for method in methods:
        method_banks = banks if method in JUMP_METHODS else [none_bank]
        for bank in method_banks:
            for seed in seeds:
                log(f"running method={method} bank={bank.name} seed={seed}")
                row, ts, final = _run_seed(potential, method, bank, cfg, cfg.get("run", {}), ref, int(seed), dev, log)
                row.update({"experiment_name": cfg.get("experiment_name", tag), "config_path": str(config_path)})
                rows.append(row); ts_all.extend(ts)
                np.savez(run_dir / "samples" / f"final_{method}_{bank.name}_seed{seed}.npz", samples=final)
    _write_csv(run_dir / "raw_metrics.csv", rows)
    _write_csv(run_dir / "timeseries.csv", ts_all)
    summary = _summarize(rows, ["experiment_name", "target_name", "method", "bank_name", "bank_scale", "bank_intensity"])
    _write_csv(run_dir / "summary_by_method.csv", summary)
    n_failed = sum(1 for r in rows if r.get("status") != "ok")
    status = {"status": "ok" if n_failed == 0 else "failed", "n_rows": len(rows), "n_failed": int(n_failed), "run_dir": str(run_dir)}
    write_json(run_dir / "run_status.json", status)
    (run_dir / "README.md").write_text(f"# {tag}\n\nConfig: `{config_path}`\n\nStatus: `{status['status']}`\n")
    log(f"DONE {run_dir} status={status['status']}")
    log.close()
    return run_dir
