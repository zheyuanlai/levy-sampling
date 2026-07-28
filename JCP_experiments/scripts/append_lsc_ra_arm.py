"""Run one LSC arm for E3/E4 through the exact notebook data path and (with
--append) append its rows to the committed results tree.

Motivation: E3 (mb3well_10d) and E4 (coupled_phi4) deployed only the multi-atom
realised estimator LSC-CP-MA, displayed as "LSC-CP-RA (A)". This adds the
genuine single-atom LSC-CP-RA arm (A=1 member of the same family) as a
comparison, WITHOUT recomputing the eight already-valid methods: the run is
deterministic in the fixed seeds, so an existing method reproduces bit-for-bit,
which is exactly how --validate proves the data path before --append trusts it.

The arm is run at the committed dt and quadrature (no re-refinement), so it is
directly comparable to the deployed methods.

Usage:
  # prove the path: reproduce the committed LSC-CP-MA row
  python -m scripts.append_lsc_ra_arm --experiment mb3well_10d --method LSC-CP-MA --validate
  # produce and append the genuine RA arm
  python -m scripts.append_lsc_ra_arm --experiment mb3well_10d --method LSC-CP-RA --append
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JCP_ROOT = HERE.parent
sys.path.insert(0, str(JCP_ROOT))

import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from src.experiments import (build_e3, build_e4, make_batched_factory,  # noqa: E402
                             make_sampler_factory, make_metrics)
from src.runner import (run_experiment_batched, checkpoint_schedule,  # noqa: E402
                        write_summary_csv, write_positions_csv,
                        write_timeseries_csv)
from src.stationarity import (collect_stationary_trajectories,  # noqa: E402
                              flat_summary_rows, write_stationarity_csv)
from src.samplers import geometric_ladder  # noqa: E402
import src.config as C  # noqa: E402

CFG = {
    "mb3well_10d": dict(
        build=build_e3, cache="cache/mb3well_10d/basin_map_v2.npz",
        quad=dict(q_theta=8, q_rho=4),
        main=["W2", "TV", "MMD", "EMC", "W2_10d"]),
    "coupled_phi4": dict(
        build=build_e4, cache="cache/coupled_phi4/basin_map_v2.npz",
        quad=dict(q_theta=32, q_rho=4),
        main=["W2", "TV", "MMD", "EMC"]),
}


def run_arm(experiment: str, method: str):
    """Return (summary_row, timeseries_rows, positions_tensor) for one method,
    computed exactly as the production notebook does."""
    c = CFG[experiment]
    exp = c["build"](device="cuda", basin_cache=str(JCP_ROOT / c["cache"]))
    cfg = exp.cfg
    metrics_fn, floors, aux = make_metrics(exp, cfg.n_particles)
    pt_betas = geometric_ladder(cfg.beta, exp.pt_beta_min, 6, exp.p_star.device)
    dt = cfg.dt                                   # committed == default (1-row refine)
    n_steps = int(round(cfg.T / dt))
    steps_per_ck = max(1, n_steps // cfg.n_checkpoints)
    ck_steps = checkpoint_schedule(n_steps)
    bfactory = make_batched_factory(exp, dt, pt_betas, cfg.seeds,
                                    n_particles=cfg.n_particles,
                                    score_kwargs=c["quad"])
    rows, method_info = run_experiment_batched(
        [method], cfg.seeds, bfactory, n_steps, steps_per_ck, dt,
        metrics_fn, exp.pot, cfg.n_particles, checkpoint_steps=ck_steps)

    summary_metrics = c["main"] + ["nonfinite_frac", "basin_map_outside_mass"]
    tmp = JCP_ROOT / "cache" / f"_tmp_summary_{experiment}_{method}.csv"
    write_summary_csv(rows, [method], cfg.seeds, summary_metrics,
                      method_info, floors, str(tmp), overwrite=True)
    with tmp.open(newline="", encoding="utf-8") as h:
        srow = next(r for r in csv.DictReader(h) if r["method"] == method)
    tmp.unlink()
    pos = exp.metric_space(method_info[method]["final_positions_all"])

    # --- stationarity (worst-basin/energy ESS), replicating the notebook's
    # charged-settling non-PT group exactly: reference-init chains, burn one
    # full horizon, then measure. Deterministic in TRACE_SEEDS, so an existing
    # method reproduces its committed ESS -- the second half of the validation.
    trace_seeds = tuple(cfg.seeds[:min(len(cfg.seeds), 4)])
    trace_chains = 8
    n_draws = min(1000, n_steps)
    steps_per_draw = max(1, n_steps // n_draws)
    settling_burn = int(round(1.0 * n_steps))
    trace_factory = make_sampler_factory(
        exp, dt, pt_betas, n_particles=trace_chains,
        score_kwargs=c["quad"], reference_init=True)
    reference_cv_means = aux["reference_cv_means"]
    cv_names = ["x"] if len(reference_cv_means) == 1 else [
        f"cv_{j}" for j in range(len(reference_cv_means))]
    reference_method = aux["sample_reference_method"]
    collected = collect_stationary_trajectories(
        sampler_factory=trace_factory, methods=[method], seeds=trace_seeds,
        n_draws=n_draws, steps_per_draw=steps_per_draw, dt=dt,
        labels_fn=exp.labels_fn, energy_fn=exp.pot.V, cv_fn=exp.metric_space,
        counter_source=exp.pot, warmup_steps=C.N_WARMUP_STEPS,
        burn_in_steps=settling_burn, equilibrium_initialized=False,
        initialization_method=(
            reference_method + ":reference_draw_then_charged_kernel_settling"),
        basin_ids=list(range(exp.p_star.numel())), cv_names=cv_names,
        basin_target_probabilities=exp.p_star.cpu().tolist(),
        reference_energy_mean=aux["reference_energy_mean"],
        reference_cv_means=reference_cv_means)
    station = collected["methods"][method]["summary"]
    return srow, rows, pos, method_info, station


def _cmp(experiment, method):
    srow, _ts, _pos, _mi, station = run_arm(experiment, method)
    committed = next(r for r in csv.DictReader(
        open(JCP_ROOT / "results" / experiment / "summary.csv"))
        if r["method"] == method)
    keys = CFG[experiment]["main"] + ["basin_KL_target", "FES_RMSE_kBT"]
    print(f"=== validate {experiment}/{method}: reproduced vs committed ===")
    ok = True
    for k in keys:
        col = f"{k}_mean"
        a, b = srow.get(col), committed.get(col)
        if a is None or b is None:
            continue
        rel = abs(float(a) - float(b)) / (abs(float(b)) + 1e-30)
        flag = "OK" if rel < 1e-6 else "**MISMATCH**"
        if rel >= 1e-6:
            ok = False
        print(f"  {col:24s} repro={float(a):.8g}  committed={float(b):.8g}  rel={rel:.2e}  {flag}")
    # stationarity: worst_basin_ess must reproduce the committed value
    comm_st = next((r for r in csv.DictReader(
        open(JCP_ROOT / "results" / experiment / "stationarity"
             / "all_methods_summary.csv")) if r["method"] == method), None)
    if comm_st is not None:
        a = float(station["worst_basin_ess"])
        b = float(comm_st["worst_basin_ess"])
        rel = abs(a - b) / (abs(b) + 1e-30)
        flag = "OK" if rel < 1e-6 else "**MISMATCH**"
        if rel >= 1e-6:
            ok = False
        print(f"  {'worst_basin_ess':24s} repro={a:.8g}  committed={b:.8g}  rel={rel:.2e}  {flag}")
    print("RESULT:", "REPRODUCED — data path is correct" if ok else "MISMATCH — do not append")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=tuple(CFG))
    ap.add_argument("--method", required=True)
    ap.add_argument("--validate", action="store_true",
                    help="reproduce a committed method's row and diff (no writes)")
    ap.add_argument("--append", action="store_true",
                    help="append this method's rows to the committed results tree")
    args = ap.parse_args(argv)
    if args.validate:
        return 0 if _cmp(args.experiment, args.method) else 1
    if args.append:
        _append(args.experiment, args.method)
        return 0
    ap.error("pass --validate or --append")


def _append(experiment, method):
    srow, ts_rows, pos, _mi, station = run_arm(experiment, method)
    resdir = JCP_ROOT / "results" / experiment
    # refuse to double-append if the method is already present
    with (resdir / "summary.csv").open(newline="", encoding="utf-8") as h:
        if any(r["method"] == method for r in csv.DictReader(h)):
            raise SystemExit(f"{method} already present in {resdir}/summary.csv; "
                             "git-restore first to re-append")
    # summary.csv: append the method row (column-union via DictWriter restval)
    _append_rows(resdir / "summary.csv", [srow])
    # metrics_timeseries.csv: append this method's checkpoint rows
    _append_rows(resdir / "metrics_timeseries.csv",
                 [r for r in ts_rows if r.get("method") == method])
    # positions.csv: append this method's terminal block (schema: method,particle,n_total,cv...)
    _append_positions(resdir / "positions.csv", method, pos)
    # stationarity: per-method summary + append flat rows to all_methods_summary
    stdir = resdir / "stationarity"
    write_stationarity_csv(str(stdir / f"{method}_summary.csv"), station)
    _append_rows(stdir / "all_methods_summary.csv", flat_summary_rows(station))
    print(f"appended {method} to {resdir} "
          f"(summary, timeseries, positions, stationarity; "
          f"worst_basin_ess={float(station['worst_basin_ess']):.1f})")


def _append_rows(path, new_rows):
    if not new_rows:
        return
    with path.open(newline="", encoding="utf-8") as h:
        reader = csv.DictReader(h)
        existing = list(reader)
        fields = list(reader.fieldnames)
    for r in new_rows:                            # union any new columns
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(existing)
        w.writerows([{k: r.get(k, "") for k in fields} for r in new_rows])


def _append_positions(path, method, pos, n_max=20_000, seed=0):
    arr = pos.detach().cpu().numpy()
    n_total = arr.shape[0]
    # match write_positions_csv exactly: n_total records the true count, but a
    # block larger than n_max is deterministically subsampled so every method's
    # block has the same row budget.
    if n_total > n_max:
        idx = np.random.default_rng(seed).choice(n_total, size=n_max, replace=False)
        arr = arr[idx]
    with path.open(newline="", encoding="utf-8") as h:
        fields = list(csv.DictReader(h).fieldnames)
    cvcols = [c for c in fields if c.startswith("cv")]
    rows = []
    for i in range(arr.shape[0]):
        row = {"method": method, "particle": i, "n_total": n_total}
        for j, cvc in enumerate(cvcols):
            row[cvc] = arr[i, j] if j < arr.shape[1] else ""
        rows.append(row)
    with path.open("a", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields, restval="")
        w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
