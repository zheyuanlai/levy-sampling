"""Append the E1 Raw-CP stationary mixing diagnostic using local CPU.

Raw-CP does not preserve the Gibbs target.  Its worst-basin ESS is therefore
reported only as an autocorrelation/mixing diagnostic for the biased Raw-CP
kernel, alongside the already-saved target-bias fields.  It must not be
interpreted as an effective sample size from the target distribution.

The trace protocol exactly matches the committed E1 non-PT stationarity run:
four seeds, eight chains per seed, a charged T-length settling period, then
1,000 draws at stride 20.  This script only appends stationarity artifacts; the
existing production relaxation results for Raw-CP are left untouched.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JCP_ROOT = HERE.parent
sys.path.insert(0, str(JCP_ROOT))

import torch

torch.set_default_dtype(torch.float64)

import src.config as C  # noqa: E402
from src.experiments import build_e1, make_metrics, make_sampler_factory  # noqa: E402
from src.samplers import geometric_ladder  # noqa: E402
from src.stationarity import (  # noqa: E402
    _json_safe,
    collect_stationary_trajectories,
    flat_summary_rows,
    write_stationarity_csv,
    write_stationarity_npz,
)


METHOD = "CP"
EXPERIMENT = "double_well"
TRACE_SEED_COUNT = 4
TRACE_CHAINS_PER_SEED = 8
TRACE_DRAWS = 1000


def _write_rows(path: Path, rows: list[dict]) -> None:
    """Rewrite a CSV with the union of existing and appended columns."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        existing = list(reader)
        fields = list(reader.fieldnames or ())
    if any(row.get("method") == METHOD for row in existing):
        raise FileExistsError(f"{METHOD} already exists in {path}")
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])
    os.replace(temporary, path)


def main() -> int:
    result_dir = JCP_ROOT / "results" / EXPERIMENT
    stationarity_dir = result_dir / "stationarity"
    summary_path = stationarity_dir / f"{METHOD}_summary.csv"
    traces_path = stationarity_dir / f"{METHOD}_traces.npz"
    all_methods_path = stationarity_dir / "all_methods_summary.csv"
    manifest_path = result_dir / "manifest.json"
    for path in (summary_path, traces_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    device = "cpu"
    exp = build_e1(device=device)
    cfg = exp.cfg
    _, _, aux = make_metrics(exp, cfg.n_particles, device=device)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    ladder_record = manifest["pt_ladder"]
    pt_betas = geometric_ladder(
        cfg.beta, exp.pt_beta_min, int(ladder_record["K"]), exp.p_star.device
    )

    n_steps = cfg.n_steps
    trace_draws = min(TRACE_DRAWS, n_steps)
    steps_per_draw = max(1, n_steps // trace_draws)
    burn_in_steps = n_steps
    trace_seeds = tuple(cfg.seeds[:TRACE_SEED_COUNT])
    trace_factory = make_sampler_factory(
        exp,
        cfg.dt,
        pt_betas,
        n_particles=TRACE_CHAINS_PER_SEED,
        score_kwargs=manifest["quadrature"]["chosen"],
        reference_init=True,
    )
    reference_method = aux["sample_reference_method"]
    initialization_method = (
        reference_method
        + ":reference_draw_then_charged_biased_raw_cp_kernel_settling"
    )
    collected = collect_stationary_trajectories(
        sampler_factory=trace_factory,
        methods=[METHOD],
        seeds=trace_seeds,
        n_draws=trace_draws,
        steps_per_draw=steps_per_draw,
        dt=cfg.dt,
        labels_fn=exp.labels_fn,
        energy_fn=exp.pot.V,
        cv_fn=exp.metric_space,
        counter_source=exp.pot,
        warmup_steps=C.N_WARMUP_STEPS,
        burn_in_steps=burn_in_steps,
        equilibrium_initialized=False,
        initialization_method=initialization_method,
        basin_ids=list(range(exp.p_star.numel())),
        cv_names=["x"],
        basin_target_probabilities=exp.p_star.cpu().tolist(),
        reference_energy_mean=aux["reference_energy_mean"],
        reference_cv_means=aux["reference_cv_means"],
    )
    result = collected["methods"][METHOD]
    summary = result["summary"]
    raw = result["raw"]

    write_stationarity_csv(summary_path, summary)
    write_stationarity_npz(
        traces_path,
        trace_times=raw["trace_times"],
        positions_t=raw["positions_t"],
        labels_t=raw["labels_t"],
        energy_t=raw["energy_t"],
        cv_t=raw["cv_t"],
        seed_index=raw["seed_index"],
        chain_index_within_seed=raw["chain_index_within_seed"],
        summary=summary,
        metadata={
            "experiment": EXPERIMENT,
            "method": METHOD,
            "device": device,
            "interpretation": "biased-kernel mixing diagnostic; not target ESS",
        },
    )
    _write_rows(all_methods_path, flat_summary_rows(summary))

    stationarity = manifest.setdefault("stationarity", {})
    stationarity.setdefault("methods", {})[METHOD] = _json_safe(summary)
    stationarity.setdefault("collection", {})[
        "charged_settling_non_targeting_raw_cp_cpu"
    ] = _json_safe(collected["collection"])
    stationarity.setdefault("protocol", {})[
        "charged_settling_non_targeting_methods"
    ] = [METHOD]
    excluded = stationarity.get("excluded_non_targeting_methods", [])
    stationarity["excluded_non_targeting_methods"] = [
        method for method in excluded if method != METHOD
    ]
    stationarity["non_targeting_mixing_diagnostics"] = {
        METHOD: {
            "device": device,
            "interpretation": "biased-kernel mixing diagnostic; not target ESS",
            "target_bias_reported_in_observable_rows": True,
        }
    }
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    with manifest_tmp.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(manifest_tmp, manifest_path)

    print(
        f"appended {EXPERIMENT}/{METHOD} CPU stationarity: "
        f"worst_basin_ess={summary['worst_basin_ess']:.8g}, "
        f"wallclock_s={summary['wallclock_s']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
