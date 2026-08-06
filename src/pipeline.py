"""The run side of the run/plot split.

A run notebook does only this::

    from src.pipeline import load_experiment, run_variants_and_save

    experiment = load_experiment("E3")
    experiment.ensure_reference()

    run_variants_and_save(experiment=experiment, method="FLA", variants=[
        {"alpha": 1.6}, {"alpha": 1.7}, {"alpha": 1.8},
    ])

Every official metric is computed here, at run time, and written to
``metrics_timeseries.csv``. Plot notebooks read those numbers; they never
recompute them, and they never call a sampler, a reference builder, or a
calibration routine.

Each variant is saved the moment it finishes, into its own atomically-renamed
run directory. A variant that fails leaves the earlier ones untouched.
"""
from __future__ import annotations

from datetime import datetime, timezone
import traceback

import numpy as np
import torch

from .calibration import CalibrationError, calibrate
from .config import (ExperimentContext, Variant, default_variants,
                     expand_variants, load_experiment, snapshot_checkpoints)
from .device import device_provenance, empty_cache
from .factory import build_sampler, sampler_requirements
from .results import (RunPaths, RunWriter, git_provenance, slugify,
                      stable_hash, utc_now)

__all__ = ["load_experiment", "run_variants_and_save", "run_variant"]


def _run_id(variant: Variant, dt: float) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{variant.slug}-dt{dt:g}-{stamp}"


def _outcome_id(variant: Variant, status: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{variant.slug}-{slugify(status)}-{stamp}"


def _seed_blocks(x: torch.Tensor, n_seeds: int, n_per_seed: int):
    for index in range(n_seeds):
        yield index, x[index * n_per_seed:(index + 1) * n_per_seed]


def run_variants_and_save(*, experiment: ExperimentContext, method: str,
                          variants=None, run_production: bool = True,
                          run_stationarity: bool = False,
                          refresh_calibration: bool = False,
                          on_error: str = "record") -> list[dict]:
    """Run every variant of one method and save each one as it finishes.

    ``variants`` is a list of parameter dicts. An entry that does not pin
    ``tame`` is expanded into canonical and tamed runs, because every
    taming-capable method runs both by default. Passing ``None`` uses the
    experiment's default grid for this method from the registry.

    ``on_error="record"`` keeps going after a failed variant and returns its
    error record; ``on_error="raise"`` stops at the first failure.
    """
    if variants is None:
        expanded = default_variants(experiment.registry,
                                    experiment.method_configs,
                                    experiment.experiment_id, method)
    else:
        expanded = expand_variants(experiment.registry,
                                   experiment.method_configs, method, variants)

    reports = []
    for variant in expanded:
        try:
            report = run_variant(
                experiment, variant, run_production=run_production,
                run_stationarity=run_stationarity,
                refresh_calibration=refresh_calibration)
        except CalibrationError as error:
            if on_error == "raise":
                raise
            # A calibration failure is a result about the method. Persist it
            # atomically before continuing to the remaining variants.
            report = _save_uncalibratable(experiment, variant, error)
            print(f"[{variant.label}] NOT CALIBRATABLE: {error.diagnosis}",
                  flush=True)
        except Exception as error:                    # noqa: BLE001
            if on_error == "raise":
                raise
            report = {
                "variant_label": variant.label,
                "method": method,
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(limit=8),
            }
            print(f"[{variant.label}] FAILED: {type(error).__name__}: {error}",
                  flush=True)
        else:
            print(f"[{variant.label}] saved to {report['run_directory']}",
                  flush=True)
        reports.append(report)
        empty_cache()
    return reports


def _save_uncalibratable(experiment: ExperimentContext, variant: Variant,
                         error: CalibrationError) -> dict:
    """Atomically persist a negative calibration outcome as first-class evidence."""
    attempt = {
        "status": "uncalibratable",
        "kind": error.kind,
        "diagnosis": error.diagnosis,
        "table": error.table,
        "next_candidate": error.next_candidate,
    }
    attempt["calibration_hash"] = stable_hash(attempt)
    run_id = _outcome_id(variant, "uncalibratable")
    streams = experiment.streams_for(variant)
    resolved = experiment.resolved_config(variant=variant,
                                          calibration=attempt)
    reference = experiment._reference
    reference_hash = (None if reference is None
                      else stable_hash(reference.describe()))
    fee = experiment._fee_calibration

    with RunWriter(experiment.paths, method=variant.method,
                   run_id=run_id) as writer:
        writer.write_yaml("resolved_config.yaml", resolved)
        writer.write_json("calibration.json", attempt)
        writer.write_json("diagnostics.json", {
            "outcome": attempt,
            "requirements": sampler_requirements(experiment, variant),
            "note": ("No production trajectory was started because no "
                     "admissible calibration was certified."),
        })
        writer.set_manifest({
            "experiment_id": experiment.experiment_id,
            "experiment_key": experiment.key,
            "method": variant.method,
            **variant.describe(),
            "tame_cap": experiment.tame_cap_for(variant),
            "dt": None,
            "particles": experiment.particles,
            "seeds": list(experiment.seeds),
            "target_hash": experiment.target_hash,
            "reference_hash": reference_hash,
            "calibration_hash": attempt["calibration_hash"],
            "fee_calibration_hash": None if fee is None else fee.hash,
            "fee_cost_unit": None if fee is None else fee.cost_unit,
            "rng": streams.provenance(),
            "device_provenance": device_provenance(experiment.device,
                                                   experiment.dtype),
            "git": git_provenance(),
            "status": "uncalibratable",
            "has_stationarity": False,
            "created_at_utc": utc_now(),
            "diagnosis": error.diagnosis,
            "calibration_kind": error.kind,
        })
        final_dir = writer.final_dir
    return {
        "variant_label": variant.label,
        "method": variant.method,
        "status": "uncalibratable",
        "diagnosis": error.diagnosis,
        "calibration_kind": error.kind,
        "calibration_table": error.table,
        "next_candidate": error.next_candidate,
        "calibration_hash": attempt["calibration_hash"],
        "run_id": run_id,
        "run_directory": str(final_dir),
    }


def run_variant(experiment: ExperimentContext, variant: Variant, *,
                run_production: bool = True, run_stationarity: bool = False,
                refresh_calibration: bool = False) -> dict:
    """Calibrate, run, and atomically save one variant."""
    from .measurements import build_measurement_suite

    requirements = sampler_requirements(experiment, variant)
    calibration = calibrate(experiment, variant, refresh=refresh_calibration)
    dt = float(calibration["dt"])

    reference = experiment.ensure_reference()
    fee_calibration = experiment.ensure_fee_calibration()
    suite = build_measurement_suite(experiment, reference)

    seeds = experiment.seeds
    n_per_seed = experiment.particles
    n_steps = experiment.steps_for(dt)
    steps = experiment.schedule_for(dt)
    snapshot_config = experiment.config.get("plot_snapshots", {}) or {}
    snapshot_steps = set(snapshot_checkpoints(
        steps, dt, snapshot_config.get("time_values", [])))
    max_snapshot_points = int(snapshot_config.get("max_points_per_seed", 2000))

    streams = experiment.streams_for(variant)
    experiment.target.counters.reset()
    with experiment.target.no_count():
        x0 = experiment.init_fn(streams, n_per_seed)
    sampler = build_sampler(experiment, variant, dt=dt, streams=streams,
                            n_per_seed=n_per_seed, x0=x0,
                            calibration=calibration)

    metric_rows: list[dict] = []
    cost_rows: list[dict] = []
    snapshots: dict[str, dict] = {}
    counter_baseline = experiment.target.raw_counters()
    diagnostics_last: dict = {}

    if not run_production:
        return {"variant_label": variant.label, "status": "skipped",
                "reason": "run_production is False",
                "calibration_hash": calibration["calibration_hash"]}

    current_step = 0
    for checkpoint in steps:
        for _ in range(checkpoint - current_step):
            sampler.step()
        current_step = checkpoint
        positions = sampler.positions()
        diagnostics_last = sampler.pop_diagnostics()
        counters = experiment.target.derived_counters(counter_baseline)
        cost_row = {
            "step": checkpoint,
            "t": checkpoint * dt,
            **fee_calibration.cost_row(counters),
        }
        total_particles = len(seeds) * n_per_seed
        cost_row["n_force_per_particle"] = counters["n_force"] / total_particles
        cost_row["n_extra_potential_equivalent_per_particle"] = (
            cost_row["n_extra_potential_equivalent"] / total_particles)
        cost_row["n_fee_per_particle"] = cost_row["n_fee"] / total_particles
        cost_rows.append(cost_row)

        with experiment.target.no_count():
            for seed_index, block in _seed_blocks(positions, len(seeds),
                                                  n_per_seed):
                row = {
                    "method": variant.method,
                    "variant_label": variant.label,
                    "variant_hash": variant.hash,
                    "metric_definition_hash": suite.definition_hash,
                    "tame": variant.tame,
                    "seed": seeds[seed_index],
                    "step": checkpoint,
                    "t": checkpoint * dt,
                    "n_fee": cost_row["n_fee"],
                    "n_fee_per_particle": cost_row["n_fee_per_particle"],
                    "n_force_per_particle": cost_row["n_force_per_particle"],
                    "n_extra_potential_equivalent_per_particle":
                        cost_row["n_extra_potential_equivalent_per_particle"],
                }
                row.update(suite.metrics(block))
                # Interval diagnostics describe the whole batched ensemble, so
                # they are attached to every seed row rather than pretending to
                # be per-seed measurements.
                row.update({key: value
                            for key, value in diagnostics_last.items()
                            if not key.endswith("_count_cumulative")})
                metric_rows.append(row)

            if checkpoint in snapshot_steps:
                snapshots[f"checkpoint_{checkpoint:09d}"] = _snapshot(
                    experiment, suite, positions, seeds, n_per_seed,
                    checkpoint, dt, cost_row, max_snapshot_points)

    terminal = sampler.positions().detach()
    stationarity = None
    if run_stationarity:
        from .stationarity import measure_stationarity

        stationarity = measure_stationarity(
            experiment, variant, dt=dt, calibration=calibration, suite=suite)

    return _save_run(experiment, variant, dt=dt, calibration=calibration,
                     metric_rows=metric_rows, cost_rows=cost_rows,
                     snapshots=snapshots, terminal=terminal,
                     diagnostics=diagnostics_last, suite=suite,
                     fee_calibration=fee_calibration, streams=streams,
                     requirements=requirements, n_steps=n_steps,
                     stationarity=stationarity)


def _snapshot(experiment, suite, positions, seeds, n_per_seed, checkpoint, dt,
              cost_row, max_points: int) -> dict:
    """Deterministically downsampled per-seed sample snapshot.

    The same indices are used for every method and every checkpoint, so the
    scatter panels compare equal point counts drawn the same way.
    """
    take = min(max_points, n_per_seed)
    index = torch.linspace(0, n_per_seed - 1, take, device=positions.device
                           ).round().long()
    blocks, seed_labels = [], []
    for seed_index, block in _seed_blocks(positions, len(seeds), n_per_seed):
        blocks.append(block[index])
        seed_labels.append(np.full(take, seeds[seed_index], dtype=np.int64))
    selected = torch.cat(blocks, dim=0)
    payload = {
        "x": selected.detach().cpu().numpy(),
        "seed": np.concatenate(seed_labels),
        "sample_index": np.tile(index.detach().cpu().numpy(), len(seeds)),
        "checkpoint_step": np.int64(checkpoint),
        "t": np.float64(checkpoint * dt),
        "n_fee": np.float64(cost_row["n_fee"]),
        "n_fee_per_particle": np.float64(cost_row["n_fee_per_particle"]),
        "n_force": np.int64(cost_row["n_force"]),
        "n_force_per_particle": np.float64(
            cost_row["n_force_per_particle"]),
        "n_extra_potential": np.int64(cost_row["n_extra_potential"]),
        "n_extra_potential_equivalent": np.float64(
            cost_row["n_extra_potential_equivalent"]),
        "n_extra_potential_equivalent_per_particle": np.float64(
            cost_row["n_extra_potential_equivalent_per_particle"]),
        "rho": np.float64(cost_row["rho"]),
        "fee_calibration_hash": cost_row["fee_calibration_hash"],
        "fee_cost_unit": cost_row["fee_cost_unit"],
        "downsample_rule": "evenly spaced indices, identical across methods",
    }
    for name, array in suite.snapshot_arrays(selected).items():
        payload[name] = array
    return payload


def _save_run(experiment, variant, *, dt, calibration, metric_rows, cost_rows,
              snapshots, terminal, diagnostics, suite, fee_calibration,
              streams, requirements, n_steps, stationarity) -> dict:
    run_id = _run_id(variant, dt)
    seeds = experiment.seeds
    resolved = experiment.resolved_config(variant=variant, dt=dt,
                                          calibration=calibration)
    resolved["resolved"]["metric_definition_hash"] = suite.definition_hash
    resolved["resolved"]["fee_calibration"] = fee_calibration.to_dict()
    resolved["resolved"]["checkpoint_costs"] = cost_rows
    with RunWriter(experiment.paths, method=variant.method,
                   run_id=run_id) as writer:
        writer.write_yaml("resolved_config.yaml", resolved)
        writer.write_json("calibration.json", calibration)
        writer.write_csv("metrics_timeseries.csv", metric_rows)
        writer.write_csv("cost_timeseries.csv", cost_rows)
        writer.write_npz(
            "terminal_samples.npz",
            x=terminal,
            seed=np.repeat(np.asarray(seeds, dtype=np.int64),
                           experiment.particles),
            **{name: array
               for name, array in suite.snapshot_arrays(terminal).items()})
        for name, payload in snapshots.items():
            writer.write_npz(f"sample_snapshots/{name}.npz", **payload)
        writer.write_json("diagnostics.json", {
            "final_interval_diagnostics": diagnostics,
            "cumulative": {key: value for key, value in diagnostics.items()
                           if key.endswith("_cumulative")},
            "sampler": _sampler_description(experiment, variant, dt,
                                            calibration),
            "measurement_suite": suite.describe(),
            "requirements": requirements,
        })
        if stationarity is not None:
            writer.write_npz("stationarity.npz", **stationarity["arrays"])
            writer.write_json("stationarity.json", stationarity["summary"])

        manifest = {
            "experiment_id": experiment.experiment_id,
            "experiment_key": experiment.key,
            "method": variant.method,
            **variant.describe(),
            "tame_cap": experiment.tame_cap_for(variant),
            "dt": dt,
            "n_steps": n_steps,
            "final_time": experiment.final_time,
            "particles": experiment.particles,
            "seeds": list(seeds),
            "checkpoint_steps": experiment.schedule_for(dt),
            "target_hash": experiment.target_hash,
            "reference_hash": experiment.reference_hash,
            "calibration_hash": calibration["calibration_hash"],
            "metric_definition_hash": suite.definition_hash,
            "fee_calibration_hash": fee_calibration.hash,
            "fee_cost_unit": fee_calibration.cost_unit,
            "fee_calibration": fee_calibration.to_dict(),
            "rng": streams.provenance(),
            "device_provenance": device_provenance(experiment.device,
                                                   experiment.dtype),
            "git": git_provenance(),
            "status": "complete",
            "has_stationarity": stationarity is not None,
            "created_at_utc": utc_now(),
        }
        writer.set_manifest(manifest)
        final_dir = writer.final_dir
    return {
        "variant_label": variant.label,
        "status": "complete",
        "run_id": run_id,
        "run_directory": str(final_dir),
        "dt": dt,
        "calibration_hash": calibration["calibration_hash"],
        "fee_calibration_hash": fee_calibration.hash,
        "n_metric_rows": len(metric_rows),
        "n_snapshots": len(snapshots),
    }


def _sampler_description(experiment, variant, dt, calibration) -> dict:
    streams = experiment.streams_for(variant, seeds=(experiment.seeds[0],))
    with experiment.target.no_count():
        sampler = build_sampler(experiment, variant, dt=dt, streams=streams,
                                n_per_seed=1, calibration=calibration)
        return sampler.describe()
