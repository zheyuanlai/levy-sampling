"""Optional per-variant stationarity diagnostics.

This is not a global stage. It is a long-chain diagnostic for ONE variant,
reporting integrated autocorrelation time, effective sample size, split R-hat,
and basin-label autocorrelation. A run notebook enables it per method cell with
``RUN_STATIONARITY = True``.

It needs re-running for a variant only when that variant's parameters, tame
flag, timestep, target, or the stationarity protocol itself changed. Every other
variant is untouched.

Chains are individual particle trajectories, all driven by the same kernel, so
the autocorrelation estimates describe the sampler rather than an ensemble
average. Split R-hat is computed across seeds on the per-seed ensemble mean,
which is the statistic that separates when different seeds sit in different
basins.
"""
from __future__ import annotations

import numpy as np
import torch

from .factory import build_sampler
from .metrics import autocorrelation_time, effective_sample_size, split_rhat

#: Defaults for the optional long-chain diagnostic: fewer particles than
#: production and a longer horizon, because this measures autocorrelation rather
#: than finite-time distributional error.
STATIONARITY_DEFAULTS = {
    "particles": 64,
    "tracked_particles": 32,
    "time_multiplier": 2.0,
    "burn_in_fraction": 0.5,
    "record_every": 1,
    "start_from_reference": True,
}


def measure_stationarity(experiment, variant, *, dt: float, calibration: dict,
                         suite, protocol: dict | None = None) -> dict:
    """Run the long-chain diagnostic for one variant."""
    protocol = {**STATIONARITY_DEFAULTS, **(protocol or {})}
    seeds = experiment.seeds
    n_per_seed = int(protocol["particles"])
    n_steps = max(2, int(round(experiment.final_time
                               * float(protocol["time_multiplier"]) / dt)))
    record_every = max(1, int(protocol["record_every"]))
    burn_in = int(float(protocol["burn_in_fraction"]) * n_steps)

    streams = experiment.streams_for(variant)
    with experiment.target.no_count():
        if protocol["start_from_reference"]:
            generator = torch.Generator(device=experiment.device)
            generator.manual_seed(20260805)
            x0 = experiment.ensure_reference().sample(
                len(seeds) * n_per_seed, generator)
        else:
            x0 = experiment.init_fn(streams, n_per_seed)
        sampler = build_sampler(experiment, variant, dt=dt, streams=streams,
                                n_per_seed=n_per_seed, x0=x0,
                                calibration=calibration)

        tracked = min(int(protocol["tracked_particles"]), n_per_seed)
        index = torch.linspace(0, n_per_seed - 1, tracked,
                               device=experiment.device).round().long()
        observable_records: list[np.ndarray] = []
        label_records: list[np.ndarray] = []
        mean_records: list[np.ndarray] = []
        for step in range(1, n_steps + 1):
            sampler.step()
            if step % record_every:
                continue
            positions = sampler.positions()
            scalar = suite.stationarity_observable(positions)
            labels = suite.labels(positions)
            per_seed = scalar.reshape(len(seeds), n_per_seed)
            observable_records.append(
                per_seed[:, index].detach().cpu().numpy())
            label_records.append(
                labels.reshape(len(seeds), n_per_seed)[:, index]
                .detach().cpu().numpy())
            mean_records.append(per_seed.mean(dim=1).detach().cpu().numpy())

    observable = np.stack(observable_records, axis=-1)      # (S, tracked, T)
    labels = np.stack(label_records, axis=-1)               # (S, tracked, T)
    means = np.stack(mean_records, axis=-1)                 # (S, T)
    keep = max(1, burn_in // record_every)
    summary = _summarize(observable[..., keep:], labels[..., keep:],
                         means[..., keep:], interval=dt * record_every,
                         protocol=protocol, n_seeds=len(seeds))
    arrays = {
        "observable": observable,
        "labels": labels,
        "seed_ensemble_mean": means,
        "burn_in_records": np.int64(keep),
        "record_every": np.int64(record_every),
        "dt": np.float64(dt),
        "n_steps": np.int64(n_steps),
    }
    return {"arrays": arrays, "summary": summary}


def _summarize(observable: np.ndarray, labels: np.ndarray, means: np.ndarray,
               *, interval: float, protocol: dict, n_seeds: int) -> dict:
    n_records = observable.shape[-1]
    flat = observable.reshape(-1, n_records)
    iats = np.array([autocorrelation_time(series) for series in flat])
    esss = np.array([effective_sample_size(series) for series in flat])
    label_iats = np.array([autocorrelation_time(series.astype(float))
                           for series in labels.reshape(-1, n_records)])
    rhat = (float(split_rhat(means))
            if means.shape[0] >= 2 and means.shape[1] >= 4 else float("nan"))
    return {
        "protocol": protocol,
        "n_seeds": int(n_seeds),
        "n_records_post_burn_in": int(n_records),
        "record_interval_time": float(interval),
        "iat_mean_steps": float(np.nanmean(iats)),
        "iat_median_steps": float(np.nanmedian(iats)),
        "iat_max_steps": float(np.nanmax(iats)),
        "iat_mean_time_units": float(np.nanmean(iats) * interval),
        "ess_per_chain_mean": float(np.nanmean(esss)),
        "ess_total": float(np.nansum(esss)),
        "basin_label_iat_mean_steps": float(np.nanmean(label_iats)),
        "basin_label_iat_max_steps": float(np.nanmax(label_iats)),
        "split_rhat_seed_ensemble_mean": rhat,
        "chain_definition": "individual particle trajectories",
        "rhat_statistic": "per-seed ensemble mean of the summary observable",
    }
