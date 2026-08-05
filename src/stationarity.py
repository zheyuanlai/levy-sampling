"""Post-burn-in stationary-trace analysis for the JCP experiments.

The routines in this module deliberately operate on *uniformly spaced*,
per-chain scalar traces rather than the sparse ensemble checkpoints used for
finite-time relaxation plots.  Input arrays use time-major layout throughout:

``labels_t``: ``(n_draws, n_chains)``
``energy_t``: ``(n_draws, n_chains)``
``cv_t``:     ``(n_draws, n_chains, n_cv)``

Effective sample sizes are sums of the per-chain ESS values, appropriate when
the chains use independent random streams.  The reported aggregate IAT is the
equivalent value ``n_draws*n_chains/ESS``.  Constant chains contribute zero
ESS and are reported explicitly instead of being interpreted as converged.
"""
from __future__ import annotations

import csv
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from .metrics import ess_from_series, iat_1d, split_rhat


SCHEMA_VERSION = 1


def _json_safe(value: Any) -> Any:
    """Convert arrays/scalars and nonfinite sentinels to strict JSON values."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _cuda_synchronize_if_available() -> None:
    """Synchronize CUDA when torch has an active CUDA backend; otherwise no-op."""
    try:
        from .device import synchronize
    except ImportError:          # torch absent: this module stays importable
        return
    synchronize()


def _as_numpy(value: Any) -> np.ndarray:
    """Convert NumPy-compatible or torch-like arrays without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def validate_trace_times(
    trace_times: Any,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> float:
    """Validate a finite, strictly increasing, uniformly spaced time grid.

    Returns the common sampling interval.  At least two time points are
    required because neither uniform spacing nor autocorrelation can be
    assessed from a single draw.
    """
    times = _as_numpy(trace_times).astype(float, copy=False)
    if times.ndim != 1:
        raise ValueError("trace_times must be one-dimensional")
    if times.size < 2:
        raise ValueError("at least two trace times are required")
    if not np.all(np.isfinite(times)):
        raise ValueError("trace_times must be finite")
    delta = np.diff(times)
    if np.any(delta <= 0):
        raise ValueError("trace_times must be strictly increasing")
    interval = float(delta[0])
    if not np.allclose(delta, interval, rtol=rtol, atol=atol):
        raise ValueError("trace_times must be uniformly spaced")
    return interval


def _validate_cost(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _evaluation_counters(source: Any) -> tuple[int, int, int]:
    """Read the repository's potential/gradient/score-quadrature counters."""
    missing = [name for name in ("n_V", "n_grad", "n_Vdelta")
               if not hasattr(source, name)]
    if missing:
        raise TypeError(
            "counter source must expose n_V, n_grad, and n_Vdelta; missing "
            + ", ".join(missing)
        )
    counters = tuple(int(getattr(source, name))
                     for name in ("n_V", "n_grad", "n_Vdelta"))
    if any(value < 0 for value in counters):
        raise ValueError("evaluation counters must be non-negative")
    return counters


@contextmanager
def _without_counter_cost(source: Any) -> Iterator[None]:
    """Exclude observation functions from the sampler's evaluation ledger."""
    if hasattr(source, "no_count"):
        with source.no_count():
            yield
        return
    before = _evaluation_counters(source)
    try:
        yield
    finally:
        source.n_V, source.n_grad, source.n_Vdelta = before


def _resolve_counter_source(sampler: Any, explicit_source: Any | None) -> Any:
    if explicit_source is not None:
        return explicit_source
    for attribute in ("pot", "potential"):
        if hasattr(sampler, attribute):
            return getattr(sampler, attribute)
    raise TypeError(
        "counter_source is required when the sampler exposes neither .pot "
        "nor .potential"
    )


def _observe_positions(
    sampler: Any,
    *,
    labels_fn: Callable[[Any], Any],
    energy_fn: Callable[[Any], Any],
    cv_fn: Callable[[Any], Any],
    counter_source: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate one post-step frame outside timing and counter accounting."""
    positions = sampler.positions()
    with _without_counter_cost(counter_source):
        labels = _as_numpy(labels_fn(positions))
        energy = _as_numpy(energy_fn(positions))
        cvs = _as_numpy(cv_fn(positions))
    positions_np = _as_numpy(positions)
    labels = np.asarray(labels)
    energy = np.asarray(energy, dtype=float)
    cvs = np.asarray(cvs, dtype=float)
    if positions_np.ndim != 2:
        raise ValueError("sampler.positions() must return shape (n_chains, d)")
    n_chains = positions_np.shape[0]
    if labels.shape != (n_chains,):
        raise ValueError("labels_fn must return shape (n_chains,)")
    if energy.shape != (n_chains,):
        raise ValueError("energy_fn must return shape (n_chains,)")
    if cvs.ndim == 1:
        cvs = cvs[:, None]
    if cvs.ndim != 2 or cvs.shape[0] != n_chains:
        raise ValueError("cv_fn must return shape (n_chains,) or (n_chains, n_cv)")
    # Copy because tensor-to-NumPy conversion can share storage with an
    # in-place sampler state update.
    return (np.array(positions_np, copy=True), np.array(labels, copy=True),
            np.array(energy, copy=True), np.array(cvs, copy=True))


def _timed_steps(
    sampler: Any,
    n_steps: int,
    *,
    synchronize_fn: Callable[[], None],
    timer_fn: Callable[[], float],
) -> float:
    """Run one sampler-only timing segment with backend synchronization."""
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if n_steps == 0:
        return 0.0
    synchronize_fn()
    start = float(timer_fn())
    for _ in range(n_steps):
        sampler.step()
    synchronize_fn()
    elapsed = float(timer_fn()) - start
    if not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError("timer returned a nonfinite or negative duration")
    return elapsed


def _validate_traces(
    trace_times: Any,
    labels_t: Any,
    energy_t: Any,
    cv_t: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    times = _as_numpy(trace_times).astype(float, copy=False)
    interval = validate_trace_times(times)
    labels_raw = _as_numpy(labels_t)
    energy = _as_numpy(energy_t).astype(float, copy=False)
    cvs = _as_numpy(cv_t).astype(float, copy=False)

    if labels_raw.ndim != 2:
        raise ValueError("labels_t must have shape (n_draws, n_chains)")
    if energy.ndim != 2 or energy.shape != labels_raw.shape:
        raise ValueError("energy_t must have the same (n_draws, n_chains) shape")
    if cvs.ndim != 3 or cvs.shape[:2] != labels_raw.shape:
        raise ValueError("cv_t must have shape (n_draws, n_chains, n_cv)")
    if labels_raw.shape[0] != times.size:
        raise ValueError("trace_times length must equal the trace draw dimension")
    if labels_raw.shape[1] < 1:
        raise ValueError("at least one chain is required")
    if cvs.shape[2] < 1:
        raise ValueError("at least one collective variable is required")
    if not np.all(np.isfinite(labels_raw)):
        raise ValueError("labels_t must be finite")
    if not np.all(labels_raw == np.rint(labels_raw)):
        raise ValueError("labels_t must contain integer basin labels")
    if not np.all(np.isfinite(energy)):
        raise ValueError("energy_t must be finite")
    if not np.all(np.isfinite(cvs)):
        raise ValueError("cv_t must be finite")

    return times, labels_raw.astype(np.int64), energy, cvs, interval


def _safe_rate(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _observable_row(
    chains: np.ndarray,
    *,
    kind: str,
    name: str,
    index: int,
    wallclock_s: float,
    gradient_evals: float,
    potential_evals: float,
    score_quadrature_evals: float,
    trace_interval: float,
    target: float | None = None,
) -> dict[str, Any]:
    """Summarize one scalar observable with chain-major input."""
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2:
        raise ValueError("observable chains must have shape (n_chains, n_draws)")

    chain_iats = np.asarray([iat_1d(chain) for chain in chains], dtype=float)
    ess = float(ess_from_series(chains))
    total_draws = int(chains.size)
    aggregate_iat = (
        float(total_draws / ess)
        if math.isfinite(ess) and ess > 0
        else float("inf") if ess == 0 else float("nan")
    )
    finite_iats = chain_iats[np.isfinite(chain_iats)]
    constant = np.asarray(
        [np.allclose(chain, chain[0]) for chain in chains], dtype=bool
    )
    sample_sd = float(np.std(chains, ddof=1)) if total_draws > 1 else float("nan")
    mean = float(np.mean(chains))
    if target is not None:
        target = float(target)
        if not math.isfinite(target):
            raise ValueError("observable target must be finite")
        signed_bias = mean - target
        absolute_bias = abs(signed_bias)
    else:
        signed_bias = absolute_bias = None
    if math.isfinite(ess) and ess > 0 and math.isfinite(sample_sd):
        mcse = float(sample_sd / math.sqrt(ess))
    else:
        # Zero empirical variance from a trapped/constant trajectory is not
        # evidence of zero Monte Carlo error.
        mcse = float("nan")

    return {
        "kind": kind,
        "name": name,
        "index": int(index),
        "mean": mean,
        "target": target,
        "signed_bias": signed_bias,
        "absolute_bias": absolute_bias,
        "sample_sd": sample_sd,
        # ``iat`` is retained as the saved-draw-unit compatibility field.
        # Multiplying by the uniform trace interval yields simulation-time
        # units and makes thinning/cadence explicit.
        "iat": aggregate_iat,
        "iat_saved_draws": aggregate_iat,
        # This is time in the numerical sampler, not molecular/physical time:
        # the enhanced dynamics are designed for equilibrium sampling rather
        # than faithful kinetic trajectories.
        "iat_sampler_time": (
            aggregate_iat * trace_interval
            if math.isfinite(aggregate_iat) else aggregate_iat
        ),
        "iat_min_chain": (
            float(np.min(finite_iats)) if finite_iats.size else float("inf")
        ),
        "iat_median_chain": (
            float(np.median(finite_iats)) if finite_iats.size else float("inf")
        ),
        "iat_max_chain": (
            float("inf")
            if np.any(np.isposinf(chain_iats))
            else float(np.max(finite_iats)) if finite_iats.size else float("nan")
        ),
        "ess": ess,
        "rhat": float(split_rhat(chains)),
        "mcse": mcse,
        "constant_chain_count": int(constant.sum()),
        "finite_iat_chain_count": int(finite_iats.size),
        "n_chains": int(chains.shape[0]),
        "n_draws_per_chain": int(chains.shape[1]),
        "ess_per_second": _safe_rate(ess, wallclock_s),
        "ess_per_gradient_eval": _safe_rate(ess, gradient_evals),
        "ess_per_potential_eval": _safe_rate(ess, potential_evals),
        "ess_per_score_quadrature_eval": _safe_rate(
            ess, score_quadrature_evals
        ),
    }


def summarize_stationary_traces(
    labels_t: Any,
    energy_t: Any,
    cv_t: Any,
    trace_times: Any,
    *,
    wallclock_s: float,
    gradient_evals: float,
    potential_evals: float,
    score_quadrature_evals: float,
    basin_ids: Sequence[int] | None = None,
    cv_names: Sequence[str] | None = None,
    basin_target_probabilities: Sequence[float] | Mapping[int, float] | None = None,
    reference_energy_mean: float | None = None,
    reference_cv_means: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute stationary-chain diagnostics for basin, energy, and CV traces.

    Inputs must already exclude burn-in.  ``basin_ids`` should enumerate all
    physically defined basins, including basins that were never visited; when
    omitted, only observed labels can be inferred.  Counts are totals for the
    analyzed run, so cost-normalized ESS values are directly comparable only
    when the accounting convention is held fixed across methods.  Optional
    targets add signed and absolute bias to the same observable row as ESS, so
    high throughput cannot be interpreted without equilibrium accuracy.
    """
    times, labels, energy, cvs, interval = _validate_traces(
        trace_times, labels_t, energy_t, cv_t
    )
    wallclock_s = _validate_cost("wallclock_s", wallclock_s)
    gradient_evals = _validate_cost("gradient_evals", gradient_evals)
    potential_evals = _validate_cost("potential_evals", potential_evals)
    score_quadrature_evals = _validate_cost(
        "score_quadrature_evals", score_quadrature_evals
    )
    if wallclock_s == 0:
        raise ValueError("wallclock_s must be positive")

    observed_ids = [int(v) for v in np.unique(labels)]
    if basin_ids is None:
        selected_basin_ids = observed_ids
    else:
        selected_basin_ids = [int(v) for v in basin_ids]
        if len(set(selected_basin_ids)) != len(selected_basin_ids):
            raise ValueError("basin_ids must be unique")
        missing = sorted(set(observed_ids) - set(selected_basin_ids))
        if missing:
            raise ValueError(f"basin_ids omit observed labels: {missing}")

    n_cv = int(cvs.shape[2])
    if cv_names is None:
        selected_cv_names = [f"cv_{j}" for j in range(n_cv)]
    else:
        selected_cv_names = [str(v) for v in cv_names]
        if len(selected_cv_names) != n_cv:
            raise ValueError("cv_names length must equal cv_t.shape[2]")

    if basin_target_probabilities is None:
        basin_targets: dict[int, float] = {}
    elif isinstance(basin_target_probabilities, Mapping):
        basin_targets = {int(key): float(value)
                         for key, value in basin_target_probabilities.items()}
        if set(basin_targets) != set(selected_basin_ids):
            raise ValueError("basin target mapping must match basin_ids exactly")
    else:
        values = [float(value) for value in basin_target_probabilities]
        if len(values) != len(selected_basin_ids):
            raise ValueError("basin target probabilities must match basin_ids")
        basin_targets = dict(zip(selected_basin_ids, values))
    if basin_targets:
        values = np.asarray(list(basin_targets.values()), dtype=float)
        if (not np.all(np.isfinite(values)) or np.any(values < 0)
                or not np.isclose(values.sum(), 1.0, rtol=1e-8, atol=1e-10)):
            raise ValueError(
                "basin target probabilities must be finite, non-negative, "
                "and sum to one"
            )

    if reference_energy_mean is not None:
        reference_energy_mean = float(reference_energy_mean)
        if not math.isfinite(reference_energy_mean):
            raise ValueError("reference_energy_mean must be finite")
    if reference_cv_means is None:
        cv_targets: list[float | None] = [None] * n_cv
    else:
        cv_targets = [float(value) for value in reference_cv_means]
        if len(cv_targets) != n_cv or not all(math.isfinite(v) for v in cv_targets):
            raise ValueError("reference_cv_means must contain one finite value per CV")

    cost_args = {
        "wallclock_s": wallclock_s,
        "gradient_evals": gradient_evals,
        "potential_evals": potential_evals,
        "score_quadrature_evals": score_quadrature_evals,
        "trace_interval": interval,
    }
    rows: list[dict[str, Any]] = []
    for basin_id in selected_basin_ids:
        indicator_t = (labels == basin_id).astype(float)
        row = _observable_row(
            indicator_t.T,
            kind="basin",
            name=f"basin_{basin_id}",
            index=basin_id,
            target=basin_targets.get(basin_id),
            **cost_args,
        )
        flips = np.diff(indicator_t, axis=0) != 0
        ever = indicator_t.astype(bool).any(axis=0)
        always = indicator_t.astype(bool).all(axis=0)
        row.update({
            "basin_id": basin_id,
            "basin_transition_count": int(flips.sum()),
            "no_basin_transition_chain_count": int((~flips.any(axis=0)).sum()),
            "unvisited_chain_count": int((~ever).sum()),
            "always_in_basin_chain_count": int(always.sum()),
            "unvisited": bool(not ever.any()),
        })
        rows.append(row)

    rows.append(_observable_row(
        energy.T, kind="energy", name="energy", index=0,
        target=reference_energy_mean, **cost_args
    ))
    for j, cv_name in enumerate(selected_cv_names):
        rows.append(_observable_row(
            cvs[:, :, j].T, kind="cv", name=cv_name, index=j,
            target=cv_targets[j], **cost_args
        ))

    basin_rows = [row for row in rows if row["kind"] == "basin"]
    worst = min(basin_rows, key=lambda row: row["ess"]) if basin_rows else None
    label_flips = np.diff(labels, axis=0) != 0
    no_label_transition = ~label_flips.any(axis=0)
    basin_index = {basin_id: j for j, basin_id in enumerate(selected_basin_ids)}
    transition_counts = np.zeros(
        (len(selected_basin_ids), len(selected_basin_ids)), dtype=np.int64)
    if labels.shape[0] > 1:
        origins = labels[:-1].reshape(-1)
        destinations = labels[1:].reshape(-1)
        for origin, destination in zip(origins, destinations):
            transition_counts[basin_index[int(origin)],
                              basin_index[int(destination)]] += 1
    row_totals = transition_counts.sum(axis=1, keepdims=True)
    transition_probabilities = np.divide(
        transition_counts, row_totals,
        out=np.zeros_like(transition_counts, dtype=float), where=row_totals > 0)

    return {
        "schema_version": SCHEMA_VERSION,
        "n_draws_per_chain": int(labels.shape[0]),
        "n_chains": int(labels.shape[1]),
        "n_cv": n_cv,
        "n_basins": len(selected_basin_ids),
        "trace_start": float(times[0]),
        "trace_stop": float(times[-1]),
        "trace_interval": interval,
        "wallclock_s": wallclock_s,
        "gradient_evals": gradient_evals,
        "potential_evals": potential_evals,
        "score_quadrature_evals": score_quadrature_evals,
        "observed_basin_ids": observed_ids,
        "basin_ids": selected_basin_ids,
        "cv_names": selected_cv_names,
        "worst_basin_name": worst["name"] if worst is not None else None,
        "worst_basin_ess": float(worst["ess"]) if worst is not None else float("nan"),
        "worst_basin_ess_per_second": (
            float(worst["ess_per_second"]) if worst is not None else float("nan")
        ),
        "worst_basin_ess_per_gradient_eval": (
            float(worst["ess_per_gradient_eval"])
            if worst is not None else float("nan")
        ),
        "worst_basin_ess_per_potential_eval": (
            float(worst["ess_per_potential_eval"])
            if worst is not None else float("nan")
        ),
        "worst_basin_ess_per_score_quadrature_eval": (
            float(worst["ess_per_score_quadrature_eval"])
            if worst is not None else float("nan")
        ),
        "diagnostics": {
            "label_transition_count": int(label_flips.sum()),
            "label_transition_count_per_chain": [
                int(v) for v in label_flips.sum(axis=0)
            ],
            "no_label_transition_chain_count": int(no_label_transition.sum()),
            "all_chains_no_label_transition": bool(no_label_transition.all()),
            # Lag-one matrices use the declared trace stride. Rows for basins
            # never observed as an origin are all zero rather than normalized
            # into a fictitious transition distribution.
            "transition_basin_ids": selected_basin_ids,
            "lag_one_transition_counts": transition_counts.tolist(),
            "lag_one_transition_probabilities": transition_probabilities.tolist(),
            "any_constant_observable_chain": bool(
                any(row["constant_chain_count"] > 0 for row in rows)
            ),
        },
        "observables": rows,
    }



def collect_stationary_trajectories(
    sampler_factory: Callable[[str, int], Any],
    methods: Sequence[str],
    seeds: Sequence[int],
    *,
    n_draws: int,
    steps_per_draw: int,
    dt: float,
    labels_fn: Callable[[Any], Any],
    energy_fn: Callable[[Any], Any],
    cv_fn: Callable[[Any], Any],
    counter_source: Any | None = None,
    warmup_steps: int = 0,
    burn_in_steps: int = 0,
    basin_ids: Sequence[int] | None = None,
    cv_names: Sequence[str] | None = None,
    basin_target_probabilities: Sequence[float] | Mapping[int, float] | None = None,
    reference_energy_mean: float | None = None,
    reference_cv_means: Sequence[float] | None = None,
    equilibrium_initialized: bool = True,
    initialization_method: str = "reference_sampler",
    synchronize_fn: Callable[[], None] | None = None,
    timer_fn: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Collect uniformly spaced stationary trajectories and summarize them.

    ``sampler_factory(method, seed)`` must return a fresh reference-initialized
    small-chain sampler with ``step()`` and ``positions()`` methods.  Set
    ``equilibrium_initialized=False`` when that reference is approximate (for
    example finite SIR); this provenance is propagated so the resulting IAT is
    not overstated as an exact-stationary diagnostic. Starting from a target
    reference where available avoids conflating burn-in relaxation with
    autocorrelation. ``warmup_steps`` runs on a throwaway sampler and is excluded
    only to absorb allocator/JIT startup, matching the production timing policy.
    ``burn_in_steps`` is a declared settling period on the analyzed sampler;
    its time and evaluation cost are included rather than hidden.

    Methods and seeds run sequentially.  CUDA is synchronized immediately
    before and after every sampler-only timing segment.  Observation functions
    run outside the timer and under the potential's ``no_count`` context, so
    wall time and evaluation deltas describe only the sampler.  The aggregate
    method summary concatenates all particles from all seeds as independent
    chains; raw arrays retain ``seed_index`` and ``chain_index_within_seed``.
    """
    selected_methods = [str(method) for method in methods]
    selected_seeds = [int(seed) for seed in seeds]
    if not isinstance(equilibrium_initialized, bool):
        raise ValueError("equilibrium_initialized must be boolean")
    initialization_method = str(initialization_method).strip()
    if not initialization_method:
        raise ValueError("initialization_method must be nonempty")
    if not selected_methods or len(set(selected_methods)) != len(selected_methods):
        raise ValueError("methods must be a non-empty unique sequence")
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("seeds must be a non-empty unique sequence")
    if isinstance(n_draws, bool) or int(n_draws) != n_draws or n_draws < 2:
        raise ValueError("n_draws must be an integer >= 2")
    if (isinstance(steps_per_draw, bool) or int(steps_per_draw) != steps_per_draw
            or steps_per_draw < 1):
        raise ValueError("steps_per_draw must be a positive integer")
    if (isinstance(warmup_steps, bool) or int(warmup_steps) != warmup_steps
            or warmup_steps < 0):
        raise ValueError("warmup_steps must be a non-negative integer")
    if (isinstance(burn_in_steps, bool) or int(burn_in_steps) != burn_in_steps
            or burn_in_steps < 0):
        raise ValueError("burn_in_steps must be a non-negative integer")
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    n_draws = int(n_draws)
    steps_per_draw = int(steps_per_draw)
    warmup_steps = int(warmup_steps)
    burn_in_steps = int(burn_in_steps)
    synchronize = synchronize_fn or _cuda_synchronize_if_available
    timer = timer_fn or time.perf_counter
    trace_times = dt * (
        burn_in_steps + steps_per_draw * np.arange(1, n_draws + 1, dtype=float)
    )
    validate_trace_times(trace_times)

    method_results: dict[str, Any] = {}
    for method in selected_methods:
        position_blocks: list[np.ndarray] = []
        label_blocks: list[np.ndarray] = []
        energy_blocks: list[np.ndarray] = []
        cv_blocks: list[np.ndarray] = []
        seed_index: list[np.ndarray] = []
        chain_index: list[np.ndarray] = []
        seed_summaries: dict[int, dict[str, Any]] = {}
        run_records: list[dict[str, Any]] = []

        for seed in selected_seeds:
            if warmup_steps:
                warm = sampler_factory(method, seed)
                for _ in range(warmup_steps):
                    warm.step()
                synchronize()
                del warm
            sampler = sampler_factory(method, seed)
            source = _resolve_counter_source(sampler, counter_source)
            counters_before = _evaluation_counters(source)
            wallclock_s = _timed_steps(
                sampler, burn_in_steps, synchronize_fn=synchronize, timer_fn=timer
            )
            positions_frames: list[np.ndarray] = []
            labels_frames: list[np.ndarray] = []
            energy_frames: list[np.ndarray] = []
            cv_frames: list[np.ndarray] = []
            for _ in range(n_draws):
                wallclock_s += _timed_steps(
                    sampler, steps_per_draw,
                    synchronize_fn=synchronize, timer_fn=timer,
                )
                positions, labels, energy, cvs = _observe_positions(
                    sampler, labels_fn=labels_fn, energy_fn=energy_fn,
                    cv_fn=cv_fn, counter_source=source,
                )
                positions_frames.append(positions)
                labels_frames.append(labels)
                energy_frames.append(energy)
                cv_frames.append(cvs)

            counters_after = _evaluation_counters(source)
            deltas = tuple(after - before for before, after
                           in zip(counters_before, counters_after))
            if any(delta < 0 for delta in deltas):
                raise RuntimeError("evaluation counters decreased during collection")
            if wallclock_s <= 0:
                raise RuntimeError("sampler wall-clock duration must be positive")

            positions_t = np.stack(positions_frames, axis=0)
            labels_t = np.stack(labels_frames, axis=0)
            energy_t = np.stack(energy_frames, axis=0)
            cv_t = np.stack(cv_frames, axis=0)
            seed_summary = summarize_stationary_traces(
                labels_t, energy_t, cv_t, trace_times,
                wallclock_s=wallclock_s,
                potential_evals=deltas[0],
                gradient_evals=deltas[1],
                score_quadrature_evals=deltas[2],
                basin_ids=basin_ids,
                cv_names=cv_names,
                basin_target_probabilities=basin_target_probabilities,
                reference_energy_mean=reference_energy_mean,
                reference_cv_means=reference_cv_means,
            )
            seed_summary.update({
                "method": method,
                "seed": seed,
                "warmup_steps": warmup_steps,
                "burn_in_steps": burn_in_steps,
                "steps_per_draw": steps_per_draw,
                "dt": dt,
                "equilibrium_initialized": equilibrium_initialized,
                "initialization_method": initialization_method,
            })
            seed_summaries[seed] = seed_summary
            diagnostics = (sampler.pop_diagnostics()
                           if hasattr(sampler, "pop_diagnostics") else {})
            run_records.append({
                "method": method,
                "seed": seed,
                "warmup_steps": warmup_steps,
                "wallclock_s": wallclock_s,
                "potential_evals": deltas[0],
                "gradient_evals": deltas[1],
                "score_quadrature_evals": deltas[2],
                "n_chains": int(labels_t.shape[1]),
                "sampler_diagnostics": diagnostics,
            })
            position_blocks.append(positions_t)
            label_blocks.append(labels_t)
            energy_blocks.append(energy_t)
            cv_blocks.append(cv_t)
            seed_index.append(np.full(labels_t.shape[1], seed, dtype=np.int64))
            chain_index.append(np.arange(labels_t.shape[1], dtype=np.int64))

        # Time and observable dimensions must match; the chain count may vary
        # by seed and is therefore concatenated rather than stacked.
        try:
            positions_all = np.concatenate(position_blocks, axis=1)
            labels_all = np.concatenate(label_blocks, axis=1)
            energy_all = np.concatenate(energy_blocks, axis=1)
            cvs_all = np.concatenate(cv_blocks, axis=1)
        except ValueError as error:
            raise ValueError(
                f"inconsistent position/CV dimensions across seeds for {method}"
            ) from error
        wallclock_total = float(sum(run["wallclock_s"] for run in run_records))
        potential_total = int(sum(run["potential_evals"] for run in run_records))
        gradient_total = int(sum(run["gradient_evals"] for run in run_records))
        score_total = int(sum(
            run["score_quadrature_evals"] for run in run_records
        ))
        aggregate_summary = summarize_stationary_traces(
            labels_all, energy_all, cvs_all, trace_times,
            wallclock_s=wallclock_total,
            potential_evals=potential_total,
            gradient_evals=gradient_total,
            score_quadrature_evals=score_total,
            basin_ids=basin_ids,
            cv_names=cv_names,
            basin_target_probabilities=basin_target_probabilities,
            reference_energy_mean=reference_energy_mean,
            reference_cv_means=reference_cv_means,
        )
        aggregate_summary.update({
            "method": method,
            "seeds": selected_seeds,
            "warmup_steps": warmup_steps,
            "burn_in_steps": burn_in_steps,
            "steps_per_draw": steps_per_draw,
            "dt": dt,
            "equilibrium_initialized": equilibrium_initialized,
            "initialization_method": initialization_method,
        })
        method_results[method] = {
            "summary": aggregate_summary,
            "seed_summaries": seed_summaries,
            "runs": run_records,
            "raw": {
                "trace_times": trace_times.copy(),
                "positions_t": positions_all,
                "labels_t": labels_all,
                "energy_t": energy_all,
                "cv_t": cvs_all,
                "seed_index": np.concatenate(seed_index),
                "chain_index_within_seed": np.concatenate(chain_index),
            },
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "methods": method_results,
        "collection": {
            "n_draws": n_draws,
            "steps_per_draw": steps_per_draw,
            "warmup_steps": warmup_steps,
            "burn_in_steps": burn_in_steps,
            "dt": dt,
            "equilibrium_initialized": equilibrium_initialized,
            "initialization_method": initialization_method,
        },
    }


def flat_summary_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return CSV-ready rows with run-level metadata repeated per observable."""
    run_fields = {
        key: summary.get(key)
        for key in (
            "schema_version", "method", "seed", "seeds", "warmup_steps",
            "burn_in_steps", "steps_per_draw", "dt", "equilibrium_initialized",
            "initialization_method",
            "n_draws_per_chain", "n_chains", "trace_start",
            "trace_stop", "trace_interval", "wallclock_s", "gradient_evals",
            "potential_evals", "score_quadrature_evals", "worst_basin_name",
            "worst_basin_ess", "worst_basin_ess_per_second",
            "worst_basin_ess_per_gradient_eval",
            "worst_basin_ess_per_potential_eval",
            "worst_basin_ess_per_score_quadrature_eval",
        )
    }
    diagnostics = summary.get("diagnostics", {})
    run_fields.update({
        "label_transition_count": diagnostics.get("label_transition_count"),
        "no_label_transition_chain_count": diagnostics.get(
            "no_label_transition_chain_count"
        ),
        "all_chains_no_label_transition": diagnostics.get(
            "all_chains_no_label_transition"
        ),
    })
    return [{**run_fields, **dict(row)} for row in summary.get("observables", [])]


def write_stationarity_csv(
    path: str | Path,
    summary: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Write one flat row per observable, refusing overwrite by default."""
    rows = flat_summary_rows(summary)
    if not rows:
        raise ValueError("summary contains no observable rows")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    fieldnames = list(rows[0])
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with output.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_stationarity_npz(
    path: str | Path,
    *,
    trace_times: Any,
    labels_t: Any,
    energy_t: Any,
    cv_t: Any,
    positions_t: Any | None = None,
    seed_index: Any | None = None,
    chain_index_within_seed: Any | None = None,
    summary: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Save raw stationary traces plus optional JSON metadata in compressed NPZ."""
    times, labels, energy, cvs, _ = _validate_traces(
        trace_times, labels_t, energy_t, cv_t
    )
    optional_arrays: dict[str, np.ndarray] = {}
    if positions_t is not None:
        positions = _as_numpy(positions_t)
        if positions.ndim != 3 or positions.shape[:2] != labels.shape:
            raise ValueError("positions_t must have shape (n_draws, n_chains, d)")
        optional_arrays["positions_t"] = np.array(positions, copy=True)
    for key, value in (("seed_index", seed_index),
                       ("chain_index_within_seed", chain_index_within_seed)):
        if value is not None:
            array = _as_numpy(value)
            if array.shape != (labels.shape[1],):
                raise ValueError(f"{key} must have shape (n_chains,)")
            optional_arrays[key] = np.array(array, copy=True)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if overwrite else "xb"
    payload = {
        "trace_times": times,
        "labels_t": labels,
        "energy_t": energy,
        "cv_t": cvs,
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "summary_json": np.asarray(json.dumps(
            _json_safe(summary or {}), sort_keys=True, allow_nan=False)),
        "metadata_json": np.asarray(json.dumps(
            _json_safe(metadata or {}), sort_keys=True, allow_nan=False)),
        **optional_arrays,
    }
    with output.open(mode) as handle:
        np.savez_compressed(handle, **payload)
    return output


def read_stationarity_npz(path: str | Path) -> dict[str, Any]:
    """Load a file written by :func:`write_stationarity_npz` without pickle."""
    with np.load(Path(path), allow_pickle=False) as data:
        result = {
            "trace_times": data["trace_times"].copy(),
            "labels_t": data["labels_t"].copy(),
            "energy_t": data["energy_t"].copy(),
            "cv_t": data["cv_t"].copy(),
            "schema_version": int(data["schema_version"]),
            "summary": json.loads(str(data["summary_json"])),
            "metadata": json.loads(str(data["metadata_json"])),
        }
        for key in ("positions_t", "seed_index", "chain_index_within_seed"):
            if key in data:
                result[key] = data[key].copy()
        return result
