from __future__ import annotations

import csv
import math
from contextlib import contextmanager
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stationarity import (  # noqa: E402
    collect_stationary_trajectories,
    flat_summary_rows,
    read_stationarity_npz,
    summarize_stationary_traces,
    validate_trace_times,
    write_stationarity_csv,
    write_stationarity_npz,
)



class _ToyPotential:
    def __init__(self):
        self.n_V = 0
        self.n_grad = 0
        self.n_Vdelta = 0

    @contextmanager
    def no_count(self):
        before = (self.n_V, self.n_grad, self.n_Vdelta)
        try:
            yield
        finally:
            self.n_V, self.n_grad, self.n_Vdelta = before

    def V(self, x):
        self.n_V += x.shape[0]
        return x[:, 0]


class _ToySampler:
    def __init__(self, potential, seed: int, *, trapped: bool = False):
        self.pot = potential
        self.seed = seed
        self.trapped = trapped
        self.x = (np.full((2, 1), -1.0) if trapped
                  else np.asarray([[-1.0], [1.0]]))
        self.n_steps = 0

    def step(self):
        n = self.x.shape[0]
        self.pot.n_V += 2 * n
        self.pot.n_grad += n
        self.pot.n_Vdelta += 3 * n
        if not self.trapped:
            self.x *= -1.0
        self.n_steps += 1

    def positions(self):
        return self.x

    def pop_diagnostics(self):
        return {"steps": self.n_steps}


class _FixedIncrementClock:
    def __init__(self, increment: float):
        self.value = 0.0
        self.increment = increment

    def __call__(self):
        value = self.value
        self.value += self.increment
        return value

def _example_traces(n_draws: int = 2_000, n_chains: int = 4):
    rng = np.random.default_rng(20260716)
    times = 5.0 + 0.25 * np.arange(n_draws)
    labels = rng.integers(0, 2, size=(n_draws, n_chains))
    energy = rng.normal(size=(n_draws, n_chains))
    cvs = np.stack([
        rng.normal(size=(n_draws, n_chains)),
        labels + 0.1 * rng.normal(size=(n_draws, n_chains)),
    ], axis=-1)
    return times, labels, energy, cvs


def test_validate_trace_times_requires_uniform_strict_finite_grid():
    assert validate_trace_times([2.0, 2.25, 2.5]) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="uniformly spaced"):
        validate_trace_times([0.0, 1.0, 2.1])
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_trace_times([0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        validate_trace_times([0.0, float("nan")])
    with pytest.raises(ValueError, match="at least two"):
        validate_trace_times([0.0])


def test_stationary_summary_reports_all_observables_and_cost_rates():
    times, labels, energy, cvs = _example_traces()
    summary = summarize_stationary_traces(
        labels, energy, cvs, times,
        wallclock_s=20.0,
        gradient_evals=80_000,
        potential_evals=160_000,
        score_quadrature_evals=320_000,
        basin_ids=[0, 1],
        cv_names=["x", "mode_coordinate"],
    )
    assert summary["trace_interval"] == pytest.approx(0.25)
    assert summary["n_draws_per_chain"] == 2_000
    assert summary["n_chains"] == 4
    assert [row["name"] for row in summary["observables"]] == [
        "basin_0", "basin_1", "energy", "x", "mode_coordinate"
    ]

    for row in summary["observables"]:
        assert row["ess"] > 0
        assert row["iat"] == pytest.approx(8_000 / row["ess"])
        assert row["iat_saved_draws"] == pytest.approx(row["iat"])
        assert row["iat_sampler_time"] == pytest.approx(0.25 * row["iat"])
        assert "iat_physical_time" not in row
        assert row["ess_per_second"] == pytest.approx(row["ess"] / 20.0)
        assert row["ess_per_gradient_eval"] == pytest.approx(row["ess"] / 80_000)
        assert row["ess_per_potential_eval"] == pytest.approx(row["ess"] / 160_000)
        assert row["ess_per_score_quadrature_eval"] == pytest.approx(
            row["ess"] / 320_000
        )
        assert row["mcse"] == pytest.approx(row["sample_sd"] / math.sqrt(row["ess"]))
        assert 0.98 < row["rhat"] < 1.03

    basin_ess = [row["ess"] for row in summary["observables"]
                 if row["kind"] == "basin"]
    assert summary["worst_basin_ess"] == min(basin_ess)
    assert summary["worst_basin_ess_per_second"] == pytest.approx(
        min(basin_ess) / 20.0
    )
    counts = np.asarray(summary["diagnostics"]["lag_one_transition_counts"])
    probabilities = np.asarray(
        summary["diagnostics"]["lag_one_transition_probabilities"])
    assert counts.shape == probabilities.shape == (2, 2)
    assert counts.sum() == (2_000 - 1) * 4
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_constant_and_no_transition_chains_are_not_false_precision():
    n_draws, n_chains = 100, 3
    times = np.arange(n_draws, dtype=float)
    labels = np.zeros((n_draws, n_chains), dtype=int)
    energy = np.ones((n_draws, n_chains))
    cvs = np.zeros((n_draws, n_chains, 1))
    summary = summarize_stationary_traces(
        labels, energy, cvs, times,
        wallclock_s=1.0,
        gradient_evals=100,
        potential_evals=100,
        score_quadrature_evals=0,
        basin_ids=[0, 1],
    )
    by_name = {row["name"]: row for row in summary["observables"]}

    assert summary["diagnostics"]["all_chains_no_label_transition"] is True
    assert summary["diagnostics"]["no_label_transition_chain_count"] == 3
    assert summary["worst_basin_ess"] == 0.0
    assert by_name["basin_1"]["unvisited"] is True
    assert by_name["basin_1"]["unvisited_chain_count"] == 3
    assert by_name["basin_0"]["always_in_basin_chain_count"] == 3
    for row in summary["observables"]:
        assert row["constant_chain_count"] == 3
        assert row["ess"] == 0.0
        assert math.isinf(row["iat"])
        assert math.isnan(row["mcse"])
        assert math.isnan(row["ess_per_score_quadrature_eval"])


def test_trace_shape_labels_and_basin_validation_fail_closed():
    times, labels, energy, cvs = _example_traces(20, 2)
    kwargs = dict(
        wallclock_s=1.0, gradient_evals=1, potential_evals=1,
        score_quadrature_evals=1,
    )
    with pytest.raises(ValueError, match="same"):
        summarize_stationary_traces(
            labels, energy[:, :1], cvs, times, **kwargs
        )
    with pytest.raises(ValueError, match="cv_t"):
        summarize_stationary_traces(
            labels, energy, cvs[:, :, 0], times, **kwargs
        )
    fractional = labels.astype(float)
    fractional[0, 0] = 0.5
    with pytest.raises(ValueError, match="integer"):
        summarize_stationary_traces(
            fractional, energy, cvs, times, **kwargs
        )
    with pytest.raises(ValueError, match="omit observed"):
        summarize_stationary_traces(
            labels, energy, cvs, times, basin_ids=[0], **kwargs
        )


def test_flat_csv_and_npz_round_trip_refuse_overwrite(tmp_path):
    times, labels, energy, cvs = _example_traces(100, 4)
    summary = summarize_stationary_traces(
        labels, energy, cvs, times,
        wallclock_s=2.0,
        gradient_evals=1_000,
        potential_evals=2_000,
        score_quadrature_evals=3_000,
        basin_ids=[0, 1],
        cv_names=["x", "y"],
    )
    rows = flat_summary_rows(summary)
    assert len(rows) == 5
    assert all("worst_basin_ess" in row for row in rows)

    csv_path = write_stationarity_csv(tmp_path / "summary.csv", summary)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        persisted = list(csv.DictReader(handle))
    assert len(persisted) == 5
    assert persisted[0]["kind"] == "basin"
    with pytest.raises(FileExistsError):
        write_stationarity_csv(csv_path, summary)

    npz_path = write_stationarity_npz(
        tmp_path / "traces.npz",
        trace_times=times,
        labels_t=labels,
        energy_t=energy,
        cv_t=cvs,
        positions_t=cvs[:, :, :1],
        seed_index=np.asarray([0, 0, 1, 1]),
        chain_index_within_seed=np.asarray([0, 1, 0, 1]),
        summary={**summary, "strict_json_sentinels": [float("nan"), float("inf")]},
        metadata={"method": "LSC-CP"},
    )
    loaded = read_stationarity_npz(npz_path)
    assert np.array_equal(loaded["trace_times"], times)
    assert np.array_equal(loaded["labels_t"], labels)
    assert np.array_equal(loaded["energy_t"], energy)
    assert np.array_equal(loaded["cv_t"], cvs)
    assert np.array_equal(loaded["positions_t"], cvs[:, :, :1])
    assert np.array_equal(loaded["seed_index"], [0, 0, 1, 1])
    assert np.array_equal(loaded["chain_index_within_seed"], [0, 1, 0, 1])
    assert loaded["summary"]["worst_basin_name"] == summary["worst_basin_name"]
    assert loaded["summary"]["strict_json_sentinels"] == ["nan", "inf"]
    assert loaded["metadata"] == {"method": "LSC-CP"}
    with pytest.raises(FileExistsError):
        write_stationarity_npz(
            npz_path, trace_times=times, labels_t=labels,
            energy_t=energy, cv_t=cvs,
        )


def test_targets_pair_bias_with_ess_in_each_observable_row():
    times, labels, energy, cvs = _example_traces(500, 4)
    summary = summarize_stationary_traces(
        labels, energy, cvs, times,
        wallclock_s=2.0,
        gradient_evals=1_000,
        potential_evals=2_000,
        score_quadrature_evals=3_000,
        basin_ids=[0, 1],
        cv_names=["x", "mode_coordinate"],
        basin_target_probabilities={0: 0.4, 1: 0.6},
        reference_energy_mean=0.25,
        reference_cv_means=[-0.5, 0.75],
    )
    targets = {
        "basin_0": 0.4, "basin_1": 0.6, "energy": 0.25,
        "x": -0.5, "mode_coordinate": 0.75,
    }
    for row in summary["observables"]:
        assert row["target"] == pytest.approx(targets[row["name"]])
        assert row["signed_bias"] == pytest.approx(
            row["mean"] - targets[row["name"]]
        )
        assert row["absolute_bias"] == pytest.approx(abs(row["signed_bias"]))
        assert "ess" in row

    without_targets = summarize_stationary_traces(
        labels, energy, cvs, times,
        wallclock_s=2.0, gradient_evals=1, potential_evals=1,
        score_quadrature_evals=1,
    )
    assert all(row["target"] is None for row in without_targets["observables"])
    assert all(row["signed_bias"] is None
               for row in without_targets["observables"])


def test_collector_records_post_step_uniform_traces_walltime_and_counters():
    potential = _ToyPotential()
    sync_calls = []
    clock = _FixedIncrementClock(0.5)

    result = collect_stationary_trajectories(
        lambda method, seed: _ToySampler(potential, seed),
        methods=["toy"], seeds=[3, 7],
        n_draws=6, steps_per_draw=3, burn_in_steps=2, dt=0.1,
        labels_fn=lambda x: (x[:, 0] > 0).astype(int),
        energy_fn=potential.V,
        cv_fn=lambda x: x,
        counter_source=potential,
        basin_ids=[0, 1], cv_names=["x"],
        basin_target_probabilities=[0.5, 0.5],
        reference_energy_mean=0.0,
        reference_cv_means=[0.0],
        synchronize_fn=lambda: sync_calls.append(1),
        timer_fn=clock,
    )
    method = result["methods"]["toy"]
    raw = method["raw"]
    summary = method["summary"]

    assert np.allclose(raw["trace_times"], [0.5, 0.8, 1.1, 1.4, 1.7, 2.0])
    assert validate_trace_times(raw["trace_times"]) == pytest.approx(0.3)
    assert raw["positions_t"].shape == (6, 4, 1)
    assert raw["labels_t"].shape == (6, 4)
    assert raw["energy_t"].shape == (6, 4)
    assert raw["cv_t"].shape == (6, 4, 1)
    assert np.array_equal(raw["seed_index"], [3, 3, 7, 7])
    assert np.array_equal(raw["chain_index_within_seed"], [0, 1, 0, 1])

    # Each seed has one burn-in timing segment plus six draw segments. The
    # injected clock gives exactly 0.5 s per synchronized segment.
    assert len(sync_calls) == 2 * (1 + 6) * 2
    assert [run["wallclock_s"] for run in method["runs"]] == [3.5, 3.5]
    assert summary["wallclock_s"] == pytest.approx(7.0)

    # Twenty sampler steps/seed and two chains: per step V=4, grad=2, Q=6.
    assert summary["potential_evals"] == 160
    assert summary["gradient_evals"] == 80
    assert summary["score_quadrature_evals"] == 240
    assert [run["sampler_diagnostics"]["steps"]
            for run in method["runs"]] == [20, 20]
    assert summary["trace_interval"] == pytest.approx(0.3)
    assert summary["equilibrium_initialized"] is True
    for row in summary["observables"]:
        assert row["target"] is not None
        assert row["absolute_bias"] == pytest.approx(0.0)


def test_collector_flags_trapped_constant_chains_and_bias():
    potential = _ToyPotential()
    result = collect_stationary_trajectories(
        lambda method, seed: _ToySampler(potential, seed, trapped=True),
        methods=["trapped"], seeds=[11],
        n_draws=10, steps_per_draw=1, dt=0.2,
        labels_fn=lambda x: (x[:, 0] > 0).astype(int),
        energy_fn=potential.V,
        cv_fn=lambda x: x,
        counter_source=potential,
        basin_ids=[0, 1], cv_names=["x"],
        basin_target_probabilities=[0.5, 0.5],
        reference_energy_mean=0.0,
        reference_cv_means=[0.0],
        synchronize_fn=lambda: None,
        timer_fn=_FixedIncrementClock(0.1),
    )
    summary = result["methods"]["trapped"]["summary"]
    by_name = {row["name"]: row for row in summary["observables"]}
    assert summary["diagnostics"]["all_chains_no_label_transition"] is True
    assert summary["worst_basin_ess"] == 0.0
    assert by_name["basin_0"]["signed_bias"] == pytest.approx(0.5)
    assert by_name["basin_1"]["signed_bias"] == pytest.approx(-0.5)
    assert by_name["energy"]["signed_bias"] == pytest.approx(-1.0)
    assert by_name["x"]["absolute_bias"] == pytest.approx(1.0)
    assert all(row["ess"] == 0.0 for row in summary["observables"])
    assert all(math.isnan(row["mcse"]) for row in summary["observables"])
