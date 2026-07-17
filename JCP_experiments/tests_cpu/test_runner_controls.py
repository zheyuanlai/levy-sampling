from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import math
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runner import (  # noqa: E402
    RefinementError,
    _metrics_agree,
    checkpoint_schedule,
    quadrature_refinement,
    refine_dt,
    run_experiment_batched,
    run_one,
    time_to_threshold,
)


class _Potential:
    def __init__(self):
        self.n_V = 0
        self.n_grad = 0
        self.n_Vdelta = 0

    def nfe(self):
        return self.n_V + self.n_grad + self.n_Vdelta

    @contextmanager
    def no_count(self):
        saved = (self.n_V, self.n_grad, self.n_Vdelta)
        try:
            yield
        finally:
            self.n_V, self.n_grad, self.n_Vdelta = saved


class _Sampler:
    def __init__(self, potential):
        self.potential = potential
        self.x = torch.zeros(1, 1, dtype=torch.float64)

    def step(self):
        self.x += 1.0
        self.potential.n_grad += 1

    def positions(self):
        return self.x

    def pop_diagnostics(self):
        return {}


def test_short_checkpoint_schedule_is_bounded_strict_and_complete():
    assert checkpoint_schedule(7) == list(range(1, 8))
    for n_steps in (1, 2, 59, 60, 61, 1_000):
        schedule = checkpoint_schedule(n_steps)
        assert schedule[-1] == n_steps
        assert all(1 <= step <= n_steps for step in schedule)
        assert all(a < b for a, b in zip(schedule, schedule[1:]))


@pytest.mark.parametrize("n_steps,steps_per_ck,expected", [
    (10, 4, [4, 8, 10]),
    (3, 8, [3]),
    (8, 4, [4, 8]),
])
def test_run_one_executes_remainder(n_steps, steps_per_ck, expected):
    potential = _Potential()

    def factory(method, seed):
        return _Sampler(potential)

    rows, info = run_one(
        "dummy", 0, factory, n_steps=n_steps, steps_per_ck=steps_per_ck,
        dt=0.1, metrics_fn=lambda x: {"position": float(x.item())},
        potential=potential, warmup=2, quiet=True,
    )
    assert [row["step"] for row in rows] == expected
    assert rows[-1]["position"] == pytest.approx(float(n_steps))
    assert info["grad_evals_per_step"] == pytest.approx(1.0)


def test_time_to_threshold_uses_only_requested_complete_seed_set():
    rows = []
    for step, requested_value in ((1, 0.1), (2, 0.1)):
        for seed in (0, 1):
            rows.append({"method": "A", "seed": seed, "step": step,
                         "t": float(step), "TV": requested_value})
        # This unrequested bad seed used to contaminate the seed mean.
        rows.append({"method": "A", "seed": 99, "step": step,
                     "t": float(step), "TV": 10.0})
    # An incomplete earlier checkpoint must not change the persistence count.
    rows.append({"method": "A", "seed": 0, "step": 0,
                 "t": 0.0, "TV": 0.0})
    assert time_to_threshold(rows, "A", (0, 1), floor=0.1, persist=2) == 1.0
    assert math.isinf(time_to_threshold(rows, "A", (), floor=0.1, persist=1))


def test_zero_reference_uses_absolute_plus_relative_tolerance():
    assert _metrics_agree(1e-13, 0.0, {}, tol=0.05)
    assert not _metrics_agree(1.0, 0.0, {}, tol=0.05)
    assert not _metrics_agree(float("nan"), 0.0, {}, tol=0.05)


def test_quadrature_refinement_fails_closed_and_preserves_table():
    settings = [{"q": 2}, {"q": 4}]
    with pytest.raises(RefinementError) as caught:
        quadrature_refinement(
            settings,
            run_terminal_fn=lambda q: {"metric": 1.0 / q},
            cert_fn=lambda q: 1.0,
            floors={}, r_max=1e-6,
        )
    error = caught.value
    assert error.kind == "quadrature" and error.status == "failed"
    assert len(error.table) == 2
    assert error.next_candidate == {"q": 4}
    assert not any(row["pass"] for row in error.table)


def test_timestep_refinement_fails_closed_instead_of_approving_next_dt():
    def terminal(dt):
        return {"method": {"metric": dt}}

    with pytest.raises(RefinementError) as caught:
        refine_dt(terminal, dt0=1.0, floors={}, tol=0.01, max_halvings=2)
    error = caught.value
    assert error.kind == "timestep" and error.status == "failed"
    assert len(error.table) == 2
    assert error.next_candidate == pytest.approx(0.25)


def test_refinements_still_return_certified_choices():
    chosen, qtable = quadrature_refinement(
        [{"q": 2}, {"q": 4}],
        run_terminal_fn=lambda q: {"metric": 1.0},
        cert_fn=lambda q: 0.0,
        floors={},
    )
    assert chosen == {"q": 2}
    assert qtable[0]["pass"] is True

    dt, dtable = refine_dt(
        lambda h: {"method": {"metric": 1.0 + 1e-4 * h}},
        dt0=1.0, floors={}, tol=0.01, max_halvings=2,
    )
    assert dt == 1.0
    assert dtable[0]["pass"] is True


class _BatchedDiagnosticSampler(_Sampler):
    def __init__(self, potential):
        super().__init__(potential)
        self.x = torch.zeros(2, 1, dtype=torch.float64)

    def pop_diagnostics(self):
        return {
            "jump_count_cumulative": 7,
            "nonfinite_proposal_count_cumulative": 2,
            "jump_rate_per_particle_time_cumulative": 1.75,
            "nonfinite_proposal_fraction_cumulative": 0.25,
        }


def test_batched_global_cumulative_counts_are_not_copied_to_seed_rows():
    potential = _Potential()
    rows, info = run_experiment_batched(
        ["dummy"], (0, 1),
        lambda method: _BatchedDiagnosticSampler(potential),
        n_steps=1, steps_per_ck=1, dt=0.1,
        metrics_fn=lambda x: {"position": float(x.mean())},
        potential=potential, n_per_seed=1, warmup=0,
    )
    terminal = [row for row in rows if row["step"] == 1]
    assert len(terminal) == 2
    assert all("jump_count_cumulative" not in row for row in terminal)
    assert all("nonfinite_proposal_count_cumulative" not in row
               for row in terminal)
    assert all(row["jump_rate_per_particle_time_cumulative"] == 1.75
               for row in terminal)
    assert all(row["nonfinite_proposal_fraction_cumulative"] == 0.25
               for row in terminal)
    assert info["dummy"]["jump_count_cumulative"] == 7
    assert info["dummy"]["nonfinite_proposal_count_cumulative"] == 2
