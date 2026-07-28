from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from launch_production import RELEASE_PREFLIGHT_TESTS  # noqa: E402
from scripts.validate_release import validate_release  # noqa: E402
from src.experiments import build_e1, make_sampler_factory  # noqa: E402
from src.manuscript import EXPERIMENTS  # noqa: E402
from src.samplers import geometric_ladder  # noqa: E402


def test_release_tree_and_frozen_results_are_complete():
    report = validate_release(ROOT, check_results=True, require_figures=True)
    assert report["status"] == "passed"
    assert tuple(report["experiments"]) == tuple(EXPERIMENTS)


def test_production_preflight_is_scoped_to_e1_e4():
    assert RELEASE_PREFLIGHT_TESTS
    assert not any(
        "e5" in path.lower() or "alanine" in path.lower()
        for path in RELEASE_PREFLIGHT_TESTS
    )


def test_all_e1_release_methods_take_a_finite_cpu_step():
    torch.set_default_dtype(torch.float64)
    experiment = build_e1(device="cpu")
    betas = geometric_ladder(
        experiment.cfg.beta, experiment.pt_beta_min, 3, "cpu"
    )
    factory = make_sampler_factory(
        experiment,
        experiment.cfg.dt,
        betas,
        n_particles=12,
        score_kwargs={"q_theta": 2, "q_rho": 2},
    )
    for method in EXPERIMENTS["double_well"].methods:
        sampler = factory(method, 0)
        sampler.step()
        positions = sampler.positions()
        assert positions.shape == (12, 1), method
        assert torch.isfinite(positions).all(), method
