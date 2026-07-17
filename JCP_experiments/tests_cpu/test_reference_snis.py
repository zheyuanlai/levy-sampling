from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.references import LaplaceMixture  # noqa: E402


class _StandardNormalPotential:
    @staticmethod
    def V(x):
        return 0.5 * x.square().sum(dim=-1)


class _NonfinitePotential:
    @staticmethod
    def V(x):
        return torch.full((x.shape[0],), float("nan"), dtype=x.dtype,
                          device=x.device)


def _matching_reference():
    means = torch.zeros(1, 1, dtype=torch.float64)
    hessians = torch.ones(1, 1, 1, dtype=torch.float64)
    energies = torch.zeros(1, dtype=torch.float64)
    return LaplaceMixture(means, hessians, energies, beta=1.0)


def test_direct_snis_weights_and_diagnostics_for_matching_proposal():
    reference = _matching_reference()
    gen = torch.Generator(device="cpu").manual_seed(3)
    x, weights, diagnostics = reference.snis_weighted_proposals(
        200, gen, _StandardNormalPotential(), beta=1.0
    )
    assert x.shape == (200, 1)
    assert torch.allclose(weights, torch.full_like(weights, 1.0 / 200),
                          atol=1e-14, rtol=1e-12)
    assert diagnostics["reference_method"] == "self_normalized_importance_sampling"
    assert diagnostics["proposal_ess"] == pytest.approx(200.0)
    assert diagnostics["proposal_ess_fraction"] == pytest.approx(1.0)
    assert diagnostics["nonfinite_log_weight_count"] == 0


def test_direct_weighted_expectation_and_category_probabilities():
    values = torch.tensor([[1.0, 10.0], [3.0, 30.0]], dtype=torch.float64)
    weights = torch.tensor([0.25, 0.75], dtype=torch.float64)
    estimate = LaplaceMixture.weighted_expectation(values, weights)
    assert torch.allclose(estimate, torch.tensor([2.5, 25.0], dtype=torch.float64))

    labels = torch.tensor([0, 1, 1])
    probabilities = LaplaceMixture.weighted_category_probabilities(
        labels, 2, torch.tensor([0.2, 0.3, 0.5])
    )
    assert torch.allclose(probabilities, torch.tensor([0.2, 0.8], dtype=torch.float64))


def test_snis_estimate_and_sir_diagnostics_are_honestly_named():
    reference = _matching_reference()
    gen = torch.Generator(device="cpu").manual_seed(7)
    estimate, diagnostics = reference.snis_estimate(
        100, gen, _StandardNormalPotential(), 1.0,
        observable=lambda x: torch.ones(x.shape[0], dtype=x.dtype),
    )
    assert estimate.item() == pytest.approx(1.0)
    assert diagnostics["reference_method"] == "self_normalized_importance_sampling"

    sample, sir_diagnostics = reference.sample_sir(
        25, gen, _StandardNormalPotential(), 1.0,
        oversample=4, return_diagnostics=True,
    )
    assert sample.shape == (25, 1)
    assert sir_diagnostics["reference_method"] == "sampling_importance_resampling"
    assert sir_diagnostics["n_proposals"] == 100
    assert sir_diagnostics["n_resampled"] == 25
    assert 0.0 < sir_diagnostics["unique_resample_fraction"] <= 1.0


def test_historical_exact_name_is_only_a_compatibility_alias():
    reference = _matching_reference()
    gen = torch.Generator(device="cpu").manual_seed(11)
    with pytest.warns(DeprecationWarning, match="not exact"):
        sample = reference.sample_exact_snis(
            10, gen, _StandardNormalPotential(), 1.0, oversample=2
        )
    assert sample.shape == (10, 1)


def test_nonfinite_importance_weights_fail_closed():
    reference = _matching_reference()
    gen = torch.Generator(device="cpu").manual_seed(13)
    with pytest.raises(FloatingPointError, match="nonfinite log weights"):
        reference.snis_weighted_proposals(10, gen, _NonfinitePotential(), 1.0)
