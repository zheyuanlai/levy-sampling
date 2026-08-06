"""Oracle counters and the force-equivalent cost model.

The point of these tests is that cost is *measured*, not assumed. A caching
change inside a sampler must move the recorded counters by itself, and no test
here may reconstruct a cost from ``steps x particles``.
"""
from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from conftest import IsotropicGaussian, make_streams
from src import fee as fee_module
from src.jumps import ShellJumpLaw
from src.potentials import CoupledQuarticChain
from src.samplers import (BAOABSampler, CompoundPoissonSampler, MALASampler,
                          ULASampler, UnboundedBox)
from src.score import DeterministicShellScore, IIDRandomAtomicScore
from src.targets import BASELINE, EXTRA, OracleCounters, Target


# ------------------------------------------------------------- target API
def test_each_oracle_entry_point_updates_its_own_counter(gaussian_target):
    x = torch.zeros(7, 1, dtype=torch.float64)
    gaussian_target.value(x, cost_class=BASELINE)
    gaussian_target.value(x[:3], cost_class=EXTRA)
    gaussian_target.force(x[:5])
    gaussian_target.value_and_force(x[:2])
    raw = gaussian_target.raw_counters()
    assert raw["n_potential_only"] == 10          # 7 baseline + 3 extra
    assert raw["n_potential_baseline"] == 7
    assert raw["n_force_only"] == 5
    assert raw["n_value_and_force"] == 2

    derived = gaussian_target.derived_counters()
    assert derived["n_force"] == 7                # force-only plus joint
    assert derived["n_extra_potential"] == 3      # joint calls are not charged twice


def test_cost_class_must_be_declared_explicitly(gaussian_target):
    with pytest.raises(ValueError, match="cost_class"):
        gaussian_target.value(torch.zeros(1, 1, dtype=torch.float64),
                              cost_class="whatever")


def test_no_count_excludes_metric_and_reference_work(gaussian_target):
    x = torch.zeros(4, 1, dtype=torch.float64)
    gaussian_target.force(x)
    before = gaussian_target.raw_counters()
    with gaussian_target.no_count():
        gaussian_target.force(x)
        gaussian_target.value(x, cost_class=EXTRA)
        gaussian_target.value_and_force(x)
    assert gaussian_target.raw_counters() == before


def test_derivation_rejects_impossible_baseline_counts():
    with pytest.raises(ValueError, match="baseline potential calls exceed"):
        OracleCounters.derive({"n_potential_only": 2, "n_potential_baseline": 5,
                               "n_force_only": 0, "n_value_and_force": 0})


# ------------------------------------------------- counters follow the code
def test_sampler_counters_reflect_actual_calls_not_a_formula(gaussian_target):
    """One force per step for ULA; one joint call per step for MALA."""
    n, steps = 16, 10
    for sampler_class, expected in (
        (ULASampler, {"n_force_only": n * steps, "n_value_and_force": 0}),
        (MALASampler, {"n_force_only": 0, "n_value_and_force": n * steps}),
    ):
        gaussian_target.counters.reset()
        sampler = sampler_class(
            target=gaussian_target, streams=make_streams(seeds=(0,)),
            x0=torch.zeros(n, 1, dtype=torch.float64), n_per_seed=n, dt=0.01,
            tame_cap=None, box=UnboundedBox())
        baseline = gaussian_target.raw_counters()
        for _ in range(steps):
            sampler.step()
        counters = gaussian_target.derived_counters(baseline)
        for key, value in expected.items():
            assert counters[key] == value, (sampler_class.__name__, key)
        assert counters["n_extra_potential"] == 0


def test_uld_force_caching_shows_up_in_the_counters(gaussian_target):
    """BAOAB caches the trailing force, so it costs one force call per step.

    If that caching were removed the counter would double on its own; nothing
    here asserts a hard-coded per-step cost.
    """
    n, steps = 16, 10
    gaussian_target.counters.reset()
    sampler = BAOABSampler(target=gaussian_target,
                           streams=make_streams(seeds=(0,)),
                           x0=torch.zeros(n, 1, dtype=torch.float64),
                           n_per_seed=n, dt=0.01, tame_cap=None,
                           box=UnboundedBox())
    baseline = gaussian_target.raw_counters()
    for _ in range(steps):
        sampler.step()
    counters = gaussian_target.derived_counters(baseline)
    assert counters["n_force"] == n * steps
    assert counters["n_extra_potential"] == 0


# ------------------------------------------------------- Levy score cost
@pytest.fixture
def one_dimensional_law(device):
    atoms = torch.tensor([[2.0], [-2.0]], dtype=torch.float64, device=device)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64, device=device)
    return ShellJumpLaw(atoms, weights, h=0.2)


@pytest.mark.parametrize("bank_size", [1, 4, 8])
def test_random_atomic_extra_potential_matches_A_times_q_theta(
        quartic_target, one_dimensional_law, bank_size):
    """Measured counters against the theoretical ``A Q_theta`` per particle-step."""
    n, steps, q_theta = 32, 5, 12
    score = IIDRandomAtomicScore(quartic_target, one_dimensional_law, 1.0,
                                 bank_size=bank_size, q_theta=q_theta)
    quartic_target.counters.reset()
    sampler = CompoundPoissonSampler(
        target=quartic_target, streams=make_streams(seeds=(0,)),
        x0=torch.zeros(n, 1, dtype=torch.float64), n_per_seed=n, dt=0.005,
        tame_cap=1.0, box=UnboundedBox(), law=one_dimensional_law,
        intensity=1.0, score=score, name="LSC-CP-RA", jump_mode="iid_bank",
        bank_size=bank_size)
    baseline = quartic_target.raw_counters()
    for _ in range(steps):
        sampler.step()
    counters = quartic_target.derived_counters(baseline)
    assert counters["n_extra_potential"] == n * steps * bank_size * q_theta
    assert counters["n_force"] == n * steps
    assert score.extra_potential_per_particle_step == bank_size * q_theta


def test_full_quadrature_extra_potential_matches_J_times_q_theta(
        quartic_target, one_dimensional_law):
    n, steps, q_theta, q_rho = 32, 5, 12, 6
    score = DeterministicShellScore(quartic_target, one_dimensional_law, 1.0,
                                    q_theta=q_theta, q_rho=q_rho)
    n_shifts = one_dimensional_law.n_atoms * q_rho
    assert score.n_shifts == n_shifts
    quartic_target.counters.reset()
    sampler = CompoundPoissonSampler(
        target=quartic_target, streams=make_streams(seeds=(0,)),
        x0=torch.zeros(n, 1, dtype=torch.float64), n_per_seed=n, dt=0.005,
        tame_cap=1.0, box=UnboundedBox(), law=one_dimensional_law,
        intensity=1.0, score=score, name="LSC-CP", jump_mode="full_law")
    baseline = quartic_target.raw_counters()
    for _ in range(steps):
        sampler.step()
    counters = quartic_target.derived_counters(baseline)
    assert counters["n_extra_potential"] == n * steps * n_shifts * q_theta
    assert score.extra_potential_per_particle_step == n_shifts * q_theta


def test_raw_compound_poisson_has_no_extra_potential_cost(quartic_target,
                                                          one_dimensional_law):
    n, steps = 32, 5
    quartic_target.counters.reset()
    sampler = CompoundPoissonSampler(
        target=quartic_target, streams=make_streams(seeds=(0,)),
        x0=torch.zeros(n, 1, dtype=torch.float64), n_per_seed=n, dt=0.005,
        tame_cap=1.0, box=UnboundedBox(), law=one_dimensional_law,
        intensity=1.0, score=None, name="Raw-CP", jump_mode="full_law")
    baseline = quartic_target.raw_counters()
    for _ in range(steps):
        sampler.step()
    assert quartic_target.derived_counters(baseline)["n_extra_potential"] == 0


# ------------------------------------------- the structured E4 fast path
@pytest.fixture
def chain_target(device):
    return Target(CoupledQuarticChain(), beta=8.0, name="chain", device=device)


def test_structured_chord_kernel_matches_direct_evaluation(chain_target):
    """The moment identity must equal ``V(x - r) - V(x)`` computed directly."""
    potential = chain_target.potential
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(64, potential.d, generator=generator, dtype=torch.float64)
    site_shift = 0.3 * torch.randn(9, 2, generator=generator,
                                   dtype=torch.float64)
    shifts = (site_shift.unsqueeze(1).expand(9, potential.n_sites, 2)
              .reshape(9, potential.d).contiguous())
    structured = potential.value_delta(x, shifts)
    direct = (potential.V(x.unsqueeze(1) - shifts.unsqueeze(0))
              - potential.V(x).unsqueeze(1))
    assert torch.allclose(structured, direct, atol=1e-10, rtol=0)


def test_structured_chords_are_counted_separately_from_generic_potentials(
        chain_target):
    """A structured kernel must never be reported as generic ``V()`` calls."""
    potential = chain_target.potential
    x = torch.zeros(8, potential.d, dtype=torch.float64)
    shifts = torch.zeros(5, potential.d, dtype=torch.float64)
    chain_target.counters.reset()
    chain_target.chord_value_delta(x, shifts)
    raw = chain_target.raw_counters()
    assert raw["n_structured_extra_chord_units"] == 40
    assert raw["n_structured_extra_particle_calls"] == 8
    assert raw["n_potential_only"] == 0


def test_structured_kernel_is_charged_a_measured_equivalent_cost(chain_target):
    calibration = fee_module.calibrate(chain_target, particle_batch_size=256,
                                       chord_counts=(8, 32, 64), warmup=3,
                                       repetitions=8)
    assert calibration.structured_seconds_per_particle_chord is not None
    counters = {"n_force": 100, "n_extra_potential": 0,
                "n_structured_extra_chord_units": 1000,
                "n_structured_extra_particle_calls": 100}
    equivalent = calibration.extra_potential_equivalent(counters)
    # Charged through the measured kernel, not as 1000 generic evaluations.
    assert equivalent > 0.0
    assert equivalent != 1000.0
    expected_seconds = (calibration.structured_fixed_seconds_per_particle * 100
                        + calibration.structured_seconds_per_particle_chord * 1000)
    assert equivalent == pytest.approx(expected_seconds / calibration.C_V,
                                       rel=1e-12)
    # N_FEE = N_F + rho * N_V_eq = N_F + C_structured_total / C_F.
    assert calibration.fee(counters) == pytest.approx(
        100 + expected_seconds / calibration.C_F, rel=1e-12)


# ------------------------------------------------------------- FEE units
def test_costs_are_per_configuration_and_batch_size_independent(gaussian_target):
    """``C_V`` and ``C_F`` are amortized per configuration, so the batch cancels."""
    small = fee_module.calibrate(gaussian_target, particle_batch_size=2048,
                                 warmup=5, repetitions=25)
    large = fee_module.calibrate(gaussian_target, particle_batch_size=16384,
                                 warmup=5, repetitions=25)
    assert small.cost_unit == fee_module.COST_UNIT == (
        "amortized_time_per_configuration")
    assert large.cost_unit == fee_module.COST_UNIT
    # Timing noise is real, but the per-configuration cost must not scale with
    # the batch size the way a per-batch time would (a factor of eight here).
    ratio = large.C_V / small.C_V
    assert 0.05 < ratio < 20.0, ratio
    assert small.rho > 0.0 and large.rho > 0.0


def test_fee_is_force_plus_rho_times_equivalent_extra_potential(gaussian_target):
    calibration = fee_module.calibrate(gaussian_target,
                                       particle_batch_size=1024, warmup=3,
                                       repetitions=10)
    counters = {"n_force": 1000, "n_extra_potential": 250}
    assert calibration.fee(counters) == pytest.approx(
        1000 + calibration.rho * 250, rel=1e-12)
    row = calibration.cost_row(counters)
    assert row["n_extra_potential_equivalent"] == 250
    assert row["fee_cost_unit"] == "amortized_time_per_configuration"
    assert row["fee_calibration_hash"] == calibration.hash


def test_calibration_hash_separates_incomparable_workloads(gaussian_target):
    first = fee_module.calibrate(gaussian_target, particle_batch_size=1024,
                                 warmup=3, repetitions=10)
    second = fee_module.calibrate(gaussian_target, particle_batch_size=4096,
                                  warmup=3, repetitions=10)
    assert first.hash != second.hash

def test_calibration_hash_includes_device_index_and_uuid(gaussian_target):
    calibration = fee_module.calibrate(
        gaussian_target, particle_batch_size=128, warmup=1, repetitions=3)
    other_index = replace(calibration, device_index=1)
    other_uuid = replace(calibration, device_uuid="GPU-different")
    assert calibration.hash != other_index.hash
    assert calibration.hash != other_uuid.hash
    assert other_index.identity_payload()["device_index"] == 1
    assert other_uuid.identity_payload()["device_uuid"] == "GPU-different"



def test_mismatched_calibrations_may_not_share_a_fee_axis(gaussian_target):
    first = fee_module.calibrate(gaussian_target, particle_batch_size=1024,
                                 warmup=3, repetitions=10)
    second = fee_module.calibrate(gaussian_target, particle_batch_size=4096,
                                  warmup=3, repetitions=10)
    assert fee_module.assert_comparable([first, first]) == first.hash
    with pytest.raises(ValueError, match="cannot share a FEE axis"):
        fee_module.assert_comparable([first, second])
    record = {"hashes": sorted([first.hash, second.hash]),
              "cost_unit_comparable": True, "workload_comparable": True,
              "merged_axis_label": "verified"}
    assert fee_module.assert_comparable([first, second],
                                        compatibility_record=record) == "verified"
    incomplete = {**record, "workload_comparable": False}
    with pytest.raises(ValueError, match="comparable workload"):
        fee_module.assert_comparable([first, second],
                                     compatibility_record=incomplete)


def test_calibration_never_pollutes_the_run_counters(gaussian_target):
    gaussian_target.counters.reset()
    fee_module.calibrate(gaussian_target, particle_batch_size=512, warmup=2,
                         repetitions=5)
    assert gaussian_target.raw_counters() == OracleCounters().snapshot()
