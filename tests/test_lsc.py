"""The Levy score, the iid random-atomic estimator, and the shared bank."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from conftest import make_streams
from src.jumps import (ShellJumpLaw, full_law_jump_increment,
                       iid_bank_jump_increment)
from src.samplers import CompoundPoissonSampler, UnboundedBox
from src.score import (DeterministicShellScore, IIDRandomAtomicScore,
                       shell_score_dense_theta)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def shell_law(device):
    atoms = torch.tensor([[2.0], [-2.0]], dtype=torch.float64, device=device)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64, device=device)
    return ShellJumpLaw(atoms, weights, h=0.2)


@pytest.fixture
def skewed_law(device):
    """Deliberately non-uniform weights: stratification and iid differ here."""
    atoms = torch.tensor([[1.5], [-2.5], [3.0]], dtype=torch.float64,
                         device=device)
    weights = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64, device=device)
    return ShellJumpLaw(atoms, weights, h=0.15)


# --------------------------------------------- deterministic score accuracy
def test_deterministic_score_matches_a_dense_theta_rule(quartic_target,
                                                        shell_law):
    """The Gauss-Legendre theta rule against a dense composite-Simpson one."""
    score = DeterministicShellScore(quartic_target, shell_law, intensity=1.0,
                                    q_theta=16, q_rho=8)
    x = torch.linspace(-1.6, 1.6, 24, dtype=torch.float64).unsqueeze(1)
    value, _ = score(x)
    dense = shell_score_dense_theta(quartic_target, shell_law, 1.0, q_rho=8,
                                    x=x, n_theta=20_001)
    scale = dense.abs().max().clamp(min=1e-12)
    assert float((value - dense).abs().max() / scale) < 1e-8


# ------------------------------------------------- iid sampling of the law
def test_bank_sampling_reproduces_the_mixture_weights(skewed_law):
    """Component frequencies must follow ``w_k``, not one draw per component."""
    streams = make_streams(seeds=(0,))
    bank = skewed_law.sample_bank(streams, "jump_bank_gen", 200_000, 4)
    flat = bank.reshape(-1, skewed_law.d)
    nearest = (flat.unsqueeze(1) - skewed_law.atoms.unsqueeze(0)).norm(
        dim=-1).argmin(dim=1)
    frequencies = torch.bincount(nearest, minlength=skewed_law.n_atoms
                                 ).to(torch.float64) / nearest.numel()
    assert torch.allclose(frequencies, skewed_law.weights, atol=3e-3)


def test_bank_slots_are_independent(skewed_law):
    """Stratification would force one slot per component; iid must not."""
    streams = make_streams(seeds=(0,))
    bank = skewed_law.sample_bank(streams, "jump_bank_gen", 50_000, 3)
    nearest = (bank.unsqueeze(2) - skewed_law.atoms.view(1, 1, -1, 1)).norm(
        dim=-1).argmin(dim=2)
    all_distinct = (nearest[:, 0] != nearest[:, 1]) & (nearest[:, 1] != nearest[:, 2])
    # Under stratification every row would carry three distinct components.
    assert float(all_distinct.to(torch.float64).mean()) < 0.5
    # Under iid sampling with these weights, all three slots landing on the
    # same component has probability sum_k w_k^3 = 0.244.
    all_same = (nearest[:, 0] == nearest[:, 1]) & (nearest[:, 1] == nearest[:, 2])
    expected = float((skewed_law.weights ** 3).sum())
    assert abs(float(all_same.to(torch.float64).mean()) - expected) < 0.01


def test_random_empirical_measure_is_unbiased(skewed_law):
    """``E[nu_hat_A] = nu`` checked on test functions."""
    streams = make_streams(seeds=(0,))
    intensity, bank_size = 1.7, 4
    bank = skewed_law.sample_bank(streams, "jump_bank_gen", 400_000, bank_size)

    def empirical(fn):
        # nu_hat_A(f) = (lambda / A) sum_j f(R_j), averaged over realisations
        return float((intensity / bank_size) * fn(bank).sum(dim=1).mean())

    def exact(fn):
        # nu(f) = lambda * E_{R ~ rho}[f(R)]
        single = skewed_law.sample_bank(make_streams(seeds=(7,)),
                                        "jump_bank_gen", 400_000, 1)
        return float(intensity * fn(single).sum(dim=1).mean())

    for name, fn in (("identity", lambda r: r[..., 0]),
                     ("square", lambda r: r[..., 0] ** 2),
                     ("bounded", lambda r: torch.tanh(r[..., 0]))):
        got, want = empirical(fn), exact(fn)
        assert abs(got - want) < 0.02 * max(abs(want), 1.0), name


# ------------------------------------------- random-atomic score estimator
def _monte_carlo_score(target, law, intensity, x, bank_size, q_theta,
                       n_replicates, seed_offset=0):
    score = IIDRandomAtomicScore(target, law, intensity, bank_size=bank_size,
                                 q_theta=q_theta)
    values = []
    for replicate in range(n_replicates):
        streams = make_streams(seeds=(seed_offset + replicate,))
        bank = law.sample_bank(streams, "jump_bank_gen", x.shape[0], bank_size)
        value, _ = score.score_for_bank(x, bank)
        values.append(value)
    return torch.stack(values)


@pytest.mark.parametrize("bank_size", [1, 4, 8])
def test_random_atomic_mean_approaches_the_full_score(quartic_target,
                                                      shell_law, bank_size):
    """For fixed ``x``, the Monte Carlo mean over banks tracks the full score.

    This is a statement about the RAW estimator, before any taming. It is
    deliberately not asserted for the tamed sampler: taming is a nonlinear map,
    so the mean of a tamed estimator is not the tamed mean.
    """
    x = torch.tensor([[0.4], [-0.9], [1.3]], dtype=torch.float64)
    full = DeterministicShellScore(quartic_target, shell_law, intensity=1.0,
                                   q_theta=24, q_rho=32)
    reference, _ = full(x)
    samples = _monte_carlo_score(quartic_target, shell_law, 1.0, x, bank_size,
                                 q_theta=24, n_replicates=1200)
    estimate = samples.mean(dim=0)
    standard_error = samples.std(dim=0) / math.sqrt(samples.shape[0])
    deviation = (estimate - reference).abs()
    assert torch.all(deviation <= 4.0 * standard_error + 1e-9), (
        f"A={bank_size}: deviation {deviation.tolist()} exceeds four standard "
        f"errors {standard_error.tolist()}")


def test_random_atomic_variance_scales_like_one_over_bank_size(quartic_target,
                                                               shell_law):
    x = torch.tensor([[0.6]], dtype=torch.float64)
    variances = {}
    for bank_size in (1, 4, 8):
        samples = _monte_carlo_score(quartic_target, shell_law, 1.0, x,
                                     bank_size, q_theta=16, n_replicates=3000,
                                     seed_offset=1000)
        variances[bank_size] = float(samples[:, 0, 0].var())
    assert variances[1] > variances[4] > variances[8]
    for bank_size in (4, 8):
        ratio = variances[1] / (bank_size * variances[bank_size])
        assert 0.75 < ratio < 1.35, (
            f"A={bank_size}: variance ratio {ratio:.3f} is not close to 1/A "
            f"scaling")


def test_bank_size_one_is_the_single_atom_estimator(quartic_target, shell_law):
    """``A = 1`` must be the plain LSC-CP-RA estimator with weight lambda."""
    score = IIDRandomAtomicScore(quartic_target, shell_law, 1.3, bank_size=1,
                                 q_theta=16)
    x = torch.tensor([[0.3], [-1.1]], dtype=torch.float64)
    bank = torch.tensor([[[1.9]], [[-2.1]]], dtype=torch.float64)
    value, _ = score.score_for_bank(x, bank)
    # Direct evaluation of -lambda R I(x, R) with the same theta rule.
    theta = score.theta.view(1, -1)
    for i in range(x.shape[0]):
        chord = x[i, 0] - theta * bank[i, 0, 0]
        with quartic_target.no_count():
            delta = (quartic_target.potential.V(chord.unsqueeze(-1))
                     - quartic_target.potential.V(x[i:i + 1]))
        integral = float((torch.exp(-quartic_target.beta * delta)
                          * torch.exp(score.log_theta_weights)).sum())
        expected = -1.3 * bank[i, 0, 0] * integral
        assert abs(float(value[i, 0]) - float(expected)) < 1e-10


# ------------------------------------------------ the shared bank and order
def test_score_and_jump_share_one_bank(quartic_target, shell_law, monkeypatch):
    """The same realised displacements must drive the score and the increment."""
    import src.samplers as samplers_module

    captured = {}
    original_increment = samplers_module.iid_bank_jump_increment

    def recording_increment(bank, streams, n_per_seed, intensity, dt):
        captured["jump_bank"] = bank.clone()
        return original_increment(bank, streams, n_per_seed, intensity, dt)

    monkeypatch.setattr(samplers_module, "iid_bank_jump_increment",
                        recording_increment)

    score = IIDRandomAtomicScore(quartic_target, shell_law, 1.0, bank_size=4,
                                 q_theta=8)
    original_score = score.score_for_bank

    def recording_score(x, bank):
        captured["score_bank"] = bank.clone()
        captured["score_state"] = x.clone()
        return original_score(x, bank)

    score.score_for_bank = recording_score

    n = 32
    x0 = torch.linspace(-1.0, 1.0, n, dtype=torch.float64).unsqueeze(1)
    sampler = CompoundPoissonSampler(
        target=quartic_target, streams=make_streams(seeds=(0,)), x0=x0,
        n_per_seed=n, dt=0.01, tame_cap=1.0, box=UnboundedBox(),
        law=shell_law, intensity=1.0, score=score, name="LSC-CP-RA",
        jump_mode="iid_bank", bank_size=4)
    state_before = sampler.x.clone()
    sampler.step()

    assert torch.equal(captured["score_bank"], captured["jump_bank"])
    # And the score saw the state at the START of the step.
    assert torch.equal(captured["score_state"], state_before)
    diagnostics = sampler.pop_diagnostics()
    assert diagnostics["bank_shared_between_score_and_noise"] is True
    assert diagnostics["bank_refresh_policy"] == "every_step"
    assert diagnostics["score_evaluation"] == "pre_step"
    assert diagnostics["splitting"] == "drift_diffusion_then_jump"


def test_bank_is_refreshed_every_step(quartic_target, shell_law, monkeypatch):
    import src.samplers as samplers_module

    banks = []
    original = samplers_module.iid_bank_jump_increment
    monkeypatch.setattr(
        samplers_module, "iid_bank_jump_increment",
        lambda bank, *args: (banks.append(bank.clone()) or original(bank, *args)))

    score = IIDRandomAtomicScore(quartic_target, shell_law, 1.0, bank_size=2,
                                 q_theta=8)
    n = 16
    sampler = CompoundPoissonSampler(
        target=quartic_target, streams=make_streams(seeds=(0,)),
        x0=torch.zeros(n, 1, dtype=torch.float64), n_per_seed=n, dt=0.01,
        tame_cap=1.0, box=UnboundedBox(), law=shell_law, intensity=1.0,
        score=score, name="LSC-CP-RA", jump_mode="iid_bank", bank_size=2)
    for _ in range(3):
        sampler.step()
    assert len(banks) == 3
    assert not torch.equal(banks[0], banks[1])
    assert not torch.equal(banks[1], banks[2])


def test_jump_is_applied_after_drift_and_diffusion(quartic_target, shell_law,
                                                   monkeypatch):
    """The increment must be added to the drifted state, not folded into it."""
    import src.samplers as samplers_module

    def zero_increment(bank, streams, n_per_seed, intensity, dt):
        counts = torch.zeros(bank.shape[0], bank.shape[1],
                             dtype=bank.dtype, device=bank.device)
        return torch.zeros_like(bank[:, 0, :]), counts

    def constant_increment(bank, streams, n_per_seed, intensity, dt):
        counts = torch.zeros(bank.shape[0], bank.shape[1],
                             dtype=bank.dtype, device=bank.device)
        return torch.full_like(bank[:, 0, :], 5.0), counts

    def run(increment_fn):
        monkeypatch.setattr(samplers_module, "iid_bank_jump_increment",
                            increment_fn)
        score = IIDRandomAtomicScore(quartic_target, shell_law, 1.0,
                                     bank_size=2, q_theta=8)
        n = 16
        sampler = CompoundPoissonSampler(
            target=quartic_target, streams=make_streams(seeds=(0,)),
            x0=torch.zeros(n, 1, dtype=torch.float64), n_per_seed=n, dt=0.01,
            tame_cap=1.0, box=UnboundedBox(), law=shell_law, intensity=1.0,
            score=score, name="LSC-CP-RA", jump_mode="iid_bank", bank_size=2)
        sampler.step()
        return sampler.positions().clone()

    without_jump = run(zero_increment)
    with_jump = run(constant_increment)
    assert torch.allclose(with_jump - without_jump,
                          torch.full_like(with_jump, 5.0), atol=1e-12)


def test_bank_poisson_counts_sum_to_the_full_rate(shell_law):
    """``sum_j N_j`` with ``N_j ~ Pois(lambda dt / A)`` is ``Pois(lambda dt)``."""
    intensity, dt = 3.0, 0.5
    expected = intensity * dt
    for bank_size in (1, 4, 8):
        streams = make_streams(seeds=(0,))
        n = 400_000
        bank = shell_law.sample_bank(streams, "jump_bank_gen", n, bank_size)
        _, counts = iid_bank_jump_increment(bank, streams, n, intensity, dt)
        total = counts.sum(dim=1)
        mean, variance = float(total.mean()), float(total.var())
        # A Poisson variable has equal mean and variance.
        assert abs(mean - expected) < 0.02 * expected
        assert abs(variance - expected) < 0.04 * expected


def test_full_law_jump_reports_cap_exceedances(shell_law):
    streams = make_streams(seeds=(0,))
    n, intensity, dt = 400_000, 1.0, 0.005
    _, applied, sampled = full_law_jump_increment(shell_law, streams, n,
                                                  intensity=intensity, dt=dt)
    assert torch.all(applied <= sampled)
    expected = intensity * dt
    standard_error = math.sqrt(expected / n)
    assert abs(float(sampled.mean()) - expected) < 4.0 * standard_error


# --------------------------------------------------------- wiring guards
def test_iid_score_cannot_be_used_with_full_law_jumps(quartic_target,
                                                      shell_law):
    score = IIDRandomAtomicScore(quartic_target, shell_law, 1.0, bank_size=4,
                                 q_theta=8)
    with pytest.raises(ValueError, match="iid_bank"):
        CompoundPoissonSampler(
            target=quartic_target, streams=make_streams(seeds=(0,)),
            x0=torch.zeros(4, 1, dtype=torch.float64), n_per_seed=4, dt=0.01,
            tame_cap=1.0, box=UnboundedBox(), law=shell_law, intensity=1.0,
            score=score, name="LSC-CP-RA", jump_mode="full_law", bank_size=4)


def test_bank_size_must_agree_between_sampler_and_score(quartic_target,
                                                        shell_law):
    score = IIDRandomAtomicScore(quartic_target, shell_law, 1.0, bank_size=4,
                                 q_theta=8)
    with pytest.raises(ValueError, match="bank size"):
        CompoundPoissonSampler(
            target=quartic_target, streams=make_streams(seeds=(0,)),
            x0=torch.zeros(4, 1, dtype=torch.float64), n_per_seed=4, dt=0.01,
            tame_cap=1.0, box=UnboundedBox(), law=shell_law, intensity=1.0,
            score=score, name="LSC-CP-RA", jump_mode="iid_bank", bank_size=8)


def test_component_stratified_estimator_is_gone():
    """The old LSC-CP-MA must be deleted, not renamed into the RA family."""
    forbidden = ("MultiAtomShellScore", "paired_multiatom", "LSC-CP-MA",
                 "RandomAtomicShellScore", "JitteredShellJumpLaw")
    offenders = []
    for path in sorted((REPOSITORY_ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        offenders.extend(f"{path.name}:{token}" for token in forbidden
                         if token in text)
    assert not offenders, offenders

    import src.score as score_module

    for token in forbidden:
        assert not hasattr(score_module, token)
