"""Paired multi-atom random-measure score/jump validation."""
import math

import torch

from tests.conftest import CACHE_DIR  # noqa: F401  (selects the GPU)
from src.certificate import certificate_grid, make_phi_family
from src.experiments import build_e1, make_batched_factory, make_sampler_factory
from src.jumps import JitteredShellJumpLaw, ShellJumpLaw
from src.potentials import DoubleWell1D
from src.samplers import CompoundPoisson, RectBox, tame
from src.score import MultiAtomShellScore, ShellScore

DEV = "cuda"


class _ZeroPotential:
    d = 1

    def V(self, x):
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)

    def grad(self, x):
        return torch.zeros_like(x)


class _RecordingMultiAtomScore(MultiAtomShellScore):
    def score_for_bank(self, x, R):
        self.bank_seen_by_score = R
        return super().score_for_bank(x, R)


def test_paired_sampler_reuses_exact_bank_for_score_and_weighted_jumps():
    """The same radial+jitter bank must determine score and jump displacement.

    Replaying the jump generator lets us reconstruct both the bank and every
    atomwise Poisson count, including unequal weights, and hence the complete
    deterministic step, including the replayed diffusion stream.
    """
    atoms = torch.tensor([[1.5], [-2.0], [0.7]], device=DEV)
    weights = torch.tensor([0.1, 0.6, 0.3], device=DEV)
    law = JitteredShellJumpLaw(atoms, weights, h=torch.tensor([0.2, 0.1, 0.05], device=DEV),
                               jitter_sigma=0.07)
    pot = _ZeroPotential()
    lam, dt, n = 3.0, 0.35, 4096
    score = _RecordingMultiAtomScore(pot, law, lam, beta=1.0,
                                      q_theta=4, m_max=float("inf"))
    x0 = torch.zeros(n, 1, device=DEV)
    gd = torch.Generator(device=DEV); gd.manual_seed(41)
    gj = torch.Generator(device=DEV); gj.manual_seed(73)
    sampler = CompoundPoisson(
        pot, x0, dt, eps=1.0, lam=lam, law=law,
        gen_diff=gd, gen_jump=gj, box=RectBox([-100.0], [100.0], DEV),
        score=score, jump_mode="paired_multiatom",
    )

    replay = torch.Generator(device=DEV); replay.manual_seed(73)
    bank = score.sample_bank(n, replay)
    rates = (lam * dt * law.weights).view(1, -1).expand(n, -1)
    counts = torch.poisson(rates, generator=replay)
    S, _ = score.score_for_bank(x0, bank)
    replay_diff = torch.Generator(device=DEV); replay_diff.manual_seed(41)
    xi = torch.randn(x0.shape, generator=replay_diff, device=DEV,
                     dtype=x0.dtype)
    expected = (x0 + dt * tame(S, dt) + math.sqrt(2.0 * dt) * xi
                + (counts.unsqueeze(-1) * bank).sum(dim=1))

    sampler.step()
    assert torch.equal(score.bank_seen_by_score, bank)
    assert torch.equal(sampler.positions(), expected)
    diag = sampler.pop_diagnostics()
    expected_count = int(counts.sum().item())
    assert diag["jump_count_mean"] == counts.sum(dim=1).mean().item()
    assert diag["jump_count_cumulative"] == expected_count
    assert diag["jump_count_applied_cumulative"] == expected_count
    assert diag["jump_rate_per_particle_time_cumulative"] == (
        expected_count / (n * dt))
    assert diag["state_box_clip_count_cumulative"] == 0
    assert diag["state_box_clip_fraction_cumulative"] == 0.0
    assert torch.allclose(counts.mean(dim=0), rates[0], atol=3e-2, rtol=0.0)
    assert abs(diag["jump_count_mean"] - lam * dt) < 4e-2
    assert diag["m_clip_fraction"] == 0.0
    # Jitter was part of the exact replayed bank rather than being redrawn for
    # jumps (the no-jitter radial support would satisfy this bound exactly).
    radial_only_distance = (bank - atoms.view(1, 3, 1)).abs().squeeze(-1)
    assert bool((radial_only_distance > law.h.view(1, -1)).any())


def test_multiatom_mc_mean_matches_exact_shell_score_without_clipping():
    """E_R[S_{nu_R}(x)] equals the deterministic full-shell score."""
    pot = DoubleWell1D()
    law = ShellJumpLaw(torch.tensor([[2.0], [-2.0]], device=DEV),
                       torch.tensor([0.35, 0.65], device=DEV),
                       torch.tensor([0.2, 0.1], device=DEV))
    lam, beta, q_theta = 0.8, 1.0, 20
    exact = ShellScore(pot, law, lam, beta, q_theta=q_theta, q_rho=32,
                       m_max=float("inf"))
    ma = MultiAtomShellScore(pot, law, lam, beta, q_theta=q_theta,
                             m_max=float("inf"))

    per_x = 50_000
    x_unique = torch.tensor([[-0.45], [0.20]], device=DEV)
    x = x_unique.repeat_interleave(per_x, dim=0)
    gen = torch.Generator(device=DEV); gen.manual_seed(123)
    bank = ma.sample_bank(x.shape[0], gen)
    estimate, diag = ma.score_for_bank(x, bank)
    estimate = estimate.reshape(2, per_x, 1).mean(dim=1)
    truth, _ = exact(x_unique)
    rel = ((estimate - truth).norm(dim=1) / truth.norm(dim=1)).max().item()
    assert rel < 8e-3, (estimate, truth, rel)
    assert diag["m_clip_fraction"].item() == 0.0


def test_frozen_multiatom_bank_has_conditional_weak_stationarity():
    """For every fixed bank, its weighted score cancels its weighted jumps."""
    pot = DoubleWell1D()
    weights = torch.tensor([0.25, 0.75], device=DEV)
    law = ShellJumpLaw(torch.tensor([[2.0], [-2.0]], device=DEV), weights, h=0.2)
    lam, beta = 0.9, 2.0
    ma = MultiAtomShellScore(pot, law, lam, beta, q_theta=32,
                             m_max=float("inf"))
    frozen = torch.tensor([[1.91], [-2.07]], device=DEV)

    class _FrozenScore:
        def log_parts(self, x):
            return ma.log_parts_for_bank(x, frozen)

    # Pin that score_for_bank and the certificate-facing log form are the same
    # implementation of the realised random measure.
    x = torch.tensor([[-1.2], [-0.3], [0.4], [1.1]], device=DEV)
    supplied = frozen.view(1, 2, 1).expand(x.shape[0], -1, -1)
    S, _ = ma.score_for_bank(x, supplied)
    M, v = ma.log_parts_for_bank(x, frozen)
    assert torch.allclose(S, -torch.exp(M).unsqueeze(1) * v,
                          rtol=2e-14, atol=2e-14)

    phis = make_phi_family(1, [0.0], 1.0, DEV, n_phi=4)
    result = certificate_grid(
        pot, _FrozenScore(), frozen, torch.log(weights), lam, beta, phis,
        lo=[-5.2], hi=[5.2], n_panels=100, nodes_per_panel=8,
    )
    assert result["max_residual"] < 1e-8, result


def test_clipping_diagnostic_matches_global_frozen_bank_accumulator():
    pot = _ZeroPotential()
    law = ShellJumpLaw(torch.tensor([[1.0], [-2.0]], device=DEV),
                       torch.tensor([0.25, 0.75], device=DEV), h=0.0)
    score = MultiAtomShellScore(pot, law, lam=10.0, beta=1.0,
                                 q_theta=4, m_max=0.0)
    x = torch.zeros(7, 1, device=DEV)
    frozen = law.atoms.clone()
    bank = frozen.unsqueeze(0).expand(x.shape[0], -1, -1)
    S, diag = score.score_for_bank(x, bank)
    M, v = score.log_parts_for_bank(x, frozen)
    expected = -torch.exp(torch.clamp(M, max=0.0)).unsqueeze(1) * v
    assert torch.allclose(S, expected, rtol=2e-15, atol=2e-15)
    assert diag["m_clip_fraction"].item() == (M > 0.0).double().mean().item() == 1.0
    assert diag["max_log_magnitude"].item() == M.max().item()


def test_multiatom_preserves_float32_device_and_dtype():
    pot = _ZeroPotential()
    law = ShellJumpLaw(torch.tensor([[1.0], [-1.0]], dtype=torch.float32),
                       torch.tensor([0.4, 0.6], dtype=torch.float32), h=0.1)
    gen = torch.Generator(device="cpu"); gen.manual_seed(5)
    score = MultiAtomShellScore(pot, law, lam=0.7, beta=1.0,
                                 q_theta=5, m_max=float("inf"), gen=gen)
    x = torch.zeros(11, 1, dtype=torch.float32)
    bank = score.sample_bank(x.shape[0])
    S, diag = score.score_for_bank(x, bank)
    assert score.theta.dtype == torch.float32
    assert bank.device == x.device and bank.dtype == x.dtype
    assert S.device == x.device and S.dtype == x.dtype
    assert diag["max_log_magnitude"].dtype == torch.float32


def test_paired_constructor_rejects_measure_mismatches():
    pot = _ZeroPotential()
    other_pot = _ZeroPotential()
    law = ShellJumpLaw(torch.tensor([[1.0], [-1.0]], device=DEV),
                       torch.tensor([0.5, 0.5], device=DEV), h=0.0)
    x0 = torch.zeros(2, 1, device=DEV)
    gd = torch.Generator(device=DEV); gd.manual_seed(1)
    gj = torch.Generator(device=DEV); gj.manual_seed(2)
    box = RectBox([-10.0], [10.0], DEV)

    correct = MultiAtomShellScore(pot, law, lam=1.0, beta=2.0, q_theta=2)
    try:
        CompoundPoisson(pot, x0, 0.1, 0.5, 1.0, law, gd, gj, box,
                        score=correct, jump_mode="full")
    except ValueError as exc:
        assert "paired_multiatom" in str(exc)
    else:
        raise AssertionError("multi-atom score was accepted with unpaired jumps")

    wrong_lam = MultiAtomShellScore(pot, law, lam=0.9, beta=2.0, q_theta=2)
    try:
        CompoundPoisson(pot, x0, 0.1, 0.5, 1.0, law, gd, gj, box,
                        score=wrong_lam, jump_mode="paired_multiatom")
    except ValueError as exc:
        assert "lambda" in str(exc)
    else:
        raise AssertionError("mismatched score/jump intensity was accepted")

    wrong_pot = MultiAtomShellScore(other_pot, law, lam=1.0, beta=2.0, q_theta=2)
    try:
        CompoundPoisson(pot, x0, 0.1, 0.5, 1.0, law, gd, gj, box,
                        score=wrong_pot, jump_mode="paired_multiatom")
    except ValueError as exc:
        assert "potential" in str(exc)
    else:
        raise AssertionError("mismatched score/sampler potential was accepted")

    wrong_beta = MultiAtomShellScore(pot, law, lam=1.0, beta=3.0, q_theta=2)
    try:
        CompoundPoisson(pot, x0, 0.1, 0.5, 1.0, law, gd, gj, box,
                        score=wrong_beta, jump_mode="paired_multiatom")
    except ValueError as exc:
        assert "eps" in str(exc)
    else:
        raise AssertionError("mismatched score/diffusion temperature was accepted")


def test_factories_deploy_paired_multiatom_without_changing_single_ra():
    exp = build_e1(device=DEV)
    betas = torch.tensor([exp.cfg.beta], device=DEV)
    factory = make_sampler_factory(exp, exp.cfg.dt, betas, n_particles=16)
    ma = factory("LSC-CP-MA", 0)
    ra = factory("LSC-CP-RA", 0)
    assert ma.jump_mode == "paired_multiatom"
    assert ra.jump_mode == "atomic"
    assert ma.score.law is ma.law

    batched_factory = make_batched_factory(
        exp, exp.cfg.dt, betas, seeds=(0, 1), n_particles=8)
    ma_batched = batched_factory("LSC-CP-MA")
    assert ma_batched.jump_mode == "paired_multiatom"
    assert ma_batched.score.law is ma_batched.law
