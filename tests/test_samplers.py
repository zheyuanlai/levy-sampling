"""Sampler correctness, with taming treated as a first-class variant."""
from __future__ import annotations

import math

import pytest
import torch

from conftest import make_streams, tight_box
from src.samplers import (BAOABSampler, CompoundPoissonSampler, FLASampler,
                          MALASampler, ParallelTemperingSampler, ULASampler,
                          geometric_ladder, tamed_drift)


# ------------------------------------------------------------ tamed drift
def test_canonical_taming_is_the_identity():
    """``cap is None`` must return the drift bit-for-bit, not merely close."""
    b = torch.randn(64, 3, dtype=torch.float64) * 1e3
    assert torch.equal(tamed_drift(b, 0.01, None), b)


def test_tamed_drift_bounds_the_displacement():
    b = torch.randn(256, 4, dtype=torch.float64) * 1e6
    dt, cap = 0.01, 1.0
    displacement = (dt * tamed_drift(b, dt, cap)).norm(dim=-1)
    assert torch.all(displacement <= cap + 1e-12)


def test_tamed_drift_survives_an_overflowing_norm():
    """A plain ``b.norm()`` would overflow to infinity and zero the drift."""
    b = torch.full((4, 2), 1e300, dtype=torch.float64)
    tamed = tamed_drift(b, 0.1, 1.0)
    assert torch.isfinite(tamed).all()
    assert torch.all(tamed.norm(dim=-1) > 0)


# -------------------------------------------------------------- tamed MALA
def _log_gaussian_density(y, mean, variance):
    d = y.shape[-1]
    return (-0.5 * ((y - mean) ** 2).sum(-1) / variance
            - 0.5 * d * math.log(2.0 * math.pi * variance))


@pytest.mark.parametrize("tame_cap", [None, 1.0])
def test_mala_proposal_log_density_difference_matches_explicit_gaussian(
        quartic_target, unbounded_box, tame_cap):
    """``log q_c(x|y) - log q_c(y|x)`` against an explicit Gaussian log-pdf."""
    streams = make_streams(seeds=(0,))
    x0 = torch.linspace(-1.5, 1.5, 32, dtype=torch.float64).unsqueeze(1)
    sampler = MALASampler(target=quartic_target, streams=streams, x0=x0,
                          n_per_seed=32, dt=0.01, tame_cap=tame_cap,
                          box=unbounded_box)
    x = sampler.x
    y = x + 0.3 * torch.randn(x.shape, generator=torch.Generator().manual_seed(7),
                              dtype=torch.float64)
    drift_x = tamed_drift(quartic_target.force(x), sampler.dt, tame_cap)
    drift_y = tamed_drift(quartic_target.force(y), sampler.dt, tame_cap)
    variance = sampler.proposal_variance

    explicit = (_log_gaussian_density(x, y + sampler.dt * drift_y, variance)
                - _log_gaussian_density(y, x + sampler.dt * drift_x, variance))
    computed = sampler._log_proposal_density_difference(x, y, drift_x, drift_y)
    assert torch.allclose(computed, explicit, atol=1e-12, rtol=0)


@pytest.mark.parametrize("tame_cap", [None, 1.0])
def test_mala_satisfies_detailed_balance_algebraically(quartic_target,
                                                       unbounded_box, tame_cap):
    """``pi(x) q(y|x) alpha(x,y) == pi(y) q(x|y) alpha(y,x)`` for the actual ratio.

    The reverse drift must be recomputed at the proposal point for this to
    hold; using the forward drift in both directions breaks it.
    """
    streams = make_streams(seeds=(0,))
    x = torch.linspace(-1.6, 1.6, 48, dtype=torch.float64).unsqueeze(1)
    sampler = MALASampler(target=quartic_target, streams=streams, x0=x,
                          n_per_seed=48, dt=0.02, tame_cap=tame_cap,
                          box=unbounded_box)
    y = x + 0.4 * torch.randn(x.shape, generator=torch.Generator().manual_seed(11),
                              dtype=torch.float64)
    beta = quartic_target.beta
    variance = sampler.proposal_variance
    drift_x = tamed_drift(quartic_target.force(x), sampler.dt, tame_cap)
    drift_y = tamed_drift(quartic_target.force(y), sampler.dt, tame_cap)

    with quartic_target.no_count():
        Vx = quartic_target.potential.V(x)
        Vy = quartic_target.potential.V(y)
    log_alpha_forward = (-beta * (Vy - Vx)
                         + sampler._log_proposal_density_difference(
                             x, y, drift_x, drift_y))
    log_alpha_reverse = (-beta * (Vx - Vy)
                         + sampler._log_proposal_density_difference(
                             y, x, drift_y, drift_x))
    log_q_forward = _log_gaussian_density(y, x + sampler.dt * drift_x, variance)
    log_q_reverse = _log_gaussian_density(x, y + sampler.dt * drift_y, variance)

    left = (-beta * Vx + log_q_forward
            + torch.clamp(log_alpha_forward, max=0.0))
    right = (-beta * Vy + log_q_reverse
             + torch.clamp(log_alpha_reverse, max=0.0))
    assert torch.allclose(left, right, atol=1e-11, rtol=0)


@pytest.mark.parametrize("tame_cap", [None, 1.0])
def test_mala_reproduces_gaussian_moments(gaussian_target, unbounded_box,
                                          tame_cap):
    """Both variants must sample ``N(0, sigma^2/beta)`` to within Monte Carlo error."""
    streams = make_streams(seeds=(0, 1))
    n_per_seed = 4000
    x0 = torch.zeros(2 * n_per_seed, 1, dtype=torch.float64)
    sampler = MALASampler(target=gaussian_target, streams=streams, x0=x0,
                          n_per_seed=n_per_seed, dt=0.05, tame_cap=tame_cap,
                          box=unbounded_box)
    for _ in range(400):
        sampler.step()
    x = sampler.positions()[:, 0]
    expected_std = gaussian_target.potential.stationary_std(gaussian_target.beta)
    assert abs(float(x.mean())) < 0.03
    assert abs(float(x.std()) - expected_std) < 0.03
    acceptance = sampler.pop_diagnostics()["mh_accept_fraction_cumulative"]
    assert 0.5 < acceptance < 1.0


def test_untamed_mala_matches_a_reference_untamed_step(quartic_target,
                                                       unbounded_box):
    """With ``tame=false`` the code path must be the plain MALA step exactly."""
    n = 64
    x0 = torch.linspace(-1.2, 1.2, n, dtype=torch.float64).unsqueeze(1)
    sampler = MALASampler(target=quartic_target, streams=make_streams(seeds=(3,)),
                          x0=x0, n_per_seed=n, dt=0.01, tame_cap=None,
                          box=unbounded_box)

    # An independent, deliberately naive implementation of the same step.
    reference_streams = make_streams(seeds=(3,))
    x = x0.clone()
    beta = quartic_target.beta
    dt = 0.01
    variance = 2.0 * dt / beta
    with quartic_target.no_count():
        Vx = quartic_target.potential.V(x)
        for _ in range(25):
            drift_x = -quartic_target.potential.grad_V(x)
            xi = reference_streams.randn("diffusion_gen", (n, 1))
            y = x + dt * drift_x + math.sqrt(variance) * xi
            Vy = quartic_target.potential.V(y)
            drift_y = -quartic_target.potential.grad_V(y)
            forward = ((y - (x + dt * drift_x)) ** 2).sum(-1)
            reverse = ((x - (y + dt * drift_y)) ** 2).sum(-1)
            log_alpha = (-beta * (Vy - Vx) + (forward - reverse) / (2.0 * variance))
            u = reference_streams.rand("mh_uniform_gen", (n,))
            accept = torch.log(u) < log_alpha
            x = torch.where(accept.unsqueeze(-1), y, x)
            Vx = torch.where(accept, Vy, Vx)

    for _ in range(25):
        sampler.step()
    assert torch.equal(sampler.positions(), x)


# ---------------------------------------------------------------- ULA / ULD
@pytest.mark.parametrize("tame_cap", [None, 1.0])
def test_uld_reaches_gaussian_equilibrium(gaussian_target, unbounded_box,
                                          tame_cap):
    streams = make_streams(seeds=(0, 1))
    n_per_seed = 4000
    x0 = torch.zeros(2 * n_per_seed, 1, dtype=torch.float64)
    sampler = BAOABSampler(target=gaussian_target, streams=streams, x0=x0,
                           n_per_seed=n_per_seed, dt=0.05, tame_cap=tame_cap,
                           box=unbounded_box, gamma=1.0)
    for _ in range(3000):
        sampler.step()
    x = sampler.positions()[:, 0]
    expected_std = gaussian_target.potential.stationary_std(gaussian_target.beta)
    assert abs(float(x.mean())) < 0.05
    # BAOAB carries O(dt^2) configurational bias; this tolerance admits it.
    assert abs(float(x.std()) - expected_std) < 0.05


def test_ula_reaches_gaussian_equilibrium(gaussian_target, unbounded_box):
    streams = make_streams(seeds=(0,))
    n = 8000
    sampler = ULASampler(target=gaussian_target, streams=streams,
                         x0=torch.zeros(n, 1, dtype=torch.float64),
                         n_per_seed=n, dt=0.01, tame_cap=None,
                         box=unbounded_box)
    for _ in range(3000):
        sampler.step()
    x = sampler.positions()[:, 0]
    expected_std = gaussian_target.potential.stationary_std(gaussian_target.beta)
    assert abs(float(x.mean())) < 0.05
    assert abs(float(x.std()) - expected_std) < 0.05


# --------------------------------------------------------------------- PT
def _pt_sampler(target, box, tame_cap, n_per_seed=3000, dt=0.05, betas=None):
    streams = make_streams(seeds=(0,))
    betas = (geometric_ladder(target.beta, 0.5, 4, target.device)
             if betas is None else betas)
    x0 = torch.zeros(n_per_seed, target.d, dtype=torch.float64)
    return ParallelTemperingSampler(
        target=target, streams=streams, x0=x0, n_per_seed=n_per_seed, dt=dt,
        tame_cap=tame_cap, box=box, betas=betas, n_swap=10)


@pytest.mark.parametrize("tame_cap", [None, 1.0])
def test_pt_every_replica_hits_its_own_stationary_moments(gaussian_target,
                                                          unbounded_box,
                                                          tame_cap):
    """Replica ``k`` must equilibrate to ``N(0, sigma^2/beta_k)``, not to the cold one."""
    sampler = _pt_sampler(gaussian_target, unbounded_box, tame_cap)
    for _ in range(600):
        sampler.step()
    sigma = gaussian_target.potential.sigma
    for k in range(sampler.n_replicas):
        expected = sigma / math.sqrt(float(sampler.betas[k]))
        observed = float(sampler.x[k, :, 0].std())
        assert abs(observed - expected) < 0.08 * expected + 0.02, (
            f"replica {k}: expected std {expected:.4f}, got {observed:.4f}")


def test_pt_swap_formula_does_not_depend_on_the_tame_flag(gaussian_target,
                                                          unbounded_box):
    """Taming changes the local kernel only; the swap acceptance is unchanged."""
    betas = geometric_ladder(gaussian_target.beta, 0.5, 4, gaussian_target.device)
    canonical = _pt_sampler(gaussian_target, unbounded_box, None, n_per_seed=256,
                            betas=betas)
    tamed = _pt_sampler(gaussian_target, unbounded_box, 1.0, n_per_seed=256,
                        betas=betas)
    # Put both samplers in the identical state, then compare one swap pass.
    state = torch.randn(canonical.x.shape, generator=torch.Generator().manual_seed(5),
                        dtype=torch.float64)
    for sampler in (canonical, tamed):
        sampler.x = state.clone()
        with gaussian_target.no_count():
            sampler.Vx = gaussian_target.potential.V(sampler.x)
        sampler._swap_pass(0)
    assert torch.equal(canonical.x, tamed.x)


def test_pt_cold_replica_matches_the_target(gaussian_target, unbounded_box):
    sampler = _pt_sampler(gaussian_target, unbounded_box, 1.0, n_per_seed=4000)
    for _ in range(800):
        sampler.step()
    cold = sampler.positions()[:, 0]
    expected = gaussian_target.potential.stationary_std(gaussian_target.beta)
    assert abs(float(cold.mean())) < 0.04
    assert abs(float(cold.std()) - expected) < 0.04


# ------------------------------------------------------------- stable noise
def test_symmetric_alpha_stable_noise_is_heavy_tailed_and_symmetric():
    streams = make_streams(seeds=(0,))
    for alpha in (1.2, 1.6, 1.9):
        draws = streams.symmetric_alpha_stable("stable_noise_gen",
                                               (200_000, 1), alpha)[:, 0]
        assert abs(float(draws.median())) < 0.05
        # A stable law with alpha < 2 has infinite variance; the empirical
        # kurtosis of a Gaussian sample of this size never reaches this.
        standardized = draws / draws.abs().median()
        assert float(standardized.abs().max()) > 50.0
        positive = float((draws > 0).to(torch.float64).mean())
        assert abs(positive - 0.5) < 0.01


def test_stable_noise_is_not_truncated():
    """A truncated stable law is not stable, so no clipping may be applied."""
    streams = make_streams(seeds=(1,))
    draws = streams.symmetric_alpha_stable("stable_noise_gen", (500_000, 1), 1.3)
    magnitudes = draws.abs().flatten().sort(descending=True).values
    # The top order statistics of a genuine stable sample spread over decades;
    # a clipped sample would pile them onto one value.
    assert float(magnitudes[0] / magnitudes[100]) > 5.0


# ------------------------------------------------------------ boundary rule
def test_out_of_box_proposals_are_rejected_not_clipped(quartic_target):
    """Every method uses one reject rule; nothing lands exactly on the wall."""
    box = tight_box(quartic_target.device, 0.5, 1)
    n = 512
    x0 = torch.zeros(n, 1, dtype=torch.float64)
    for sampler in (
        ULASampler(target=quartic_target, streams=make_streams(seeds=(0,)),
                   x0=x0, n_per_seed=n, dt=0.05, tame_cap=None, box=box),
        FLASampler(target=quartic_target, streams=make_streams(seeds=(0,)),
                   x0=x0, n_per_seed=n, dt=0.05, tame_cap=1.0, box=box,
                   alpha=1.5),
        BAOABSampler(target=quartic_target, streams=make_streams(seeds=(0,)),
                     x0=x0, n_per_seed=n, dt=0.05, tame_cap=None, box=box),
    ):
        for _ in range(200):
            sampler.step()
        x = sampler.positions()
        assert bool(box.contains(x).all()), f"{sampler.name} left the box"
        on_wall = (x.abs() - 0.5).abs() < 1e-15
        assert not bool(on_wall.any()), (
            f"{sampler.name} produced a state exactly on the boundary, which is "
            "the signature of clipping rather than rejection")
        diagnostics = sampler.pop_diagnostics()
        assert diagnostics["boundary_rule"] == "reject"
        assert diagnostics["boundary_reject_count_cumulative"] > 0


def test_mala_rejects_out_of_box_proposals_without_clipping(quartic_target):
    box = tight_box(quartic_target.device, 0.5, 1)
    n = 512
    sampler = MALASampler(target=quartic_target,
                          streams=make_streams(seeds=(0,)),
                          x0=torch.zeros(n, 1, dtype=torch.float64),
                          n_per_seed=n, dt=0.05, tame_cap=None, box=box)
    for _ in range(200):
        sampler.step()
    x = sampler.positions()
    assert bool(box.contains(x).all())
    assert not bool(((x.abs() - 0.5).abs() < 1e-15).any())
    diagnostics = sampler.pop_diagnostics()
    assert diagnostics["boundary_reject_count_cumulative"] > 0
