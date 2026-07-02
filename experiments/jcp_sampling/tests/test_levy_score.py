
import torch

from experiments.jcp_sampling.core.jump_banks import double_well_shell, minima_edge_graph
from experiments.jcp_sampling.core.levy_score import stationary_levy_score
from experiments.jcp_sampling.core.potentials import DoubleWell1D, TripleWell1D


def test_levy_score_chunking_agrees():
    pot = DoubleWell1D(beta=4.0)
    bank = double_well_shell(scale=1.0, intensity=0.2)
    x = torch.linspace(-1.5, 1.5, 17).reshape(-1, 1)
    s1 = stationary_levy_score(pot.potential, x, bank, pot.beta, n_theta=8, jump_chunk=8, particle_chunk=None)
    s2 = stationary_levy_score(pot.potential, x, bank, pot.beta, n_theta=8, jump_chunk=1, particle_chunk=5)
    assert torch.allclose(s1, s2, atol=1e-5)
    assert torch.isfinite(s1).all()


def test_levy_score_symmetry_at_origin():
    pot = DoubleWell1D(beta=4.0)
    bank = double_well_shell(scale=1.0, intensity=0.2)
    s = stationary_levy_score(pot.potential, torch.zeros(1, 1), bank, pot.beta, n_theta=12)
    assert abs(float(s.item())) < 1e-5


def test_stationarity_identity_1d():
    """Numerically verify d/dx[S_nu(x) pi(x)] = sum_e rate_e [pi(x-r_e) - pi(x)] on a grid.

    This is the continuous-time stationarity condition the Levy score is designed to satisfy:
    the divergence of the score-drift probability current cancels the jump gain/loss current,
    so pi ~ exp(-beta V) is invariant. Holds for any finite bank; checked here for a double well.
    """
    beta = 1.0
    Vfn = lambda z: 0.25 * (z[..., 0] ** 2 - 1.0) ** 2
    bank = double_well_shell(minima=(-1.0, 1.0), scale=1.0, intensity=0.5).to(dtype=torch.float64)
    x = torch.linspace(-2.0, 2.0, 4001, dtype=torch.float64).reshape(-1, 1)
    S = stationary_levy_score(Vfn, x, bank, beta, n_theta=48, exponent_clip=1e9, score_clip=None)[:, 0]
    xs = x[:, 0]
    pi = torch.exp(-beta * Vfn(x))
    lhs = torch.gradient(S * pi, spacing=(xs,))[0]
    rates = bank.intensity * bank.weights
    rhs = torch.zeros_like(xs)
    for w, r in zip(rates, bank.vectors[:, 0]):
        rhs = rhs + w * (torch.exp(-beta * Vfn((xs - r).reshape(-1, 1))) - pi)
    sl = slice(50, -50)
    rel = (lhs[sl] - rhs[sl]).abs().max() / rhs[sl].abs().max()
    assert float(rel) < 1e-3


def test_stationarity_identity_triple_well_actual_banks():
    """Same continuous-time stationarity check on the manuscript's triple-well adjacent bank.

    Confirms the closed-form Levy score preserves the Gaussian-mixture Gibbs target for the
    actual multi-mode jump support used in the experiments, not just the double well.
    """
    tw = TripleWell1D(eps=0.08)
    beta = tw.beta
    Vfn = lambda z: tw.potential(z)
    bank = minima_edge_graph(tw.minima(), [(0, 1), (1, 2)], intensity=1.0, symmetric=True).to(dtype=torch.float64)
    x = torch.linspace(-5.5, 5.5, 6001, dtype=torch.float64).reshape(-1, 1)
    S = stationary_levy_score(Vfn, x, bank, beta, n_theta=48, exponent_clip=1e9, score_clip=None)[:, 0]
    xs = x[:, 0]
    pi = torch.exp(-beta * Vfn(x))
    lhs = torch.gradient(S * pi, spacing=(xs,))[0]
    rates = bank.intensity * bank.weights
    rhs = torch.zeros_like(xs)
    for w, r in zip(rates, bank.vectors[:, 0]):
        rhs = rhs + w * (torch.exp(-beta * Vfn((xs - r).reshape(-1, 1))) - pi)
    mask = (pi > pi.max() * 1e-6) & (xs > -5.0) & (xs < 5.0)
    rel = (lhs[mask] - rhs[mask]).abs().max() / rhs[mask].abs().max()
    assert float(rel) < 1e-2
