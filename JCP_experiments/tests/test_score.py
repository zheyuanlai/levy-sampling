"""Score validation (spec section 9.1)."""
import math
import os

import numpy as np
import torch

from tests.conftest import CACHE_DIR
from src.potentials import CoupledPhi4, DoubleWell1D, MoG40
from src.jumps import ShellJumpLaw
from src.score import (MoG40Score, ShellScore, log_bracket, mog40_score_brute,
                       shell_score_brute_theta)

DEV = "cuda"


def test_mog40_closed_form_vs_brute():
    """Closed form vs 3-D brute quadrature (M_phi=512, Q_rho=200, Q_theta=200)
    at matched directions: validates the analytic rho/theta treatment
    (erf/erfcx branches) to < 1e-8 relative. Run once, cached."""
    pot = MoG40(device=DEV)
    xtest = torch.stack([
        pot.mu[0] + 0.3,                    # near a mode
        0.5 * (pot.mu[0] + pot.mu[1]),      # between modes
        pot.mu.mean(0),                     # mid-cloud
        pot.mu[7] + 1.0,
    ])
    cache = os.path.join(CACHE_DIR, "mog40_brute.npz")
    if os.path.exists(cache):
        S_brute = torch.as_tensor(np.load(cache)["S"], device=DEV)
    else:
        S_brute = mog40_score_brute(xtest, pot.mu, 4.0, 15.0, 1.0, 512, 200, 200)
        np.savez(cache, S=S_brute.cpu().numpy())
    closed = MoG40Score(pot.mu, 4.0, 15.0, 1.0, m_phi=512)
    S_cf, _ = closed(xtest)
    rel = ((S_cf - S_brute).norm(dim=1) / S_brute.norm(dim=1)).max().item()
    assert rel < 1e-8, rel


def test_generic_shell_vs_dense_theta():
    """Shell score vs a dense composite-Simpson theta integral (200k nodes),
    same rho quadrature. At Q_theta=48 the GL rule is spectrally converged
    even at boundary-layer points, so this isolates the log-space
    implementation to < 1e-10 relative. (Quadrature-ORDER adequacy at the
    production Q_theta is settled separately by the refinement study and the
    stationarity certificate.)"""
    pot = DoubleWell1D()
    law = ShellJumpLaw(torch.tensor([[2.0], [-2.0]], device=DEV),
                       torch.tensor([0.5, 0.5], device=DEV), 0.2)
    xs = torch.tensor([[-1.0], [-0.5], [0.7], [1.3]], device=DEV)
    S_dense = shell_score_brute_theta(pot, law, 1.0, 8.0, 8, xs)
    sc = ShellScore(pot, law, 1.0, 8.0, 48, 8)
    S, _ = sc(xs)
    rel = ((S - S_dense).norm(dim=1) / S_dense.norm(dim=1)).max().item()
    assert rel < 1e-10, rel


def test_log_bracket_vs_mpmath():
    """Branched log-bracket vs mpmath at 3000 digits, including the extreme
    tails where the naive form has 100% relative error (m=30) or underflows
    entirely (|m|=90)."""
    import mpmath as mp
    mp.mp.dps = 3000
    a, b = 4.0, 15.0

    def bracket_mp(m):
        F = lambda z: z * mp.erf(z / mp.sqrt(2)) + mp.sqrt(2 / mp.pi) * mp.e ** (-z * z / 2)
        return F(b - m) - F(a - m) + (b - a) * mp.erf(m / mp.sqrt(2))

    ms = [-90.0, -16.0, 0.0, 9.0, 15.0, 20.0, 30.0, 90.0]
    lb = log_bracket(torch.tensor(ms, device=DEV), a, b).tolist()
    for m, l in zip(ms, lb):
        ref = float(mp.log(bracket_mp(m)))
        assert abs(l - ref) < 1e-9, (m, l, ref)
    # regression: the naive (unbranched) form is catastrophically wrong at
    # m = 30 -- verify the branched form is NOT equal to the naive one there
    m30 = torch.tensor([30.0], device=DEV)
    F = lambda z: z * torch.erf(z / math.sqrt(2)) + math.sqrt(2 / math.pi) * torch.exp(-0.5 * z * z)
    naive = F(b - m30) - F(a - m30) + (b - a) * torch.erf(m30 / math.sqrt(2))
    truth = float(mp.log(bracket_mp(30.0)))
    assert naive.item() <= 0.0 or abs(math.log(max(naive.item(), 1e-300)) - truth) > 1.0


def test_phi4_moment_delta_vs_lattice():
    """Moment-based homogeneous energy delta vs direct full-lattice
    V(q-r) - V(q): < 1e-13 absolute (spec 4.4)."""
    pot = CoupledPhi4()
    gen = torch.Generator(device=DEV)
    gen.manual_seed(7)
    # production-like states: near coherent minima with fluctuation
    base = torch.tensor([-1.0, -1.0], device=DEV).repeat(pot.Ns)
    x = base + 0.3 * torch.randn(128, 24, generator=gen, device=DEV)
    # production-like shifts: phase-to-phase differences with shell jitter
    D = torch.tensor([[2.0, 2.0], [2.0, 0.0], [0.0, 2.0], [-2.0, 2.0]],
                     device=DEV) + 0.05 * torch.randn(4, 2, generator=gen, device=DEV)
    R = D.unsqueeze(1).expand(-1, pot.Ns, 2).reshape(-1, 24)
    dv_m = pot.V_delta_homogeneous(x, D)
    dv_d = pot._V_raw(x.unsqueeze(1) - R.unsqueeze(0)) - pot._V_raw(x).unsqueeze(1)
    assert (dv_m - dv_d).abs().max().item() < 1e-13
