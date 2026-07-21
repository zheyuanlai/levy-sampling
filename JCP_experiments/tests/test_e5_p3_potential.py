"""E5 P3 gate: whitened internal-coordinate potential U_eff, Jacobian-free chord.

Gate (task S P3):
  * grad vs central finite differences to 1e-6;
  * Jacobian-free check: for a pure-torsion shift r (phi/psi slots) and random
    q_tilde, theta,
        | U_eff(q_tilde - theta r) - U_eff(q_tilde)
          - (U(x(q_tilde - theta r)) - U(x(q_tilde))) | < 1e-10   (max over batch).
    If this is NOT machine-zero the design premise is wrong -- STOP.
"""
from __future__ import annotations

import numpy as np
import torch

from tests.conftest import CACHE_DIR  # noqa: F401  (selects the GPU)
from src.e5_alanine.potential import AlanineDipeptideBAT, e5_beta

DEV = "cuda"


def _pot():
    return AlanineDipeptideBAT(device=DEV)


def _states(pot, n=16, scale=0.05, seed=0):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    qt_ref = pot.q_ref * pot.Dinv
    return qt_ref.unsqueeze(0) + scale * torch.randn(n, pot.d, generator=g,
                                                     device=DEV)


def test_beta_and_whitening():
    pot = _pot()
    # beta = 1/(kB T) at 300 K
    assert abs(pot.beta - e5_beta(300.0)) < 1e-15
    assert abs(1.0 / pot.beta - 2.4943388) < 1e-4
    assert pot.d == 60
    # torsions unwhitened, bonds/angles scaled down (stiff)
    assert bool((pot.D[pot.torsion_slots_t] == 1.0).all())
    assert float(pot.D[pot.bond_slots_t].max()) < 0.02
    assert float(pot.D[pot.angle_slots_t].max()) < 0.2


def test_grad_matches_finite_differences():
    pot = _pot()
    qt = _states(pot, n=6, seed=1)
    g = pot.grad(qt)
    h = 1e-6
    eye = torch.eye(pot.d, device=DEV, dtype=torch.float64)
    for k in range(qt.shape[0]):
        with pot.no_count():
            fd = (pot._V_raw(qt[k].unsqueeze(0) + h * eye)
                  - pot._V_raw(qt[k].unsqueeze(0) - h * eye)) / (2.0 * h)
        rel = (g[k] - fd).norm() / (fd.norm() + 1e-30)
        assert float(rel) < 1e-6, (k, float(rel))


def test_jacobian_free_pure_torsion_chord():
    pot = _pot()
    qt = _states(pot, n=64, seed=2)
    g = torch.Generator(device=DEV)
    g.manual_seed(3)
    theta = torch.rand(qt.shape[0], generator=g, device=DEV)

    def _resid(r):
        lhs = pot._V_raw(qt - theta.unsqueeze(1) * r) - pot._V_raw(qt)
        rhs = (pot._U_cart_from_qt(qt - theta.unsqueeze(1) * r)
               - pot._U_cart_from_qt(qt))
        return (lhs - rhs).abs().max().item()

    # phi/psi-only shift
    r = torch.zeros(pot.d, device=DEV)
    r[pot.phi_slot], r[pot.psi_slot] = 0.7, -0.5
    res_phipsi = _resid(r)
    # all-torsion shift
    r2 = torch.zeros(pot.d, device=DEV)
    r2[pot.torsion_slots_t] = torch.linspace(-0.4, 0.4, pot.torsion_slots_t.numel(),
                                             device=DEV)
    res_all = _resid(r2)
    print(f"jacobian-free residual: phi/psi={res_phipsi:.2e} all-torsions={res_all:.2e}")
    assert res_phipsi < 1e-10, res_phipsi
    assert res_all < 1e-10, res_all


def test_to_cv_and_counters():
    pot = _pot()
    qt = _states(pot, n=8, seed=4)
    cv = pot.to_cv(qt)
    assert cv.shape == (8, 2)
    assert bool(((cv > -np.pi - 1e-9) & (cv <= np.pi + 1e-9)).all())
    pot.reset_counters()
    pot.V(qt)
    pot.grad(qt)
    assert pot.n_V == 8 and pot.n_grad == 8
    # base V_delta path increments n_Vdelta and is finite
    R = torch.zeros(3, pot.d, device=DEV)
    R[:, pot.phi_slot] = torch.tensor([0.1, 0.2, 0.3], device=DEV)
    dv = pot.V_delta(qt, R)
    assert dv.shape == (8, 3) and bool(torch.isfinite(dv).all())
    assert pot.n_Vdelta == 8 * 3
