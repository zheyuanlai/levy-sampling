"""Weak stationarity residual R(phi) (spec sections 8, 9.2).

R is evaluated with the UNCAPPED log-space score magnitude (the deployed
tamed drift step is identical to the uncapped one up to ~e^{-M_MAX}, and
that saturation defect is asserted separately)."""
import os

import torch

from tests.conftest import CACHE_DIR
from src.config import Q_RHO, Q_THETA
from src.potentials import (DoubleWell1D, MB_CRITICAL, MoG40,
                            MuellerBrownLatent2D, TransformedMuellerBrown10D,
                            muller_brown_2d_grad, newton_refine)
from src.jumps import AnnulusJumpLaw, ShellJumpLaw, gauss_legendre_01
from src.score import MoG40Score, ShellScore
from src.certificate import (certificate_grid, certificate_importance,
                             make_phi_family)

DEV = "cuda"


def _e1_setup():
    pot = DoubleWell1D()
    law = ShellJumpLaw(torch.tensor([[2.0], [-2.0]], device=DEV),
                       torch.tensor([0.5, 0.5], device=DEV), 0.2)
    phis = make_phi_family(1, [0.0], 1.0, DEV)
    shifts, logw = law.quadrature_shifts(Q_RHO)
    score = ShellScore(pot, law, 1.0, 8.0, Q_THETA, Q_RHO)
    return pot, score, shifts, logw, phis


def test_e1_generous_box():
    """Production quadrature orders, box extended a full jump length beyond
    the target's effective support: R < 1e-6 and the M_MAX saturation defect
    on the deployed tamed step is negligible."""
    pot, score, shifts, logw, phis = _e1_setup()
    res = certificate_grid(pot, score, shifts, logw, 1.0, 8.0, phis,
                           [-5.2], [5.2], n_panels=120, nodes_per_panel=8)
    assert res["max_residual"] < 1e-6, res["max_residual"]
    assert res["clip_tamed_step_defect"] < 1e-12
    for i in range(len(phis)):
        assert abs(res[f"phi_{i}"]["jump_term"]) > 1e-4     # phis are informative


def test_e1_tight_box_regression():
    """A deliberately tight box must reproduce a large residual: order-one
    contributions to the identity live where pi is tiny and S is enormous.
    Guards against anyone shrinking the integration domain."""
    pot, score, shifts, logw, phis = _e1_setup()
    res = certificate_grid(pot, score, shifts, logw, 1.0, 8.0, phis,
                           [-1.3], [1.3], n_panels=60, nodes_per_panel=8)
    assert res["max_residual"] > 1e-2, res["max_residual"]


def test_e2_mog40():
    pot = MoG40(device=DEV)
    law = AnnulusJumpLaw(4.0, 15.0, DEV)
    score = MoG40Score(pot.mu, 4.0, 15.0, 1.0, m_phi=32)
    phis = make_phi_family(2, [0.0, 0.0], 30.0, DEV)
    shifts, logw = law.quadrature_shifts(8, 64)   # fine continuous-nu J side
    res = certificate_grid(pot, score, shifts, logw, 1.0, 8.0, phis,
                           [-60.0, -60.0], [60.0, 60.0],
                           n_panels=120, nodes_per_panel=6, chunk=8192)
    assert res["max_residual"] < 1e-6, res["max_residual"]


def test_e3_reduced_latent():
    """E3 reduces exactly to latent 2D: jumps and test functions act on
    z_{1:2} only, dot products are affine-invariant, the aux Gaussian
    factorises. Per-atom shell half-widths carry the x-space h into z."""
    potr = MuellerBrownLatent2D(s=40.0)
    pot10 = TransformedMuellerBrown10D(device=DEV)
    zs = {k: newton_refine(muller_brown_2d_grad,
                           torch.tensor(MB_CRITICAL[k][0], device=DEV))
          for k in ("min_A", "min_B", "min_C")}
    zA, zB, zC = zs["min_A"], zs["min_B"], zs["min_C"]
    dz = torch.stack([zB - zC, zC - zB, zC - zA, zA - zC])
    atoms_x = pot10.from_latent(torch.cat([dz, torch.zeros(4, 8, device=DEV)], 1))
    h_x = 0.1 * atoms_x.norm(dim=1).min().item()
    h_z = h_x * dz.norm(dim=1) / atoms_x.norm(dim=1)
    law = ShellJumpLaw(dz, torch.full((4,), 0.25, device=DEV), h_z)
    score = ShellScore(potr, law, 1.0, 8.0, Q_THETA, Q_RHO)
    shifts, logw = law.quadrature_shifts(Q_RHO)
    phis = make_phi_family(2, [0.0, 0.8], 0.8, DEV)
    res = certificate_grid(potr, score, shifts, logw, 1.0, 8.0, phis,
                           [-4.2, -2.7], [4.2, 4.7],
                           n_panels=130, nodes_per_panel=8, chunk=8192)
    assert res["max_residual"] < 1e-6, res["max_residual"]
    assert res["clip_tamed_step_defect"] < 1e-12


def test_e4_importance():
    """24D: shifted-form residual by self-normalised importance sampling from
    the Laplace mixture; exactly equivalent to the deployed quadrature score
    provided the M_MAX cap never fires on the sampled region (asserted)."""
    from src.experiments import build_e4
    exp = build_e4(device=DEV, basin_cache=os.path.join(CACHE_DIR, "phi4_basins.npz"))
    theta, w_theta = gauss_legendre_01(Q_THETA, DEV)
    shifts, logw = exp.law.quadrature_shifts(Q_RHO)
    phis = make_phi_family(24, exp.extras["means24"][0].tolist(), 1.5, DEV, n_phi=4)
    res = certificate_importance(exp.pot, shifts, logw, theta, w_theta,
                                 1.0, 8.0, phis, exp.extras["laplace"],
                                 n_samples=200_000)
    assert res["max_residual"] < 1e-6, res["max_residual"]
    gen = torch.Generator(device=DEV)
    gen.manual_seed(11)
    xs = exp.extras["laplace"].sample(100_000, gen)
    score = exp.make_score()
    M, _ = score.log_parts(xs)
    assert M.max().item() < 600.0        # cap never fires where pi has mass
