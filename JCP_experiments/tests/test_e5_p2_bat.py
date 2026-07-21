"""E5 P2 gate: differentiable BAT transform + analytic log|Jacobian|.

Gate (task S P2):
  * round trips to_bat(to_cartesian(q)) ~ q and to_cartesian(to_bat(x)) ~ x
    (after frame fixing) to 1e-10;
  * autograd d x / d q vs central finite differences to 1e-6;
  * log_jacobian vs numerical log|det| of the internal Jacobian block to 1e-6;
  * torsion-independence: max torsion-derivative of log_jacobian < 1e-10 (printed),
    while bond/angle derivatives match the analytic form.
"""
from __future__ import annotations

import numpy as np
import torch

from tests.conftest import CACHE_DIR  # noqa: F401  (selects the GPU)
from src.e5_alanine.system import build_alanine_system, dihedral as np_dih
from src.e5_alanine.bat import BATTransform

DEV = "cuda"


def _configs(n=32):
    ala = build_alanine_system()
    rng = np.random.default_rng(11)
    p0 = ala.positions_nm
    X = np.stack([p0] + [p0 + s * rng.standard_normal(p0.shape)
                         for s in list(np.linspace(0.005, 0.04, n - 1))])
    return ala, torch.tensor(X.reshape(X.shape[0], -1), device=DEV)


def test_phi_psi_are_bat_torsions():
    ala, x = _configs()
    bat = BATTransform(device=DEV)
    q = bat.to_bat(x)
    Xnp = x.reshape(x.shape[0], 22, 3).cpu().numpy()
    phi_np = np_dih(Xnp, (4, 6, 8, 14))
    psi_np = np_dih(Xnp, (6, 8, 14, 16))

    def _werr(a, b):
        return np.abs(((a - b + np.pi) % (2 * np.pi)) - np.pi).max()

    assert _werr(q[:, bat.phi_slot].cpu().numpy(), phi_np) < 1e-10
    assert _werr(q[:, bat.psi_slot].cpu().numpy(), psi_np) < 1e-10
    assert bat.n_internal == 60
    assert (len(bat.bond_slots), len(bat.angle_slots),
            len(bat.torsion_slots)) == (21, 20, 19)


def test_round_trips():
    ala, x = _configs()
    bat = BATTransform(device=DEV)
    q = bat.to_bat(x)
    # internal -> cart -> internal is exact
    assert (bat.to_bat(bat.to_cartesian(q)) - q).abs().max().item() < 1e-10
    # cart -> internal -> cart is idempotent after frame fixing
    xff = bat.to_cartesian(q)
    assert (bat.to_cartesian(bat.to_bat(xff)) - xff).abs().max().item() < 1e-10


def test_jacobian_vs_finite_differences():
    ala, x = _configs(4)
    bat = BATTransform(device=DEV)
    q = bat.to_bat(x)
    h = 1e-6
    for k in range(q.shape[0]):
        qi = q[k]
        Jauto = torch.autograd.functional.jacobian(bat.to_cartesian, qi)  # (66,60)
        eye = torch.eye(bat.n_internal, device=DEV, dtype=torch.float64)
        Jfd = ((bat.to_cartesian(qi + h * eye) - bat.to_cartesian(qi - h * eye))
               / (2.0 * h)).T                                             # (66,60)
        rel = (Jauto - Jfd).norm() / (Jfd.norm() + 1e-30)
        assert float(rel) < 1e-6, (k, float(rel))


def test_log_jacobian_matches_numeric_logdet():
    ala, x = _configs(6)
    bat = BATTransform(device=DEV)
    q = bat.to_bat(x)
    for k in range(q.shape[0]):
        qi = q[k]
        Jfull = torch.autograd.functional.jacobian(bat.to_cartesian, qi)  # (66,60)
        Jfree = Jfull[bat.free_cart_idx, :]                               # (60,60)
        _, logdet = torch.linalg.slogdet(Jfree)
        analytic = bat.log_jacobian(qi)
        assert abs(float(logdet) - float(analytic)) < 1e-6, (k, float(logdet),
                                                             float(analytic))


def test_torsion_independence_and_analytic_form():
    ala, x = _configs(16)
    bat = BATTransform(device=DEV)
    q = bat.to_bat(x).clone().requires_grad_(True)
    lj = bat.log_jacobian(q).sum()
    (g,) = torch.autograd.grad(lj, q)
    tor_deriv = g[:, bat.torsion_slots_t].abs().max().item()
    print(f"max |d log_jac / d torsion| = {tor_deriv:.3e}")
    assert tor_deriv < 1e-10

    # analytic derivatives: d/d r23 = 1/r23, d/d b = 2/b, d/d a = cot a,
    # d/d r12 = d/d a123 = 0
    qd = q.detach()
    ana = torch.zeros_like(qd)
    ana[:, bat.i_r23] = 1.0 / qd[:, bat.i_r23]
    ana[:, bat.bond_slots_t[2:]] = 2.0 / qd[:, bat.bond_slots_t[2:]]
    ana[:, bat.angle_slots_t[1:]] = 1.0 / torch.tan(qd[:, bat.angle_slots_t[1:]])
    assert (g - ana).abs().max().item() < 1e-9, (g - ana).abs().max().item()
    # r12 and a123 carry no Jacobian weight
    assert g[:, bat.i_r12].abs().max().item() < 1e-12
    assert g[:, bat.i_a123].abs().max().item() < 1e-12
