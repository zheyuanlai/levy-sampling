"""Generous-box certificate pre-flight gate for run_production.sh.

Builds the experiment and computes the weak-stationarity residual max R on the
generous domain; exits 0 iff max R < TOL (production may proceed), else 1
(production must REFUSE the experiment). The exact-quadrature certificate is a
stricter (or equivalent) gate than the deployed RA-LSC per-atom residual.

Usage:  JCP_GPU=4 python scripts/certificate_gate.py <experiment_name>
        experiment_name in {double_well, mog40, mb3well_10d, coupled_phi4}
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))          # JCP_experiments/
from src.gpu_guard import select_gpu
select_gpu(os.environ.get("JCP_GPU", "4"))
import torch
torch.set_default_dtype(torch.float64)

from src.config import Q_THETA, Q_RHO
from src.certificate import (certificate_grid, certificate_importance,
                             make_phi_family)
from src.jumps import ShellJumpLaw, gauss_legendre_01
from src.score import ShellScore, MoG40Score

TOL = float(os.environ.get("JCP_CERT_TOL", "1e-6"))   # tighten/loosen for tests
DEV = "cuda"
_CACHE = os.path.join(os.path.dirname(_HERE), "tests", ".cache")


def cert_e1() -> float:
    from src.experiments import build_e1
    exp = build_e1(device=DEV)
    phis = make_phi_family(1, [0.0], 1.0, DEV)
    sh, lw = exp.law.quadrature_shifts(64)
    sc = ShellScore(exp.pot, exp.law, exp.cfg.lam, exp.cfg.beta, Q_THETA, Q_RHO)
    return certificate_grid(exp.pot, sc, sh, lw, exp.cfg.lam, exp.cfg.beta, phis,
                            [-5.2], [5.2], n_panels=120, nodes_per_panel=8)["max_residual"]


def cert_e2() -> float:
    from src.experiments import build_e2
    exp = build_e2(device=DEV)
    sc = MoG40Score(exp.pot.mu, 4.0, 15.0, exp.cfg.lam, m_phi=32)
    sh, lw = exp.law.quadrature_shifts(8, 64)
    phis = make_phi_family(2, [0.0, 0.0], 30.0, DEV)
    return certificate_grid(exp.pot, sc, sh, lw, exp.cfg.lam, exp.cfg.beta, phis,
                            [-60.0, -60.0], [60.0, 60.0], n_panels=120,
                            nodes_per_panel=6, chunk=8192)["max_residual"]


def cert_e3() -> float:
    from src.experiments import build_e3
    from src.potentials import MB3Latent2D
    exp = build_e3(device=DEV, basin_cache=os.path.join(_CACHE, "mb3_basins.npz"))
    potr = MB3Latent2D()
    dz = exp.extras["atoms_z"][:, :2]
    h_z = exp.extras["h"] * dz.norm(dim=1) / exp.law.atoms.norm(dim=1)
    law_r = ShellJumpLaw(dz, exp.law.weights.clone(), h_z)
    sc = ShellScore(potr, law_r, exp.cfg.lam, exp.cfg.beta, Q_THETA, Q_RHO)
    sh, lw = law_r.quadrature_shifts(64)
    phis = make_phi_family(2, [0.0, 0.5], 0.8, DEV)
    return certificate_grid(potr, sc, sh, lw, exp.cfg.lam, exp.cfg.beta, phis,
                            exp.extras["cert_lo"], exp.extras["cert_hi"],
                            n_panels=200, nodes_per_panel=10, chunk=8192)["max_residual"]


def cert_e4() -> float:
    from src.experiments import build_e4
    exp = build_e4(device=DEV, basin_cache=os.path.join(_CACHE, "phi4_basins.npz"))
    theta, w_theta = gauss_legendre_01(Q_THETA, DEV)
    sh, lw = exp.law.quadrature_shifts(Q_RHO)
    shj, lwj = exp.law.quadrature_shifts(64)
    phis = make_phi_family(24, exp.extras["means24"][0].tolist(), 1.5, DEV, n_phi=4)
    return certificate_importance(exp.pot, sh, lw, theta, w_theta, exp.cfg.lam,
                                  exp.cfg.beta, phis, exp.extras["laplace"],
                                  n_samples=200_000, nu_shifts_jump=shj,
                                  nu_logw_jump=lwj)["max_residual"]


_GATES = {"double_well": cert_e1, "mog40": cert_e2,
          "mb3well_10d": cert_e3, "coupled_phi4": cert_e4}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in _GATES:
        print(f"usage: certificate_gate.py <{'|'.join(_GATES)}>", file=sys.stderr)
        return 2
    name = sys.argv[1]
    R = float(_GATES[name]())
    ok = R < TOL
    print(f"{name}: generous-box certificate max R = {R:.3e}  (tol {TOL})  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
