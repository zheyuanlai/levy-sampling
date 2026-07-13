"""RA-LSC atomwise certificate.

RandomAtomicShellScore relies on the fact that for EVERY fixed realised shift R
the atomic sub-generator A_{eps,R} is individually mu-invariant. We verify this
directly: build the single-atom (fixed-R) score as a ShellScore over a 1-atom
law {R} (h=0, so its quadrature is just R) and run the weak-stationarity
certificate with the single jump shift R. The per-atom residual is a STRICTER
statement than the mixture certificate (the mixture residual is bounded by the
max over atoms).
"""
import torch

from tests.conftest import CACHE_DIR  # noqa: F401  (selects the GPU)
from src.config import Q_THETA, Q_RHO
from src.potentials import DoubleWell1D
from src.jumps import ShellJumpLaw
from src.score import ShellScore
from src.certificate import certificate_grid, make_phi_family

DEV = "cuda"


def _single_atom_score(pot, r, lam, beta, q_theta):
    """ShellScore over the single fixed atom r (h=0): its log_parts is exactly
    the fixed-R integrand RandomAtomicShellScore evaluates per step."""
    atoms = r.reshape(1, -1)
    law = ShellJumpLaw(atoms, torch.ones(1, dtype=torch.float64, device=DEV), h=0.0)
    return ShellScore(pot, law, lam, beta, q_theta, q_rho=1)


def test_atomwise_certificate_generous_box():
    """Every realised shift on the (a, rho_q) grid: R(phi) < 1e-6 on the
    generous box (production quadrature orders)."""
    pot = DoubleWell1D()
    beta, lam = 8.0, 1.0
    phis = make_phi_family(1, [0.0], 1.0, DEV)
    base_law = ShellJumpLaw(torch.tensor([[2.0], [-2.0]], dtype=torch.float64, device=DEV),
                            torch.tensor([0.5, 0.5], dtype=torch.float64, device=DEV), 0.2)
    shifts, _ = base_law.quadrature_shifts(Q_RHO)      # (2*Q_RHO, 1) realised shifts
    worst = 0.0
    for r in shifts:
        score = _single_atom_score(pot, r, lam, beta, Q_THETA)
        res = certificate_grid(pot, score, r.reshape(1, -1),
                               torch.zeros(1, dtype=torch.float64, device=DEV),
                               lam, beta, phis, [-5.2], [5.2],
                               n_panels=120, nodes_per_panel=8)
        worst = max(worst, res["max_residual"])
    assert worst < 1e-6, worst


def test_atomwise_tight_box_regression():
    """A deliberately tight box must expose a large residual for a fixed atom
    too: order-one identity mass lives where pi is tiny and S is enormous.
    Guards against anyone shrinking the certificate domain."""
    pot = DoubleWell1D()
    r = torch.tensor([[2.0]], dtype=torch.float64, device=DEV)
    score = _single_atom_score(pot, r, 1.0, 8.0, Q_THETA)
    phis = make_phi_family(1, [0.0], 1.0, DEV)
    res = certificate_grid(pot, score, r,
                           torch.zeros(1, dtype=torch.float64, device=DEV),
                           1.0, 8.0, phis, [-1.3], [1.3],
                           n_panels=60, nodes_per_panel=8)
    assert res["max_residual"] > 1e-2, res["max_residual"]
