"""RA-LSC jump/score mismatch must break invariance.

The RA estimator is correct only because the SAME realised R drives both the
score drift and the jump. If the jump uses atom r_a but the score is built for a
different atom r_b, mu-invariance MUST fail -- the certificate residual has to be
significantly non-zero. A correct implementation therefore FAILS this residual
check (formulation note condition 17.2).
"""
import torch

from tests.conftest import CACHE_DIR  # noqa: F401  (selects the GPU)
from src.config import Q_THETA
from src.potentials import DoubleWell1D
from src.jumps import ShellJumpLaw
from src.score import ShellScore
from src.certificate import certificate_grid, make_phi_family

DEV = "cuda"


def _single_atom_score(pot, r, lam, beta, q_theta):
    atoms = r.reshape(1, -1)
    law = ShellJumpLaw(atoms, torch.ones(1, dtype=torch.float64, device=DEV), h=0.0)
    return ShellScore(pot, law, lam, beta, q_theta, q_rho=1)


def test_mismatched_jump_and_score_is_nonzero():
    pot = DoubleWell1D()
    r_a = torch.tensor([[2.0]], dtype=torch.float64, device=DEV)   # jump uses r_a
    r_b = torch.tensor([[-2.0]], dtype=torch.float64, device=DEV)  # score built for r_b
    score_b = _single_atom_score(pot, r_b, 1.0, 8.0, Q_THETA)
    phis = make_phi_family(1, [0.0], 1.0, DEV)
    res = certificate_grid(pot, score_b, r_a,
                           torch.zeros(1, dtype=torch.float64, device=DEV),
                           1.0, 8.0, phis, [-5.2], [5.2],
                           n_panels=120, nodes_per_panel=8)
    # drift (for r_b) cannot cancel the jump (for r_a): residual is order one
    assert res["max_residual"] > 1e-2, res["max_residual"]
