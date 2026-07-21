"""E5 P5 gate: BAT torsion jump law (relay + drop-forbidden) + RA/MA + R(phi).

Gate (task S P5):
  * R(phi) on a generous torus domain below the documented threshold, reported
    alongside the tight-domain reading and the exact domains;
  * retained atoms have bounded chords (no persistent M_MAX saturation);
  * the dropped-atom log matches the geography;
  * score direction sanity across the phi barrier.

Certificate form: 60-D rules out a quadrature grid, so -- as for E4 (24-D) -- the
residual is the SHIFTED form, which reduces to the pointwise theta-quadrature
defect with no O(1) cancellation left to Monte Carlo. mu expectations are taken
against the reweighted metadynamics pool. The DIRECT form is reported but not
gated: its integrand p.S is O(1) exactly where p is exponentially small, so it is
not mu-estimable in high dimension.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from tests.conftest import CACHE_DIR  # noqa: F401  (selects the GPU)
from src.config import M_MAX, Q_THETA
from src.score import RandomAtomicShellScore, MultiAtomShellScore
from src.e5_alanine.build_reference import CACHE_DIR as E5_CACHE

DEV = "cuda"

MAX_RESIDUAL_SHIFTED = 1e-6      # declared threshold, matching E1-E4
MAX_CHORD_SATURATION = 1e-3      # per-atom fraction allowed at M_MAX


def _exp():
    if not os.path.exists(os.path.join(E5_CACHE, "reference.npz")):
        pytest.skip("E5 reference cache not generated")
    from src.experiments import build_e5_alanine
    return build_e5_alanine(device=DEV)


def test_jump_law_geography_and_screen():
    exp = _exp()
    rec = exp.extras["jump_design"]
    print(f"atoms: {rec['n_retained']} retained / {rec['n_candidates']} candidates, "
          f"h = {rec['h']:.4f}, cap = {rec['cp_drift_cap']:.4f}")
    for info in rec["dropped"]:
        print(f"  DROPPED {info['src']}->{info['dst']} "
              f"(dphi={info['dphi_deg']:.0f}, dpsi={info['dpsi_deg']:.0f}): {info['reason']}")
    assert rec["n_retained"] >= 2
    # every retained atom has a bounded chord (no persistent M_MAX saturation)
    for info in rec["retained"]:
        assert info["frac_saturating_M_MAX"] <= MAX_CHORD_SATURATION, info
        assert info["nonfinite_fraction"] == 0.0, info
    # +- pairs present (both homotopy directions)
    assert rec["plus_minus_pairs_present"]
    # basin pairs with no direct atom must be reachable by relay
    for r in rec["relay_pairs"]:
        assert r["connected"], r
    # h and the drift cap follow the E3/E4 rule
    assert abs(rec["cp_drift_cap"] - 2.0 * rec["h"]) < 1e-12
    assert abs(rec["h"] - 0.1 * rec["min_atom_norm"]) < 1e-12
    assert abs(exp.cp_drift_cap - 2.0 * rec["h"]) < 1e-12


def test_atoms_are_pure_torsion():
    """Atoms move only phi and psi: this is what makes the chord Jacobian-free."""
    exp = _exp()
    pot, atoms = exp.pot, exp.law.atoms
    keep = {int(pot.phi_slot), int(pot.psi_slot)}
    others = [i for i in range(pot.d) if i not in keep]
    assert float(atoms[:, others].abs().max()) == 0.0
    # and the shell jitter stays in the same plane
    gen = torch.Generator(device=DEV)
    gen.manual_seed(1)
    R = exp.law.sample(256, gen)
    assert float(R[:, others].abs().max()) == 0.0


def test_certificate_generous_vs_tight_domain():
    exp = _exp()
    pot, ref = exp.pot, exp.extras["reference"]
    from src.e5_alanine.certificate import (certificate_atomwise_weighted,
                                            torus_phi_family, tight_domain_mask)
    phis = torus_phi_family(pot, exp.law.atoms)
    score = RandomAtomicShellScore(pot, exp.law, exp.cfg.lam, exp.cfg.beta, Q_THETA)

    generous = certificate_atomwise_weighted(
        pot, ref.qt, ref.weights, exp.law.atoms, exp.law.weights,
        exp.cfg.lam, exp.cfg.beta, phis, q_theta=Q_THETA, score=score)
    # tight domain: a single basin core, which truncates the identity
    core = ref.minima[ref.deepest_basin()]
    mask = tight_domain_mask(ref.cvs, core, 0.6)
    tight = certificate_atomwise_weighted(
        pot, ref.qt, ref.weights, exp.law.atoms, exp.law.weights,
        exp.cfg.lam, exp.cfg.beta, phis, q_theta=Q_THETA, mask=mask, score=score)

    print(f"GENEROUS domain (full torus, (-pi,pi]^2, n={generous['n_samples']}): "
          f"R_shifted = {generous['max_residual_shifted']:.3e}, "
          f"R_direct = {generous.get('max_residual_direct', float('nan')):.3e}")
    print(f"TIGHT domain (|dphi|,|dpsi| < 0.6 rad of the deepest basin, "
          f"n={tight['n_samples']}): R_shifted = {tight['max_residual_shifted']:.3e}, "
          f"R_direct = {tight.get('max_residual_direct', float('nan')):.3e}")

    # the gate is the shifted-form residual on the generous domain
    assert generous["max_residual_shifted"] < MAX_RESIDUAL_SHIFTED, \
        generous["max_residual_shifted"]
    assert generous["mixture_residual_shifted"] < MAX_RESIDUAL_SHIFTED
    # test functions must be informative (non-vanishing jump term)
    jumps = [v["jump_term"] for k, v in generous.items()
             if k.startswith("atom") and isinstance(v, dict)]
    assert max(abs(j) for j in jumps) > 1e-4
    # the deployed M_MAX cap must not fire where mu has mass
    assert generous["max_log_magnitude"] < M_MAX


def test_score_direction_across_the_phi_barrier():
    exp = _exp()
    pot, ref = exp.pot, exp.extras["reference"]
    from src.e5_alanine.jump_design import score_direction_sanity
    score = RandomAtomicShellScore(pot, exp.law, exp.cfg.lam, exp.cfg.beta, Q_THETA)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(7)
    out = score_direction_sanity(pot, ref, exp.law, score, gen, n=512)
    for k, v in out.items():
        print(f"  {k}: mean S_phi = {v['mean_score_phi']:+.4f}, "
              f"mean torus displacement to the deepest basin = "
              f"{v['mean_disp_to_deepest']:+.4f}, "
              f"sign agreement = {v['sign_agreement']:.3f}")
    assert out["pos_phi"] is not None and out["neg_phi"] is not None
    # From the sparsely populated positive-phi island the correction must point
    # back toward the dominant cluster ALONG THE TORUS-SHORT PATH. For this force
    # field that path runs through +-180 (10.9 kJ/mol) not through phi ~ 0
    # (32.8 kJ/mol), so the correct statement is agreement of signs with the
    # torus displacement, not a fixed sign.
    pos = out["pos_phi"]
    assert np.sign(pos["mean_score_phi"]) == np.sign(pos["mean_disp_to_deepest"]), pos
    assert pos["sign_agreement"] > 0.5, pos
    # and the correction is much stronger from the sparse island than from the
    # dominant cluster, where the jump flux is already near balance
    assert abs(pos["mean_score_phi"]) > abs(out["neg_phi"]["mean_score_phi"]), out


def test_multiatom_score_constructs_against_the_law():
    """MA validates law geometry/units/weights; also checks eps = 1/beta."""
    exp = _exp()
    ma = MultiAtomShellScore(exp.pot, exp.law, exp.cfg.lam, exp.cfg.beta, Q_THETA)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(3)
    R = ma.sample_bank(64, gen)
    assert R.shape == (64, exp.law.A, exp.pot.d)
    q = exp.extras["reference"].sample(64, gen)
    S, diag = ma.score_for_bank(q, R)
    assert S.shape == q.shape and bool(torch.isfinite(S).all())
    assert float(diag["m_clip_fraction"]) == 0.0
    assert abs(exp.cfg.beta * exp.cfg.eps - 1.0) < 1e-12
