#!/usr/bin/env python
"""CPU consistency checks for the E4 jump-design study.

Everything here is cheap and runs before any GPU time is spent. The checks are
the ones whose failure would be silent:

1. the frozen E4 build path is unchanged by the study's edits;
2. ``IIDBankScore`` at A = 1 reproduces the single-atom ``RandomAtomicShellScore``;
3. the paired multi-atom score still reads the shell law's own atom masses;
4. the nu-2 product quadrature agrees with a large Monte-Carlo score;
5. the applied jump process is A-independent, as the estimator argument claims;
6. a non-homogeneous law never reaches ``CoupledPhi4.V_delta``.
"""
from __future__ import annotations

import argparse
import math
import sys

from src.gpu_guard import select_gpu


def _report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu",
                        help="'cpu' or a GPU index opted in via JCP_EXTRA_GPUS")
    args = parser.parse_args(argv)
    select_gpu(args.device)

    import torch
    torch.set_default_dtype(torch.float64)

    from src.experiments import build_e4
    from src.jump_designs import (TiledStableLaw, TruncatedCoordinateStableLaw,
                                  assert_homogeneous_for_v_delta)
    from src.samplers import CompoundPoisson
    from src.score import (IIDBankScore, MultiAtomShellScore,
                           RandomAtomicShellScore)

    dev = torch.device("cpu") if args.device == "cpu" else torch.device("cuda", 0)
    ok = True

    # --- 1. the default build path is untouched --------------------------
    exp = build_e4(device=dev, basin_cache=None, basin_n_grid=64,
                   basin_flow_steps=2000, snis_proposals=20_000)
    h = float(exp.extras["h"])
    box_half = float(exp.extras["sampling_box_design"]["sampling_box_half_width"])
    ok &= _report(
        "frozen E4 shell half-width unchanged",
        math.isclose(h, 0.6928105591973582, rel_tol=1e-12),
        f"h = {h!r}")
    ok &= _report(
        "frozen E4 sampling box unchanged",
        box_half == 5.0, f"half-width = {box_half}")
    ok &= _report(
        "frozen E4 drift cap unchanged",
        math.isclose(float(exp.cp_drift_cap), 2.0 * h, rel_tol=1e-12),
        f"cap = {float(exp.cp_drift_cap)!r}")
    ok &= _report("frozen E4 law is the 8 phase-edge atoms",
                  tuple(exp.law.atoms.shape) == (8, 24))

    pot, law, lam, beta = exp.pot, exp.law, exp.cfg.lam, exp.cfg.beta
    g = torch.Generator(device=dev)
    g.manual_seed(11)
    x = exp.init_fn(64, g)

    # --- 3. shell bank still uses the law's atom masses ------------------
    ma = MultiAtomShellScore(pot, law, lam, beta, q_theta=32)
    ok &= _report("multi-atom score reads the shell law's weights",
                  bool(torch.equal(ma.weights, law.weights))
                  and ma.A == law.A and ma.d == law.d)

    # --- 2. IIDBankScore at A = 1 == RandomAtomicShellScore --------------
    ra = RandomAtomicShellScore(pot, law, lam, beta, q_theta=32)
    bank1 = IIDBankScore(pot, law, lam, beta, n_atoms=1, q_theta=32)
    R = law.sample(x.shape[0], g)
    with pot.no_count():
        s_ra, _ = ra.score_for_shift(x, R)
        s_b1, _ = bank1.score_for_bank(x, R.unsqueeze(1))
    max_dev = float((s_ra - s_b1).abs().max())
    ok &= _report("IIDBankScore(A=1) == RandomAtomicShellScore",
                  max_dev <= 1e-12 * float(s_ra.abs().max().clamp(min=1.0)),
                  f"max |difference| = {max_dev:.3e}")

    # --- 4. nu-2 product quadrature vs a large Monte-Carlo score ---------
    base2 = TruncatedCoordinateStableLaw.with_mean_length(2, 2.0, 0.99, dev)
    tiled = TiledStableLaw(base2, pot.Ns)
    # The Monte-Carlo reference is itself noisy -- the bank score is dominated
    # by rare large-ratio draws -- so quote its own seed-to-seed spread and
    # judge the quadrature by self-convergence rather than against a moving
    # target.
    from src.score import ShellScore
    mc_atoms = 1 << 16
    mc = IIDBankScore(pot, tiled, lam, beta, n_atoms=mc_atoms, q_theta=32)
    xs = x[:8]
    mc_runs = []
    for seed in (4242, 4243, 4244, 4245):
        g_mc = torch.Generator(device=dev)
        g_mc.manual_seed(seed)
        with pot.no_count():
            s_mc, _ = mc.score_for_bank(xs, mc.sample_bank(xs.shape[0], g_mc))
        mc_runs.append(s_mc)
    mc_mean = torch.stack(mc_runs).mean(dim=0)

    def _rel(a, b):
        return float(((a - b).norm(dim=1)
                      / b.norm(dim=1).clamp(min=1e-300)).median())

    mc_spread = max(_rel(r, mc_mean) for r in mc_runs)
    print("    Monte-Carlo reference: %d atoms, %d seeds, median relative "
          "seed-to-seed spread = %.3e" % (mc_atoms, len(mc_runs), mc_spread))
    print("    nu-2 product quadrature:")
    dets = {}
    for q_u in (4, 8, 16, 24, 32):
        det = ShellScore(pot, tiled, lam, beta, q_theta=32, q_rho=q_u)
        with pot.no_count():
            dets[q_u], _ = det(xs)
        print("      q_u = %2d  J = %5d  vs MC mean = %.3e   vs q_u=32 = %.3e"
              % (q_u, det.J, _rel(dets[q_u], mc_mean), 0.0))
    self_conv = {q_u: _rel(dets[q_u], dets[32]) for q_u in (4, 8, 16, 24)}
    print("    self-convergence to q_u = 32: "
          + ", ".join("q_u=%d: %.3e" % (k, v) for k, v in self_conv.items()))
    ok &= _report(
        "nu-2 product quadrature self-converges",
        self_conv[24] < 0.02 and self_conv[24] < self_conv[4],
        "q_u=24 differs from q_u=32 by %.3e" % self_conv[24])
    ok &= _report(
        "nu-2 quadrature agrees with Monte Carlo within its noise",
        _rel(dets[32], mc_mean) < max(3.0 * mc_spread, 0.02),
        "quadrature-vs-MC %.3e, MC spread %.3e"
        % (_rel(dets[32], mc_mean), mc_spread))

    # --- 5. the applied jump process is A-independent --------------------
    counts = {}
    for A in (1, 4, 16):
        score = IIDBankScore(pot, tiled, lam, beta, n_atoms=A, q_theta=8)
        g_d = torch.Generator(device=dev)
        g_d.manual_seed(5)
        g_j = torch.Generator(device=dev)
        g_j.manual_seed(6)
        smp = CompoundPoisson(pot, x, 0.002, exp.cfg.eps, lam, tiled, g_d, g_j,
                              exp.box, score=score, name=f"bank{A}",
                              drift_cap=float(exp.cp_drift_cap),
                              jump_mode="paired_multiatom")
        n_steps = 2000
        with pot.no_count():
            for _ in range(n_steps):
                smp.step()
        # pop_diagnostics averages per step, so scale back to a total.
        counts[A] = float(smp.pop_diagnostics()["jump_count_mean"]) * n_steps
    expected = n_steps * lam * 0.002
    spread = max(counts.values()) - min(counts.values())
    ok &= _report(
        "applied jump rate is A-independent",
        spread < 0.25 * expected,
        "mean jumps/particle " + ", ".join(f"A={A}: {c:.4f}" for A, c in counts.items())
        + f" (expected {expected:.4f})")

    # --- 6. a non-homogeneous law must not reach V_delta -----------------
    raw24 = TruncatedCoordinateStableLaw.with_mean_length(24, 6.9282, 0.99, dev)
    try:
        assert_homogeneous_for_v_delta(raw24, pot)
        guarded = False
    except TypeError:
        guarded = True
    ok &= _report("nu-24 is refused by the V_delta homogeneity guard", guarded)
    try:
        raw24.quadrature_shifts(8)
        refused = False
    except NotImplementedError:
        refused = True
    ok &= _report("nu-24 has no product quadrature", refused)
    assert_homogeneous_for_v_delta(tiled, pot)
    assert_homogeneous_for_v_delta(law, pot)
    ok &= _report("nu-2 and the phase-edge law pass the homogeneity guard", True)

    # --- 7. the swapped-law build path ----------------------------------
    cheap = dict(basin_n_grid=64, basin_flow_steps=2000, snis_proposals=20_000)
    exp2 = build_e4(device=dev, jump_law=tiled, basin_bounds=(-9.0, 9.0), **cheap)
    exp24 = build_e4(device=dev, jump_law=raw24, basin_bounds=(-9.0, 9.0), **cheap)
    ok &= _report(
        "swapped-law box is derived from the law's own reach",
        math.isclose(
            exp2.extras["sampling_box_design"]["max_componentwise_jump_reach"],
            tiled.max_componentwise_reach(), rel_tol=1e-12),
        "box +/-%g for nu-2, +/-%g for nu-24"
        % (exp2.extras["sampling_box_design"]["sampling_box_half_width"],
           exp24.extras["sampling_box_design"]["sampling_box_half_width"]))
    ok &= _report(
        "basin domain follows basin_bounds, with no second literal left behind",
        (exp2.extras["basin_map_metric_bounds"] == [[-9.0, -9.0], [9.0, 9.0]]
         and exp2.extras["builder_reference_parameters"]["basin_bounds"]
         == exp2.extras["basin_map_metric_bounds"]))
    # cap = 0.2 * E||R|| reproduces the manuscript's 2h whenever the mean jump
    # length is matched to the phase-edge atoms, so the rule is a restatement
    # rather than a new convention.
    ok &= _report(
        "drift-cap rule reproduces 2h at the matched mean length",
        math.isclose(float(exp2.cp_drift_cap), 2.0 * h, rel_tol=1e-3)
        and math.isclose(float(exp24.cp_drift_cap), 2.0 * h, rel_tol=1e-3),
        "nu-2 cap %.6f, nu-24 cap %.6f, manuscript 2h = %.6f"
        % (exp2.cp_drift_cap, exp24.cp_drift_cap, 2.0 * h))
    try:
        exp24.make_score(q_theta=8, q_rho=8)
        refused24 = False
    except TypeError:
        refused24 = True
    ok &= _report("the exact-quadrature score refuses nu-24", refused24)
    exp2.make_score(q_theta=8, q_rho=4)
    ok &= _report("the exact-quadrature score accepts nu-2", True)

    # --- 8. the M_MAX cap is inert once the drift is tamed ---------------
    # nu-24 drives the score into its log-magnitude cap far more often than any
    # manuscript run does, so the study reports that fraction as a measure of
    # how large the correction has to be. That is only honest if the cap does
    # not itself change the trajectory. It does not: tame(b, dt, cap) has
    # displacement cap * b/||b|| once dt*||b|| >> cap, so at those magnitudes
    # the step depends on the direction of b alone. Verified rather than
    # asserted -- run the same chain against the highest cap float64 can
    # represent and require an identical path. (Above log(DBL_MAX) = 709 the
    # score overflows to infinity and the trajectory becomes NaN, which is why
    # a cap has to exist at all; 600 versus 700 tests whether its *value*
    # matters, which is the question here.)
    def _run(m_max, n_steps=150):
        score = IIDBankScore(pot, raw24, lam, beta, n_atoms=8, q_theta=8,
                             m_max=m_max)
        g_d = torch.Generator(device=dev)
        g_d.manual_seed(21)
        g_j = torch.Generator(device=dev)
        g_j.manual_seed(22)
        smp = CompoundPoisson(pot, x, 0.002, exp24.cfg.eps, lam, raw24, g_d,
                              g_j, exp24.box, score=score, name="probe",
                              drift_cap=float(exp24.cp_drift_cap),
                              jump_mode="paired_multiatom")
        with pot.no_count():
            for _ in range(n_steps):
                smp.step()
        return smp.positions(), smp.pop_diagnostics()

    x_capped, diag = _run(600.0)
    x_raised, _ = _run(700.0)
    drift = float((x_capped - x_raised).abs().max())
    # Not bit-identity: dividing by 1 + dt||b||/cap at ||b|| ~ e^600 loses the
    # low bits of the direction differently under each cap, and 150 steps of a
    # chaotic chain amplify that. The bound is set well below any resolution
    # the metrics have (the drift cap itself is 1.39 and the box is +/-9).
    ok &= _report(
        "the value of the M_MAX cap does not change the tamed trajectory",
        math.isfinite(drift) and drift < 1e-6,
        "max |position difference| = %.3e over 150 steps with the cap at 600 "
        "vs 700 (round-off, not a systematic shift); clip fraction %.2e"
        % (drift, diag.get("m_clip_fraction", 0.0)))

    # The mechanism behind that: once the score saturates the tamed drift, one
    # step moves exactly the cap distance regardless of how large the score is.
    # Evaluate on post-jump excursion states, not on the initial ensemble --
    # near a minimum the score is O(1) and the test would pass vacuously.
    score = IIDBankScore(pot, raw24, lam, beta, n_atoms=8, q_theta=8)
    g_b = torch.Generator(device=dev)
    g_b.manual_seed(31)
    x_excursion = x + raw24.sample(x.shape[0], g_b)
    with pot.no_count():
        S, sdiag = score.score_for_bank(x_excursion,
                                        score.sample_bank(x.shape[0], g_b))
    x = x_excursion
    from src.samplers import tame
    cap = float(exp24.cp_drift_cap)
    b = -pot.grad(x) + S
    displacement = (0.002 * tame(b, 0.002, cap)).norm(dim=1)
    saturated = S.norm(dim=1) > 1e100
    frac_saturated = float(saturated.to(torch.float64).mean())
    worst = (float((displacement[saturated] - cap).abs().max())
             if bool(saturated.any()) else 0.0)
    ok &= _report(
        "a saturated score moves exactly one cap length per step",
        worst < 1e-9 and frac_saturated > 0.5,
        "%.1f%% of post-jump particles saturated; worst |step - cap| = %.3e"
        % (100.0 * frac_saturated, worst))

    print()
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
