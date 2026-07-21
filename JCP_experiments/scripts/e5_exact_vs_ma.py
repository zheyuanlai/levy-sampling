"""E5: cross-validate the deployed paired multi-atom arm against the exact
deterministic-quadrature arm.

E5's deployed estimator is LSC-CP-MA, because a black-box force field affords
only chord ENERGIES; the exact ShellScore additionally needs the jump law's
closed-form rho-quadrature and costs q_theta*A*q_rho = 768 chord points per
particle per step (8x the MA bank), i.e. ~21 h at the production ensemble.
Rather than deploy it, we validate against it in two ways:

  (A) POINTWISE, and this is the sharp test: MA is an unbiased estimator of the
      exact score, E_bank[S_MA(x)] = S_exact(x). Averaging MA over many banks at
      fixed reference states must converge to the exact score. Costs minutes.
  (B) END-TO-END: run the exact arm's dynamics on a reduced seed set for the
      full horizon and compare basin occupancies against the MA arm on the same
      seeds. Costs a few hours.

Usage: python -m scripts.e5_exact_vs_ma [--seeds 2] [--steps 40000] [--banks 4096]
"""
from __future__ import annotations
import argparse, json, os, time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--banks", type=int, default=4096)
    ap.add_argument("--states", type=int, default=256)
    ap.add_argument("--out", default="results/e5_exact_vs_ma.json")
    args = ap.parse_args()

    from src.experiments import build_e5_alanine
    from src.score import MultiAtomShellScore
    from src.samplers import CompoundPoisson
    from src.config import diffusion_seed, jump_seed, init_seed, Q_THETA

    dev = "cuda"
    exp = build_e5_alanine(device=dev)
    cfg = exp.cfg
    steps = args.steps if args.steps is not None else int(round(cfg.T / cfg.dt))
    out: dict = {"config": {"seeds": args.seeds, "steps": steps,
                            "banks": args.banks, "states": args.states,
                            "n_particles": cfg.n_particles, "dt": cfg.dt,
                            "beta": cfg.beta, "q_theta": Q_THETA}}

    # ---- (A) pointwise unbiasedness: E_bank[S_MA] -> S_exact ----------------
    ref = exp.extras["reference"]
    g = torch.Generator(device=dev); g.manual_seed(20260722)
    x = ref.sample(args.states, g)
    exact = exp.make_score()
    S_exact, _ = exact(x)
    ma = MultiAtomShellScore(exp.pot, exp.law, cfg.lam, cfg.beta, Q_THETA)
    acc = torch.zeros_like(S_exact)
    t0 = time.time()
    done = 0
    while done < args.banks:
        b = min(256, args.banks - done)
        for _ in range(b):
            R = ma.sample_bank(x.shape[0], g)
            S, _ = ma.score_for_bank(x, R)
            acc += S
        done += b
    S_ma = acc / args.banks
    # compare on the phi/psi components, which are the only ones the jumps drive
    sl = [int(exp.pot.phi_slot), int(exp.pot.psi_slot)]
    num = (S_ma[:, sl] - S_exact[:, sl]).norm(dim=1)
    den = S_exact[:, sl].norm(dim=1).clamp(min=1e-300)
    out["pointwise"] = {
        "n_banks": args.banks, "n_states": int(x.shape[0]),
        "median_rel_err": float((num / den).median()),
        "mean_rel_err": float((num / den).mean()),
        "p90_rel_err": float((num / den).quantile(0.9)),
        "corr_phi": float(torch.corrcoef(torch.stack(
            [S_ma[:, sl[0]], S_exact[:, sl[0]]]))[0, 1]),
        "seconds": round(time.time() - t0, 1),
    }
    print("(A) pointwise:", json.dumps(out["pointwise"]))
    del exact, ma, acc
    torch.cuda.empty_cache()

    # ---- (B) end-to-end dynamics on a reduced seed set ----------------------
    blocks = []
    for s in cfg.seeds[:args.seeds]:
        gi = torch.Generator(device=dev); gi.manual_seed(init_seed(s))
        blocks.append(exp.init_fn(cfg.n_particles, gi))
    x0 = torch.cat(blocks, 0)
    island = set(exp.extras["positive_phi_basins"])
    K = int(exp.p_star.shape[0])

    def _run(name):
        gd = torch.Generator(device=dev); gd.manual_seed(diffusion_seed(
            "LSC-CP" if name == "exact" else "LSC-CP-MA", 0))
        gj = torch.Generator(device=dev); gj.manual_seed(jump_seed(0))
        if name == "exact":
            sc = exp.make_score(); mode = "full"
        else:
            sc = MultiAtomShellScore(exp.pot, exp.law, cfg.lam, cfg.beta, Q_THETA)
            mode = "paired_multiatom"
        smp = CompoundPoisson(exp.pot, x0, cfg.dt, cfg.eps, cfg.lam, exp.law,
                              gd, gj, exp.box, score=sc, name=name,
                              drift_cap=exp.cp_drift_cap, jump_mode=mode)
        t = time.time()
        for i in range(steps):
            smp.step()
        pos = smp.positions()
        lab = exp.labels_fn(pos)
        p_hat = torch.stack([(lab == k).to(torch.float64).mean() for k in range(K)])
        return {
            "hours": round((time.time() - t) / 3600, 3),
            "p_hat": [round(float(v), 5) for v in p_hat],
            "island_occupancy": round(float(sum(
                (lab == k).to(torch.float64).mean() for k in island)), 5),
            "basin_L1": round(float((p_hat - exp.p_star).abs().sum()), 4),
            "nonfinite": int((~torch.isfinite(pos)).any(dim=-1).sum()),
        }

    for name in ("ma", "exact"):
        out[name] = _run(name)
        print(f"(B) {name}:", json.dumps(out[name]))
        torch.cuda.empty_cache()

    out["p_star"] = [round(float(v), 5) for v in exp.p_star]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
