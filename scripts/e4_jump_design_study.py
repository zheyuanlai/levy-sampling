#!/usr/bin/env python
"""E4 supplementary study: how much does the choice of jump measure nu matter?

The manuscript's E4 uses eight atoms read straight off the phi^4 phase square.
This study replaces nu with the symmetric alpha-stable family FLA already uses
as the uncorrected heavy-tailed control -- a measure that encodes nothing about
the four coherent phases -- in two forms:

``nu2``   a 2-dimensional per-site displacement tiled coherently across all 12
          sites, the traditional way jumps are composed for a coupled chain.
          Homogeneous, so it also supports a deterministic product quadrature.
``nu24``  the same coordinatewise law applied directly in 24 dimensions, as FLA
          applies its noise. No product rule fits, so only the realised-
          displacement estimator can run it -- which is the point.

The target, reference, collective variable and metrics are the manuscript's,
untouched; only nu moves.  Nothing here writes into ``results/coupled_phi4/`` or
registers a fifth manuscript experiment, so the frozen release stays valid.

Wall-clock is deliberately not reported: this runs on a shared GPU. Physical
time, NFE and score-quadrature evaluations are hardware-independent and are.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

from src.gpu_guard import select_gpu

HERE = Path(__file__).resolve().parent.parent

# Design grid, mirroring scripts/e4_jump_design_calibrate.py.
BASELINE_MEAN_LENGTH = 6.928105591973582
REFERENCE_TRUNCATION = 0.99
REFERENCE_SCALE = 1.0
TRUNCATION_LEVELS = (0.90, 0.95, 0.99)
SCALE_MULTIPLIERS = (0.5, 1.0, 2.0)
BANK_SIZES = (1, 2, 4, 8, 16, 32)
# Converged reference for nu-24, which has no deterministic quadrature. One
# doubling past the top of the sweep is enough to show whether the estimator
# has settled; 128 would cost four GPU-hours on its own for the same answer.
REFERENCE_BANK = 64
SENSITIVITY_BANK = 8
# Metric checkpoints. The manuscript uses 60 + 160 to resolve plotted curves and
# their running averages across E1--E4; this study reports terminal values and
# coarse trajectories, and metrics cost ~0.2 s per seed per checkpoint
# regardless of arm, which at 36 arms is hours of pure diagnostic time.
N_DENSE_CHECKPOINTS = 20
N_SPARSE_CHECKPOINTS = 40
Q_THETA = 32          # chord nodes, as selected by E4's quadrature refinement
Q_U = 16              # nu-2 product-rule order; within 0.3% of q_u = 32
BASIN_CELL = 0.01     # frozen E4 basin-map cell size, held constant
SHARED_BASIN_HALF_WIDTH = 18.0   # widest box in the grid; see the calibration table

STAGES = {
    # seeds, particles per seed, T
    "smoke": (1, 200, 1.0),
    "pilot": (4, 1000, 20.0),
    "production": (16, 1000, 100.0),
}


def design_configurations():
    """(design, q, L) points: the bank sweep at the reference, the sensitivity
    sweep on the two axes through it."""
    out, seen = [], set()
    for design in ("nu2", "nu24"):
        for q in TRUNCATION_LEVELS:
            for L in SCALE_MULTIPLIERS:
                if not (q == REFERENCE_TRUNCATION or L == REFERENCE_SCALE):
                    continue
                key = (design, q, L)
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "design": design, "truncation_mass": q, "scale": L,
                    "box_reach_multiplier": 1.0,
                    "is_reference": (q == REFERENCE_TRUNCATION
                                     and L == REFERENCE_SCALE),
                })
    # Box-sensitivity control. A heavy-tailed nu keeps particles out in
    # excursions where a second jump can reach the numerical wall, so the
    # reference point of each design is repeated with the box sized for two
    # jumps instead of one. If the metrics move, the finding is about the box;
    # if they do not, it is about nu.
    for design in ("nu2", "nu24"):
        out.append({
            "design": design, "truncation_mass": REFERENCE_TRUNCATION,
            "scale": REFERENCE_SCALE, "box_reach_multiplier": 2.0,
            "is_reference": False,
        })
    # Headline first: the two reference points carry the bank sweep and the
    # design comparison, then the box controls that defend them, then the
    # sensitivity sweep. An interrupted campaign then still has a result.
    out.sort(key=lambda c: (not c["is_reference"],
                            c["box_reach_multiplier"] == 1.0,
                            c["design"], c["truncation_mass"], c["scale"]))
    return out


def arms_for(config: dict) -> list[str]:
    """Which samplers a configuration runs.

    The reference point carries the full bank sweep; the sensitivity points
    carry one bank size so the two axes stay separable. Raw-CP runs everywhere
    -- without the uncorrected ablation on the identical nu, a good corrected
    result cannot be attributed to the score rather than to the jump law.

    Each design gets exactly one ground truth for the bank estimator, and only
    at the reference point where the sweep needs it: nu-2 has the deterministic
    product quadrature, so it does not also need a large bank; nu-24 has no
    quadrature at all, so a converged large bank is the only reference
    available. Running both everywhere would roughly double the campaign for no
    extra information.
    """
    arms = ["CP"]
    if config["is_reference"]:
        arms += [f"LSC-CP-RA-{A}" for A in BANK_SIZES]
        if config["design"] == "nu2":
            arms.append("LSC-CP")
        else:
            arms.append(f"LSC-CP-RA-{REFERENCE_BANK}")
    else:
        arms.append(f"LSC-CP-RA-{SENSITIVITY_BANK}")
    return arms


def build_law(design: str, truncation_mass: float, scale: float, device,
              n_sites: int = 12):
    from src.jump_designs import TiledStableLaw, TruncatedCoordinateStableLaw
    target = scale * BASELINE_MEAN_LENGTH
    if design == "nu24":
        return TruncatedCoordinateStableLaw.with_mean_length(
            24, target, truncation_mass, device)
    base = TruncatedCoordinateStableLaw.with_mean_length(
        2, target / math.sqrt(n_sites), truncation_mass, device)
    return TiledStableLaw(base, n_sites)


def config_key(config: dict) -> str:
    key = "%s_q%s_L%s" % (config["design"],
                          str(config["truncation_mass"]).replace(".", ""),
                          str(config["scale"]).replace(".", ""))
    if config.get("box_reach_multiplier", 1.0) != 1.0:
        key += "_box%g" % config["box_reach_multiplier"]
    return key


def make_factory(exp, dt, x0_blocks, device):
    """Fresh sampler per arm name, all arms sharing one initial ensemble.

    Built here rather than through ``src.experiments.make_batched_factory``
    because the bank size A is an arm-level knob that the manuscript's fixed
    method names cannot express -- and because leaving that factory alone keeps
    the frozen release path untouched.
    """
    import torch
    from src.config import diffusion_seed, jump_seed
    from src.samplers import CompoundPoisson
    from src.score import IIDBankScore

    lam, beta, eps = exp.cfg.lam, exp.cfg.beta, exp.cfg.eps

    def factory(arm: str):
        x0 = x0_blocks
        gen = torch.Generator(device=device)
        # Distinct diffusion stream per arm, derived from the arm name. crc32
        # rather than hash(): Python randomises string hashing per process, so
        # hash() would make the run irreproducible across invocations.
        gen.manual_seed(diffusion_seed("LSC-CP-MA", 0)
                        + (zlib.crc32(arm.encode()) % 100_000))
        g_jump = torch.Generator(device=device)
        g_jump.manual_seed(jump_seed(0))
        common = dict(name=arm, drift_cap=float(exp.cp_drift_cap))
        if arm == "CP":
            return CompoundPoisson(exp.pot, x0, dt, eps, lam, exp.law, gen,
                                   g_jump, exp.box, score=None, **common)
        if arm == "LSC-CP":
            score = exp.make_score(q_theta=Q_THETA, q_rho=Q_U)
            return CompoundPoisson(exp.pot, x0, dt, eps, lam, exp.law, gen,
                                   g_jump, exp.box, score=score, **common)
        if arm.startswith("LSC-CP-RA-"):
            A = int(arm.rsplit("-", 1)[1])
            score = IIDBankScore(exp.pot, exp.law, lam, beta, n_atoms=A,
                                 q_theta=Q_THETA)
            return CompoundPoisson(exp.pot, x0, dt, eps, lam, exp.law, gen,
                                   g_jump, exp.box, score=score,
                                   jump_mode="paired_multiatom", **common)
        raise ValueError(f"unknown arm {arm!r}")

    return factory


def run_configuration(config, stage, device, out_root, basin_cache, force):
    import torch
    from src.config import init_seed
    from src.experiments import build_e4, make_metrics
    from src.runner import (checkpoint_schedule, run_experiment_batched,
                            write_manifest, write_summary_csv,
                            write_timeseries_csv)

    n_seeds, n_particles, final_time = STAGES[stage]
    key = config_key(config)
    out_dir = out_root / key
    if out_dir.exists() and not force:
        print(f"  {key}: already present, skipping (use --force to redo)")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)

    law = build_law(config["design"], config["truncation_mass"],
                    config["scale"], device)
    exp = build_e4(device=device, jump_law=law,
                   box_reach_multiplier=config.get("box_reach_multiplier", 1.0),
                   basin_bounds=(-SHARED_BASIN_HALF_WIDTH,
                                 SHARED_BASIN_HALF_WIDTH),
                   basin_n_grid=int(round(2.0 * SHARED_BASIN_HALF_WIDTH
                                          / BASIN_CELL)),
                   basin_cache=basin_cache)

    dt = exp.cfg.dt
    n_steps = int(round(final_time / dt))
    seeds = tuple(range(n_seeds))
    blocks = []
    for seed in seeds:
        g = torch.Generator(device=device)
        g.manual_seed(init_seed(seed))
        blocks.append(exp.init_fn(n_particles, g))
    x0 = torch.cat(blocks, dim=0)

    metrics_fn, floors, metric_aux = make_metrics(
        exp, n_particles, device=device)
    arms = arms_for(config)
    factory = make_factory(exp, dt, x0, device)

    print(f"  {key}: {len(arms)} arms x {n_steps} steps "
          f"({n_seeds} seeds x {n_particles} particles), box +/-"
          f"{exp.extras['sampling_box_design']['sampling_box_half_width']}, "
          f"drift cap {exp.cp_drift_cap:.4f}", flush=True)

    exp.pot.reset_counters()
    t0 = time.perf_counter()
    rows, method_info = run_experiment_batched(
        arms, seeds, factory, n_steps,
        steps_per_ck=max(1, n_steps // 50), dt=dt, metrics_fn=metrics_fn,
        potential=exp.pot, n_per_seed=n_particles,
        checkpoint_steps=checkpoint_schedule(
            n_steps, n_dense=N_DENSE_CHECKPOINTS,
            n_sparse=N_SPARSE_CHECKPOINTS))
    elapsed = time.perf_counter() - t0

    write_timeseries_csv(rows, str(out_dir / "metrics_timeseries.csv"),
                         overwrite=force)
    write_summary_csv(rows, arms, seeds, ["W2", "MMD", "TV"], method_info,
                      floors, str(out_dir / "summary.csv"), overwrite=force)
    for info in method_info.values():
        info.pop("final_positions_seed0", None)
        info.pop("final_positions_all", None)
    write_manifest(
        str(out_dir / "manifest.json"), overwrite=force,
        study="e4_jump_design",
        configuration=config,
        arms=arms,
        stage=stage,
        jump_law=exp.extras["jump_law_description"],
        cp_drift_cap=float(exp.cp_drift_cap),
        sampling_box_design=exp.extras["sampling_box_design"],
        basin_map_metric_bounds=exp.extras["basin_map_metric_bounds"],
        p_star=exp.p_star.tolist(),
        q_theta=Q_THETA,
        q_u=Q_U,
        config={"d": exp.cfg.d, "N": n_particles, "T": final_time,
                "dt": dt, "beta": exp.cfg.beta, "lam": exp.cfg.lam,
                "seeds": n_seeds},
        bias_floors=floors,
        method_info=method_info,
        elapsed_s=elapsed,
        wallclock_reported=False,
        wallclock_note=("run on a shared GPU; timings are not comparable to "
                        "the manuscript's dedicated-device protocol"),
        clamp_fallback_fraction=float(exp.law.fallback_fraction()),
        metric_reference=metric_aux.get("sample_reference_method"),
    )
    print(f"  {key}: done in {elapsed / 60.0:.1f} min", flush=True)
    return out_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), default="pilot")
    parser.add_argument("--gpu", default=os.environ.get("JCP_GPU", "2"),
                        help="GPU index (opt in with JCP_EXTRA_GPUS) or 'cpu'")
    parser.add_argument("--designs", default="nu2,nu24")
    parser.add_argument("--only-reference", action="store_true",
                        help="run only the reference (q, L) point of each design")
    parser.add_argument("--configs", default=None,
                        help="comma-separated configuration keys to run, or "
                             "substrings of them; useful for sharding a long "
                             "campaign across sessions")
    parser.add_argument("--force", action="store_true",
                        help="rerun configurations whose output already exists")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)

    select_gpu(args.gpu)
    import torch
    torch.set_default_dtype(torch.float64)
    from src.device import DEFAULT_DEVICE

    device = DEFAULT_DEVICE
    out_root = Path(args.output_root) if args.output_root else (
        HERE / "results" / "e4_jump_design" / args.stage)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = HERE / "cache" / "e4_jump_design"
    cache_dir.mkdir(parents=True, exist_ok=True)
    basin_cache = str(cache_dir / ("basin_shared_%g_%d.npz" % (
        SHARED_BASIN_HALF_WIDTH,
        int(round(2.0 * SHARED_BASIN_HALF_WIDTH / BASIN_CELL)))))

    wanted = {d.strip() for d in args.designs.split(",") if d.strip()}
    configs = [c for c in design_configurations() if c["design"] in wanted]
    if args.only_reference:
        configs = [c for c in configs if c["is_reference"]]
    if args.configs:
        wanted_keys = [k.strip() for k in args.configs.split(",") if k.strip()]
        configs = [c for c in configs
                   if any(k in config_key(c) for k in wanted_keys)]
        if not configs:
            print(f"no configuration matches {args.configs!r}", file=sys.stderr)
            return 1

    print(f"stage={args.stage} device={device} configurations={len(configs)}")
    print(f"output {out_root}")
    for config in configs:
        run_configuration(config, args.stage, device, out_root, basin_cache,
                          args.force)
    print("study stage complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
