#!/usr/bin/env python
"""Calibration table for the E4 jump-design study. CPU only, seconds to run.

Every configuration of the study is defined by a design (nu-2 composed or nu-24
FLA-matched), a truncation level q = P(|eta_i| <= c), and a scale multiplier L
on the mean jump length. This script resolves each of them and prints the
quantities that decide whether a run is affordable and whether its diagnostics
can pass: sigma, c, E||R||, the componentwise and collective-variable reaches,
the implied sampling box, the implied basin-map domain and grid, the drift cap,
and the per-step score cost.

Read this before spending GPU time. The heavy tail is the part of the design
most likely to force a change.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

from src.gpu_guard import select_gpu

# The manuscript's phase-edge law: eight coherent atoms of norm sqrt(12)*2.
BASELINE_MEAN_LENGTH = 6.928105591973582
# Frozen E4 basin-map cell size, 8.0 / 800. Held constant as the domain grows.
BASIN_CELL = 0.01
# Chord nodes selected by E4's own quadrature refinement.
Q_THETA = 32
# Product-rule order for the nu-2 deterministic arm: q_u = 16 sits within 0.3%
# of q_u = 32 and well inside the Monte-Carlo reference's own noise.
Q_U = 16

# Truncation levels q = P(|eta_i| <= c). q = 0.99 gives c = 7.29, i.e. single
# coordinate displacements up to ~3.8 times the 2.0 phase spacing -- genuinely
# heavy relative to the structure the jump has to find. Going to q = 0.999
# (c = 26.7) admits displacements twelve times the phase spacing, which lands
# every jump in a region where exp(-beta dV) underflows; it forces a +/-28 box
# and a 31M-cell basin map to measure a foregone conclusion, so the sweep goes
# lighter instead of heavier.
TRUNCATION_LEVELS = (0.90, 0.95, 0.99)
SCALE_MULTIPLIERS = (0.5, 1.0, 2.0)
REFERENCE_TRUNCATION = 0.99
REFERENCE_SCALE = 1.0
BANK_SIZES = (1, 2, 4, 8, 16, 32)
REFERENCE_BANK = 128


def configurations():
    """(design, q, L) grid: the bank sweep at the reference point, the
    sensitivity sweep at a fixed bank."""
    seen, out = set(), []
    for design in ("nu2", "nu24"):
        for q in TRUNCATION_LEVELS:
            for L in SCALE_MULTIPLIERS:
                is_reference = (q == REFERENCE_TRUNCATION and L == REFERENCE_SCALE)
                on_axis = is_reference or q == REFERENCE_TRUNCATION or L == REFERENCE_SCALE
                if not on_axis:
                    continue
                key = (design, q, L)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"design": design, "truncation_mass": q, "scale": L,
                            "is_reference": is_reference})
    return out


def build_law(design: str, truncation_mass: float, scale: float, device,
              n_sites: int = 12):
    from src.jump_designs import TiledStableLaw, TruncatedCoordinateStableLaw
    target = scale * BASELINE_MEAN_LENGTH
    if design == "nu24":
        return TruncatedCoordinateStableLaw.with_mean_length(
            24, target, truncation_mass, device)
    if design == "nu2":
        # Coherent tiling amplifies the per-site length by sqrt(n_sites), so
        # match the per-site scale that reproduces the same 24-dimensional
        # mean jump length.
        base = TruncatedCoordinateStableLaw.with_mean_length(
            2, target / math.sqrt(n_sites), truncation_mass, device)
        return TiledStableLaw(base, n_sites)
    raise ValueError(f"unknown design {design!r}")


def resolve(config: dict, pieces: dict, device) -> dict:
    from src.experiments import _phi4_sampling_box_design
    law = build_law(config["design"], config["truncation_mass"],
                    config["scale"], device)
    box_design = _phi4_sampling_box_design(
        pieces["means24"], pieces["atoms"], pieces["h"], pieces["hessians"],
        beta=pieces["beta"], pt_beta_min=pieces["pt_beta_min"],
        componentwise_reach=float(law.max_componentwise_reach()))
    box_half = float(box_design["sampling_box_half_width"])
    # States are clipped into the box componentwise and qbar is a mean of site
    # coordinates, so pinning the basin domain to the box makes the
    # outside-mass diagnostic structurally zero.
    n_grid = int(round(2.0 * box_half / BASIN_CELL))
    row = dict(config)
    row.update({
        "sigma": float(getattr(law, "base", law).sigma),
        "c": float(getattr(law, "base", law).c),
        "mean_length": float(law.mean_length()),
        "max_componentwise_reach": float(law.max_componentwise_reach()),
        "max_reach": float(law.max_reach()),
        "metric_reach": float(law.metric_reach()
                              if hasattr(law, "metric_reach")
                              else law.max_reach() / math.sqrt(12)),
        "box_half_width": box_half,
        "basin_half_width": box_half,
        "basin_n_grid": n_grid,
        "basin_cells_millions": n_grid * n_grid / 1e6,
        "drift_cap": 0.2 * float(law.mean_length()),
        "exact_cost": (Q_U * Q_U * Q_THETA if config["design"] == "nu2" else None),
    })
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", default=None,
                        help="write the resolved grid as JSON")
    args = parser.parse_args(argv)
    select_gpu("cpu")

    import torch
    torch.set_default_dtype(torch.float64)
    from src.experiments import build_e4

    device = torch.device("cpu")
    # Cheap build: only the phase-edge geometry, Hessians and beta are needed.
    base = build_e4(device=device, basin_n_grid=8, basin_flow_steps=10,
                    snis_proposals=1000)
    pieces = {
        "means24": base.extras["means24"],
        "atoms": base.law.atoms,
        "h": base.extras["h"],
        "hessians": base.extras["hessians"],
        "beta": base.cfg.beta,
        "pt_beta_min": base.pt_beta_min,
    }

    rows = [resolve(c, pieces, device) for c in configurations()]

    print("Baseline (manuscript phase-edge law): E||R|| = %.4f, componentwise "
          "reach = %.4f, box +/-%d, basin +/-4 (n_grid 800), drift cap %.4f"
          % (BASELINE_MEAN_LENGTH,
             base.extras["sampling_box_design"]["max_componentwise_jump_reach"],
             base.extras["sampling_box_design"]["sampling_box_half_width"],
             base.cp_drift_cap))
    print()
    header = ("design    q      L    sigma      c     E||R||  comp.reach "
              " qbar.reach   box  n_grid  cells/M  drift cap")
    print(header)
    print("-" * len(header))
    for r in rows:
        print("%-6s %5.3f %4.1f %8.4f %7.3f %8.4f %10.3f %11.3f %5d %7d %8.2f %10.4f%s"
              % (r["design"], r["truncation_mass"], r["scale"], r["sigma"],
                 r["c"], r["mean_length"], r["max_componentwise_reach"],
                 r["metric_reach"], r["box_half_width"], r["basin_n_grid"],
                 r["basin_cells_millions"], r["drift_cap"],
                 "  <- reference" if r["is_reference"] else ""))

    widest = max(r["box_half_width"] for r in rows)
    shared_grid = int(round(2.0 * widest / BASIN_CELL))
    print()
    print("Shared basin-map domain for the whole study: +/-%g, n_grid %d "
          "(%.1fM cells at the frozen cell size %g)"
          % (widest, shared_grid, shared_grid ** 2 / 1e6, BASIN_CELL))
    print("One domain for every configuration means one p_star, so the designs "
          "are scored against the same reference.")
    print()
    print("Score cost, chord energies per particle per step (q_theta = %d):" % Q_THETA)
    print("  manuscript exact LSC-CP (8 atoms x q_rho 4) : %5d" % (8 * 4 * Q_THETA))
    print("  manuscript LSC-CP-RA (8)                    : %5d" % (8 * Q_THETA))
    print("  nu-2 exact product rule (q_u = %d)          : %5d"
          % (Q_U, Q_U * Q_U * Q_THETA))
    for A in BANK_SIZES:
        print("  LSC-CP-RA (%3d)                             : %5d" % (A, A * Q_THETA))
    print("  LSC-CP-RA (%3d) converged reference          : %5d"
          % (REFERENCE_BANK, REFERENCE_BANK * Q_THETA))

    if args.json_out:
        payload = {
            "baseline_mean_length": BASELINE_MEAN_LENGTH,
            "basin_cell": BASIN_CELL,
            "q_theta": Q_THETA,
            "q_u": Q_U,
            "bank_sizes": list(BANK_SIZES),
            "reference_bank": REFERENCE_BANK,
            "shared_basin_half_width": widest,
            "shared_basin_n_grid": shared_grid,
            "configurations": rows,
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
