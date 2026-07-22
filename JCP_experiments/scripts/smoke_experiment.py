"""Run a small, real dynamics smoke for one JCP experiment.

This gate builds the requested physical model, instantiates every method that
will appear in the corresponding production notebook, advances each sampler,
and validates finite in-box states, energies, basin labels, diagnostics, and
compute counters.  It is deliberately not an equilibrium or accuracy claim.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
JCP_ROOT = HERE.parent
sys.path.insert(0, str(JCP_ROOT))

EXPERIMENT_NOTEBOOK_METHODS = {
    "double_well": "ULA,MALA,FLA,BAOAB,PT,CP,LSC-CP,LSC-CP-RA",
    "mog40": "ULA,MALA,FLA,BAOAB,PT,CP,LSC-CP,LSC-CP-RA",
    "mb3well_10d": "ULA,MALA,FLA,BAOAB,PT,CP,LSC-CP,LSC-CP-MA",
    "coupled_phi4": "ULA,MALA,FLA,BAOAB,PT,CP,LSC-CP,LSC-CP-MA",
    "alanine_dipeptide": "ULA,MALA,FLA,BAOAB,PT,CP,LSC-CP,LSC-CP-MA",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False,
                  default=str)
        handle.write("\n")


def _parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(item.strip() for item in value.split(",") if item.strip())
    if not methods or len(set(methods)) != len(methods):
        raise argparse.ArgumentTypeError("--methods must be a nonempty unique CSV")
    return methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True,
                        choices=tuple(EXPERIMENT_NOTEBOOK_METHODS))
    parser.add_argument("--methods", required=True, type=_parse_methods)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--q-theta", type=int, default=4)
    parser.add_argument("--q-rho", type=int, default=2)
    parser.add_argument("--pt-replicas", type=int, default=4)
    parser.add_argument("--basin-n-grid", type=int, default=48)
    parser.add_argument("--basin-flow-steps", type=int, default=800)
    parser.add_argument("--basin-mass-n-quad", type=int, default=96)
    parser.add_argument("--reference-grid-size", type=int, default=128)
    parser.add_argument("--snis-proposals", type=int, default=2_048)
    parser.add_argument("--max-score-clip-fraction", type=float, default=0.01)
    parser.add_argument("--max-state-box-clip-fraction", type=float, default=0.01)
    parser.add_argument(
        "--max-jump-boundary-clip-fraction", type=float, default=0.0)
    parser.add_argument(
        "--max-basin-map-outside-fraction", type=float, default=0.0)
    parser.add_argument("--max-jump-cap-hits", type=int, default=0)
    parser.add_argument("--min-mala-acceptance", type=float, default=0.01)
    parser.add_argument("--min-pt-swap-acceptance", type=float, default=0.01)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
            "particles", "steps", "q_theta", "q_rho", "basin_n_grid",
            "basin_flow_steps", "basin_mass_n_quad", "reference_grid_size",
            "snis_proposals"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.pt_replicas < 2:
        raise ValueError("--pt-replicas must be at least 2")
    for name in (
            "max_score_clip_fraction", "max_state_box_clip_fraction",
            "max_jump_boundary_clip_fraction",
            "max_basin_map_outside_fraction",
            "min_mala_acceptance", "min_pt_swap_acceptance"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must lie in [0, 1]")
    if args.max_jump_cap_hits < 0:
        raise ValueError("--max-jump-cap-hits must be non-negative")
    # The gate's contract is that every method a job will run has first passed a
    # bounded real-dynamics smoke. Requiring set EQUALITY additionally assumed
    # that one job always runs the whole matrix -- true until E5, whose exact
    # arm costs ~25 h at the production ensemble and therefore runs as its own
    # method shard on its own GPU. A shard smokes exactly the methods it runs,
    # so the safety property is intact; whole-matrix coverage is restored by the
    # union-of-shards check at merge time (scripts/merge_method_shards.py).
    # An unregistered method is still refused: a shard may only narrow.
    registered = set(EXPERIMENT_NOTEBOOK_METHODS[args.experiment].split(","))
    supplied = set(args.methods)
    if not supplied:
        raise ValueError("smoke methods must name at least one method")
    extra = supplied - registered
    if extra:
        raise ValueError(
            "smoke methods must be a subset of the production method matrix; "
            f"extra={sorted(extra)}, registered={sorted(registered)}"
        )


def _requires_score_quadrature(experiment: str, method: str) -> bool:
    """Whether this LSC implementation should charge V(x-theta r) calls.

    Every deployed LSC arm now integrates numerically, so all of them charge.
    (E2 previously deployed the analytic MoG40 score, which exercised a real
    score path with n_Vdelta == 0; that arm has been retired to a comparator.)
    """
    return method.startswith("LSC-CP")


def _strictly_finite_mapping(mapping: dict, *, context: str) -> None:
    for key, value in mapping.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise RuntimeError(f"{context}: nonfinite diagnostic {key}={value}")


def _build_experiment(name: str, device: str, cache_dir: Path,
                      args: argparse.Namespace):
    """Build with smoke-only coarse references isolated under this run."""
    from src.experiments import (build_e1, build_e2, build_e3, build_e4,
                                 build_e5_alanine)

    cache_dir.mkdir(parents=True, exist_ok=False)
    if name == "double_well":
        return build_e1(device=device)
    if name == "mog40":
        return build_e2(device=device)
    if name == "alanine_dipeptide":
        # E5 has no gradient-flow basin map: the partition is a torus Voronoi
        # around the metadynamics FES minima, so the smoke reuses the committed
        # reference cache rather than building a coarse one.
        return build_e5_alanine(device=device, n_particles=args.particles)
    cache = str(cache_dir / "basin_map.npz")
    common = dict(
        basin_cache=cache,
        basin_n_grid=args.basin_n_grid,
        basin_flow_steps=args.basin_flow_steps,
    )
    if name == "mb3well_10d":
        return build_e3(
            device=device, **common,
            basin_mass_n_quad=args.basin_mass_n_quad,
            reference_grid_shape=(args.reference_grid_size,
                                  args.reference_grid_size),
        )
    if name == "coupled_phi4":
        return build_e4(
            device=device, **common, snis_proposals=args.snis_proposals)
    raise ValueError(name)


def run_smoke(args: argparse.Namespace) -> dict:
    """Execute the model/method smoke and persist immutable raw artifacts."""
    _validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started_at = _utc_now()

    # Persist only requested inputs before GPU/model/reference construction so
    # a failed builder still leaves the exact smoke request behind.
    source_config = {
        "schema_version": 1,
        "kind": "smoke_request",
        "experiment": args.experiment,
        "methods": list(args.methods),
        "particles": args.particles,
        "steps": args.steps,
        "q_theta": args.q_theta,
        "q_rho": args.q_rho,
        "pt_replicas": args.pt_replicas,
        "smoke_reference_settings": {
            "basin_n_grid": args.basin_n_grid,
            "basin_flow_steps": args.basin_flow_steps,
            "basin_mass_n_quad": args.basin_mass_n_quad,
            "reference_grid_shape": [args.reference_grid_size,
                                     args.reference_grid_size],
            "snis_proposals": args.snis_proposals,
            "cache_scope": "this smoke artifact directory only",
        },
        "failure_thresholds": {
            "max_score_clip_fraction": args.max_score_clip_fraction,
            "max_state_box_clip_fraction": args.max_state_box_clip_fraction,
            "max_jump_boundary_clip_fraction": (
                args.max_jump_boundary_clip_fraction),
            "max_basin_map_outside_fraction": (
                args.max_basin_map_outside_fraction),
            "max_jump_cap_hits": args.max_jump_cap_hits,
            "min_mala_acceptance": args.min_mala_acceptance,
            "min_pt_swap_acceptance": args.min_pt_swap_acceptance,
        },
    }
    # Pure JSON is valid YAML 1.2.
    with (output_dir / "original_config.yaml").open(
            "x", encoding="utf-8") as handle:
        json.dump(source_config, handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")

    # gpu_guard must run before importing torch.
    from src.gpu_guard import select_gpu
    select_gpu(int(os.environ.get("JCP_GPU", "4")))
    import torch
    torch.set_default_dtype(torch.float64)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("smoke requires exactly one visible CUDA GPU")

    from src.experiments import make_sampler_factory
    from src.runner import hardware_manifest
    from src.samplers import geometric_ladder

    exp = _build_experiment(
        args.experiment, "cuda", output_dir / "smoke_cache", args)
    if exp.name != args.experiment:
        raise RuntimeError(f"builder returned {exp.name!r}, expected {args.experiment!r}")
    p_star = exp.p_star.reshape(-1)
    if (p_star.numel() < 1
            or not bool(torch.isfinite(p_star).all().item())
            or bool((p_star < 0).any().item())
            or float(p_star.sum().item()) <= 0.0
            or not math.isclose(float(p_star.sum().item()), 1.0,
                                rel_tol=1e-10, abs_tol=1e-12)):
        raise RuntimeError(f"invalid smoke target basin probabilities: {p_star}")
    reference_basin_map_outside = float(
        exp.extras.get("reference_diagnostics", {}).get(
            "weighted_basin_map_outside_mass", 0.0))
    if reference_basin_map_outside > args.max_basin_map_outside_fraction:
        raise RuntimeError(
            "reference basin-map outside mass "
            f"{reference_basin_map_outside} exceeds "
            f"{args.max_basin_map_outside_fraction}")
    pt_betas = geometric_ladder(
        exp.cfg.beta, exp.pt_beta_min, args.pt_replicas, exp.p_star.device)
    factory = make_sampler_factory(
        exp, exp.cfg.dt, pt_betas, n_particles=args.particles,
        score_kwargs={"q_theta": args.q_theta, "q_rho": args.q_rho},
    )

    _write_json_exclusive(output_dir / "resolved_config.json", {
        **source_config,
        "kind": "smoke_resolved",
        "resolved_model": {
            "experiment_name": exp.name,
            "dimension": exp.cfg.d,
            "dt": exp.cfg.dt,
            "beta": exp.cfg.beta,
            "lambda": exp.cfg.lam,
            "potential_type": type(exp.pot).__name__,
            "jump_law_type": type(exp.law).__name__,
        },
        "builder_reference_parameters": exp.extras.get(
            "builder_reference_parameters", {}),
        "sampling_box": {
            "lower": exp.box.lo.detach().cpu().tolist(),
            "upper": exp.box.hi.detach().cpu().tolist(),
        },
        "sampling_box_design": exp.extras.get("sampling_box_design"),
        "basin_cache_provenance": exp.extras.get(
            "basin_cache_provenance"),
        "p_star": [float(v) for v in p_star.detach().cpu()],
        "reference_diagnostics": exp.extras.get("reference_diagnostics"),
        "pt_betas": [float(v) for v in pt_betas.detach().cpu()],
        "device": "cuda",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    })

    rows: list[dict] = []
    for method in args.methods:
        n_v0, n_g0, n_q0 = exp.pot.n_V, exp.pot.n_grad, exp.pot.n_Vdelta
        sampler = factory(method, 0)
        for step in range(args.steps):
            sampler.step()
            positions = sampler.positions()
            if positions.shape != (args.particles, exp.cfg.d):
                raise RuntimeError(
                    f"{method}: state shape {tuple(positions.shape)} != "
                    f"{(args.particles, exp.cfg.d)}"
                )
            if not bool(torch.isfinite(positions).all().item()):
                raise RuntimeError(f"{method}: nonfinite state at step {step + 1}")
        positions = sampler.positions()
        if not bool(exp.box.contains(positions).all().item()):
            raise RuntimeError(f"{method}: final state outside configured box")
        with exp.pot.no_count():
            energies = exp.pot.V(positions)
            labels = exp.labels_fn(positions)
            basin_map_bounds = exp.extras.get("basin_map_metric_bounds")
            if basin_map_bounds is None:
                basin_map_outside_fraction = 0.0
            else:
                metric_positions = exp.metric_space(positions)[:, :2]
                basin_lo = torch.as_tensor(
                    basin_map_bounds[0], dtype=metric_positions.dtype,
                    device=metric_positions.device)
                basin_hi = torch.as_tensor(
                    basin_map_bounds[1], dtype=metric_positions.dtype,
                    device=metric_positions.device)
                basin_inside = ((metric_positions >= basin_lo)
                                & (metric_positions <= basin_hi)).all(dim=1)
                basin_map_outside_fraction = float(
                    (~basin_inside).to(torch.float64).mean().item())
        if not bool(torch.isfinite(energies).all().item()):
            raise RuntimeError(f"{method}: nonfinite final energy")
        if labels.shape != (args.particles,) or labels.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise RuntimeError(f"{method}: invalid basin label array")
        if not bool(((labels >= 0) & (labels < exp.p_star.numel())).all().item()):
            raise RuntimeError(f"{method}: basin label outside target support")
        diagnostics = sampler.pop_diagnostics()
        _strictly_finite_mapping(diagnostics, context=method)
        if method.startswith("LSC-CP"):
            required_score_diagnostics = {"m_clip_fraction", "max_log_magnitude"}
            missing_score_diagnostics = required_score_diagnostics - diagnostics.keys()
            if missing_score_diagnostics:
                raise RuntimeError(
                    f"{method}: score path did not report "
                    f"{sorted(missing_score_diagnostics)}")
            lifetime_score_clip = float(diagnostics.get(
                "score_clip_fraction_cumulative",
                diagnostics["m_clip_fraction"]))
            if lifetime_score_clip > args.max_score_clip_fraction:
                raise RuntimeError(
                    f"{method}: lifetime score clipping "
                    f"{lifetime_score_clip} exceeds "
                    f"{args.max_score_clip_fraction}")
        state_clip = float(diagnostics.get(
            "state_box_clip_fraction_cumulative", 0.0))
        if state_clip > args.max_state_box_clip_fraction:
            raise RuntimeError(
                f"{method}: state-box clipping {state_clip} exceeds "
                f"{args.max_state_box_clip_fraction}")
        jump_boundary_clip = float(diagnostics.get(
            "jump_boundary_clip_fraction_per_applied_jump_cumulative", 0.0))
        # Gate pi-targeting LSC-CP methods only. Raw CP's invariant law != pi
        # and its boundary contact is the documented defect: recorded in the
        # manifest diagnostics, not gated (same convention as the dt and
        # basin-map gates).
        if (method.startswith("LSC-CP")
                and jump_boundary_clip > args.max_jump_boundary_clip_fraction):
            raise RuntimeError(
                f"{method}: jump-boundary clipping per applied jump "
                f"{jump_boundary_clip} exceeds "
                f"{args.max_jump_boundary_clip_fraction}")
        if (method not in ("FLA", "CP", "CP-RA")
                and basin_map_outside_fraction
                > args.max_basin_map_outside_fraction):
            raise RuntimeError(
                f"{method}: target-preserving basin-map outside fraction "
                f"{basin_map_outside_fraction} exceeds "
                f"{args.max_basin_map_outside_fraction}")
        cap_hits = int(diagnostics.get("jump_cap_hit_count_cumulative", 0))
        if cap_hits > args.max_jump_cap_hits:
            raise RuntimeError(
                f"{method}: jump-cap hits {cap_hits} exceed "
                f"{args.max_jump_cap_hits}")
        nonfinite_proposals = int(diagnostics.get(
            "nonfinite_proposal_count_cumulative", 0))
        if nonfinite_proposals != 0:
            raise RuntimeError(
                f"{method}: observed {nonfinite_proposals} nonfinite proposals")
        if method == "MALA":
            acceptance = float(diagnostics.get(
                "mala_accept_fraction_cumulative", -1.0))
            if acceptance < args.min_mala_acceptance:
                raise RuntimeError(
                    f"MALA acceptance {acceptance} is below "
                    f"{args.min_mala_acceptance}")
        if method == "PT":
            swap_acceptance = float(diagnostics.get(
                "pt_swap_accept_fraction_cumulative", -1.0))
            if swap_acceptance < args.min_pt_swap_acceptance:
                raise RuntimeError(
                    f"PT swap acceptance {swap_acceptance} is below "
                    f"{args.min_pt_swap_acceptance}")
        row = {
            "method": method,
            "particles": args.particles,
            "steps": args.steps,
            "state_finite": True,
            "state_inside_box": True,
            "energy_mean": float(energies.mean().item()),
            "energy_min": float(energies.min().item()),
            "energy_max": float(energies.max().item()),
            "basins_observed": int(torch.unique(labels).numel()),
            "basin_map_outside_fraction": basin_map_outside_fraction,
            "potential_evals": int(exp.pot.n_V - n_v0),
            "gradient_evals": int(exp.pot.n_grad - n_g0),
            "score_quadrature_evals": int(exp.pot.n_Vdelta - n_q0),
        }
        row.update(diagnostics)
        _strictly_finite_mapping(row, context=method)
        if row["gradient_evals"] <= 0:
            raise RuntimeError(f"{method}: no gradient evaluation was recorded")
        score_requires_quadrature = _requires_score_quadrature(exp.name, method)
        analytic_mog40_score = (
            method.startswith("LSC-CP") and not score_requires_quadrature)
        if (score_requires_quadrature
                and row["score_quadrature_evals"] <= 0):
            raise RuntimeError(f"{method}: no Levy-score quadrature was recorded")
        if analytic_mog40_score and row["score_quadrature_evals"] != 0:
            raise RuntimeError(
                "MoG40 analytic LSC-CP unexpectedly used potential quadrature")
        rows.append(row)

    columns = sorted(set().union(*(row.keys() for row in rows)))
    with (output_dir / "smoke_metrics.csv").open(
            "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "schema_version": 1,
        "status": "success",
        "kind": "per_experiment_dynamics_smoke",
        "experiment": args.experiment,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "methods": list(args.methods),
        "config": source_config,
        "p_star": [float(v) for v in p_star.detach().cpu()],
        "reference_diagnostics": exp.extras.get("reference_diagnostics"),
        "basin_cache_provenance": exp.extras.get("basin_cache_provenance"),
        "rows": rows,
        "hardware": hardware_manifest(),
    }
    _write_json_exclusive(output_dir / "smoke_manifest.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_smoke(args)
    except BaseException as exc:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = output_dir / "failure_manifest.json"
        if not failure_path.exists():
            _write_json_exclusive(failure_path, {
                "schema_version": 1,
                "status": "failed",
                "experiment": args.experiment,
                "finished_at_utc": _utc_now(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            })
        raise
    print(
        f"smoke passed: {result['experiment']} "
        f"({len(result['methods'])} methods)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
