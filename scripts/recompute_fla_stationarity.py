"""Recompute genuine FLA worst-basin ESS traces for the frozen E1--E4 release.

FLA does not preserve the target, so its ESS is a biased-kernel mixing
diagnostic and must be read with W2/MMD/basin-TV.  This script follows the
same reference-start, charged-settling protocol as the other stationarity
summaries.  It writes the compact CSV/manifest artifacts used by manuscript
plotting and keeps the larger raw traces in the ignored ``cache/`` tree.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import N_WARMUP_STEPS  # noqa: E402
from src.experiments import (  # noqa: E402
    build_e1,
    build_e2,
    build_e3,
    build_e4,
    make_sampler_factory,
)
from src.runner import json_safe  # noqa: E402
from src.potentials import mb3_2d_grad, phi4_W_grad  # noqa: E402
from src.stationarity import (  # noqa: E402
    collect_stationary_trajectories,
    flat_summary_rows,
    summarize_stationary_traces,
    write_stationarity_csv,
    write_stationarity_npz,
)


BUILDERS = {
    "double_well": build_e1,
    "mog40": build_e2,
    "mb3well_10d": build_e3,
    "coupled_phi4": build_e4,
}


def _without_wallclock(value):
    """Remove hardware-dependent timing outputs from release-facing objects."""
    if isinstance(value, dict):
        return {
            key: _without_wallclock(child)
            for key, child in value.items()
            if "wallclock" not in str(key).lower()
            and not str(key).lower().endswith("_per_second")
            and key != "timing_hardware"
        }
    if isinstance(value, list):
        return [_without_wallclock(child) for child in value]
    return value


def _first_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle), None)
    if row is None:
        raise ValueError(f"empty CSV: {path}")
    return row


def _build_experiment(name: str, cache_dir: Path):
    print(f"{name}: constructing CPU target/reference objects", flush=True)
    if name == "mb3well_10d":
        return build_e3(
            "cpu",
            basin_cache=str(cache_dir / "basin_map_v2.npz"),
            # This placeholder map is not used for analysis. Frozen positions
            # are classified below by direct gradient flow, avoiding a
            # 600x600x40k CPU precomputation.
            basin_n_grid=16,
            basin_flow_steps=1,
            reference_grid_shape=(256, 256),
            basin_mass_n_quad=32,
        )
    if name == "coupled_phi4":
        return build_e4(
            "cpu",
            basin_cache=str(cache_dir / "basin_map_v2.npz"),
            basin_n_grid=16,
            basin_flow_steps=1,
            # The frozen manifest supplies target moments/masses; this smaller
            # SNIS pool is used only to initialize chains before a full-T burn.
            snis_proposals=20_000,
        )
    return BUILDERS[name]("cpu")


def _direct_flow_labels(name: str, exp, positions_t) -> torch.Tensor:
    """Classify saved positions by accelerated direct basin gradient flow."""
    positions = torch.as_tensor(positions_t, dtype=torch.float64)
    flat = positions.reshape(-1, positions.shape[-1])
    if name == "mb3well_10d":
        z = exp.pot.to_latent(flat)[:, :2]
        grad_fn = mb3_2d_grad
        minima = exp.extras["Z3"]
        lo = torch.tensor([-2.0, -1.3], dtype=torch.float64)
        hi = torch.tensor([1.9, 2.6], dtype=torch.float64)
    elif name == "coupled_phi4":
        z = exp.metric_space(flat)
        grad_fn = phi4_W_grad
        minima = exp.extras["minima_2d"]
        lo = torch.tensor([-4.0, -4.0], dtype=torch.float64)
        hi = torch.tensor([4.0, 4.0], dtype=torch.float64)
    else:
        return exp.labels_fn(flat).reshape(positions.shape[:2])

    # Same normalized gradient-flow time horizon (6.0) as the production basin
    # map, with a 6.67x larger step. Basin attraction is invariant to this
    # reparameterization away from separatrices. A production-step audit on a
    # deterministic subset below fails closed if the acceleration changes any
    # assigned label.
    def flow(points: torch.Tensor, dt: float, steps: int) -> torch.Tensor:
        current = points.clone()
        for _ in range(steps):
            gradient = grad_fn(current)
            norm = gradient.norm(dim=1, keepdim=True)
            current = current - dt * gradient / (1.0 + dt * norm)
            current = torch.clamp(current, lo, hi)
        distance = (
            current.unsqueeze(1) - minima.unsqueeze(0)
        ).square().sum(dim=-1)
        return distance.argmin(dim=1)

    accelerated = flow(z, 1.0e-3, 6_000)
    audit_index = torch.linspace(
        0, max(z.shape[0] - 1, 0), min(512, z.shape[0]),
        dtype=torch.float64,
    ).round().to(torch.long)
    audited = flow(z[audit_index], 1.5e-4, 40_000)
    mismatch = int((accelerated[audit_index] != audited).sum().item())
    if mismatch:
        raise RuntimeError(
            f"{name}: accelerated basin flow disagrees with production flow "
            f"on {mismatch}/{audit_index.numel()} audited positions"
        )
    return accelerated.reshape(positions.shape[:2])


def _rewrite_all_methods(
    path: Path,
    fla_rows: list[dict],
    method_order: list[str],
) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = [row for row in reader if row.get("method") != "FLA"]
    rows.extend(fla_rows)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    rank = {method: index for index, method in enumerate(method_order)}
    rows.sort(key=lambda row: (
        rank.get(str(row.get("method")), len(rank)),
        str(row.get("kind", "")),
        int(float(row.get("index") or -1)),
    ))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def recompute(name: str, *, overwrite: bool) -> dict:
    result_dir = ROOT / "results" / name
    stationarity_dir = result_dir / "stationarity"
    manifest_path = result_dir / "manifest.json"
    output_path = stationarity_dir / "FLA_summary.csv"
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} exists; pass --overwrite to replace it"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = manifest["stationarity"]["protocol"]
    template = _first_csv_row(stationarity_dir / "ULA_summary.csv")
    cache_dir = ROOT / "cache" / "fla_stationarity" / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    exp = _build_experiment(name, cache_dir)

    exp.p_star = torch.tensor(
        manifest["p_star"], dtype=torch.float64, device="cpu"
    )
    n_chains = int(template["n_chains"]) // len(json.loads(template["seeds"]))
    seeds = tuple(json.loads(template["seeds"]))
    n_draws = int(template["n_draws_per_chain"])
    steps_per_draw = int(template["steps_per_draw"])
    burn_in_steps = int(protocol["settling_burn_steps"])
    dt = float(manifest["config"]["dt"])
    chosen_quadrature = manifest.get("quadrature", {}).get("chosen") or {}
    pt_betas = torch.tensor(
        [float(manifest["config"]["beta"])], dtype=torch.float64
    )
    factory = make_sampler_factory(
        exp,
        dt,
        pt_betas,
        n_particles=n_chains,
        score_kwargs=chosen_quadrature,
        reference_init=True,
    )

    reference = manifest["reference"]
    print(
        f"{name}: FLA, {len(seeds)} seeds x {n_chains} chains, "
        f"burn={burn_in_steps}, draws={n_draws}, stride={steps_per_draw}",
        flush=True,
    )
    collected = collect_stationary_trajectories(
        sampler_factory=factory,
        methods=["FLA"],
        seeds=seeds,
        n_draws=n_draws,
        steps_per_draw=steps_per_draw,
        dt=dt,
        labels_fn=exp.labels_fn,
        energy_fn=exp.pot.V,
        cv_fn=exp.metric_space,
        counter_source=exp.pot,
        warmup_steps=int(template.get("warmup_steps") or N_WARMUP_STEPS),
        burn_in_steps=burn_in_steps,
        basin_ids=list(range(len(manifest["p_star"]))),
        cv_names=["x"] if len(reference["reference_cv_means"]) == 1 else [
            f"cv_{index}"
            for index in range(len(reference["reference_cv_means"]))
        ],
        basin_target_probabilities=manifest["p_star"],
        reference_energy_mean=float(reference["reference_energy_mean"]),
        reference_cv_means=reference["reference_cv_means"],
        equilibrium_initialized=False,
        initialization_method=(
            reference["sample_reference_method"]
            + ":reference_draw_then_charged_kernel_settling"
        ),
    )
    result = collected["methods"]["FLA"]
    summary = result["summary"]
    raw = result["raw"]
    if name in ("mb3well_10d", "coupled_phi4"):
        print(f"{name}: assigning saved FLA traces by direct basin flow", flush=True)
        labels_t = _direct_flow_labels(name, exp, raw["positions_t"])
        raw["labels_t"] = labels_t.numpy()
        summary = summarize_stationary_traces(
            raw["labels_t"],
            raw["energy_t"],
            raw["cv_t"],
            raw["trace_times"],
            wallclock_s=summary["wallclock_s"],
            gradient_evals=summary["gradient_evals"],
            potential_evals=summary["potential_evals"],
            score_quadrature_evals=summary["score_quadrature_evals"],
            basin_ids=list(range(len(manifest["p_star"]))),
            cv_names=["x"] if len(reference["reference_cv_means"]) == 1 else [
                f"cv_{index}"
                for index in range(len(reference["reference_cv_means"]))
            ],
            basin_target_probabilities=manifest["p_star"],
            reference_energy_mean=float(reference["reference_energy_mean"]),
            reference_cv_means=reference["reference_cv_means"],
        )
        summary.update({
            "method": "FLA",
            "seeds": list(seeds),
            "warmup_steps": int(
                template.get("warmup_steps") or N_WARMUP_STEPS
            ),
            "burn_in_steps": burn_in_steps,
            "steps_per_draw": steps_per_draw,
            "dt": dt,
            "equilibrium_initialized": False,
            "initialization_method": (
                reference["sample_reference_method"]
                + ":reference_draw_then_charged_kernel_settling"
            ),
        })
    release_summary = _without_wallclock(summary)
    write_stationarity_csv(output_path, release_summary, overwrite=overwrite)
    write_stationarity_npz(
        cache_dir / "FLA_traces.npz",
        trace_times=raw["trace_times"],
        positions_t=raw["positions_t"],
        labels_t=raw["labels_t"],
        energy_t=raw["energy_t"],
        cv_t=raw["cv_t"],
        seed_index=raw["seed_index"],
        chain_index_within_seed=raw["chain_index_within_seed"],
        summary=summary,
        metadata={"experiment": name, "method": "FLA", "device": "cpu"},
        overwrite=True,
    )

    method_order = list(manifest["plot"]["methods"])
    _rewrite_all_methods(
        stationarity_dir / "all_methods_summary.csv",
        flat_summary_rows(release_summary),
        method_order,
    )
    stationarity = manifest["stationarity"]
    stationarity["methods"]["FLA"] = json_safe(release_summary)
    stationarity["collection"]["charged_settling_non_targeting_fla"] = (
        _without_wallclock(collected["collection"])
    )
    settled = stationarity["protocol"].setdefault(
        "charged_settling_non_targeting_methods", []
    )
    if "FLA" not in settled:
        settled.append("FLA")
    excluded = stationarity.get("excluded_non_targeting_methods", [])
    stationarity["excluded_non_targeting_methods"] = [
        method for method in excluded if method != "FLA"
    ]
    diagnostics = stationarity.setdefault(
        "non_targeting_mixing_diagnostics", {}
    )
    diagnostics["FLA"] = {
        "interpretation": "biased-kernel mixing diagnostic; not target ESS",
        "must_report_with_target_bias": True,
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(json_safe(manifest), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    print(
        f"{name}: worst-basin ESS={summary['worst_basin_ess']:.6g}",
        flush=True,
    )
    return {
        "experiment": name,
        "worst_basin_ess": summary["worst_basin_ess"],
        "gradient_evals": summary["gradient_evals"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        default=",".join(BUILDERS),
        help="comma-separated subset of " + ",".join(BUILDERS),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    names = [item.strip() for item in args.experiments.split(",") if item.strip()]
    unknown = sorted(set(names) - set(BUILDERS))
    if not names or unknown:
        parser.error(f"invalid experiments: {unknown or names}")
    report = [recompute(name, overwrite=args.overwrite) for name in names]
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
