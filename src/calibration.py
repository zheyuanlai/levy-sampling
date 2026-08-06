"""Per-variant numerical calibration, cached by content hash.

The dependency graph is conditional. A variant runs only the calibrations it
actually needs:

    target
     |- reference                       (shared, built once)
     |- ULA / FLA / ULD variant  -> dt
     |- MALA variant             -> dt + acceptance
     |- PT variant               -> ladder tuning + replica-kernel dt
     |- LSC variant              -> score certificate + quadrature + dt

Canonical and tamed variants of the same method are separate variants and
calibrate separately: different timesteps, different acceptance, and for
parallel tempering a different ladder. A canonical ladder is never reused for a
tamed run.

Results are cached at

    protocols/<target-hash>/<method>/<variant-calibration-hash>.json

so changing a figure, a font, or an unrelated method never triggers a
recalibration, while changing a variant's parameters or the target does.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from . import metrics as sampling_metrics
import numpy as np
import torch

from .factory import build_sampler, sampler_requirements
from .results import json_safe, stable_hash
from .samplers import geometric_ladder

#: How many times the Metropolis acceptance search may double or halve the
#: timestep before giving up. Reaching the band from below costs
#: ``log2(dt_band / initial_dt) + 1`` iterations, so this bounds how far the
#: search can travel from ``initial_dt``. Overridable per method or per
#: experiment under ``calibration.dt.acceptance_search_iterations``.
ACCEPTANCE_SEARCH_ITERATIONS = 12

#: Pilot sizes. Small enough to calibrate quickly, large enough that the
#: comparison is not dominated by Monte Carlo noise. Recorded in the
#: calibration payload so a reader can see what the decision was based on.
PILOT_DEFAULTS = {
    "particles": 512,
    "seeds": 2,
    "time_fraction": 0.25,
    "ladder_pilot_steps": 4000,
    "ladder_burn_fraction": 0.5,
}


class CalibrationError(RuntimeError):
    """A refinement grid was exhausted without a certified choice.

    ``table`` preserves every attempted comparison for the run manifest.
    ``next_candidate`` is a suggestion for extending the grid, not a certified
    setting, and must never be promoted to production automatically.

    ``diagnosis`` says WHY in one line. The distinction that matters is between
    a variant that merely needs a finer grid and one that is unstable at every
    timestep tried -- an untamed Levy-score sampler whose boundary rejection
    rate does not fall as the timestep shrinks is the second kind, and that is a
    result about the method, not a defect in the search.
    """

    def __init__(self, kind: str, table: list[dict], next_candidate=None,
                 diagnosis: str | None = None) -> None:
        self.kind = kind
        self.table = table
        self.next_candidate = next_candidate
        self.diagnosis = diagnosis or _diagnose(table)
        detail = (f"{kind} refinement failed after {len(table)} comparison(s): "
                  f"{self.diagnosis}")
        if next_candidate is not None:
            detail += f"; next unverified candidate is {next_candidate!r}"
        super().__init__(detail)


def _diagnose(table: list[dict]) -> str:
    """One line saying which gate failed, and whether shrinking dt would help."""
    if not table:
        return "no comparisons were attempted"
    persistent = [row for row in table if row.get("stability_problems")]
    if len(persistent) == len(table):
        names = sorted({problem[0] for row in table
                        for problem in row["stability_problems"]})
        first = table[0]["stability_problems"][0]
        last = table[-1]["stability_problems"][0]
        trend = ("and it does not improve as the timestep shrinks"
                 if len(first) > 1 and len(last) > 1
                 and abs(float(last[1] or 0.0)) >= 0.5 * abs(float(first[1] or 0.0))
                 else "though it improves as the timestep shrinks")
        return (f"unstable at every timestep tried ({', '.join(names)}) {trend}")
    keys = sorted({failure["key"] for row in table
                   for failure in row.get("agreement_failures", ())})
    if keys:
        return f"no timestep agreed with its halving on {', '.join(keys)}"
    return "the grid was exhausted"


# --------------------------------------------------------------- cache keys
def calibration_key(context, variant, pilot: dict) -> str:
    """Content hash of everything a calibration decision depends on."""
    method_config = context.method_configs.get(variant.method, {})
    return stable_hash({
        "experiment": context.experiment_id,
        "variant": variant.describe(),
        "protocol": {
            "final_time": context.final_time,
            "initial_dt": context.config["protocol"]["initial_dt"],
        },
        "score": context.config.get("score"),
        "taming": {"cap": context.tame_cap_for(variant)},
        "calibration_rules": context.config.get("calibration"),
        "method_rules": {key: value for key, value in method_config.items()
                         if key not in ("source",)},
        "pilot": pilot,
    })


def calibration_path(context, variant, key: str) -> Path:
    directory = (context.paths.protocols_dir / context.target_hash
                 / variant.method)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{key}.json"


def load_cached(context, variant, key: str) -> dict | None:
    path = calibration_path(context, variant, key)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def store(context, variant, key: str, payload: dict) -> None:
    path = calibration_path(context, variant, key)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True,
                   allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


# ----------------------------------------------------------------- pilots
def _pilot_streams(context, variant, seeds):
    return context.streams_for(variant, seeds=seeds)


def _pilot_summary(context, variant, dt: float, pilot: dict,
                   calibration: dict | None = None) -> dict:
    """Run a short pilot and return scalar summaries plus stability flags.

    Deliberately reference-free: dt selection compares a setting against its own
    halving, so it cannot be contaminated by reference construction and can run
    before the reference exists.
    """
    seeds = tuple(range(int(pilot["seeds"])))
    n_per_seed = int(pilot["particles"])
    n_steps = max(1, int(round(context.final_time * float(pilot["time_fraction"])
                               / float(dt))))
    streams = _pilot_streams(context, variant, seeds)
    requirements = sampler_requirements(context, variant)
    ess_trace: list[torch.Tensor] = []
    trace_start = n_steps // 2
    with context.target.no_count():
        sampler = build_sampler(context, variant, dt=dt, streams=streams,
                                n_per_seed=n_per_seed,
                                calibration=calibration)
        for step in range(n_steps):
            sampler.step()
            if requirements.get("ess") and step >= trace_start:
                positions = sampler.positions()
                coordinate_trace = _summary_coordinate(context, positions)
                ess_trace.append(coordinate_trace.reshape(
                    len(seeds), n_per_seed).mean(dim=1))
        diagnostics = sampler.pop_diagnostics()
        x = sampler.positions()
        finite = torch.isfinite(x).all(dim=-1)
        summary = {
            "n_steps": n_steps,
            "nonfinite_fraction": float((~finite).to(torch.float64).mean().item()),
            "boundary_reject_fraction": float(
                diagnostics.get("boundary_reject_fraction_cumulative", 0.0)),
        }
        if requirements.get("ess"):
            if len(ess_trace) >= 2:
                trace = torch.stack(ess_trace, dim=1)
                temporal_ess = sampling_metrics.effective_sample_size(
                    trace.detach().cpu().numpy())
                temporal_fraction = temporal_ess / float(trace.numel())
            else:
                temporal_ess = math.nan
                temporal_fraction = math.nan
            summary["temporal_ess"] = float(temporal_ess)
            summary["temporal_ess_fraction"] = float(temporal_fraction)
            summary["temporal_ess_draws_per_seed"] = len(ess_trace)
        if not bool(finite.any().item()):
            summary.update({key: math.nan for key in _AGREEMENT_KEYS})
            summary.update({f"{key}_se": math.nan for key in _AGREEMENT_KEYS})
            return {**summary, **_acceptance_fields(diagnostics)}
        clean = x[finite]
        coordinate = _summary_coordinate(context, clean)
        energy = context.target.potential.V(clean)
        summary["n_effective"] = int(clean.shape[0])
        summary.update(_scalar_summary("summary", coordinate))
        summary.update(_scalar_summary("energy", energy))
    return {**summary, **_acceptance_fields(diagnostics)}


#: Bootstrap settings for the pilot standard errors. Frozen so a calibration is
#: reproducible.
_BOOTSTRAP_REPLICATES = 200
_BOOTSTRAP_SEED = 20260805


def _statistics(values: torch.Tensor) -> dict[str, float]:
    """Robust location and scale statistics of a pilot sample.

    Deliberately robust. The uncorrected methods put real mass in the tails --
    raw compound Poisson pushes particles far up the quartic wall -- so a sample
    standard deviation of the energy is dominated by a handful of particles and
    wanders by tens of percent between independent pilots. An interquartile
    range does not, so it can actually detect a discretisation trend.
    """
    quantiles = torch.quantile(
        values, torch.tensor([0.25, 0.5, 0.75], dtype=values.dtype,
                             device=values.device))
    return {
        "mean": float(values.mean().item()),
        "abs_mean": float(values.abs().mean().item()),
        "median": float(quantiles[1].item()),
        "iqr": float((quantiles[2] - quantiles[0]).item()),
    }


def _scalar_summary(prefix: str, values: torch.Tensor) -> dict:
    """Pilot statistics AND their bootstrap standard errors.

    The standard errors matter: a timestep is compared against its own halving
    on an independently drawn pilot, so the difference is discretisation bias
    plus Monte Carlo noise. A nonparametric bootstrap over particles gives valid
    errors for every statistic here, including the quantiles, without assuming
    the sample is anywhere near Gaussian.
    """
    point = _statistics(values)
    n = int(values.numel())
    generator = torch.Generator(device=values.device)
    generator.manual_seed(_BOOTSTRAP_SEED)
    index = torch.randint(0, n, (_BOOTSTRAP_REPLICATES, n),
                          generator=generator, device=values.device)
    resampled = values[index]
    replicate_quantiles = torch.quantile(
        resampled, torch.tensor([0.25, 0.5, 0.75], dtype=values.dtype,
                                device=values.device), dim=1)
    replicates = {
        "mean": resampled.mean(dim=1),
        "abs_mean": resampled.abs().mean(dim=1),
        "median": replicate_quantiles[1],
        "iqr": replicate_quantiles[2] - replicate_quantiles[0],
    }
    out = {}
    for name, value in point.items():
        out[f"{prefix}_{name}"] = value
        out[f"{prefix}_{name}_se"] = float(
            replicates[name].std(unbiased=True).item())
    return out


def _acceptance_fields(diagnostics: dict) -> dict:
    out = {}
    for key, value in diagnostics.items():
        keep = (key in ("mh_accept_fraction_cumulative",
                        "swap_accept_fraction_cumulative",
                        "score_clip_fraction_cumulative",
                        "round_trip_count_cumulative",
                        "round_trip_rate_cumulative")
                or (key.startswith("replica_")
                    and key.endswith("_mh_accept_fraction_cumulative")))
        if keep:
            out[key] = float(value)
    return out


def _summary_coordinate(context, x: torch.Tensor) -> torch.Tensor:
    """A low-dimensional scalar summary used only for dt agreement checks."""
    potential = context.target.potential
    if hasattr(potential, "collective_variable"):
        return potential.collective_variable(x)[:, 0]
    if hasattr(potential, "order_parameter"):
        return potential.order_parameter(x)[:, 0]
    return x[:, 0]


#: Scalars compared between a timestep and its halving.
_AGREEMENT_KEYS = ("summary_mean", "summary_abs_mean", "summary_median",
                   "summary_iqr", "energy_median", "energy_iqr")

#: How many combined standard errors of slack the agreement check allows on top
#: of the relative tolerance. Three keeps the false-rejection rate per scalar
#: near a quarter of a percent.
_NOISE_ALLOWANCE = 3.0


def _values_agree(coarse: float, fine: float, tolerance: float, scale: float,
                  noise: float) -> bool:
    if not (math.isfinite(coarse) and math.isfinite(fine)):
        return False
    return abs(coarse - fine) <= tolerance * max(abs(fine), scale) + noise


def _summaries_agree(coarse: dict, fine: dict, tolerance: float
                     ) -> tuple[bool, list]:
    """Compare a timestep against its halving, allowing for pilot noise.

    Two pilots at different timesteps are independent samples, so their
    difference is discretisation bias plus Monte Carlo noise. Only the bias part
    should gate the choice, so the tolerance is widened by the combined standard
    error of the two estimates.
    """
    failures = []
    # A common scale keeps a summary that happens to sit near zero from
    # demanding absurd absolute agreement.
    scale = max(abs(fine.get("summary_iqr", 0.0)),
                abs(fine.get("energy_iqr", 0.0)), 1e-6)
    for key in _AGREEMENT_KEYS:
        if key not in coarse or key not in fine:
            continue
        combined_se = math.hypot(coarse.get(f"{key}_se", 0.0),
                                 fine.get(f"{key}_se", 0.0))
        noise = _NOISE_ALLOWANCE * combined_se
        if not _values_agree(coarse[key], fine[key], tolerance, scale, noise):
            failures.append({
                "key": key,
                "coarse": round(coarse[key], 8),
                "fine": round(fine[key], 8),
                "difference": round(abs(coarse[key] - fine[key]), 8),
                "allowance": round(tolerance * max(abs(fine[key]), scale)
                                   + noise, 8),
            })
    return (not failures), failures


def _stable(summary: dict, requirements: dict, rules: dict,
            gates: dict | None = None) -> tuple[bool, list]:
    """Hard failure modes that no amount of averaging can excuse."""
    gates = gates or {}
    problems = []
    max_nonfinite = float(gates.get("max_nonfinite_fraction", 0.0))
    max_reject = float(gates.get("max_boundary_reject_fraction", 0.02))
    if summary["nonfinite_fraction"] > max_nonfinite:
        problems.append(("nonfinite_fraction", summary["nonfinite_fraction"]))
    if summary["boundary_reject_fraction"] > max_reject:
        problems.append(("boundary_reject_fraction",
                         summary["boundary_reject_fraction"]))
    if requirements["acceptance"]:
        # Only a COLLAPSED acceptance is an instability. An acceptance above the
        # efficient band means the timestep is smaller than it needs to be, and
        # halving dt can only push acceptance further up, so treating the upper
        # edge as a failure here would make the refinement loop unsatisfiable.
        # The upper edge is handled by the acceptance search instead.
        low = rules.get("target_acceptance", (0.2, 0.9))[0]
        acceptance = summary.get("mh_accept_fraction_cumulative")
        if acceptance is None or acceptance < low:
            problems.append(("mh_acceptance_collapsed", acceptance))
        if requirements.get("replica_acceptance"):
            replica_acceptance = [
                float(value) for key, value in summary.items()
                if (key.startswith("replica_")
                    and key.endswith("_mh_accept_fraction_cumulative"))
            ]
            if (not replica_acceptance
                    or min(replica_acceptance) < float(low)):
                problems.append(("replica_mh_acceptance_collapsed",
                                 replica_acceptance))
    if requirements.get("ess"):
        ess_fraction = summary.get("temporal_ess_fraction")
        minimum_ess = float(rules.get("minimum_ess_fraction", 0.05))
        if (ess_fraction is None or not math.isfinite(float(ess_fraction))
                or float(ess_fraction) < minimum_ess):
            problems.append(("temporal_ess_fraction", ess_fraction))
    return (not problems), problems


# ------------------------------------------------------------ dt refinement
def calibrate_dt(context, variant, *, pilot: dict,
                 calibration: dict | None = None) -> dict:
    """Largest timestep on a dyadic grid whose summaries match its halving.

    Each variant chooses its own timestep. There is no global ``dt_final``:
    canonical and tamed, and every hyperparameter value, calibrate separately.
    """
    rules = dict(context.config["calibration"]["dt"])
    method_rules = (context.method_configs.get(variant.method, {})
                    .get("calibration", {}) or {}).get("dt", {}) or {}
    tolerance = float(method_rules.get("tolerance", rules.get("tolerance", 0.05)))
    max_halvings = int(method_rules.get("max_halvings",
                                        rules.get("max_halvings", 4)))
    acceptance_iterations = int(method_rules.get(
        "acceptance_search_iterations",
        rules.get("acceptance_search_iterations",
                  ACCEPTANCE_SEARCH_ITERATIONS)))
    acceptance_rules = _acceptance_rules(context, variant)
    requirements = sampler_requirements(context, variant)
    stability_gates = dict(
        context.config["calibration"].get("stability", {}) or {})

    dt = float(context.config["protocol"]["initial_dt"])
    table: list[dict] = []
    cache: dict[float, dict] = {}

    def summary_at(value: float) -> dict:
        if value not in cache:
            cache[value] = _pilot_summary(context, variant, value, pilot,
                                          calibration)
        return cache[value]

    acceptance_search = None
    if requirements["acceptance"]:
        dt, acceptance_search = _search_acceptance_band(
            summary_at, dt, acceptance_rules["target_acceptance"],
            max_iterations=acceptance_iterations)

    for _ in range(max_halvings):
        coarse = summary_at(dt)
        fine = summary_at(dt / 2.0)
        stable, stability_problems = _stable(coarse, requirements,
                                             acceptance_rules, stability_gates)
        agrees, failures = _summaries_agree(coarse, fine, tolerance)
        acceptance_ok = True
        if requirements["acceptance"]:
            acceptance = coarse.get("mh_accept_fraction_cumulative")
            low, high = acceptance_rules["target_acceptance"]
            acceptance_ok = bool(
                acceptance is not None and low <= acceptance <= high)
            if not acceptance_ok:
                stability_problems.append(
                    ("mh_acceptance_outside_target_band", acceptance))
        row = {
            "dt": dt,
            "pass": bool(stable and agrees and acceptance_ok),
            "stability_problems": stability_problems,
            "agreement_failures": failures,
            "summary": coarse,
            "summary_half": fine,
        }
        table.append(row)
        if row["pass"]:
            return {"dt": dt, "dt_table": table, "tolerance": tolerance,
                    "criteria": method_rules.get("criteria",
                                                 rules.get("criteria", [])),
                    "acceptance_search": acceptance_search,
                    "pilot": pilot}
        dt = dt / 2.0
    raise CalibrationError("timestep", table, next_candidate=dt)


def _search_acceptance_band(summary_at, dt: float, band, *,
                            max_iterations: int = ACCEPTANCE_SEARCH_ITERATIONS
                            ) -> tuple[float, dict]:
    """Move the timestep until the Metropolis acceptance lands in the band.

    Acceptance falls as the timestep grows, so this search has to be able to go
    up as well as down. Running it before the agreement refinement means the
    refinement starts from an efficient timestep rather than from one that is
    accurate but needlessly small.

    The iteration count bounds how far the timestep can travel from
    ``initial_dt``: reaching the band from below needs
    ``log2(dt_band / initial_dt) + 1`` iterations. It was fixed at 8, which on
    E2 stopped the search at ``dt = 1.28`` with acceptance still 0.93 while the
    band is only reached at ``dt = 5.12``. Iterations are cheap because the loop
    breaks as soon as the band is attained or the timestep oscillates, so an
    unnecessarily generous budget costs nothing when the band is close.
    """
    low, high = float(band[0]), float(band[1])
    history = []
    best_dt, best_distance = dt, math.inf
    for _ in range(max_iterations):
        acceptance = summary_at(dt).get("mh_accept_fraction_cumulative")
        if acceptance is None:
            break
        history.append({"dt": dt, "acceptance": acceptance})
        distance = (0.0 if low <= acceptance <= high
                    else min(abs(acceptance - low), abs(acceptance - high)))
        if distance < best_distance:
            best_dt, best_distance = dt, distance
        if distance == 0.0:
            break
        dt = dt * 2.0 if acceptance > high else dt / 2.0
        if any(abs(entry["dt"] - dt) < 1e-18 for entry in history):
            break                                        # oscillating
    return best_dt, {
        "target_band": [low, high],
        "history": history,
        "selected_dt": best_dt,
        "band_attained": best_distance == 0.0,
        "rationale": "acceptance decreases with dt, so the search moves both ways",
    }


def _acceptance_rules(context, variant) -> dict:
    calibration_config = context.config.get("calibration", {})
    family = context.registry["methods"][variant.method].get("family",
                                                             variant.method)
    if family == "MALA":
        ess_rules = dict((context.method_configs.get(variant.method, {})
                          .get("calibration", {}) or {}).get("ess", {}) or {})
        return {
            "target_acceptance": tuple(
                calibration_config.get("mala", {}).get(
                    "target_acceptance", (0.4, 0.75))),
            "minimum_ess_fraction": float(
                ess_rules.get("minimum_pilot_fraction", 0.05)),
        }
    if family == "PT":
        return {"target_acceptance": tuple(
            calibration_config.get("mala", {}).get("target_acceptance",
                                                   (0.3, 0.9)))}
    return {}


# ----------------------------------------------------------- PT ladder tuning
def tune_pt_ladder(context, variant, dt: float, *, pilot: dict,
                   enforce_mixing: bool = True) -> dict:
    """Pick the replica count so mean swap acceptance lands in the target band.

    Acceptance is measured only on the post-burn-in half of the pilot. Every
    replica starts from the same cold ensemble, so early swaps see near-equal
    potentials across the ladder and accept at a transiently inflated rate; a
    short pilot measured from step zero would silently under-ladder PT.

    Canonical and tamed PT tune independently, because the tamed local kernel
    changes the within-replica mixing and therefore the ladder PT needs.
    """
    rules = context.config["calibration"]["pt"]
    method_ladder = dict(
        context.method_configs.get(variant.method, {}).get("ladder") or {})
    target_band = tuple(method_ladder.get(
        "target_swap_acceptance",
        rules.get("target_swap_acceptance", (0.2, 0.4))))
    beta_min = float(context.pt_beta_min)
    beta_max = float(context.beta)
    k_cap = int(rules.get("k_cap", 64))
    n_replicas = int(rules.get("k_initial", 8))
    pilot_steps = int(rules.get(
        "pilot_steps", pilot.get("ladder_pilot_steps", 4000)))
    burn = int(float(method_ladder.get(
        "burn_fraction", pilot.get(
            "ladder_burn_fraction", PILOT_DEFAULTS["ladder_burn_fraction"])))
               * pilot_steps)
    seeds = tuple(range(int(pilot["seeds"])))
    n_per_seed = int(pilot["particles"])

    history: dict[int, float] = {}
    mixing_history: dict[int, dict] = {}
    for _ in range(10):
        betas = geometric_ladder(beta_max, beta_min, n_replicas, context.device)
        streams = _pilot_streams(context, variant, seeds)
        with context.target.no_count():
            sampler = build_sampler(
                context, variant, dt=dt, streams=streams,
                n_per_seed=n_per_seed,
                calibration={"pt_betas": betas.tolist()})
            for _ in range(burn):
                sampler.step()
            sampler.reset_diagnostics()                    # discard transient
            for _ in range(pilot_steps - burn):
                sampler.step()
            diagnostics = sampler.pop_diagnostics()
        acceptance = float(diagnostics.get("swap_accept_fraction_cumulative", 0.0))
        history[n_replicas] = acceptance
        replica_acceptance = [
            float(diagnostics.get(
                f"replica_{replica}_mh_accept_fraction_cumulative",
                float("nan")))
            for replica in range(n_replicas)
        ]
        mixing_history[n_replicas] = {
            "swap_acceptance": acceptance,
            "replica_acceptance": replica_acceptance,
            "round_trip_count": int(
                diagnostics.get("round_trip_count_cumulative", 0)),
            "round_trip_rate": float(
                diagnostics.get("round_trip_rate_cumulative", 0.0)),
        }
        if acceptance < target_band[0]:
            n_replicas = min(int(math.ceil(n_replicas * 1.5)), k_cap)
        elif acceptance > target_band[1]:
            n_replicas = max(2, n_replicas - 1 if n_replicas <= 4
                             else int(math.floor(n_replicas * 0.75)))
        else:
            break
        if n_replicas in history:                          # oscillating
            break

    def distance(value: float) -> float:
        if target_band[0] <= value <= target_band[1]:
            return 0.0
        return min(abs(value - target_band[0]), abs(value - target_band[1]))

    best = min(history, key=lambda k: distance(history[k]))
    betas = geometric_ladder(beta_max, beta_min, best, context.device)
    minimum_round_trips = int(method_ladder.get("min_round_trips", 1))
    selected_mixing = mixing_history[best]
    band_attained = bool(target_band[0] <= history[best] <= target_band[1])
    mixing_attained = bool(
        selected_mixing["round_trip_count"] >= minimum_round_trips)
    gate_pass = bool(
        band_attained and (mixing_attained or not enforce_mixing))
    gate = {
        "n_replicas": best,
        "swap_acceptance": history[best],
        "target_band": list(target_band),
        "band_attained": band_attained,
        **selected_mixing,
        "minimum_round_trips": minimum_round_trips,
        "mixing_attained": mixing_attained,
        "mixing_gate_enforced": bool(enforce_mixing),
        "pass": gate_pass,
        "stability_problems": ([] if gate_pass else [
            ("swap_acceptance_band", history[best]) if not band_attained
            else ("round_trip_count",
                  selected_mixing["round_trip_count"])
        ]),
    }
    # Only the enforcing pass may fail the run. The first ladder pass is
    # invoked with enforce_mixing=False to gate placement before the local
    # timestep is known, and raising there defeats that: the ladder is always
    # retuned after dt calibration, and it is that second, enforcing pass which
    # decides whether PT is admissible.
    if enforce_mixing and not gate["pass"]:
        raise CalibrationError(
            "pt_ladder", [gate],
            diagnosis=(
                "no PT ladder candidate met the frozen swap acceptance "
                "band and final-timestep round-trip requirement"))
    return {
        "pt_betas": betas.tolist(),
        "pt_tuning": {
            "n_replicas": best,
            "beta_max": beta_max,
            "beta_min": beta_min,
            "ratio": (beta_min / beta_max) ** (1.0 / (best - 1)),
            "swap_acceptance": history[best],
            "target_band": list(target_band),
            "band_attained": band_attained,
            "history": {str(k): v for k, v in history.items()},
            "candidate_mixing_history": {
                str(k): value for k, value in mixing_history.items()},
            "replica_acceptance": selected_mixing["replica_acceptance"],
            "round_trip_count": selected_mixing["round_trip_count"],
            "round_trip_rate": selected_mixing["round_trip_rate"],
            "minimum_round_trips": minimum_round_trips,
            "mixing_attained": mixing_attained,
            "mixing_gate_enforced": bool(enforce_mixing),
            "tuned_for_tame_variant": bool(variant.tame),
            "pilot_steps": pilot_steps,
            "burn_in_steps": burn,
        },
    }


# ------------------------------------------------- score quadrature calibration
def calibrate_quadrature(context, variant, *, n_probe: int = 256,
                         seed: int = 20260805) -> dict:
    """Choose the score quadrature by self-convergence against a finer rule.

    The quadrature depends on the target and jump law but not on the sampler
    timestep, so it is shared across the timesteps of a given target. Only the
    full deterministic estimator needs it; the iid random-atomic estimator has
    no radial rule to refine.
    """
    from .score import DeterministicShellScore

    score_config = context.config["score"]
    base = {"q_theta": int(score_config["q_theta"]),
            "q_rho": int(score_config["q_rho"])}
    law_kwargs = ({"m_phi": int(score_config["m_phi"])}
                  if "m_phi" in score_config else {})
    fine = {"q_theta": base["q_theta"] * 2, "q_rho": base["q_rho"] * 2}
    fine_law_kwargs = ({"m_phi": law_kwargs["m_phi"] * 2}
                       if law_kwargs else {})

    generator = torch.Generator(device=context.device)
    generator.manual_seed(int(seed))
    streams = context.streams_for(variant, seeds=(0,))
    with context.target.no_count():
        probe = context.init_fn(streams, n_probe)
        coarse_score = DeterministicShellScore(
            context.target, context.law, context.intensity, q_theta=base["q_theta"],
            q_rho=base["q_rho"], **law_kwargs)
        fine_score = DeterministicShellScore(
            context.target, context.law, context.intensity, q_theta=fine["q_theta"],
            q_rho=fine["q_rho"], **fine_law_kwargs)
        coarse_M, coarse_v = coarse_score.log_parts(probe)
        fine_M, fine_v = fine_score.log_parts(probe)
        # Compare in log space: the magnitudes span hundreds of decades, so a
        # relative error on the score vector itself would be meaningless.
        #
        # The comparison must be on the SCORE's log-magnitude, log|S| = M +
        # log|v|, not on M alone. ``log_parts`` returns S = -exp(M) v with M
        # only the per-particle maximum exponent, so refining the rule moves
        # weight between M and v with no change to the score: doubling the node
        # count halves every quadrature weight, which drops M by log 2 and
        # doubles |v|. Comparing M alone therefore reported a fixed ~0.693
        # discrepancy for a perfectly converged rule and failed every LSC-CP
        # variant on E1-E3. The tolerances below are unchanged; only the
        # quantity they are applied to is corrected.
        tiny = torch.finfo(coarse_v.dtype).tiny
        coarse_log_magnitude = (
            coarse_M + torch.log(coarse_v.norm(dim=1).clamp_min(tiny)))
        fine_log_magnitude = (
            fine_M + torch.log(fine_v.norm(dim=1).clamp_min(tiny)))
        log_difference = (coarse_log_magnitude - fine_log_magnitude).abs()
        direction_difference = (
            (coarse_v / coarse_v.norm(dim=1, keepdim=True).clamp_min(tiny))
            - (fine_v / fine_v.norm(dim=1, keepdim=True).clamp_min(tiny))
        ).norm(dim=1)
    quadrature_rules = ((context.method_configs.get(variant.method, {})
                         .get("calibration", {}) or {})
                        .get("quadrature", {}) or {})
    tolerance = float(quadrature_rules.get(
        "tolerance",
        context.config["calibration"]["dt"].get("tolerance", 0.05)))
    p95_log = float(torch.quantile(log_difference, 0.95).item())
    p95_multiplier = float(quadrature_rules.get(
        "p95_tolerance_multiplier", 4.0))
    p95_direction = float(torch.quantile(direction_difference, 0.95).item())
    passed = bool(
        float(log_difference.median().item()) <= tolerance
        and float(direction_difference.median().item()) <= tolerance
        and p95_log <= p95_multiplier * tolerance
        and p95_direction <= p95_multiplier * tolerance)
    gate = {
        "setting": base,
        "reference_setting": fine,
        "median_abs_log_magnitude_difference": float(
            log_difference.median().item()),
        "p95_abs_log_magnitude_difference": p95_log,
        "max_abs_log_magnitude_difference": float(
            log_difference.max().item()),
        "median_direction_difference": float(
            direction_difference.median().item()),
        "p95_direction_difference": p95_direction,
        "max_direction_difference": float(direction_difference.max().item()),
        "tolerance": tolerance,
        "p95_tolerance": p95_multiplier * tolerance,
        "n_probe": int(n_probe),
        "shared_across_timesteps": True,
        "passed": passed,
    }
    if not passed:
        raise CalibrationError(
            "quadrature", [gate], next_candidate=fine,
            diagnosis=("score quadrature did not converge against the doubled "
                       "rule at the frozen log-magnitude/direction tolerances"))
    return {
        "quadrature": base,
        "quadrature_check": gate,
    }


def calibrate_score_certificate(context, variant,
                                calibration: dict) -> dict:
    """Certify the score estimator contract required by the method registry.

    Full quadrature receives a numerical self-convergence certificate. The
    random-atomic family receives a structural unbiasedness certificate: iid
    draws from the full law, refreshed each step, with one empirical measure
    shared by score and jump noise.
    """
    entry = context.registry["methods"][variant.method]
    estimator = entry.get("estimator_type")
    if estimator == "deterministic_quadrature":
        check = dict(calibration.get("quadrature_check") or {})
        certificate = {
            "kind": "deterministic_score_quadrature_self_convergence",
            "residual_definition": (
                "max(median absolute log-magnitude difference, "
                "median direction difference) against doubled quadrature"),
            "residual": max(
                float(check["median_abs_log_magnitude_difference"]),
                float(check["median_direction_difference"])),
            "tolerance": float(check["tolerance"]),
            "passed": bool(check.get("passed")),
            "quadrature": calibration.get("quadrature"),
            "reference_setting": check.get("reference_setting"),
        }
    elif estimator == "iid_random_atomic":
        method = context.method_configs[variant.method]
        bank = dict(method.get("bank") or {})
        checks = {
            "iid_from_full_mixture":
                bank.get("sampling") == "iid_from_full_mixture",
            "refreshed_every_step":
                bank.get("refresh_policy") == "every_step",
            "shared_between_score_and_noise":
                bank.get("shared_between_score_and_noise") is True,
            "positive_bank_size": int(variant.parameters.get("A", 0)) > 0,
        }
        certificate = {
            "kind": "iid_empirical_levy_measure_unbiasedness_contract",
            "identity": (
                "E[S_A(x)] equals the full-law score because every bank atom "
                "is iid from rho=nu/Lambda"),
            "bank_size": int(variant.parameters["A"]),
            "checks": checks,
            "passed": all(checks.values()),
            "scope": (
                "structural estimator certificate; Monte Carlo convergence "
                "is covered by the independent score-estimator regression"),
        }
    else:
        raise CalibrationError(
            "score_certificate", [],
            diagnosis=f"unsupported score estimator type {estimator!r}")

    if not certificate["passed"]:
        raise CalibrationError(
            "score_certificate", [certificate],
            diagnosis="the configured score estimator failed its certificate")
    return {"score_certificate": certificate}


# --------------------------------------------------------------- entry point
def calibrate(context, variant, *, refresh: bool = False,
              pilot: dict | None = None) -> dict:
    """Calibrate one variant, reusing a cached record whenever it matches."""
    pilot = {**PILOT_DEFAULTS, **(pilot or {})}
    key = calibration_key(context, variant, pilot)
    if not refresh:
        cached = load_cached(context, variant, key)
        if cached is not None:
            return cached

    requirements = sampler_requirements(context, variant)
    payload: dict = {
        "calibration_hash": key,
        "experiment_id": context.experiment_id,
        "method": variant.method,
        "variant_label": variant.label,
        "variant_hash": variant.hash,
        "tame": variant.tame,
        "tame_cap": context.tame_cap_for(variant),
        "requirements": requirements,
        "pilot": pilot,
    }

    if requirements["quadrature"]:
        payload.update(calibrate_quadrature(context, variant))
    if requirements["score_certificate"]:
        payload.update(calibrate_score_certificate(context, variant,
                                                   payload))

    if requirements["pt_ladder"]:
        # Tune the ladder at the starting timestep, then calibrate the local
        # kernel's timestep with that ladder in place.
        initial_dt = float(context.config["protocol"]["initial_dt"])
        payload.update(tune_pt_ladder(
            context, variant, initial_dt, pilot=pilot, enforce_mixing=False))

    payload.update(calibrate_dt(context, variant, pilot=pilot,
                                calibration=payload))

    if requirements["pt_ladder"]:
        # Always retune at the selected local-kernel timestep and enforce the
        # mixing certificate there. The initial pass deliberately gates only
        # ladder placement, so it cannot reject PT before dt is calibrated.
        payload.update(tune_pt_ladder(
            context, variant, payload["dt"], pilot=pilot,
            enforce_mixing=True))

    store(context, variant, key, payload)
    return payload
