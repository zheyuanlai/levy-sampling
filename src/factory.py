"""Construct a sampler for one variant.

Shared by the calibration pilots and the production runs so the two can never
drift apart: a variant is calibrated with exactly the object it later runs.
"""
from __future__ import annotations

import torch

from .samplers import (BAOABSampler, CompoundPoissonSampler, FLASampler,
                       MALASampler, ParallelTemperingSampler, ULASampler,
                       geometric_ladder)
from .score import DeterministicShellScore, IIDRandomAtomicScore

_SAMPLER_CLASSES = {
    "src.samplers.ULASampler": ULASampler,
    "src.samplers.MALASampler": MALASampler,
    "src.samplers.FLASampler": FLASampler,
    "src.samplers.BAOABSampler": BAOABSampler,
    "src.samplers.ParallelTemperingSampler": ParallelTemperingSampler,
    "src.samplers.CompoundPoissonSampler": CompoundPoissonSampler,
}


def build_score(context, variant, calibration: dict | None = None):
    """The score object for a variant, or ``None`` for the uncorrected methods.

    The score quadrature may be shared across timesteps at a fixed target and
    jump law, so it is taken from the calibration record when one is supplied.
    """
    entry = context.registry["methods"][variant.method]
    if not entry.get("uses_score", False):
        return None
    score_config = dict(context.config["score"])
    if calibration:
        score_config.update(calibration.get("quadrature", {}) or {})
    q_theta = int(score_config["q_theta"])
    m_max = float(score_config.get("m_max", 600.0))
    estimator = entry.get("estimator_type")
    if estimator == "iid_random_atomic":
        return IIDRandomAtomicScore(
            context.target, context.law, context.intensity,
            bank_size=int(variant.parameters["A"]), q_theta=q_theta,
            m_max=m_max)
    if estimator == "deterministic_quadrature":
        law_kwargs = {}
        if "m_phi" in score_config:
            law_kwargs["m_phi"] = int(score_config["m_phi"])
        return DeterministicShellScore(
            context.target, context.law, context.intensity, q_theta=q_theta,
            q_rho=int(score_config["q_rho"]), m_max=m_max, **law_kwargs)
    raise ValueError(
        f"method {variant.method!r} declares an unknown estimator type "
        f"{estimator!r}")


def pt_ladder(context, calibration: dict | None, dt: float) -> torch.Tensor:
    """The parallel-tempering ladder for a variant.

    Canonical and tamed PT tune separately, so this always comes from that
    variant's own calibration record; there is no shared default ladder.
    """
    if calibration and calibration.get("pt_betas"):
        return torch.as_tensor(calibration["pt_betas"], dtype=torch.float64,
                               device=context.device)
    raise ValueError(
        "parallel tempering needs a tuned ladder; run calibration for this "
        "variant first")


def build_sampler(context, variant, *, dt: float, streams, n_per_seed: int,
                  x0=None, calibration: dict | None = None):
    """Instantiate the sampler for ``variant`` at timestep ``dt``."""
    entry = context.registry["methods"][variant.method]
    sampler_class = _SAMPLER_CLASSES[entry["implementation"]]
    if x0 is None:
        x0 = context.init_fn(streams, n_per_seed)
    common = {
        "target": context.target,
        "streams": streams,
        "x0": x0,
        "n_per_seed": n_per_seed,
        "dt": float(dt),
        "tame_cap": context.tame_cap_for(variant),
        "box": context.box,
    }
    parameters = {key: value for key, value in variant.parameters.items()
                  if key != "tame"}
    family = entry.get("family", variant.method)

    if family in ("ULA", "MALA"):
        return sampler_class(**common)
    if family == "FLA":
        return sampler_class(**common, alpha=float(parameters["alpha"]))
    if family == "ULD":
        return sampler_class(**common, gamma=float(parameters.get("gamma", 1.0)))
    if family == "PT":
        return sampler_class(**common, betas=pt_ladder(context, calibration, dt),
                             n_swap=int(parameters.get("n_swap", 10)))
    if family in ("CP", "LSC-CP", "LSC-CP-RA"):
        score = build_score(context, variant, calibration)
        jump_mode = "iid_bank" if family == "LSC-CP-RA" else "full_law"
        return sampler_class(
            **common, law=context.law, intensity=context.intensity,
            score=score, name=variant.method, jump_mode=jump_mode,
            bank_size=int(parameters.get("A", 1)))
    raise ValueError(f"no sampler wiring for family {family!r}")


def sampler_requirements(context, variant) -> dict:
    """Which calibrations this variant actually needs.

    The dependency graph is conditional: a parallel-tempering ladder is tuned
    only when parallel tempering runs, a score certificate and quadrature
    calibration only when a corrected method runs, and a full quadrature
    calibration only for the deterministic full-quadrature estimator.
    """
    entry = context.registry["methods"][variant.method]
    family = entry.get("family", variant.method)
    return {
        "dt": True,
        "pt_ladder": family == "PT",
        "acceptance": family in ("MALA", "PT"),
        "score_certificate": bool(entry.get("uses_score", False)),
        "quadrature": entry.get("estimator_type") == "deterministic_quadrature",
        "ess": family == "MALA",
        "replica_acceptance": family == "PT",
        "round_trip": family == "PT",
    }
