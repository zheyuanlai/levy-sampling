"""Per-experiment wiring: jump law, numerical box, and initial condition.

These builders turn a resolved experiment configuration plus a ``Target`` into
the remaining objects a sampler needs. Nothing here hard-codes a parameter that
also lives in YAML: every number is read from the configuration, and anything
derived from geometry (shell half-width, drift cap, box half-width) is computed
here and written back into the resolved configuration so the run manifest
records the value, not just the rule that produced it.
"""
from __future__ import annotations

import math

import torch

from .jumps import AnnulusJumpLaw, ShellJumpLaw
from .potentials import (PHASES, QUARTIC_CHAIN_MINIMA,
                         muller_brown_3well_grad, refined_minima,
                         site_potential_grad)
from .samplers import LatentRectBox, RectBox, UnboundedBox


def _reject_unknown_keys(section: dict, allowed: set[str], *,
                         section_name: str = "", **kwargs) -> None:
    """Fail on a configuration key this builder does not understand.

    Retired options are deleted from the schema rather than defaulted, so a
    stale configuration that still carries one is an error instead of a setting
    that is silently ignored.
    """
    label = section_name or kwargs.get("section", "configuration section")
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(
            f"{label} has unrecognised key(s) {unknown}; known keys are "
            f"{sorted(allowed)}")


def build_components(experiment_id: str, config: dict, target, device) -> dict:
    builder = _COMPONENT_BUILDERS.get(experiment_id)
    if builder is None:
        raise KeyError(f"no component builder registered for {experiment_id!r}")
    return builder(config, target, device)


# ======================================================================== E1
def _build_e1(config: dict, target, device) -> dict:
    law_config = config["jump_law"]
    atoms = torch.as_tensor(law_config["atoms"], dtype=torch.float64,
                            device=device)
    weights = torch.as_tensor(law_config["weights"], dtype=torch.float64,
                              device=device)
    law = ShellJumpLaw(atoms, weights, h=float(law_config["h"]))

    boundary = config["boundary"]
    box = RectBox(boundary["lo"], boundary["hi"], device)

    init = config["initial_condition"]
    center = torch.as_tensor(init["center"], dtype=torch.float64, device=device)
    sigma = float(init["sigma"])

    def init_fn(streams, n_per_seed):
        return center + sigma * streams.randn("init_gen", (n_per_seed, 1))

    return {
        "law": law,
        "box": box,
        "init_fn": init_fn,
        "cp_cap": float(config["taming"]["cp_cap"]),
        "default_cap": float(config["taming"]["default_cap"]),
        "pt_beta_min": float(config["calibration"]["pt"]["beta_min"]),
        "resolved": {
            "jump_law": law.describe(),
            "boundary": box.describe(),
            "taming": {"default_cap": float(config["taming"]["default_cap"]),
                       "cp_cap": float(config["taming"]["cp_cap"])},
        },
    }


# ======================================================================== E2
def _build_e2(config: dict, target, device) -> dict:
    law_config = config["jump_law"]
    law = AnnulusJumpLaw(law_config["inner_radius"], law_config["outer_radius"],
                         device)

    boundary = config["boundary"]
    box = RectBox(boundary["lo"], boundary["hi"], device)

    init = config["initial_condition"]
    component = int(init["component"])
    center = target.extras["component_means"][component]
    sigma = float(init["sigma"])

    def init_fn(streams, n_per_seed):
        return center + sigma * streams.randn("init_gen", (n_per_seed, 2))

    return {
        "law": law,
        "box": box,
        "init_fn": init_fn,
        "cp_cap": float(config["taming"]["cp_cap"]),
        "default_cap": float(config["taming"]["default_cap"]),
        "pt_beta_min": float(config["calibration"]["pt"]["beta_min"]),
        "resolved": {
            "jump_law": law.describe(),
            "boundary": box.describe(),
            "taming": {"default_cap": float(config["taming"]["default_cap"]),
                       "cp_cap": float(config["taming"]["cp_cap"])},
        },
    }


# ======================================================================== E3
def _build_e3(config: dict, target, device) -> dict:
    potential = target.potential
    minima = target.extras["latent_minima"]

    # Relay atoms {+-r_BA, +-r_BC} through the middle hub B. There is no direct
    # A-C atom: that chord overshoots the field-zero region between the wells.
    edges = config["jump_law"]["relay_edges"]
    latent_displacements = []
    for source, destination in edges:
        latent_displacements.append(minima[destination] - minima[source])
    padding = torch.zeros(potential.d - 2, dtype=torch.float64, device=device)
    atoms_latent = torch.stack(
        [torch.cat([displacement, padding])
         for displacement in latent_displacements])
    atoms = potential.from_latent(atoms_latent)
    n_atoms = atoms.shape[0]
    weights = torch.full((n_atoms,), 1.0 / n_atoms, dtype=torch.float64,
                         device=device)
    h = 0.1 * float(atoms.norm(dim=1).min().item())
    law = ShellJumpLaw(atoms, weights, h=h)
    cp_cap = 2.0 * h

    boundary = config["boundary"]
    box = LatentRectBox(boundary["latent_lo"], boundary["latent_hi"], potential)

    init = config["initial_condition"]
    seed_minimum = minima[init["minimum"]]
    sigma = float(init["sigma"])
    dimension = potential.d

    def init_fn(streams, n_per_seed):
        latent = sigma * streams.randn("init_gen", (n_per_seed, dimension))
        latent[:, :2] += seed_minimum
        return potential.from_latent(latent)

    pt_config = config["calibration"]["pt"]
    beta_min = pt_config.get("beta_min")
    if beta_min is None:
        # The hot replica must cross the A<->B barrier: beta_min * b(A<->B) ~ 2.
        beta_min = 2.0 / float(target.extras["barrier_AB"])

    return {
        "law": law,
        "box": box,
        "init_fn": init_fn,
        "cp_cap": cp_cap,
        "default_cap": float(config["taming"]["default_cap"]),
        "pt_beta_min": float(beta_min),
        "resolved": {
            "jump_law": {**law.describe(), "relay_edges": edges,
                         "h": h, "h_rule": "0.1 * min ||r_a||"},
            "boundary": box.describe(),
            "taming": {"default_cap": float(config["taming"]["default_cap"]),
                       "cp_cap": cp_cap, "cp_cap_rule": "2h"},
            "pt_beta_min": float(beta_min),
            "barrier_AB": float(target.extras["barrier_AB"]),
            "barrier_BC": float(target.extras["barrier_BC"]),
        },
    }


# ======================================================================== E4
def _build_e4(config: dict, target, device) -> dict:
    potential = target.potential
    minima_2d = target.extras["refined_site_minima"]
    coherent_states = target.extras["coherent_states"]
    hessians = target.extras["coherent_hessians"]
    n_sites = potential.n_sites

    # Eight edge atoms of the phase square. The two diagonal pairs are dropped:
    # their coherent chords cross the field-zero hilltop at the centre, so a
    # diagonal transition relays through a mixed phase in two hops instead.
    diagonals = ({0, 3}, {1, 2})
    atom_rows, edge_pairs = [], []
    for i in range(len(PHASES)):
        for j in range(len(PHASES)):
            if i == j or {i, j} in diagonals:
                continue
            displacement = minima_2d[j] - minima_2d[i]
            atom_rows.append(displacement.unsqueeze(0)
                             .expand(n_sites, 2).reshape(potential.d))
            edge_pairs.append((PHASES[i], PHASES[j]))
    atoms = torch.stack(atom_rows)
    n_atoms = atoms.shape[0]
    weights = torch.full((n_atoms,), 1.0 / n_atoms, dtype=torch.float64,
                         device=device)
    h = 0.1 * float(atoms.norm(dim=1).min().item())
    _reject_unknown_keys(config["jump_law"], {
        "kind", "drop_diagonals", "weights_uniform", "h_rule", "intensity"},
        section="E4 jump_law")
    law = ShellJumpLaw(atoms, weights, h=h)
    cp_cap = 2.0 * h

    pt_beta_min = float(config["calibration"]["pt"]["beta_min"])
    box_design = sampling_box_design(
        coherent_states, atoms, h, hessians, beta=target.beta,
        pt_beta_min=pt_beta_min,
        tail_probability=float(config["boundary"]["tail_probability"]))
    half_width = box_design["sampling_box_half_width"]
    box = RectBox([-half_width] * potential.d, [half_width] * potential.d,
                  device)

    init = config["initial_condition"]
    phase_index = PHASES.index(init["phase"])
    center = coherent_states[phase_index]
    sigma = float(init["sigma"])
    dimension = potential.d

    def init_fn(streams, n_per_seed):
        return center + sigma * streams.randn("init_gen",
                                              (n_per_seed, dimension))

    return {
        "law": law,
        "box": box,
        "init_fn": init_fn,
        "cp_cap": cp_cap,
        "default_cap": float(config["taming"]["default_cap"]),
        "pt_beta_min": pt_beta_min,
        "edge_pairs": edge_pairs,
        "resolved": {
            "jump_law": {**law.describe(), "edge_pairs": edge_pairs,
                         "h": h, "h_rule": "0.1 * min ||r_a||",
                         "displacement_noise": "none"},
            "boundary": {**box.describe(), "design": box_design},
            "taming": {"default_cap": float(config["taming"]["default_cap"]),
                       "cp_cap": cp_cap, "cp_cap_rule": "2h"},
        },
    }


def sampling_box_design(means: torch.Tensor, atoms: torch.Tensor, h: float,
                        hessians: torch.Tensor, *, beta: float,
                        pt_beta_min: float,
                        tail_probability: float = 1e-8) -> dict:
    """Conservative high-probability numerical box for an unbounded target.

    No finite box is an exact support bound here, so the half-width comes from a
    simultaneous Laplace-mixture component envelope with a declared union-bound
    tail budget, padded by one maximum componentwise displacement of the shell
    law, also covering the hottest parallel-tempering envelope, and rounded up.
    This is an overflow guard, not a truncation of the physical model.
    """
    if (means.ndim != 2 or atoms.ndim != 2 or hessians.ndim != 3
            or means.shape[1] != atoms.shape[1]
            or hessians.shape[1:] != (means.shape[1], means.shape[1])):
        raise ValueError("incompatible means, atoms, or Hessians")
    if not 0.0 < tail_probability < 1.0:
        raise ValueError("tail_probability must lie strictly in (0, 1)")
    if beta <= 0.0 or pt_beta_min <= 0.0:
        raise ValueError("beta and pt_beta_min must be positive")

    atom_norms = atoms.norm(dim=1, keepdim=True)
    units = atoms / atom_norms
    max_jump_reach = float((atoms.abs() + float(h) * units.abs()).amax().item())

    n_modes, dimension = means.shape
    quantile_probability = 1.0 - tail_probability / (2.0 * n_modes * dimension)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=means.dtype, device=means.device),
        torch.tensor(1.0, dtype=means.dtype, device=means.device))
    quantile = float(normal.icdf(torch.tensor(
        quantile_probability, dtype=means.dtype, device=means.device)).item())
    inverse_hessians = torch.linalg.inv(hessians)
    std_target = float(torch.sqrt(torch.diagonal(
        inverse_hessians / float(beta), dim1=-2, dim2=-1)).amax().item())
    std_hottest = float(torch.sqrt(torch.diagonal(
        inverse_hessians / float(pt_beta_min), dim1=-2, dim2=-1)).amax().item())
    phase_extent = float(means.abs().amax().item())
    target_envelope = phase_extent + quantile * std_target
    hottest_envelope = phase_extent + quantile * std_hottest
    required = max(target_envelope + max_jump_reach, hottest_envelope)
    half_width = float(math.ceil(required))
    if half_width < required:
        raise AssertionError("the rounded sampling box is not conservative")
    return {
        "formula": ("ceil(max(B_beta(alpha) + R_inf, B_beta_min(alpha))); "
                    "B_b(alpha) = max|mu| + Phi^{-1}(1 - alpha/(2 K d)) "
                    "max sqrt(diag(H_k^{-1})/b); R_inf = max(|r_ac| + h_a |u_ac|)"),
        "tail_probability_union_bound": float(tail_probability),
        "normal_quantile": quantile,
        "n_phase_modes": int(n_modes),
        "dimension": int(dimension),
        "phase_component_extent": phase_extent,
        "max_component_std_beta": std_target,
        "max_component_std_beta_min": std_hottest,
        "target_phase_envelope_half_width": target_envelope,
        "hottest_pt_envelope_half_width": hottest_envelope,
        "max_componentwise_jump_reach": max_jump_reach,
        "required_half_width_before_rounding": required,
        "sampling_box_half_width": half_width,
        "jump_safe_core_half_width": half_width - max_jump_reach,
    }


_COMPONENT_BUILDERS = {
    "E1": _build_e1,
    "E2": _build_e2,
    "E3": _build_e3,
    "E4": _build_e4,
}
