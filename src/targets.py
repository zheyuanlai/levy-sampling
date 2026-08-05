"""The counted target oracle: ``value``, ``force``, ``value_and_force``.

Every sampler reaches the potential energy surface only through this facade,
and every call updates a counter at the moment it happens. Nothing infers cost
from ``steps x particles`` after the fact, so a caching change inside MALA or
BAOAB moves the recorded cost by itself.

Raw counters
------------
``n_potential_only``      potential-only evaluations (all of them)
``n_potential_baseline``  the subset explicitly tagged as baseline algorithm work
``n_force_only``          force-only evaluations
``n_value_and_force``     joint evaluations

Derived
-------
``n_force = n_force_only + n_value_and_force``
``n_extra_potential = n_potential_only - n_potential_baseline``

A joint call returns the current point's potential as a by-product, so it is
never also charged as an extra potential evaluation. Whether a potential-only
call is baseline or extra is declared by the caller at the call site; it is
never guessed afterwards.

Structured chord kernels
------------------------
The coupled quartic chain evaluates ``V(x - theta r) - V(x)`` through an exact
moment identity rather than generic potential calls. Those units accumulate in
``n_structured_extra_chord_units`` and are converted to a potential-equivalent
count by the measured FEE calibration; they are never passed off as generic
``V()`` calls.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math

import numpy as np
import torch

from .device import DTYPE, resolve_device
from .potentials import (CoupledQuarticChain, DoubleWell1D, MoG40,
                         MullerBrown3Well10D, MullerBrown3WellLatent2D,
                         MB3_CRITICAL, PHASES, QUARTIC_CHAIN_MINIMA,
                         muller_brown_3well, muller_brown_3well_grad,
                         refined_minima, site_potential, site_potential_grad)

BASELINE = "baseline"
EXTRA = "extra"
_COST_CLASSES = (BASELINE, EXTRA)


@dataclass
class OracleCounters:
    """Point counts of what a sampler actually asked the target for."""

    n_potential_only: int = 0
    n_potential_baseline: int = 0
    n_force_only: int = 0
    n_value_and_force: int = 0
    n_structured_extra_chord_units: int = 0
    #: Chord counts per structured call, kept so the FEE model can charge the
    #: fixed per-particle part of the kernel separately from the per-chord part.
    n_structured_extra_particle_calls: int = 0

    def reset(self) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, 0)

    def snapshot(self) -> dict[str, int]:
        return {name: int(getattr(self, name))
                for name in self.__dataclass_fields__}

    def since(self, baseline: dict[str, int]) -> dict[str, int]:
        return {name: int(getattr(self, name)) - int(baseline.get(name, 0))
                for name in self.__dataclass_fields__}

    @staticmethod
    def derive(raw: dict[str, int]) -> dict[str, int]:
        n_force = int(raw["n_force_only"]) + int(raw["n_value_and_force"])
        n_extra_potential = (int(raw["n_potential_only"])
                             - int(raw["n_potential_baseline"]))
        if n_extra_potential < 0:
            raise ValueError(
                "baseline potential calls exceed total potential-only calls; "
                "a call site tagged its cost class incorrectly")
        return {**raw, "n_force": n_force,
                "n_extra_potential": n_extra_potential}


class Target:
    """A potential energy surface plus its oracle accounting."""

    def __init__(self, potential, beta: float, *, name: str,
                 device=None, dtype: torch.dtype = DTYPE) -> None:
        self.potential = potential
        self.beta = float(beta)
        self.name = str(name)
        self.d = int(potential.d)
        self.device = resolve_device(device)
        self.dtype = dtype
        self.counters = OracleCounters()
        self._counting = True
        #: Filled in by the experiment builders; consumed by references,
        #: metrics, and plotting. Never used to decide oracle cost.
        self.extras: dict = {}

    @property
    def eps(self) -> float:
        return 1.0 / self.beta

    # -- counting control --------------------------------------------------
    @contextmanager
    def no_count(self):
        """Exclude the enclosed calls from the oracle counters.

        Metric evaluation, reference construction, calibration pilots, and
        plotting all run inside this context, so the recorded cost is sampler
        work only.
        """
        previous = self._counting
        self._counting = False
        try:
            yield self
        finally:
            self._counting = previous

    def _count(self, field_name: str, amount: int) -> None:
        if self._counting and amount:
            setattr(self.counters, field_name,
                    getattr(self.counters, field_name) + int(amount))

    @staticmethod
    def _n_points(x: torch.Tensor) -> int:
        return int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1

    # -- the oracle API ----------------------------------------------------
    def value(self, x: torch.Tensor, *, cost_class: str = BASELINE
              ) -> torch.Tensor:
        """``V(x)``, charged as a potential-only evaluation.

        ``cost_class`` must be stated explicitly at every call site: ``baseline``
        for ordinary algorithm work, ``extra`` for the additional evaluations a
        method needs on top of one force per step (the Levy-score chords).
        """
        if cost_class not in _COST_CLASSES:
            raise ValueError(
                f"cost_class must be one of {_COST_CLASSES}, got {cost_class!r}")
        n = self._n_points(x)
        self._count("n_potential_only", n)
        if cost_class == BASELINE:
            self._count("n_potential_baseline", n)
        return self.potential.V(x)

    def force(self, x: torch.Tensor) -> torch.Tensor:
        """``-grad V(x)``, charged as a force-only evaluation."""
        self._count("n_force_only", self._n_points(x))
        return -self.potential.grad_V(x)

    def value_and_force(self, x: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(V(x), -grad V(x))``, charged once as a joint evaluation."""
        self._count("n_value_and_force", self._n_points(x))
        return self.potential.V(x), -self.potential.grad_V(x)

    # -- Levy-score chord energies ----------------------------------------
    def chord_value_delta(self, x: torch.Tensor,
                          shifts: torch.Tensor) -> torch.Tensor:
        """``V(x - r) - V(x)`` for shared shifts ``(J, d)``; extra potential.

        The base point's energy is evaluated once and reused across all shifts,
        so only the ``N * J`` shifted configurations are charged.
        """
        n, j = x.shape[0], shifts.shape[0]
        if self.potential.structured_value_delta:
            self._count("n_structured_extra_chord_units", n * j)
            self._count("n_structured_extra_particle_calls", n)
        else:
            self._count("n_potential_only", n * j)
        return self.potential.value_delta(x, shifts)

    def chord_value_delta_pointwise(self, x: torch.Tensor,
                                    y: torch.Tensor) -> torch.Tensor:
        """``V(y) - V(x)`` for per-particle chord points ``(N, ..., d)``.

        Used by the iid random-atomic estimator, whose chord points differ from
        particle to particle because every particle carries its own bank.
        """
        n = x.shape[0]
        chords = int(y.numel() // (n * x.shape[-1]))
        if self.potential.structured_value_delta:
            self._count("n_structured_extra_chord_units", n * chords)
            self._count("n_structured_extra_particle_calls", n)
        else:
            self._count("n_potential_only", n * chords)
        return self.potential.value_delta_pointwise(x, y)

    # -- convenience -------------------------------------------------------
    def log_target(self, x: torch.Tensor, *, cost_class: str = BASELINE
                   ) -> torch.Tensor:
        return -self.beta * self.value(x, cost_class=cost_class)

    def raw_counters(self) -> dict[str, int]:
        return self.counters.snapshot()

    def derived_counters(self, baseline: dict[str, int] | None = None
                         ) -> dict[str, int]:
        raw = (self.counters.snapshot() if baseline is None
               else self.counters.since(baseline))
        return OracleCounters.derive(raw)


# ============================================================ E1 double well
def build_e1_target(config: dict, device=None) -> Target:
    target_config = config["target"]
    potential = DoubleWell1D()
    target = Target(potential, target_config["beta"], name="double_well",
                    device=device)
    target.extras.update({
        "minima": [-1.0, 1.0],
        "barrier_height": 1.0,
        "kramers_time": DoubleWell1D.kramers_time(target.beta),
        "basin_labels": ["left", "right"],
    })
    return target


# ================================================================= E2 MoG40
def build_e2_target(config: dict, device=None) -> Target:
    target_config = config["target"]
    device = resolve_device(device)
    potential = MoG40(
        beta=target_config["beta"],
        n_components=target_config.get("n_components", 40),
        center_range=tuple(target_config.get("center_range", (-40.0, 40.0))),
        center_seed=target_config.get("center_seed", 0),
        device=device)
    target = Target(potential, target_config["beta"], name="mog40",
                    device=device)
    distances = torch.cdist(potential.mu, potential.mu)
    distances.fill_diagonal_(float("inf"))
    target.extras.update({
        "component_means": potential.mu,
        "n_components": potential.n_components,
        "nearest_neighbour_distance": float(distances.min().item()),
    })
    return target


# ========================================================= E3 Muller-Brown
def build_e3_target(config: dict, device=None) -> Target:
    target_config = config["target"]
    device = resolve_device(device)
    potential = MullerBrown3Well10D(
        sigma_aux=target_config.get("sigma_aux", 0.4),
        embedding_seed=target_config.get("embedding_seed", 12345),
        singular_values=tuple(target_config.get("singular_values", (0.75, 1.45))),
        device=device)
    target = Target(potential, target_config["beta"],
                    name="muller_brown_3well_10d", device=device)
    minima = {key: refined_minima(muller_brown_3well_grad, MB3_CRITICAL,
                                  [key], device)[0]
              for key in ("A", "B", "C")}
    saddles = {key: refined_minima(muller_brown_3well_grad, MB3_CRITICAL,
                                   [key], device)[0]
               for key in ("S_AB", "S_BC")}
    barrier_ab = float((muller_brown_3well(saddles["S_AB"].unsqueeze(0))
                        - muller_brown_3well(minima["B"].unsqueeze(0))).item())
    barrier_bc = float((muller_brown_3well(saddles["S_BC"].unsqueeze(0))
                        - muller_brown_3well(minima["B"].unsqueeze(0))).item())
    target.extras.update({
        "latent_minima": minima,
        "latent_saddles": saddles,
        "latent_minima_stack": torch.stack(
            [minima["A"], minima["B"], minima["C"]]),
        "basin_labels": ["A", "B", "C"],
        "barrier_AB": barrier_ab,
        "barrier_BC": barrier_bc,
        "latent_potential": MullerBrown3WellLatent2D(),
        "to_latent": potential.to_latent,
        "from_latent": potential.from_latent,
        "collective_variable": potential.collective_variable,
    })
    return target


# ================================================== E4 coupled quartic chain
def build_e4_target(config: dict, device=None) -> Target:
    target_config = config["target"]
    device = resolve_device(device)
    potential = CoupledQuarticChain(
        n_sites=target_config.get("n_sites", 12),
        kappa=target_config.get("kappa", 2.5),
        coefficients=target_config.get("site_coefficients"))
    target = Target(potential, target_config["beta"],
                    name="coupled_quartic_chain", device=device)

    grad_fn = (lambda v: site_potential_grad(v, potential.coefficients))
    minima_2d = refined_minima(grad_fn, QUARTIC_CHAIN_MINIMA, PHASES, device)
    coherent_states = (minima_2d.unsqueeze(1)
                       .expand(len(PHASES), potential.n_sites, 2)
                       .reshape(len(PHASES), potential.d).contiguous())
    hessians = _coherent_hessians(potential, coherent_states)
    target.extras.update({
        "phases": list(PHASES),
        "refined_site_minima": minima_2d,
        "coherent_states": coherent_states,
        "coherent_hessians": hessians,
        "coherent_site_energies": site_potential(minima_2d,
                                                 potential.coefficients),
        "n_sites": potential.n_sites,
        "order_parameter": potential.order_parameter,
    })
    return target


def _coherent_hessians(potential, coherent_states: torch.Tensor
                       ) -> torch.Tensor:
    from torch.autograd.functional import hessian as autograd_hessian

    blocks = []
    for k in range(coherent_states.shape[0]):
        hessian = autograd_hessian(
            lambda q: potential.V(q.unsqueeze(0))[0],
            coherent_states[k].clone())
        blocks.append(0.5 * (hessian + hessian.T))
    return torch.stack(blocks)


#: Resolved by ``src.config.load_experiment`` from the ``builder`` key.
TARGET_BUILDERS = {
    "src.targets.build_e1_target": build_e1_target,
    "src.targets.build_e2_target": build_e2_target,
    "src.targets.build_e3_target": build_e3_target,
    "src.targets.build_e4_target": build_e4_target,
}
