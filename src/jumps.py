"""Jump laws: the normalized displacement measure ``rho = nu / lambda``.

Every law exposes one sampling primitive, ``sample_bank``, which draws
``bank_size`` displacements **iid from the full law** for every particle. For a
finite mixture ``rho = sum_k w_k q_k`` that means drawing a component index from
``w`` and then a displacement from that component, independently for every bank
slot. Taking one fixed draw per component is component stratification, a
different estimator, and is not available here.

All draws go through ``src.rng.EnsembleStreams`` so each seed owns its
generators and per-seed shapes never depend on the batch size.
"""
from __future__ import annotations

import math

import numpy as np
import torch

#: Poisson truncation for full-law compound-Poisson jumps. With lambda*dt <= 0.01
#: the probability of exceeding this is below 1e-20; the sampler still records
#: sampled-versus-applied counts so a cap hit is explicit rather than silent.
K_MAX_JUMPS = 8


def gauss_legendre_01(n: int, device, dtype=torch.float64):
    """Gauss-Legendre nodes on [0,1] with probability weights (summing to 1)."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return (torch.as_tensor(0.5 * (nodes + 1.0), dtype=dtype, device=device),
            torch.as_tensor(0.5 * weights, dtype=dtype, device=device))


def gauss_legendre_m11(n: int, device, dtype=torch.float64):
    """Gauss-Legendre nodes on [-1,1] with probability weights, matching
    ``Unif(-1, 1)``; rescale the nodes by ``h`` for ``rho ~ Unif(-h, h)``."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return (torch.as_tensor(nodes, dtype=dtype, device=device),
            torch.as_tensor(0.5 * weights, dtype=dtype, device=device))


class JumpLaw:
    """Interface shared by every jump law."""

    d: int

    def sample_bank(self, streams, stream_name: str, n_per_seed: int,
                    bank_size: int) -> torch.Tensor:
        """``(S * n_per_seed, bank_size, d)`` iid draws from the full law."""
        raise NotImplementedError

    def quadrature_shifts(self, **kwargs):
        """``(shifts (J, d), log_weights (J,))`` for the deterministic score."""
        raise NotImplementedError

    def max_reach(self) -> float:
        raise NotImplementedError

    def describe(self) -> dict:
        raise NotImplementedError


class ShellJumpLaw(JumpLaw):
    """Finite-atom shell law.

    ``r = r_a + rho u_a`` with ``a ~ w``, ``rho ~ Unif(-h_a, h_a)`` and
    ``u_a = r_a / ||r_a||``. ``h`` may be scalar or per atom.
    """

    def __init__(self, atoms: torch.Tensor, weights, h) -> None:
        if (not isinstance(atoms, torch.Tensor) or atoms.ndim != 2
                or atoms.shape[0] < 1 or atoms.shape[1] < 1
                or not atoms.is_floating_point()):
            raise ValueError("atoms must be a nonempty rank-2 floating tensor")
        if not bool(torch.isfinite(atoms).all().item()):
            raise ValueError("atoms must be finite")
        norms = atoms.norm(dim=1, keepdim=True)
        if (not bool(torch.isfinite(norms).all().item())
                or bool((norms <= 0).any().item())):
            raise ValueError("every shell atom must have finite nonzero norm")
        weights_t = torch.as_tensor(weights, dtype=atoms.dtype,
                                    device=atoms.device)
        if weights_t.shape != (atoms.shape[0],):
            raise ValueError("weights must have shape (number of atoms,)")
        total = weights_t.sum()
        if (not bool(torch.isfinite(weights_t).all().item())
                or bool((weights_t < 0).any().item())
                or float(total.item()) <= 0.0):
            raise ValueError(
                "weights must be finite, nonnegative, and have positive sum")
        h_t = torch.as_tensor(h, dtype=atoms.dtype, device=atoms.device)
        try:
            h_t = h_t.expand(atoms.shape[0]).clone()
        except RuntimeError as exc:
            raise ValueError("h must be scalar or one entry per atom") from exc
        if (not bool(torch.isfinite(h_t).all().item())
                or bool((h_t < 0).any().item())):
            raise ValueError("shell half-width h must be finite and nonnegative")

        self.atoms = atoms
        self.weights = weights_t / total
        self.d = atoms.shape[1]
        self.n_atoms = atoms.shape[0]
        self.h = h_t
        self.units = atoms / norms

    def sample_bank(self, streams, stream_name: str, n_per_seed: int,
                    bank_size: int) -> torch.Tensor:
        n_per_seed, bank_size = int(n_per_seed), int(bank_size)
        # One component index per (particle, bank slot), drawn from the full
        # mixture weights -- not one fixed draw per component.
        index = streams.categorical(stream_name, self.weights,
                                    n_per_seed * bank_size)
        index = index.reshape(-1, bank_size)
        radial = ((streams.rand(stream_name, (n_per_seed, bank_size)) * 2.0 - 1.0)
                  * self.h[index])
        return self.atoms[index] + radial.unsqueeze(-1) * self.units[index]

    def quadrature_shifts(self, q_rho: int, **_):
        """All ``r_{a,q} = r_a + rho_q h_a u_a`` with probability weights.

        A normalized quadrature rule for the declared shell law, not the
        continuous sampling measure itself.
        """
        rho, w_rho = gauss_legendre_m11(q_rho, self.atoms.device,
                                        self.atoms.dtype)
        radial = self.h.view(-1, 1) * rho.view(1, -1)
        shifts = (self.atoms.unsqueeze(1)
                  + radial.unsqueeze(-1) * self.units.unsqueeze(1))
        weights = self.weights.unsqueeze(1) * w_rho.unsqueeze(0)
        return shifts.reshape(-1, self.d), torch.log(weights.reshape(-1))

    def max_reach(self) -> float:
        return float((self.atoms.norm(dim=1) + self.h).max())

    def describe(self) -> dict:
        return {
            "kind": "shell",
            "n_atoms": int(self.n_atoms),
            "dimension": int(self.d),
            "atom_norms": self.atoms.norm(dim=1).tolist(),
            "weights": self.weights.tolist(),
            "h": self.h.tolist(),
            "max_reach": self.max_reach(),
        }


class AnnulusJumpLaw(JumpLaw):
    """``r = rho u_phi`` with ``rho ~ Unif[a, b]`` and ``phi ~ Unif[0, 2 pi)``.

    Deliberately generic: it encodes a plausible mode-spacing scale and no mode
    locations at all.
    """

    d = 2

    def __init__(self, inner_radius: float, outer_radius: float, device) -> None:
        self.a = float(inner_radius)
        self.b = float(outer_radius)
        if not 0.0 <= self.a < self.b:
            raise ValueError("require 0 <= inner_radius < outer_radius")
        self.device = device

    def sample_bank(self, streams, stream_name: str, n_per_seed: int,
                    bank_size: int) -> torch.Tensor:
        n_per_seed, bank_size = int(n_per_seed), int(bank_size)
        radius = self.a + (self.b - self.a) * streams.rand(
            stream_name, (n_per_seed, bank_size))
        phi = 2.0 * math.pi * streams.rand(stream_name, (n_per_seed, bank_size))
        return torch.stack([radius * torch.cos(phi),
                            radius * torch.sin(phi)], dim=-1)

    def quadrature_shifts(self, q_rho: int, m_phi: int = 64, **_):
        """Tensor quadrature of the annulus: ``m_phi`` trapezoid directions by
        ``q_rho`` Gauss-Legendre radii, with probability weights."""
        nodes, weights = np.polynomial.legendre.leggauss(q_rho)
        radius = torch.as_tensor(0.5 * (self.b - self.a) * (nodes + 1.0) + self.a,
                                 dtype=torch.float64, device=self.device)
        w_radius = torch.as_tensor(0.5 * weights, dtype=torch.float64,
                                   device=self.device)
        phi = (torch.arange(m_phi, dtype=torch.float64, device=self.device)
               * (2.0 * math.pi / m_phi))
        directions = torch.stack([torch.cos(phi), torch.sin(phi)], dim=1)
        shifts = radius.view(-1, 1, 1) * directions.view(1, -1, 2)
        weight = (w_radius / m_phi).view(-1, 1).expand(-1, m_phi)
        return shifts.reshape(-1, 2), torch.log(weight.reshape(-1))

    def max_reach(self) -> float:
        return self.b

    def describe(self) -> dict:
        return {
            "kind": "annulus",
            "dimension": 2,
            "inner_radius": self.a,
            "outer_radius": self.b,
            "max_reach": self.max_reach(),
        }


def full_law_jump_increment(law: JumpLaw, streams, n_per_seed: int,
                            intensity: float, dt: float, *,
                            k_max: int = K_MAX_JUMPS):
    """Compound-Poisson increment for the full law.

    Returns ``(increment, applied_counts, sampled_counts)``. ``N ~ Poisson(
    lambda dt)`` displacements are drawn iid from ``rho``; the fixed unrolled
    loop applies at most ``k_max`` of them and reports both counts so a cap hit
    is visible instead of silently truncated.
    """
    rates = torch.full((int(n_per_seed),), float(intensity) * float(dt),
                       device=streams.device, dtype=streams.dtype)
    sampled = streams.poisson("poisson_gen", rates)
    bank = law.sample_bank(streams, "jump_bank_gen", n_per_seed, k_max)
    order = torch.arange(k_max, device=bank.device,
                         dtype=sampled.dtype).view(1, -1)
    mask = (sampled.unsqueeze(1) > order).to(bank.dtype)
    increment = (mask.unsqueeze(-1) * bank).sum(dim=1)
    applied = torch.clamp(sampled, max=float(k_max))
    return increment, applied, sampled


def iid_bank_jump_increment(bank: torch.Tensor, streams, n_per_seed: int,
                            intensity: float, dt: float):
    """Compound-Poisson increment driven by an already-drawn iid bank.

    ``N_j ~ Poisson(lambda dt / A)`` and ``Delta L = sum_j N_j R_j``, so the
    total count is ``Poisson(lambda dt)`` exactly and the same bank that built
    the score builds the noise.
    """
    bank_size = int(bank.shape[1])
    rate = float(intensity) * float(dt) / bank_size
    rates = torch.full((int(n_per_seed), bank_size), rate,
                       device=streams.device, dtype=streams.dtype)
    counts = streams.poisson("poisson_gen", rates)
    increment = (counts.unsqueeze(-1) * bank).sum(dim=1)
    return increment, counts
