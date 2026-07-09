"""Jump-law utilities for Phase 16A experiments.

The functions here implement finite-activity compound Poisson jumps.  They do
not run simulations by themselves; notebooks choose the model, time step, and
diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AtomJump:
    """Finite atomic jump law with total intensity ``lam``."""

    atoms: np.ndarray
    weights: np.ndarray
    lam: float
    name: str = "atom"
    edge_count: int | None = None
    jump_type: str = "atom"

    def __post_init__(self) -> None:
        self.atoms = np.asarray(self.atoms, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        if self.atoms.ndim != 2:
            raise ValueError("atoms must have shape (n_atoms, dim)")
        if len(self.atoms) != len(self.weights):
            raise ValueError("weights must match atoms")
        total = float(np.sum(self.weights))
        if total <= 0:
            raise ValueError("weights must have positive sum")
        self.weights = self.weights / total
        if self.edge_count is None:
            self.edge_count = len(self.atoms)


@dataclass
class EdgeShellJump:
    """Shell-thickened edge jump law.

    For each center r0, samples r0 + rho * r0 / |r0| with
    rho uniformly distributed on [-h_shell, h_shell].
    """

    centers: np.ndarray
    weights: np.ndarray
    lam: float
    h_shell: float
    name: str = "edge-shell"
    edge_count: int | None = None
    jump_type: str = "shell"

    def __post_init__(self) -> None:
        self.centers = np.asarray(self.centers, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        if self.centers.ndim != 2:
            raise ValueError("centers must have shape (n_centers, dim)")
        if len(self.centers) != len(self.weights):
            raise ValueError("weights must match centers")
        total = float(np.sum(self.weights))
        if total <= 0:
            raise ValueError("weights must have positive sum")
        self.weights = self.weights / total
        self.h_shell = float(self.h_shell)
        if self.h_shell < 0:
            raise ValueError("h_shell must be nonnegative")
        if self.edge_count is None:
            self.edge_count = len(self.centers)

    @property
    def atoms(self) -> np.ndarray:
        """Compatibility alias for code that inventories jump centers."""

        return self.centers

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        ids = rng.choice(len(self.weights), size=size, p=self.weights)
        r0 = self.centers[ids]
        norm = np.linalg.norm(r0, axis=1, keepdims=True)
        u = r0 / np.maximum(norm, 1e-14)
        rho = rng.uniform(-self.h_shell, self.h_shell, size=(size, 1))
        return r0 + rho * u


def apply_compound_poisson_atom_jumps(z, jump, rng, dt):
    """Apply compound Poisson atom jumps to a batch of samples."""

    z = np.asarray(z, dtype=float)
    n = z.shape[0]

    counts = rng.poisson(jump.lam * dt, size=n)
    total = int(counts.sum())

    if total == 0:
        return z, counts

    owners = np.repeat(np.arange(n), counts)
    atom_ids = rng.choice(len(jump.atoms), size=total, p=jump.weights)

    out = z.copy()
    np.add.at(out, owners, jump.atoms[atom_ids])

    return out, counts


def apply_compound_poisson_shell_jumps(z, jump, rng, dt):
    """Apply compound Poisson shell-thickened edge jumps."""

    z = np.asarray(z, dtype=float)
    n = z.shape[0]

    counts = rng.poisson(jump.lam * dt, size=n)
    total = int(counts.sum())

    if total == 0:
        return z, counts

    owners = np.repeat(np.arange(n), counts)
    increments = jump.sample(rng, total)

    out = z.copy()
    np.add.at(out, owners, increments)

    return out, counts


def apply_bernoulli_atom_jumps_for_debug(z, jump, rng, dt):
    """Old one-jump-per-step approximation retained only for debugging."""

    z = np.asarray(z, dtype=float)
    event = rng.random(z.shape[0]) < min(1.0, jump.lam * dt)
    n_events = int(np.sum(event))
    if n_events == 0:
        return z, np.zeros(z.shape[0], dtype=int)
    atom_ids = rng.choice(len(jump.atoms), size=n_events, p=jump.weights)
    out = z.copy()
    out[event] = out[event] + jump.atoms[atom_ids]
    counts = np.zeros(z.shape[0], dtype=int)
    counts[event] = 1
    return out, counts
