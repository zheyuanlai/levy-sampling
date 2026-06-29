"""Core samplers: ULA, MALA, FLMC, LSB-MC.

All samplers share a small interface so the experiment driver can treat them
uniformly and account for compute fairly:

    sampler = Sampler(target, dt, ...)
    state   = sampler.init_state(n_seeds, n_particles, seed, device)
    state, diag = sampler.step(state, gen)          # diag: {'acc': ..., ...}
    samples = sampler.final_samples(state)          # (n_seeds, n_particles, dim)

Each sampler maintains ``grad_evals`` (number of batch ∇V evaluations) and
``pot_evals`` (number of batch V evaluations, with the LSB-MC Lévy-score charged
its full quadrature cost) so that compute can be compared fairly across methods
in addition to wall-clock time.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

from .targets import Target
from .utils import (flmc_c_alpha, randn_like_g, sample_symmetric_alpha_stable,
                    sanitize, tame)


class Sampler:
    name = "base"
    uses_temps = False

    def __init__(self, target: Target, dt: float, tame_cap: float = 1.0, **kw):
        self.target = target
        self.dt = float(dt)
        self.sigma = target.sigma
        self.clip = target.state_clip
        self.tame_cap = float(tame_cap)
        self.grad_evals = 0
        self.pot_evals = 0

    def init_state(self, n_seeds: int, n_particles: int, seed: int, device):
        return self.target.init_particles(n_seeds, n_particles, seed, device)

    def step(self, x: torch.Tensor, gen: torch.Generator) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError

    def final_samples(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ULA(Sampler):
    name = "ULA"

    def step(self, x, gen):
        f = self.target.force(x)
        self.grad_evals += 1
        x = (x + tame(f, self.dt, self.tame_cap)
             + self.sigma * math.sqrt(self.dt) * randn_like_g(x, gen))
        return sanitize(x, self.clip), {}


class MALA(Sampler):
    name = "MALA"

    def step(self, x, gen):
        dt = self.dt
        f = self.target.force(x)
        xi = randn_like_g(x, gen)
        y = x + 0.5 * dt * f + math.sqrt(dt) * xi
        y = sanitize(y, self.clip)
        fy = self.target.force(y)
        self.grad_evals += 2
        lq_fwd = -((y - x - 0.5 * dt * f) ** 2).sum(-1) / (2 * dt)
        lq_rev = -((x - y - 0.5 * dt * fy) ** 2).sum(-1) / (2 * dt)
        log_a = (self.target.log_prob(y) + lq_rev) - (self.target.log_prob(x) + lq_fwd)
        self.pot_evals += 2
        u = torch.rand(log_a.shape, generator=gen, device=x.device, dtype=x.dtype)
        acc = torch.log(u.clamp_min(1e-30)) < log_a
        x_new = torch.where(acc.unsqueeze(-1), y, x)
        return x_new, {"acc": acc.float().mean().item()}


class FLMC(Sampler):
    name = "FLMC"

    def __init__(self, target, dt, alpha: float = 1.5, xi_clip: float = 100.0, **kw):
        super().__init__(target, dt, **kw)
        self.alpha = alpha
        self.c_alpha = flmc_c_alpha(alpha)
        self.xi_clip = xi_clip

    def step(self, x, gen):
        drift = self.c_alpha * self.target.force(x)
        self.grad_evals += 1
        xi = sample_symmetric_alpha_stable(x.shape, self.alpha, gen, x.device, x.dtype)
        xi = xi.clamp(-self.xi_clip, self.xi_clip)
        x = x + tame(drift, self.dt, self.tame_cap) + (self.dt ** (1.0 / self.alpha)) * xi
        return sanitize(x, self.clip), {}


class LSBMC(Sampler):
    """Lévy-Score-Based Monte Carlo with a compound-Poisson finite jump bank.

    The drift is  -∇V + S_L  where S_L is the Lévy-score correction supplied by
    the target (1-D grid for ManyWell, generic finite-bank quadrature otherwise).
    The score is what preserves the invariant distribution under the jumps, so it
    is never removed.
    """
    name = "LSBMC"

    def step(self, x, gen):
        f = self.target.force(x)
        self.grad_evals += 1
        s = self.target.levy_score(x)
        # charge the Lévy-score quadrature its batch-equivalent potential cost
        self.pot_evals += self._levy_cost()
        drift = f + s
        x = (x + tame(drift, self.dt, self.tame_cap)
             + self.sigma * math.sqrt(self.dt) * randn_like_g(x, gen))
        x = self.target.apply_jumps(x, gen, self.dt)
        return sanitize(x, self.clip), {}

    def _levy_cost(self) -> int:
        # ManyWell: precomputed grid lookup (negligible). Generic bank: n_theta * M.
        t = self.target
        if hasattr(t, "_theta_nodes") and hasattr(t, "_jumps"):
            return int(t._theta_nodes.shape[0] * t._jumps.shape[0])
        return 1


SAMPLER_REGISTRY = {
    "ULA": ULA,
    "MALA": MALA,
    "FLMC": FLMC,
    "LSBMC": LSBMC,
}
