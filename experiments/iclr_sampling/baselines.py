"""New baselines: Parallel Tempering, HMC, Underdamped Langevin (BAOAB).

These follow the same ``Sampler`` interface as ``samplers.py`` (init_state /
step / final_samples) and maintain ``grad_evals`` / ``pot_evals`` counters so
compute is reported fairly (PT runs a kernel per replica; HMC runs L leapfrog
gradients per proposal).
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch

from .samplers import Sampler
from .utils import sanitize


class ParallelTempering(Sampler):
    """Replica exchange with a geometric inverse-temperature ladder.

    State shape: (n_seeds, n_temps, n_particles, dim).  Within each replica we
    run a tempered MALA kernel targeting exp(-beta_l V); adjacent replicas
    attempt swaps every ``swap_interval`` steps.  Final samples are the cold
    (beta=1) chain.  Vectorised across seeds, temperatures and particles.
    """
    name = "PT"
    uses_temps = True

    def __init__(self, target, dt, n_temps: int = 6, beta_min: float = 0.05,
                 swap_interval: int = 5, **kw):
        super().__init__(target, dt, **kw)
        self.n_temps = int(n_temps)
        self.beta_min = float(beta_min)
        self.swap_interval = int(swap_interval)
        betas = np.geomspace(1.0, beta_min, n_temps).astype(np.float32)
        self._betas_np = betas
        self.betas = None              # set on device in init_state
        self._t = 0
        self._swap_parity = 0
        self.swap_attempts = 0
        self.swap_accepts = 0

    def init_state(self, n_seeds, n_particles, seed, device):
        base = self.target.init_particles(n_seeds, n_particles, seed, device)  # (S,N,d)
        x = base.unsqueeze(1).repeat(1, self.n_temps, 1, 1).contiguous()       # (S,T,N,d)
        self.betas = torch.tensor(self._betas_np, device=device).view(1, -1, 1, 1)
        self._t = 0
        self.swap_attempts = 0
        self.swap_accepts = 0
        return x

    def step(self, x, gen):
        dt = self.dt
        beta = self.betas                                   # (1,T,1,1)
        f = self.target.force(x)                            # (S,T,N,d)
        xi = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
        bf = beta * f
        y = x + 0.5 * dt * bf + math.sqrt(dt) * xi
        y = sanitize(y, self.clip)
        fy = self.target.force(y)
        self.grad_evals += 2 * self.n_temps
        lq_fwd = -((y - x - 0.5 * dt * bf) ** 2).sum(-1) / (2 * dt)
        lq_rev = -((x - y - 0.5 * dt * beta * fy) ** 2).sum(-1) / (2 * dt)
        # tempered log target: beta * log_prob = -beta V
        bV_x = self.betas.squeeze(-1) * self.target.potential(x)
        bV_y = self.betas.squeeze(-1) * self.target.potential(y)
        self.pot_evals += 2 * self.n_temps
        log_a = (-bV_y + lq_rev) - (-bV_x + lq_fwd)
        u = torch.rand(log_a.shape, generator=gen, device=x.device, dtype=x.dtype)
        acc = torch.log(u.clamp_min(1e-30)) < log_a         # (S,T,N)
        x = torch.where(acc.unsqueeze(-1), y, x)
        diag = {"acc": acc.float().mean().item()}

        self._t += 1
        if self._t % self.swap_interval == 0 and self.n_temps > 1:
            x, sa = self._swap(x, gen)
            diag["swap_acc"] = sa
        return x, diag

    def _swap(self, x, gen):
        V = self.target.potential(x)                        # (S,T,N)
        self.pot_evals += self.n_temps
        betas = self.betas.view(-1)                         # (T,)
        parity = self._swap_parity
        self._swap_parity ^= 1
        ls = list(range(parity, self.n_temps - 1, 2))
        if not ls:
            return x, float("nan")
        attempts = 0
        accepts = 0
        for l in ls:
            bi, bj = betas[l], betas[l + 1]
            Vi, Vj = V[:, l, :], V[:, l + 1, :]             # (S,N)
            log_a = (bi - bj) * (Vi - Vj)
            u = torch.rand(log_a.shape, generator=gen, device=x.device, dtype=x.dtype)
            sw = torch.log(u.clamp_min(1e-30)) < log_a      # (S,N)
            mask = sw.unsqueeze(-1)
            xi_l = x[:, l, :, :]
            xi_l1 = x[:, l + 1, :, :]
            new_l = torch.where(mask, xi_l1, xi_l)
            new_l1 = torch.where(mask, xi_l, xi_l1)
            x[:, l, :, :] = new_l
            x[:, l + 1, :, :] = new_l1
            # update V cache for swapped configs (so chained pairs are consistent)
            Vl, Vl1 = V[:, l, :], V[:, l + 1, :]
            V[:, l, :] = torch.where(sw, Vl1, Vl)
            V[:, l + 1, :] = torch.where(sw, Vl, Vl1)
            attempts += sw.numel()
            accepts += int(sw.sum().item())
        self.swap_attempts += attempts
        self.swap_accepts += accepts
        return x, (accepts / max(attempts, 1))

    def final_samples(self, x):
        return x[:, 0, :, :]            # cold (beta=1) chain


class HMC(Sampler):
    """Hamiltonian Monte Carlo with leapfrog integration and MH correction.

    Vectorised over (n_seeds, n_particles).  Unit mass, momentum resampled each
    iteration.  Charges L+1 gradient evaluations per proposal.
    """
    name = "HMC"

    def __init__(self, target, dt, n_leapfrog: int = 10, **kw):
        super().__init__(target, dt, **kw)
        self.L = int(n_leapfrog)
        self.eps = float(dt)

    def step(self, x, gen):
        eps, L = self.eps, self.L
        p = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
        x0, p0 = x, p
        V0 = self.target.potential(x)
        grad = self.target.grad_potential(x)               # ∇V
        self.grad_evals += 1
        self.pot_evals += 1
        xc = x
        pc = p - 0.5 * eps * grad
        for i in range(L):
            xc = xc + eps * pc
            g = self.target.grad_potential(xc)
            self.grad_evals += 1
            if i < L - 1:
                pc = pc - eps * g
            else:
                pc = pc - 0.5 * eps * g
        xc = sanitize(xc, self.clip)
        V1 = self.target.potential(xc)
        self.pot_evals += 1
        H0 = V0 + 0.5 * (p0 ** 2).sum(-1)
        H1 = V1 + 0.5 * (pc ** 2).sum(-1)
        log_a = H0 - H1
        u = torch.rand(log_a.shape, generator=gen, device=x.device, dtype=x.dtype)
        acc = torch.log(u.clamp_min(1e-30)) < log_a
        x_new = torch.where(acc.unsqueeze(-1), xc, x0)
        return x_new, {"acc": acc.float().mean().item()}


class UnderdampedLangevin(Sampler):
    """Underdamped (kinetic) Langevin via the BAOAB splitting; no MH correction.

    Targets exp(-V) at unit temperature with unit mass.  Carries momentum across
    steps and reuses the end-of-step force at the next step's first kick, so the
    amortised cost is one gradient evaluation per step.
    """
    name = "ULD"

    def __init__(self, target, dt, friction: float = 2.0, **kw):
        super().__init__(target, dt, **kw)
        self.gamma = float(friction)
        self._p = None
        self._force = None

    def init_state(self, n_seeds, n_particles, seed, device):
        x = self.target.init_particles(n_seeds, n_particles, seed, device)
        g = torch.Generator(device=device); g.manual_seed(seed + 4242)
        self._p = torch.randn(x.shape, generator=g, device=device)
        self._force = self.target.force(x)
        self.grad_evals += 1
        return x

    def step(self, x, gen):
        dt = self.dt
        c = math.exp(-self.gamma * dt)
        noise_scale = math.sqrt(max(1.0 - c * c, 0.0))
        p = self._p
        f = self._force                                    # force = -∇V at x
        # B (half kick) + A (half drift)
        p = p + 0.5 * dt * f
        x = x + 0.5 * dt * p
        # O (Ornstein-Uhlenbeck thermostat)
        xi = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
        p = c * p + noise_scale * xi
        # A (half drift) + B (half kick) using force at the new position
        x = x + 0.5 * dt * p
        x = sanitize(x, self.clip)
        f_new = self.target.force(x)
        self.grad_evals += 1
        p = p + 0.5 * dt * f_new
        self._p = p
        self._force = f_new
        return x, {}


BASELINE_REGISTRY = {
    "PT": ParallelTempering,
    "HMC": HMC,
    "ULD": UnderdampedLangevin,
}
