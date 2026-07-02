
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .jump_banks import FiniteJumpBank
from .levy_score import stationary_levy_score


def torch_generator(seed: int, device) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def sanitize(x: torch.Tensor, clip: float) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)


def tame(drift: torch.Tensor, dt: float, cap: float = 1.0) -> torch.Tensor:
    nrm = drift.norm(dim=-1, keepdim=True)
    return float(dt) * drift / (1.0 + (float(dt) / float(cap)) * nrm)


@dataclass
class SamplerDiagnostics:
    grad_evals: int = 0
    pot_evals: int = 0
    levy_quadrature_evals: int = 0
    jump_events: int = 0
    acceptance_sum: float = 0.0
    acceptance_n: int = 0
    swap_sum: float = 0.0
    swap_n: int = 0
    levy_score_norm_mean_sum: float = 0.0
    levy_score_norm_max: float = 0.0
    levy_exponent_clipped_frac_sum: float = 0.0
    levy_diag_n: int = 0

    def as_dict(self) -> dict:
        return {
            "grad_evals": self.grad_evals,
            "pot_evals": self.pot_evals,
            "levy_quadrature_evals": self.levy_quadrature_evals,
            "jump_events": self.jump_events,
            "acceptance_rate": self.acceptance_sum / self.acceptance_n if self.acceptance_n else float("nan"),
            "swap_acceptance_rate": self.swap_sum / self.swap_n if self.swap_n else float("nan"),
            "levy_score_norm_mean": self.levy_score_norm_mean_sum / self.levy_diag_n if self.levy_diag_n else 0.0,
            "levy_score_norm_max": self.levy_score_norm_max,
            "levy_exponent_clipped_frac": self.levy_exponent_clipped_frac_sum / self.levy_diag_n if self.levy_diag_n else 0.0,
        }


class Sampler:
    name = "base"

    def __init__(self, potential, dt: float, tame_cap: float = 1.0, **kwargs):
        self.potential = potential
        self.dt = float(dt)
        self.tame_cap = float(tame_cap)
        self.diag = SamplerDiagnostics()

    def init_state(self, n: int, seed: int, device):
        return self.potential.initial_state(n, seed, device)

    def step(self, x, gen):
        raise NotImplementedError

    def final_samples(self, x):
        return x


class OverdampedLangevin(Sampler):
    name = "ULA"

    def step(self, x, gen):
        f = self.potential.force(x); self.diag.grad_evals += 1
        noise = math.sqrt(2.0 / self.potential.beta * self.dt) * torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
        return sanitize(x + tame(f, self.dt, self.tame_cap) + noise, self.potential.state_clip), {}


class MALA(Sampler):
    name = "MALA"

    def step(self, x, gen):
        dt = self.dt; beta = self.potential.beta
        f = beta * self.potential.force(x); self.diag.grad_evals += 1
        y = sanitize(x + 0.5 * dt * f + math.sqrt(dt) * torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype), self.potential.state_clip)
        fy = beta * self.potential.force(y); self.diag.grad_evals += 1
        lq_fwd = -((y - x - 0.5 * dt * f) ** 2).sum(-1) / (2 * dt)
        lq_rev = -((x - y - 0.5 * dt * fy) ** 2).sum(-1) / (2 * dt)
        Vx = self.potential.potential(x); Vy = self.potential.potential(y); self.diag.pot_evals += 2
        log_a = (-beta * Vy + lq_rev) - (-beta * Vx + lq_fwd)
        u = torch.rand(log_a.shape, generator=gen, device=x.device, dtype=x.dtype)
        acc = torch.log(u.clamp_min(1e-30)) < log_a
        self.diag.acceptance_sum += float(acc.float().mean().item()); self.diag.acceptance_n += 1
        return torch.where(acc.unsqueeze(-1), y, x), {"acc": float(acc.float().mean().item())}


class BAOAB(Sampler):
    name = "BAOAB"

    def __init__(self, potential, dt, friction: float = 2.0, **kwargs):
        super().__init__(potential, dt, **kwargs)
        self.friction = float(friction)
        self.p = None
        self.force_cache = None

    def init_state(self, n, seed, device):
        x = self.potential.initial_state(n, seed, device)
        g = torch_generator(seed + 551, device)
        self.p = torch.randn(x.shape, generator=g, device=device) / math.sqrt(self.potential.beta)
        self.force_cache = self.potential.force(x); self.diag.grad_evals += 1
        return x

    def step(self, x, gen):
        dt = self.dt
        c = math.exp(-self.friction * dt)
        sigma_p = math.sqrt((1 - c*c) / self.potential.beta)
        p = self.p + 0.5 * dt * self.force_cache
        x = x + 0.5 * dt * p
        p = c * p + sigma_p * torch.randn(p.shape, generator=gen, device=x.device, dtype=x.dtype)
        x = sanitize(x + 0.5 * dt * p, self.potential.state_clip)
        f = self.potential.force(x); self.diag.grad_evals += 1
        self.p = p + 0.5 * dt * f
        self.force_cache = f
        return x, {}


class HMC(Sampler):
    name = "HMC"

    def __init__(self, potential, dt, n_leapfrog: int = 8, **kwargs):
        super().__init__(potential, dt, **kwargs)
        self.n_leapfrog = int(n_leapfrog)

    def step(self, x, gen):
        eps = self.dt; beta = self.potential.beta
        p0 = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
        x0 = x
        V0 = beta * self.potential.potential(x0); self.diag.pot_evals += 1
        p = p0 - 0.5 * eps * beta * self.potential.gradient(x0); self.diag.grad_evals += 1
        xc = x0
        for i in range(self.n_leapfrog):
            xc = sanitize(xc + eps * p, self.potential.state_clip)
            g = beta * self.potential.gradient(xc); self.diag.grad_evals += 1
            p = p - (eps if i < self.n_leapfrog - 1 else 0.5 * eps) * g
        V1 = beta * self.potential.potential(xc); self.diag.pot_evals += 1
        H0 = V0 + 0.5 * (p0 ** 2).sum(-1)
        H1 = V1 + 0.5 * (p ** 2).sum(-1)
        log_a = H0 - H1
        u = torch.rand(log_a.shape, generator=gen, device=x.device, dtype=x.dtype)
        acc = torch.log(u.clamp_min(1e-30)) < log_a
        self.diag.acceptance_sum += float(acc.float().mean().item()); self.diag.acceptance_n += 1
        return torch.where(acc.unsqueeze(-1), xc, x0), {"acc": float(acc.float().mean().item())}


class ParallelTempering(Sampler):
    name = "PT"

    def __init__(self, potential, dt, n_temps: int = 4, beta_min_factor: float = 0.2,
                 swap_interval: int = 5, **kwargs):
        super().__init__(potential, dt, **kwargs)
        self.n_temps = int(n_temps)
        self.beta_min_factor = float(beta_min_factor)
        self.swap_interval = int(swap_interval)
        self.step_i = 0
        self.factors = None

    def init_state(self, n, seed, device):
        base = self.potential.initial_state(n, seed, device)
        self.factors = torch.tensor(torch.logspace(0, math.log10(self.beta_min_factor), self.n_temps).tolist(), device=device, dtype=base.dtype)
        return base.unsqueeze(0).repeat(self.n_temps, 1, 1)

    def step(self, x, gen):
        dt = self.dt; beta0 = self.potential.beta
        fac = self.factors.view(-1, 1, 1)
        f = beta0 * fac * self.potential.force(x); self.diag.grad_evals += self.n_temps
        y = sanitize(x + 0.5 * dt * f + math.sqrt(dt) * torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype), self.potential.state_clip)
        fy = beta0 * fac * self.potential.force(y); self.diag.grad_evals += self.n_temps
        lq_fwd = -((y - x - 0.5 * dt * f) ** 2).sum(-1) / (2 * dt)
        lq_rev = -((x - y - 0.5 * dt * fy) ** 2).sum(-1) / (2 * dt)
        Vx = self.potential.potential(x); Vy = self.potential.potential(y); self.diag.pot_evals += 2 * self.n_temps
        log_a = (-beta0 * fac.squeeze(-1) * Vy + lq_rev) - (-beta0 * fac.squeeze(-1) * Vx + lq_fwd)
        u = torch.rand(log_a.shape, generator=gen, device=x.device, dtype=x.dtype)
        acc = torch.log(u.clamp_min(1e-30)) < log_a
        x = torch.where(acc.unsqueeze(-1), y, x)
        self.diag.acceptance_sum += float(acc.float().mean().item()); self.diag.acceptance_n += 1
        self.step_i += 1
        if self.step_i % self.swap_interval == 0:
            x = self._swap(x, gen)
        return x, {"acc": float(acc.float().mean().item())}

    def _swap(self, x, gen):
        V = self.potential.potential(x); self.diag.pot_evals += self.n_temps
        beta = self.potential.beta * self.factors
        accepts = []
        for i in range(self.n_temps - 1):
            log_a = (beta[i] - beta[i+1]) * (V[i] - V[i+1])
            u = torch.rand(log_a.shape, generator=gen, device=x.device, dtype=x.dtype)
            sw = torch.log(u.clamp_min(1e-30)) < log_a
            xi, xj = x[i].clone(), x[i+1].clone()
            x[i] = torch.where(sw.unsqueeze(-1), xj, xi)
            x[i+1] = torch.where(sw.unsqueeze(-1), xi, xj)
            accepts.append(float(sw.float().mean().item()))
        if accepts:
            self.diag.swap_sum += float(sum(accepts) / len(accepts)); self.diag.swap_n += 1
        return x

    def final_samples(self, x):
        return x[0]


class LevyScoreJumpDiffusion(Sampler):
    name = "LSBMC"

    def __init__(self, potential, dt, bank: FiniteJumpBank, n_theta: int = 8,
                 jump_chunk: int = 64, particle_chunk: int | None = None,
                 exponent_clip: float = 60.0, score_clip: float = 100.0,
                 use_score: bool = True, **kwargs):
        super().__init__(potential, dt, **kwargs)
        self.bank = bank
        self.n_theta = int(n_theta)
        self.jump_chunk = int(jump_chunk)
        self.particle_chunk = particle_chunk
        self.exponent_clip = float(exponent_clip)
        self.score_clip = float(score_clip)
        # use_score=True -> LSC-CP (corrected); False -> raw compound-Poisson (no drift correction).
        # Both draw the Brownian increment then the jump increment from the same generator in the
        # same order, so with a shared seed they use identical noise + jump realizations (the
        # manuscript's "same jump times and jump law") and differ only by the stationary drift.
        self.use_score = bool(use_score)

    def step(self, x, gen):
        bank = self.bank.to(device=x.device, dtype=x.dtype)
        f = self.potential.force(x); self.diag.grad_evals += 1
        if self.use_score:
            S, sdiag = stationary_levy_score(self.potential.potential, x, bank, self.potential.beta,
                                             n_theta=self.n_theta, particle_chunk=self.particle_chunk,
                                             jump_chunk=self.jump_chunk, exponent_clip=self.exponent_clip,
                                             score_clip=self.score_clip, return_diagnostics=True)
            self.diag.levy_quadrature_evals += int(sdiag["levy_quadrature_evals"])
            self.diag.pot_evals += int(sdiag["levy_quadrature_evals"])
            self.diag.levy_score_norm_mean_sum += float(sdiag["levy_score_norm_mean"])
            self.diag.levy_score_norm_max = max(self.diag.levy_score_norm_max, float(sdiag["levy_score_norm_max"]))
            self.diag.levy_exponent_clipped_frac_sum += float(sdiag["levy_exponent_clipped_frac"])
            self.diag.levy_diag_n += 1
        else:
            S = torch.zeros_like(x)
            sdiag = {"levy_score_norm_mean": 0.0}
        noise = math.sqrt(2.0 / self.potential.beta * self.dt) * torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
        x = sanitize(x + tame(f + S, self.dt, self.tame_cap) + noise, self.potential.state_clip)
        inc, jdiag = bank.sample_increment(x.shape[:-1], gen, self.dt, device=x.device, dtype=x.dtype)
        self.diag.jump_events += int(jdiag["jump_events"])
        return sanitize(x + inc, self.potential.state_clip), {**sdiag, **jdiag}


SAMPLERS = {
    "ULA": OverdampedLangevin,
    "MALA": MALA,
    "BAOAB": BAOAB,
    "HMC": HMC,
    "PT": ParallelTempering,
    "LSBMC": LevyScoreJumpDiffusion,
}
