"""The seven samplers. All share the same tamed drift map

    tame(b) = b / (1 + dt ||b||)                    (per-particle norm)

applied to every method's drift (ULA, MALA proposal, FLA, BAOAB force, raw
CP, LSC-CP). Tamed MALA remains exact because the proposal density
q(y|x) = N(y; x + dt*tame(b(x)), 2 eps dt I) is used consistently in both
directions of the MH ratio; asymmetric taming would make taming a hidden
variable in the comparison.

Discipline inside step(): no .item(), no .cpu(), no print, no host sync.
Diagnostics accumulate as device tensors and are popped at checkpoints.
"""
from __future__ import annotations

import math

import torch

from .jumps import apply_poisson_jumps
from .config import K_MAX_JUMPS


def tame(b: torch.Tensor, dt: float) -> torch.Tensor:
    # overflow-safe norm: with the Levy score, ||b|| can exceed e^354 and
    # b.norm() would overflow to inf, silently zeroing the tamed drift.
    m = b.abs().amax(dim=-1, keepdim=True).clamp(min=1.0)
    n = m * (b / m).norm(dim=-1, keepdim=True)
    return b / (1.0 + dt * n)


# ------------------------------------------------------------------- boxes
class RectBox:
    """Axis-aligned box in the sampling coordinates."""

    def __init__(self, lo, hi, device) -> None:
        self.lo = torch.as_tensor(lo, dtype=torch.float64, device=device)
        self.hi = torch.as_tensor(hi, dtype=torch.float64, device=device)

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        return ((x >= self.lo) & (x <= self.hi)).all(dim=-1)

    def clip(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.lo, self.hi)


class LatentRectBox:
    """Box specified in latent coordinates z = x B^{-T} (E3)."""

    def __init__(self, lo, hi, potential) -> None:
        dev = potential.B.device
        self.lo = torch.as_tensor(lo, dtype=torch.float64, device=dev)
        self.hi = torch.as_tensor(hi, dtype=torch.float64, device=dev)
        self.pot = potential

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pot.to_latent(x)
        return ((z >= self.lo) & (z <= self.hi)).all(dim=-1)

    def clip(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.clamp(self.pot.to_latent(x), self.lo, self.hi)
        return self.pot.from_latent(z)


# ------------------------------------------------------------- diagnostics
class SamplerBase:
    name = "base"

    def __init__(self) -> None:
        self._sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}
        self._maxes: dict[str, torch.Tensor] = {}

    def _acc(self, key: str, val: torch.Tensor) -> None:
        if key in self._sums:
            self._sums[key] = self._sums[key] + val
            self._counts[key] += 1
        else:
            self._sums[key] = val.clone()
            self._counts[key] = 1

    def _acc_max(self, key: str, val: torch.Tensor) -> None:
        if key in self._maxes:
            self._maxes[key] = torch.maximum(self._maxes[key], val)
        else:
            self._maxes[key] = val.clone()

    def pop_diagnostics(self) -> dict[str, float]:
        """Host sync happens HERE (checkpoints only), never inside step()."""
        out = {k: (self._sums[k] / self._counts[k]).item() for k in self._sums}
        out.update({k: v.item() for k, v in self._maxes.items()})
        self._sums, self._counts, self._maxes = {}, {}, {}
        return out

    def positions(self) -> torch.Tensor:
        return self.x

    def step(self) -> None:
        raise NotImplementedError


# --------------------------------------------------------------------- ULA
class ULA(SamplerBase):
    name = "ULA"

    def __init__(self, pot, x0, dt, eps, gen, box) -> None:
        super().__init__()
        self.pot, self.x, self.dt, self.eps, self.gen, self.box = pot, x0.clone(), dt, eps, gen, box
        self._noise = math.sqrt(2.0 * eps * dt)

    def step(self) -> None:
        g = self.pot.grad(self.x)
        xi = torch.randn(self.x.shape, generator=self.gen, device=self.x.device,
                         dtype=self.x.dtype)
        self.x = self.box.clip(self.x + self.dt * tame(-g, self.dt) + self._noise * xi)


# -------------------------------------------------------------------- MALA
class MALA(SamplerBase):
    """MALA whose proposal matches the ULA step: with grad log pi = -beta grad V,
    Y = X - (h beta / 2) grad V + sqrt(h) xi equals the ULA step iff
    h beta / 2 = dt AND h = 2 eps dt; both give h = 2 dt / beta = dt/4 at beta=8.

    Proposals are never clipped before the accept step (that would silently
    break exactness); out-of-box proposals are auto-rejected, which is valid
    MH for the box-restricted target."""

    name = "MALA"

    def __init__(self, pot, x0, dt, beta, gen, box) -> None:
        super().__init__()
        self.pot, self.x, self.dt, self.beta, self.gen, self.box = pot, x0.clone(), dt, beta, gen, box
        self.h = 2.0 * dt / beta                      # = 2 eps dt
        self.Vx = pot.V(self.x)
        self.gx = pot.grad(self.x)

    def step(self) -> None:
        x, dt, h = self.x, self.dt, self.h
        mu_x = x + dt * tame(-self.gx, dt)
        xi = torch.randn(x.shape, generator=self.gen, device=x.device, dtype=x.dtype)
        y = mu_x + math.sqrt(h) * xi
        Vy = self.pot.V(y)
        gy = self.pot.grad(y)
        mu_y = y + dt * tame(-gy, dt)
        fwd = ((y - mu_x) ** 2).sum(-1)
        bwd = ((x - mu_y) ** 2).sum(-1)
        log_alpha = -self.beta * (Vy - self.Vx) - (bwd - fwd) / (2.0 * h)
        u = torch.rand(x.shape[0], generator=self.gen, device=x.device, dtype=x.dtype)
        acc = (torch.log(u) < log_alpha) & self.box.contains(y)
        accf = acc.to(x.dtype)
        self.x = torch.where(acc.unsqueeze(-1), y, x)
        self.Vx = torch.where(acc, Vy, self.Vx)
        self.gx = torch.where(acc.unsqueeze(-1), gy, self.gx)
        self._acc("mala_accept", accf.mean())


# --------------------------------------------------------------------- FLA
_FLA_ALPHA = 1.7
_FLA_C = math.gamma(_FLA_ALPHA - 1.0) / math.gamma(_FLA_ALPHA / 2.0) ** 2


def sample_sas(shape, alpha: float, gen: torch.Generator, device) -> torch.Tensor:
    """Chambers-Mallows-Stuck: per-coordinate symmetric alpha-stable S-alpha-S(1).
    No tail clipping - a truncated stable is not stable."""
    Phi = (torch.rand(shape, generator=gen, device=device, dtype=torch.float64) - 0.5) * math.pi
    W = -torch.log(torch.rand(shape, generator=gen, device=device, dtype=torch.float64))
    return (torch.sin(alpha * Phi) / torch.cos(Phi) ** (1.0 / alpha)
            * (torch.cos((1.0 - alpha) * Phi) / W) ** ((1.0 - alpha) / alpha))


class FLA(SamplerBase):
    """Fractional Langevin (Simsekli, ICML 2017, sec 3.3):
    X <- X + dt tame(-c_alpha grad U) + dt^{1/alpha} xi_alpha, U = beta V.
    The *uncorrected nonlocal* comparator: heavy tails cross barriers, but
    the invariant law is not pi."""

    name = "FLA"

    def __init__(self, pot, x0, dt, beta, gen, box, alpha: float = _FLA_ALPHA) -> None:
        super().__init__()
        self.pot, self.x, self.dt, self.beta, self.gen, self.box = pot, x0.clone(), dt, beta, gen, box
        self.alpha = alpha
        self.c_alpha = math.gamma(alpha - 1.0) / math.gamma(alpha / 2.0) ** 2
        self._noise = dt ** (1.0 / alpha)

    def step(self) -> None:
        g = self.pot.grad(self.x)                     # grad U = beta grad V
        drift = tame(-self.c_alpha * self.beta * g, self.dt)
        xi = sample_sas(self.x.shape, self.alpha, self.gen, self.x.device)
        self.x = self.box.clip(self.x + self.dt * drift + self._noise * xi)


# ------------------------------------------------------------------- BAOAB
class BAOAB(SamplerBase):
    """Kinetic Langevin, BAOAB splitting (NOT HMC: no accept/reject, carries
    O(dt^2) configurational bias). Unit mass, gamma = 1. The O-step is the
    exact OU solution: dp = -gamma p dt + sqrt(2 gamma eps) dW gives variance
    eps (1 - e^{-2 gamma dt}) by Ito isometry. Trailing force is cached as
    the next step's leading B (one gradient per step)."""

    name = "BAOAB"

    def __init__(self, pot, x0, dt, eps, gen, box, gamma: float = 1.0) -> None:
        super().__init__()
        self.pot, self.x, self.dt, self.eps, self.gen, self.box = pot, x0.clone(), dt, eps, gen, box
        self.gamma = gamma
        self.c1 = math.exp(-gamma * dt)
        self.c2 = math.sqrt(eps * (1.0 - self.c1 ** 2))
        self.p = math.sqrt(eps) * torch.randn(x0.shape, generator=gen,
                                              device=x0.device, dtype=x0.dtype)
        self.f = tame(-pot.grad(self.x), dt)          # cached tamed force

    def step(self) -> None:
        dt = self.dt
        p = self.p + 0.5 * dt * self.f                                    # B
        q = self.x + 0.5 * dt * p                                         # A
        xi = torch.randn(q.shape, generator=self.gen, device=q.device, dtype=q.dtype)
        p = self.c1 * p + self.c2 * xi                                    # O
        q = q + 0.5 * dt * p                                              # A
        self.f = tame(-self.pot.grad(q), dt)                              # B (cached)
        self.p = p + 0.5 * dt * self.f
        self.x = self.box.clip(q)


# ---------------------------------------------------------------------- PT
class ParallelTempering(SamplerBase):
    """MALA-within-replica parallel tempering. K replicas at beta_k
    (geometric ladder), replica k using h_k = 2 dt / beta_k, so every replica
    takes the SAME tamed drift step and only the noise scale differs.

    Swaps of adjacent pairs (alternating even/odd offsets) every n_swap
    steps; the joint target is prod_k pi_k and the swap is a deterministic
    involution, so
        alpha_swap = min{1, exp[(beta_i - beta_{i+1})(V(x_i) - V(x_{i+1}))]}.
    V values are cached by MALA, so swaps are free in evaluation count.
    State is (K, N, d): the replica index is a batch dimension. Metrics use
    the cold replica only; wall-clock includes all K replicas."""

    name = "PT"

    def __init__(self, pot, x0, dt, betas: torch.Tensor, gen, box,
                 n_swap: int = 10) -> None:
        super().__init__()
        self.pot, self.dt, self.gen, self.box = pot, dt, gen, box
        self.betas = betas                                    # (K,) descending, betas[0]=8
        self.K = betas.shape[0]
        self.n_swap = n_swap
        self.x = x0.unsqueeze(0).repeat(self.K, 1, 1)         # (K, N, d)
        self.h = (2.0 * dt / betas).view(-1, 1)               # (K, 1)
        self.Vx = pot.V(self.x)                               # (K, N)
        self.gx = pot.grad(self.x)
        self._step_count = 0

    def step(self) -> None:
        x, dt = self.x, self.dt
        mu_x = x + dt * tame(-self.gx, dt)
        xi = torch.randn(x.shape, generator=self.gen, device=x.device, dtype=x.dtype)
        y = mu_x + torch.sqrt(self.h).unsqueeze(-1) * xi
        Vy = self.pot.V(y)
        gy = self.pot.grad(y)
        mu_y = y + dt * tame(-gy, dt)
        fwd = ((y - mu_x) ** 2).sum(-1)
        bwd = ((x - mu_y) ** 2).sum(-1)
        log_alpha = (-self.betas.view(-1, 1) * (Vy - self.Vx)
                     - (bwd - fwd) / (2.0 * self.h))
        u = torch.rand(x.shape[:2], generator=self.gen, device=x.device, dtype=x.dtype)
        acc = (torch.log(u) < log_alpha) & self.box.contains(y)
        self.x = torch.where(acc.unsqueeze(-1), y, x)
        self.Vx = torch.where(acc, Vy, self.Vx)
        self.gx = torch.where(acc.unsqueeze(-1), gy, self.gx)
        self._acc("mala_accept", acc.to(torch.float64).mean())

        self._step_count += 1
        if self._step_count % self.n_swap == 0:
            offset = (self._step_count // self.n_swap) % 2
            self._swap_pass(offset)

    def _swap_pass(self, offset: int) -> None:
        accs = []
        for i in range(offset, self.K - 1, 2):
            log_a = (self.betas[i] - self.betas[i + 1]) * (self.Vx[i] - self.Vx[i + 1])
            u = torch.rand(self.x.shape[1], generator=self.gen,
                           device=self.x.device, dtype=torch.float64)
            sw = torch.log(u) < log_a                          # (N,)
            swf = sw.unsqueeze(-1)
            # materialise both branches BEFORE writing (x[i], x[i+1] are views)
            new_xi = torch.where(swf, self.x[i + 1], self.x[i])
            new_xj = torch.where(swf, self.x[i], self.x[i + 1])
            self.x[i], self.x[i + 1] = new_xi, new_xj
            new_Vi = torch.where(sw, self.Vx[i + 1], self.Vx[i])
            new_Vj = torch.where(sw, self.Vx[i], self.Vx[i + 1])
            self.Vx[i], self.Vx[i + 1] = new_Vi, new_Vj
            new_gi = torch.where(swf, self.gx[i + 1], self.gx[i])
            new_gj = torch.where(swf, self.gx[i], self.gx[i + 1])
            self.gx[i], self.gx[i + 1] = new_gi, new_gj
            accs.append(sw.to(torch.float64).mean())
        if accs:
            self._acc("pt_swap_accept", torch.stack(accs).mean())

    def positions(self) -> torch.Tensor:
        return self.x[0]                                       # cold replica


def geometric_ladder(beta_max: float, beta_min: float, K: int, device) -> torch.Tensor:
    r = (beta_min / beta_max) ** (1.0 / (K - 1))
    return beta_max * torch.as_tensor([r ** k for k in range(K)],
                                      dtype=torch.float64, device=device)


def tune_ladder(pot, x0, dt, box, beta_max: float, beta_min: float,
                pilot_steps: int = 20_000, burn_frac: float = 0.5,
                target=(0.2, 0.4), K0: int = 8,
                K_cap: int = 64, seed: int = 1234) -> tuple[torch.Tensor, dict]:
    """Pick K (geometric ladder, fixed endpoints) so the mean swap acceptance
    over a pilot run lands in [0.2, 0.4]. Acceptance is measured only on the
    post-burn-in half of the pilot: all replicas start at the cold x0, so
    early swaps (near-equal V across the ladder) accept at a transiently
    inflated rate, and a short pilot would silently under-ladder PT.
    Returns (betas, tuning record)."""
    dev = x0.device
    K = K0
    record = {}
    best = None
    burn = int(burn_frac * pilot_steps)
    for _ in range(10):
        betas = geometric_ladder(beta_max, beta_min, K, dev)
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)
        pt = ParallelTempering(pot, x0, dt, betas, gen, box)
        for _s in range(burn):
            pt.step()
        pt.pop_diagnostics()                     # discard transient
        for _s in range(pilot_steps - burn):
            pt.step()
        diag = pt.pop_diagnostics()
        acc = diag.get("pt_swap_accept", 0.0)
        record[K] = acc
        best = (K, betas, acc)
        if acc < target[0]:
            K = min(int(math.ceil(K * 1.5)), K_cap)
        elif acc > target[1]:
            K = max(2, K - 1 if K <= 4 else int(math.floor(K * 0.75)))
        else:
            break
        if K in record:                    # oscillating; keep closest
            break
    # keep the tried K whose acceptance is closest to the target band
    def _dist(a):
        return 0.0 if target[0] <= a <= target[1] else min(abs(a - target[0]),
                                                           abs(a - target[1]))
    K = min(record, key=lambda k: _dist(record[k]))
    acc = record[K]
    betas = geometric_ladder(beta_max, beta_min, K, dev)
    r = (beta_min / beta_max) ** (1.0 / (K - 1))
    return betas, {"K": K, "r": r, "beta_min": beta_min, "beta_max": beta_max,
                   "swap_acceptance": acc, "history": record,
                   "band_attained": bool(target[0] <= acc <= target[1])}


# ------------------------------------------------------- raw CP and LSC-CP
class CompoundPoisson(SamplerBase):
    """Shared discretisation for raw CP (score=None) and LSC-CP (score set):

        X1 = X + dt b(X)/(1 + dt||b||) + sqrt(2 eps dt) xi,
        X_{n+1} = X1 + sum_{k<=N_n} A_k,  N_n ~ Poisson(lam dt), A_k ~ nu.

    b = -grad V (raw CP) or -grad V + S_{nu,beta} (LSC-CP). The jump stream
    (gen_jump) is a dedicated generator seeded identically for both methods,
    so their jump times and increments are pathwise identical."""

    def __init__(self, pot, x0, dt, eps, lam, law, gen_diff, gen_jump, box,
                 score=None, name: str | None = None) -> None:
        super().__init__()
        self.pot, self.x, self.dt, self.eps, self.lam = pot, x0.clone(), dt, eps, lam
        self.law, self.gen_diff, self.gen_jump, self.box = law, gen_diff, gen_jump, box
        self.score = score
        self.name = name or ("LSC-CP" if score is not None else "CP")
        self._noise = math.sqrt(2.0 * eps * dt)

    def step(self) -> None:
        g = self.pot.grad(self.x)
        b = -g
        if self.score is not None:
            S, sdiag = self.score(self.x)
            b = b + S
            self._acc("m_clip_fraction", sdiag["m_clip_fraction"])
            self._acc_max("max_log_magnitude", sdiag["max_log_magnitude"])
        xi = torch.randn(self.x.shape, generator=self.gen_diff,
                         device=self.x.device, dtype=self.x.dtype)
        x1 = self.x + self.dt * tame(b, self.dt) + self._noise * xi
        x1, counts = apply_poisson_jumps(x1, self.law, self.lam, self.dt,
                                         self.gen_jump, K_MAX_JUMPS)
        self.x = self.box.clip(x1)
        self._acc("jump_count_mean", counts.mean())
