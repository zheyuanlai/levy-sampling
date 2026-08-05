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


def tame(b: torch.Tensor, dt: float, cap: float = 1.0) -> torch.Tensor:
    """Tamed drift; the displacement dt*tame(b) is bounded by `cap`.

    cap = 1 is the shared default for every method. A smaller cap may be set
    for the coupled CP/LSC-CP pair on landscapes where the corrective score
    forms long transport tubes: a single O(1) hop scatters landers out of
    the tube (into side basins), while steps bounded by the jump-shell scale
    follow the tube (E3's design-ladder ablation measures this).

    Overflow-safe norm: with the Levy score, ||b|| can exceed e^354 and
    b.norm() would overflow to inf, silently zeroing the tamed drift."""
    m = b.abs().amax(dim=-1, keepdim=True).clamp(min=1.0)
    n = m * (b / m).norm(dim=-1, keepdim=True)
    return b / (1.0 + dt * n / cap)


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
        # Interval diagnostics are reset after each checkpoint.  Cumulative
        # diagnostics deliberately persist for the lifetime of the analyzed
        # sampler (production warm-up uses a separate throwaway instance).
        self._sums: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}
        self._maxes: dict[str, torch.Tensor] = {}
        self._cumulative: dict[str, torch.Tensor] = {}
        self._cumulative_ratio_numerators: dict[str, torch.Tensor] = {}
        self._cumulative_ratio_denominators: dict[str, torch.Tensor] = {}
        self._static_diagnostics: dict[str, int | float] = {}

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

    def _acc_cumulative(self, key: str, val: torch.Tensor) -> None:
        """Accumulate a device scalar without synchronizing the host."""
        if key in self._cumulative:
            self._cumulative[key] = self._cumulative[key] + val
        else:
            self._cumulative[key] = val.clone()

    def _acc_cumulative_ratio(self, key: str, numerator: torch.Tensor,
                              denominator: torch.Tensor) -> None:
        """Accumulate numerator/denominator for a lifetime-to-date ratio."""
        if key in self._cumulative_ratio_numerators:
            self._cumulative_ratio_numerators[key] = (
                self._cumulative_ratio_numerators[key] + numerator)
            self._cumulative_ratio_denominators[key] = (
                self._cumulative_ratio_denominators[key] + denominator)
        else:
            self._cumulative_ratio_numerators[key] = numerator.clone()
            self._cumulative_ratio_denominators[key] = denominator.clone()

    def _clip_state(self, candidate: torch.Tensor) -> torch.Tensor:
        """Clip finite boundary crossings without masking numerical explosions.

        Nonfinite particle rows remain nonfinite so run-level safety gates see
        the failure; they are counted separately from ordinary box clipping.
        """
        finite = torch.isfinite(candidate).all(dim=-1)
        outside = ~self.box.contains(candidate)
        finite_outside = finite & outside
        clipped = finite_outside.to(torch.int64).sum()
        nonfinite = (~finite).to(torch.int64).sum()
        proposed = torch.as_tensor(outside.numel(), dtype=torch.int64,
                                   device=outside.device)
        self._acc_cumulative("state_box_clip_count_cumulative", clipped)
        self._acc_cumulative("nonfinite_proposal_count_cumulative", nonfinite)
        self._acc_cumulative_ratio(
            "state_box_clip_fraction_cumulative", clipped, proposed)
        self._acc_cumulative_ratio(
            "nonfinite_proposal_fraction_cumulative", nonfinite, proposed)
        clipped_candidate = self.box.clip(candidate)
        return torch.where(finite.unsqueeze(-1), clipped_candidate, candidate)

    def _record_score_diagnostics(self, diagnostics: dict) -> None:
        """Record interval and lifetime score-clipping diagnostics."""
        fraction = diagnostics["m_clip_fraction"]
        count = diagnostics["m_clip_count"]
        total = diagnostics["m_clip_total"]
        if (count.dtype != torch.int64 or total.dtype != torch.int64
                or count.device != fraction.device
                or total.device != fraction.device):
            raise TypeError("score clip counts must be device-resident int64 scalars")
        self._acc("m_clip_fraction", fraction)
        self._acc_max("max_log_magnitude", diagnostics["max_log_magnitude"])
        self._acc_cumulative("score_clip_count_cumulative", count)
        self._acc_cumulative_ratio(
            "score_clip_fraction_cumulative", count, total)

    def _record_outside_proposals(self, candidate: torch.Tensor,
                                  inside: torch.Tensor) -> None:
        """Distinguish finite box rejections from nonfinite MH proposals."""
        finite = torch.isfinite(candidate).all(dim=-1)
        finite_outside = finite & (~inside)
        outside = finite_outside.to(torch.int64).sum()
        nonfinite = (~finite).to(torch.int64).sum()
        proposed = torch.as_tensor(inside.numel(), dtype=torch.int64,
                                   device=inside.device)
        self._acc_cumulative(
            "outside_proposal_reject_count_cumulative", outside)
        self._acc_cumulative("nonfinite_proposal_count_cumulative", nonfinite)
        self._acc_cumulative_ratio(
            "outside_proposal_reject_fraction_cumulative", outside, proposed)
        self._acc_cumulative_ratio(
            "nonfinite_proposal_fraction_cumulative", nonfinite, proposed)

    def pop_diagnostics(self) -> dict[str, float | int]:
        """Host sync happens HERE (checkpoints only), never inside step()."""
        out = {k: (self._sums[k] / self._counts[k]).item() for k in self._sums}
        out.update({k: v.item() for k, v in self._maxes.items()})
        out.update({k: v.item() for k, v in self._cumulative.items()})
        out.update({
            key: torch.where(
                self._cumulative_ratio_denominators[key] > 0,
                self._cumulative_ratio_numerators[key]
                / self._cumulative_ratio_denominators[key],
                torch.zeros_like(
                    self._cumulative_ratio_numerators[key],
                    dtype=torch.float64)).item()
            for key in self._cumulative_ratio_numerators
        })
        out.update(self._static_diagnostics)
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
        candidate = self.x + self.dt * tame(-g, self.dt) + self._noise * xi
        self.x = self._clip_state(candidate)


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
        inside = self.box.contains(y)
        self._record_outside_proposals(y, inside)
        acc = (torch.log(u) < log_alpha) & inside
        accf = acc.to(x.dtype)
        self.x = torch.where(acc.unsqueeze(-1), y, x)
        self.Vx = torch.where(acc, Vy, self.Vx)
        self.gx = torch.where(acc.unsqueeze(-1), gy, self.gx)
        accepted = acc.to(torch.int64).sum()
        proposed = torch.as_tensor(
            acc.numel(), dtype=torch.int64, device=acc.device)
        self._acc("mala_accept", accf.mean())
        self._acc_cumulative("mala_accept_count_cumulative", accepted)
        self._acc_cumulative("mala_proposal_count_cumulative", proposed)
        self._acc_cumulative_ratio(
            "mala_accept_fraction_cumulative", accepted, proposed)


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
        candidate = self.x + self.dt * drift + self._noise * xi
        self.x = self._clip_state(candidate)


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
        q = self._clip_state(q)
        self.f = tame(-self.pot.grad(q), dt)                              # B (cached)
        self.p = p + 0.5 * dt * self.f
        self.x = q


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
        inside = self.box.contains(y)
        self._record_outside_proposals(y, inside)
        acc = (torch.log(u) < log_alpha) & inside
        self.x = torch.where(acc.unsqueeze(-1), y, x)
        self.Vx = torch.where(acc, Vy, self.Vx)
        self.gx = torch.where(acc.unsqueeze(-1), gy, self.gx)
        accepted = acc.to(torch.int64).sum()
        proposed = torch.as_tensor(
            acc.numel(), dtype=torch.int64, device=acc.device)
        self._acc("mala_accept", acc.to(torch.float64).mean())
        self._acc_cumulative("mala_accept_count_cumulative", accepted)
        self._acc_cumulative("mala_proposal_count_cumulative", proposed)
        self._acc_cumulative_ratio(
            "mala_accept_fraction_cumulative", accepted, proposed)

        self._step_count += 1
        if self._step_count % self.n_swap == 0:
            offset = (self._step_count // self.n_swap) % 2
            self._swap_pass(offset)

    def _swap_pass(self, offset: int) -> None:
        accepted_counts = []
        proposal_counts = []
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
            accepted_counts.append(sw.to(torch.int64).sum())
            proposal_counts.append(torch.as_tensor(
                sw.numel(), dtype=torch.int64, device=sw.device))
        if accepted_counts:
            accepted = torch.stack(accepted_counts).sum()
            proposed = torch.stack(proposal_counts).sum()
            self._acc(
                "pt_swap_accept", accepted.to(torch.float64) / proposed)
            self._acc_cumulative("pt_swap_accept_count_cumulative", accepted)
            self._acc_cumulative("pt_swap_proposal_count_cumulative", proposed)
            self._acc_cumulative_ratio(
                "pt_swap_accept_fraction_cumulative", accepted, proposed)

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

    b = -grad V (raw CP) or -grad V + S_{nu,beta} (LSC-CP). ``gen_jump``
    is a dedicated generator. Full-law CP/LSC-CP and the single-atom RA pair
    can use identically seeded streams for pathwise coupling. Paired-MA instead
    draws a stratified bank and atomwise counts from that stream; it has no
    full-law raw-CP pathwise counterpart."""

    def __init__(self, pot, x0, dt, eps, lam, law, gen_diff, gen_jump, box,
                 score=None, name: str | None = None,
                 drift_cap: float = 1.0, jump_mode: str = "full") -> None:
        super().__init__()
        self.pot, self.x, self.dt, self.eps, self.lam = pot, x0.clone(), dt, eps, lam
        self.law, self.gen_diff, self.gen_jump, self.box = law, gen_diff, gen_jump, box
        self.score = score
        self.name = name or ("LSC-CP" if score is not None else "CP")
        self.drift_cap = float(drift_cap)      # same cap for CP and LSC-CP
        self._noise = math.sqrt(2.0 * eps * dt)
        # "full": full-law jumps (redraw r per jump) + exact-quadrature score,
        #         the deployed exact LSC-CP / raw-CP pair.
        # "atomic": single displacement R_n per step drives BOTH the score
        #         (RandomAtomicShellScore) and the jump -- the RA estimator.
        # "paired_multiatom": a realised bank R_{n,a} drives BOTH the
        #         MultiAtomShellScore and independent atomwise Poisson jumps
        #         N_{n,a} ~ Pois(lam*w_a*dt).
        if jump_mode not in ("full", "atomic", "paired_multiatom"):
            raise ValueError(f"unknown jump_mode {jump_mode!r}")
        is_multiatom_score = (score is not None
                              and hasattr(score, "sample_bank")
                              and hasattr(score, "score_for_bank"))
        if is_multiatom_score and jump_mode != "paired_multiatom":
            raise ValueError("a multi-atom score must use paired_multiatom jumps")
        if jump_mode == "paired_multiatom":
            if not is_multiatom_score:
                raise ValueError("paired_multiatom requires a multi-atom score")
            if score.law is not law:
                raise ValueError("paired_multiatom score and jumps must share the same law")
            if score.potential is not pot:
                raise ValueError("paired_multiatom score and sampler must share the same potential")
            if not math.isclose(float(score.lam), float(lam), rel_tol=1e-14,
                                abs_tol=0.0):
                raise ValueError("paired_multiatom score and jumps must share lambda")
            if not math.isclose(float(score.beta) * float(eps), 1.0,
                                rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError("paired_multiatom requires eps = 1 / score.beta")
            # A shell bank advertises its geometry through .atoms; a continuous
            # law has none, so fall back to the law's own device/dtype.
            atoms = getattr(law, "atoms", None)
            law_device = (atoms.device if atoms is not None
                          else getattr(law, "device", self.x.device))
            law_dtype = (atoms.dtype if atoms is not None
                         else getattr(law, "dtype", self.x.dtype))
            # Resolve through an empty tensor so an index-free 'cuda' compares
            # equal to the state's 'cuda:0'.
            law_device = torch.empty(0, device=law_device).device
            if law_device != self.x.device or law_dtype != self.x.dtype:
                raise ValueError("paired_multiatom law and state must share device and dtype")
        self.jump_mode = jump_mode
        if jump_mode == "full":
            # Full-law mode uses a fixed unrolled jump loop.  The sampled
            # Poisson count is retained separately so a cap hit is explicit.
            self._static_diagnostics["jump_cap_k"] = int(K_MAX_JUMPS)

    def step(self) -> None:
        if self.jump_mode == "atomic":
            self._step_atomic()
        elif self.jump_mode == "paired_multiatom":
            self._step_paired_multiatom()
        else:
            self._step_full()

    def _record_jump_counts(self, sampled: torch.Tensor,
                            applied: torch.Tensor) -> None:
        """Record exact occurrences, applied jumps, and lifetime jump rates."""
        sampled_by_particle = (sampled if sampled.ndim == 1
                               else sampled.sum(dim=tuple(range(1, sampled.ndim))))
        applied_by_particle = (applied if applied.ndim == 1
                               else applied.sum(dim=tuple(range(1, applied.ndim))))
        sampled_total = sampled_by_particle.to(torch.int64).sum()
        applied_total = applied_by_particle.to(torch.int64).sum()
        particle_time = torch.as_tensor(
            sampled_by_particle.numel() * self.dt,
            dtype=torch.float64, device=sampled.device)
        self._acc_cumulative("jump_count_cumulative", sampled_total)
        self._acc_cumulative("jump_count_applied_cumulative", applied_total)
        self._acc_cumulative_ratio(
            "jump_rate_per_particle_time_cumulative",
            sampled_total, particle_time)
        self._acc_cumulative_ratio(
            "jump_applied_rate_per_particle_time_cumulative",
            applied_total, particle_time)

        if self.jump_mode == "full":
            excess_by_particle = (
                sampled_by_particle.to(torch.int64)
                - applied_by_particle.to(torch.int64))
            cap_hits = (excess_by_particle > 0).to(torch.int64).sum()
            proposals = torch.as_tensor(
                sampled_by_particle.numel(), dtype=torch.int64,
                device=sampled.device)
            self._acc_cumulative(
                "jump_cap_hit_count_cumulative", cap_hits)
            self._acc_cumulative(
                "jump_cap_excess_count_cumulative", excess_by_particle.sum())
            self._acc_cumulative_ratio(
                "jump_cap_hit_fraction_cumulative", cap_hits, proposals)

    def _record_jump_boundary(self, before_jump: torch.Tensor,
                              after_jump: torch.Tensor,
                              applied: torch.Tensor) -> None:
        """Count finite box exits conditional on actually applied jumps.

        The generic state-box diagnostic mixes drift/diffusion and jump exits.
        This diagnostic isolates particles whose finite pre-jump state was
        inside the numerical box but whose finite post-jump candidate was not.
        For rare multi-jump steps all applied jumps on such a particle are
        charged; the exact applied-jump denominator is recorded separately.
        """
        applied_by_particle = (applied if applied.ndim == 1
                               else applied.sum(dim=tuple(
                                   range(1, applied.ndim))))
        applied_i64 = applied_by_particle.to(torch.int64)
        eligible = ((applied_i64 > 0)
                    & torch.isfinite(before_jump).all(dim=-1)
                    & self.box.contains(before_jump))
        post_finite = torch.isfinite(after_jump).all(dim=-1)
        boundary_exit = (eligible & post_finite
                         & (~self.box.contains(after_jump)))
        denominator = applied_i64[eligible].sum()
        clipped = applied_i64[boundary_exit].sum()
        self._acc_cumulative(
            "jump_boundary_clip_count_cumulative", clipped)
        self._acc_cumulative(
            "jump_boundary_applied_count_cumulative", denominator)
        self._acc_cumulative_ratio(
            "jump_boundary_clip_fraction_per_applied_jump_cumulative",
            clipped, denominator)

    def _step_full(self) -> None:
        g = self.pot.grad(self.x)
        b = -g
        if self.score is not None:
            S, sdiag = self.score(self.x)
            b = b + S
            self._record_score_diagnostics(sdiag)
        xi = torch.randn(self.x.shape, generator=self.gen_diff,
                         device=self.x.device, dtype=self.x.dtype)
        before_jump = (self.x + self.dt * tame(b, self.dt, self.drift_cap)
                       + self._noise * xi)
        x1, applied_counts, sampled_counts = apply_poisson_jumps(
            before_jump, self.law, self.lam, self.dt, self.gen_jump,
            K_MAX_JUMPS, return_sampled_counts=True)
        self._record_jump_boundary(before_jump, x1, applied_counts)
        self._record_jump_counts(sampled_counts, applied_counts)
        self.x = self._clip_state(x1)
        self._acc("jump_count_mean", applied_counts.mean())

    def _step_atomic(self) -> None:
        """RA step: one displacement R_n per particle drives score AND jump.

        Conditions enforced (RA invariance, formulation note 17.2-17.4):
        (i)  R_n is drawn at the START of the step, from the SHARED jump stream
             (raw-CP score=None and RA-LSC score-set are pathwise coupled on
             (R_n, M_n)); law.sample takes no state, so R_n is independent of x;
        (ii) the SAME R_n is passed to the score and used for the jump;
        (iii) the score drift acts even when M_n == 0 (it is continuous drift,
             not part of the jump).
        """
        n = self.x.shape[0]
        R_n = self.law.sample(n, self.gen_jump)                  # (N, d)
        M_n = torch.poisson(torch.full((n,), self.lam * self.dt,
                                       device=self.x.device, dtype=self.x.dtype),
                            generator=self.gen_jump)             # (N,)
        g = self.pot.grad(self.x)
        b = -g
        if self.score is not None:
            S, sdiag = self.score.score_for_shift(self.x, R_n)   # SAME R_n
            b = b + S
            self._record_score_diagnostics(sdiag)
        xi = torch.randn(self.x.shape, generator=self.gen_diff,
                         device=self.x.device, dtype=self.x.dtype)
        before_jump = (self.x + self.dt * tame(b, self.dt, self.drift_cap)
                       + self._noise * xi)
        # M_n >= 2 jumps in one step share the single R_n (coincides with
        # full-law CP only as h -> 0; negligible at lam*dt ~ 5e-3).
        x1 = before_jump + M_n.unsqueeze(1) * R_n
        self._record_jump_boundary(before_jump, x1, M_n)
        self._record_jump_counts(M_n, M_n)
        self.x = self._clip_state(x1)
        self._acc("jump_count_mean", M_n.mean())

    def _step_paired_multiatom(self) -> None:
        """Paired MA step using one random finite measure per particle.

        Draw the state-independent bank ``R_{n,a} ~ q_a`` at the start of the
        step, then use it in both the weighted score and atomwise jump process

            N_{n,a} ~ Pois(lam * w_a * dt),
            Delta J_n = sum_a N_{n,a} R_{n,a}.

        Conditional on the frozen bank, score and jumps correspond to exactly
        the same measure ``lam * sum_a w_a delta_{R_{n,a}}``. Counts are not
        truncated: sampled multiplicities are applied exactly.
        """
        n = self.x.shape[0]
        R = self.score.sample_bank(n, self.gen_jump)             # (N, A, d)
        # Bank weights come from the score: for a shell bank they are the law's
        # atom masses, for an i.i.d. bank over a continuous law they are 1/A.
        rates = (self.lam * self.dt * self.score.weights).view(1, -1).expand(n, -1)
        counts = torch.poisson(rates, generator=self.gen_jump)   # (N, A)

        g = self.pot.grad(self.x)
        S, sdiag = self.score.score_for_bank(self.x, R)          # SAME R
        b = -g + S
        self._record_score_diagnostics(sdiag)

        xi = torch.randn(self.x.shape, generator=self.gen_diff,
                         device=self.x.device, dtype=self.x.dtype)
        before_jump = (self.x + self.dt * tame(b, self.dt, self.drift_cap)
                       + self._noise * xi)
        x1 = before_jump + (counts.unsqueeze(-1) * R).sum(dim=1)
        self._record_jump_boundary(before_jump, x1, counts)
        self._record_jump_counts(counts, counts)
        self.x = self._clip_state(x1)
        self._acc("jump_count_mean", counts.sum(dim=1).mean())
