"""The samplers: ULA, MALA, FLA, ULD (BAOAB), PT, Raw-CP, LSC-CP, LSC-CP-RA(A).

Taming
------
Every method that supports taming runs two variants. The canonical variant uses
the bare drift ``b``; the tamed variant uses

    b_c(x) = b(x) / (1 + dt ||b(x)|| / c).

For MALA and PT this changes the proposal, so the Metropolis-Hastings ratio uses
the ACTUAL tamed proposal density in both directions, with the reverse drift
``b_c(y)`` recomputed at the proposal point. Taming is never switched off to
avoid the proposal-density question, and the two variants are calibrated
separately.

Boundary rule
-------------
The targets live on R^d. Where a finite numerical box is needed it is enforced
by ONE rule shared by every method: a proposal that leaves the box (or turns
non-finite) is rejected and the state is kept. Nothing is clipped anywhere.

Discipline inside ``step()``
----------------------------
No ``.item()``, no ``.cpu()``, no printing, no host synchronisation. Diagnostics
accumulate as device tensors and are popped at checkpoints.
"""
from __future__ import annotations

import math

import torch

from .jumps import K_MAX_JUMPS, full_law_jump_increment, iid_bank_jump_increment


def tamed_drift(b: torch.Tensor, dt: float, cap: float | None) -> torch.Tensor:
    """Tamed drift; the displacement ``dt * b_c`` is bounded by ``cap``.

    ``cap is None`` means the canonical variant: the drift is returned
    unchanged, bit-for-bit.

    The norm is computed overflow-safely. With the Levy score ``||b||`` can
    exceed ``exp(354)``, where a plain ``b.norm()`` overflows to infinity and
    would silently zero the tamed drift.
    """
    if cap is None:
        return b
    scale = b.abs().amax(dim=-1, keepdim=True).clamp(min=1.0)
    norm = scale * (b / scale).norm(dim=-1, keepdim=True)
    return b / (1.0 + dt * norm / float(cap))


# ------------------------------------------------------------------ boxes
class RectBox:
    """Axis-aligned box in the sampling coordinates."""

    kind = "rect"

    def __init__(self, lo, hi, device, dtype=torch.float64) -> None:
        self.lo = torch.as_tensor(lo, dtype=dtype, device=device)
        self.hi = torch.as_tensor(hi, dtype=dtype, device=device)

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        return ((x >= self.lo) & (x <= self.hi)).all(dim=-1)

    def describe(self) -> dict:
        return {"kind": self.kind, "lo": self.lo.tolist(),
                "hi": self.hi.tolist()}


class LatentRectBox:
    """Box specified in latent coordinates ``z = x B^{-T}`` (E3)."""

    kind = "latent_rect"

    def __init__(self, lo, hi, potential, dtype=torch.float64) -> None:
        device = potential.B.device
        self.lo = torch.as_tensor(lo, dtype=dtype, device=device)
        self.hi = torch.as_tensor(hi, dtype=dtype, device=device)
        self.potential = potential

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        z = self.potential.to_latent(x)
        return ((z >= self.lo) & (z <= self.hi)).all(dim=-1)

    def describe(self) -> dict:
        return {"kind": self.kind, "latent_lo": self.lo.tolist(),
                "latent_hi": self.hi.tolist()}


class UnboundedBox:
    """No numerical box at all; every finite proposal is admissible."""

    kind = "unbounded"

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)

    def describe(self) -> dict:
        return {"kind": self.kind}


# ------------------------------------------------------------ diagnostics
class SamplerBase:
    """Shared diagnostics, boundary handling, and checkpoint plumbing."""

    name = "base"
    family = "base"

    def __init__(self, *, target, streams, n_per_seed: int, dt: float,
                 tame_cap: float | None, box) -> None:
        self.target = target
        self.streams = streams
        self.n_per_seed = int(n_per_seed)
        self.n_seeds = len(streams.seeds)
        self.dt = float(dt)
        self.tame_cap = None if tame_cap is None else float(tame_cap)
        self.box = box
        self._interval_sums: dict[str, torch.Tensor] = {}
        self._interval_counts: dict[str, int] = {}
        self._interval_maxima: dict[str, torch.Tensor] = {}
        self._cumulative: dict[str, torch.Tensor] = {}
        self._ratio_numerators: dict[str, torch.Tensor] = {}
        self._ratio_denominators: dict[str, torch.Tensor] = {}
        self._static: dict[str, int | float | str | bool] = {
            "tamed": self.tame_cap is not None,
            "tame_cap": float("nan") if self.tame_cap is None else self.tame_cap,
            "boundary_rule": "reject",
        }

    @property
    def tamed(self) -> bool:
        return self.tame_cap is not None

    # -- accumulation ------------------------------------------------------
    def _accumulate(self, key: str, value: torch.Tensor) -> None:
        if key in self._interval_sums:
            self._interval_sums[key] = self._interval_sums[key] + value
            self._interval_counts[key] += 1
        else:
            self._interval_sums[key] = value.clone()
            self._interval_counts[key] = 1

    def _accumulate_max(self, key: str, value: torch.Tensor) -> None:
        if key in self._interval_maxima:
            self._interval_maxima[key] = torch.maximum(
                self._interval_maxima[key], value)
        else:
            self._interval_maxima[key] = value.clone()

    def _accumulate_cumulative(self, key: str, value: torch.Tensor) -> None:
        if key in self._cumulative:
            self._cumulative[key] = self._cumulative[key] + value
        else:
            self._cumulative[key] = value.clone()

    def _accumulate_ratio(self, key: str, numerator: torch.Tensor,
                          denominator: torch.Tensor) -> None:
        if key in self._ratio_numerators:
            self._ratio_numerators[key] = self._ratio_numerators[key] + numerator
            self._ratio_denominators[key] = (self._ratio_denominators[key]
                                             + denominator)
        else:
            self._ratio_numerators[key] = numerator.clone()
            self._ratio_denominators[key] = denominator.clone()

    def pop_diagnostics(self) -> dict[str, float | int | str | bool]:
        """Host synchronisation happens HERE only, never inside ``step()``."""
        out = {key: (self._interval_sums[key] / self._interval_counts[key]).item()
               for key in self._interval_sums}
        out.update({key: value.item()
                    for key, value in self._interval_maxima.items()})
        out.update({key: value.item() for key, value in self._cumulative.items()})
        for key, numerator in self._ratio_numerators.items():
            denominator = self._ratio_denominators[key]
            out[key] = torch.where(
                denominator > 0,
                numerator / denominator,
                torch.zeros_like(numerator, dtype=torch.float64)).item()
        out.update(self._static)
        self._interval_sums, self._interval_counts = {}, {}
        self._interval_maxima = {}
        return out

    def reset_diagnostics(self) -> None:
        """Reset interval and cumulative accumulators at a declared pilot boundary."""
        self._interval_sums, self._interval_counts = {}, {}
        self._interval_maxima = {}
        self._cumulative = {}
        self._ratio_numerators, self._ratio_denominators = {}, {}

    # -- shared boundary rule ---------------------------------------------
    def _apply_reject(self, current: torch.Tensor,
                      candidate: torch.Tensor) -> torch.Tensor:
        """Keep the state whenever a proposal leaves the box or is non-finite."""
        finite = torch.isfinite(candidate).all(dim=-1)
        inside = self.box.contains(candidate)
        accept = finite & inside
        self._record_boundary(finite, inside)
        return torch.where(accept.unsqueeze(-1), candidate, current)

    def _record_boundary(self, finite: torch.Tensor,
                         inside: torch.Tensor) -> None:
        rejected_outside = (finite & (~inside)).to(torch.int64).sum()
        nonfinite = (~finite).to(torch.int64).sum()
        proposed = torch.as_tensor(finite.numel(), dtype=torch.int64,
                                   device=finite.device)
        self._accumulate_cumulative("boundary_reject_count_cumulative",
                                    rejected_outside)
        self._accumulate_cumulative("nonfinite_proposal_count_cumulative",
                                    nonfinite)
        self._accumulate_ratio("boundary_reject_fraction_cumulative",
                               rejected_outside, proposed)
        self._accumulate_ratio("nonfinite_proposal_fraction_cumulative",
                               nonfinite, proposed)

    def _record_score_diagnostics(self, diagnostics: dict) -> None:
        self._accumulate("m_clip_fraction", diagnostics["m_clip_fraction"])
        self._accumulate_max("max_log_magnitude",
                             diagnostics["max_log_magnitude"])
        self._accumulate_cumulative("score_clip_count_cumulative",
                                    diagnostics["m_clip_count"])
        self._accumulate_ratio("score_clip_fraction_cumulative",
                               diagnostics["m_clip_count"],
                               diagnostics["m_clip_total"])

    def _record_acceptance(self, accepted: torch.Tensor,
                           proposed: torch.Tensor, prefix: str = "mh") -> None:
        self._accumulate(f"{prefix}_accept",
                         accepted.to(torch.float64) / proposed.to(torch.float64))
        self._accumulate_cumulative(f"{prefix}_accept_count_cumulative", accepted)
        self._accumulate_cumulative(f"{prefix}_proposal_count_cumulative",
                                    proposed)
        self._accumulate_ratio(f"{prefix}_accept_fraction_cumulative",
                               accepted, proposed)

    # -- interface ---------------------------------------------------------
    def positions(self) -> torch.Tensor:
        return self.x

    def step(self) -> None:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"method": self.name, "family": self.family, "dt": self.dt,
                "tamed": self.tamed, "tame_cap": self.tame_cap,
                "boundary": self.box.describe(), "boundary_rule": "reject"}


# ------------------------------------------------------------------- ULA
class ULASampler(SamplerBase):
    """Overdamped Langevin, Euler-Maruyama, no accept/reject."""

    name = "ULA"
    family = "ULA"

    def __init__(self, *, target, streams, x0, n_per_seed, dt, tame_cap, box,
                 **_) -> None:
        super().__init__(target=target, streams=streams, n_per_seed=n_per_seed,
                         dt=dt, tame_cap=tame_cap, box=box)
        self.x = x0.clone()
        self._noise_scale = math.sqrt(2.0 * target.eps * self.dt)

    def step(self) -> None:
        b = self.target.force(self.x)
        xi = self.streams.randn("diffusion_gen",
                                (self.n_per_seed, self.x.shape[-1]))
        candidate = (self.x + self.dt * tamed_drift(b, self.dt, self.tame_cap)
                     + self._noise_scale * xi)
        self.x = self._apply_reject(self.x, candidate)


# ------------------------------------------------------------------ MALA
class MALASampler(SamplerBase):
    """MALA with the actual tamed proposal density.

    Proposal, with ``b_c`` the tamed (or, canonically, bare) drift:

        Y = X + dt b_c(X) + sqrt(2 dt / beta) xi
        q_c(y|x) = N(y; x + dt b_c(x), (2 dt / beta) I)
        log alpha = -beta [V(y) - V(x)] + log q_c(x|y) - log q_c(y|x)

    ``b_c(y)`` is recomputed at ``y``, so the reverse density is the density the
    sampler would actually have used from ``y``. Conditional on the current
    state the proposal is Gaussian with known mean and isotropic covariance, so
    no Jacobian term appears.

    Proposals are never moved before the accept step -- clipping first would
    silently break exactness. Out-of-box proposals are rejected, which is valid
    Metropolis-Hastings for the box-restricted target and is the same boundary
    rule every other method uses.
    """

    name = "MALA"
    family = "MALA"

    def __init__(self, *, target, streams, x0, n_per_seed, dt, tame_cap, box,
                 **_) -> None:
        super().__init__(target=target, streams=streams, n_per_seed=n_per_seed,
                         dt=dt, tame_cap=tame_cap, box=box)
        self.x = x0.clone()
        self.beta = float(target.beta)
        #: proposal variance 2 dt / beta = 2 eps dt, matching the ULA step
        self.proposal_variance = 2.0 * self.dt / self.beta
        self.Vx, self.bx = target.value_and_force(self.x)

    def _log_proposal_density_difference(self, x, y, drift_x, drift_y):
        """``log q_c(x|y) - log q_c(y|x)`` for the isotropic tamed Gaussian."""
        mean_forward = x + self.dt * drift_x
        mean_reverse = y + self.dt * drift_y
        forward = ((y - mean_forward) ** 2).sum(-1)
        reverse = ((x - mean_reverse) ** 2).sum(-1)
        return (forward - reverse) / (2.0 * self.proposal_variance)

    def step(self) -> None:
        x = self.x
        drift_x = tamed_drift(self.bx, self.dt, self.tame_cap)
        xi = self.streams.randn("diffusion_gen", (self.n_per_seed, x.shape[-1]))
        y = x + self.dt * drift_x + math.sqrt(self.proposal_variance) * xi
        Vy, by = self.target.value_and_force(y)
        drift_y = tamed_drift(by, self.dt, self.tame_cap)
        log_alpha = (-self.beta * (Vy - self.Vx)
                     + self._log_proposal_density_difference(
                         x, y, drift_x, drift_y))
        u = self.streams.rand("mh_uniform_gen", (self.n_per_seed,))
        finite = torch.isfinite(y).all(dim=-1) & torch.isfinite(log_alpha)
        inside = self.box.contains(y)
        self._record_boundary(finite, inside)
        accept = (torch.log(u) < log_alpha) & inside & finite
        self.x = torch.where(accept.unsqueeze(-1), y, x)
        self.Vx = torch.where(accept, Vy, self.Vx)
        self.bx = torch.where(accept.unsqueeze(-1), by, self.bx)
        proposed = torch.as_tensor(accept.numel(), dtype=torch.int64,
                                   device=accept.device)
        self._record_acceptance(accept.to(torch.int64).sum(), proposed)


# ------------------------------------------------------------------- FLA
class FLASampler(SamplerBase):
    """Fractional Langevin: the UNCORRECTED nonlocal comparator.

        X <- X + dt b_c(-c_alpha grad U) + dt^{1/alpha} xi_alpha,   U = beta V,
        c_alpha = Gamma(alpha - 1) / Gamma(alpha / 2)^2.

    Heavy tails cross barriers, but the invariant law is not the target.
    """

    name = "FLA"
    family = "FLA"

    def __init__(self, *, target, streams, x0, n_per_seed, dt, tame_cap, box,
                 alpha: float = 1.7, **_) -> None:
        super().__init__(target=target, streams=streams, n_per_seed=n_per_seed,
                         dt=dt, tame_cap=tame_cap, box=box)
        self.x = x0.clone()
        self.alpha = float(alpha)
        self.beta = float(target.beta)
        self.c_alpha = (math.gamma(self.alpha - 1.0)
                        / math.gamma(self.alpha / 2.0) ** 2)
        self._noise_scale = self.dt ** (1.0 / self.alpha)
        self._static["alpha"] = self.alpha

    def step(self) -> None:
        # force = -grad V, so -c_alpha grad U = c_alpha beta * force.
        b = self.c_alpha * self.beta * self.target.force(self.x)
        xi = self.streams.symmetric_alpha_stable(
            "stable_noise_gen", (self.n_per_seed, self.x.shape[-1]), self.alpha)
        candidate = (self.x + self.dt * tamed_drift(b, self.dt, self.tame_cap)
                     + self._noise_scale * xi)
        self.x = self._apply_reject(self.x, candidate)


# ------------------------------------------------------------------- ULD
class BAOABSampler(SamplerBase):
    """Underdamped Langevin dynamics, BAOAB splitting.

    ULD is the method name used in every figure and table; BAOAB is only the
    integrator. Unit mass, friction ``gamma``. The O-step is the exact
    Ornstein-Uhlenbeck solution. The trailing force is cached as the next step's
    leading B, so the oracle counters see one force call per step because the
    sampler makes one, not because a formula says so.

    This carries O(dt^2) configurational bias: there is no accept/reject step.
    """

    name = "ULD"
    family = "ULD"
    integrator = "BAOAB"

    def __init__(self, *, target, streams, x0, n_per_seed, dt, tame_cap, box,
                 gamma: float = 1.0, **_) -> None:
        super().__init__(target=target, streams=streams, n_per_seed=n_per_seed,
                         dt=dt, tame_cap=tame_cap, box=box)
        self.x = x0.clone()
        self.gamma = float(gamma)
        eps = target.eps
        self.c1 = math.exp(-self.gamma * self.dt)
        self.c2 = math.sqrt(eps * (1.0 - self.c1 ** 2))
        self.p = math.sqrt(eps) * streams.randn(
            "diffusion_gen", (self.n_per_seed, x0.shape[-1]))
        self.f = tamed_drift(target.force(self.x), self.dt, self.tame_cap)
        self._static["gamma"] = self.gamma
        self._static["integrator"] = self.integrator

    def step(self) -> None:
        dt = self.dt
        p = self.p + 0.5 * dt * self.f                                  # B
        q = self.x + 0.5 * dt * p                                       # A
        xi = self.streams.randn("diffusion_gen",
                                (self.n_per_seed, q.shape[-1]))
        p = self.c1 * p + self.c2 * xi                                  # O
        q = q + 0.5 * dt * p                                            # A
        finite = torch.isfinite(q).all(dim=-1)
        inside = self.box.contains(q)
        accept = finite & inside
        self._record_boundary(finite, inside)
        # One shared reject rule: an excursion keeps the whole previous state,
        # position and momentum together.
        q = torch.where(accept.unsqueeze(-1), q, self.x)
        f = tamed_drift(self.target.force(q), dt, self.tame_cap)         # B
        p = torch.where(accept.unsqueeze(-1), p + 0.5 * dt * f, self.p)
        self.x, self.p, self.f = q, p, f


# -------------------------------------------------------------------- PT
class ParallelTemperingSampler(SamplerBase):
    """Parallel tempering with tamed MALA inside each replica.

    Replica ``k`` runs at its own inverse temperature ``beta_k``:

        Y_k = X_k + dt b_c(X_k) + sqrt(2 dt / beta_k) xi_k,

    and its acceptance uses ``beta_k`` with the actual tamed forward and reverse
    proposal densities, ``b_c(Y_k)`` recomputed at ``Y_k``.

    Taming changes only the local kernel. The swap of adjacent replicas is a
    deterministic involution on a product target, so

        alpha_swap = 1 ^ exp[(beta_i - beta_j) (V(X_i) - V(X_j))]

    regardless of the tame flag. Potentials are cached by the local kernel, so
    swaps cost no evaluations. State is ``(K, S*N, d)``; metrics use the cold
    replica, and the oracle counters cover all replicas by construction.
    """

    name = "PT"
    family = "PT"

    def __init__(self, *, target, streams, x0, n_per_seed, dt, tame_cap, box,
                 betas: torch.Tensor, n_swap: int = 10, **_) -> None:
        super().__init__(target=target, streams=streams, n_per_seed=n_per_seed,
                         dt=dt, tame_cap=tame_cap, box=box)
        self.betas = betas
        self.n_replicas = int(betas.shape[0])
        self.n_swap = int(n_swap)
        self.x = x0.unsqueeze(0).repeat(self.n_replicas, 1, 1)
        #: per-replica proposal variance 2 dt / beta_k
        self.proposal_variance = (2.0 * self.dt / betas).view(-1, 1)
        self.Vx, self.bx = target.value_and_force(self.x)
        self._step_count = 0
        n_walkers = int(self.x.shape[1])
        self._walker_ids = torch.arange(
            self.n_replicas, dtype=torch.long, device=self.x.device
        )[:, None].expand(self.n_replicas, n_walkers).clone()
        self._seen_hot = torch.zeros(
            self.n_replicas, n_walkers, dtype=torch.bool, device=self.x.device)
        self._static["n_replicas"] = self.n_replicas
        self._static["n_swap"] = self.n_swap

    def step(self) -> None:
        x = self.x
        drift_x = tamed_drift(self.bx, self.dt, self.tame_cap)
        # Per-seed shape (K, N, d): the replica index is a leading batch
        # dimension, so seed blocks still concatenate along the particle axis.
        xi = self.streams.randn(
            "diffusion_gen", (self.n_replicas, self.n_per_seed, x.shape[-1]),
            cat_dim=1)
        y = (x + self.dt * drift_x
             + torch.sqrt(self.proposal_variance).unsqueeze(-1) * xi)
        Vy, by = self.target.value_and_force(y)
        drift_y = tamed_drift(by, self.dt, self.tame_cap)
        mean_forward = x + self.dt * drift_x
        mean_reverse = y + self.dt * drift_y
        forward = ((y - mean_forward) ** 2).sum(-1)
        reverse = ((x - mean_reverse) ** 2).sum(-1)
        log_alpha = (-self.betas.view(-1, 1) * (Vy - self.Vx)
                     + (forward - reverse) / (2.0 * self.proposal_variance))
        u = self.streams.rand("mh_uniform_gen",
                              (self.n_replicas, self.n_per_seed), cat_dim=1)
        finite = torch.isfinite(y).all(dim=-1) & torch.isfinite(log_alpha)
        inside = self.box.contains(y)
        self._record_boundary(finite, inside)
        accept = (torch.log(u) < log_alpha) & inside & finite
        self.x = torch.where(accept.unsqueeze(-1), y, x)
        self.Vx = torch.where(accept, Vy, self.Vx)
        self.bx = torch.where(accept.unsqueeze(-1), by, self.bx)
        proposed = torch.as_tensor(accept.numel(), dtype=torch.int64,
                                   device=accept.device)
        self._record_acceptance(accept.to(torch.int64).sum(), proposed)
        for replica in range(self.n_replicas):
            replica_proposed = torch.as_tensor(
                accept[replica].numel(), dtype=torch.int64,
                device=accept.device)
            self._record_acceptance(
                accept[replica].to(torch.int64).sum(), replica_proposed,
                prefix=f"replica_{replica}_mh")

        self._step_count += 1
        if self._step_count % self.n_swap == 0:
            self._swap_pass((self._step_count // self.n_swap) % 2)

    def _swap_pass(self, offset: int) -> None:
        accepted_counts, proposal_counts = [], []
        for i in range(offset, self.n_replicas - 1, 2):
            log_alpha = ((self.betas[i] - self.betas[i + 1])
                         * (self.Vx[i] - self.Vx[i + 1]))
            u = self.streams.rand("mh_uniform_gen", (self.n_per_seed,))
            swap = torch.log(u) < log_alpha
            swap_column = swap.unsqueeze(-1)
            # Materialise both branches before writing: x[i] and x[i+1] are views.
            new_xi = torch.where(swap_column, self.x[i + 1], self.x[i])
            new_xj = torch.where(swap_column, self.x[i], self.x[i + 1])
            self.x[i], self.x[i + 1] = new_xi, new_xj
            new_vi = torch.where(swap, self.Vx[i + 1], self.Vx[i])
            new_vj = torch.where(swap, self.Vx[i], self.Vx[i + 1])
            self.Vx[i], self.Vx[i + 1] = new_vi, new_vj
            new_bi = torch.where(swap_column, self.bx[i + 1], self.bx[i])
            new_bj = torch.where(swap_column, self.bx[i], self.bx[i + 1])
            self.bx[i], self.bx[i + 1] = new_bi, new_bj
            new_wi = torch.where(
                swap, self._walker_ids[i + 1], self._walker_ids[i])
            new_wj = torch.where(
                swap, self._walker_ids[i], self._walker_ids[i + 1])
            self._walker_ids[i], self._walker_ids[i + 1] = new_wi, new_wj
            accepted_counts.append(swap.to(torch.int64).sum())
            proposal_counts.append(torch.as_tensor(
                swap.numel(), dtype=torch.int64, device=swap.device))
        if accepted_counts:
            self._record_acceptance(torch.stack(accepted_counts).sum(),
                                    torch.stack(proposal_counts).sum(),
                                    prefix="swap")
        self._record_round_trips()

    def _record_round_trips(self) -> None:
        """Count labelled walkers that visit hot and subsequently return cold."""
        columns = torch.arange(self._walker_ids.shape[1],
                               device=self._walker_ids.device)
        hot_ids = self._walker_ids[-1]
        self._seen_hot[hot_ids, columns] = True
        cold_ids = self._walker_ids[0]
        returned = self._seen_hot[cold_ids, columns]
        completed = returned.to(torch.int64).sum()
        self._seen_hot[cold_ids, columns] = False
        opportunities = torch.as_tensor(
            columns.numel(), dtype=torch.int64, device=columns.device)
        self._accumulate_cumulative(
            "round_trip_count_cumulative", completed)
        self._accumulate_ratio(
            "round_trip_rate_cumulative", completed, opportunities)

    def reset_diagnostics(self) -> None:
        super().reset_diagnostics()
        self._seen_hot.zero_()

    def positions(self) -> torch.Tensor:
        return self.x[0]


def geometric_ladder(beta_max: float, beta_min: float, n_replicas: int,
                     device) -> torch.Tensor:
    ratio = (beta_min / beta_max) ** (1.0 / (n_replicas - 1))
    return beta_max * torch.as_tensor(
        [ratio ** k for k in range(n_replicas)], dtype=torch.float64,
        device=device)


# ---------------------------------------------- compound-Poisson family
class CompoundPoissonSampler(SamplerBase):
    """Raw-CP, LSC-CP, and LSC-CP-RA(A) on one fixed single-step order.

        score at X_n  ->  drift-diffusion  ->  compound-Poisson jump

    with ``b = -grad V`` (Raw-CP) or ``b = -grad V + S(X_n)`` (the corrected
    variants). No splitting-order ablation is run; the order is fixed and
    recorded in the resolved config.

    ``jump_mode="full_law"`` draws ``N ~ Poisson(lambda dt)`` displacements iid
    from the full law. ``jump_mode="iid_bank"`` is LSC-CP-RA(A): ONE bank
    ``R_1..R_A`` is drawn iid from the full law at the start of the step, handed
    to the score, and then reused for the increment with
    ``N_j ~ Poisson(lambda dt / A)``. Score and noise therefore see the identical
    random empirical measure, and the bank is refreshed every step.
    """

    family = "CP"

    def __init__(self, *, target, streams, x0, n_per_seed, dt, tame_cap, box,
                 law, intensity: float, score=None, name: str = "Raw-CP",
                 jump_mode: str = "full_law", bank_size: int = 1, **_) -> None:
        super().__init__(target=target, streams=streams, n_per_seed=n_per_seed,
                         dt=dt, tame_cap=tame_cap, box=box)
        if jump_mode not in ("full_law", "iid_bank"):
            raise ValueError(f"unknown jump_mode {jump_mode!r}")
        if jump_mode == "iid_bank" and score is not None:
            if getattr(score, "estimator_type", None) != "iid_random_atomic":
                raise ValueError(
                    "iid_bank jumps require the iid random-atomic score")
            if int(score.bank_size) != int(bank_size):
                raise ValueError(
                    "the sampler bank size and the score bank size must agree")
        if jump_mode == "full_law" and score is not None:
            if getattr(score, "estimator_type", None) == "iid_random_atomic":
                raise ValueError(
                    "the iid random-atomic score must use iid_bank jumps")
        self.x = x0.clone()
        self.law = law
        self.intensity = float(intensity)
        self.score = score
        self.name = str(name)
        self.family = "LSC-CP-RA" if jump_mode == "iid_bank" else (
            "LSC-CP" if score is not None else "CP")
        self.jump_mode = jump_mode
        self.bank_size = int(bank_size)
        self._noise_scale = math.sqrt(2.0 * target.eps * self.dt)
        self._static["score_evaluation"] = "pre_step"
        self._static["splitting"] = "drift_diffusion_then_jump"
        self._static["jump_mode"] = jump_mode
        if jump_mode == "full_law":
            self._static["jump_cap_k"] = int(K_MAX_JUMPS)
        else:
            self._static["bank_size"] = self.bank_size
            self._static["bank_refresh_policy"] = "every_step"
            self._static["bank_shared_between_score_and_noise"] = True

    def step(self) -> None:
        if self.jump_mode == "iid_bank":
            self._step_iid_bank()
        else:
            self._step_full_law()

    # -- Raw-CP and full LSC-CP -------------------------------------------
    def _step_full_law(self) -> None:
        b = self.target.force(self.x)
        if self.score is not None:
            correction, diagnostics = self.score(self.x)
            b = b + correction
            self._record_score_diagnostics(diagnostics)
        xi = self.streams.randn("diffusion_gen",
                                (self.n_per_seed, self.x.shape[-1]))
        drifted = (self.x + self.dt * tamed_drift(b, self.dt, self.tame_cap)
                   + self._noise_scale * xi)
        increment, applied, sampled = full_law_jump_increment(
            self.law, self.streams, self.n_per_seed, self.intensity, self.dt)
        candidate = drifted + increment
        self._record_jump_counts(sampled, applied)
        self.x = self._apply_reject(self.x, candidate)
        self._accumulate("jump_count_mean", applied.mean())

    # -- LSC-CP-RA(A) ------------------------------------------------------
    def _step_iid_bank(self) -> None:
        """One iid bank per particle per step, shared by the score and the jump.

        The bank is drawn before the score is evaluated and takes no state
        argument, so its independence from the current state is structural.
        """
        bank = self.law.sample_bank(self.streams, "jump_bank_gen",
                                    self.n_per_seed, self.bank_size)
        b = self.target.force(self.x)
        if self.score is not None:
            correction, diagnostics = self.score.score_for_bank(self.x, bank)
            b = b + correction
            self._record_score_diagnostics(diagnostics)
        xi = self.streams.randn("diffusion_gen",
                                (self.n_per_seed, self.x.shape[-1]))
        drifted = (self.x + self.dt * tamed_drift(b, self.dt, self.tame_cap)
                   + self._noise_scale * xi)
        increment, counts = iid_bank_jump_increment(
            bank, self.streams, self.n_per_seed, self.intensity, self.dt)
        candidate = drifted + increment
        total = counts.sum(dim=1)
        self._record_jump_counts(total, total)
        self.x = self._apply_reject(self.x, candidate)
        self._accumulate("jump_count_mean", total.mean())

    def _record_jump_counts(self, sampled: torch.Tensor,
                            applied: torch.Tensor) -> None:
        sampled_total = sampled.to(torch.int64).sum()
        applied_total = applied.to(torch.int64).sum()
        particle_time = torch.as_tensor(
            sampled.shape[0] * self.dt, dtype=torch.float64,
            device=sampled.device)
        self._accumulate_cumulative("jump_count_cumulative", sampled_total)
        self._accumulate_cumulative("jump_count_applied_cumulative",
                                    applied_total)
        self._accumulate_ratio("jump_rate_per_particle_time_cumulative",
                               sampled_total, particle_time)
        if self.jump_mode == "full_law":
            excess = sampled.to(torch.int64) - applied.to(torch.int64)
            cap_hits = (excess > 0).to(torch.int64).sum()
            proposals = torch.as_tensor(sampled.shape[0], dtype=torch.int64,
                                        device=sampled.device)
            self._accumulate_cumulative("jump_cap_hit_count_cumulative",
                                        cap_hits)
            self._accumulate_cumulative("jump_cap_excess_count_cumulative",
                                        excess.sum())
            self._accumulate_ratio("jump_cap_hit_fraction_cumulative",
                                   cap_hits, proposals)

    def describe(self) -> dict:
        record = super().describe()
        record.update({
            "jump_mode": self.jump_mode,
            "intensity": self.intensity,
            "jump_law": self.law.describe(),
            "score_evaluation": "pre_step",
            "splitting": "drift_diffusion_then_jump",
        })
        if self.score is not None:
            record["score"] = self.score.describe()
        return record
