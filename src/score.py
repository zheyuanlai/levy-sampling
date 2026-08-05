"""The stationary Levy score and its iid random-atomic estimator.

    S_{nu,beta}(x) = -lambda int rho(dr) r int_0^1 exp[-beta (V(x - theta r) - V(x))] dtheta

Naive evaluation overflows: the per-direction theta integral is strictly
positive but spans hundreds of orders of magnitude. Everything is accumulated in
log space -- log-sum-exp over quadrature nodes, then the per-particle maximum
exponent ``M(x)`` is factored out to leave an O(1) direction vector ``v(x)``:

    S(x) = -exp(min(M(x), M_MAX)) v(x).

Because every sampler tames its drift, when ``||S||`` is astronomical only its
direction matters, and ``v`` preserves that direction exactly. There is no clip
on vector components and none on the raw log-ratio; the only cap is ``M_MAX``,
and the fraction of particles reaching it is recorded.

Two estimators live here.

``DeterministicShellScore`` integrates the declared law with Gauss-Legendre
rules, costing ``J * q_theta`` extra potential evaluations per particle per step.

``IIDRandomAtomicScore`` is the LSC-CP-RA(A) estimator. The sampler draws one
bank ``R_1..R_A`` iid from the full normalized law and hands the SAME tensor to
this score and to the compound-Poisson increment, so both use the identical
random empirical Levy measure

    nu_hat_A = (lambda / A) sum_j delta_{R_j},

at a cost of ``A * q_theta`` extra potential evaluations per particle per step.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .jumps import gauss_legendre_01

#: Cap on the log-magnitude of the score. Only the magnitude is capped; the
#: direction is exact.
M_MAX = 600.0

SQRT_PI_OVER_2 = math.sqrt(math.pi / 2.0)
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
SQRT2 = math.sqrt(2.0)


def _log_parts(log_terms: torch.Tensor, vectors: torch.Tensor,
               log_prefactor: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor out the per-particle maximum exponent.

    ``log_terms`` is ``(N, J)`` log magnitudes and ``vectors`` is ``(J, d)``.
    Returns ``(M, v)`` with ``S = -exp(M) v`` exactly. Cancellation between
    ``+r`` and ``-r`` atoms inside ``v`` is genuine physics and is preserved.
    """
    M = log_terms.max(dim=1).values + log_prefactor
    scaled = torch.exp(log_terms - (M - log_prefactor).unsqueeze(1))
    return M, scaled @ vectors


def _clip_diagnostics(M: torch.Tensor, m_max: float) -> dict:
    """Device-resident clipping count, denominator, fraction, and maximum."""
    clipped = M > m_max
    count = clipped.to(torch.int64).sum()
    total = torch.as_tensor(M.numel(), dtype=torch.int64, device=M.device)
    return {
        "m_clip_count": count,
        "m_clip_total": total,
        "m_clip_fraction": count.to(torch.float64) / total,
        "max_log_magnitude": M.max(),
    }


def _finalize(M: torch.Tensor, v: torch.Tensor,
              m_max: float) -> tuple[torch.Tensor, dict]:
    S = -torch.exp(torch.clamp(M, max=m_max)).unsqueeze(1) * v
    return S, _clip_diagnostics(M, m_max)


class DeterministicShellScore:
    """Full LSC-CP score: deterministic quadrature of the declared jump law.

    The theta and radial expectations are approximated by normalized
    Gauss-Legendre rules; the finite quadrature measure is not literally the
    sampler's continuous law, and the certificate measures the residual.
    """

    estimator_type = "deterministic_quadrature"

    def __init__(self, target, law, intensity: float, *, q_theta: int,
                 q_rho: int, m_max: float = M_MAX, chunk: int | None = None,
                 max_block_elements: int = 1 << 23, **law_quadrature_kwargs
                 ) -> None:
        self.target = target
        self.law = law
        self.intensity = float(intensity)
        self.beta = float(target.beta)
        self.m_max = float(m_max)
        device = target.device
        theta, w_theta = gauss_legendre_01(q_theta, device)
        shifts, log_weights = law.quadrature_shifts(
            q_rho=q_rho, **law_quadrature_kwargs)
        self.shifts = shifts
        self.log_shift_weights = log_weights
        self.log_theta_weights = torch.log(w_theta)
        self.q_theta = int(q_theta)
        self.n_shifts = int(shifts.shape[0])
        # All theta_p * r_j chord displacements, flattened once.
        self.chord_shifts = (theta.view(-1, 1, 1)
                             * shifts.view(1, *shifts.shape)).reshape(
                                 -1, shifts.shape[1])
        per_particle = max(1, self.q_theta * self.n_shifts)
        self.chunk = (int(chunk) if chunk is not None
                      else max(1, int(max_block_elements) // per_particle))

    @property
    def extra_potential_per_particle_step(self) -> int:
        return self.q_theta * self.n_shifts

    def _log_parts_block(self, x: torch.Tensor):
        n = x.shape[0]
        chord_delta = self.target.chord_value_delta(
            x, self.chord_shifts).reshape(n, self.q_theta, self.n_shifts)
        log_integral = torch.logsumexp(
            self.log_theta_weights.view(1, -1, 1) - self.beta * chord_delta,
            dim=1)
        log_terms = self.log_shift_weights.unsqueeze(0) + log_integral
        return _log_parts(log_terms, self.shifts, math.log(self.intensity))

    def log_parts(self, x: torch.Tensor):
        """Uncapped ``(M, v)``; blocked over particles, bit-identical unblocked."""
        n = x.shape[0]
        if n <= self.chunk:
            return self._log_parts_block(x)
        magnitudes, directions = [], []
        for start in range(0, n, self.chunk):
            M, v = self._log_parts_block(x[start:start + self.chunk])
            magnitudes.append(M)
            directions.append(v)
        return torch.cat(magnitudes, dim=0), torch.cat(directions, dim=0)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        M, v = self.log_parts(x)
        return _finalize(M, v, self.m_max)

    def describe(self) -> dict:
        return {
            "estimator_type": self.estimator_type,
            "q_theta": self.q_theta,
            "n_quadrature_shifts": self.n_shifts,
            "m_max": self.m_max,
            "extra_potential_per_particle_step":
                self.extra_potential_per_particle_step,
        }


class IIDRandomAtomicScore:
    """LSC-CP-RA(A): the score of a random empirical Levy measure.

    Given a bank ``R_1..R_A`` drawn iid from the full normalized law ``rho``,

        nu_hat_A = (lambda / A) sum_j delta_{R_j},
        S_A(x)   = -(lambda / A) sum_j R_j I(x, R_j),
        I(x, R)  = int_0^1 exp[-beta (V(x - theta R) - V(x))] dtheta.

    ``A`` is an ordinary Monte Carlo bank size, a hyperparameter of this one
    estimator family. This object never draws the bank: the sampler draws it,
    passes the identical tensor here and to the jump increment, and refreshes it
    every step. That keeps the bank independent of the current state and makes
    "score and noise share a bank" structural rather than a convention.
    """

    estimator_type = "iid_random_atomic"
    bank_refresh_policy = "every_step"
    bank_shared_between_score_and_noise = True

    def __init__(self, target, law, intensity: float, *, bank_size: int,
                 q_theta: int, m_max: float = M_MAX) -> None:
        self.target = target
        self.law = law
        self.intensity = float(intensity)
        self.beta = float(target.beta)
        self.bank_size = int(bank_size)
        self.q_theta = int(q_theta)
        self.m_max = float(m_max)
        if self.bank_size < 1:
            raise ValueError("bank size A must be a positive integer")
        if self.intensity <= 0.0 or self.beta <= 0.0 or self.q_theta < 1:
            raise ValueError("lambda, beta, and q_theta must be positive")
        theta, w_theta = gauss_legendre_01(q_theta, target.device)
        self.theta = theta
        self.log_theta_weights = torch.log(w_theta)
        # lambda / A is the constant weight of each bank atom.
        self.log_atom_weight = math.log(self.intensity / self.bank_size)

    @property
    def extra_potential_per_particle_step(self) -> int:
        return self.q_theta * self.bank_size

    def score_for_bank(self, x: torch.Tensor,
                       bank: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Score of the realised bank ``(N, A, d)`` at states ``x`` ``(N, d)``."""
        n, d = x.shape
        A, q_theta = self.bank_size, self.q_theta
        if bank.shape != (n, A, d):
            raise ValueError(
                f"bank must have shape {(n, A, d)}, got {tuple(bank.shape)}")
        if bank.device != x.device or bank.dtype != x.dtype:
            raise ValueError("bank must share device and dtype with the state")
        # chord points y = x - theta_p R_j  ->  (N, A, Qt, d)
        chord_points = (x.view(n, 1, 1, d)
                        - self.theta.view(1, 1, q_theta, 1) * bank.view(n, A, 1, d))
        chord_delta = self.target.chord_value_delta_pointwise(x, chord_points)
        log_integral = torch.logsumexp(
            self.log_theta_weights.view(1, 1, q_theta) - self.beta * chord_delta,
            dim=2)
        log_terms = self.log_atom_weight + log_integral            # (N, A)
        # One global accumulator per particle. Capping per atom before summing
        # would change the relative atom weights whenever clipping binds, and
        # the result would no longer be the score of the frozen bank.
        M = log_terms.max(dim=1).values
        scaled = torch.exp(log_terms - M.unsqueeze(1))
        v = (scaled.unsqueeze(-1) * bank).sum(dim=1)
        S = -torch.exp(torch.clamp(M, max=self.m_max)).unsqueeze(1) * v
        return S, _clip_diagnostics(M, self.m_max)

    def log_parts_for_frozen_bank(self, x: torch.Tensor, bank: torch.Tensor):
        """Uncapped ``(M, v)`` for one bank ``(A, d)`` shared by all states.

        Certificate-facing form satisfying ``S_R(x) = -exp(M(x)) v(x)``.
        """
        n, d = x.shape
        A, q_theta = self.bank_size, self.q_theta
        if bank.shape != (A, d):
            raise ValueError(
                f"frozen bank must have shape {(A, d)}, got {tuple(bank.shape)}")
        chord_points = (x.view(n, 1, 1, d)
                        - self.theta.view(1, 1, q_theta, 1) * bank.view(1, A, 1, d))
        chord_delta = self.target.chord_value_delta_pointwise(x, chord_points)
        log_integral = torch.logsumexp(
            self.log_theta_weights.view(1, 1, q_theta) - self.beta * chord_delta,
            dim=2)
        return _log_parts(log_integral, bank, self.log_atom_weight)

    def describe(self) -> dict:
        return {
            "estimator_type": self.estimator_type,
            "bank_size": self.bank_size,
            "bank_refresh_policy": self.bank_refresh_policy,
            "bank_shared_between_score_and_noise":
                self.bank_shared_between_score_and_noise,
            "bank_sampling": "iid_from_full_mixture",
            "q_theta": self.q_theta,
            "m_max": self.m_max,
            "extra_potential_per_particle_step":
                self.extra_potential_per_particle_step,
        }


# =============================================== closed-form E2 comparator
def _g_hat(z: torch.Tensor) -> torch.Tensor:
    """``sqrt(2/pi) - z erfcx(z/sqrt 2)`` for ``z >= 0``."""
    return SQRT_2_OVER_PI - z * torch.special.erfcx(z / SQRT2)


def _F(z: torch.Tensor) -> torch.Tensor:
    """``F(z) = z erf(z/sqrt 2) + sqrt(2/pi) exp(-z^2/2)``; ``F'`` is even."""
    return (z * torch.special.erf(z / SQRT2)
            + SQRT_2_OVER_PI * torch.exp(-0.5 * z * z))


def log_bracket(m: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """``log[F(b-m) - F(a-m) + (b-a) erf(m/sqrt 2)]``, branched for stability.

    The naive form has 100% relative error at ``m = 30``; branching so the O(m)
    parts that cancel analytically are never formed numerically brings that to
    about 1e-12, and the factorisation keeps every branch in range.
    """
    ba = b - a
    # Every branch is evaluated on the full tensor and selected with `where`:
    # out-of-branch values may overflow but are discarded. Boolean-mask indexing
    # would launch scatter kernels and host-sync inside the step loop.
    high = m >= b
    low = m <= 0.0

    bracket_mid = _F(b - m) - _F(a - m) + ba * torch.special.erf(m / SQRT2)
    log_mid = torch.log(bracket_mid)

    inner_high = (_g_hat(m - b)
                  - _g_hat(m - a) * torch.exp(-0.5 * ba * (2.0 * m - a - b))
                  - ba * torch.special.erfcx(m / SQRT2)
                  * torch.exp(-0.5 * b * (2.0 * m - b)))
    log_high = -0.5 * (m - b) ** 2 + torch.log(inner_high)

    inner_low = (ba * torch.special.erfcx(-m / SQRT2)
                 + _g_hat(b - m) * torch.exp(b * m - 0.5 * b * b)
                 - _g_hat(a - m) * torch.exp(a * m - 0.5 * a * a))
    log_low = -0.5 * m * m + torch.log(inner_low)

    return torch.where(high, log_high, torch.where(low, log_low, log_mid))


class MoG40ClosedFormScore:
    """Analytic LSC drift for the equal-weight mixture and the annulus law.

    Zero potential evaluations: the theta and radial integrals are analytic, and
    only the angular integral uses a trapezoid rule, which is spectrally accurate
    for a periodic integrand. NOT deployed -- it needs the mixture means, so it
    does not generalise. Retained as the exactness comparator that validates the
    numerical annulus quadrature in tests and in the certificate.
    """

    estimator_type = "closed_form_comparator"

    def __init__(self, mu: torch.Tensor, inner_radius: float,
                 outer_radius: float, intensity: float, m_phi: int = 32,
                 m_max: float = M_MAX) -> None:
        self.mu = mu
        self.a, self.b = float(inner_radius), float(outer_radius)
        self.intensity = float(intensity)
        self.m_phi = int(m_phi)
        self.m_max = float(m_max)
        phi = (torch.arange(m_phi, dtype=torch.float64, device=mu.device)
               * (2.0 * math.pi / m_phi))
        self.directions = torch.stack([torch.cos(phi), torch.sin(phi)], dim=1)

    def log_parts(self, x: torch.Tensor):
        offsets = x.unsqueeze(1) - self.mu.unsqueeze(0)
        square = (offsets * offsets).sum(-1)
        log_omega = (-0.5 * square) - torch.logsumexp(-0.5 * square, dim=1,
                                                      keepdim=True)
        m = torch.einsum("nkd,ld->nkl", offsets, self.directions)
        log_terms = (log_omega.unsqueeze(-1) + 0.5 * m * m
                     + log_bracket(m, self.a, self.b)
                     + math.log(SQRT_PI_OVER_2))
        log_h = torch.logsumexp(log_terms, dim=1)
        log_prefactor = math.log(
            self.intensity / (self.m_phi * (self.b - self.a)))
        return _log_parts(log_h, self.directions, log_prefactor)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        M, v = self.log_parts(x)
        return _finalize(M, v, self.m_max)


# ================================================================ validators
def shell_score_dense_theta(target, law, intensity: float, q_rho: int,
                            x: torch.Tensor, n_theta: int = 200_001
                            ) -> torch.Tensor:
    """Shell score with the same radial rule but a dense composite-Simpson
    theta integral (``n_theta`` odd). Validates the Gauss-Legendre theta rule."""
    device = x.device
    shifts, log_weights = law.quadrature_shifts(q_rho=q_rho)
    weights = torch.exp(log_weights)
    theta = torch.linspace(0.0, 1.0, n_theta, dtype=torch.float64, device=device)
    simpson = torch.ones(n_theta, dtype=torch.float64, device=device)
    simpson[1:-1:2] = 4.0
    simpson[2:-1:2] = 2.0
    simpson = simpson * ((theta[1] - theta[0]) / 3.0)
    beta = float(target.beta)
    score = torch.zeros_like(x)
    with target.no_count():
        base = target.potential.V(x)
        for j in range(shifts.shape[0]):
            logs = []
            for start in range(0, n_theta, 4096):
                nodes = theta[start:start + 4096]
                chord = x.unsqueeze(1) - nodes.view(1, -1, 1) * shifts[j]
                exponent = beta * (base.unsqueeze(1) - target.potential.V(chord))
                logs.append(exponent + torch.log(simpson[start:start + 4096]).view(1, -1))
            log_integral = torch.logsumexp(torch.cat(logs, dim=1), dim=1)
            score -= (intensity * weights[j]
                      * torch.exp(log_integral)).unsqueeze(1) * shifts[j]
    return score


def mog40_score_brute(x: torch.Tensor, mu: torch.Tensor, a: float, b: float,
                      intensity: float, m_phi: int = 512, q_rho: int = 200,
                      q_theta: int = 200) -> torch.Tensor:
    """Brute-force three-dimensional quadrature of the annulus score.

    Trapezoid in the angle, Gauss-Legendre in radius and theta. Slow; used once
    by the tests as an independent check on the closed form.
    """
    device = x.device
    phi = (torch.arange(m_phi, dtype=torch.float64, device=device)
           * (2.0 * math.pi / m_phi))
    directions = torch.stack([torch.cos(phi), torch.sin(phi)], dim=1)
    nodes_r, weights_r = np.polynomial.legendre.leggauss(q_rho)
    radius = torch.as_tensor(0.5 * (b - a) * (nodes_r + 1.0) + a, device=device)
    w_radius = torch.as_tensor(0.5 * (b - a) * weights_r, device=device)
    nodes_t, weights_t = np.polynomial.legendre.leggauss(q_theta)
    theta = torch.as_tensor(0.5 * (nodes_t + 1.0), device=device)
    w_theta = torch.as_tensor(0.5 * weights_t, device=device)

    def log_pi(y: torch.Tensor) -> torch.Tensor:
        diff = y.unsqueeze(-2) - mu
        return torch.logsumexp(-0.5 * (diff * diff).sum(-1), dim=-1)

    base = log_pi(x)
    score = torch.zeros(x.shape[0], 2, dtype=torch.float64, device=device)
    for direction in range(m_phi):
        shift = ((theta.view(-1, 1) * radius.view(1, -1)).unsqueeze(-1)
                 * directions[direction])
        y = x.view(-1, 1, 1, 2) - shift.unsqueeze(0)
        ratio = torch.exp(log_pi(y) - base.view(-1, 1, 1))
        inner = torch.einsum("t,r,ntr->n", w_theta, w_radius * radius, ratio)
        score -= inner.unsqueeze(1) * directions[direction]
    # The trapezoid weight 2pi/M and the 1/(2pi) prefactor cancel to 1/M.
    return score * (intensity / (m_phi * (b - a)))
