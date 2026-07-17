"""Levy score S_{nu,beta}: log-space accumulation, closed forms, validators.

S_{nu,beta}(x) = -lam * int nu(dr) r int_0^1 exp[-beta (V(x - theta r) - V(x))] d theta

Naive evaluation overflows: the per-direction theta integral is strictly
positive but spans hundreds of orders of magnitude at beta = 8. We therefore
accumulate in log space (log-sum-exp over quadrature nodes), extract the
global max exponent M(x), form the O(1) direction vector v(x), and return

    S(x) = -lam * exp(min(M(x), M_MAX)) * v(x),         M_MAX = 600.

Because every sampler tames its drift, when ||S|| is astronomical only its
*direction* matters, and v preserves the direction exactly. No score_clip on
vector components, no log_clip on the raw log-ratio; the only cap is M_MAX,
and the fraction of particles hitting it is logged (`m_clip_fraction`).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .config import M_MAX
from .jumps import ShellJumpLaw, gauss_legendre_01

SQRT_PI_OVER_2 = math.sqrt(math.pi / 2.0)
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
SQRT2 = math.sqrt(2.0)


def _log_parts(log_terms: torch.Tensor, vecs: torch.Tensor,
               log_prefactor: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Steps 2-3 of the log-space accumulator.

    log_terms: (N, J) log magnitudes; vecs: (J, d) directions.
    Returns (M, v) with M = per-particle max log magnitude (including the
    constant prefactor) and v the O(1) direction vector, so that the exact
    score is S = -exp(M) * v. Cancellation between +r and -r atoms inside v
    is genuine physics, not error, and is preserved exactly.
    """
    M = log_terms.max(dim=1).values + log_prefactor          # (N,)
    scaled = torch.exp(log_terms - (M - log_prefactor).unsqueeze(1))  # O(1)
    v = scaled @ vecs                                        # (N, d)
    return M, v


def _clip_diagnostics(M: torch.Tensor, m_max: float) -> dict:
    """Exact device-resident clipping count, denominator, fraction, and max."""
    clipped = M > m_max
    count = clipped.to(torch.int64).sum()
    total = torch.as_tensor(M.numel(), dtype=torch.int64, device=M.device)
    return {
        "m_clip_count": count,
        "m_clip_total": total,
        "m_clip_fraction": count.to(torch.float64) / total,
        "max_log_magnitude": M.max(),
    }


def _finalize(M: torch.Tensor, v: torch.Tensor, m_max: float) -> tuple[torch.Tensor, dict]:
    """Step 4: S = -exp(min(M, M_MAX)) * v; log exact clipping counts."""
    S = -torch.exp(torch.clamp(M, max=m_max)).unsqueeze(1) * v
    return S, _clip_diagnostics(M, m_max)


def _realized_chord_deltas(potential, x: torch.Tensor,
                            y: torch.Tensor) -> torch.Tensor:
    """Evaluate per-particle chord energies under the score ledger.

    ``y`` has shape ``(N, ..., d)`` and contains particle-specific shifts, so
    the public ``V_delta(x, R)`` broadcasting interface cannot represent it.
    Repository potentials expose ``_V_raw``; using it here while incrementing
    ``n_Vdelta`` records every quadrature energy difference as a Lévy-score
    evaluation rather than mixing that cost into ordinary potential calls.
    Lightweight external/test potentials fall back to their public ``V`` API.
    """
    n = x.shape[0]
    d = x.shape[-1]
    n_quad = int(y.numel() // (n * d))
    if (hasattr(potential, "_V_raw") and hasattr(potential, "n_Vdelta")):
        potential.n_Vdelta += n * n_quad
        v0 = potential._V_raw(x)
        vy = potential._V_raw(y)
    else:
        v0 = potential.V(x)
        vy = potential.V(y.reshape(-1, d)).reshape(y.shape[:-1])
    return vy - v0.reshape((n,) + (1,) * (vy.ndim - 1))


# ================================================================ 4.2 shell
class ShellScore:
    """Generic shell score (E1, E3, and E4 via the moment-exact V_delta).

    nu: centres {r_a} (weights w_a), shell half-thickness h. The theta/rho
    expectations under that declared continuous law are approximated by
    normalized Gauss--Legendre rules; the finite quadrature measure is not
    literally the sampler's continuous nu.
    """

    def __init__(self, potential, law: ShellJumpLaw, lam: float, beta: float,
                 q_theta: int, q_rho: int, m_max: float = M_MAX) -> None:
        self.potential = potential
        self.law = law
        self.lam = float(lam)
        self.beta = float(beta)
        self.m_max = float(m_max)
        dev = law.atoms.device
        theta, w_theta = gauss_legendre_01(q_theta, dev)     # (Qt,), sum(w)=1
        shifts, logw = law.quadrature_shifts(q_rho)          # (J, d), (J,)
        self.r_aq = shifts                                   # (J, d)
        self.logw_aq = logw                                  # (J,)
        self.log_w_theta = torch.log(w_theta)                # (Qt,)
        # all theta_p * r_j shift vectors, flattened once
        self.R_all = (theta.view(-1, 1, 1) * shifts.view(1, *shifts.shape))  # (Qt,J,d)
        self.R_flat = self.R_all.reshape(-1, shifts.shape[1])
        self.q_theta = q_theta
        self.J = shifts.shape[0]

    def log_parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(M, v) with S = -exp(M) v exactly (no cap applied here)."""
        n = x.shape[0]
        dV = self.potential.V_delta(x, self.R_flat).reshape(n, self.q_theta, self.J)
        # log I_j = LSE_p [ log w_hat_p + beta (V(x) - V(x - theta_p r_j)) ]
        log_I = torch.logsumexp(self.log_w_theta.view(1, -1, 1) - self.beta * dV, dim=1)
        log_terms = self.logw_aq.unsqueeze(0) + log_I        # (N, J)
        return _log_parts(log_terms, self.r_aq, math.log(self.lam))

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        M, v = self.log_parts(x)
        return _finalize(M, v, self.m_max)


# ================================================== random-atomic estimator
class RandomAtomicShellScore:
    """Unbiased single-atom estimator of the exact Levy score `ShellScore`.

    Given ONE realised displacement R per particle -- drawn by the sampler from
    the SAME measure nu the jump uses (rho = nu/lambda, so the Radon-Nikodym
    weight w(R) = lambda is constant) -- the per-particle score is

        S_R(x) = -lambda R exp( LSE_p[ log w_p + beta (V(x) - V(x - theta_p R)) ] ),

    with theta_p, w_p Gauss-Legendre nodes/probability-weights on [0,1]. This is
    the fixed-R integrand of the exact score, whose nu-average is S_{nu,beta}:

      * unbiased:      E_{R~rho}[hat A_{eps,R}] = A_{eps,nu}     (Fubini);
      * atomwise mu-invariant: for EVERY fixed R, int A_{eps,R} phi dmu = 0
        (chord fundamental-theorem-of-calculus identity).

    It is a PRACTICAL ESTIMATOR of `ShellScore`, not a separate method. We do NOT
    claim finite-refresh spectral-gap transfer or exact target-preservation of
    the Euler-Poisson discretisation. Cost is q_theta score-quadrature energy differences per particle
    per step (vs q_theta * A * q_rho for the exact quadrature), and the jump law
    need only be *sampleable* -- no closed-form rho/atom quadrature is required.

    Interface: the sampler draws R and calls `score_for_shift(x, R)`; the score
    never draws R itself (so the same R is used for score and jump, and R stays
    independent of the current state x -- both required by the invariance proof).
    Generic in the jump law: works for any `law.sample(n, gen) -> (n, d)` and any
    potential exposing `.V` (shell laws E1/E3/E4 and the E2 annulus law alike).
    """

    def __init__(self, potential, law, lam: float, beta: float,
                 q_theta: int = None, m_max: float = M_MAX) -> None:
        from .config import Q_THETA
        self.potential = potential
        self.law = law
        self.lam = float(lam)
        self.beta = float(beta)
        self.m_max = float(m_max)
        self.q_theta = int(q_theta if q_theta is not None else Q_THETA)
        if hasattr(law, "atoms"):
            dev = law.atoms.device
        else:
            dev = getattr(law, "device", torch.device("cpu"))
        theta, w_theta = gauss_legendre_01(self.q_theta, dev)   # (Qt,), sum(w)=1
        self.theta = theta
        self.log_w_theta = torch.log(w_theta)                   # (Qt,)
        self.log_lam = math.log(self.lam)

    def score_for_shift(self, x: torch.Tensor,
                        R: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """S_R(x) for one realised displacement R (N, d) per particle.

        Direction is exactly -R; only the scalar magnitude exp(M) is tamed
        (M_MAX cap), mirroring `_finalize` but per particle with a single atom.
        """
        n, d = x.shape
        # Chord points y = x - theta_p R -> (N, Qt, d).  The helper
        # records N*Qt score-quadrature energy differences in n_Vdelta.
        y = x.unsqueeze(1) - self.theta.view(1, -1, 1) * R.unsqueeze(1)
        dV = _realized_chord_deltas(self.potential, x, y)        # (N, Qt)
        # log I = LSE_p[ log w_p + beta (V(x) - V(x - theta_p R)) ]
        log_I = torch.logsumexp(self.log_w_theta.view(1, -1) - self.beta * dV, dim=1)
        M = self.log_lam + log_I                                 # (N,)
        S = -torch.exp(torch.clamp(M, max=self.m_max)).unsqueeze(1) * R
        return S, _clip_diagnostics(M, self.m_max)


class MultiAtomShellScore:
    """Paired multi-atom random-measure Levy score.

    The realised per-particle bank is

        R = (R_1, ..., R_A),   R_a ~ q_a,

    where ``q_a`` is atom ``a``'s radial shell (and optional jitter) law. It
    defines the random finite measure

        nu_R(dr) = lambda * sum_a w_a delta_{R_a}(dr).

    ``score_for_bank`` evaluates the exact (up to theta quadrature/clipping)
    score of *that same realised measure*:

        S(x) = -lambda sum_a w_a R_a
               exp(LSE_p[log w_p + beta(V(x)-V(x-theta_p R_a))]).

    The paired sampler then uses independent counts

        N_a ~ Poisson(lambda * w_a * dt)

    and jumps by ``sum_a N_a R_a``. Conditional on every frozen bank R, the
    drift and jump terms therefore use the identical ``nu_R`` and obey the
    chord stationarity identity atom by atom. Averaging over R recovers the
    original shell law.

    Cost is A*q_theta score-quadrature energy differences/step (vs q_theta for
    single-atom RA, and A*q_rho*q_theta for exact quadrature). ``sample_bank``
    is deliberately state-free; the sampler draws the bank before evaluating
    the score and passes the identical tensor to ``score_for_bank`` and the
    jump update.
    """

    def __init__(self, potential, law, lam: float, beta: float,
                 q_theta: int = None, m_max: float = M_MAX,
                 gen: torch.Generator = None) -> None:
        from .config import Q_THETA
        self.potential = potential
        self.law = law
        self.lam = float(lam)
        self.beta = float(beta)
        self.m_max = float(m_max)
        self.q_theta = int(q_theta if q_theta is not None else Q_THETA)
        required = ("atoms", "weights", "h", "units", "A", "d")
        missing = [name for name in required if not hasattr(law, name)]
        if missing:
            raise TypeError(
                "paired multi-atom score requires a finite shell bank; "
                f"missing {', '.join(missing)}"
            )
        if self.lam <= 0.0 or self.beta <= 0.0 or self.q_theta <= 0:
            raise ValueError("lambda, beta, and q_theta must be positive")
        if math.isnan(self.m_max):
            raise ValueError("m_max must not be NaN")
        atom_norms = law.atoms.norm(dim=1, keepdim=True)
        if (law.atoms.shape != (law.A, law.d)
                or law.h.shape != (law.A,)
                or law.units.shape != law.atoms.shape
                or not bool(torch.isfinite(law.atoms).all().item())
                or not bool(torch.isfinite(atom_norms).all().item())
                or bool((atom_norms <= 0).any().item())
                or not bool(torch.isfinite(law.h).all().item())
                or bool((law.h < 0).any().item())
                or not bool(torch.isfinite(law.units).all().item())):
            raise ValueError(
                "multi-atom shell geometry/units must be finite, nonzero, "
                "and have valid shapes with h >= 0")
        expected_units = law.atoms / atom_norms
        rtol = 2e-5 if law.atoms.dtype == torch.float32 else 2e-12
        atol = 2e-6 if law.atoms.dtype == torch.float32 else 2e-14
        if (not torch.allclose(law.units.norm(dim=1),
                               torch.ones(law.A, dtype=law.units.dtype,
                                          device=law.units.device),
                               rtol=rtol, atol=atol)
                or not torch.allclose(law.units, expected_units,
                                      rtol=rtol, atol=atol)):
            raise ValueError(
                "multi-atom shell units must be normalized and aligned with atoms")
        if (not bool(torch.isfinite(law.weights).all().item())
                or bool((law.weights < 0).any().item())
                or not math.isclose(float(law.weights.sum().item()), 1.0,
                                    rel_tol=1e-12, abs_tol=1e-14)):
            raise ValueError("multi-atom weights must be finite, nonnegative, and normalized")
        dev = law.atoms.device
        theta, w_theta = gauss_legendre_01(
            self.q_theta, dev, dtype=law.atoms.dtype)
        self.theta = theta
        self.log_w_theta = torch.log(w_theta)
        self.log_lam = math.log(self.lam)
        self.gen = gen if gen is not None else torch.Generator(device=dev)

    def sample_bank(self, n: int, gen: torch.Generator | None = None) -> torch.Tensor:
        """Draw one conditional-shell displacement per atom and particle.

        This method has no state argument, making independence of the refreshed
        random measure from the current chain state structural. For a jittered
        shell law, the same Gaussian jitter used by its marginal sampler is
        applied independently to every realised atom.
        """
        if isinstance(n, bool) or int(n) != n or n < 1:
            raise ValueError("n must be a positive integer")
        n = int(n)
        gen = self.gen if gen is None else gen
        A, d = self.law.A, self.law.d
        dev, dtype = self.law.atoms.device, self.law.atoms.dtype
        rho = ((torch.rand(n, A, generator=gen, device=dev, dtype=dtype) * 2.0 - 1.0)
               * self.law.h.view(1, A))
        R = (self.law.atoms.view(1, A, d)
             + rho.unsqueeze(-1) * self.law.units.view(1, A, d))
        jitter_sigma = float(getattr(self.law, "jitter_sigma", 0.0))
        if jitter_sigma > 0.0:
            R = R + jitter_sigma * torch.randn(
                R.shape, generator=gen, device=dev, dtype=dtype)
        return R

    def score_for_bank(self, x: torch.Tensor,
                       R: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Evaluate the score of a supplied realised bank ``R``.

        ``R[n, a]`` is the atom used for particle ``n`` and component ``a``.
        The sampler retains this tensor and uses it again in the paired jump.
        """
        n, d = x.shape
        A, Qt = self.law.A, self.q_theta
        if R.shape != (n, A, d):
            raise ValueError(f"bank must have shape {(n, A, d)}, got {tuple(R.shape)}")
        if R.device != x.device or R.dtype != x.dtype:
            raise ValueError("bank must have the same device and dtype as x")
        # chord points y = x - theta_p R_a  -> (N, A, Qt, d)
        y = x.view(n, 1, 1, d) - self.theta.view(1, 1, Qt, 1) * R.view(n, A, 1, d)
        dV = _realized_chord_deltas(self.potential, x, y)        # (N, A, Qt)
        log_I = torch.logsumexp(self.log_w_theta.view(1, 1, Qt) - self.beta * dV, dim=2)
        # One global log accumulator per particle, matching ShellScore.  A
        # per-atom cap before summation would change relative atom weights when
        # clipping binds and would no longer be the score of the frozen bank.
        log_terms = (self.log_lam + torch.log(self.law.weights).view(1, A)
                     + log_I)                                   # (N, A)
        M = log_terms.max(dim=1).values                          # (N,)
        scaled = torch.exp(log_terms - M.unsqueeze(1))           # (N, A)
        v = (scaled.unsqueeze(-1) * R).sum(dim=1)                # (N, d)
        S = -torch.exp(torch.clamp(M, max=self.m_max)).unsqueeze(1) * v
        return S, _clip_diagnostics(M, self.m_max)

    def log_parts_for_bank(self, x: torch.Tensor,
                           R: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Uncapped ``(M, v)`` for a bank frozen across all states ``x``.

        This certificate-facing form satisfies ``S_R(x) = -exp(M(x)) v(x)``.
        Production per-particle banks use ``score_for_bank`` instead.
        """
        n, d = x.shape
        A, Qt = self.law.A, self.q_theta
        if R.shape != (A, d):
            raise ValueError(f"frozen bank must have shape {(A, d)}, got {tuple(R.shape)}")
        if R.device != x.device or R.dtype != x.dtype:
            raise ValueError("frozen bank must have the same device and dtype as x")
        y = x.view(n, 1, 1, d) - self.theta.view(1, 1, Qt, 1) * R.view(1, A, 1, d)
        dV = _realized_chord_deltas(self.potential, x, y)        # (N, A, Qt)
        log_I = torch.logsumexp(self.log_w_theta.view(1, 1, Qt) - self.beta * dV, dim=2)
        log_terms = torch.log(self.law.weights).view(1, A) + log_I
        return _log_parts(log_terms, R, self.log_lam)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compatibility score-only path; paired sampling retains the bank."""
        return self.score_for_bank(x, self.sample_bank(x.shape[0]))


# ============================================================== 4.3 MoG40
def _g_hat(z: torch.Tensor) -> torch.Tensor:
    """g_hat(z) = sqrt(2/pi) - z erfcx(z/sqrt(2)) for z >= 0 (so that
    g(z) = g_hat(z) e^{-z^2/2} and F(z) = |z| + g(|z|))."""
    return SQRT_2_OVER_PI - z * torch.special.erfcx(z / SQRT2)


def _F(z: torch.Tensor) -> torch.Tensor:
    """F(z) = z erf(z/sqrt 2) + sqrt(2/pi) e^{-z^2/2}; F' = erf(z/sqrt 2), even."""
    return z * torch.special.erf(z / SQRT2) + SQRT_2_OVER_PI * torch.exp(-0.5 * z * z)


def log_bracket(m: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """log of B(m) = F(b-m) - F(a-m) + (b-a) erf(m/sqrt 2)  ( > 0 ).

    Branched so the O(m) parts that cancel *analytically* in the outer
    regimes are never formed numerically (the naive form has 100% relative
    error at m = 30; this form is ~1e-12), and factored so nothing under- or
    overflows anywhere on the real line.
    """
    ba = b - a
    # All three branches are evaluated on the full tensor and selected with
    # torch.where: out-of-branch values may overflow to inf/nan but are
    # discarded by the selection. (Boolean-mask indexing here would both
    # launch slow scatter kernels and -- via .any() -- host-sync inside the
    # sampler step loop.)
    hi = m >= b
    lo = m <= 0.0

    # 0 < m < b: direct, safe
    B_mid = _F(b - m) - _F(a - m) + ba * torch.special.erf(m / SQRT2)
    log_mid = torch.log(B_mid)

    # m >= b: B = e^{-(m-b)^2/2} [ ghat(m-b) - ghat(m-a) e^{-(b-a)(2m-a-b)/2}
    #                              - (b-a) erfcx(m/sqrt2) e^{-b(2m-b)/2} ]
    inner_hi = (_g_hat(m - b)
                - _g_hat(m - a) * torch.exp(-0.5 * ba * (2.0 * m - a - b))
                - ba * torch.special.erfcx(m / SQRT2) * torch.exp(-0.5 * b * (2.0 * m - b)))
    log_hi = -0.5 * (m - b) ** 2 + torch.log(inner_hi)

    # m <= 0: B = e^{-m^2/2} [ (b-a) erfcx(-m/sqrt2) + ghat(b-m) e^{bm-b^2/2}
    #                          - ghat(a-m) e^{am-a^2/2} ]
    inner_lo = (ba * torch.special.erfcx(-m / SQRT2)
                + _g_hat(b - m) * torch.exp(b * m - 0.5 * b * b)
                - _g_hat(a - m) * torch.exp(a * m - 0.5 * a * a))
    log_lo = -0.5 * m * m + torch.log(inner_lo)

    return torch.where(hi, log_hi, torch.where(lo, log_lo, log_mid))


class MoG40Score:
    """Closed-form LSC drift for the equal-weight MoG target and the annulus
    jump law r = rho u_phi, rho ~ Unif[a,b], phi ~ Unif[0, 2pi).

    Zero potential evaluations: theta and rho integrals are analytic (erf),
    only the phi integral uses an M_phi-point trapezoid rule (spectrally
    accurate for the periodic integrand).
    """

    def __init__(self, mu: torch.Tensor, a: float, b: float, lam: float,
                 m_phi: int = 32, m_max: float = M_MAX) -> None:
        self.mu = mu                       # (K, 2)
        self.a, self.b = float(a), float(b)
        self.lam = float(lam)
        self.m_phi = m_phi
        self.m_max = float(m_max)
        dev = mu.device
        phi = torch.arange(m_phi, dtype=torch.float64, device=dev) * (2.0 * math.pi / m_phi)
        self.u = torch.stack([torch.cos(phi), torch.sin(phi)], dim=1)  # (M, 2)

    def log_parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = x.unsqueeze(1) - self.mu.unsqueeze(0)          # (N, K, 2)
        sq = (d_k * d_k).sum(-1)                             # (N, K)
        log_omega = (-0.5 * sq) - torch.logsumexp(-0.5 * sq, dim=1, keepdim=True)
        m = torch.einsum("nkd,ld->nkl", d_k, self.u)         # (N, K, M)
        logB = log_bracket(m, self.a, self.b)                # (N, K, M)
        log_terms = log_omega.unsqueeze(-1) + 0.5 * m * m + logB + math.log(SQRT_PI_OVER_2)
        log_H = torch.logsumexp(log_terms, dim=1)            # (N, M)
        log_pref = math.log(self.lam / (self.m_phi * (self.b - self.a)))
        return _log_parts(log_H, self.u, log_pref)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        M, v = self.log_parts(x)
        return _finalize(M, v, self.m_max)


# ============================================================== validators
def mog40_score_brute(x: torch.Tensor, mu: torch.Tensor, a: float, b: float,
                      lam: float, m_phi: int = 512, q_rho: int = 200,
                      q_theta: int = 200) -> torch.Tensor:
    """Brute-force 3-D quadrature of
    S(x) = -lam/(2 pi (b-a)) intint int rho u_phi pi(x - theta rho u)/pi(x)
    (trapezoid in phi, GL in rho and theta). Slow; used once by tests."""
    dev = x.device
    phi = torch.arange(m_phi, dtype=torch.float64, device=dev) * (2.0 * math.pi / m_phi)
    u = torch.stack([torch.cos(phi), torch.sin(phi)], dim=1)            # (M, 2)
    xr, wr = np.polynomial.legendre.leggauss(q_rho)
    rho = torch.as_tensor(0.5 * (b - a) * (xr + 1.0) + a, device=dev)   # (Qr,)
    w_rho = torch.as_tensor(0.5 * (b - a) * wr, device=dev)
    xt, wt = np.polynomial.legendre.leggauss(q_theta)
    theta = torch.as_tensor(0.5 * (xt + 1.0), device=dev)               # (Qt,)
    w_theta = torch.as_tensor(0.5 * wt, device=dev)

    def log_pi(y: torch.Tensor) -> torch.Tensor:
        diff = y.unsqueeze(-2) - mu
        return torch.logsumexp(-0.5 * (diff * diff).sum(-1), dim=-1)

    lp0 = log_pi(x)                                                     # (N,)
    S = torch.zeros(x.shape[0], 2, dtype=torch.float64, device=dev)
    for l in range(m_phi):                                              # chunk in phi
        shift = (theta.view(-1, 1) * rho.view(1, -1)).unsqueeze(-1) * u[l]  # (Qt,Qr,2)
        y = x.view(-1, 1, 1, 2) - shift.unsqueeze(0)                    # (N,Qt,Qr,2)
        ratio = torch.exp(log_pi(y) - lp0.view(-1, 1, 1))               # (N,Qt,Qr)
        inner = torch.einsum("t,r,ntr->n", w_theta, w_rho * rho, ratio)  # (N,)
        S -= inner.unsqueeze(1) * u[l]
    return S * (lam / (m_phi * (b - a)) * (2.0 * math.pi) / (2.0 * math.pi))
    # note: trapezoid weight 2pi/M and the 1/(2pi) prefactor cancel to 1/M


def shell_score_brute_theta(potential, law: ShellJumpLaw, lam: float, beta: float,
                            q_rho: int, x: torch.Tensor,
                            n_theta: int = 200_001) -> torch.Tensor:
    """Shell score with the SAME rho quadrature but a dense composite-Simpson
    theta integral (n_theta odd). Validates the GL theta quadrature."""
    dev = x.device
    shifts, logw = law.quadrature_shifts(q_rho)              # (J, d)
    w_aq = torch.exp(logw)
    theta = torch.linspace(0.0, 1.0, n_theta, dtype=torch.float64, device=dev)
    w = torch.ones(n_theta, dtype=torch.float64, device=dev)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    w = w * ((theta[1] - theta[0]) / 3.0)                    # Simpson weights, sum ~ 1
    S = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float64, device=dev)
    v0 = potential.V(x)                                      # (N,)
    for j in range(shifts.shape[0]):                         # chunk over shifts
        acc = None
        chunk = 4096
        logs = []
        for s in range(0, n_theta, chunk):
            th = theta[s:s + chunk]
            y = x.unsqueeze(1) - th.view(1, -1, 1) * shifts[j]
            g = beta * (v0.unsqueeze(1) - potential.V(y))    # (N, c)
            logs.append(g + torch.log(w[s:s + chunk]).view(1, -1))
        log_I = torch.logsumexp(torch.cat(logs, dim=1), dim=1)   # (N,)
        contrib = w_aq[j] * torch.exp(log_I)
        S -= lam * contrib.unsqueeze(1) * shifts[j]
        del logs, acc
    return S
