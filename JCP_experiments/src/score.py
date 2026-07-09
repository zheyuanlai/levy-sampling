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


def _finalize(M: torch.Tensor, v: torch.Tensor, m_max: float) -> tuple[torch.Tensor, dict]:
    """Step 4: S = -exp(min(M, M_MAX)) * v; log the clip fraction."""
    S = -torch.exp(torch.clamp(M, max=m_max)).unsqueeze(1) * v
    diag = {
        "m_clip_fraction": (M > m_max).to(torch.float64).mean(),
        "max_log_magnitude": M.max(),
    }
    return S, diag


# ================================================================ 4.2 shell
class ShellScore:
    """Generic shell score (E1, E3, and E4 via the moment-exact V_delta).

    nu: centres {r_a} (weights w_a), shell half-thickness h; the theta and rho
    integrals use Gauss-Legendre probability weights so the quadrature measure
    is exactly the sampler's nu.
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
    out = torch.empty_like(m)

    hi = m >= b
    lo = m <= 0.0
    mid = ~(hi | lo)

    if mid.any():
        mm = m[mid]
        B = _F(b - mm) - _F(a - mm) + ba * torch.special.erf(mm / SQRT2)
        out[mid] = torch.log(B)

    if hi.any():
        mm = m[hi]
        # B = e^{-(m-b)^2/2} [ ghat(m-b) - ghat(m-a) e^{-(b-a)(2m-a-b)/2}
        #                      - (b-a) erfcx(m/sqrt2) e^{-b(2m-b)/2} ]
        inner = (_g_hat(mm - b)
                 - _g_hat(mm - a) * torch.exp(-0.5 * ba * (2.0 * mm - a - b))
                 - ba * torch.special.erfcx(mm / SQRT2) * torch.exp(-0.5 * b * (2.0 * mm - b)))
        out[hi] = -0.5 * (mm - b) ** 2 + torch.log(inner)

    if lo.any():
        mm = m[lo]
        # B = e^{-m^2/2} [ (b-a) erfcx(-m/sqrt2) + ghat(b-m) e^{bm-b^2/2}
        #                  - ghat(a-m) e^{am-a^2/2} ]
        inner = (ba * torch.special.erfcx(-mm / SQRT2)
                 + _g_hat(b - mm) * torch.exp(b * mm - 0.5 * b * b)
                 - _g_hat(a - mm) * torch.exp(a * mm - 0.5 * a * a))
        out[lo] = -0.5 * mm * mm + torch.log(inner)

    return out


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
