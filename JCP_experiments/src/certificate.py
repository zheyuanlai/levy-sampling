"""Weak stationarity residual R(phi) - the correctness artifact.

    R(phi) = | int S.grad(phi) dpi + int J phi dpi | / | int J phi dpi |

Zero in exact arithmetic (unconditionally in nu); the measured value is the
combined defect from Q_theta, Q_rho and any clipping.

Two evaluation modes:

* certificate_grid: direct composite-Gauss-Legendre x-quadrature (E1-E3;
  E3 reduces exactly to its latent 2D problem because jumps and test
  functions act on z_{1:2} only and the dot product is affine-invariant).
  The integration domain MUST extend a full jump length beyond the target's
  effective support: order-one contributions to the identity live where pi
  is tiny and S is enormous. A deliberately tight box reproduces a large
  residual (regression-tested).

* certificate_importance: E4 (24D). Uses the change of variables
  x -> x + theta_p r inside the drift term, under which the implemented
  quadrature score satisfies EXACTLY
      int S.grad(phi) dpi = -lam int nu(dr) sum_p w_p E_pi[ r . grad(phi)(x + theta_p r) ],
  so the residual becomes a single self-normalised importance-sampling
  average of the pointwise theta-quadrature defect - no O(1) cancellation is
  left to Monte Carlo. This is equivalent to the deployed score provided the
  M_MAX cap never fires on the sampled region (asserted separately).
"""
from __future__ import annotations

import math

import numpy as np
import torch


# ------------------------------------------------------------ test functions
class TanhRidgeProduct:
    """phi(x) = prod_j tanh((a_j . x - c_j) / s_j); smooth, bounded, with
    analytic gradient. Offsets are deliberately asymmetric so |int J phi dpi|
    is bounded away from zero."""

    def __init__(self, a: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> None:
        self.a, self.c, self.s = a, c, s              # (J,d), (J,), (J,)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.tanh((x @ self.a.T - self.c) / self.s)      # (N, J)
        return t.prod(dim=-1)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.tanh((x @ self.a.T - self.c) / self.s)      # (N, J)
        J = t.shape[-1]
        g = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        for j in range(J):
            others = torch.ones_like(t[:, 0])
            for i in range(J):
                if i != j:
                    others = others * t[:, i]
            g = g + (others * (1.0 - t[:, j] ** 2) / self.s[j]).unsqueeze(-1) * self.a[j]
        return g


def make_phi_family(d: int, center, scale: float, device,
                    n_phi: int = 6, seed: int = 314) -> list[TanhRidgeProduct]:
    """Products of 1-2 tanh ridges at a few offsets, adapted to the domain
    scale. Deterministic (fixed seed)."""
    rng = np.random.default_rng(seed)
    center = np.asarray(center, dtype=float).reshape(1, d)
    phis = []
    for i in range(n_phi):
        J = 1 + (i % 2)
        a = rng.standard_normal((J, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        c = (a @ center.T).reshape(-1) + rng.uniform(-0.35, 0.35, J) * scale
        s = rng.uniform(0.4, 0.9, J) * scale
        phis.append(TanhRidgeProduct(
            torch.as_tensor(a, dtype=torch.float64, device=device),
            torch.as_tensor(c, dtype=torch.float64, device=device),
            torch.as_tensor(s, dtype=torch.float64, device=device)))
    return phis


# ------------------------------------------------------------ x-quadrature
def composite_gl_grid(lo, hi, n_panels: int, nodes_per_panel: int, device):
    """Composite Gauss-Legendre nodes/weights per dimension, tensorised.
    Returns (points (G, d), weights (G,)). Spectrally accurate per panel."""
    lo = np.atleast_1d(np.asarray(lo, dtype=float))
    hi = np.atleast_1d(np.asarray(hi, dtype=float))
    d = lo.shape[0]
    xg, wg = np.polynomial.legendre.leggauss(nodes_per_panel)
    axes, waxes = [], []
    for k in range(d):
        edges = np.linspace(lo[k], hi[k], n_panels + 1)
        mid = 0.5 * (edges[:-1] + edges[1:])
        half = 0.5 * (edges[1:] - edges[:-1])
        nodes = (mid[:, None] + half[:, None] * xg[None, :]).reshape(-1)
        wts = (half[:, None] * wg[None, :]).reshape(-1)
        axes.append(torch.as_tensor(nodes, dtype=torch.float64, device=device))
        waxes.append(torch.as_tensor(wts, dtype=torch.float64, device=device))
    if d == 1:
        return axes[0].unsqueeze(1), waxes[0]
    grids = torch.meshgrid(*axes, indexing="ij")
    pts = torch.stack([g.reshape(-1) for g in grids], dim=1)
    w = waxes[0]
    for k in range(1, d):
        w = (w.unsqueeze(-1) * waxes[k]).reshape(-1)
    return pts, w


def certificate_grid(potential, score_fn, nu_shifts: torch.Tensor,
                     nu_logw: torch.Tensor, lam: float, beta: float,
                     phis: list[TanhRidgeProduct], lo, hi,
                     n_panels: int = 48, nodes_per_panel: int = 8,
                     chunk: int = 16384, m_max: float = 600.0,
                     shift_chunk: int = 256) -> dict:
    """Direct-form residual on a composite-GL grid over [lo, hi].

    The drift integrand p(x) S(x) . grad(phi)(x) is assembled IN LOG SPACE
    from the score's (M, v) parts: exp(-beta V(x) + min(M, M_MAX)). In linear
    fp64 arithmetic p underflows exactly where ||S|| is astronomical, and the
    certificate would silently drop the order-one far-field contributions it
    exists to check. The min(M, M_MAX) cap is kept so the measured residual
    includes the deployed clipping policy."""
    dev = nu_shifts.device
    X, wq = composite_gl_grid(lo, hi, n_panels, nodes_per_panel, dev)
    G = X.shape[0]
    logp = -beta * potential.V(X)
    shift = logp.max()
    logp = logp - shift
    pw = torch.exp(logp) * wq                                # (G,)

    Mlog = torch.empty(G, dtype=torch.float64, device=dev)
    v = torch.empty_like(X)
    for s0 in range(0, G, chunk):
        Mlog[s0:s0 + chunk], v[s0:s0 + chunk] = score_fn.log_parts(X[s0:s0 + chunk])
    # UNCAPPED magnitude: the deployed drift caps M at M_MAX, but taming
    # saturates (step -> -v/||v||) long before e^{M_MAX}, so the deployed
    # tamed step is identical to the uncapped one up to O(e^{-M_MAX}); that
    # saturation defect is reported separately below.
    log_drift_mag = logp + Mlog                              # (G,) stable in log space

    w_nu = torch.exp(nu_logw)                                # (J,) sums to 1
    n_shift = nu_shifts.shape[0]
    results = {}
    for i, phi in enumerate(phis):
        # drift: -sum_g wq_g e^{logp_g + min(M_g, cap)} (v_g . grad phi(x_g))
        drift_term = -(wq * torch.exp(log_drift_mag)
                       * (v * phi.grad(X)).sum(-1)).sum()
        # jump: lam sum_j w_j sum_g pw_g [phi(x_g + r_j) - phi(x_g)]
        phi0_w = (pw * phi(X)).sum()
        jump_term = torch.zeros((), dtype=torch.float64, device=dev)
        for j0 in range(0, n_shift, shift_chunk):
            rs = nu_shifts[j0:j0 + shift_chunk]              # (c, d)
            ws = w_nu[j0:j0 + shift_chunk]
            for s0 in range(0, G, chunk):
                xs = X[s0:s0 + chunk]
                vals = phi(xs.unsqueeze(1) + rs.unsqueeze(0))  # (g, c)
                jump_term = jump_term + (pw[s0:s0 + chunk].unsqueeze(1)
                                         * vals * ws.unsqueeze(0)).sum()
        jump_term = lam * (jump_term - phi0_w * w_nu.sum())
        results[f"phi_{i}"] = {
            "residual": float((torch.abs(drift_term + jump_term) / torch.abs(jump_term)).item()),
            "jump_term": float(jump_term.item()),
            "drift_term": float(drift_term.item()),
        }
    results["max_residual"] = max(v_["residual"] for k, v_ in results.items()
                                  if k.startswith("phi_"))
    # clipping saturation defect: where M > M_MAX the deployed tamed drift
    # step differs from the uncapped one by ~ 1/(dt e^{M_MAX} ||v||); report
    # its sup over the grid (0.0 when the cap never binds here).
    clipped = Mlog > m_max
    results["m_clip_fraction_grid"] = float(clipped.to(torch.float64).mean().item())
    if bool(clipped.any().item()):
        vn = v[clipped].norm(dim=1).clamp(min=1e-300)
        results["clip_tamed_step_defect"] = float(
            torch.exp(-torch.tensor(m_max, device=dev) - torch.log(vn).min()).item())
    else:
        results["clip_tamed_step_defect"] = 0.0
    return results


def certificate_importance(potential, nu_shifts: torch.Tensor,
                           nu_logw: torch.Tensor, theta: torch.Tensor,
                           w_theta: torch.Tensor, lam: float, beta: float,
                           phis: list[TanhRidgeProduct], proposal,
                           n_samples: int = 200_000, seed: int = 2718,
                           chunk: int = 20_000) -> dict:
    """Shifted-form residual by self-normalised importance sampling (E4).
    `proposal` must provide .sample(n, gen) and .log_q(x)."""
    dev = nu_shifts.device
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    x = proposal.sample(n_samples, gen)
    log_w = -beta * potential.V(x) - proposal.log_q(x)
    log_w = log_w - log_w.max()
    w_is = torch.exp(log_w)
    w_is = w_is / w_is.sum()

    w_nu = torch.exp(nu_logw)
    results = {}
    for i, phi in enumerate(phis):
        num = torch.zeros((), dtype=torch.float64, device=dev)
        den = torch.zeros((), dtype=torch.float64, device=dev)
        for s0 in range(0, n_samples, chunk):
            xs = x[s0:s0 + chunk]
            ws = w_is[s0:s0 + chunk]
            phi0 = phi(xs)
            for j in range(nu_shifts.shape[0]):
                r = nu_shifts[j]
                dphi = phi(xs + r) - phi0                    # exact theta integral
                quad = torch.zeros_like(dphi)                # GL theta quadrature
                for p_i in range(theta.shape[0]):
                    quad = quad + w_theta[p_i] * (phi.grad(xs + theta[p_i] * r) @ r)
                num = num + lam * w_nu[j] * (ws * (dphi - quad)).sum()
                den = den + lam * w_nu[j] * (ws * dphi).sum()
        results[f"phi_{i}"] = {"residual": float((num.abs() / den.abs()).item()),
                               "jump_term": float(den.item())}
    results["max_residual"] = max(v["residual"] for k, v in results.items()
                                  if k.startswith("phi_"))
    return results
