"""P5: weak-stationarity certificate R(phi) for E5 (atomwise, torus, 60-D).

Why this differs from E1-E3's ``certificate_grid``
--------------------------------------------------
The state space is 60-dimensional, so a tensor-product quadrature grid is out.
Following E4 (24-D), we use the SHIFTED form: under the change of variables
x -> x + theta_p r inside the drift term, the implemented quadrature score
satisfies exactly

    int S_r . grad(f) dmu = -lam * sum_p w_p  E_mu[ r . grad(f)(x + theta_p r) ],

so the residual collapses to a self-normalised average of the POINTWISE
theta-quadrature defect, with no O(1) cancellation left to Monte Carlo.  Unlike
E4 we do not need a Laplace-mixture proposal: the well-tempered metadynamics
reference pool already IS a weighted sample of mu (``E5Reference``), so mu
expectations are taken directly against its importance weights.

The certificate is ATOMWISE (the stricter statement, per S1.3): for every frozen
displacement r the paired jump+drift kernel is individually mu-invariant, and
the mixture residual is bounded by the max over atoms.

Why the DIRECT form is reported but not gated
---------------------------------------------
The direct residual needs int p(x) S(x).grad(f)(x) dx.  Because
S_r(x) ~ e^{+beta U(x)}, the integrand p.S is O(1) exactly where p is
exponentially small -- the far field that carries order-one identity mass.  A
mu-weighted sample essentially never visits it, so the direct form is NOT
mu-estimable in 60-D; this is precisely why E4 (and E5) certify with the shifted
form.  We still report it on a generous and a tight domain to document the
effect, with the tight domain showing the truncation blow-up that E1-E3 see when
their integration box is shrunk.

Test functions
--------------
On the torus the admissible test functions are PERIODIC.  We use
f(q) = sin(m phi + n psi + c), which is smooth, bounded, 2pi-periodic in both
CVs, and has an analytic gradient.  Following the repo's high-dimensional lesson
(random ridges have a.r_hat ~ 1/sqrt(d) and are blind to the jump direction),
the (m, n) are chosen JUMP-ALIGNED: each retained atom contributes a mode whose
wavevector is along its own (dphi, dpsi).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..jumps import gauss_legendre_01


class TorusSine:
    """f(q) = sin(m*phi + n*psi + c) on the whitened internal state.

    phi and psi are whitened slots with D = 1, so they are the internal torsion
    angles themselves and f is exactly 2pi-periodic in each -- an admissible test
    function on the torus (a tanh ridge would not be).
    """

    def __init__(self, m: float, n: float, c: float, phi_slot: int,
                 psi_slot: int, d: int) -> None:
        self.m, self.n, self.c = float(m), float(n), float(c)
        self.phi_slot, self.psi_slot, self.d = phi_slot, psi_slot, d

    def _arg(self, q: torch.Tensor) -> torch.Tensor:
        return self.m * q[..., self.phi_slot] + self.n * q[..., self.psi_slot] + self.c

    def __call__(self, q: torch.Tensor) -> torch.Tensor:
        return torch.sin(self._arg(q))

    def grad(self, q: torch.Tensor) -> torch.Tensor:
        g = torch.zeros_like(q)
        cos = torch.cos(self._arg(q))
        g[..., self.phi_slot] = self.m * cos
        g[..., self.psi_slot] = self.n * cos
        return g


def torus_phi_family(pot, atoms: torch.Tensor, n_extra: int = 3,
                     seed: int = 314) -> list[TorusSine]:
    """Jump-aligned periodic test functions, plus a few generic low modes."""
    rng = np.random.default_rng(seed)
    phis = []
    # one mode per atom direction, normalised so the wavevector is O(1)
    for a in range(atoms.shape[0]):
        dphi = float(atoms[a, pot.phi_slot])
        dpsi = float(atoms[a, pot.psi_slot])
        nrm = math.hypot(dphi, dpsi)
        if nrm < 1e-12:
            continue
        phis.append(TorusSine(dphi / nrm, dpsi / nrm, float(rng.uniform(-1, 1)),
                              pot.phi_slot, pot.psi_slot, pot.d))
    # generic low-order lattice modes (integer wavevectors are exactly periodic)
    for (m, n) in [(1, 0), (0, 1), (1, 1), (1, -1), (2, 1)][:n_extra + 2]:
        phis.append(TorusSine(m, n, float(rng.uniform(-1, 1)),
                              pot.phi_slot, pot.psi_slot, pot.d))
    return phis


def certificate_atomwise_weighted(pot, samples: torch.Tensor,
                                  weights: torch.Tensor, atoms: torch.Tensor,
                                  atom_weights: torch.Tensor, lam: float,
                                  beta: float, phis: list, *,
                                  q_theta: int = 16,
                                  mask: torch.Tensor | None = None,
                                  score=None,
                                  max_samples: int | None = 4000) -> dict:
    """Atomwise shifted-form (gated) and direct-form (reported) residuals.

    samples/weights: a weighted mu-sample (the metadynamics reference pool).
    mask: optional boolean restriction defining a sub-domain; weights are
    renormalised over it, which is how the tight-domain reading is produced.
    """
    dev = samples.device
    if mask is not None:
        samples = samples[mask]
        weights = weights[mask]
    if max_samples is not None and samples.shape[0] > max_samples:
        # stride-thin the (serially correlated) pool to bound memory/time; the
        # weights ride along, so this stays a valid weighted mu-estimator
        step = int(np.ceil(samples.shape[0] / max_samples))
        samples = samples[::step]
        weights = weights[::step]
    w = weights / weights.sum()
    theta, w_theta = gauss_legendre_01(q_theta, dev)

    out = {"n_samples": int(samples.shape[0])}
    worst_shift, worst_direct = 0.0, 0.0
    mix_num_s = torch.zeros((), dtype=torch.float64, device=dev)
    mix_num_d = torch.zeros((), dtype=torch.float64, device=dev)
    mix_den = torch.zeros((), dtype=torch.float64, device=dev)
    max_log_mag = -float("inf")

    for a in range(atoms.shape[0]):
        r = atoms[a]
        wa = float(atom_weights[a])
        Rfix = r.unsqueeze(0).expand(samples.shape[0], -1).contiguous()
        S_dir = None
        if score is not None:
            with pot.no_count():
                S_dir, diag = score.score_for_shift(samples, Rfix)
            max_log_mag = max(max_log_mag, float(diag["max_log_magnitude"]))
        for i, f in enumerate(phis):
            f0 = f(samples)
            f1 = f(samples + r)
            J = lam * (w * (f1 - f0)).sum()
            # shifted drift: lam * sum_p w_p E[ grad f(x + theta_p r) . r ]
            chord = samples.unsqueeze(1) + theta.view(1, -1, 1) * r
            gdot = (f.grad(chord) * r).sum(-1)                    # (N, Qt)
            Dsh = lam * (w.unsqueeze(1) * w_theta.view(1, -1) * gdot).sum()
            res_s = float((torch.abs(J - Dsh) / torch.abs(J)).item())
            worst_shift = max(worst_shift, res_s)
            out[f"atom{a}_phi{i}"] = {"jump_term": float(J.item()),
                                      "residual_shifted": res_s}
            if S_dir is not None:
                Ddir = (w * (S_dir * f.grad(samples)).sum(-1)).sum()
                res_d = float((torch.abs(J + Ddir) / torch.abs(J)).item())
                worst_direct = max(worst_direct, res_d)
                out[f"atom{a}_phi{i}"]["residual_direct"] = res_d
                mix_num_d = mix_num_d + wa * (J + Ddir)
            mix_num_s = mix_num_s + wa * (J - Dsh)
            mix_den = mix_den + wa * J

    out["max_residual_shifted"] = worst_shift
    if score is not None:
        out["max_residual_direct"] = worst_direct
        out["max_log_magnitude"] = max_log_mag
        out["mixture_residual_direct"] = float(
            (torch.abs(mix_num_d) / torch.abs(mix_den)).item())
    out["mixture_residual_shifted"] = float(
        (torch.abs(mix_num_s) / torch.abs(mix_den)).item())
    return out


def tight_domain_mask(cvs: torch.Tensor, center, half_width: float) -> torch.Tensor:
    """Samples within `half_width` (per CV) of `center` -- the tight domain."""
    c = torch.as_tensor(center, dtype=cvs.dtype, device=cvs.device)
    d = (cvs - c).abs()
    d = torch.minimum(d, 2.0 * np.pi - d)
    return (d < half_width).all(dim=-1)
