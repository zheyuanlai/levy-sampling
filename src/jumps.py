"""Jump laws nu (probability measures on R^d) and their samplers.

The invariant enforced by tests: score integrals and jumps target the same
declared jump law. Deterministic Gauss--Legendre rules approximate expectations
under that continuous law; paired realised-bank MA has a literal conditional
finite-measure identity.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def gauss_legendre_01(n: int, device, dtype=torch.float64):
    """GL nodes/weights on [0,1]; weights sum to 1 (probability weights)."""
    x, w = np.polynomial.legendre.leggauss(n)
    nodes = torch.as_tensor(0.5 * (x + 1.0), dtype=dtype, device=device)
    weights = torch.as_tensor(0.5 * w, dtype=dtype, device=device)
    return nodes, weights


def gauss_legendre_m11(n: int, device, dtype=torch.float64):
    """GL nodes on [-1,1] with *probability* weights (sum to 1), matching a
    Unif(-1,1) variable; rescale nodes by h for rho ~ Unif(-h, h)."""
    x, w = np.polynomial.legendre.leggauss(n)
    nodes = torch.as_tensor(x, dtype=dtype, device=device)
    weights = torch.as_tensor(0.5 * w, dtype=dtype, device=device)
    return nodes, weights


class ShellJumpLaw:
    """Finite-atom shell law: r = r_a + rho u_a, a ~ w, rho ~ Unif(-h_a, h_a),
    u_a = r_a / ||r_a||.  h may be a scalar or per-atom (A,)."""

    def __init__(self, atoms: torch.Tensor, weights: torch.Tensor,
                 h: float | torch.Tensor) -> None:
        if (not isinstance(atoms, torch.Tensor) or atoms.ndim != 2
                or atoms.shape[0] < 1 or atoms.shape[1] < 1
                or not atoms.is_floating_point()):
            raise ValueError("atoms must be a nonempty rank-2 floating tensor")
        if not bool(torch.isfinite(atoms).all().item()):
            raise ValueError("atoms must be finite")
        norms = atoms.norm(dim=1, keepdim=True)
        if (not bool(torch.isfinite(norms).all().item())
                or bool((norms <= 0).any().item())):
            raise ValueError("every shell atom must have finite nonzero norm")
        weights_t = torch.as_tensor(
            weights, dtype=atoms.dtype, device=atoms.device)
        if weights_t.shape != (atoms.shape[0],):
            raise ValueError("weights must have shape (number of atoms,)")
        weight_sum = weights_t.sum()
        if (not bool(torch.isfinite(weights_t).all().item())
                or bool((weights_t < 0).any().item())
                or not bool(torch.isfinite(weight_sum).item())
                or float(weight_sum.item()) <= 0.0):
            raise ValueError("weights must be finite, nonnegative, and have positive sum")
        h_t = torch.as_tensor(h, dtype=atoms.dtype, device=atoms.device)
        try:
            h_t = h_t.expand(atoms.shape[0]).clone()
        except RuntimeError as exc:
            raise ValueError("h must be scalar or have one entry per atom") from exc
        if (not bool(torch.isfinite(h_t).all().item())
                or bool((h_t < 0).any().item())):
            raise ValueError("shell half-width h must be finite and nonnegative")

        self.atoms = atoms                                    # (A, d)
        self.weights = weights_t / weight_sum                 # (A,)
        self.d = atoms.shape[1]
        self.A = atoms.shape[0]
        self.h = h_t                                          # (A,)
        self.units = atoms / norms                            # (A, d)

    def sample(self, n: int, gen: torch.Generator) -> torch.Tensor:
        dev = self.atoms.device
        a = torch.multinomial(self.weights.expand(n, -1), 1, generator=gen).squeeze(1)
        rho = (torch.rand(n, 1, generator=gen, device=dev) * 2.0 - 1.0) * self.h[a].unsqueeze(1)
        return self.atoms[a] + rho * self.units[a]

    def quadrature_shifts(self, q_rho: int):
        """All r_{a,q} = r_a + rho_q h_a u_a with probability weights.

        Returns (shifts (A*Qr, d), log_weights (A*Qr,)); weights are
        w_a * w_hat_q and sum to 1. This is a normalized quadrature rule for
        the declared shell law, not the continuous sampling measure itself."""
        dev = self.atoms.device
        rho, wr = gauss_legendre_m11(q_rho, dev)
        radial = self.h.view(-1, 1) * rho.view(1, -1)                 # (A, Qr)
        shifts = self.atoms.unsqueeze(1) + radial.unsqueeze(-1) * self.units.unsqueeze(1)
        w = self.weights.unsqueeze(1) * wr.unsqueeze(0)               # (A, Qr)
        return shifts.reshape(-1, self.d), torch.log(w.reshape(-1))

    def max_reach(self) -> float:
        return float((self.atoms.norm(dim=1) + self.h).max())


class JitteredShellJumpLaw(ShellJumpLaw):
    """Shell law plus fresh per-draw isotropic Gaussian jitter:
    r = r_a + rho u_a + sigma xi,  xi ~ N(0, I_d).

    SAMPLED-BANK LSC ONLY. Random-atomic and paired multi-atom scores integrate
    the chord for the same realised displacement(s) used by their jump update,
    so neither needs closed-form quadrature over the jitter distribution.
    ``quadrature_shifts`` is unsupported because the jittered measure has no
    finite-atom representation; deterministic exact-quadrature scores and
    certificates therefore cannot use this law. Off by default (sigma = 0).
    """

    def __init__(self, atoms, weights, h, jitter_sigma: float) -> None:
        super().__init__(atoms, weights, h)
        self.jitter_sigma = float(jitter_sigma)
        if (not math.isfinite(self.jitter_sigma)
                or self.jitter_sigma < 0.0):
            raise ValueError("jitter_sigma must be finite and nonnegative")

    def sample(self, n: int, gen: torch.Generator) -> torch.Tensor:
        r = super().sample(n, gen)
        if self.jitter_sigma > 0.0:
            r = r + self.jitter_sigma * torch.randn(
                r.shape, generator=gen, device=r.device, dtype=r.dtype)
        return r

    def quadrature_shifts(self, q_rho: int):
        raise NotImplementedError(
            "JitteredShellJumpLaw has no finite-atom quadrature "
            "(sampled-bank RA/MA LSC only)")


class AnnulusJumpLaw:
    """E2 law: r = rho u_phi, rho ~ Unif[a, b], phi ~ Unif[0, 2 pi). d = 2.

    Deliberately generic: it encodes only a plausible mode-spacing scale
    [a, b], no mode locations."""

    d = 2

    def __init__(self, a: float, b: float, device) -> None:
        self.a, self.b = float(a), float(b)
        self.device = device

    def sample(self, n: int, gen: torch.Generator) -> torch.Tensor:
        dev = self.device
        rho = self.a + (self.b - self.a) * torch.rand(n, generator=gen, device=dev)
        phi = 2.0 * math.pi * torch.rand(n, generator=gen, device=dev)
        return torch.stack([rho * torch.cos(phi), rho * torch.sin(phi)], dim=1)

    def quadrature_shifts(self, q_rho: int, m_phi: int):
        """Tensor quadrature of nu (used by the certificate's jump side):
        m_phi trapezoid directions x GL-q_rho radii with probability weights.
        The closed-form score integrates rho and theta analytically, so a
        fine q_rho here isolates the direction-quadrature defect."""
        dev = self.device
        x, w = np.polynomial.legendre.leggauss(q_rho)
        rho = torch.as_tensor(0.5 * (self.b - self.a) * (x + 1.0) + self.a,
                              dtype=torch.float64, device=dev)
        wr = torch.as_tensor(0.5 * w, dtype=torch.float64, device=dev)   # sums to 1
        phi = torch.arange(m_phi, dtype=torch.float64, device=dev) * (2.0 * math.pi / m_phi)
        u = torch.stack([torch.cos(phi), torch.sin(phi)], dim=1)         # (M, 2)
        shifts = rho.view(-1, 1, 1) * u.view(1, -1, 2)                   # (Qr, M, 2)
        wgt = (wr / m_phi).view(-1, 1).expand(-1, m_phi)                 # (Qr, M)
        return shifts.reshape(-1, 2), torch.log(wgt.reshape(-1))

    def max_reach(self) -> float:
        return self.b


def apply_poisson_jumps(
    x: torch.Tensor,
    law,
    lam: float,
    dt: float,
    gen: torch.Generator,
    k_max: int = 8,
    *,
    return_sampled_counts: bool = False,
):
    """Apply capped full-law Poisson jumps without hiding cap exceedances.

    The default two-value return remains ``(x_new, jumps_applied)`` for legacy
    callers.  ``return_sampled_counts=True`` additionally returns the exact
    sampled Poisson multiplicities before the fixed ``k_max`` implementation
    cap.  This lets production diagnostics distinguish requested occurrences
    from applied jumps and fail closed if the nominally negligible tail fires.

    The fixed loop performs no host synchronization.  Jumps are always drawn
    (active or not) so paired raw CP and LSC-CP consume identical streams.
    """
    n = x.shape[0]
    counts = torch.poisson(torch.full(
        (n,), lam * dt, device=x.device, dtype=x.dtype), generator=gen)
    # one batched draw of all k_max candidates (identical stream for any two
    # samplers sharing `gen`; ~6x fewer kernel launches than a k-loop)
    A = law.sample(n * k_max, gen).reshape(k_max, n, x.shape[1])
    ks = torch.arange(k_max, device=x.device, dtype=counts.dtype).unsqueeze(1)
    mask = (counts.unsqueeze(0) > ks).to(x.dtype)            # (k_max, n)
    x = x + (mask.unsqueeze(-1) * A).sum(dim=0)
    applied = torch.clamp(counts, max=k_max)
    if return_sampled_counts:
        return x, applied, counts
    return x, applied
