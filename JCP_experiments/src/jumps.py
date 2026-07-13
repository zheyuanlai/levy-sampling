"""Jump laws nu (probability measures on R^d) and their samplers.

The invariant enforced by tests: the nu used by the Levy score (quadrature)
and the nu used to generate jumps in the sampler are the SAME measure.
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
        assert atoms.ndim == 2
        self.atoms = atoms                                    # (A, d)
        self.weights = weights / weights.sum()                # (A,)
        self.d = atoms.shape[1]
        self.A = atoms.shape[0]
        h_t = torch.as_tensor(h, dtype=atoms.dtype, device=atoms.device)
        self.h = h_t.expand(self.A).clone()                   # (A,)
        self.units = atoms / atoms.norm(dim=1, keepdim=True)  # (A, d)

    def sample(self, n: int, gen: torch.Generator) -> torch.Tensor:
        dev = self.atoms.device
        a = torch.multinomial(self.weights.expand(n, -1), 1, generator=gen).squeeze(1)
        rho = (torch.rand(n, 1, generator=gen, device=dev) * 2.0 - 1.0) * self.h[a].unsqueeze(1)
        return self.atoms[a] + rho * self.units[a]

    def quadrature_shifts(self, q_rho: int):
        """All r_{a,q} = r_a + rho_q h_a u_a with probability weights.

        Returns (shifts (A*Qr, d), log_weights (A*Qr,)); weights are
        w_a * w_hat_q and sum to 1, matching the sampling measure exactly."""
        dev = self.atoms.device
        rho, wr = gauss_legendre_m11(q_rho, dev)
        radial = self.h.view(-1, 1) * rho.view(1, -1)                 # (A, Qr)
        shifts = self.atoms.unsqueeze(1) + radial.unsqueeze(-1) * self.units.unsqueeze(1)
        w = self.weights.unsqueeze(1) * wr.unsqueeze(0)               # (A, Qr)
        return shifts.reshape(-1, self.d), torch.log(w.reshape(-1))

    def max_reach(self) -> float:
        return float((self.atoms.norm(dim=1) + self.h).max())


class JitteredShellJumpLaw(ShellJumpLaw):
    """Shell law + fresh per-draw transverse Gaussian jitter:
    r = r_a + rho u_a + sigma xi,  xi ~ N(0, I_d).

    RA-LSC ONLY. The random-atomic score integrates the chord for the realised
    r, so no closed-form quadrature over the jump law is needed -- continuous
    jitter is free. `quadrature_shifts` is therefore unsupported (there is no
    finite-atom representation of the jittered nu), so the exact-quadrature
    score and the certificate cannot use this law. Off by default (sigma = 0)."""

    def __init__(self, atoms, weights, h, jitter_sigma: float) -> None:
        super().__init__(atoms, weights, h)
        self.jitter_sigma = float(jitter_sigma)

    def sample(self, n: int, gen: torch.Generator) -> torch.Tensor:
        r = super().sample(n, gen)
        if self.jitter_sigma > 0.0:
            r = r + self.jitter_sigma * torch.randn(
                r.shape, generator=gen, device=r.device, dtype=r.dtype)
        return r

    def quadrature_shifts(self, q_rho: int):
        raise NotImplementedError(
            "JitteredShellJumpLaw has no finite-atom quadrature (RA-LSC only)")


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


def apply_poisson_jumps(x: torch.Tensor, law, lam: float, dt: float,
                        gen: torch.Generator, k_max: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """x <- x + sum_{k=1}^{N} A_k, N ~ Poisson(lam dt), A_k ~ nu i.i.d.

    Vectorised with a fixed unrolled loop of k_max candidate jumps (no host
    sync). For lam*dt <= 0.01 the truncation P(N > 8) < 1e-20 per particle
    per step. Jumps are always *drawn* (active or not) so that two samplers
    sharing this generator consume identical streams (pathwise coupling of
    raw CP and LSC-CP).

    Returns (x_new, jumps_applied (N,))."""
    n = x.shape[0]
    counts = torch.poisson(torch.full((n,), lam * dt, device=x.device), generator=gen)
    # one batched draw of all k_max candidates (identical stream for any two
    # samplers sharing `gen`; ~6x fewer kernel launches than a k-loop)
    A = law.sample(n * k_max, gen).reshape(k_max, n, x.shape[1])
    ks = torch.arange(k_max, device=x.device, dtype=counts.dtype).unsqueeze(1)
    mask = (counts.unsqueeze(0) > ks).to(x.dtype)            # (k_max, n)
    x = x + (mask.unsqueeze(-1) * A).sum(dim=0)
    return x, torch.clamp(counts, max=k_max)
