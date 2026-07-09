"""Reference samplers and basin partitions.

E1: inverse-CDF on a dense grid (tail mass < 1e-30 outside the box).
E2: exact i.i.d. draws from the mixture (see MoG40.sample_exact).
E3: grid inverse-CDF on the 2D MB latent marginal x exact Gaussian aux,
    pushed through z -> z B^T.
E4: harmonic (Laplace) mixture - a REFERENCE, not ground truth; a long PT
    chain is run as a cross-check.
"""
from __future__ import annotations

import math

import numpy as np
import torch


# ---------------------------------------------------------------- E1 (1D)
class Grid1DInverseCDF:
    def __init__(self, log_density, lo: float, hi: float, n_grid: int = 200_001,
                 device="cuda") -> None:
        x = torch.linspace(lo, hi, n_grid, dtype=torch.float64, device=device)
        logp = log_density(x)
        p = torch.exp(logp - logp.max())
        cdf = torch.cumsum(0.5 * (p[1:] + p[:-1]) * (x[1:] - x[:-1]), dim=0)
        self.cdf = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), cdf])
        self.cdf = self.cdf / self.cdf[-1].clone()
        self.x = x

    def sample(self, n: int, gen: torch.Generator) -> torch.Tensor:
        u = torch.rand(n, generator=gen, device=self.x.device, dtype=torch.float64)
        idx = torch.clamp(torch.searchsorted(self.cdf, u), 1, self.x.shape[0] - 1)
        c0, c1 = self.cdf[idx - 1], self.cdf[idx]
        frac = (u - c0) / torch.clamp(c1 - c0, min=1e-300)
        return (self.x[idx - 1] + frac * (self.x[idx] - self.x[idx - 1])).unsqueeze(1)


# ---------------------------------------------------------------- 2D grids
class Grid2DSampler:
    """Categorical over fine cells + uniform jitter within the cell."""

    def __init__(self, log_density, lo, hi, shape=(2400, 2400), device="cuda") -> None:
        self.lo = torch.as_tensor(lo, dtype=torch.float64, device=device)
        self.hi = torch.as_tensor(hi, dtype=torch.float64, device=device)
        self.shape = shape
        nx, ny = shape
        xs = torch.linspace(float(lo[0]), float(hi[0]), nx + 1, dtype=torch.float64, device=device)
        ys = torch.linspace(float(lo[1]), float(hi[1]), ny + 1, dtype=torch.float64, device=device)
        cx = 0.5 * (xs[1:] + xs[:-1])
        cy = 0.5 * (ys[1:] + ys[:-1])
        gx, gy = torch.meshgrid(cx, cy, indexing="ij")
        pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
        logp = log_density(pts)
        p = torch.exp(logp - logp.max())
        self.cdf = torch.cumsum(p, dim=0)
        self.cdf = self.cdf / self.cdf[-1].clone()
        self.cell = torch.stack([(xs[1] - xs[0]).reshape(()), (ys[1] - ys[0]).reshape(())])
        self.centers = pts

    def sample(self, n: int, gen: torch.Generator) -> torch.Tensor:
        dev = self.cdf.device
        u = torch.rand(n, generator=gen, device=dev, dtype=torch.float64)
        idx = torch.clamp(torch.searchsorted(self.cdf, u), max=self.cdf.shape[0] - 1)
        jitter = (torch.rand(n, 2, generator=gen, device=dev, dtype=torch.float64) - 0.5) * self.cell
        return self.centers[idx] + jitter


class MB10DReference:
    """Latent 2D MB marginal (grid inverse-CDF) x N(0, eps sigma_aux^2 I_8),
    pushed through z -> z B^T."""

    def __init__(self, pot, lo2d, hi2d, beta: float, shape=(2400, 2400)) -> None:
        from .potentials import muller_brown_2d
        self.pot = pot
        self.beta = beta
        self.grid = Grid2DSampler(
            lambda z: -(beta / pot.s) * muller_brown_2d(z), lo2d, hi2d,
            shape=shape, device=pot.B.device)
        self.aux_std = math.sqrt((1.0 / beta)) * pot.sigma_aux

    def sample(self, n: int, gen: torch.Generator) -> torch.Tensor:
        z2 = self.grid.sample(n, gen)
        aux = self.aux_std * torch.randn(n, 8, generator=gen,
                                         device=z2.device, dtype=torch.float64)
        return self.pot.from_latent(torch.cat([z2, aux], dim=1))


# ---------------------------------------------------------------- E4 Laplace
class LaplaceMixture:
    """Harmonic reference: phase k with weight ~ exp(-beta V_k)/sqrt(det H_k),
    fluctuations N(0, eps H_k^{-1}). Provides log_q for importance sampling."""

    def __init__(self, means: torch.Tensor, hessians: torch.Tensor,
                 energies: torch.Tensor, beta: float) -> None:
        self.means = means                                   # (K, d)
        self.beta = beta
        K, d = means.shape
        self.d = d
        sign, logdet = torch.linalg.slogdet(hessians)
        assert bool((sign > 0).all().item()), "Hessians must be SPD"
        log_w = -beta * energies - 0.5 * logdet
        self.log_weights = log_w - torch.logsumexp(log_w, dim=0)
        self.weights = torch.exp(self.log_weights)
        # fluctuation covariance eps H^{-1} = (beta H)^{-1}
        cov = torch.linalg.inv(hessians) / beta
        self.chol = torch.linalg.cholesky(cov)               # (K, d, d)
        self.prec = hessians * beta                          # (K, d, d)
        _, logdet_cov = torch.linalg.slogdet(cov)
        self.log_norm = -0.5 * (d * math.log(2.0 * math.pi) + logdet_cov)  # (K,)

    def sample(self, n: int, gen: torch.Generator) -> torch.Tensor:
        dev = self.means.device
        k = torch.multinomial(self.weights.expand(n, -1), 1, generator=gen).squeeze(1)
        z = torch.randn(n, self.d, generator=gen, device=dev, dtype=torch.float64)
        return self.means[k] + torch.einsum("nij,nj->ni", self.chol[k], z)

    def log_q(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - self.means.unsqueeze(0)      # (N, K, d)
        quad = torch.einsum("nkd,kde,nke->nk", diff, self.prec, diff)
        comp = self.log_weights + self.log_norm - 0.5 * quad
        return torch.logsumexp(comp, dim=1)


# ---------------------------------------------------------------- basins
class GradientFlowBasinMap2D:
    """Basin-of-attraction map of a 2D potential on a grid, computed by
    damped gradient descent to convergence, cached to .npz. Assignment of
    samples = nearest-cell lookup."""

    def __init__(self, grad_fn, minima: torch.Tensor, lo, hi,
                 n_grid: int = 600, device="cuda", cache: str | None = None,
                 dt_flow: float = 1.5e-4, n_flow: int = 40_000) -> None:
        # dt_flow is set by the stiffest Hessian eigenvalue among the 2D
        # potentials used here (Mueller-Brown ~6e3: dt*lam ~ 0.9 < 2), and
        # the tamed step caps wall gradients at unit displacement.
        import os
        self.lo = torch.as_tensor(lo, dtype=torch.float64, device=device)
        self.hi = torch.as_tensor(hi, dtype=torch.float64, device=device)
        self.n_grid = n_grid
        self.minima = minima
        if cache is not None and os.path.exists(cache):
            data = np.load(cache)
            self.labels = torch.as_tensor(data["labels"], device=device)
            return
        xs = torch.linspace(float(lo[0]), float(hi[0]), n_grid, dtype=torch.float64, device=device)
        ys = torch.linspace(float(lo[1]), float(hi[1]), n_grid, dtype=torch.float64, device=device)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        z = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
        for _ in range(n_flow):
            g = grad_fn(z)
            gn = g.norm(dim=1, keepdim=True)
            z = z - dt_flow * g / (1.0 + dt_flow * gn)       # tamed flow, stable
            z = torch.clamp(z, self.lo, self.hi)
        d2 = ((z.unsqueeze(1) - minima.unsqueeze(0)) ** 2).sum(-1)
        self.labels = d2.argmin(dim=1).reshape(n_grid, n_grid)
        if cache is not None:
            np.savez(cache, labels=self.labels.cpu().numpy())

    def assign(self, z2: torch.Tensor) -> torch.Tensor:
        frac = (z2 - self.lo) / (self.hi - self.lo)
        ij = torch.clamp((frac * self.n_grid).long(), 0, self.n_grid - 1)
        return self.labels[ij[:, 0], ij[:, 1]]

    def p_star(self, log_density, n_quad: int = 1200) -> torch.Tensor:
        """Target occupancy by grid quadrature of the (unnormalised) density."""
        dev = self.lo.device
        xs = torch.linspace(float(self.lo[0]), float(self.hi[0]), n_quad,
                            dtype=torch.float64, device=dev)
        ys = torch.linspace(float(self.lo[1]), float(self.hi[1]), n_quad,
                            dtype=torch.float64, device=dev)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        z = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
        logp = log_density(z)
        p = torch.exp(logp - logp.max())
        lab = self.assign(z)
        K = self.minima.shape[0]
        mass = torch.zeros(K, dtype=torch.float64, device=dev)
        mass.scatter_add_(0, lab, p)
        return mass / mass.sum()
