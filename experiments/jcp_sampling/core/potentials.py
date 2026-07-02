
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class BasePotential:
    name: str
    dim: int
    beta: float = 1.0
    state_clip: float = 50.0

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def force(self, x: torch.Tensor) -> torch.Tensor:
        return -self.gradient(x)

    def minima(self) -> torch.Tensor:
        raise NotImplementedError

    def basin_labels(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def target_basin_probs(self, device=None) -> Optional[torch.Tensor]:
        return None

    def slow_cv(self, x: torch.Tensor) -> torch.Tensor:
        """Per-particle scalar slow collective variable used for IAT/ESS.

        Must be a *continuous / low-cardinality* coordinate that couples to the slow
        inter-basin mode and does not saturate. The default (basin label as float) is fine
        for low-cardinality basin sets; high-dimensional targets override with a bounded
        collective variable (e.g. a deep-well count) so the ensemble mean keeps fluctuating.
        """
        return self.basin_labels(x).to(torch.float32)

    def reference(self, n: int, seed: int, device=None) -> torch.Tensor:
        raise NotImplementedError

    def initial_state(self, n: int, seed: int, device=None) -> torch.Tensor:
        raise NotImplementedError

    def observables(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"energy": self.potential(x)}

    def metadata(self) -> dict:
        return {"name": self.name, "dim": self.dim, "beta": self.beta, "state_clip": self.state_clip}


def _inverse_cdf_1d(V_fn, beta: float, lo: float, hi: float, grid_n: int, n: int, seed: int):
    rng = np.random.default_rng(seed)
    grid = np.linspace(lo, hi, int(grid_n))
    logp = -float(beta) * V_fn(grid)
    logp -= logp.max()
    p = np.exp(logp)
    dx = grid[1] - grid[0]
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[:-1] + p[1:]) * dx)])
    cdf /= cdf[-1]
    u = rng.random(n)
    return np.interp(u, cdf, grid)


class DoubleWell1D(BasePotential):
    def __init__(self, beta: float = 8.0, state_clip: float = 4.0):
        super().__init__("double_well", 1, float(beta), float(state_clip))

    def potential(self, x):
        z = x[..., 0]
        return 0.25 * (z * z - 1.0) ** 2

    def gradient(self, x):
        z = x[..., 0]
        return (z * (z * z - 1.0)).unsqueeze(-1)

    def minima(self):
        return torch.tensor([[-1.0], [1.0]], dtype=torch.float32)

    def basin_labels(self, x):
        return (x[..., 0] >= 0).long()

    def target_basin_probs(self, device=None):
        return torch.tensor([0.5, 0.5], device=device, dtype=torch.float32)

    def reference(self, n, seed, device=None):
        arr = _inverse_cdf_1d(lambda z: 0.25 * (z*z - 1.0)**2, self.beta, -3.0, 3.0, 20001, n, seed)
        return torch.tensor(arr[:, None], device=device, dtype=torch.float32)

    def initial_state(self, n, seed, device=None):
        g = torch.Generator(device=device); g.manual_seed(int(seed))
        return -1.0 + 0.08 * torch.randn(n, 1, generator=g, device=device)

    def slow_cv(self, x):
        return x[..., 0]

    def observables(self, x):
        return {"energy": self.potential(x), "x": x[..., 0], "x2": x[..., 0] ** 2,
                "right_basin": (x[..., 0] >= 0).float()}


class TripleWell1D(BasePotential):
    """Triple-well target defined as a normalized Gaussian mixture ``p_star``.

    We set ``V(x) = -eps * log p_star(x)`` and ``beta = 1/eps`` so that ``exp(-beta V) = p_star``
    exactly (the sampler's Gibbs target *is* the mixture, independent of eps). ``eps`` then only
    controls the ratio of local-diffusion to jump time scales: a small ``eps`` slows the local
    overdamped diffusion (``drift = eps * grad log p_star``, noise ``sqrt(2 eps)``) so the three
    modes become metastable, which is the manuscript's operating regime (eps=0.08). The jump law
    and the stationary Levy-score density ratios ``p_star(x-theta r)/p_star(x)`` are eps-independent.
    Modes m=(-3,0,3), scales s=(0.50,0.75,0.50), weights (5/21, 3/7, 1/3).
    """

    def __init__(self, eps: float = 0.08, state_clip: float = 6.0):
        super().__init__("triple_well", 1, 1.0 / float(eps), float(state_clip))
        self.eps = float(eps)
        self.modes = torch.tensor([-3.0, 0.0, 3.0], dtype=torch.float32)
        self.scales = torch.tensor([0.50, 0.75, 0.50], dtype=torch.float32)
        self.mix = torch.tensor([5.0 / 21.0, 3.0 / 7.0, 1.0 / 3.0], dtype=torch.float32)
        self.split = (-1.5, 1.5)  # basin boundaries between the three modes
        self._probs_cache = None

    def _log_components(self, z):
        m = self.modes.to(z.device, z.dtype)
        s = self.scales.to(z.device, z.dtype)
        w = self.mix.to(z.device, z.dtype)
        zc = z[..., None]
        log_phi = -0.5 * math.log(2 * math.pi) - torch.log(s) - 0.5 * ((zc - m) / s) ** 2
        return torch.log(w) + log_phi

    def log_pstar(self, z):
        return torch.logsumexp(self._log_components(z), dim=-1)

    def potential(self, x):
        return -self.eps * self.log_pstar(x[..., 0])

    def gradient(self, x):
        z = x[..., 0]
        m = self.modes.to(z.device, z.dtype)
        s = self.scales.to(z.device, z.dtype)
        r = torch.softmax(self._log_components(z), dim=-1)          # responsibilities (...,3)
        dlog = (r * (-(z[..., None] - m) / s ** 2)).sum(-1)         # d/dz log p_star
        return (-self.eps * dlog).unsqueeze(-1)

    def minima(self):
        return self.modes.clone().unsqueeze(-1)

    def basin_labels(self, x):
        z = x[..., 0]
        bounds = torch.tensor(self.split, device=z.device, dtype=z.dtype)
        return torch.bucketize(z, bounds)

    def target_basin_probs(self, device=None):
        if self._probs_cache is None:
            grid = np.linspace(-9.0, 9.0, 60001)
            logp = self.log_pstar(torch.tensor(grid, dtype=torch.float32)).numpy()
            p = np.exp(logp - logp.max()); dx = grid[1] - grid[0]; p /= p.sum() * dx
            lab = np.digitize(grid, list(self.split))
            probs = np.array([p[lab == k].sum() * dx for k in range(3)], dtype=float)
            self._probs_cache = probs / probs.sum()
        return torch.tensor(self._probs_cache, device=device, dtype=torch.float32)

    def reference(self, n, seed, device=None):
        rng = np.random.default_rng(seed)
        comp = rng.choice(3, size=n, p=self.mix.numpy())
        z = rng.normal(self.modes.numpy()[comp], self.scales.numpy()[comp])
        return torch.tensor(z[:, None], device=device, dtype=torch.float32)

    def initial_state(self, n, seed, device=None):
        # start every chain in the left mode to test inter-mode communication
        g = torch.Generator(device=device); g.manual_seed(int(seed))
        return -3.0 + 0.1 * torch.randn(n, 1, generator=g, device=device)

    def slow_cv(self, x):
        return x[..., 0]

    def observables(self, x):
        z = x[..., 0]
        return {"energy": self.potential(x), "x": z,
                "middle_mass": ((z > self.split[0]) & (z < self.split[1])).float(),
                "left_mass": (z <= self.split[0]).float(),
                "right_mass": (z >= self.split[1]).float()}

    def metadata(self):
        md = super().metadata(); md.update({"eps": self.eps, "modes": self.modes.tolist()}); return md


class FourWell2D(BasePotential):
    def __init__(self, beta: float = 4.0, state_clip: float = 4.0):
        super().__init__("four_well", 2, float(beta), float(state_clip))

    def potential(self, x):
        return ((x[..., 0] ** 2 - 1.0) ** 2 + (x[..., 1] ** 2 - 1.0) ** 2)

    def gradient(self, x):
        g = torch.empty_like(x)
        g[..., 0] = 4.0 * x[..., 0] * (x[..., 0] ** 2 - 1.0)
        g[..., 1] = 4.0 * x[..., 1] * (x[..., 1] ** 2 - 1.0)
        return g

    def minima(self):
        return torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=torch.float32)

    def basin_labels(self, x):
        sx = (x[..., 0] >= 0).long()
        sy = (x[..., 1] >= 0).long()
        return 2 * sx + sy

    def target_basin_probs(self, device=None):
        return torch.full((4,), 0.25, device=device, dtype=torch.float32)

    def reference(self, n, seed, device=None):
        x = _inverse_cdf_1d(lambda z: (z*z - 1.0)**2, self.beta, -3.0, 3.0, 20001, n, seed)
        y = _inverse_cdf_1d(lambda z: (z*z - 1.0)**2, self.beta, -3.0, 3.0, 20001, n, seed + 19)
        return torch.tensor(np.stack([x, y], axis=1), device=device, dtype=torch.float32)

    def initial_state(self, n, seed, device=None):
        g = torch.Generator(device=device); g.manual_seed(int(seed))
        base = torch.tensor([-1.0, -1.0], device=device)
        return base + 0.08 * torch.randn(n, 2, generator=g, device=device)

    def slow_cv(self, x):
        # number of positive coordinates (0..2); starts at 0 in basin (-1,-1), ~1 at equilibrium
        return (x >= 0).to(x.dtype).sum(-1)

    def observables(self, x):
        return {"energy": self.potential(x), "x": x[..., 0], "y": x[..., 1],
                "r2": (x ** 2).sum(-1)}


class MuellerBrown2D(BasePotential):
    PARAMS = torch.tensor([
        [-200.0, -1.0, 0.0, -10.0, 1.0, 0.0],
        [-100.0, -1.0, 0.0, -10.0, 0.0, 0.5],
        [-170.0, -6.5, 11.0, -6.5, -0.5, 1.5],
        [15.0, 0.7, 0.6, 0.7, -1.0, 1.0],
    ], dtype=torch.float32)

    def __init__(self, beta: float = 12.0, scale: float = 0.05, state_clip: float = 5.0,
                 grid_n: int = 120):
        super().__init__("muller_brown", 2, float(beta), float(state_clip))
        self.scale = float(scale)
        self.grid_n = int(grid_n)
        self.domain = (-1.8, 1.2, -0.4, 2.1)
        self._minima = torch.tensor([[-0.558, 1.442], [0.623, 0.028], [-0.050, 0.467]], dtype=torch.float32)
        self._grid_cache = None

    def potential(self, x):
        p = self.PARAMS.to(device=x.device, dtype=x.dtype)
        X = x[..., 0][..., None]
        Y = x[..., 1][..., None]
        A, a, b, c, x0, y0 = [p[:, i] for i in range(6)]
        dx = X - x0; dy = Y - y0
        val = (A * self.scale) * torch.exp((a * dx * dx + b * dx * dy + c * dy * dy).clamp(-80.0, 80.0))
        return val.sum(-1)

    def gradient(self, x):
        p = self.PARAMS.to(device=x.device, dtype=x.dtype)
        X = x[..., 0][..., None]
        Y = x[..., 1][..., None]
        A, a, b, c, x0, y0 = [p[:, i] for i in range(6)]
        dx = X - x0; dy = Y - y0
        exp_term = (A * self.scale) * torch.exp((a * dx * dx + b * dx * dy + c * dy * dy).clamp(-80.0, 80.0))
        gx = (exp_term * (2 * a * dx + b * dy)).sum(-1)
        gy = (exp_term * (b * dx + 2 * c * dy)).sum(-1)
        return torch.stack([gx, gy], dim=-1)

    def minima(self):
        return self._minima.clone()

    def basin_labels(self, x):
        m = self._minima.to(device=x.device, dtype=x.dtype)
        return torch.cdist(x.reshape(-1, 2), m).argmin(-1).reshape(x.shape[:-1])

    def _grid_reference_np(self):
        if self._grid_cache is not None:
            return self._grid_cache
        xmin, xmax, ymin, ymax = self.domain
        gx = np.linspace(xmin, xmax, self.grid_n)
        gy = np.linspace(ymin, ymax, self.grid_n)
        X, Y = np.meshgrid(gx, gy, indexing="xy")
        pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
        with torch.no_grad():
            V = self.potential(pts).cpu().numpy().reshape(self.grid_n, self.grid_n)
        logp = -self.beta * V; logp -= np.max(logp)
        P = np.exp(logp)
        dx = gx[1] - gx[0]; dy = gy[1] - gy[0]
        P /= (P.sum() * dx * dy)
        labels = self.basin_labels(torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)).numpy().reshape(P.shape)
        probs = np.array([(P[labels == k].sum() * dx * dy) for k in range(3)], dtype=float)
        self._grid_cache = (gx, gy, P, probs / probs.sum())
        return self._grid_cache

    def grid_reference(self):
        return self._grid_reference_np()

    def target_basin_probs(self, device=None):
        _, _, _, probs = self._grid_reference_np()
        return torch.tensor(probs, device=device, dtype=torch.float32)

    def reference(self, n, seed, device=None):
        gx, gy, P, _ = self._grid_reference_np()
        rng = np.random.default_rng(seed)
        flat = P.ravel(); flat = flat / flat.sum()
        idx = rng.choice(flat.size, size=n, p=flat)
        iy, ix = np.unravel_index(idx, P.shape)
        dx = gx[1] - gx[0]; dy = gy[1] - gy[0]
        x = gx[ix] + rng.uniform(-0.5 * dx, 0.5 * dx, size=n)
        y = gy[iy] + rng.uniform(-0.5 * dy, 0.5 * dy, size=n)
        return torch.tensor(np.stack([x, y], axis=1), device=device, dtype=torch.float32)

    def initial_state(self, n, seed, device=None):
        g = torch.Generator(device=device); g.manual_seed(int(seed))
        base = self._minima[0].to(device=device)
        return base + 0.05 * torch.randn(n, 2, generator=g, device=device)

    def slow_cv(self, x):
        return x[..., 0]

    def observables(self, x):
        return {"energy": self.potential(x), "x": x[..., 0], "y": x[..., 1], "r2": (x ** 2).sum(-1)}

    def metadata(self):
        md = super().metadata(); md.update({"scale": self.scale, "domain": self.domain, "grid_n": self.grid_n}); return md


class TransformedMuellerBrown10D(BasePotential):
    """Muller-Brown embedded in ``dim`` dimensions and linearly mixed.

    Latent energy ``U10(z) = U_MB(z1,z2) + (1/(2 sigma^2)) sum_{l>=3} z_l^2`` (sigma=0.75). The
    sampler evolves the *mixed* coordinate ``x = z B^T`` for a fixed seeded orthogonal ``B`` (so
    ``z = x B``); basin labels and jump directions are computed by projecting back to ``z``.
    ``beta = 1/eps`` (eps=0.5). Only the 2D latent block is metastable; the aux dims are Gaussian.
    """

    def __init__(self, eps: float = 0.5, dim: int = 10, sigma_aux: float = 0.75,
                 scale: float = 0.02, mix_seed: int = 0, state_clip: float = 8.0, grid_n: int = 150):
        super().__init__("muller10d", int(dim), 1.0 / float(eps), float(state_clip))
        self.eps = float(eps)
        self.sigma_aux = float(sigma_aux)
        self.mb = MuellerBrown2D(beta=1.0 / float(eps), scale=scale, grid_n=grid_n)
        rng = np.random.default_rng(mix_seed)
        Q, _ = np.linalg.qr(rng.standard_normal((int(dim), int(dim))))  # orthogonal mixing
        self.B = torch.tensor(Q, dtype=torch.float32)                   # x = z @ B^T ; z = x @ B
        self._mb_min = self.mb.minima()                                 # (3,2)

    def _to_latent(self, x):
        return x @ self.B.to(x.device, x.dtype)

    def _to_mixed(self, z):
        return z @ self.B.to(z.device, z.dtype).T

    def potential(self, x):
        z = self._to_latent(x)
        aux = 0.5 / self.sigma_aux ** 2 * (z[..., 2:] ** 2).sum(-1)
        return self.mb.potential(z[..., :2]) + aux

    def gradient(self, x):
        z = self._to_latent(x)
        gz = torch.empty_like(z)
        gz[..., :2] = self.mb.gradient(z[..., :2])
        gz[..., 2:] = z[..., 2:] / self.sigma_aux ** 2
        return gz @ self.B.to(x.device, x.dtype).T                      # dV/dx = gz @ B^T

    def minima(self):
        zmin = torch.zeros(self._mb_min.shape[0], self.dim)
        zmin[:, :2] = self._mb_min
        return self._to_mixed(zmin)

    def basin_labels(self, x):
        return self.mb.basin_labels(self._to_latent(x)[..., :2])

    def target_basin_probs(self, device=None):
        return self.mb.target_basin_probs(device=device)

    def reference(self, n, seed, device=None):
        z2d = self.mb.reference(n, seed, device="cpu").numpy()
        rng = np.random.default_rng(seed + 7)
        aux_std = self.sigma_aux / math.sqrt(self.beta)
        aux = rng.normal(0.0, aux_std, size=(n, self.dim - 2))
        zf = torch.tensor(np.concatenate([z2d, aux], axis=1), dtype=torch.float32)
        return self._to_mixed(zf).to(device=device)

    def initial_state(self, n, seed, device=None):
        g = torch.Generator(device=device); g.manual_seed(int(seed))
        aux_std = self.sigma_aux / math.sqrt(self.beta)
        z = torch.zeros(n, self.dim, device=device)
        z[:, :2] = self._mb_min[0].to(device) + 0.05 * torch.randn(n, 2, generator=g, device=device)
        z[:, 2:] = aux_std * torch.randn(n, self.dim - 2, generator=g, device=device)
        return self._to_mixed(z)

    def slow_cv(self, x):
        return self._to_latent(x)[..., 0]

    def observables(self, x):
        z = self._to_latent(x)
        return {"energy": self.potential(x), "z1": z[..., 0], "z2": z[..., 1],
                "aux_r2": (z[..., 2:] ** 2).sum(-1)}

    def metadata(self):
        md = super().metadata()
        md.update({"eps": self.eps, "sigma_aux": self.sigma_aux, "mix": "orthogonal"})
        return md


class ManyWell(BasePotential):
    A, B, C = -0.5, -6.0, 1.0

    def __init__(self, n_blocks: int = 4, beta: float = 1.0, state_clip: float = 10.0):
        super().__init__(f"manywell_d{2*int(n_blocks)}", 2 * int(n_blocks), float(beta), float(state_clip))
        self.n_blocks = int(n_blocks)
        roots = np.roots([4 * self.C, 0.0, 2 * self.B, self.A])
        real = np.sort(roots[np.abs(roots.imag) < 1e-9].real)
        self.d_left, self.d_saddle, self.d_right = map(float, real)
        self.well_sep = self.d_right - self.d_left
        grid = np.linspace(-5, 5, 200001)
        logp = -self.beta * (self.A * grid + self.B * grid**2 + self.C * grid**4)
        logp -= logp.max(); p = np.exp(logp); dx = grid[1] - grid[0]
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[:-1] + p[1:]) * dx)]); cdf /= cdf[-1]
        self._grid = grid; self._cdf = cdf
        mask = grid > self.d_saddle
        self.p_deep = float(np.trapezoid((p / (p.sum() * dx))[mask], grid[mask]))

    def potential(self, x):
        d = x[..., 0::2]; v = x[..., 1::2]
        return (self.A * d + self.B * d**2 + self.C * d**4).sum(-1) + 0.5 * (v**2).sum(-1)

    def gradient(self, x):
        g = torch.empty_like(x)
        d = x[..., 0::2]; v = x[..., 1::2]
        g[..., 0::2] = self.A + 2 * self.B * d + 4 * self.C * d**3
        g[..., 1::2] = v
        return g

    def minima(self):
        pts = []
        for mask in range(2 ** min(self.n_blocks, 8)):
            x = torch.zeros(self.dim)
            for b in range(self.n_blocks):
                x[2*b] = self.d_right if ((mask >> b) & 1) else self.d_left
            pts.append(x)
            if len(pts) >= 2 ** min(self.n_blocks, 8):
                break
        return torch.stack(pts)

    def block_deep(self, x):
        return x[..., 0::2] > self.d_saddle

    def basin_labels(self, x):
        deep = self.block_deep(x).long()
        if self.n_blocks <= 20:
            pow2 = (2 ** torch.arange(self.n_blocks, device=x.device)).long()
            return (deep * pow2).sum(-1)
        return deep.sum(-1)

    def reference(self, n, seed, device=None):
        rng = np.random.default_rng(seed)
        u = rng.random((n, self.n_blocks))
        d = np.interp(u.ravel(), self._cdf, self._grid).reshape(n, self.n_blocks)
        v = rng.normal(size=(n, self.n_blocks)) / math.sqrt(self.beta)
        x = np.empty((n, self.dim), dtype=np.float32)
        x[:, 0::2] = d; x[:, 1::2] = v
        return torch.tensor(x, device=device, dtype=torch.float32)

    def initial_state(self, n, seed, device=None):
        g = torch.Generator(device=device); g.manual_seed(int(seed))
        x = torch.zeros(n, self.dim, device=device)
        x[..., 0::2] = self.d_left + 0.05 * torch.randn(n, self.n_blocks, generator=g, device=device)
        x[..., 1::2] = 0.1 * torch.randn(n, self.n_blocks, generator=g, device=device)
        return x

    def slow_cv(self, x):
        # number of deep blocks (0..n_blocks); bounded, fluctuates around n_blocks*p_deep -- no saturation
        return self.block_deep(x).to(x.dtype).sum(-1)

    def observables(self, x):
        deep = self.block_deep(x).float()
        return {"energy": self.potential(x), "deep_count": deep.sum(-1), "deep_frac": deep.mean(-1)}

    def metadata(self):
        md = super().metadata(); md.update({"n_blocks": self.n_blocks, "p_deep": self.p_deep, "well_sep": self.well_sep}); return md



class LennardJones2D(BasePotential):
    """N-atom Lennard-Jones cluster in 2D (sigma=eps=1), coordinates flattened to (2N,).

    Provides the physics primitives for the E9 obstruction study: energy, analytic gradient,
    center-of-mass removal, a rotation-invariant sorted-pair-distance descriptor for isomer
    labeling, and quench (steepest descent) to the inherent structure. The fixed-bank Levy-score
    jump is known to be incompatible with the cluster's continuous rotational symmetry; this class
    is used to demonstrate that obstruction rather than to claim an equilibrium efficiency win.
    """

    def __init__(self, n_atoms: int = 7, beta: float = 5.0, state_clip: float = 6.0,
                 spatial_dim: int = 2):
        super().__init__(f"lj{int(n_atoms)}_{int(spatial_dim)}d",
                         int(spatial_dim) * int(n_atoms), float(beta), float(state_clip))
        self.n_atoms = int(n_atoms)
        self.spatial_dim = int(spatial_dim)

    def _pos(self, x):
        return x.reshape(*x.shape[:-1], self.n_atoms, self.spatial_dim)

    def _pair(self, x):
        p = self._pos(x)
        diff = p[..., :, None, :] - p[..., None, :, :]
        r2 = (diff ** 2).sum(-1)
        return p, diff, r2

    def potential(self, x):
        _, _, r2 = self._pair(x)
        iu = torch.triu_indices(self.n_atoms, self.n_atoms, offset=1, device=x.device)
        r2u = r2[..., iu[0], iu[1]].clamp_min(1e-6)
        inv6 = r2u ** (-3)
        return (4.0 * (inv6 ** 2 - inv6)).sum(-1)

    def gradient(self, x):
        p, diff, r2 = self._pair(x)
        eye = torch.eye(self.n_atoms, device=x.device, dtype=torch.bool)
        r2 = r2.clone(); r2[..., eye] = 1.0
        inv2 = 1.0 / r2.clamp_min(1e-6)
        inv6 = inv2 ** 3
        # dV/dr2 * 2 factor folded: force magnitude coeff per pair
        coeff = 4.0 * (-12.0 * inv6 ** 2 + 6.0 * inv6) * inv2  # dV/dr_ij * (1/r_ij), per ordered pair
        coeff[..., eye] = 0.0
        g = (coeff[..., :, :, None] * diff).sum(-2)  # dV/dx_i = sum_j coeff_ij (p_i - p_j)
        return g.reshape(*x.shape[:-1], self.dim)

    def remove_com(self, x):
        p = self._pos(x)
        p = p - p.mean(-2, keepdim=True)
        return p.reshape(*x.shape[:-1], self.dim)

    def descriptor(self, x):
        """Rotation/translation/permutation-invariant sorted pairwise distances."""
        _, _, r2 = self._pair(x)
        iu = torch.triu_indices(self.n_atoms, self.n_atoms, offset=1, device=x.device)
        d = r2[..., iu[0], iu[1]].clamp_min(1e-12).sqrt()
        return torch.sort(d, dim=-1).values

    def quench(self, x, n_steps: int = 400, lr: float = 2e-3):
        """Steepest-descent quench to the inherent structure (for isomer assignment)."""
        z = self.remove_com(x).clone()
        for _ in range(int(n_steps)):
            g = self.gradient(z)
            gn = g.norm(dim=-1, keepdim=True).clamp_min(1.0)
            z = z - lr * g / gn  # normalized step for stiff LJ forces
            z = self.remove_com(z)
        return z

    def minima(self):
        raise NotImplementedError("LJ isomer catalogue is discovered by quenching, not analytic")

    def basin_labels(self, x):
        # requires an isomer catalogue; handled in the E9 analysis, not the generic pipeline
        return torch.zeros(x.shape[:-1], dtype=torch.long, device=x.device)


class AlanineTorus2D(BasePotential):
    """2D Ramachandran (phi, psi) torsion sampler on the torus [-pi, pi)^2.

    The target is a documented analytic surrogate for the alanine-dipeptide free-energy surface: a
    mixture of wrapped Gaussians at the metastable conformational basins (beta/C7eq, alpha_R, C7ax,
    alpha_L). We set V = -eps * log p_star(wrap(x)), so the Gibbs target is the mixture; the
    potential is periodic, distances use the minimum image, and jumps wrap around the torus. This
    is a surrogate (not an MD-derived FES) and is labeled as such in the report.
    """

    def __init__(self, eps: float = 0.4, state_clip: float = 1.0e6):
        super().__init__("alanine_torus", 2, 1.0 / float(eps), float(state_clip))
        self.eps = float(eps)
        deg = torch.tensor([[-150.0, 150.0], [-70.0, -35.0], [60.0, -60.0], [65.0, 40.0]])
        self.centers = deg * math.pi / 180.0
        self.scales = torch.tensor([0.45, 0.40, 0.45, 0.40])
        self.mix = torch.tensor([0.35, 0.40, 0.15, 0.10])
        self.domain = (-math.pi, math.pi, -math.pi, math.pi)
        self._probs_cache = None

    @staticmethod
    def _wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _log_components(self, x):
        m = self.centers.to(x.device, x.dtype)
        s = self.scales.to(x.device, x.dtype)
        w = self.mix.to(x.device, x.dtype)
        d = self._wrap(x[..., None, :] - m)
        r2 = (d ** 2).sum(-1)
        logN = -math.log(2 * math.pi) - 2 * torch.log(s) - 0.5 * r2 / (s ** 2)
        return torch.log(w) + logN

    def log_pstar(self, x):
        return torch.logsumexp(self._log_components(x), dim=-1)

    def potential(self, x):
        return -self.eps * self.log_pstar(self._wrap(x))

    def gradient(self, x):
        xw = self._wrap(x)
        m = self.centers.to(x.device, x.dtype)
        s = self.scales.to(x.device, x.dtype)
        d = self._wrap(xw[..., None, :] - m)
        r = torch.softmax(self._log_components(xw), dim=-1)
        dlog = (r[..., None] * (-d / (s[..., None] ** 2))).sum(-2)   # wrap slope 1 a.e.
        return -self.eps * dlog

    def minima(self):
        return self.centers.clone()

    def basin_labels(self, x):
        m = self.centers.to(x.device, x.dtype)
        d = self._wrap(self._wrap(x)[..., None, :] - m)
        return (d ** 2).sum(-1).argmin(-1)

    def target_basin_probs(self, device=None):
        if self._probs_cache is None:
            g = np.linspace(-math.pi, math.pi, 401)
            X, Y = np.meshgrid(g, g, indexing="xy")
            pts = torch.tensor(np.stack([X.ravel(), Y.ravel()], 1), dtype=torch.float32)
            logp = self.log_pstar(pts).numpy()
            p = np.exp(logp - logp.max()); p /= p.sum()
            lab = self.basin_labels(pts).numpy()
            probs = np.array([p[lab == k].sum() for k in range(self.centers.shape[0])])
            self._probs_cache = probs / probs.sum()
        return torch.tensor(self._probs_cache, device=device, dtype=torch.float32)

    def reference(self, n, seed, device=None):
        rng = np.random.default_rng(seed)
        comp = rng.choice(self.centers.shape[0], size=n, p=self.mix.numpy())
        c = self.centers.numpy()[comp]; s = self.scales.numpy()[comp]
        x = self._wrap(torch.tensor(c + rng.normal(0, 1, size=(n, 2)) * s[:, None], dtype=torch.float32))
        return x.to(device=device)

    def initial_state(self, n, seed, device=None):
        g = torch.Generator(device=device); g.manual_seed(int(seed))
        base = self.centers[1].to(device)   # start in alpha_R
        return self._wrap(base + 0.1 * torch.randn(n, 2, generator=g, device=device))

    def slow_cv(self, x):
        return torch.cos(self._wrap(x)[..., 0])   # bounded, periodic-safe

    def observables(self, x):
        xw = self._wrap(x)
        return {"energy": self.potential(x), "phi": xw[..., 0], "psi": xw[..., 1]}

    def metadata(self):
        md = super().metadata(); md.update({"eps": self.eps, "surrogate": True}); return md


def build_potential(cfg: dict) -> BasePotential:
    kind = cfg.get("kind", cfg.get("target", ""))
    target_cfg = dict(cfg.get("target_cfg", cfg))
    beta = float(target_cfg.get("beta", cfg.get("beta", 1.0)))
    if kind == "double_well":
        return DoubleWell1D(beta=beta, state_clip=target_cfg.get("state_clip", 4.0))
    if kind == "triple_well":
        return TripleWell1D(eps=target_cfg.get("eps", 0.08), state_clip=target_cfg.get("state_clip", 6.0))
    if kind == "muller10d":
        return TransformedMuellerBrown10D(
            eps=target_cfg.get("eps", 0.5), dim=target_cfg.get("dim", 10),
            sigma_aux=target_cfg.get("sigma_aux", 0.75), scale=target_cfg.get("scale", 0.02),
            mix_seed=target_cfg.get("mix_seed", 0), state_clip=target_cfg.get("state_clip", 8.0),
            grid_n=target_cfg.get("grid_n", 150))
    if kind == "four_well":
        return FourWell2D(beta=beta, state_clip=target_cfg.get("state_clip", 4.0))
    if kind == "muller_brown":
        return MuellerBrown2D(beta=beta, scale=target_cfg.get("scale", 0.05), state_clip=target_cfg.get("state_clip", 5.0), grid_n=target_cfg.get("grid_n", 120))
    if kind == "manywell":
        return ManyWell(n_blocks=target_cfg.get("n_blocks", 4), beta=beta, state_clip=target_cfg.get("state_clip", 10.0))
    if kind == "alanine_torus":
        return AlanineTorus2D(eps=target_cfg.get("eps", 0.4), state_clip=target_cfg.get("state_clip", 1.0e6))
    raise ValueError(f"unknown potential kind: {kind}")
