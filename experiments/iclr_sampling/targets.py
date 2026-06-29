"""Target distributions for the ICLR sampling suite.

Every target exposes the density  p(x) ∝ exp(-V(x))  (we fix sigma^2 = 2 so the
repo convention exp(-2V/sigma^2) reduces to exp(-V)).  Targets also provide the
ingredients the LSB-MC sampler needs:

* ``potential(x)`` / ``grad_potential(x)`` / ``force(x) = -grad_potential``
* ``sample_reference(n, seed, device)``  (exact or high-quality reference)
* ``init_particles(n_seeds, n, seed, device)``  (metastable initialisation)
* ``levy_score(x)``  (the Lévy-score drift correction S_L)
* ``apply_jumps(x, gen, dt)``  (the compound-Poisson jump operator)
* mode bookkeeping for the sampling-quality metrics.

The jump bank and its matching Lévy score are *target specific and principled*
(double-well separation for ManyWell, mixture geometry for MoG, label-switching
symmetry for the Bayesian GMM) — they are never arbitrary "bad jumps".
"""
from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .utils import gauss_legendre_01

SIGMA = math.sqrt(2.0)          # sigma^2 = 2  ->  p ∝ exp(-V)
SIGMA2 = 2.0


# --------------------------------------------------------------------------- #
# Generic finite-bank Lévy-score correction
# --------------------------------------------------------------------------- #
def finite_bank_levy_score(
    potential_fn,
    x: torch.Tensor,
    jumps: torch.Tensor,
    rates: torch.Tensor,
    theta_nodes: torch.Tensor,
    theta_weights: torch.Tensor,
    clamp: float = 60.0,
    chunk: int = 64,
    s_clip: float = 100.0,
) -> torch.Tensor:
    """S_L(x) = -∫_0^1 Σ_j λ_j r_j exp(-(V(x-θ r_j) - V(x))) dθ   (sigma^2 = 2).

    Computed by Gauss-Legendre quadrature over θ∈[0,1] and chunked over the jump
    bank to bound GPU memory.  ``jumps`` is (M, d), ``rates`` is (M,), ``x`` is
    (..., d).  Returns S_L with the same shape as ``x``.
    """
    Vx = potential_fn(x)                                  # (...,)
    S = torch.zeros_like(x)
    M = jumps.shape[0]
    for t in range(theta_nodes.shape[0]):
        theta = theta_nodes[t]
        tw = theta_weights[t]
        for j0 in range(0, M, chunk):
            jb = jumps[j0:j0 + chunk]                     # (m, d)
            rb = rates[j0:j0 + chunk]                     # (m,)
            xq = x.unsqueeze(-2) - theta * jb             # (..., m, d)
            Vq = potential_fn(xq)                         # (..., m)
            w = torch.exp((Vx.unsqueeze(-1) - Vq).clamp(-clamp, clamp))  # (..., m)
            # contribution  -tw * Σ_m (λ_m w_m) r_m
            S = S - tw * torch.einsum("...m,md->...d", w * rb, jb)
    return S.clamp(-s_clip, s_clip)


def apply_finite_bank_jumps(
    x: torch.Tensor,
    jumps: torch.Tensor,
    rates: torch.Tensor,
    gen: torch.Generator,
    dt: float,
) -> torch.Tensor:
    """Compound-Poisson jump operator for a finite additive jump bank.

    Each jump type j fires independently with probability λ_j dt per step and
    adds its displacement r_j.  Fully vectorised over the leading (particle/seed)
    dimensions and the bank.
    """
    lead = x.shape[:-1]
    M = jumps.shape[0]
    p = (rates * dt).clamp(0.0, 1.0)                       # (M,)
    u = torch.rand((*lead, M), generator=gen, device=x.device, dtype=x.dtype)
    fire = (u < p).to(x.dtype)                             # (..., M)
    delta = torch.einsum("...m,md->...d", fire, jumps)
    return x + delta


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class Target:
    name: str = "base"
    dim: int = 0
    sigma: float = SIGMA
    has_modes: bool = False
    n_modes: int = 0
    jump_bank_size: int = 0
    state_clip: float = 50.0
    high_dim: bool = False     # use sliced-W2 instead of exact OT in metrics

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def grad_potential(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def force(self, x: torch.Tensor) -> torch.Tensor:
        return -self.grad_potential(x)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -self.potential(x)

    def sample_reference(self, n: int, seed: int, device) -> Optional[torch.Tensor]:
        return None

    def init_particles(self, n_seeds: int, n: int, seed: int, device) -> torch.Tensor:
        raise NotImplementedError

    # --- LSB-MC hooks (default: generic finite bank) ----------------------- #
    def levy_score(self, x: torch.Tensor) -> torch.Tensor:
        return finite_bank_levy_score(
            self.potential, x, self._jumps, self._rates,
            self._theta_nodes, self._theta_weights, chunk=self._jump_chunk)

    def apply_jumps(self, x: torch.Tensor, gen: torch.Generator, dt: float) -> torch.Tensor:
        return apply_finite_bank_jumps(x, self._jumps, self._rates, gen, dt)

    # --- mode bookkeeping (discrete-mode targets) -------------------------- #
    def assign_modes(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def target_mode_weights(self, device) -> torch.Tensor:
        raise NotImplementedError

    def metadata(self) -> Dict:
        return {"name": self.name, "dim": self.dim, "sigma": self.sigma,
                "jump_bank_size": self.jump_bank_size, "n_modes": self.n_modes}


# --------------------------------------------------------------------------- #
# ManyWell:  d = 2 * n_blocks independent 2D blocks
# --------------------------------------------------------------------------- #
class ManyWellTarget(Target):
    """16-style asymmetric double-well blocks (preserves manywell.ipynb design).

    Each block has a double-well coordinate d (even index) with
    U_d(d) = a d + b d^2 + c d^4  and a standard-Gaussian coordinate v (odd).
    The jump acts only on the double-well coordinate with magnitude r~U[a,b]
    chosen to bracket the well separation (~3.463).  The Lévy score is the exact
    1-D correction precomputed on a grid, identical in spirit to the notebook.
    """

    A_COEF, B_COEF, C_COEF = -0.5, -6.0, 1.0

    def __init__(self, n_blocks: int = 16, device=None,
                 lambda_block: float = 0.1, jump_r_min: float = 3.2,
                 jump_r_max: float = 3.8, n_quad: int = 256,
                 grid_size: int = 20001, state_clip: float = 10.0):
        self.n_blocks = int(n_blocks)
        self.dim = 2 * self.n_blocks
        self.name = f"manywell_d{self.dim}"
        self.sigma = SIGMA
        self.lambda_block = lambda_block
        self.jump_r_min = jump_r_min
        self.jump_r_max = jump_r_max
        self.state_clip = state_clip
        self.high_dim = True
        self.jump_bank_size = 2 * self.n_blocks      # sign x block channels (continuous r)
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # --- stationary points of the 1D double well --------------------- #
        roots = np.roots([4 * self.C_COEF, 0.0, 2 * self.B_COEF, self.A_COEF])
        real = np.sort(roots[np.abs(roots.imag) < 1e-9].real)
        self.d_left, self.d_saddle, self.d_right = (float(real[0]), float(real[1]),
                                                    float(real[2]))
        self.well_sep = self.d_right - self.d_left

        # --- 1D reference inverse CDF and p_deep ------------------------- #
        dg = np.linspace(-5.0, 5.0, 200001)
        logp = -self._U1d_np(dg)
        logp -= logp.max()
        pd = np.exp(logp)
        Z = np.trapezoid(pd, dg)
        pd_n = pd / Z
        dx = dg[1] - dg[0]
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pd_n[:-1] + pd_n[1:]) * dx)])
        self._dg, self._cdf = dg, cdf
        mask_deep = dg > self.d_saddle
        self.p_deep = float(np.trapezoid(pd_n[mask_deep], dg[mask_deep]))

        # --- 1D Lévy score grid (float64 -> float32) --------------------- #
        self._grid, self._score = self._precompute_levy_grid(n_quad, grid_size)

    # --- potential / gradient ------------------------------------------- #
    def _U1d_np(self, d):
        return self.A_COEF * d + self.B_COEF * d ** 2 + self.C_COEF * d ** 4

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        d = x[..., 0::2]
        v = x[..., 1::2]
        Ud = self.A_COEF * d + self.B_COEF * d ** 2 + self.C_COEF * d ** 4
        return Ud.sum(-1) + 0.5 * (v ** 2).sum(-1)

    def grad_potential(self, x: torch.Tensor) -> torch.Tensor:
        d = x[..., 0::2]
        v = x[..., 1::2]
        g = torch.empty_like(x)
        g[..., 0::2] = self.A_COEF + 2 * self.B_COEF * d + 4 * self.C_COEF * d ** 3
        g[..., 1::2] = v
        return g

    # --- reference / init ------------------------------------------------ #
    def sample_reference(self, n: int, seed: int, device) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        nw = self.n_blocks
        u = rng.uniform(0.0, self._cdf[-1], size=(n, nw))
        d = np.interp(u.ravel(), self._cdf, self._dg).reshape(n, nw)
        v = rng.standard_normal((n, nw))
        x = np.empty((n, self.dim), dtype=np.float64)
        x[:, 0::2] = d
        x[:, 1::2] = v
        return torch.tensor(x, device=device, dtype=torch.float32)

    def init_particles(self, n_seeds, n, seed, device):
        g = torch.Generator(device=device); g.manual_seed(seed)
        x = torch.empty(n_seeds, n, self.dim, device=device, dtype=torch.float32)
        x[..., 0::2] = self.d_left + 0.05 * torch.randn(
            n_seeds, n, self.n_blocks, generator=g, device=device)
        x[..., 1::2] = 0.10 * torch.randn(
            n_seeds, n, self.n_blocks, generator=g, device=device)
        return x

    # --- Lévy score (1D grid, even coords only) -------------------------- #
    def _precompute_levy_grid(self, n_quad, grid_size):
        d = torch.linspace(-5.0, 5.0, grid_size, dtype=torch.float64)
        nodes, wts = np.polynomial.legendre.leggauss(n_quad)
        nodes = torch.tensor(nodes, dtype=torch.float64)
        wts = torch.tensor(wts, dtype=torch.float64)

        def panel(lo, hi):
            u = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
            wq = 0.5 * (hi - lo) * wts
            return u, wq

        u1, w1 = panel(0.0, self.jump_r_min)
        u2, w2 = panel(self.jump_r_min, self.jump_r_max)
        u = torch.cat([u1, u2])
        wq = torch.cat([w1, w2])
        Wu = torch.where(u <= self.jump_r_min,
                         torch.full_like(u, self.jump_r_max - self.jump_r_min),
                         self.jump_r_max - u)

        def U1d_t(dd):
            return self.A_COEF * dd + self.B_COEF * dd ** 2 + self.C_COEF * dd ** 4

        dc = d[:, None]; uu = u[None, :]
        Ud = U1d_t(dc)
        ep = torch.exp(torch.clamp(Ud - U1d_t(dc + uu), -60.0, 60.0))
        em = torch.exp(torch.clamp(Ud - U1d_t(dc - uu), -60.0, 60.0))
        integ = ((wq * Wu)[None, :] * (ep - em)).sum(dim=1)
        S = self.lambda_block / (2.0 * (self.jump_r_max - self.jump_r_min)) * integ
        S = S.clamp(-100.0, 100.0)
        return d.to(torch.float32).to(self.device), S.to(torch.float32).to(self.device)

    def _interp1d(self, q):
        grid, vals = self._grid, self._score
        G = grid.shape[0]
        q = q.contiguous()
        idx = torch.bucketize(q, grid).clamp(1, G - 1)
        x0, x1 = grid[idx - 1], grid[idx]
        y0, y1 = vals[idx - 1], vals[idx]
        w = (q - x0) / (x1 - x0)
        return y0 + w * (y1 - y0)

    def levy_score(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        out[..., 0::2] = self._interp1d(x[..., 0::2])
        return out

    def apply_jumps(self, x, gen, dt):
        d_shape = x[..., 0::2].shape
        dev = x.device
        jump = torch.rand(d_shape, generator=gen, device=dev) < self.lambda_block * dt
        r = self.jump_r_min + (self.jump_r_max - self.jump_r_min) * \
            torch.rand(d_shape, generator=gen, device=dev)
        sgn = torch.where(torch.rand(d_shape, generator=gen, device=dev) < 0.5, -1.0, 1.0)
        x = x.clone()
        x[..., 0::2] = x[..., 0::2] + jump.to(x.dtype) * sgn * r
        return x

    # --- ManyWell count-mode bookkeeping --------------------------------- #
    def block_deep(self, x: torch.Tensor) -> torch.Tensor:
        """Boolean (..., n_blocks): block is in the deep (right) well."""
        return x[..., 0::2] > self.d_saddle

    def deep_count(self, x: torch.Tensor) -> torch.Tensor:
        """Number of deep blocks per particle, shape (...)."""
        return self.block_deep(x).sum(-1)

    def metadata(self):
        md = super().metadata()
        md.update({"n_blocks": self.n_blocks, "p_deep": self.p_deep,
                   "well_sep": self.well_sep, "d_saddle": self.d_saddle,
                   "jump_r_min": self.jump_r_min, "jump_r_max": self.jump_r_max,
                   "lambda_block": self.lambda_block,
                   "jump_note": "continuous magnitude r~U[a,b] per block, 2 signs"})
        return md


# --------------------------------------------------------------------------- #
# Mixture of Gaussians in 2D
# --------------------------------------------------------------------------- #
class MoGTarget(Target):
    """K-component isotropic Gaussian mixture in 2D (uniform weights, unit var).

    Centres are drawn once from a fixed seed (reproducing the MoG40 design,
    extended to arbitrary K).  The LSB-MC jump bank is a *principled directed
    kNN graph of centre differences*  r_ij = mu_j - mu_i for j in kNN(i), which
    matches the target geometry; its Lévy score is the generic finite-bank
    correction.  Particles initialise near a single mode (mode-discovery task).
    """

    def __init__(self, n_modes: int = 40, device=None, center_seed: int = 0,
                 center_range: float = 40.0, comp_sigma: float = 1.0,
                 k_neighbors: int = 8, lam_total: float = 1.0,
                 n_theta: int = 8, jump_chunk: int = 64, state_clip: float = 80.0):
        self.n_modes = int(n_modes)
        self.dim = 2
        self.name = f"mog_K{self.n_modes}"
        self.sigma = SIGMA
        self.comp_sigma = comp_sigma
        self.state_clip = state_clip
        self.has_modes = True
        self.high_dim = False
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        rng = np.random.default_rng(center_seed)
        mu = rng.uniform(-center_range, center_range, size=(self.n_modes, 2))
        self.mu = torch.tensor(mu, device=device, dtype=torch.float32)

        # --- principled kNN centre-difference jump bank ------------------ #
        self._build_knn_bank(k_neighbors, lam_total)
        self._jump_chunk = jump_chunk
        tn, tw = gauss_legendre_01(n_theta, device, torch.float32)
        self._theta_nodes, self._theta_weights = tn, tw

    def _build_knn_bank(self, k_neighbors, lam_total):
        with torch.no_grad():
            d2 = torch.cdist(self.mu, self.mu) ** 2          # (K, K)
            k = min(k_neighbors, self.n_modes - 1)
            d2.fill_diagonal_(float("inf"))
            nn = torch.topk(d2, k, dim=1, largest=False).indices  # (K, k)
            src = torch.arange(self.n_modes, device=self.device).unsqueeze(1).expand(-1, k)
            jumps = self.mu[nn] - self.mu[src.reshape(-1)].reshape(self.n_modes, k, 2)
            jumps = jumps.reshape(-1, 2)                      # (K*k, 2)
        self._jumps = jumps
        M = jumps.shape[0]
        self.jump_bank_size = M
        # distribute a fixed total intensity over the bank
        self._rates = torch.full((M,), lam_total / max(M, 1), device=self.device,
                                 dtype=torch.float32)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        d = x.unsqueeze(-2) - self.mu                        # (..., K, 2)
        d2 = (d ** 2).sum(-1) / (self.comp_sigma ** 2)       # (..., K)
        return -torch.logsumexp(-0.5 * d2, dim=-1)

    def grad_potential(self, x: torch.Tensor) -> torch.Tensor:
        d = x.unsqueeze(-2) - self.mu                        # (..., K, 2)
        d2 = (d ** 2).sum(-1) / (self.comp_sigma ** 2)
        w = torch.softmax(-0.5 * d2, dim=-1)                 # responsibilities
        return torch.einsum("...k,...kd->...d", w, d) / (self.comp_sigma ** 2)

    def sample_reference(self, n, seed, device):
        rng = np.random.default_rng(seed)
        k = rng.integers(0, self.n_modes, size=n)
        mu = self.mu.cpu().numpy()
        x = mu[k] + self.comp_sigma * rng.standard_normal((n, 2))
        return torch.tensor(x, device=device, dtype=torch.float32)

    def init_particles(self, n_seeds, n, seed, device):
        g = torch.Generator(device=device); g.manual_seed(seed)
        x0 = self.mu[0] + 0.5 * torch.randn(n_seeds, n, 2, generator=g, device=device)
        return x0

    def assign_modes(self, x: torch.Tensor) -> torch.Tensor:
        d2 = torch.cdist(x.reshape(-1, 2), self.mu) ** 2
        return d2.argmin(dim=1).reshape(x.shape[:-1])

    def target_mode_weights(self, device):
        return torch.full((self.n_modes,), 1.0 / self.n_modes, device=device)

    def metadata(self):
        md = super().metadata()
        md.update({"center_range": float(self.mu.abs().max().item()),
                   "comp_sigma": self.comp_sigma,
                   "jump_note": "directed kNN centre-difference bank"})
        return md


# --------------------------------------------------------------------------- #
# Bayesian Gaussian-mixture posterior with label switching
# --------------------------------------------------------------------------- #
class BayesGMMTarget(Target):
    """Posterior over component means of a K-component Gaussian mixture.

    Data model:  y_i ~ (1/K) Σ_k N(mu_k, sigma_y^2 I),  prior mu_k ~ N(0, tau^2 I).
    Parameter theta = (mu_1, ..., mu_K) ∈ R^{K p}.  The posterior is invariant
    under the K! label permutations, giving K! symmetric modes.  The LSB-MC jump
    bank is the set of differences between permuted MAP modes — a principled,
    state-independent design that follows directly from the label-switching
    symmetry.  Reference samples are produced by symmetrising a long single-mode
    chain across all permutations (the posterior is exactly permutation-symmetric).
    """

    def __init__(self, K: int = 3, p: int = 2, n_data: int = 300,
                 sigma_y: float = 1.0, tau: float = 10.0, data_seed: int = 0,
                 true_radius: float = 6.0, device=None,
                 lam_total: float = 1.0, n_theta: int = 8, jump_chunk: int = 32,
                 state_clip: float = 40.0):
        self.K = int(K)
        self.p = int(p)
        self.dim = self.K * self.p
        self.n_data = int(n_data)
        self.sigma_y = sigma_y
        self.tau = tau
        self.name = f"bayes_gmm_K{self.K}_p{self.p}"
        self.sigma = SIGMA
        self.state_clip = state_clip
        self.has_modes = True
        self.high_dim = False
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # --- synthetic data from well-separated true means --------------- #
        rng = np.random.default_rng(data_seed)
        angles = np.linspace(0, 2 * np.pi, self.K, endpoint=False)
        true_mu = np.zeros((self.K, self.p))
        true_mu[:, 0] = true_radius * np.cos(angles)
        if self.p > 1:
            true_mu[:, 1] = true_radius * np.sin(angles)
        self.true_mu = true_mu
        labels = rng.integers(0, self.K, size=self.n_data)
        y = true_mu[labels] + sigma_y * rng.standard_normal((self.n_data, self.p))
        self.y = torch.tensor(y, device=device, dtype=torch.float32)        # (n_data, p)

        # --- permutations and (later) permuted MAP modes ----------------- #
        self.perms = list(itertools.permutations(range(self.K)))
        self.n_modes = len(self.perms)                        # K!

        # --- MAP estimate (KMeans init + gradient refinement) ------------ #
        self.map_theta = self._find_map(data_seed)            # (dim,)
        self.mode_centers = self._build_permuted_modes(self.map_theta)  # (K!, dim)

        # --- principled label-permutation jump bank ---------------------- #
        self._build_perm_bank(lam_total)
        self._jump_chunk = jump_chunk
        tn, tw = gauss_legendre_01(n_theta, device, torch.float32)
        self._theta_nodes, self._theta_weights = tn, tw

    # --- potential: negative log posterior ------------------------------- #
    def _theta_to_mu(self, theta: torch.Tensor) -> torch.Tensor:
        """(..., K*p) -> (..., K, p)."""
        return theta.reshape(*theta.shape[:-1], self.K, self.p)

    def potential(self, theta: torch.Tensor) -> torch.Tensor:
        mu = self._theta_to_mu(theta)                          # (..., K, p)
        # log-likelihood: Σ_i logsumexp_k [ -||y_i - mu_k||^2 / (2 sigma_y^2) ] - const
        # broadcast data over the leading sample dims.
        lead = mu.shape[:-2]
        y = self.y.reshape(*([1] * len(lead)), self.n_data, 1, self.p)  # (...,N,1,p)
        muE = mu.unsqueeze(-3)                                 # (..., 1, K, p)
        d2 = ((y - muE) ** 2).sum(-1) / (self.sigma_y ** 2)    # (..., N, K)
        ll = torch.logsumexp(-0.5 * d2, dim=-1).sum(-1)        # (...,)
        # prior:  Σ_k ||mu_k||^2 / (2 tau^2)
        prior = (mu ** 2).sum((-1, -2)) / (2.0 * self.tau ** 2)
        return -ll + prior

    def grad_potential(self, theta: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            t = theta.detach().requires_grad_(True)
            V = self.potential(t).sum()
            g, = torch.autograd.grad(V, t)
        return g.detach()

    def _find_map(self, seed) -> torch.Tensor:
        from sklearn.cluster import KMeans

        y_np = self.y.cpu().numpy()
        best_theta, best_V = None, float("inf")
        for init in range(4):
            km = KMeans(n_clusters=self.K, n_init=5, random_state=seed + init).fit(y_np)
            theta0 = torch.tensor(km.cluster_centers_.reshape(-1), device=self.device,
                                  dtype=torch.float32)
            theta = theta0.clone().requires_grad_(True)
            opt = torch.optim.Adam([theta], lr=0.05)
            for _ in range(800):
                opt.zero_grad()
                loss = self.potential(theta)
                loss.backward()
                opt.step()
            with torch.no_grad():
                V = float(self.potential(theta).item())
            if V < best_V:
                best_V, best_theta = V, theta.detach().clone()
        self.map_energy = best_V
        return best_theta

    def _build_permuted_modes(self, theta: torch.Tensor) -> torch.Tensor:
        mu = self._theta_to_mu(theta)                          # (K, p)
        modes = []
        for perm in self.perms:
            modes.append(mu[list(perm)].reshape(-1))
        return torch.stack(modes, dim=0)                       # (K!, dim)

    def _build_perm_bank(self, lam_total):
        modes = self.mode_centers                              # (M, dim)
        M = modes.shape[0]
        diffs = (modes.unsqueeze(0) - modes.unsqueeze(1)).reshape(-1, self.dim)  # (M*M, dim)
        # drop the zero (identity) differences
        nz = diffs.norm(dim=1) > 1e-6
        self._jumps = diffs[nz]
        nb = self._jumps.shape[0]
        self.jump_bank_size = nb
        self._rates = torch.full((nb,), lam_total / max(nb, 1), device=self.device,
                                 dtype=torch.float32)

    # --- reference: symmetrise a long single-mode chain ------------------ #
    def sample_reference(self, n, seed, device, n_burn: int = 4000,
                         n_collect: int = 8000, dt: float = 2e-3):
        """High-quality reference: run a long MALA chain confined to one mode,
        then replicate across all K! permutations (the posterior is exactly
        label-symmetric).  Returns ~n permutation-balanced samples."""
        g = torch.Generator(device=device); g.manual_seed(seed + 777)
        n_chains = max(64, n // (self.n_modes * 4))
        x = self.map_theta.to(device).unsqueeze(0).repeat(n_chains, 1)
        x = x + 0.05 * torch.randn(x.shape, generator=g, device=device)
        collected = []
        total = n_burn + n_collect
        thin = max(1, n_collect // max(1, (n // self.n_modes) // n_chains + 1))
        for step in range(total):
            x, _ = _mala_step_plain(self, x, g, dt)
            if step >= n_burn and (step - n_burn) % thin == 0:
                collected.append(x.clone())
        base = torch.cat(collected, dim=0)                     # single-mode cloud
        # symmetrise across permutations
        mu = self._theta_to_mu(base)                           # (S, K, p)
        out = []
        for perm in self.perms:
            out.append(mu[:, list(perm), :].reshape(base.shape[0], -1))
        ref = torch.cat(out, dim=0)
        idx = torch.randperm(ref.shape[0], generator=g, device=device)[:n]
        return ref[idx]

    def init_particles(self, n_seeds, n, seed, device):
        """Initialise all particles in ONE permutation mode (label-switching
        discovery task: a good sampler must populate all K! modes)."""
        g = torch.Generator(device=device); g.manual_seed(seed)
        base = self.mode_centers[0].to(device)
        x0 = base + 0.1 * torch.randn(n_seeds, n, self.dim, generator=g, device=device)
        return x0

    def assign_modes(self, x: torch.Tensor) -> torch.Tensor:
        d2 = torch.cdist(x.reshape(-1, self.dim), self.mode_centers) ** 2
        return d2.argmin(dim=1).reshape(x.shape[:-1])

    def target_mode_weights(self, device):
        return torch.full((self.n_modes,), 1.0 / self.n_modes, device=device)

    def metadata(self):
        md = super().metadata()
        md.update({"K": self.K, "p": self.p, "n_data": self.n_data,
                   "sigma_y": self.sigma_y, "tau": self.tau,
                   "map_energy": getattr(self, "map_energy", None),
                   "true_mu": self.true_mu.tolist(),
                   "jump_note": "differences between permuted MAP modes"})
        return md


# --------------------------------------------------------------------------- #
# Minimal MALA step used internally for the Bayes-GMM reference generator.
# (The full, instrumented samplers live in samplers.py / baselines.py.)
# --------------------------------------------------------------------------- #
def _mala_step_plain(target: Target, x: torch.Tensor, gen: torch.Generator, dt: float):
    f = target.force(x)
    xi = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    y = x + 0.5 * dt * f + math.sqrt(dt) * xi
    fy = target.force(y)
    lq_fwd = -((y - x - 0.5 * dt * f) ** 2).sum(-1) / (2 * dt)
    lq_rev = -((x - y - 0.5 * dt * fy) ** 2).sum(-1) / (2 * dt)
    log_a = (target.log_prob(y) + lq_rev) - (target.log_prob(x) + lq_fwd)
    u = torch.rand(log_a.shape, generator=gen, device=x.device, dtype=x.dtype)
    acc = torch.log(u.clamp_min(1e-30)) < log_a
    x_new = torch.where(acc.unsqueeze(-1), y, x)
    return x_new, acc.float().mean().item()


TARGET_REGISTRY = {
    "manywell": ManyWellTarget,
    "mog": MoGTarget,
    "bayes_gmm": BayesGMMTarget,
}
