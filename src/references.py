"""Reference samplers and basin partitions.

E1: inverse-CDF on a dense grid (tail mass < 1e-30 outside the box).
E2: exact i.i.d. draws from the mixture (see MoG40.sample_exact).
E3: grid inverse-CDF on the 2D MB latent marginal x exact Gaussian aux,
    pushed through z -> z B^T.
E4: harmonic (Laplace) mixture - a REFERENCE, not ground truth; a long PT
    chain is run as a cross-check.
"""
from __future__ import annotations

import hashlib
import math
import os
import warnings

import numpy as np
import torch

from .device import DEFAULT_DEVICE


# ---------------------------------------------------------------- E1 (1D)
class Grid1DInverseCDF:
    def __init__(self, log_density, lo: float, hi: float, n_grid: int = 200_001,
                 device=DEFAULT_DEVICE) -> None:
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

    def __init__(self, log_density, lo, hi, shape=(2400, 2400), device=DEFAULT_DEVICE) -> None:
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


class Latent2DGaussianReference:
    """Generic 10D reference: latent 2D marginal (grid inverse-CDF on a
    given log-density) x N(0, eps sigma_aux^2 I_8), pushed through
    z -> z B^T."""

    def __init__(self, pot, latent_log_density, lo2d, hi2d, beta: float,
                 shape=(2400, 2400)) -> None:
        self.pot = pot
        self.beta = beta
        self.grid = Grid2DSampler(latent_log_density, lo2d, hi2d,
                                  shape=shape, device=pot.B.device)
        self.aux_std = math.sqrt((1.0 / beta)) * pot.sigma_aux

    def sample(self, n: int, gen: torch.Generator) -> torch.Tensor:
        z2 = self.grid.sample(n, gen)
        aux = self.aux_std * torch.randn(n, 8, generator=gen,
                                         device=z2.device, dtype=torch.float64)
        return self.pot.from_latent(torch.cat([z2, aux], dim=1))


class MB10DReference(Latent2DGaussianReference):
    """Original E3 reference (kept for the unit tests / history)."""

    def __init__(self, pot, lo2d, hi2d, beta: float, shape=(2400, 2400)) -> None:
        from .potentials import muller_brown_2d
        super().__init__(pot, lambda z: -(beta / pot.s) * muller_brown_2d(z),
                         lo2d, hi2d, beta, shape)


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

    def snis_weighted_proposals(
            self, m: int, gen: torch.Generator, potential, beta: float
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Draw proposal points and return normalized SNIS weights.

        This is the statistically efficient interface for target expectations:
        callers should apply the weights directly rather than resampling.  The
        returned diagnostics describe proposal quality; they do not certify an
        exact or independent target sample.
        """
        if m < 1:
            raise ValueError("m must be positive")
        if beta <= 0:
            raise ValueError("beta must be positive")
        x = self.sample(m, gen)
        log_w = -beta * potential.V(x) - self.log_q(x)
        if not bool(torch.isfinite(log_w).all().item()):
            n_bad = int((~torch.isfinite(log_w)).sum().item())
            raise FloatingPointError(
                f"SNIS produced {n_bad}/{m} nonfinite log weights"
            )
        weights = torch.softmax(log_w, dim=0)
        ess = float((1.0 / torch.sum(weights.square())).item())
        entropy_ess = float(torch.exp(
            -torch.sum(weights * torch.log(weights.clamp_min(1e-300)))
        ).item())
        diagnostics = {
            "reference_method": "self_normalized_importance_sampling",
            "n_proposals": int(m),
            "proposal_ess": ess,
            "proposal_ess_fraction": ess / m,
            "entropy_ess": entropy_ess,
            "max_normalized_weight": float(weights.max().item()),
            "weight_cv2": max(m / ess - 1.0, 0.0),
            "nonfinite_log_weight_count": 0,
        }
        return x, weights, diagnostics

    @staticmethod
    def weighted_expectation(values: torch.Tensor,
                             weights: torch.Tensor) -> torch.Tensor:
        """Direct SNIS estimate along the first axis of ``values``."""
        if values.ndim == 0 or values.shape[0] != weights.numel():
            raise ValueError("values first dimension must match weights")
        if not bool(torch.isfinite(values).all().item()):
            raise FloatingPointError("SNIS observable contains nonfinite values")
        weights = weights.reshape(-1).to(device=values.device, dtype=values.dtype)
        if (not bool(torch.isfinite(weights).all().item())
                or bool((weights < 0).any().item())):
            raise ValueError("weights must be finite and nonnegative")
        total = weights.sum()
        if float(total.item()) <= 0.0:
            raise ValueError("weights must have a positive sum")
        weights = weights / total
        return torch.tensordot(weights, values, dims=([0], [0]))

    @staticmethod
    def weighted_category_probabilities(labels: torch.Tensor, n_categories: int,
                                        weights: torch.Tensor) -> torch.Tensor:
        """Direct SNIS category/basin probabilities without SIR resampling."""
        if n_categories < 1:
            raise ValueError("n_categories must be positive")
        labels = labels.reshape(-1).to(dtype=torch.long)
        weights = weights.reshape(-1).to(device=labels.device, dtype=torch.float64)
        if labels.numel() != weights.numel():
            raise ValueError("labels and weights must have the same length")
        if labels.numel() and bool(((labels < 0) | (labels >= n_categories)).any().item()):
            raise ValueError("labels must lie in [0, n_categories)")
        total = weights.sum()
        if (not bool(torch.isfinite(weights).all().item())
                or bool((weights < 0).any().item())
                or float(total.item()) <= 0.0):
            raise ValueError("weights must be finite, nonnegative, and have a positive sum")
        probabilities = torch.zeros(n_categories, dtype=weights.dtype,
                                    device=weights.device)
        probabilities.scatter_add_(0, labels, weights / total)
        return probabilities

    def snis_estimate(self, m: int, gen: torch.Generator, potential, beta: float,
                      observable) -> tuple[torch.Tensor, dict]:
        """Estimate ``E_pi[observable(X)]`` directly with normalized weights."""
        x, weights, diagnostics = self.snis_weighted_proposals(
            m, gen, potential, beta
        )
        values = observable(x)
        return self.weighted_expectation(values, weights), diagnostics

    def sample_sir(self, n: int, gen: torch.Generator, potential, beta: float,
                   oversample: int = 16, return_diagnostics: bool = False):
        """Approximate target sample by sampling-importance-resampling (SIR).

        For finite proposal count the result is neither exact nor i.i.d. from
        the target: resampled points share the same random weighted proposal
        pool.  Prefer :meth:`snis_estimate` for expectations and basin masses.
        """
        if n < 1:
            raise ValueError("n must be positive")
        if oversample < 1:
            raise ValueError("oversample must be positive")
        x, weights, diagnostics = self.snis_weighted_proposals(
            n * oversample, gen, potential, beta
        )
        idx = torch.multinomial(weights, n, replacement=True, generator=gen)
        sample = x[idx]
        diagnostics = dict(diagnostics)
        diagnostics.update({
            "reference_method": "sampling_importance_resampling",
            "n_resampled": int(n),
            "oversample": int(oversample),
            "unique_resample_fraction": float(torch.unique(idx).numel() / n),
        })
        return (sample, diagnostics) if return_diagnostics else sample

    def sample_exact_snis(self, n: int, gen: torch.Generator, potential,
                          beta: float, oversample: int = 16) -> torch.Tensor:
        """Deprecated compatibility alias for :meth:`sample_sir`.

        The historical name was incorrect: finite self-normalized importance
        resampling is an approximate SIR reference, not an exact target sampler.
        """
        warnings.warn(
            "sample_exact_snis is not exact; use sample_sir or direct "
            "snis_estimate instead", DeprecationWarning, stacklevel=2
        )
        return self.sample_sir(n, gen, potential, beta, oversample)

    def snis_ess_fraction(self, potential, beta: float, gen: torch.Generator,
                          m: int = 50_000) -> float:
        """Proposal-weight ESS fraction, not the ESS of an i.i.d. target sample."""
        _, _, diagnostics = self.snis_weighted_proposals(m, gen, potential, beta)
        return float(diagnostics["proposal_ess_fraction"])


# ---------------------------------------------------------------- basins
class GradientFlowBasinMap2D:
    """Validated cached basin-of-attraction map for a 2D potential.

    Cache reuse is permitted only when the saved grid, domain, minima, and
    flow integrator metadata exactly match the requested construction. Legacy
    label-only caches are rejected by default; an explicit analysis-only opt-in
    can load them with ``cache_validation_status='legacy_unverified'``.
    """

    _CACHE_SCHEMA_VERSION = 1
    _CACHE_METADATA_KEYS = (
        "n_grid", "lo", "hi", "minima", "dt_flow", "n_flow",
    )

    def __init__(self, grad_fn, minima: torch.Tensor, lo, hi,
                 n_grid: int = 600, device=DEFAULT_DEVICE, cache: str | None = None,
                 dt_flow: float = 1.5e-4, n_flow: int = 40_000,
                 *, allow_legacy_unverified: bool = False) -> None:
        # dt_flow is set by the stiffest Hessian eigenvalue among the 2D
        # potentials used here (Mueller-Brown ~6e3: dt*lam ~ 0.9 < 2), and
        # the tamed step caps wall gradients at unit displacement.
        if isinstance(n_grid, bool) or int(n_grid) != n_grid or n_grid < 2:
            raise ValueError("n_grid must be an integer >= 2")
        if isinstance(n_flow, bool) or int(n_flow) != n_flow or n_flow < 1:
            raise ValueError("n_flow must be a positive integer")
        if not math.isfinite(float(dt_flow)) or float(dt_flow) <= 0.0:
            raise ValueError("dt_flow must be finite and positive")
        self.lo = torch.as_tensor(lo, dtype=torch.float64, device=device)
        self.hi = torch.as_tensor(hi, dtype=torch.float64, device=device)
        self.minima = torch.as_tensor(minima, dtype=torch.float64, device=device)
        if (self.lo.shape != (2,) or self.hi.shape != (2,)
                or self.minima.ndim != 2 or self.minima.shape[1] != 2
                or self.minima.shape[0] < 1):
            raise ValueError("lo/hi must be 2-vectors and minima must have shape (K, 2)")
        if (not bool(torch.isfinite(self.lo).all().item())
                or not bool(torch.isfinite(self.hi).all().item())
                or not bool(torch.isfinite(self.minima).all().item())
                or not bool((self.hi > self.lo).all().item())):
            raise ValueError("basin domain/minima must be finite with hi > lo")
        self.n_grid = int(n_grid)
        self.dt_flow = float(dt_flow)
        self.n_flow = int(n_flow)
        self.cache_path = os.path.abspath(cache) if cache is not None else None
        self.cache_sha256: str | None = None
        self.cache_validation_status = "not_requested"
        expected = {
            "n_grid": np.asarray(self.n_grid, dtype=np.int64),
            "lo": self.lo.detach().cpu().numpy(),
            "hi": self.hi.detach().cpu().numpy(),
            "minima": self.minima.detach().cpu().numpy(),
            "dt_flow": np.asarray(self.dt_flow, dtype=np.float64),
            "n_flow": np.asarray(self.n_flow, dtype=np.int64),
        }

        if self.cache_path is not None and os.path.exists(self.cache_path):
            with np.load(self.cache_path, allow_pickle=False) as data:
                if "labels" not in data.files:
                    raise ValueError(f"basin cache {self.cache_path} has no labels array")
                labels = np.asarray(data["labels"])
                self._validate_cached_labels(labels)
                missing = [key for key in self._CACHE_METADATA_KEYS
                           if key not in data.files]
                if "cache_schema_version" not in data.files:
                    missing.append("cache_schema_version")
                if missing:
                    if not allow_legacy_unverified:
                        raise ValueError(
                            f"legacy/incomplete basin cache {self.cache_path} is missing "
                            f"metadata {sorted(missing)}; refusing unverified reuse")
                    warnings.warn(
                        f"explicitly loading legacy basin cache {self.cache_path} "
                        "without construction metadata; results are unverified",
                        RuntimeWarning, stacklevel=2)
                    self.cache_validation_status = "legacy_unverified"
                else:
                    schema = int(np.asarray(data["cache_schema_version"]).item())
                    mismatches = []
                    if schema != self._CACHE_SCHEMA_VERSION:
                        mismatches.append(
                            f"cache_schema_version={schema} (expected {self._CACHE_SCHEMA_VERSION})")
                    for key, expected_value in expected.items():
                        actual = np.asarray(data[key])
                        if not np.array_equal(actual, expected_value):
                            mismatches.append(key)
                    if mismatches:
                        raise ValueError(
                            f"basin cache metadata mismatch for {self.cache_path}: "
                            + ", ".join(mismatches))
                    self.cache_validation_status = "validated"
            self.labels = torch.as_tensor(labels, dtype=torch.long, device=device)
            self.cache_sha256 = self._sha256(self.cache_path)
            return

        xs = torch.linspace(float(self.lo[0]), float(self.hi[0]), self.n_grid,
                            dtype=torch.float64, device=device)
        ys = torch.linspace(float(self.lo[1]), float(self.hi[1]), self.n_grid,
                            dtype=torch.float64, device=device)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        z = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
        for _ in range(self.n_flow):
            g = grad_fn(z)
            gn = g.norm(dim=1, keepdim=True)
            z = z - self.dt_flow * g / (1.0 + self.dt_flow * gn)
            z = torch.clamp(z, self.lo, self.hi)
        d2 = ((z.unsqueeze(1) - self.minima.unsqueeze(0)) ** 2).sum(-1)
        self.labels = d2.argmin(dim=1).reshape(self.n_grid, self.n_grid)
        if self.cache_path is not None:
            np.savez(
                self.cache_path,
                labels=self.labels.detach().cpu().numpy(),
                cache_schema_version=np.asarray(
                    self._CACHE_SCHEMA_VERSION, dtype=np.int64),
                **expected,
            )
            self.cache_validation_status = "created_validated"
            self.cache_sha256 = self._sha256(self.cache_path)

    @staticmethod
    def _sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _validate_cached_labels(self, labels: np.ndarray) -> None:
        expected_shape = (self.n_grid, self.n_grid)
        if labels.shape != expected_shape:
            raise ValueError(
                f"basin cache labels shape {labels.shape} != {expected_shape}")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("basin cache labels must have an integer dtype")
        if labels.size and (labels.min() < 0 or labels.max() >= self.minima.shape[0]):
            raise ValueError("basin cache labels lie outside the declared minima")

    def cache_provenance(self) -> dict:
        """JSON-safe validation status and construction metadata."""
        return {
            "path": self.cache_path,
            "sha256": self.cache_sha256,
            "validation_status": self.cache_validation_status,
            "cache_schema_version": self._CACHE_SCHEMA_VERSION,
            "n_grid": self.n_grid,
            "lo": self.lo.detach().cpu().tolist(),
            "hi": self.hi.detach().cpu().tolist(),
            "minima": self.minima.detach().cpu().tolist(),
            "dt_flow": self.dt_flow,
            "n_flow": self.n_flow,
        }

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
