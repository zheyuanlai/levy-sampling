"""Potential energy surfaces for E1-E4, as pure functions of position.

Nothing in this module counts anything. Oracle accounting lives entirely in
``src.targets``, which wraps these surfaces and records what a sampler actually
asked for. Keeping the physics uncounted means a caching change in a sampler
shows up in the counters by itself instead of needing a matching edit to a cost
formula.

Conventions
-----------
* ``x`` has shape ``(..., d)``; ``V`` returns ``(...,)``; ``grad_V`` returns
  ``(..., d)``.
* Everything is ``float64``.
* ``value_delta(x, R)`` returns ``V(x - R) - V(x)`` for shifts ``R`` of shape
  ``(J, d)`` and ``x`` of shape ``(N, d)``, giving ``(N, J)``. The coupled chain
  overrides it with an exact O(N_s) moment identity for homogeneous shifts.
"""
from __future__ import annotations

import math

import numpy as np
import torch


class Potential:
    """A potential energy surface. Subclasses implement ``V`` and ``grad_V``."""

    d: int = 1
    name: str = "potential"
    #: True when ``value_delta`` uses a closed-form structured kernel rather
    #: than generic evaluations of ``V`` at shifted configurations. FEE charges
    #: such a kernel its measured equivalent cost, never a pretend count of
    #: generic potential calls.
    structured_value_delta: bool = False

    def V(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def grad_V(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def value_delta(self, x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """``V(x - R) - V(x)``; ``x`` is ``(N, d)`` and ``R`` is ``(J, d)``."""
        return self.V(x.unsqueeze(1) - R.unsqueeze(0)) - self.V(x).unsqueeze(1)

    def value_delta_pointwise(self, x: torch.Tensor,
                              y: torch.Tensor) -> torch.Tensor:
        """``V(y) - V(x)`` for particle-specific chord points.

        ``y`` has shape ``(N, ..., d)`` with per-particle shifts, which the
        broadcasting form above cannot express.
        """
        base = self.V(x)
        shifted = self.V(y)
        return shifted - base.reshape((x.shape[0],) + (1,) * (shifted.ndim - 1))

    def n_chord_units(self, n_particles: int, n_chords: int) -> int:
        """Configurations charged for a chord batch of this shape."""
        return int(n_particles) * int(n_chords)


# ======================================================================== E1
class DoubleWell1D(Potential):
    """``V(x) = (x^2 - 1)^2``: minima at +-1, saddle at 0, barrier height 1."""

    d = 1
    name = "double_well_1d"

    def V(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0]
        return (x1 * x1 - 1.0) ** 2

    def grad_V(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0]
        return (4.0 * x1 * (x1 * x1 - 1.0)).unsqueeze(-1)

    @staticmethod
    def kramers_time(beta: float = 8.0) -> float:
        return 2.0 * math.pi / math.sqrt(8.0 * 4.0) * math.exp(beta * 1.0)


# ======================================================================== E2
class MoG40(Potential):
    """``V(x) = -(1/beta) log sum_k exp(-||x - mu_k||^2 / 2)`` on R^2.

    The ``1/beta`` prefactor makes ``pi ~ exp(-beta V)`` an exactly equal-weight
    mixture of ``N(mu_k, I_2)`` for every beta; beta only rescales barriers.
    """

    d = 2
    name = "mog40"

    def __init__(self, beta: float = 8.0, n_components: int = 40,
                 center_range: tuple[float, float] = (-40.0, 40.0),
                 center_seed: int = 0, device=None) -> None:
        self.beta = float(beta)
        self.n_components = int(n_components)
        rng = np.random.default_rng(int(center_seed))
        centers = rng.uniform(center_range[0], center_range[1],
                              size=(self.n_components, 2))
        self.mu = torch.as_tensor(centers, dtype=torch.float64, device=device)

    def _sq_dists(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(-2) - self.mu
        return (diff * diff).sum(-1)

    def component_log_density(self, x: torch.Tensor) -> torch.Tensor:
        """``log N(x; mu_k, I_2)`` for every component, shape ``(..., K)``."""
        return -0.5 * self._sq_dists(x) - math.log(2.0 * math.pi)

    def V(self, x: torch.Tensor) -> torch.Tensor:
        return -(1.0 / self.beta) * torch.logsumexp(
            -0.5 * self._sq_dists(x), dim=-1)

    def grad_V(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(-2) - self.mu
        weights = torch.softmax(-0.5 * (diff * diff).sum(-1), dim=-1)
        return (1.0 / self.beta) * (weights.unsqueeze(-1) * diff).sum(-2)

    def sample_exact(self, n: int, generator: torch.Generator) -> torch.Tensor:
        """Exact i.i.d. draws from the equal-weight mixture."""
        device = self.mu.device
        k = torch.randint(0, self.n_components, (n,), generator=generator,
                          device=device)
        z = torch.randn(n, 2, generator=generator, device=device,
                        dtype=torch.float64)
        return self.mu[k] + z


# ======================================================================== E3
# Standard Muller-Brown functional form with the depth parameters retuned so the
# three deep wells are equal (V = -0.7957). The stock surface is multimodal OR
# metastable but never both; equalising the depths makes temperature a free dial.
# At beta = 24 the target is trimodal AND metastable on two timescales:
# beta*b(A<->B) = 11.1 (slow) and beta*b(B<->C) = 4.0 (moderate).
# Tuple layout: (D_k, a_k, b_k, c_k, x_k, y_k).
_MB3 = (
    (-1.6607, -1.0, 0.0, -10.0, 1.0, 0.0),
    (-1.0, -1.0, 0.0, -10.0, 0.0, 0.5),
    (-1.0218, -6.5, 11.0, -6.5, -0.5, 1.5),
    (0.15, 0.7, 0.6, 0.7, -1.0, 1.0),
)

#: Newton-refinement seeds; the reference builder refines and asserts these.
#: A = top-left, B = middle hub, C = right (the initial well).
MB3_CRITICAL = {
    "A": ((-0.5870, 1.4130), -0.7957),
    "B": ((-0.0650, 0.4750), -0.7957),
    "C": ((0.5740, 0.0390), -0.7957),
    "S_AB": ((-0.9160, 0.6660), -0.3323),
    "S_BC": ((0.2660, 0.2470), -0.6310),
}


def muller_brown_3well(z: torch.Tensor) -> torch.Tensor:
    """The retuned Muller-Brown surface on the latent pair ``(..., 2)``."""
    z1, z2 = z[..., 0], z[..., 1]
    out = torch.zeros_like(z1)
    for depth, a, b, c, x0, y0 in _MB3:
        dx, dy = z1 - x0, z2 - y0
        out = out + depth * torch.exp(a * dx * dx + b * dx * dy + c * dy * dy)
    return out


def muller_brown_3well_grad(z: torch.Tensor) -> torch.Tensor:
    z1, z2 = z[..., 0], z[..., 1]
    g1 = torch.zeros_like(z1)
    g2 = torch.zeros_like(z1)
    for depth, a, b, c, x0, y0 in _MB3:
        dx, dy = z1 - x0, z2 - y0
        e = depth * torch.exp(a * dx * dx + b * dx * dy + c * dy * dy)
        g1 = g1 + e * (2.0 * a * dx + b * dy)
        g2 = g2 + e * (b * dx + 2.0 * c * dy)
    return torch.stack([g1, g2], dim=-1)


class MullerBrown3Well10D(Potential):
    """``U(z) = V_MB(z1,z2) + ||z_{3:10}||^2 / (2 sigma_aux^2)``, sampled in
    ``x = z B^T`` with ``B = Q diag(s)``.

    The collective variable is the LATENT pair ``z_{1:2} = (x B^{-T})_{1:2}``,
    never the first two sampling coordinates.
    """

    d = 10
    name = "muller_brown_3well_10d"

    def __init__(self, sigma_aux: float = 0.4, embedding_seed: int = 12345,
                 singular_values: tuple[float, float] = (0.75, 1.45),
                 device=None) -> None:
        self.sigma_aux = float(sigma_aux)
        rng = np.random.default_rng(int(embedding_seed))
        Q, _ = np.linalg.qr(rng.standard_normal((self.d, self.d)))
        B = Q @ np.diag(np.linspace(singular_values[0], singular_values[1],
                                    self.d))
        self.B = torch.as_tensor(B, dtype=torch.float64, device=device)
        self.B_inv = torch.as_tensor(np.linalg.inv(B), dtype=torch.float64,
                                     device=device)

    def to_latent(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.B_inv.T

    def from_latent(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.B.T

    def collective_variable(self, x: torch.Tensor) -> torch.Tensor:
        """The two-dimensional CV ``z_{1:2}``."""
        return self.to_latent(x)[..., :2]

    def V(self, x: torch.Tensor) -> torch.Tensor:
        z = self.to_latent(x)
        aux = z[..., 2:]
        return (muller_brown_3well(z[..., :2])
                + 0.5 * (aux * aux).sum(-1) / self.sigma_aux ** 2)

    def grad_V(self, x: torch.Tensor) -> torch.Tensor:
        z = self.to_latent(x)
        gz = torch.zeros_like(z)
        gz[..., :2] = muller_brown_3well_grad(z[..., :2])
        gz[..., 2:] = z[..., 2:] / self.sigma_aux ** 2
        return gz @ self.B_inv


class MullerBrown3WellLatent2D(Potential):
    """Reduced latent surface used by the E3 score certificate.

    Jumps and test functions act on ``z_{1:2}`` only, dot products are
    affine-invariant, and the auxiliary Gaussian factorises out exactly, so the
    10D stationarity residual equals this 2D one.
    """

    d = 2
    name = "muller_brown_3well_latent2d"

    def V(self, z: torch.Tensor) -> torch.Tensor:
        return muller_brown_3well(z)

    def grad_V(self, z: torch.Tensor) -> torch.Tensor:
        return muller_brown_3well_grad(z)


# ======================================================================== E4
#: Site-potential tilt terms. Chosen so beta*dW_max = 0.44 across phases, inside
#: the regime where the tamed fixed-step integrator realises the correction's
#: detailed-balance return flux, while the phases stay distinguishably
#: non-uniform and the coherent barriers stay at beta*b ~ 7.8-8.2.
QUARTIC_CHAIN_COEFFICIENTS = {"cxy": -0.0125, "hx": 0.0075, "hy": 0.015}

#: Newton-refinement seeds for the four coherent minima of the site potential.
QUARTIC_CHAIN_MINIMA = {
    "--": ((-1.0025, -1.0034), -0.0351),
    "-+": ((-0.9994, 0.9965), 0.0200),
    "+-": ((0.9975, -1.0003), 0.0050),
    "++": ((1.0006, 0.9997), 0.0100),
}
PHASES = ("--", "-+", "+-", "++")


def site_potential(v: torch.Tensor, coefficients=None) -> torch.Tensor:
    """Two-component quartic site potential ``W`` on ``(..., 2)``."""
    c = QUARTIC_CHAIN_COEFFICIENTS if coefficients is None else coefficients
    x, y = v[..., 0], v[..., 1]
    return ((x * x - 1.0) ** 2 + (y * y - 1.0) ** 2
            + c["cxy"] * x * y + c["hx"] * x + c["hy"] * y)


def site_potential_grad(v: torch.Tensor, coefficients=None) -> torch.Tensor:
    c = QUARTIC_CHAIN_COEFFICIENTS if coefficients is None else coefficients
    x, y = v[..., 0], v[..., 1]
    gx = 4.0 * x * (x * x - 1.0) + c["cxy"] * y + c["hx"]
    gy = 4.0 * y * (y * y - 1.0) + c["cxy"] * x + c["hy"]
    return torch.stack([gx, gy], dim=-1)


class CoupledQuarticChain(Potential):
    """One-dimensional two-component coupled quartic chain, ``N_s`` sites in R^2.

    ``V(q) = kappa/(2 delta) sum_i ||q_{i+1} - q_i||^2 + delta sum_i W(q_i)``
    with periodic indices and ``delta = 1/N_s``.

    This is a two-component coupled quartic chain, not a scalar phi^4 model; the
    order parameter ``m = (1/N_s) sum_i q_i`` is a two-component magnetization.

    For a homogeneous shift ``r = 1_{N_s} (x) dvec`` the gradient energy is
    exactly invariant, so ``V(q - r) - V(q)`` is a fixed polynomial in ``dvec``
    whose coefficients are per-particle moments computed once in O(N_s). That
    structured kernel is charged its measured equivalent cost in FEE.
    """

    name = "coupled_quartic_chain"
    structured_value_delta = True

    def __init__(self, n_sites: int = 12, kappa: float = 2.5,
                 coefficients=None) -> None:
        self.n_sites = int(n_sites)
        self.d = 2 * self.n_sites
        self.kappa = float(kappa)
        self.coefficients = dict(
            QUARTIC_CHAIN_COEFFICIENTS if coefficients is None else coefficients)
        self.delta = 1.0 / self.n_sites

    def sites(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-1], self.n_sites, 2)

    def order_parameter(self, x: torch.Tensor) -> torch.Tensor:
        """Two-component magnetization ``m = (1/N_s) sum_i q_i``."""
        return self.sites(x).mean(dim=-2)

    def V(self, x: torch.Tensor) -> torch.Tensor:
        q = self.sites(x)
        dq = torch.roll(q, shifts=-1, dims=-2) - q
        gradient_energy = (self.kappa / (2.0 * self.delta)) * (dq * dq).sum((-1, -2))
        site_energy = self.delta * site_potential(q, self.coefficients).sum(-1)
        return gradient_energy + site_energy

    def grad_V(self, x: torch.Tensor) -> torch.Tensor:
        q = self.sites(x)
        laplacian = 2.0 * q - torch.roll(q, 1, dims=-2) - torch.roll(q, -1, dims=-2)
        g = ((self.kappa / self.delta) * laplacian
             + self.delta * site_potential_grad(q, self.coefficients))
        return g.reshape(*x.shape)

    # -- static observables ------------------------------------------------
    def energy_per_site(self, x: torch.Tensor) -> torch.Tensor:
        return self.V(x) / self.n_sites

    def coherence(self, x: torch.Tensor) -> torch.Tensor:
        """``G(q) = (1/N_s) sum_i ||q_{i+1} - q_i||^2`` with periodic indices."""
        q = self.sites(x)
        dq = torch.roll(q, shifts=-1, dims=-2) - q
        return (dq * dq).sum(-1).mean(-1)

    def two_point_correlation(self, x: torch.Tensor) -> torch.Tensor:
        """``C(r) = (1/N_s) sum_i <q_i . q_{i+r}>`` for r = 0..floor(N_s/2)."""
        q = self.sites(x)
        max_lag = self.n_sites // 2
        return torch.stack(
            [(q * torch.roll(q, shifts=-r, dims=-2)).sum(-1).mean(-1)
             for r in range(max_lag + 1)], dim=-1)

    def site_phase_labels(self, x: torch.Tensor,
                          minima: torch.Tensor) -> torch.Tensor:
        """Nearest refined coherent minimum for every site, shape ``(N, N_s)``."""
        q = self.sites(x)
        distances = (q.unsqueeze(-2) - minima).norm(dim=-1)
        return distances.argmin(dim=-1)

    def kink_density(self, x: torch.Tensor,
                     minima: torch.Tensor) -> torch.Tensor:
        """Fraction of periodic neighbour pairs whose site phase labels differ."""
        labels = self.site_phase_labels(x, minima)
        return (labels != torch.roll(labels, shifts=-1, dims=-1)).to(
            x.dtype).mean(-1)

    # -- structured homogeneous chord kernel -------------------------------
    def moments(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        q = self.sites(x)
        xs, ys = q[..., 0], q[..., 1]
        return {
            "x1": xs.sum(-1), "x2": (xs * xs).sum(-1), "x3": (xs ** 3).sum(-1),
            "y1": ys.sum(-1), "y2": (ys * ys).sum(-1), "y3": (ys ** 3).sum(-1),
        }

    def value_delta_homogeneous(self, x: torch.Tensor,
                                D: torch.Tensor) -> torch.Tensor:
        """``V(x - 1 (x) d) - V(x)`` for per-site shifts ``D`` of shape ``(J, 2)``.

        Exact, and O(N_s + J) per particle instead of O(N_s J).
        """
        m = self.moments(x)
        dx, dy = D[:, 0].unsqueeze(0), D[:, 1].unsqueeze(0)
        n_sites = float(self.n_sites)
        x1, x2, x3 = m["x1"].unsqueeze(1), m["x2"].unsqueeze(1), m["x3"].unsqueeze(1)
        y1, y2, y3 = m["y1"].unsqueeze(1), m["y2"].unsqueeze(1), m["y3"].unsqueeze(1)
        c = self.coefficients
        # per-axis quartic: ((u-d)^2-1)^2 - (u^2-1)^2
        #   = -4 d u^3 + 6 d^2 u^2 + (4 d - 4 d^3) u + d^4 - 2 d^2
        quartic = (-4.0 * dx * x3 + 6.0 * dx * dx * x2
                   + (4.0 * dx - 4.0 * dx ** 3) * x1
                   + n_sites * (dx ** 4 - 2.0 * dx * dx)
                   - 4.0 * dy * y3 + 6.0 * dy * dy * y2
                   + (4.0 * dy - 4.0 * dy ** 3) * y1
                   + n_sites * (dy ** 4 - 2.0 * dy * dy))
        # bilinear cross term: c_xy[(x-dx)(y-dy) - xy] = c_xy(-x dy - y dx + dx dy)
        cross = c["cxy"] * (-(x1 * dy) - (y1 * dx) + n_sites * dx * dy)
        linear = -(c["hx"] * dx + c["hy"] * dy) * n_sites
        return self.delta * (quartic + cross + linear)

    def _homogeneous_site_shifts(self, R: torch.Tensor) -> torch.Tensor:
        shaped = R.reshape(R.shape[0], self.n_sites, 2)
        if not torch.allclose(shaped, shaped[:, :1, :].expand_as(shaped)):
            raise ValueError(
                "the coupled-chain structured chord kernel requires homogeneous "
                "shifts r = 1_{N_s} (x) d")
        return shaped[:, 0, :]

    def value_delta(self, x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        return self.value_delta_homogeneous(x, self._homogeneous_site_shifts(R))

    def value_delta_pointwise(self, x: torch.Tensor,
                              y: torch.Tensor) -> torch.Tensor:
        """Structured form for per-particle chord points ``y = x - theta R``.

        The displacement is recovered as ``x - y`` and must be homogeneous.
        """
        displacement = x.reshape(x.shape[0], *([1] * (y.ndim - 2)), self.d) - y
        flat = displacement.reshape(-1, self.d)
        shaped = flat.reshape(-1, self.n_sites, 2)
        if not torch.allclose(shaped, shaped[:, :1, :].expand_as(shaped)):
            raise ValueError(
                "the coupled-chain structured chord kernel requires homogeneous "
                "shifts r = 1_{N_s} (x) d")
        site_shift = shaped[:, 0, :].reshape(*displacement.shape[:-1], 2)
        m = self.moments(x)
        n_sites = float(self.n_sites)
        c = self.coefficients
        lead = (x.shape[0],) + (1,) * (site_shift.ndim - 2)
        x1, x2, x3 = (m["x1"].reshape(lead), m["x2"].reshape(lead),
                      m["x3"].reshape(lead))
        y1, y2, y3 = (m["y1"].reshape(lead), m["y2"].reshape(lead),
                      m["y3"].reshape(lead))
        dx, dy = site_shift[..., 0], site_shift[..., 1]
        quartic = (-4.0 * dx * x3 + 6.0 * dx * dx * x2
                   + (4.0 * dx - 4.0 * dx ** 3) * x1
                   + n_sites * (dx ** 4 - 2.0 * dx * dx)
                   - 4.0 * dy * y3 + 6.0 * dy * dy * y2
                   + (4.0 * dy - 4.0 * dy ** 3) * y1
                   + n_sites * (dy ** 4 - 2.0 * dy * dy))
        cross = c["cxy"] * (-(x1 * dy) - (y1 * dx) + n_sites * dx * dy)
        linear = -(c["hx"] * dx + c["hy"] * dy) * n_sites
        return self.delta * (quartic + cross + linear)

    def kink_energy(self) -> float:
        return (4.0 / 3.0) * math.sqrt(2.0 * self.kappa)


# =================================================================== utilities
def newton_refine(grad_fn, z0: torch.Tensor, n_iter: int = 60) -> torch.Tensor:
    """Newton on ``grad_fn = 0`` with a finite-difference Jacobian (fp64)."""
    z = z0.clone()
    dim = z.numel()
    h = 1e-6
    for _ in range(n_iter):
        g = grad_fn(z.unsqueeze(0))[0]
        jacobian = torch.zeros(dim, dim, dtype=z.dtype, device=z.device)
        for j in range(dim):
            e = torch.zeros_like(z)
            e[j] = h
            jacobian[:, j] = (grad_fn((z + e).unsqueeze(0))[0]
                              - grad_fn((z - e).unsqueeze(0))[0]) / (2.0 * h)
        step = torch.linalg.solve(jacobian, g)
        z = z - step
        if step.abs().max() < 1e-13:
            break
    return z


def refined_minima(grad_fn, seeds: dict, keys, device) -> torch.Tensor:
    """Newton-refine a set of named critical-point seeds into a stacked tensor."""
    refined = []
    for key in keys:
        z0 = torch.as_tensor(seeds[key][0], dtype=torch.float64, device=device)
        refined.append(newton_refine(grad_fn, z0))
    return torch.stack(refined)
