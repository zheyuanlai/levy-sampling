"""Potentials V, grad V (and shifted energy differences) for E1-E4.

Conventions
-----------
* x always has shape (..., d); V returns (...,); grad returns (..., d).
* dtype is torch.float64 everywhere (set globally by the caller).
* `V_delta(x, R)` returns V(x - R) - V(x) for a batch of shifts R (J, d)
  -> (N, J). The generic implementation broadcasts; CoupledPhi4 overrides it
  with the O(N_s) moment trick for homogeneous shifts.
* Evaluation counters (`n_V`, `n_grad`, `n_Vdelta`) count *points evaluated*
  (host-side shape arithmetic only; no device sync).
"""
from __future__ import annotations

import math

import numpy as np
import torch


class Potential:
    d: int = 1
    name: str = "potential"

    def __init__(self) -> None:
        self.reset_counters()

    def reset_counters(self) -> None:
        self.n_V = 0
        self.n_grad = 0
        self.n_Vdelta = 0

    # -- interface ---------------------------------------------------------
    def V(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def V_delta(self, x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """V(x - R) - V(x); x: (N, d), R: (J, d) -> (N, J)."""
        self.n_Vdelta += x.shape[0] * R.shape[0]
        shifted = x.unsqueeze(1) - R.unsqueeze(0)          # (N, J, d)
        v0 = self._V_raw(x)                                # (N,)
        return self._V_raw(shifted) - v0.unsqueeze(1)

    # -- internals: _V_raw does not bump counters (used by V and V_delta) --
    def _V_raw(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ======================================================================= E1
class DoubleWell1D(Potential):
    """V(x) = (x^2 - 1)^2, minima at +-1, saddle 0, Delta V = 1."""

    d = 1
    name = "double_well_1d"

    def _V_raw(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0]
        return (x1 * x1 - 1.0) ** 2

    def V(self, x: torch.Tensor) -> torch.Tensor:
        self.n_V += int(np.prod(x.shape[:-1]))
        return self._V_raw(x)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        self.n_grad += int(np.prod(x.shape[:-1]))
        x1 = x[..., 0]
        return (4.0 * x1 * (x1 * x1 - 1.0)).unsqueeze(-1)

    @staticmethod
    def kramers_time(beta: float = 8.0) -> float:
        # tau = 2*pi / sqrt(V''(min) |V''(saddle)|) * exp(beta * DeltaV)
        return 2.0 * math.pi / math.sqrt(8.0 * 4.0) * math.exp(beta * 1.0)


# ======================================================================= E2
class MoG40(Potential):
    """V(x) = -(1/beta) log sum_k exp(-||x - mu_k||^2 / 2)  on R^2.

    The 1/beta prefactor makes pi ~ exp(-beta V) an *equal-weight mixture of
    N(mu_k, I_2)* exactly, independent of beta; beta only rescales barriers.
    """

    d = 2
    name = "mog40"

    def __init__(self, beta: float = 8.0, device: str | torch.device = "cuda") -> None:
        super().__init__()
        self.beta = beta
        rng = np.random.default_rng(0)
        mu = rng.uniform(-40.0, 40.0, size=(40, 2))
        self.mu = torch.as_tensor(mu, dtype=torch.float64, device=device)  # (40,2)

    def _sq_dists(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(-2) - self.mu                    # (..., 40, 2)
        return (diff * diff).sum(-1)                        # (..., 40)

    def _V_raw(self, x: torch.Tensor) -> torch.Tensor:
        return -(1.0 / self.beta) * torch.logsumexp(-0.5 * self._sq_dists(x), dim=-1)

    def V(self, x: torch.Tensor) -> torch.Tensor:
        self.n_V += int(np.prod(x.shape[:-1]))
        return self._V_raw(x)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        self.n_grad += int(np.prod(x.shape[:-1]))
        diff = x.unsqueeze(-2) - self.mu                    # (..., 40, 2)
        w = torch.softmax(-0.5 * (diff * diff).sum(-1), dim=-1)  # (..., 40)
        return (1.0 / self.beta) * (w.unsqueeze(-1) * diff).sum(-2)

    def sample_exact(self, n: int, gen: torch.Generator) -> torch.Tensor:
        """Exact i.i.d. draws from the equal-weight mixture of N(mu_k, I2)."""
        dev = self.mu.device
        k = torch.randint(0, 40, (n,), generator=gen, device=dev)
        z = torch.randn(n, 2, generator=gen, device=dev)
        return self.mu[k] + z


# ======================================================================= E3
_MB_A = (-200.0, -100.0, -170.0, 15.0)
_MB_a = (-1.0, -1.0, -6.5, 0.7)
_MB_b = (0.0, 0.0, 11.0, 0.6)
_MB_c = (-10.0, -10.0, -6.5, 0.7)
_MB_x = (1.0, 0.0, -0.5, -1.0)
_MB_y = (0.0, 0.5, 1.5, 1.0)

# verified critical points of U_MB (asserted in the notebook / tests)
MB_CRITICAL = {
    "min_A": ((-0.5582, 1.4417), -146.70),
    "min_B": ((0.6235, 0.0280), -108.17),
    "min_C": ((-0.0500, 0.4667), -80.77),
    "saddle_S1": ((-0.8220, 0.6243), -40.66),   # A <-> C
    "saddle_S2": ((0.2125, 0.2930), -72.25),    # C <-> B
}


def muller_brown_2d(z: torch.Tensor) -> torch.Tensor:
    """U_MB on (..., 2)."""
    z1, z2 = z[..., 0], z[..., 1]
    out = torch.zeros_like(z1)
    for A, a, b, c, x0, y0 in zip(_MB_A, _MB_a, _MB_b, _MB_c, _MB_x, _MB_y):
        dx, dy = z1 - x0, z2 - y0
        out = out + A * torch.exp(a * dx * dx + b * dx * dy + c * dy * dy)
    return out


def muller_brown_2d_grad(z: torch.Tensor) -> torch.Tensor:
    z1, z2 = z[..., 0], z[..., 1]
    g1 = torch.zeros_like(z1)
    g2 = torch.zeros_like(z1)
    for A, a, b, c, x0, y0 in zip(_MB_A, _MB_a, _MB_b, _MB_c, _MB_x, _MB_y):
        dx, dy = z1 - x0, z2 - y0
        e = A * torch.exp(a * dx * dx + b * dx * dy + c * dy * dy)
        g1 = g1 + e * (2.0 * a * dx + b * dy)
        g2 = g2 + e * (b * dx + 2.0 * c * dy)
    return torch.stack([g1, g2], dim=-1)


class TransformedMuellerBrown10D(Potential):
    """U(z) = U_MB(z1,z2)/s + ||z_{3:10}||^2/(2 sigma_aux^2), sampled in x = z B^T.

    B = Q diag(linspace(0.75, 1.45, 10)), Q from QR of a default_rng(12345)
    standard normal.  V(x) = U(x B^{-T});  row-vector gradient
    grad_x V = (grad_z U) B^{-1}.
    """

    d = 10
    name = "muller_brown_10d"
    s = 40.0
    sigma_aux = 0.4

    def __init__(self, device: str | torch.device = "cuda") -> None:
        super().__init__()
        rng = np.random.default_rng(12345)
        Q, _ = np.linalg.qr(rng.standard_normal((10, 10)))
        B = Q @ np.diag(np.linspace(0.75, 1.45, 10))
        self.B = torch.as_tensor(B, dtype=torch.float64, device=device)
        self.Binv = torch.as_tensor(np.linalg.inv(B), dtype=torch.float64, device=device)

    def to_latent(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.Binv.T          # z = x B^{-T}

    def from_latent(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.B.T             # x = z B^T

    def _U_latent(self, z: torch.Tensor) -> torch.Tensor:
        aux = z[..., 2:]
        return muller_brown_2d(z[..., :2]) / self.s + 0.5 * (aux * aux).sum(-1) / self.sigma_aux**2

    def _V_raw(self, x: torch.Tensor) -> torch.Tensor:
        return self._U_latent(self.to_latent(x))

    def V(self, x: torch.Tensor) -> torch.Tensor:
        self.n_V += int(np.prod(x.shape[:-1]))
        return self._V_raw(x)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        self.n_grad += int(np.prod(x.shape[:-1]))
        z = self.to_latent(x)
        gz = torch.zeros_like(z)
        gz[..., :2] = muller_brown_2d_grad(z[..., :2]) / self.s
        gz[..., 2:] = z[..., 2:] / self.sigma_aux**2
        return gz @ self.Binv           # (grad_z U) B^{-1}


# ======================================================================= E4
PHI4_W_COEFFS = dict(cxy=-0.05, hx=0.03, hy=0.06)

# verified coherent minima of W (asserted in the notebook / tests)
PHI4_MINIMA = {
    "--": ((-1.0099, -1.0135), -0.1412),
    "-+": ((-0.9976, 0.9860), 0.0792),
    "+-": ((0.9898, -1.0013), 0.0196),
    "++": ((1.0025, 0.9988), 0.0400),
}
PHI4_ESCAPE_BARRIERS = {"--": 1.082, "-+": 0.892, "+-": 0.921, "++": 0.990}
PHI4_LAPLACE_MASSES = {"--": 0.583, "-+": 0.106, "+-": 0.169, "++": 0.142}


def phi4_W(v: torch.Tensor) -> torch.Tensor:
    """Site potential W on (..., 2)."""
    x, y = v[..., 0], v[..., 1]
    return ((x * x - 1.0) ** 2 + (y * y - 1.0) ** 2
            - 0.05 * x * y + 0.03 * x + 0.06 * y)


def phi4_W_grad(v: torch.Tensor) -> torch.Tensor:
    x, y = v[..., 0], v[..., 1]
    gx = 4.0 * x * (x * x - 1.0) - 0.05 * y + 0.03
    gy = 4.0 * y * (y * y - 1.0) - 0.05 * x + 0.06
    return torch.stack([gx, gy], dim=-1)


class CoupledPhi4(Potential):
    """Periodic Ginzburg-Landau chain: N_s = 12 sites q_i in R^2, d = 24.

    V(q) = kappa/(2 delta) sum_i ||q_{i+1} - q_i||^2 + delta sum_i W(q_i).

    For a *homogeneous* shift r = 1_{N_s} (x) dvec the gradient energy is
    exactly invariant, so V(q - r) - V(q) is a fixed polynomial in dvec whose
    coefficients are per-particle moments (computed once per call in O(N_s)).
    """

    name = "coupled_phi4"
    Ns = 12
    d = 24
    kappa = 2.5

    def __init__(self) -> None:
        super().__init__()
        self.delta = 1.0 / self.Ns

    def _sites(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-1], self.Ns, 2)

    def _V_raw(self, x: torch.Tensor) -> torch.Tensor:
        q = self._sites(x)
        dq = torch.roll(q, shifts=-1, dims=-2) - q
        grad_energy = (self.kappa / (2.0 * self.delta)) * (dq * dq).sum((-1, -2))
        site_energy = self.delta * phi4_W(q).sum(-1)
        return grad_energy + site_energy

    def V(self, x: torch.Tensor) -> torch.Tensor:
        self.n_V += int(np.prod(x.shape[:-1]))
        return self._V_raw(x)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        self.n_grad += int(np.prod(x.shape[:-1]))
        q = self._sites(x)
        lap = 2.0 * q - torch.roll(q, 1, dims=-2) - torch.roll(q, -1, dims=-2)
        g = (self.kappa / self.delta) * lap + self.delta * phi4_W_grad(q)
        return g.reshape(*x.shape)

    # ---- moment-exact homogeneous energy difference -----------------------
    def moments(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        q = self._sites(x)
        xs, ys = q[..., 0], q[..., 1]
        return dict(
            x1=xs.sum(-1), x2=(xs * xs).sum(-1), x3=(xs ** 3).sum(-1),
            y1=ys.sum(-1), y2=(ys * ys).sum(-1), y3=(ys ** 3).sum(-1),
            xy=(xs * ys).sum(-1),
        )

    def V_delta_homogeneous(self, x: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        """V(x - 1 (x) d) - V(x) for per-site shifts D: (J, 2) -> (N, J)."""
        self.n_Vdelta += x.shape[0] * D.shape[0]
        m = self.moments(x)                                  # each (N,)
        dx, dy = D[:, 0].unsqueeze(0), D[:, 1].unsqueeze(0)  # (1, J)
        Ns = float(self.Ns)
        x1, x2, x3 = m["x1"].unsqueeze(1), m["x2"].unsqueeze(1), m["x3"].unsqueeze(1)
        y1, y2, y3 = m["y1"].unsqueeze(1), m["y2"].unsqueeze(1), m["y3"].unsqueeze(1)
        xy1 = m["xy"].unsqueeze(1)
        del xy1  # cross moment not needed: the xy term is bilinear, see below
        # quartic part per axis: ((u-d)^2-1)^2 - (u^2-1)^2
        #   = -4 d u^3 + 6 d^2 u^2 + (4 d - 4 d^3) u + d^4 - 2 d^2
        quart = (-4.0 * dx * x3 + 6.0 * dx * dx * x2 + (4.0 * dx - 4.0 * dx ** 3) * x1
                 + Ns * (dx ** 4 - 2.0 * dx * dx)
                 - 4.0 * dy * y3 + 6.0 * dy * dy * y2 + (4.0 * dy - 4.0 * dy ** 3) * y1
                 + Ns * (dy ** 4 - 2.0 * dy * dy))
        # cross: -0.05[(x-dx)(y-dy) - xy] = 0.05(x dy + y dx) - 0.05 dx dy
        cross = 0.05 * (x1 * dy + y1 * dx) - 0.05 * Ns * dx * dy
        # linear fields: 0.03(-dx) + 0.06(-dy) per site
        lin = -(0.03 * dx + 0.06 * dy) * Ns
        return self.delta * (quart + cross + lin)

    def V_delta(self, x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """Homogeneous fast path. R rows must be 1_{N_s} (x) d (checked once)."""
        Rq = R.reshape(R.shape[0], self.Ns, 2)
        if not getattr(self, "_homog_checked", False):
            assert torch.allclose(Rq, Rq[:, :1, :].expand_as(Rq)), \
                "phi4 V_delta fast path requires homogeneous shifts"
            self._homog_checked = True
        return self.V_delta_homogeneous(x, Rq[:, 0, :])

    def kink_energy(self) -> float:
        # sigma = int_{-1}^{1} sqrt(2 kappa (1-x^2)^2) dx = 4/3 sqrt(2 kappa)
        return (4.0 / 3.0) * math.sqrt(2.0 * self.kappa)


class MuellerBrownLatent2D(Potential):
    """Reduced latent 2D potential U_MB(z)/s (used by the E3 certificate:
    jumps and test functions act on z_{1:2} only, dot products are
    affine-invariant, and the aux Gaussian factorises out exactly, so the
    full 10D residual equals this 2D one)."""

    d = 2
    name = "muller_brown_latent2d"

    def __init__(self, s: float = 40.0) -> None:
        super().__init__()
        self.s = s

    def _V_raw(self, z: torch.Tensor) -> torch.Tensor:
        return muller_brown_2d(z) / self.s

    def V(self, z: torch.Tensor) -> torch.Tensor:
        self.n_V += int(np.prod(z.shape[:-1]))
        return self._V_raw(z)

    def grad(self, z: torch.Tensor) -> torch.Tensor:
        self.n_grad += int(np.prod(z.shape[:-1]))
        return muller_brown_2d_grad(z) / self.s


# ================================================================ utilities
def newton_refine(grad_fn, z0: torch.Tensor, n_iter: int = 60) -> torch.Tensor:
    """Newton on grad_fn = 0 with finite-difference Jacobian (fp64, small dim)."""
    z = z0.clone()
    dim = z.numel()
    h = 1e-6
    for _ in range(n_iter):
        g = grad_fn(z.unsqueeze(0))[0]
        J = torch.zeros(dim, dim, dtype=z.dtype, device=z.device)
        for j in range(dim):
            e = torch.zeros_like(z)
            e[j] = h
            J[:, j] = (grad_fn((z + e).unsqueeze(0))[0]
                       - grad_fn((z - e).unsqueeze(0))[0]) / (2.0 * h)
        step = torch.linalg.solve(J, g)
        z = z - step
        if step.abs().max() < 1e-13:
            break
    return z
