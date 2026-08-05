"""Alpha-stable jump measures nu for the E4 jump-design study.

The manuscript's four experiments all use a nu whose atoms were chosen from the
target's basin geometry.  This module supplies the complementary case: a nu
drawn from the *same* symmetric alpha-stable family that FLA already uses as
the uncorrected heavy-tailed control, which encodes nothing about the four
coherent phases of the phi^4 chain.

Construction
------------
A genuine alpha-stable Levy measure has infinite activity and is therefore not
a compound-Poisson nu at all.  Restricting it to a finite shell is what makes it
admissible in the weak stationarity identity, not a numerical convenience.  We
truncate *coordinatewise*, so the law factorises and a deterministic quadrature
exists in low dimension:

    kappa_{sigma,c} = Law(sigma * eta),
    eta in R^m with i.i.d. eta_i ~ SaS(1) conditioned on |eta_i| <= c.

Coordinates come from :func:`src.samplers.sample_sas`, the Chambers--Mallows--
Stuck routine FLA itself uses, so the increment family is literally FLA's.

Two designs are built on this:

``TruncatedCoordinateStableLaw(m=24)``
    applied directly to the state.  Site displacements are independent, so a
    jump destroys chain coherence.  Not homogeneous -- it must never reach
    ``CoupledPhi4.V_delta`` (see ``assert_not_homogeneous_only``).  A product
    quadrature would need q_u**24 nodes, so only the realised-displacement
    estimators can run: this is the obstruction the study is about.

``TiledStableLaw`` over ``TruncatedCoordinateStableLaw(m=2)``
    the traditional composed design: draw one per-site displacement and tile it
    coherently across all N_s sites.  Homogeneous, so the moment-exact
    ``V_delta`` fast path stays valid and a q_u**2 product quadrature is cheap.

Interface follows the duck-typed contract of :mod:`src.jumps`: ``sample(n, gen)``
state-free, ``quadrature_shifts(...) -> (shifts, log_weights)`` with weights
summing to one, ``max_reach()``, plus ``.d`` and ``.device``.
"""
from __future__ import annotations

import math

import torch

from .jumps import gauss_legendre_01
from .samplers import sample_sas

# Bounded coordinatewise rejection.  After k passes the residual per-coordinate
# probability of still being out of range is (1 - q)**(k+1); at the loosest
# level used here (q = 0.95) sixteen passes give 5e-21, so the clamp fallback is
# a formality.  It is counted anyway -- a silent fallback would change the
# declared law, and the deterministic quadrature is built from the *exact*
# truncated quantile, so the two would then disagree.
#
# Redrawing the whole tensor each pass avoids a host synchronisation, but the
# expected number of passes actually needed is ~2, so we test for cleanliness
# every ``_REJECTION_SYNC_EVERY`` passes and stop.  That costs at most a couple
# of syncs per step instead of fourteen wasted draws.
_REJECTION_ROUNDS: int = 16
_REJECTION_SYNC_EVERY: int = 4

# Monte-Carlo sample used to calibrate E||eta|| once per law.  Fixed seed: the
# resulting sigma is part of the declared measure and must not move between
# runs.
_CALIBRATION_SAMPLES: int = 1 << 20
_CALIBRATION_SEED: int = 20260805


def sas_abs_quantile(alpha: float, mass: float) -> float:
    """c with P(|eta| <= c) = ``mass`` for eta ~ SaS(1).

    scipy's ``levy_stable`` with beta = 0 uses the same normalisation as
    :func:`src.samplers.sample_sas` (verified against 4e6 draws to 3 decimal
    places out to the 0.995 quantile), so its quantile function can be used
    directly.
    """
    if not 0.0 < mass < 1.0:
        raise ValueError("truncation mass must lie strictly between 0 and 1")
    from scipy.stats import levy_stable
    return float(levy_stable.ppf(0.5 * (1.0 + mass), alpha, 0.0))


def _truncated_quantile(alpha: float, mass: float, u) -> torch.Tensor:
    """Quantile of SaS(1) conditioned on |eta| <= c, evaluated at u in [0,1].

    By symmetry F(-c) = (1 - mass)/2 and F(c) - F(-c) = mass, so the truncated
    quantile is just the untruncated one reparameterised onto the retained
    probability window.  No table inversion is needed.
    """
    from scipy.stats import levy_stable
    u_np = u.detach().cpu().numpy() if isinstance(u, torch.Tensor) else u
    p = 0.5 * (1.0 - mass) + mass * u_np
    return torch.as_tensor(levy_stable.ppf(p, alpha, 0.0), dtype=torch.float64)


class TruncatedCoordinateStableLaw:
    """kappa = Law(sigma * eta), eta_i i.i.d. SaS(alpha) conditioned on |eta_i| <= c.

    ``truncation_mass`` is q = P(|eta_i| <= c); c is derived from it, so the
    heaviness of the retained tail is the knob rather than an opaque cutoff.
    ``sigma`` may be given directly or solved for from a target mean jump length
    via :meth:`with_mean_length`.
    """

    def __init__(self, m: int, sigma: float, truncation_mass: float,
                 device, alpha: float = 1.7,
                 dtype: torch.dtype = torch.float64) -> None:
        if int(m) != m or m < 1:
            raise ValueError("m must be a positive integer")
        if not (1.0 < float(alpha) < 2.0):
            raise ValueError("alpha must lie in (1, 2) for nu to have a finite "
                             "first moment")
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma must be finite and positive")
        self.d = int(m)
        self.m = int(m)
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.truncation_mass = float(truncation_mass)
        self.c = sas_abs_quantile(self.alpha, self.truncation_mass)
        self.device = device
        self.dtype = dtype
        # Device-resident counter for the clamp fallback; read at checkpoints.
        self.fallback_count = torch.zeros((), dtype=torch.int64, device=device)
        self.draw_count = torch.zeros((), dtype=torch.int64, device=device)

    # -- construction helpers ---------------------------------------------
    @staticmethod
    def mean_norm_unit_scale(m: int, truncation_mass: float,
                             alpha: float = 1.7, device="cpu") -> float:
        """E||eta|| for the truncated law at sigma = 1, by fixed-seed MC."""
        probe = TruncatedCoordinateStableLaw(
            m, 1.0, truncation_mass, device, alpha=alpha)
        gen = torch.Generator(device=device)
        gen.manual_seed(_CALIBRATION_SEED)
        total = 0.0
        remaining = _CALIBRATION_SAMPLES
        block = max(1, (1 << 22) // max(1, m))
        drawn = 0
        while remaining > 0:
            n = min(block, remaining)
            total += float(probe.sample(n, gen).norm(dim=1).sum().item())
            drawn += n
            remaining -= n
        return total / drawn

    @classmethod
    def with_mean_length(cls, m: int, mean_length: float,
                         truncation_mass: float, device,
                         alpha: float = 1.7) -> "TruncatedCoordinateStableLaw":
        """Law whose mean jump length E||R|| equals ``mean_length``.

        E||R|| = sigma * E||eta|| is linear in sigma, so one MC estimate of
        E||eta|| at unit scale determines sigma exactly.
        """
        unit = cls.mean_norm_unit_scale(m, truncation_mass, alpha=alpha,
                                        device=device)
        law = cls(m, float(mean_length) / unit, truncation_mass, device,
                  alpha=alpha)
        law._mean_length = float(mean_length)
        return law

    def mean_length(self) -> float:
        """E||R||, the scale the drift cap and the fairness match are set by."""
        cached = getattr(self, "_mean_length", None)
        if cached is None:
            cached = self.sigma * self.mean_norm_unit_scale(
                self.m, self.truncation_mass, alpha=self.alpha,
                device=self.device)
            self._mean_length = cached
        return cached

    # -- nu interface ------------------------------------------------------
    def sample(self, n: int, gen: torch.Generator) -> torch.Tensor:
        """State-free draw of n increments, shape (n, m)."""
        shape = (int(n), self.m)
        r = sample_sas(shape, self.alpha, gen, self.device)
        outside = r.abs() > self.c
        for k in range(_REJECTION_ROUNDS):
            fresh = sample_sas(shape, self.alpha, gen, self.device)
            r = torch.where(outside, fresh, r)
            outside = r.abs() > self.c
            if (k + 1) % _REJECTION_SYNC_EVERY == 0 and not bool(outside.any()):
                break
        # Formality: clamp whatever survived every independent rejection.
        self.fallback_count += outside.to(torch.int64).sum()
        self.draw_count += r.numel()
        r = torch.where(outside, torch.sign(r) * self.c, r)
        return (self.sigma * r).to(self.dtype)

    def quadrature_shifts(self, q_u: int):
        """Product Gauss--Legendre-in-probability rule; m = 2 only.

        Nodes are quantiles of the truncated marginal at Gauss--Legendre points
        of the probability variable, so the weights are exact probabilities of
        the declared law and sum to one -- the direct analogue of
        ``gauss_legendre_m11`` being a quadrature of Unif(-h, h).

        A product rule needs q_u**m nodes, which is why the 24-dimensional
        design has no deterministic score.
        """
        if self.m != 2:
            raise NotImplementedError(
                f"a product quadrature of the coordinatewise stable law needs "
                f"q_u**{self.m} nodes; only m = 2 is affordable. Use the "
                f"realised-displacement estimator (IIDBankScore) instead.")
        u, w = gauss_legendre_01(int(q_u), self.device)          # sum(w) = 1
        nodes = _truncated_quantile(self.alpha, self.truncation_mass, u)
        nodes = self.sigma * nodes.to(device=self.device, dtype=self.dtype)
        w = w.to(self.dtype)
        shifts = torch.stack([
            nodes.repeat_interleave(int(q_u)),
            nodes.repeat(int(q_u)),
        ], dim=1)                                                # (q_u**2, 2)
        weights = (w.unsqueeze(1) * w.unsqueeze(0)).reshape(-1)  # (q_u**2,)
        return shifts, torch.log(weights)

    def max_reach(self) -> float:
        """Bound on ||r||; the componentwise bound is ``sigma * c``."""
        return self.sigma * self.c * math.sqrt(self.m)

    def max_componentwise_reach(self) -> float:
        return self.sigma * self.c

    def fallback_fraction(self) -> float:
        drawn = int(self.draw_count.item())
        return float(self.fallback_count.item()) / drawn if drawn else 0.0

    def describe(self) -> dict:
        return {
            "type": "TruncatedCoordinateStableLaw",
            "m": self.m,
            "alpha": self.alpha,
            "sigma": self.sigma,
            "truncation_mass": self.truncation_mass,
            "c": self.c,
            "mean_length": self.mean_length(),
            "max_reach": self.max_reach(),
            "max_componentwise_reach": self.max_componentwise_reach(),
            "rejection_rounds": _REJECTION_ROUNDS,
            "clamp_fallback_fraction": self.fallback_fraction(),
        }


class TiledStableLaw:
    """Coherent tiling of a low-dimensional base law across ``n_sites`` sites.

    ``r = 1_{n_sites} (x) base_draw`` in the flat layout (x0, y0, x1, y1, ...),
    matching how the manuscript's phase-edge atoms are built.  Because every
    site receives the same displacement, the chain's gradient energy is exactly
    invariant and ``CoupledPhi4.V_delta``'s moment-exact fast path applies.
    """

    def __init__(self, base, n_sites: int) -> None:
        if int(n_sites) != n_sites or n_sites < 1:
            raise ValueError("n_sites must be a positive integer")
        self.base = base
        self.n_sites = int(n_sites)
        self.site_dim = int(base.d)
        self.d = self.n_sites * self.site_dim
        self.device = base.device
        self.dtype = getattr(base, "dtype", torch.float64)

    def _tile(self, r: torch.Tensor) -> torch.Tensor:
        return r.unsqueeze(1).expand(r.shape[0], self.n_sites,
                                     self.site_dim).reshape(r.shape[0], self.d)

    def sample(self, n: int, gen: torch.Generator) -> torch.Tensor:
        return self._tile(self.base.sample(n, gen))

    def quadrature_shifts(self, *args, **kwargs):
        shifts, log_weights = self.base.quadrature_shifts(*args, **kwargs)
        return self._tile(shifts).contiguous(), log_weights

    def max_reach(self) -> float:
        return self.base.max_reach() * math.sqrt(self.n_sites)

    def max_componentwise_reach(self) -> float:
        return self.base.max_componentwise_reach()

    def mean_length(self) -> float:
        """E||R|| in the full state space; tiling is coherent, so the per-site
        length is amplified by sqrt(n_sites)."""
        return self.base.mean_length() * math.sqrt(self.n_sites)

    def metric_reach(self) -> float:
        """Bound on the shift of the collective variable qbar = mean_i q_i.

        Tiling is coherent, so qbar moves by exactly the base displacement.
        This is what the basin map's domain has to cover.
        """
        return self.base.max_reach()

    def fallback_fraction(self) -> float:
        return self.base.fallback_fraction()

    def describe(self) -> dict:
        base = dict(self.base.describe())
        base["base_type"] = base.pop("type")
        base.update({
            "type": "TiledStableLaw",
            "n_sites": self.n_sites,
            "d": self.d,
            "mean_length": self.mean_length(),
            "per_site_mean_length": self.base.mean_length(),
            "max_reach": self.max_reach(),
            "max_componentwise_reach": self.max_componentwise_reach(),
            "metric_reach": self.metric_reach(),
        })
        return base


def is_homogeneous_law(law) -> bool:
    """True when every draw is 1_{n_sites} (x) d, i.e. safe for V_delta."""
    return isinstance(law, TiledStableLaw) or hasattr(law, "atoms")


def assert_homogeneous_for_v_delta(law, potential) -> None:
    """Fail loudly rather than let ``CoupledPhi4.V_delta`` corrupt a shift.

    That method caches its homogeneity assertion after the first call and then
    silently reduces later batches to their first site, so a non-homogeneous
    law reaching it produces wrong energies without any error.
    """
    if hasattr(potential, "V_delta_homogeneous") and not is_homogeneous_law(law):
        raise TypeError(
            f"{type(law).__name__} is not homogeneous; {type(potential).__name__}"
            ".V_delta would silently reduce each shift to its first site. Use a "
            "realised-displacement score (IIDBankScore), which evaluates chord "
            "energies through _V_raw.")
