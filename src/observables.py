"""Deterministic physical observables of a sampled configuration ensemble.

Everything here is a pure function of tensors: no randomness, no file I/O
beyond the basin-map cache, no plotting, no experiment-specific paths. The E3
collective variable, the E4 two-component order parameter and its
susceptibility, and the chain's energy, coherence, two-point correlation, kink
density, heat capacity and Binder cumulant all live in this one module, next to
the single basin map that labels both E3 basins and E4 phases.

Oracle accounting
-----------------
Every function that reaches a ``Target`` does so inside ``target.no_count()``.
Observable evaluation is analysis, not sampler work, and must never move the
oracle counters.

Weighted variants
-----------------
Each scalar or matrix observable the E4 SNIS cross-check needs also has a
``*_weighted`` form taking normalized weights ``w`` (``w_k >= 0``,
``sum_k w_k = 1``; the weights are renormalized defensively). First moments use
``sum_k w_k f_k``. Second moments use the reliability-weight denominator
``1 - sum_k w_k^2``, which equals ``(N-1)/N`` at uniform weights and therefore
reproduces the ``N-1`` sample convention of the unweighted estimators exactly.

Phase labels
------------
Points that fall outside the basin-map domain are labelled ``OUTSIDE_LABEL``
and their mass is reported separately. They are never clamped into a boundary
basin: for E4 a phase-to-phase jump can park the order parameter well beyond
the map, and clamping would relabel exactly the transport the study measures.
"""
from __future__ import annotations

import hashlib
import math
import os
import warnings

import numpy as np
import torch

from .device import resolve_device

#: Sentinel label for a point outside the basin-map domain. Never a phase.
OUTSIDE_LABEL = -1

#: Smallest admissible ``1 - sum_k w_k^2``. At or below this the weighted
#: sample carries no second-moment information (weighted ESS ~ 1).
_MIN_RELIABILITY = 1e-12


# ============================================================ shape/weight guards
def _check_matrix(values: torch.Tensor, name: str,
                  width: int | None = None) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(values).__name__}")
    if values.ndim != 2:
        raise ValueError(
            f"{name} must have shape (N, {'d' if width is None else width}), "
            f"got {tuple(values.shape)}")
    if width is not None and values.shape[1] != width:
        raise ValueError(
            f"{name} must have shape (N, {width}), got {tuple(values.shape)}")
    if values.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample")
    return values


def _check_vector(values: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(values).__name__}")
    if values.ndim != 1:
        raise ValueError(f"{name} must have shape (N,), got {tuple(values.shape)}")
    if values.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample")
    return values


def _check_order_parameter(m: torch.Tensor) -> torch.Tensor:
    return _check_matrix(m, "order parameter m", width=2)


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return value


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _normalized_weights(weights: torch.Tensor, n_samples: int) -> torch.Tensor:
    """Validate and renormalize SNIS weights against a sample count."""
    if not isinstance(weights, torch.Tensor):
        raise TypeError(
            f"weights must be a torch.Tensor, got {type(weights).__name__}")
    w = weights.reshape(-1).to(torch.float64)
    if w.numel() != n_samples:
        raise ValueError(
            f"weights must have one entry per sample: got {w.numel()} weights "
            f"for {n_samples} samples")
    if not bool(torch.isfinite(w).all().item()):
        raise ValueError("weights must be finite")
    if bool((w < 0).any().item()):
        raise ValueError("weights must be nonnegative")
    total = w.sum()
    if float(total.item()) <= 0.0:
        raise ValueError("weights must have a positive sum")
    return w / total


def _reliability_denominator(w: torch.Tensor) -> torch.Tensor:
    """``1 - sum_k w_k^2``, the weighted analogue of the ``N-1`` denominator."""
    denominator = 1.0 - (w * w).sum()
    if float(denominator.item()) <= _MIN_RELIABILITY:
        raise ValueError(
            "weights are too concentrated for a second-moment estimate: "
            f"1 - sum w^2 = {float(denominator.item()):.3e}; the weighted "
            "effective sample size is ~1")
    return denominator


def _weighted_sum(values: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.tensordot(w.to(values.dtype).to(values.device), values,
                           dims=([0], [0]))


# ================================================================= sample means
def sample_mean(values: torch.Tensor) -> torch.Tensor:
    """``(1/N) sum_k f_k``, averaging over the leading sample axis."""
    if values.ndim < 1 or values.shape[0] < 1:
        raise ValueError(
            f"values must have a nonempty leading sample axis, got "
            f"{tuple(values.shape)}")
    return values.mean(dim=0)


def sample_mean_weighted(values: torch.Tensor,
                         weights: torch.Tensor) -> torch.Tensor:
    """``sum_k w_k f_k`` for normalized weights ``w``."""
    if values.ndim < 1 or values.shape[0] < 1:
        raise ValueError(
            f"values must have a nonempty leading sample axis, got "
            f"{tuple(values.shape)}")
    return _weighted_sum(values, _normalized_weights(weights, values.shape[0]))


# ==================================================================== E3 the CV
def latent_cv(target, x: torch.Tensor) -> torch.Tensor:
    """The E3 collective variable ``z_{1:2} = (x B^{-T})_{1:2}``, shape ``(..., 2)``.

    The CV is the LATENT pair, never the first two sampling coordinates: the
    surface is sampled in ``x = z B^T`` with a dense ``B = Q diag(s)``, so every
    sampling coordinate mixes all ten latent directions and ``x_{1:2}`` is not a
    collective variable at all.
    """
    potential = target.potential
    if not hasattr(potential, "collective_variable"):
        raise TypeError(
            f"{type(potential).__name__} has no collective_variable; latent_cv "
            "is defined for the embedded Muller-Brown surface")
    if x.ndim < 1 or x.shape[-1] != potential.d:
        raise ValueError(
            f"configurations must have shape (..., {potential.d}), got "
            f"{tuple(x.shape)}")
    with target.no_count():
        return potential.collective_variable(x)


# ============================================================ E4 chain accessors
def _chain(target):
    potential = target.potential
    for attribute in ("n_sites", "sites", "order_parameter", "energy_per_site",
                      "coherence", "two_point_correlation", "kink_density"):
        if not hasattr(potential, attribute):
            raise TypeError(
                f"{type(potential).__name__} is not a coupled quartic chain: "
                f"missing {attribute!r}")
    return potential


def _check_configurations(potential, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or x.shape[1] != potential.d:
        raise ValueError(
            f"configurations must have shape (N, {potential.d}) for a "
            f"{potential.n_sites}-site chain, got {tuple(x.shape)}")
    if x.shape[0] < 1:
        raise ValueError("configurations must contain at least one sample")
    return x


def _check_site_minima(minima: torch.Tensor) -> torch.Tensor:
    if not isinstance(minima, torch.Tensor):
        raise TypeError(
            f"site_minima must be a torch.Tensor, got {type(minima).__name__}")
    if minima.ndim != 2 or minima.shape[1] != 2 or minima.shape[0] < 1:
        raise ValueError(
            f"site_minima must have shape (K, 2) with K >= 1, got "
            f"{tuple(minima.shape)}")
    if not bool(torch.isfinite(minima).all().item()):
        raise ValueError("site_minima must be finite")
    return minima


# ============================================================ E4 order parameter
def order_parameter(target, x: torch.Tensor) -> torch.Tensor:
    """``m = (1/N_s) sum_i q_i``, the two-component magnetization, shape ``(N, 2)``.

    This is a two-component coupled quartic chain, so ``m`` is a vector in R^2,
    not a scalar phi^4 magnetization.
    """
    potential = _chain(target)
    _check_configurations(potential, x)
    with target.no_count():
        return potential.order_parameter(x)


def order_parameter_norm(m: torch.Tensor) -> torch.Tensor:
    """``||m||`` per sample, shape ``(N,)``."""
    return _check_order_parameter(m).norm(dim=1)


def order_parameter_mean(m: torch.Tensor) -> torch.Tensor:
    """``E[m] = (1/N) sum_k m_k``, shape ``(2,)``."""
    return sample_mean(_check_order_parameter(m))


def order_parameter_mean_weighted(m: torch.Tensor,
                                  weights: torch.Tensor) -> torch.Tensor:
    """``E_w[m] = sum_k w_k m_k``, shape ``(2,)``."""
    m = _check_order_parameter(m)
    return _weighted_sum(m, _normalized_weights(weights, m.shape[0]))


# ==================================================================== E4 phases
def phase_labels(m: torch.Tensor, basin_map) -> torch.Tensor:
    """Assign each order parameter ``m_k`` to a phase, shape ``(N,)``.

    ``basin_map`` must expose ``assign(points) -> labels`` and
    ``outside(points) -> bool mask``. Points outside the map domain get
    ``OUTSIDE_LABEL = -1``; they are never clamped into a boundary basin.
    """
    m = _check_order_parameter(m)
    for attribute in ("assign", "outside"):
        if not callable(getattr(basin_map, attribute, None)):
            raise TypeError(
                f"basin_map must provide a callable {attribute!r}; got "
                f"{type(basin_map).__name__}")
    labels = basin_map.assign(m).reshape(-1).to(torch.long)
    outside = basin_map.outside(m).reshape(-1)
    if labels.numel() != m.shape[0] or outside.numel() != m.shape[0]:
        raise ValueError(
            f"basin_map returned {labels.numel()} labels and {outside.numel()} "
            f"outside flags for {m.shape[0]} points")
    sentinel = torch.full_like(labels, OUTSIDE_LABEL)
    return torch.where(outside.to(torch.bool), sentinel, labels)


def _check_labels(labels: torch.Tensor, n_phases: int) -> torch.Tensor:
    n_phases = _positive_int(n_phases, "n_phases")
    labels = labels.reshape(-1).to(torch.long)
    if labels.numel() < 1:
        raise ValueError("labels must contain at least one sample")
    bad = (labels != OUTSIDE_LABEL) & ((labels < 0) | (labels >= n_phases))
    if bool(bad.any().item()):
        raise ValueError(
            f"labels must lie in [0, {n_phases}) or equal OUTSIDE_LABEL "
            f"({OUTSIDE_LABEL}); found {int(bad.sum().item())} out-of-range labels")
    return labels


def phase_probabilities(labels: torch.Tensor,
                        n_phases: int = 4) -> tuple[torch.Tensor, float]:
    """``(p, outside_mass)`` with ``p_s = #{k : l_k = s} / #{k : l_k != -1}``.

    The phase probabilities are computed over the non-outside samples only and
    sum to one there. The outside mass ``#{k : l_k = -1} / N`` is reported
    separately and is never folded into a phase; if every sample is outside the
    map domain the probabilities are all zero and the outside mass is 1.
    """
    labels = _check_labels(labels, n_phases)
    n_total = labels.numel()
    inside = labels != OUTSIDE_LABEL
    n_inside = int(inside.sum().item())
    outside_mass = float(n_total - n_inside) / float(n_total)
    probabilities = torch.zeros(n_phases, dtype=torch.float64,
                                device=labels.device)
    if n_inside:
        contribution = torch.full((n_inside,), 1.0 / n_inside,
                                  dtype=torch.float64, device=labels.device)
        probabilities.scatter_add_(0, labels[inside], contribution)
    return probabilities, outside_mass


def phase_probabilities_weighted(labels: torch.Tensor, weights: torch.Tensor,
                                 n_phases: int = 4) -> tuple[torch.Tensor, float]:
    """``(p, outside_mass)`` with ``p_s = sum_{k: l_k = s} w_k / sum_{k: l_k != -1} w_k``.

    Same convention as :func:`phase_probabilities`: outside mass is the weight
    ``sum_{k: l_k = -1} w_k`` reported separately, never folded into a phase.
    """
    labels = _check_labels(labels, n_phases)
    w = _normalized_weights(weights, labels.numel()).to(labels.device)
    inside = labels != OUTSIDE_LABEL
    inside_mass = w[inside].sum()
    outside_mass = float((1.0 - inside_mass).clamp_min(0.0).item())
    probabilities = torch.zeros(n_phases, dtype=torch.float64,
                                device=labels.device)
    if float(inside_mass.item()) > 0.0:
        probabilities.scatter_add_(0, labels[inside], w[inside] / inside_mass)
    return probabilities, outside_mass


# ==================================================================== E4 energy
def energy_per_site(target, x: torch.Tensor) -> torch.Tensor:
    """``V(q) / N_s`` per sample, shape ``(N,)``."""
    potential = _chain(target)
    _check_configurations(potential, x)
    with target.no_count():
        return potential.energy_per_site(x)


def energy_per_site_mean(energies_per_site: torch.Tensor) -> torch.Tensor:
    """``E[V/N_s] = (1/N) sum_k V_k / N_s``."""
    return sample_mean(_check_vector(energies_per_site, "energies_per_site"))


def energy_per_site_mean_weighted(energies_per_site: torch.Tensor,
                                  weights: torch.Tensor) -> torch.Tensor:
    """``E_w[V/N_s] = sum_k w_k V_k / N_s``."""
    e = _check_vector(energies_per_site, "energies_per_site")
    return _weighted_sum(e, _normalized_weights(weights, e.shape[0]))


def energy_per_site_variance(energies_per_site: torch.Tensor) -> torch.Tensor:
    """``Var[V/N_s]`` with the ``N-1`` denominator."""
    e = _check_vector(energies_per_site, "energies_per_site")
    if e.shape[0] < 2:
        raise ValueError("a variance needs at least two samples")
    centered = e - e.mean()
    return (centered * centered).sum() / (e.shape[0] - 1)


def energy_per_site_variance_weighted(energies_per_site: torch.Tensor,
                                      weights: torch.Tensor) -> torch.Tensor:
    """``sum_k w_k (V_k/N_s - E_w[V/N_s])^2 / (1 - sum_k w_k^2)``."""
    e = _check_vector(energies_per_site, "energies_per_site")
    w = _normalized_weights(weights, e.shape[0]).to(e.device)
    centered = e - _weighted_sum(e, w)
    return (w.to(e.dtype) * centered * centered).sum() / _reliability_denominator(w)


# ============================================================ E4 susceptibility
def susceptibility(m: torch.Tensor, beta: float,
                   n_sites: int) -> torch.Tensor:
    """``chi = beta * N_s * Cov(m)``, a ``(2, 2)`` matrix.

    ``Cov(m)`` is the sample covariance of the two-component order parameter
    with the ``N-1`` denominator.
    """
    m = _check_order_parameter(m)
    beta = _positive(beta, "beta")
    n_sites = _positive_int(n_sites, "n_sites")
    if m.shape[0] < 2:
        raise ValueError("a covariance needs at least two samples")
    centered = m - m.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (m.shape[0] - 1)
    return beta * n_sites * covariance


def susceptibility_weighted(m: torch.Tensor, weights: torch.Tensor,
                            beta: float, n_sites: int) -> torch.Tensor:
    """``chi = beta * N_s * Cov_w(m)`` with the weighted covariance

    ``Cov_w(m) = sum_k w_k (m_k - E_w[m]) (m_k - E_w[m])^T / (1 - sum_k w_k^2)``,
    which reduces to the ``N-1`` sample covariance at uniform weights.
    """
    m = _check_order_parameter(m)
    beta = _positive(beta, "beta")
    n_sites = _positive_int(n_sites, "n_sites")
    w = _normalized_weights(weights, m.shape[0]).to(m.device)
    centered = m - _weighted_sum(m, w)
    covariance = ((w.to(m.dtype).unsqueeze(1) * centered).T @ centered
                  / _reliability_denominator(w))
    return beta * n_sites * covariance


def relative_frobenius_error(estimate: torch.Tensor,
                             reference: torch.Tensor) -> float:
    """``||estimate - reference||_F / ||reference||_F``."""
    if estimate.shape != reference.shape:
        raise ValueError(
            f"estimate shape {tuple(estimate.shape)} != reference shape "
            f"{tuple(reference.shape)}")
    denominator = float(reference.to(torch.float64).norm().item())
    if denominator <= 0.0:
        raise ValueError(
            "reference has zero Frobenius norm; a relative error is undefined")
    difference = (estimate.to(torch.float64) - reference.to(torch.float64))
    return float(difference.norm().item()) / denominator


# ================================================================= E4 coherence
def coherence(target, x: torch.Tensor) -> torch.Tensor:
    """``G(q) = (1/N_s) sum_i ||q_{i+1} - q_i||^2``, periodic indices, shape ``(N,)``."""
    potential = _chain(target)
    _check_configurations(potential, x)
    with target.no_count():
        return potential.coherence(x)


def coherence_mean(coherences: torch.Tensor) -> torch.Tensor:
    """``E[G] = (1/N) sum_k G_k``."""
    return sample_mean(_check_vector(coherences, "coherences"))


def coherence_mean_weighted(coherences: torch.Tensor,
                            weights: torch.Tensor) -> torch.Tensor:
    """``E_w[G] = sum_k w_k G_k``."""
    g = _check_vector(coherences, "coherences")
    return _weighted_sum(g, _normalized_weights(weights, g.shape[0]))


# ====================================================== E4 two-point correlation
def two_point_correlation(target, x: torch.Tensor) -> torch.Tensor:
    """``C(r) = (1/N_s) sum_i q_i . q_{i+r}`` for ``r = 0..floor(N_s/2)``.

    Returned per sample with shape ``(N, R+1)``, ``R = floor(N_s/2)``, so the
    caller decides whether to average uniformly or with SNIS weights.
    """
    potential = _chain(target)
    _check_configurations(potential, x)
    with target.no_count():
        return potential.two_point_correlation(x)


def _check_correlations(C: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(C, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(C).__name__}")
    if C.ndim != 2 or C.shape[0] < 1 or C.shape[1] < 1:
        raise ValueError(
            f"{name} must have shape (N, R+1) with N, R+1 >= 1, got "
            f"{tuple(C.shape)}")
    return C


def two_point_correlation_mean(C: torch.Tensor) -> torch.Tensor:
    """``E[C(r)] = (1/N) sum_k C_k(r)``, shape ``(R+1,)``."""
    return sample_mean(_check_correlations(C, "C"))


def two_point_correlation_mean_weighted(C: torch.Tensor,
                                        weights: torch.Tensor) -> torch.Tensor:
    """``E_w[C(r)] = sum_k w_k C_k(r)``, shape ``(R+1,)``."""
    C = _check_correlations(C, "C")
    return _weighted_sum(C, _normalized_weights(weights, C.shape[0]))


def connected_two_point_correlation(C: torch.Tensor,
                                    mean_m: torch.Tensor) -> torch.Tensor:
    """``C_conn(r) = C(r) - ||E[m]||^2``, same shape as ``C``."""
    if not isinstance(C, torch.Tensor):
        raise TypeError(f"C must be a torch.Tensor, got {type(C).__name__}")
    if C.ndim < 1 or C.shape[-1] < 1:
        raise ValueError(
            f"C must have a nonempty trailing lag axis, got {tuple(C.shape)}")
    if mean_m.shape != (2,):
        raise ValueError(
            f"mean_m must be the mean order parameter of shape (2,), got "
            f"{tuple(mean_m.shape)}")
    return C - (mean_m * mean_m).sum()


def correlation_relative_l2(C_hat: torch.Tensor,
                            C_star: torch.Tensor) -> float:
    """``sqrt( sum_r (C_hat(r) - C_star(r))^2 / sum_r C_star(r)^2 )``."""
    if C_hat.shape != C_star.shape:
        raise ValueError(
            f"C_hat shape {tuple(C_hat.shape)} != C_star shape "
            f"{tuple(C_star.shape)}")
    hat = C_hat.to(torch.float64)
    star = C_star.to(torch.float64)
    denominator = float((star * star).sum().item())
    if denominator <= 0.0:
        raise ValueError(
            "reference correlation has zero L2 norm; a relative error is undefined")
    difference = hat - star
    return math.sqrt(float((difference * difference).sum().item()) / denominator)


# ============================================================== E4 kink density
def kink_density(target, x: torch.Tensor,
                 site_minima: torch.Tensor) -> torch.Tensor:
    """``K(q) = (1/N_s) sum_i 1{l_i != l_{i+1}}`` with periodic indices, shape ``(N,)``.

    Site labels are ``l_i = argmin_s ||q_i - mu_s||`` over the refined
    homogeneous minima ``mu_s in R^2`` passed as ``site_minima`` of shape
    ``(K, 2)``.
    """
    potential = _chain(target)
    _check_configurations(potential, x)
    minima = _check_site_minima(site_minima)
    with target.no_count():
        return potential.kink_density(x, minima.to(device=x.device, dtype=x.dtype))


def kink_density_mean(kink_densities: torch.Tensor) -> torch.Tensor:
    """``E[K] = (1/N) sum_k K_k``."""
    return sample_mean(_check_vector(kink_densities, "kink_densities"))


def kink_density_mean_weighted(kink_densities: torch.Tensor,
                               weights: torch.Tensor) -> torch.Tensor:
    """``E_w[K] = sum_k w_k K_k``."""
    k = _check_vector(kink_densities, "kink_densities")
    return _weighted_sum(k, _normalized_weights(weights, k.shape[0]))


# =========================================================== E4 heat capacity
def heat_capacity_per_site(energies: torch.Tensor, beta: float,
                           n_sites: int) -> float:
    """``c_V = beta^2 / N_s * Var[V(q)]`` with the ``N-1`` denominator.

    ``energies`` are TOTAL configurational energies ``V(q)``, not per-site
    energies; passing ``V/N_s`` here understates ``c_V`` by ``N_s^2``.
    """
    energies = _check_vector(energies, "energies")
    beta = _positive(beta, "beta")
    n_sites = _positive_int(n_sites, "n_sites")
    if energies.shape[0] < 2:
        raise ValueError("a variance needs at least two samples")
    centered = energies - energies.mean()
    variance = float(((centered * centered).sum() / (energies.shape[0] - 1)).item())
    return beta * beta / n_sites * variance


def heat_capacity_per_site_weighted(energies: torch.Tensor,
                                    weights: torch.Tensor, beta: float,
                                    n_sites: int) -> float:
    """``c_V = beta^2 / N_s * sum_k w_k (V_k - E_w[V])^2 / (1 - sum_k w_k^2)``.

    ``energies`` are TOTAL configurational energies ``V(q)``, not per-site.
    """
    energies = _check_vector(energies, "energies")
    beta = _positive(beta, "beta")
    n_sites = _positive_int(n_sites, "n_sites")
    w = _normalized_weights(weights, energies.shape[0]).to(energies.device)
    centered = energies - _weighted_sum(energies, w)
    variance = float(((w.to(energies.dtype) * centered * centered).sum()
                      / _reliability_denominator(w)).item())
    return beta * beta / n_sites * variance


# ========================================================= E4 Binder cumulant
def binder_cumulant(m: torch.Tensor) -> float:
    """``U_4 = 1 - E[||m||^4] / (2 E[||m||^2]^2)``, the O(2) convention.

    The main text reports only this vector cumulant, appropriate for the
    two-component order parameter: the isotropic Gaussian limit gives
    ``E[||m||^4] = 2 E[||m||^2]^2`` and hence ``U_4 = 0``, while a single
    frozen direction gives ``U_4 = 1/2``.
    """
    m = _check_order_parameter(m)
    squared = (m * m).sum(dim=1)
    second = squared.mean()
    fourth = (squared * squared).mean()
    denominator = 2.0 * float(second.item()) ** 2
    if denominator <= 0.0:
        raise ValueError(
            "E[||m||^2] is zero; the Binder cumulant is undefined")
    return 1.0 - float(fourth.item()) / denominator


def binder_cumulant_weighted(m: torch.Tensor, weights: torch.Tensor) -> float:
    """``U_4 = 1 - E_w[||m||^4] / (2 E_w[||m||^2]^2)`` with normalized weights."""
    m = _check_order_parameter(m)
    w = _normalized_weights(weights, m.shape[0]).to(m.device)
    squared = (m * m).sum(dim=1)
    second = _weighted_sum(squared, w)
    fourth = _weighted_sum(squared * squared, w)
    denominator = 2.0 * float(second.item()) ** 2
    if denominator <= 0.0:
        raise ValueError(
            "E_w[||m||^2] is zero; the Binder cumulant is undefined")
    return 1.0 - float(fourth.item()) / denominator


def binder_cumulant_component(m_component: torch.Tensor) -> float:
    """``U_4 = 1 - E[m^4] / (3 E[m^2]^2)`` for one Cartesian component of ``m``.

    The per-axis scalar convention, kept for diagnostics only: the main text
    uses the vector :func:`binder_cumulant`.
    """
    m_component = _check_vector(m_component, "m_component")
    squared = m_component * m_component
    second = squared.mean()
    fourth = (squared * squared).mean()
    denominator = 3.0 * float(second.item()) ** 2
    if denominator <= 0.0:
        raise ValueError("E[m^2] is zero; the Binder cumulant is undefined")
    return 1.0 - float(fourth.item()) / denominator


def binder_cumulant_component_weighted(m_component: torch.Tensor,
                                       weights: torch.Tensor) -> float:
    """``U_4 = 1 - E_w[m^4] / (3 E_w[m^2]^2)`` for one Cartesian component."""
    m_component = _check_vector(m_component, "m_component")
    w = _normalized_weights(weights, m_component.shape[0]).to(m_component.device)
    squared = m_component * m_component
    second = _weighted_sum(squared, w)
    fourth = _weighted_sum(squared * squared, w)
    denominator = 3.0 * float(second.item()) ** 2
    if denominator <= 0.0:
        raise ValueError("E_w[m^2] is zero; the Binder cumulant is undefined")
    return 1.0 - float(fourth.item()) / denominator


# ============================================================ observable sets
def e4_observable_set(target, x: torch.Tensor, *, basin_map,
                      site_minima: torch.Tensor, n_phases: int = 4) -> dict:
    """Every E4 observable of one unweighted sample, as a plain dict.

    Keys: ``order_parameter_mean`` ``(2,)``, ``phase_probabilities``
    ``(n_phases,)``, ``phase_outside_mass``, ``energy_per_site_mean``,
    ``energy_per_site_variance``, ``susceptibility`` ``(2, 2)``,
    ``coherence_mean``, ``two_point_correlation`` ``(R+1,)``,
    ``connected_two_point_correlation`` ``(R+1,)``, ``kink_density_mean``,
    ``heat_capacity_per_site``, ``binder_cumulant``.

    Heat capacity is taken from the TOTAL energies ``V = N_s * (V/N_s)``.
    """
    potential = _chain(target)
    _check_configurations(potential, x)
    m = order_parameter(target, x)
    labels = phase_labels(m, basin_map)
    probabilities, outside_mass = phase_probabilities(labels, n_phases)
    per_site = energy_per_site(target, x)
    correlations = two_point_correlation_mean(two_point_correlation(target, x))
    mean_m = order_parameter_mean(m)
    return {
        "order_parameter_mean": mean_m,
        "phase_probabilities": probabilities,
        "phase_outside_mass": outside_mass,
        "energy_per_site_mean": energy_per_site_mean(per_site),
        "energy_per_site_variance": energy_per_site_variance(per_site),
        "susceptibility": susceptibility(m, target.beta, potential.n_sites),
        "coherence_mean": coherence_mean(coherence(target, x)),
        "two_point_correlation": correlations,
        "connected_two_point_correlation": connected_two_point_correlation(
            correlations, mean_m),
        "kink_density_mean": kink_density_mean(
            kink_density(target, x, site_minima)),
        "heat_capacity_per_site": heat_capacity_per_site(
            per_site * potential.n_sites, target.beta, potential.n_sites),
        "binder_cumulant": binder_cumulant(m),
    }


def e4_observable_set_weighted(target, x: torch.Tensor, weights: torch.Tensor,
                               *, basin_map, site_minima: torch.Tensor,
                               n_phases: int = 4) -> dict:
    """The same E4 observables under normalized SNIS weights, same keys.

    With uniform weights this reproduces :func:`e4_observable_set` exactly up to
    floating point.
    """
    potential = _chain(target)
    _check_configurations(potential, x)
    w = _normalized_weights(weights, x.shape[0])
    m = order_parameter(target, x)
    labels = phase_labels(m, basin_map)
    probabilities, outside_mass = phase_probabilities_weighted(labels, w, n_phases)
    per_site = energy_per_site(target, x)
    correlations = two_point_correlation_mean_weighted(
        two_point_correlation(target, x), w)
    mean_m = order_parameter_mean_weighted(m, w)
    return {
        "order_parameter_mean": mean_m,
        "phase_probabilities": probabilities,
        "phase_outside_mass": outside_mass,
        "energy_per_site_mean": energy_per_site_mean_weighted(per_site, w),
        "energy_per_site_variance": energy_per_site_variance_weighted(per_site, w),
        "susceptibility": susceptibility_weighted(m, w, target.beta,
                                                  potential.n_sites),
        "coherence_mean": coherence_mean_weighted(coherence(target, x), w),
        "two_point_correlation": correlations,
        "connected_two_point_correlation": connected_two_point_correlation(
            correlations, mean_m),
        "kink_density_mean": kink_density_mean_weighted(
            kink_density(target, x, site_minima), w),
        "heat_capacity_per_site": heat_capacity_per_site_weighted(
            per_site * potential.n_sites, w, target.beta, potential.n_sites),
        "binder_cumulant": binder_cumulant_weighted(m, w),
    }


# ============================================================== the basin map
class GradientFlowBasinMap2D:
    """One cached gradient-flow basin map for a 2D landscape.

    Both the E3 basins (on the latent CV ``z_{1:2}``) and the E4 phases (on the
    order parameter ``m``) are labelled by this one class, so a sample can never
    be labelled by two different conventions: one grid, one flow integrator, one
    nearest-minimum rule, one out-of-domain sentinel.

    Construction precomputes an ``n_grid x n_grid`` grid over ``[lo, hi]``, runs
    tamed gradient-flow descent from every cell centre, and labels the cell by
    its nearest minimum. Labels are cached to an ``.npz`` together with the
    construction metadata; reuse is permitted only when the saved grid, domain,
    minima, and integrator settings match the request exactly. Legacy label-only
    caches are rejected unless ``allow_legacy_unverified`` is set, and the cache
    file's sha256 is recorded in :meth:`cache_provenance`.

    Points outside ``[lo, hi]`` are labelled ``OUTSIDE_LABEL = -1`` by
    :meth:`assign` and flagged by :meth:`outside`; they are never clamped into a
    boundary cell. Clamping would silently relabel exactly the long-range
    transport these experiments measure, since a phase-to-phase jump can move
    the order parameter far beyond the map domain.
    """

    _CACHE_SCHEMA_VERSION = 1
    _CACHE_METADATA_KEYS = ("n_grid", "lo", "hi", "minima", "dt_flow", "n_flow")

    def __init__(self, grad_fn, minima: torch.Tensor, lo, hi,
                 n_grid: int = 600, device=None, cache: str | None = None,
                 dt_flow: float = 1.5e-4, n_flow: int = 40_000,
                 *, allow_legacy_unverified: bool = False) -> None:
        # dt_flow is set by the stiffest Hessian eigenvalue among the 2D
        # landscapes used here (Muller-Brown ~6e3: dt*lam ~ 0.9 < 2), and the
        # tamed step caps wall gradients at unit displacement.
        if isinstance(n_grid, bool) or int(n_grid) != n_grid or n_grid < 2:
            raise ValueError("n_grid must be an integer >= 2")
        if isinstance(n_flow, bool) or int(n_flow) != n_flow or n_flow < 1:
            raise ValueError("n_flow must be a positive integer")
        if not math.isfinite(float(dt_flow)) or float(dt_flow) <= 0.0:
            raise ValueError("dt_flow must be finite and positive")
        device = resolve_device(device)
        self.device = device
        self.lo = torch.as_tensor(lo, dtype=torch.float64, device=device)
        self.hi = torch.as_tensor(hi, dtype=torch.float64, device=device)
        self.minima = torch.as_tensor(minima, dtype=torch.float64, device=device)
        if (self.lo.shape != (2,) or self.hi.shape != (2,)
                or self.minima.ndim != 2 or self.minima.shape[1] != 2
                or self.minima.shape[0] < 1):
            raise ValueError(
                "lo/hi must be 2-vectors and minima must have shape (K, 2)")
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
            self.labels = self._load_cache(expected, allow_legacy_unverified,
                                           device)
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

    # -- cache -------------------------------------------------------------
    def _load_cache(self, expected: dict, allow_legacy_unverified: bool,
                    device) -> torch.Tensor:
        with np.load(self.cache_path, allow_pickle=False) as data:
            if "labels" not in data.files:
                raise ValueError(
                    f"basin cache {self.cache_path} has no labels array")
            labels = np.asarray(data["labels"])
            self._validate_cached_labels(labels)
            missing = [key for key in self._CACHE_METADATA_KEYS
                       if key not in data.files]
            if "cache_schema_version" not in data.files:
                missing.append("cache_schema_version")
            if missing:
                if not allow_legacy_unverified:
                    raise ValueError(
                        f"legacy/incomplete basin cache {self.cache_path} is "
                        f"missing metadata {sorted(missing)}; refusing "
                        "unverified reuse")
                warnings.warn(
                    f"explicitly loading legacy basin cache {self.cache_path} "
                    "without construction metadata; results are unverified",
                    RuntimeWarning, stacklevel=3)
                self.cache_validation_status = "legacy_unverified"
            else:
                schema = int(np.asarray(data["cache_schema_version"]).item())
                mismatches = []
                if schema != self._CACHE_SCHEMA_VERSION:
                    mismatches.append(
                        f"cache_schema_version={schema} "
                        f"(expected {self._CACHE_SCHEMA_VERSION})")
                for key, expected_value in expected.items():
                    if not np.array_equal(np.asarray(data[key]), expected_value):
                        mismatches.append(key)
                if mismatches:
                    raise ValueError(
                        f"basin cache metadata mismatch for {self.cache_path}: "
                        + ", ".join(mismatches))
                self.cache_validation_status = "validated"
        return torch.as_tensor(labels, dtype=torch.long, device=device)

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
        if labels.size and (labels.min() < 0
                            or labels.max() >= self.minima.shape[0]):
            raise ValueError(
                "basin cache labels lie outside the declared minima")

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

    # -- labelling ---------------------------------------------------------
    def _check_points(self, points: torch.Tensor) -> torch.Tensor:
        if not isinstance(points, torch.Tensor):
            raise TypeError(
                f"points must be a torch.Tensor, got {type(points).__name__}")
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(
                f"points must have shape (N, 2), got {tuple(points.shape)}")
        if not bool(torch.isfinite(points).all().item()):
            raise ValueError(
                "points must be finite; a nonfinite coordinate cannot be "
                "assigned to a basin")
        return points.to(device=self.lo.device, dtype=torch.float64)

    def outside(self, points: torch.Tensor) -> torch.Tensor:
        """Boolean mask of points outside the closed domain ``[lo, hi]``."""
        p = self._check_points(points)
        return ((p < self.lo) | (p > self.hi)).any(dim=1)

    def assign(self, points: torch.Tensor) -> torch.Tensor:
        """Basin label per point, ``OUTSIDE_LABEL = -1`` outside ``[lo, hi]``."""
        p = self._check_points(points)
        outside = ((p < self.lo) | (p > self.hi)).any(dim=1)
        fraction = (p - self.lo) / (self.hi - self.lo)
        ij = torch.clamp((fraction * self.n_grid).long(), 0, self.n_grid - 1)
        labels = self.labels[ij[:, 0], ij[:, 1]]
        return torch.where(outside, torch.full_like(labels, OUTSIDE_LABEL),
                           labels)

    def p_star(self, log_density, n_quad: int = 1200) -> torch.Tensor:
        """Basin masses of ``exp(log_density)`` by grid quadrature over ``[lo, hi]``.

        The unnormalised density is evaluated on an ``n_quad x n_quad`` grid and
        summed per basin; the result is normalised to sum to one over the
        domain, so it is a conditional occupancy given the domain.
        """
        if isinstance(n_quad, bool) or int(n_quad) != n_quad or n_quad < 2:
            raise ValueError("n_quad must be an integer >= 2")
        device = self.lo.device
        xs = torch.linspace(float(self.lo[0]), float(self.hi[0]), int(n_quad),
                            dtype=torch.float64, device=device)
        ys = torch.linspace(float(self.lo[1]), float(self.hi[1]), int(n_quad),
                            dtype=torch.float64, device=device)
        gx, gy = torch.meshgrid(xs, ys, indexing="ij")
        z = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
        logp = log_density(z)
        p = torch.exp(logp - logp.max())
        labels = self.assign(z)
        if bool((labels == OUTSIDE_LABEL).any().item()):
            raise RuntimeError(
                "quadrature nodes were labelled outside their own domain; the "
                "basin map bounds are inconsistent")
        mass = torch.zeros(self.minima.shape[0], dtype=torch.float64,
                           device=device)
        mass.scatter_add_(0, labels, p)
        total = mass.sum()
        if float(total.item()) <= 0.0:
            raise ValueError(
                "the quadrature density has zero mass over the basin domain")
        return mass / total
