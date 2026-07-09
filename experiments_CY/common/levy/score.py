"""Levy-score correction utilities for atom and shell jump laws."""

from __future__ import annotations

import numpy as np

FLOAT_LOG_MAX = float(np.log(np.finfo(float).max) - 2.0)


def _quadrature_nodes(n: int, a: float = 0.0, b: float = 1.0):
    x, w = np.polynomial.legendre.leggauss(int(n))
    nodes = 0.5 * (b - a) * x + 0.5 * (a + b)
    weights = 0.5 * (b - a) * w
    return nodes, weights


def _ratio_from_log(log_ratio, log_clip):
    """Exponentiate a log-ratio and report score-changing clipping.

    ``log_clip=None`` uses only the floating-point overflow guard.  Passing a
    smaller value applies mathematical truncation and is reported separately.
    """

    lr = np.asarray(log_ratio, dtype=float)
    overflow_clip = FLOAT_LOG_MAX
    if log_clip is None:
        effective_clip = overflow_clip
        score_clip_count = 0
    else:
        effective_clip = min(float(log_clip), overflow_clip)
        score_clip_count = int(np.sum((lr < -effective_clip) | (lr > effective_clip)))
    overflow_count = int(np.sum((lr < -overflow_clip) | (lr > overflow_clip)))
    return np.exp(np.clip(lr, -effective_clip, effective_clip)), score_clip_count, overflow_count, effective_clip


def levy_score_atoms(
    z,
    jump,
    energy_fn,
    eps,
    theta_nodes=None,
    theta_weights=None,
    n_theta=12,
    log_clip=60.0,
    score_clip=None,
    return_diagnostics=False,
):
    """Compute the atom-law Levy-score correction.

    ``energy_fn`` returns the potential energy V(z).  The density ratio is
    exp(-(V(z-theta*r)-V(z))/eps).
    """

    z = np.asarray(z, dtype=float)
    if theta_nodes is None or theta_weights is None:
        theta_nodes, theta_weights = _quadrature_nodes(n_theta, 0.0, 1.0)
    theta_nodes = np.asarray(theta_nodes, dtype=float)
    theta_weights = np.asarray(theta_weights, dtype=float)

    S = np.zeros_like(z)
    E0 = energy_fn(z)
    clip_count = 0
    overflow_count = 0
    total_count = 0
    max_log_ratio = 0.0
    effective_log_clip = FLOAT_LOG_MAX

    for r, wm in zip(jump.atoms, jump.weights):
        acc = np.zeros_like(z)
        for theta, wtheta in zip(theta_nodes, theta_weights):
            E_shift = energy_fn(z - theta * r)
            log_ratio = -(E_shift - E0) / eps
            max_log_ratio = max(max_log_ratio, float(np.max(np.abs(log_ratio))))
            ratio, n_clip, n_overflow, effective_log_clip = _ratio_from_log(log_ratio, log_clip)
            clip_count += n_clip
            overflow_count += n_overflow
            total_count += log_ratio.size
            acc += wtheta * ratio[:, None] * r
        S -= jump.lam * wm * acc

    raw = S.copy()
    if score_clip is not None:
        S = np.clip(S, -score_clip, score_clip)

    diagnostics = {
        "score_clip_fraction": float(np.mean(S != raw)) if score_clip is not None else 0.0,
        "logratio_clip_fraction": float(clip_count / max(total_count, 1)),
        "score_changing_logratio_clip_fraction": float(clip_count / max(total_count, 1)),
        "overflow_guard_logratio_clip_fraction": float(overflow_count / max(total_count, 1)),
        "effective_log_clip": float(effective_log_clip),
        "max_score_norm": float(np.max(np.linalg.norm(raw, axis=1))) if raw.size else 0.0,
        "max_log_ratio": float(max_log_ratio),
        "n_theta": int(len(theta_nodes)),
        "n_rho": 1,
        "rho_n": 1,
    }
    return (S, diagnostics) if return_diagnostics else S


def levy_score_edge_shell(
    z,
    jump,
    energy_fn,
    eps,
    theta_nodes,
    theta_weights,
    rho_nodes,
    rho_weights,
    log_clip=60.0,
    score_clip=None,
    return_diagnostics=False,
):
    """Compute the Levy-score correction for shell-thickened edge jumps."""

    z = np.asarray(z, dtype=float)
    theta_nodes = np.asarray(theta_nodes, dtype=float)
    theta_weights = np.asarray(theta_weights, dtype=float)
    rho_nodes = np.asarray(rho_nodes, dtype=float)
    rho_weights = np.asarray(rho_weights, dtype=float)

    S = np.zeros_like(z)
    E0 = energy_fn(z)
    clip_count = 0
    overflow_count = 0
    total_count = 0
    max_log_ratio = 0.0
    effective_log_clip = FLOAT_LOG_MAX

    for r0, wm in zip(jump.centers, jump.weights):
        norm = np.linalg.norm(r0)
        u = r0 / max(norm, 1e-14)

        for rho, wrho in zip(rho_nodes, rho_weights):
            r = r0 + rho * u
            acc = np.zeros_like(z)

            for theta, wtheta in zip(theta_nodes, theta_weights):
                E_shift = energy_fn(z - theta * r)
                log_ratio = -(E_shift - E0) / eps
                max_log_ratio = max(max_log_ratio, float(np.max(np.abs(log_ratio))))
                ratio, n_clip, n_overflow, effective_log_clip = _ratio_from_log(log_ratio, log_clip)
                clip_count += n_clip
                overflow_count += n_overflow
                total_count += log_ratio.size
                acc += wtheta * ratio[:, None] * r

            S -= jump.lam * wm * wrho * acc

    raw = S.copy()
    if score_clip is not None:
        S = np.clip(S, -score_clip, score_clip)

    diagnostics = {
        "score_clip_fraction": float(np.mean(S != raw)) if score_clip is not None else 0.0,
        "logratio_clip_fraction": float(clip_count / max(total_count, 1)),
        "score_changing_logratio_clip_fraction": float(clip_count / max(total_count, 1)),
        "overflow_guard_logratio_clip_fraction": float(overflow_count / max(total_count, 1)),
        "effective_log_clip": float(effective_log_clip),
        "max_score_norm": float(np.max(np.linalg.norm(raw, axis=1))) if raw.size else 0.0,
        "max_log_ratio": float(max_log_ratio),
        "h_shell": float(getattr(jump, "h_shell", 0.0)),
        "n_theta": int(len(theta_nodes)),
        "n_rho": int(len(rho_nodes)),
        "rho_n": int(len(rho_nodes)),
    }

    return (S, diagnostics) if return_diagnostics else S
