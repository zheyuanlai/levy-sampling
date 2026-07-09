"""Metric utilities with explicit names for Phase 16A outputs."""

from __future__ import annotations

import numpy as np


def mode_entropy_metrics(p):
    p = np.asarray(p, dtype=float)
    p = p / np.sum(p)
    p_safe = np.maximum(p, 1e-300)

    H = -np.sum(p_safe * np.log(p_safe))
    K = len(p)

    return {
        "entropy": float(H),
        "entropy_normalized": float(H / np.log(K)),
        "effective_mode_count": float(np.exp(H)),
        "effective_mode_fraction": float(np.exp(H) / K),
    }


def row_normalize_transition_matrix(counts):
    counts = np.asarray(counts, dtype=float)
    rowsum = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, rowsum, out=np.zeros_like(counts), where=rowsum > 0)


def transition_rows_from_counts(counts, labels, method, matrix_kind):
    probs = row_normalize_transition_matrix(counts)
    rows = []
    for i, from_label in enumerate(labels):
        for j, to_label in enumerate(labels):
            rows.append(
                {
                    "method": method,
                    "matrix_kind": matrix_kind,
                    "from_state": from_label,
                    "to_state": to_label,
                    "count": float(counts[i, j]),
                    "probability": float(probs[i, j]),
                }
            )
    return rows


def compute_recorded_phase_transition_matrix(label_series, n_states):
    counts = np.zeros((n_states, n_states), dtype=float)
    if len(label_series) < 2:
        return counts
    prev = np.asarray(label_series[0], dtype=int)
    for cur in label_series[1:]:
        cur = np.asarray(cur, dtype=int)
        for a, b in zip(prev, cur):
            counts[int(a), int(b)] += 1.0
        prev = cur
    return counts


def compute_jump_step_transition_matrix(before_labels, after_labels, n_states):
    """Count label changes across time steps containing at least one jump."""

    before_labels = np.asarray(before_labels, dtype=int)
    after_labels = np.asarray(after_labels, dtype=int)
    counts = np.zeros((n_states, n_states), dtype=float)
    for a, b in zip(before_labels, after_labels):
        counts[int(a), int(b)] += 1.0
    return counts


def compute_jump_event_transition_matrix(before_labels, after_labels, n_states):
    """Backward-compatible alias for jump-step transition counts."""

    return compute_jump_step_transition_matrix(before_labels, after_labels, n_states)


def cost_proxy_fields(n_steps, n_particles, n_atoms, n_theta, n_rho, uses_score=True):
    """Return coarse cost proxies for jump application and score quadrature.

    Raw compound-Poisson methods do not evaluate the Levy-score quadrature, so
    their ``score_cost_proxy`` is zero.  The old ``cost_proxy`` field is kept as
    a backward-compatible alias to ``total_cost_proxy``.
    """

    n_rho = int(n_rho)
    jump_cost = int(n_steps) * int(n_particles)
    score_cost = (
        int(n_steps) * int(n_particles) * int(n_atoms) * int(n_theta) * n_rho
        if uses_score
        else 0
    )
    fields = {
        "n_atoms": int(n_atoms),
        "n_theta": int(n_theta),
        "n_rho": n_rho,
        "rho_n": n_rho,
        "jump_cost_proxy": int(jump_cost),
        "score_cost_proxy": int(score_cost),
        "total_cost_proxy": int(jump_cost + score_cost),
    }
    fields["cost_proxy"] = fields["total_cost_proxy"]
    return fields
