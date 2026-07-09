"""Muller--Brown-specific diagnostics for Phase 16A."""

from __future__ import annotations

import numpy as np


def build_basin_map_labels(
    grid_points,
    grid_shape,
    minima,
    grad_U,
    step=6e-4,
    max_iter=160,
    tol=1e-7,
    nearest_classifier=None,
):
    """Precompute gradient-flow basin labels on a 2D latent grid."""

    grid_points = np.asarray(grid_points, dtype=float)
    z = grid_points.copy()
    converged = np.zeros(z.shape[0], dtype=bool)
    final_norm = np.full(z.shape[0], np.inf, dtype=float)
    for _ in range(int(max_iter)):
        g = np.asarray(grad_U(z), dtype=float)
        gnorm = np.linalg.norm(g, axis=1)
        final_norm = gnorm
        converged |= gnorm < tol
        if np.all(converged):
            break
        active = ~converged
        z[active] = z[active] - step * g[active] / (1.0 + step * gnorm[active, None])
    labels = np.argmin(((z[:, None, :] - np.asarray(minima)[None, :, :]) ** 2).sum(axis=2), axis=1)
    diagnostics = {
        "final_gradient_norm_mean": float(np.mean(final_norm)) if len(final_norm) else 0.0,
        "final_gradient_norm_max": float(np.max(final_norm)) if len(final_norm) else 0.0,
        "basin_label_converged_fraction": float(np.mean(converged)) if len(converged) else 1.0,
        "basin_label_failure_fraction": float(1.0 - np.mean(converged)) if len(converged) else 0.0,
    }
    basin_map_labels = labels.reshape(tuple(grid_shape))
    metadata = dict(diagnostics)
    metadata.update(
        grid_shape=f"{int(grid_shape[0])}x{int(grid_shape[1])}",
        n_grid_points=int(grid_points.shape[0]),
        basin_flow_step=float(step),
        basin_flow_max_iter=int(max_iter),
        basin_flow_tol=float(tol),
    )
    if nearest_classifier is not None:
        nearest = np.asarray(nearest_classifier(grid_points), dtype=int)
        metadata["nearest_minimum_disagreement_rate"] = (
            float(np.mean(labels != nearest)) if len(labels) else 0.0
        )
    return basin_map_labels, metadata


def lookup_basin_map_labels(points, gx, gy, basin_map_labels):
    """Assign latent points by nearest-cell lookup in a precomputed basin map."""

    points = np.asarray(points, dtype=float)
    gx = np.asarray(gx, dtype=float)
    gy = np.asarray(gy, dtype=float)
    labels = np.asarray(basin_map_labels, dtype=int)
    if labels.shape != (len(gy), len(gx)):
        raise ValueError("basin_map_labels must have shape (len(gy), len(gx))")
    ix = np.searchsorted(gx, points[:, 0], side="left")
    iy = np.searchsorted(gy, points[:, 1], side="left")
    ix = np.clip(ix, 0, len(gx) - 1)
    iy = np.clip(iy, 0, len(gy) - 1)
    left = np.maximum(ix - 1, 0)
    down = np.maximum(iy - 1, 0)
    ix = np.where(np.abs(points[:, 0] - gx[left]) <= np.abs(points[:, 0] - gx[ix]), left, ix)
    iy = np.where(np.abs(points[:, 1] - gy[down]) <= np.abs(points[:, 1] - gy[iy]), down, iy)
    return labels[iy, ix].astype(int)


def gradient_flow_label(z0, minima, grad_U, step=1e-3, max_iter=5000, tol=1e-8):
    """Assign a latent point to a basin by gradient descent."""

    z = np.array(z0, dtype=float)

    for _ in range(max_iter):
        g = grad_U(z[None, :])[0] if z.ndim == 1 else grad_U(z)
        if np.linalg.norm(g) < tol:
            break
        z = z - step * g

    dists = np.linalg.norm(minima - z[None, :], axis=1)
    return int(np.argmin(dists))


def gradient_flow_labels(z, minima, grad_U, step=1e-3, max_iter=5000, tol=1e-8):
    z = np.asarray(z, dtype=float)
    return np.array(
        [gradient_flow_label(row, minima, grad_U, step=step, max_iter=max_iter, tol=tol) for row in z],
        dtype=int,
    )


def gradient_flow_labels_with_diagnostics(z, minima, grad_U, step=1e-3, max_iter=5000, tol=1e-8):
    """Assign basins by gradient descent and report convergence diagnostics."""

    z = np.asarray(z, dtype=float)
    labels = []
    final_norms = []
    converged = []
    for row in z:
        cur = np.array(row, dtype=float)
        norm = np.inf
        for _ in range(int(max_iter)):
            g = grad_U(cur[None, :])[0] if cur.ndim == 1 else grad_U(cur)
            norm = float(np.linalg.norm(g))
            if norm < tol:
                break
            cur = cur - step * g
        labels.append(int(np.argmin(np.linalg.norm(minima - cur[None, :], axis=1))))
        final_norms.append(norm)
        converged.append(norm < tol)
    labels = np.array(labels, dtype=int)
    final_norms = np.array(final_norms, dtype=float)
    converged = np.array(converged, dtype=bool)
    nearest = np.argmin(((z[:, None, :] - minima[None, :, :]) ** 2).sum(axis=2), axis=1)
    return labels, {
        "final_gradient_norm_mean": float(np.mean(final_norms)) if len(final_norms) else 0.0,
        "final_gradient_norm_max": float(np.max(final_norms)) if len(final_norms) else 0.0,
        "basin_label_converged_fraction": float(np.mean(converged)) if len(converged) else 1.0,
        "basin_label_failure_fraction": float(1.0 - np.mean(converged)) if len(converged) else 0.0,
        "nearest_minimum_disagreement_rate": float(np.mean(labels != nearest)) if len(labels) else 0.0,
    }


def mass_from_labels(labels, n_states):
    labels = np.asarray(labels, dtype=int)
    return np.bincount(labels, minlength=n_states) / max(len(labels), 1)


def weak_observable_error(values, target_mean):
    return abs(float(np.mean(values)) - float(target_mean))


def round_trip_count(label_series, start_state=0, target_state=None):
    if target_state is None:
        target_state = int(np.max([np.max(x) for x in label_series if len(x)]))
    count = 0
    seen_target = False
    prev_majority = start_state
    for labels in label_series:
        if len(labels) == 0:
            continue
        majority = int(np.bincount(np.asarray(labels, dtype=int)).argmax())
        if majority == target_state:
            seen_target = True
        if seen_target and majority == start_state and prev_majority != start_state:
            count += 1
            seen_target = False
        prev_majority = majority
    return int(count)


def dwell_time_by_majority(label_series, times, n_states):
    times = np.asarray(times, dtype=float)
    dwell = np.zeros(n_states, dtype=float)
    if len(label_series) < 2:
        return dwell
    majorities = []
    for labels in label_series:
        labels = np.asarray(labels, dtype=int)
        majorities.append(int(np.bincount(labels, minlength=n_states).argmax()))
    for k in range(len(majorities) - 1):
        dwell[majorities[k]] += max(0.0, float(times[k + 1] - times[k]))
    return dwell
