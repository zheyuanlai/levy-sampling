#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from flmc_utils import step_flmc_nd
from high_dim_output.benchmark_metrics import (
    BENCHMARK_METHODS,
    compute_benchmark_metrics,
    get_default_benchmark_config,
    init_benchmark_history,
    make_metric_rng,
    plot_benchmark_metrics_figure,
    save_benchmark_metadata_json,
    save_benchmark_metrics_csv,
)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(THIS_DIR, "lennard_jones_output")

N_ATOMS = 7
SPATIAL_DIM = 2
FLAT_DIM = N_ATOMS * SPATIAL_DIM
DESCRIPTOR_DIM = N_ATOMS * (N_ATOMS - 1) // 2
PAIR_I, PAIR_J = np.triu_indices(N_ATOMS, k=1)

DEFAULT_T_STAR = 0.05
DEFAULT_SIGMA_NOISE = float(np.sqrt(2.0 * DEFAULT_T_STAR))
DEFAULT_R_MIN = 1e-3
DEFAULT_CLUSTER_MAX_PAIR_DISTANCE = 3.6

# Canonical ordered list of EMC checkpoint metrics.
# Both run_simulation() and main() reference this constant so that the
# metric name lists stay synchronized when metrics are added or removed.
_EMC_METRIC_NAMES = [
    "lj7_emc",
    "lj7_assigned_fraction",
    "lj7_hard_mode_coverage",
    "lj7_conditional_mode_entropy",
]


@dataclass(frozen=True)
class LJ7Paths:
    output_dir: str
    minima_raw: str
    minima_descriptors: str
    minima_energies: str
    minima_metadata: str
    minima_summary: str
    reference_descriptors: str
    reference_metadata: str
    benchmark_base: str
    benchmark_metadata: str
    emc_benchmark_base: str
    mode_occupancy_final_by_seed: str
    mode_occupancy_final_summary: str


def get_lj7_paths(output_dir: str = DEFAULT_OUTPUT_DIR) -> LJ7Paths:
    output_dir = os.path.abspath(output_dir)
    return LJ7Paths(
        output_dir=output_dir,
        minima_raw=os.path.join(output_dir, "minima_raw.npy"),
        minima_descriptors=os.path.join(output_dir, "minima_descriptors.npy"),
        minima_energies=os.path.join(output_dir, "minima_energies.csv"),
        minima_metadata=os.path.join(output_dir, "minima_metadata.json"),
        minima_summary=os.path.join(output_dir, "minima_summary.csv"),
        reference_descriptors=os.path.join(output_dir, "reference_descriptors_lj7_2d.npy"),
        reference_metadata=os.path.join(output_dir, "reference_metadata_lj7_2d.json"),
        benchmark_base=os.path.join(output_dir, "benchmark_metrics_lennard_jones_n7_d2"),
        benchmark_metadata=os.path.join(output_dir, "metrics_benchmark_lennard_jones_n7_d2.json"),
        emc_benchmark_base=os.path.join(output_dir, "emc_metrics_lennard_jones_n7_d2"),
        mode_occupancy_final_by_seed=os.path.join(output_dir, "mode_occupancy_final_by_seed.csv"),
        mode_occupancy_final_summary=os.path.join(output_dir, "mode_occupancy_final_summary.csv"),
    )


def ensure_output_dir(output_dir: str = DEFAULT_OUTPUT_DIR) -> LJ7Paths:
    paths = get_lj7_paths(output_dir)
    os.makedirs(paths.output_dir, exist_ok=True)
    return paths


def resolve_unique_output_base(base_path: str) -> str:
    """Avoid overwriting an existing benchmark figure/CSV bundle."""
    if not (
        os.path.exists(f"{base_path}.png")
        or os.path.exists(f"{base_path}.pdf")
        or os.path.exists(f"{base_path}.csv")
    ):
        return base_path

    suffix = 1
    while True:
        candidate = f"{base_path}_run{suffix:02d}"
        if not (
            os.path.exists(f"{candidate}.png")
            or os.path.exists(f"{candidate}.pdf")
            or os.path.exists(f"{candidate}.csv")
        ):
            return candidate
        suffix += 1


def resolve_unique_json_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    suffix = 1
    while True:
        candidate = f"{stem}_run{suffix:02d}{ext}"
        if not os.path.exists(candidate):
            return candidate
        suffix += 1


def _ensure_state_batch(R: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(R, dtype=float)
    if arr.ndim == 2 and arr.shape == (N_ATOMS, SPATIAL_DIM):
        return arr[None, ...], True
    if arr.ndim == 3 and arr.shape[1:] == (N_ATOMS, SPATIAL_DIM):
        return arr, False
    raise ValueError(f"Expected shape ({N_ATOMS}, {SPATIAL_DIM}) or (batch, {N_ATOMS}, {SPATIAL_DIM}).")


def flatten_states(R: np.ndarray) -> np.ndarray:
    batch, squeeze = _ensure_state_batch(R)
    flat = batch.reshape(batch.shape[0], FLAT_DIM)
    return flat[0] if squeeze else flat


def unflatten_states(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1 and arr.size == FLAT_DIM:
        return arr.reshape(N_ATOMS, SPATIAL_DIM)
    if arr.ndim == 2 and arr.shape[1] == FLAT_DIM:
        return arr.reshape(arr.shape[0], N_ATOMS, SPATIAL_DIM)
    raise ValueError(f"Expected shape ({FLAT_DIM},) or (batch, {FLAT_DIM}).")


def remove_center_of_mass(R: np.ndarray) -> np.ndarray:
    """
    Project the free cluster into the COM=0 gauge.

    This is mandatory for the benchmark because the raw LJ7(2d) system is
    translation-invariant.
    """
    batch, squeeze = _ensure_state_batch(R)
    centered = batch - np.mean(batch, axis=1, keepdims=True)
    return centered[0] if squeeze else centered


def sample_com_free_directions(
    rng: np.random.Generator,
    n_dirs: int,
) -> np.ndarray:
    """
    Sample approximately isotropic directions in the COM=0 subspace.
    """
    directions = []
    while len(directions) < int(n_dirs):
        raw = rng.standard_normal((int(n_dirs), N_ATOMS, SPATIAL_DIM))
        raw = remove_center_of_mass(raw)
        norms = np.linalg.norm(raw.reshape(raw.shape[0], -1), axis=1)
        mask = norms > 1e-12
        if not np.any(mask):
            continue
        raw = raw[mask]
        norms = norms[mask][:, None, None]
        directions.extend((raw / norms).tolist())
    return np.asarray(directions[: int(n_dirs)], dtype=float)


def safe_pair_distances(R: np.ndarray, r_min: float = DEFAULT_R_MIN) -> np.ndarray:
    batch, squeeze = _ensure_state_batch(R)
    disp = batch[:, PAIR_I, :] - batch[:, PAIR_J, :]
    norms = np.linalg.norm(disp, axis=2)
    safe = np.maximum(norms, float(r_min))
    return safe[0] if squeeze else safe


def sorted_pair_distance_descriptor(R: np.ndarray, r_min: float = DEFAULT_R_MIN) -> np.ndarray:
    dists = safe_pair_distances(R, r_min=r_min)
    desc = np.sort(dists, axis=-1)
    if desc.ndim == 1:
        if desc.shape[0] != DESCRIPTOR_DIM:
            raise ValueError("Descriptor dimension mismatch.")
        return desc
    if desc.shape[1] != DESCRIPTOR_DIM:
        raise ValueError("Descriptor dimension mismatch.")
    return desc


def total_energy(R: np.ndarray, r_min: float = DEFAULT_R_MIN) -> np.ndarray:
    batch, squeeze = _ensure_state_batch(R)
    dists = safe_pair_distances(batch, r_min=r_min)
    inv_r6 = dists ** (-6.0)
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6, axis=1)
    return float(energy[0]) if squeeze else energy


def grad_energy(R: np.ndarray, r_min: float = DEFAULT_R_MIN) -> np.ndarray:
    batch, squeeze = _ensure_state_batch(R)
    disp = batch[:, PAIR_I, :] - batch[:, PAIR_J, :]
    dists = safe_pair_distances(batch, r_min=r_min)
    inv_r2 = dists ** (-2.0)
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    coef = 24.0 * (inv_r6 - 2.0 * inv_r12) * inv_r2
    pair_grad = coef[:, :, None] * disp

    grad = np.zeros_like(batch)
    for idx, (i, j) in enumerate(zip(PAIR_I, PAIR_J)):
        grad[:, i, :] += pair_grad[:, idx, :]
        grad[:, j, :] -= pair_grad[:, idx, :]

    grad = remove_center_of_mass(grad)
    return grad[0] if squeeze else grad


def total_energy_flat(X: np.ndarray, r_min: float = DEFAULT_R_MIN) -> np.ndarray:
    return total_energy(remove_center_of_mass(unflatten_states(X)), r_min=r_min)


def grad_energy_flat(X: np.ndarray, r_min: float = DEFAULT_R_MIN) -> np.ndarray:
    grad = grad_energy(remove_center_of_mass(unflatten_states(X)), r_min=r_min)
    return flatten_states(grad)


def gradU_energy_flat(
    X: np.ndarray,
    temperature: float,
    r_min: float = DEFAULT_R_MIN,
) -> np.ndarray:
    """
    Paper-faithful FLMC mapping for the Lennard-Jones benchmark.

    Target:
        pi(R) ∝ exp(-E(R) / T_star)
    Therefore:
        U(R) = -log pi(R) = E(R) / T_star + const
        gradU(R) = gradE(R) / T_star
    and the FLMC drift is
        -c_alpha * gradU(R) = -(c_alpha / T_star) gradE(R).
    """
    return grad_energy_flat(X, r_min=r_min) / float(temperature)


def tamed_increment(drift: np.ndarray, dt: float) -> np.ndarray:
    batch, squeeze = _ensure_state_batch(drift)
    norms = np.linalg.norm(batch.reshape(batch.shape[0], -1), axis=1)[:, None, None]
    incr = float(dt) * batch / (1.0 + float(dt) * norms)
    return incr[0] if squeeze else incr


def _objective_value_and_grad(x_flat: np.ndarray, r_min: float) -> tuple[float, np.ndarray]:
    state = remove_center_of_mass(unflatten_states(x_flat))
    value = total_energy(state, r_min=r_min)
    grad = flatten_states(grad_energy(state, r_min=r_min))
    return float(value), np.asarray(grad, dtype=float)


def local_minimize(
    R0: np.ndarray,
    r_min: float = DEFAULT_R_MIN,
    maxiter: int = 800,
    gtol: float = 1e-8,
    ftol: float = 1e-12,
) -> dict:
    """
    Local minimization wrapper around SciPy L-BFGS-B.
    """
    x0 = flatten_states(remove_center_of_mass(R0))
    result = minimize(
        fun=lambda x: _objective_value_and_grad(x, r_min=r_min),
        x0=x0,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(maxiter),
            "gtol": float(gtol),
            "ftol": float(ftol),
            "maxls": 50,
        },
    )
    coords = remove_center_of_mass(unflatten_states(result.x))
    return {
        "coords": coords,
        "energy": float(total_energy(coords, r_min=r_min)),
        "descriptor": sorted_pair_distance_descriptor(coords, r_min=r_min),
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nit": int(getattr(result, "nit", 0)),
        "nfev": int(getattr(result, "nfev", 0)),
    }


def random_initial_configuration(
    rng: np.random.Generator,
    init_scale: float = 1.2,
) -> np.ndarray:
    """
    Diverse random starts for minima discovery.
    """
    scale = float(init_scale) * np.exp(float(rng.normal(loc=0.0, scale=0.35)))
    if rng.random() < 0.5:
        R = rng.standard_normal((N_ATOMS, SPATIAL_DIM))
    else:
        ang = rng.uniform(0.0, 2.0 * np.pi, size=N_ATOMS)
        rad = scale * (0.6 + 0.8 * rng.random(N_ATOMS))
        R = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
    R += 0.15 * scale * rng.standard_normal((N_ATOMS, SPATIAL_DIM))
    return remove_center_of_mass(R)


def _descriptor_is_duplicate(
    descriptor: np.ndarray,
    existing: list[dict],
    minima_tol: float,
) -> int | None:
    for idx, item in enumerate(existing):
        dist = np.linalg.norm(descriptor - item["descriptor"])
        if dist < float(minima_tol):
            return idx
    return None


def is_compact_cluster_descriptor(
    descriptor: np.ndarray,
    cluster_max_pair_distance: float = DEFAULT_CLUSTER_MAX_PAIR_DISTANCE,
) -> bool:
    """
    Reject nearly dissociated pseudo-minima during minima preprocessing.

    The free LJ7 cluster is not coercive in the relative coordinates, so local
    optimization can terminate at diffuse configurations with tiny gradients.
    EMC should be anchored to the four bonded minima only, hence the compact
    cluster filter in descriptor space.
    """
    return float(np.max(descriptor)) <= float(cluster_max_pair_distance)


def load_or_discover_lj7_minima(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    cluster_max_pair_distance: float = DEFAULT_CLUSTER_MAX_PAIR_DISTANCE,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Load the minima bundle if it already exists on disk, otherwise run
    discover_four_minima() and save it.

    This is the canonical entry point for EMC preprocessing.  It wraps
    discover_four_minima() so that callers do not need to remember the
    load/discover branching logic themselves.

    Parameters
    ----------
    output_dir : str
        Directory that contains (or will contain) the minima bundle files.
    cluster_max_pair_distance : float
        Passed to discover_four_minima() when discovery is required.
    force : bool
        If True, ignore cached files and rediscover minima from scratch.
    verbose : bool
        Print progress messages during discovery.

    Returns
    -------
    dict with keys 'paths', 'raw', 'descriptors', 'energies', 'metadata'.
    """
    return discover_four_minima(
        output_dir=output_dir,
        cluster_max_pair_distance=cluster_max_pair_distance,
        force=force,
        verbose=verbose,
    )


def assign_descriptors_to_minima(
    descriptors: np.ndarray,
    minima_descriptors: np.ndarray,
    cluster_max_pair_distance: float,
    assignment_radius: float | None = None,
) -> dict:
    """
    Assign each sample descriptor to one of the four LJ7(2d) minima.

    The assignment is done in the sorted pair-distance descriptor space (R^21)
    that is invariant to translation, rotation, and atom relabeling.

    Steps:
    1. Compact-cluster filter: if max(descriptor) > cluster_max_pair_distance
       the sample is marked unassigned (label = -1) because it is likely a
       nearly dissociated configuration that does not belong to any bonded well.
    2. Nearest-minimum distance: compute Euclidean distance to each of the
       four minima descriptors.
    3. Radius test: assign to the nearest minimum only if the distance is
       within assignment_radius.  Samples between wells are marked unassigned
       rather than forced into the nearest basin.

    Parameters
    ----------
    descriptors : np.ndarray of shape (n, 21)
        Sorted pair-distance descriptors of the current sampler ensemble.
    minima_descriptors : np.ndarray of shape (4, 21)
        Sorted pair-distance descriptors of the four compact minima.
    cluster_max_pair_distance : float
        Compact-cluster filter threshold (same value used during discovery).
    assignment_radius : float or None
        Radius in descriptor space for hard assignment.  If None, chosen
        automatically as:
            assignment_radius = 0.35 * min_{a != b} ||d_a - d_b||_2
        The factor 0.35 places the boundary at 35 % of the minimum separation
        between any two minima, ensuring non-overlapping Voronoi-like balls.

    Returns
    -------
    dict with keys:
        "labels"            : int array of shape (n,), values in {-1,0,1,2,3}
        "nearest_distances" : float array of shape (n,), Euclidean distance to
                              the nearest minimum (-1 entries have inf or the
                              distance to nearest, depending on filter outcome)
        "assignment_radius" : float (auto-computed or as provided)
        "assigned_fraction" : float in [0, 1]
        "counts"            : int array of shape (4,), per-mode assigned counts
    """
    descriptors = np.asarray(descriptors, dtype=float)
    minima_descriptors = np.asarray(minima_descriptors, dtype=float)
    n = descriptors.shape[0]
    n_modes = minima_descriptors.shape[0]

    # Auto assignment radius: 35 % of minimum inter-minimum separation
    if assignment_radius is None:
        sep = float("inf")
        for a in range(n_modes):
            for b in range(n_modes):
                if a != b:
                    d = float(np.linalg.norm(minima_descriptors[a] - minima_descriptors[b]))
                    if d < sep:
                        sep = d
        assignment_radius = 0.35 * sep

    labels = np.full(n, -1, dtype=int)
    nearest_distances = np.full(n, float("inf"), dtype=float)

    for i in range(n):
        desc = descriptors[i]
        # Compact-cluster filter: reject near-dissociated configurations
        if float(np.max(desc)) > float(cluster_max_pair_distance):
            continue
        # Distance to each minimum
        dists = np.array(
            [float(np.linalg.norm(desc - minima_descriptors[k])) for k in range(n_modes)]
        )
        nearest_idx = int(np.argmin(dists))
        nearest_dist = float(dists[nearest_idx])
        nearest_distances[i] = nearest_dist
        if nearest_dist <= float(assignment_radius):
            labels[i] = nearest_idx

    assigned_fraction = float(np.sum(labels >= 0)) / float(n) if n > 0 else 0.0
    counts = np.array(
        [int(np.sum(labels == k)) for k in range(n_modes)], dtype=int
    )
    return {
        "labels": labels,
        "nearest_distances": nearest_distances,
        "assignment_radius": float(assignment_radius),
        "assigned_fraction": assigned_fraction,
        "counts": counts,
    }


def quench_and_assign_states_to_minima(
    states: np.ndarray,
    minima_descriptors: np.ndarray,
    r_min: float = DEFAULT_R_MIN,
    cluster_max_pair_distance: float = DEFAULT_CLUSTER_MAX_PAIR_DISTANCE,
    assignment_radius: float | None = None,
    local_maxiter: int = 400,
    local_gtol: float = 1e-6,
) -> dict:
    """
    Assign sampler states to the four LJ7(2d) minima via local minimization.

    Each state is quenched (locally minimized from its current position) before
    assignment.  The quenched descriptor — not the raw sample descriptor — is
    then compared to the four canonical minima descriptors.  This makes
    assignment independent of the instantaneous thermal fluctuations and instead
    reflects which basin of attraction the sample currently occupies.

    Why quench-based assignment is scientifically stronger than raw descriptor
    nearest-ball assignment:
    - The sorted pair-distance descriptor of a thermally excited sample can sit
      far from every minimum descriptor even when the sample is firmly inside one
      basin.  With a finite assignment radius, such samples are marked unassigned
      (label = -1), underestimating well occupancy.
    - Quenching maps each sample deterministically to the nearest local minimum
      of the PES, which is the correct definition of "which basin does this point
      belong to."  The quenched descriptor is much closer to its canonical
      minimum than the raw descriptor, so assignment is more reliable.
    - Basin boundaries in coordinate space do not correspond to spheres in
      descriptor space, so sphere-based assignment is geometry-blind.  Quenching
      respects the true geometry of the PES.

    Performance note:
    This function calls local_minimize() once per sample.  For n=256 samples at
    each of 21 checkpoints × 4 methods × 5 seeds, roughly 108 k quenches are
    run.  Each quench is a 14-D L-BFGS-B minimization with maxiter=400; in
    practice, well-initialized samples converge in far fewer iterations.  Runtime
    is typically 1–5 s per checkpoint depending on hardware.

    Parameters
    ----------
    states : np.ndarray of shape (n, 7, 2) or (n, 14)
        Current sampler ensemble in coordinate space.
    minima_descriptors : np.ndarray of shape (4, 21)
        Sorted pair-distance descriptors of the four compact minima.
    r_min : float
        Soft lower bound on pair distances used in local_minimize().
    cluster_max_pair_distance : float
        Compact-cluster filter: quenched configurations with max pair distance
        above this threshold are marked unassigned.
    assignment_radius : float or None
        Radius in quenched-descriptor space for well assignment.  If None,
        auto-computed as 0.35 × min inter-minimum separation (same rule as
        assign_descriptors_to_minima).
    local_maxiter : int
        Maximum L-BFGS-B iterations per quench.
    local_gtol : float
        Gradient tolerance for the L-BFGS-B quench.

    Returns
    -------
    dict with keys:
        "labels"               : int array of shape (n,), values in {-1,0,1,2,3}
        "nearest_distances"    : float array of shape (n,)
        "assignment_radius"    : float
        "assigned_fraction"    : float in [0, 1]
        "counts"               : int array of shape (4,)
        "quenched_descriptors" : np.ndarray of shape (n, 21)
        "quenched_energies"    : np.ndarray of shape (n,)
    """
    # Accept either (n, N_ATOMS, SPATIAL_DIM) or (n, FLAT_DIM)
    arr = np.asarray(states, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == FLAT_DIM:
        arr = unflatten_states(arr)
    elif arr.ndim != 3 or arr.shape[1:] != (N_ATOMS, SPATIAL_DIM):
        raise ValueError(
            f"states must have shape (n, {N_ATOMS}, {SPATIAL_DIM}) or (n, {FLAT_DIM})."
        )
    n = arr.shape[0]
    minima_descriptors = np.asarray(minima_descriptors, dtype=float)
    n_modes = minima_descriptors.shape[0]

    # Auto assignment radius
    if assignment_radius is None:
        sep = float("inf")
        for a in range(n_modes):
            for b in range(n_modes):
                if a != b:
                    d = float(np.linalg.norm(minima_descriptors[a] - minima_descriptors[b]))
                    if d < sep:
                        sep = d
        assignment_radius = 0.35 * sep

    quenched_descriptors = np.zeros((n, int(DESCRIPTOR_DIM)), dtype=float)
    quenched_energies = np.zeros(n, dtype=float)
    labels = np.full(n, -1, dtype=int)
    nearest_distances = np.full(n, float("inf"), dtype=float)

    for i in range(n):
        result = local_minimize(
            arr[i],
            r_min=float(r_min),
            maxiter=int(local_maxiter),
            gtol=float(local_gtol),
        )
        q_desc = result["descriptor"]
        q_energy = result["energy"]
        quenched_descriptors[i] = q_desc
        quenched_energies[i] = q_energy

        # Compact-cluster filter on quenched descriptor
        if float(np.max(q_desc)) > float(cluster_max_pair_distance):
            continue

        dists = np.array(
            [float(np.linalg.norm(q_desc - minima_descriptors[k])) for k in range(n_modes)]
        )
        nearest_idx = int(np.argmin(dists))
        nearest_dist = float(dists[nearest_idx])
        nearest_distances[i] = nearest_dist
        if nearest_dist <= float(assignment_radius):
            labels[i] = nearest_idx

    assigned_fraction = float(np.sum(labels >= 0)) / float(n) if n > 0 else 0.0
    counts = np.array(
        [int(np.sum(labels == k)) for k in range(n_modes)], dtype=int
    )
    return {
        "labels": labels,
        "nearest_distances": nearest_distances,
        "assignment_radius": float(assignment_radius),
        "assigned_fraction": assigned_fraction,
        "counts": counts,
        "quenched_descriptors": quenched_descriptors,
        "quenched_energies": quenched_energies,
    }


def compute_emc_from_labels(labels: np.ndarray, n_modes: int = 4) -> float:
    """
    Entropic Mode Coverage (EMC) scalar in [0, 1].

    Formula:
        Let f   = assigned_fraction = (# samples with label >= 0) / n
        Let p_k = (# samples in mode k among assigned) / (# assigned)
        Let H   = -sum_k p_k log(p_k)   (Shannon entropy, sum over nonzero p_k)
        EMC     = f * exp(H) / n_modes

    Interpretation:
    - exp(H) / n_modes lies in [1/n_modes, 1]:
        * Equals 1 when assigned mass is uniform across all modes.
        * Equals 1/n_modes when all assigned mass collapses to one mode.
    - Multiplying by f penalises methods that scatter many samples outside
      all bonded wells (non-compact configurations), reflecting the fact that
      unassigned samples contribute neither to exploration nor to coverage.
    - EMC = 1 iff every sample is assigned AND distributed uniformly.
    - EMC = 0 iff no sample is assigned.

    Scientific note:
    EMC is an *exploration* metric that measures diversity across bonded wells
    under the compact-cluster constraint.  It is NOT a standalone equilibrium-
    fidelity metric and does not verify within-well density accuracy.  Report
    together with Sinkhorn / MMD.
    """
    labels = np.asarray(labels, dtype=int)
    n = labels.size
    if n == 0:
        return 0.0
    n_assigned = int(np.sum(labels >= 0))
    if n_assigned == 0:
        return 0.0
    assigned_fraction = float(n_assigned) / float(n)
    assigned_labels = labels[labels >= 0]
    counts = np.array([int(np.sum(assigned_labels == k)) for k in range(n_modes)])
    p = counts.astype(float) / float(n_assigned)
    nonzero = p[p > 0.0]
    H = float(-np.sum(nonzero * np.log(nonzero)))
    return float(assigned_fraction * np.exp(H) / float(n_modes))


def compute_hard_mode_coverage(
    labels: np.ndarray,
    n_modes: int = 4,
    min_fraction: float = 0.01,
) -> float:
    """
    Hard mode coverage: fraction of modes with empirical occupancy >= min_fraction.

    A mode is 'covered' when at least min_fraction of *all* samples (not just
    assigned ones) land in it.  This is deliberately conservative: a method
    that assigns 99 % of its mass to one well and 1 % elsewhere gets partial
    credit only if the secondary wells each exceed the threshold individually.

    Parameters
    ----------
    labels : int array of shape (n,)
        As returned by assign_descriptors_to_minima.  Unassigned samples
        (label = -1) do not count toward any mode.
    n_modes : int
        Total number of modes.
    min_fraction : float
        Threshold fraction over all n samples required to count a mode as covered.

    Returns
    -------
    float in [0, 1]: covered_modes / n_modes
    """
    labels = np.asarray(labels, dtype=int)
    n = labels.size
    if n == 0:
        return 0.0
    covered = sum(
        1 for k in range(n_modes)
        if float(np.sum(labels == k)) / float(n) >= float(min_fraction)
    )
    return float(covered) / float(n_modes)


def compute_conditional_mode_entropy(labels: np.ndarray, n_modes: int = 4) -> float:
    """
    Conditional mode entropy: exp(H) / n_modes conditioned on assigned samples.

    This is the intra-basin diversity component of EMC, isolated from the
    assigned-fraction penalty.  It answers the question: given that a sample
    IS assigned to a well, how uniformly are the assigned samples spread across
    the four wells?

    Formula:
        Let p_k = (# assigned in mode k) / (# assigned)
        H = -sum_k p_k log(p_k)   (over nonzero p_k)
        return exp(H) / n_modes

    - Returns 1.0 when assigned mass is exactly uniform across all modes.
    - Returns 1/n_modes when all assigned mass collapses to one mode.
    - Returns 0.0 when no sample is assigned.

    Relationship to EMC:
        EMC = assigned_fraction * compute_conditional_mode_entropy(labels)
    """
    labels = np.asarray(labels, dtype=int)
    n_assigned = int(np.sum(labels >= 0))
    if n_assigned == 0:
        return 0.0
    assigned_labels = labels[labels >= 0]
    counts = np.array([int(np.sum(assigned_labels == k)) for k in range(n_modes)])
    p = counts.astype(float) / float(n_assigned)
    nonzero = p[p > 0.0]
    H = float(-np.sum(nonzero * np.log(nonzero)))
    return float(np.exp(H) / float(n_modes))


def compute_mode_occupancy(labels: np.ndarray, n_modes: int = 4) -> np.ndarray:
    """
    Return empirical fraction of all samples (assigned or not) in each mode.

    Unassigned samples (label = -1) are not counted in any mode, so the
    returned array will not sum to 1 when some samples are unassigned.

    Returns
    -------
    np.ndarray of shape (n_modes,) with values in [0, 1].
    """
    labels = np.asarray(labels, dtype=int)
    n = labels.size
    if n == 0:
        return np.zeros(n_modes, dtype=float)
    return np.array(
        [float(np.sum(labels == k)) / float(n) for k in range(n_modes)],
        dtype=float,
    )


def save_minima_bundle(
    minima: list[dict],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    metadata: dict | None = None,
) -> LJ7Paths:
    paths = ensure_output_dir(output_dir)
    minima_sorted = sorted(
        minima,
        key=lambda item: (float(item["energy"]), item["descriptor"].tolist()),
    )
    raw = np.stack([item["coords"] for item in minima_sorted], axis=0)
    descriptors = np.stack([item["descriptor"] for item in minima_sorted], axis=0)

    np.save(paths.minima_raw, raw)
    np.save(paths.minima_descriptors, descriptors)

    with open(paths.minima_energies, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "energy"])
        for idx, item in enumerate(minima_sorted):
            writer.writerow([idx, float(item["energy"])])

    payload = {
        "n_minima": int(raw.shape[0]),
        "descriptor_dim": int(descriptors.shape[1]),
    }
    if metadata:
        payload.update(metadata)
    with open(paths.minima_metadata, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return paths


def load_minima_bundle(output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    paths = get_lj7_paths(output_dir)
    with open(paths.minima_metadata, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    raw = np.load(paths.minima_raw)
    descriptors = np.load(paths.minima_descriptors)
    energies = []
    with open(paths.minima_energies, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            energies.append(float(row["energy"]))
    return {
        "paths": paths,
        "raw": raw,
        "descriptors": descriptors,
        "energies": np.asarray(energies, dtype=float),
        "metadata": metadata,
    }


def discover_four_minima(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    seed: int = 2026,
    init_scale: float = 1.2,
    r_min: float = DEFAULT_R_MIN,
    minima_tol: float = 1e-4,
    cluster_max_pair_distance: float = DEFAULT_CLUSTER_MAX_PAIR_DISTANCE,
    max_attempts: int = 4000,
    no_new_minima_patience: int = 300,
    local_maxiter: int = 800,
    local_gtol: float = 1e-8,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    paths = ensure_output_dir(output_dir)
    if (
        not force
        and os.path.exists(paths.minima_raw)
        and os.path.exists(paths.minima_descriptors)
        and os.path.exists(paths.minima_energies)
        and os.path.exists(paths.minima_metadata)
    ):
        return load_minima_bundle(output_dir)

    rng = np.random.default_rng(int(seed))
    unique_minima: list[dict] = []
    extra_distinct = []
    attempts_since_new = 0
    total_attempts = 0
    rejected_noncompact = 0

    for attempt_idx in range(int(max_attempts)):
        total_attempts += 1
        R0 = random_initial_configuration(rng, init_scale=init_scale)
        result = local_minimize(
            R0,
            r_min=r_min,
            maxiter=local_maxiter,
            gtol=local_gtol,
        )
        if not is_compact_cluster_descriptor(
            result["descriptor"],
            cluster_max_pair_distance=cluster_max_pair_distance,
        ):
            rejected_noncompact += 1
            continue
        duplicate_idx = _descriptor_is_duplicate(
            result["descriptor"],
            unique_minima,
            minima_tol=minima_tol,
        )
        if duplicate_idx is None:
            unique_minima.append(result)
            attempts_since_new = 0
            if verbose:
                print(
                    f"[minima] new minimum #{len(unique_minima)} at attempt {attempt_idx + 1} "
                    f"with energy {result['energy']:.10f}",
                    flush=True,
                )
            if len(unique_minima) > 4:
                extra_distinct.append(result)
        else:
            attempts_since_new += 1
            current = unique_minima[duplicate_idx]
            if result["energy"] < current["energy"]:
                unique_minima[duplicate_idx] = result

        if len(unique_minima) >= 4 and attempts_since_new >= int(no_new_minima_patience):
            break

    unique_minima = sorted(
        unique_minima,
        key=lambda item: (float(item["energy"]), item["descriptor"].tolist()),
    )
    if extra_distinct:
        # More than 4 compact minima found.  Keep the 4 with lowest energy.
        # This can happen if minima_tol is tight relative to the descriptor-space
        # separation between nearly degenerate structures.  We warn but continue.
        warnings.warn(
            f"More than 4 distinct compact minima found ({len(unique_minima)} total). "
            "Keeping the 4 lowest-energy ones. Consider increasing minima_tol if this "
            "is unexpected.",
            RuntimeWarning,
        )
        unique_minima = unique_minima[:4]
    if len(unique_minima) < 4:
        raise RuntimeError(
            f"Expected at least 4 distinct minima, found only {len(unique_minima)} after "
            f"{total_attempts} attempts. Increase max_attempts, decrease minima_tol, or "
            "relax cluster_max_pair_distance."
        )

    metadata = {
        "benchmark": "LJ7(2d)",
        "temperature_reduced": DEFAULT_T_STAR,
        "seed": int(seed),
        "r_min": float(r_min),
        "minima_tol": float(minima_tol),
        "cluster_max_pair_distance": float(cluster_max_pair_distance),
        "init_scale": float(init_scale),
        "max_attempts": int(max_attempts),
        "attempts_used": int(total_attempts),
        "rejected_noncompact": int(rejected_noncompact),
        "no_new_minima_patience": int(no_new_minima_patience),
        "local_maxiter": int(local_maxiter),
        "local_gtol": float(local_gtol),
        "descriptor_definition": "sorted pair distances of shape (21,)",
        "minima_filter": (
            "Only compact bonded-cluster minima are retained. Candidates with "
            "max(sorted pair distance) above cluster_max_pair_distance are rejected "
            "to exclude nearly dissociated pseudo-minima from EMC preprocessing."
        ),
    }
    save_minima_bundle(unique_minima, output_dir=output_dir, metadata=metadata)
    return load_minima_bundle(output_dir)


def brownian_noise(
    rng: np.random.Generator,
    n_samples: int,
    sigma_noise: float,
    dt: float,
) -> np.ndarray:
    noise = rng.standard_normal((int(n_samples), N_ATOMS, SPATIAL_DIM))
    return sigma_noise * np.sqrt(float(dt)) * noise


def step_ula(
    states: np.ndarray,
    dt: float,
    sigma_noise: float,
    r_min: float,
    rng: np.random.Generator,
) -> np.ndarray:
    drift = -grad_energy(states, r_min=r_min)
    proposal = states + tamed_increment(drift, dt) + brownian_noise(
        rng,
        n_samples=states.shape[0],
        sigma_noise=sigma_noise,
        dt=dt,
    )
    return remove_center_of_mass(proposal)


def step_mala(
    states: np.ndarray,
    dt: float,
    temperature: float,
    sigma_noise: float,
    r_min: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    drift_x = -grad_energy(states, r_min=r_min)
    mean_x = states + tamed_increment(drift_x, dt)
    proposal = mean_x + brownian_noise(
        rng,
        n_samples=states.shape[0],
        sigma_noise=sigma_noise,
        dt=dt,
    )
    proposal = remove_center_of_mass(proposal)

    drift_y = -grad_energy(proposal, r_min=r_min)
    mean_y = proposal + tamed_increment(drift_y, dt)

    e_x = total_energy(states, r_min=r_min)
    e_y = total_energy(proposal, r_min=r_min)

    noise_var = max((sigma_noise ** 2) * float(dt), 1e-12)
    diff_fwd = proposal - mean_x
    diff_bwd = states - mean_y
    log_q_y_given_x = -np.sum(diff_fwd.reshape(diff_fwd.shape[0], -1) ** 2, axis=1) / (2.0 * noise_var)
    log_q_x_given_y = -np.sum(diff_bwd.reshape(diff_bwd.shape[0], -1) ** 2, axis=1) / (2.0 * noise_var)
    log_alpha = (-e_y / float(temperature) + log_q_x_given_y) - (
        -e_x / float(temperature) + log_q_y_given_x
    )
    accept = np.log(rng.random(states.shape[0])) < log_alpha
    updated = states.copy()
    updated[accept] = proposal[accept]
    return updated, float(np.mean(accept))


def step_flmc(
    states: np.ndarray,
    dt: float,
    temperature: float,
    alpha: float,
    r_min: float,
    rng: np.random.Generator,
    step_cap: float | None = 1.25,
) -> np.ndarray:
    """
    Temperature-scaled FLMC step for LJ7(2d).

    Uses the temperature-aware parameterisation:
        X_{n+1} = X_n - dt * c_alpha * gradE(X_n) + (T_star * dt)^(1/alpha) * xi_n

    where gradE is the raw LJ energy gradient (NOT divided by T_star) and the
    noise scale is (T_star * dt)^(1/alpha).  Compared to the original
    temperature-unaware form (gradU = gradE/T_star, noise = dt^(1/alpha)*xi):

      - At alpha=2 this reduces *exactly* to tamed ULA (same drift and noise).
      - The drift multiplier drops from c_alpha/T_star (~24x at T*=0.05) to
        c_alpha (~1.18x), eliminating the large Euler-Maruyama discretisation
        error that caused the cluster to expand in the original form.
      - The noise is scaled down by T_star^(1/alpha) relative to the original,
        appropriate for the cold Boltzmann target at T_star=0.05.

    Both forms target exp(-E/T_star) in continuous time; only the discrete-time
    accuracy differs.

    Multidimensional stable motion uses independent coordinate-wise SαS(1)
    entries.  We remove center-of-mass motion after the step because the
    benchmark state space is represented modulo translations.

    step_cap : float or None
        Maximum per-sample displacement (Euclidean norm in flat space) allowed
        per step.  Alpha-stable noise (alpha < 2) has infinite variance; the
        cap prevents rare catastrophically large noise realisations from pushing
        LJ atoms through each other.  Analogous to jump_cap in step_lsbmc.
    """
    flat = flatten_states(states)
    flat_new = step_flmc_nd(
        x=flat,
        dt=float(dt),
        alpha=float(alpha),
        gradU_fn=lambda arr: grad_energy_flat(arr, r_min=r_min),
        rng=rng,
        clip_bounds=None,
        sigma=float(temperature),
    )
    if step_cap is not None:
        delta = flat_new - flat
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        too_large = norms > float(step_cap)
        delta = np.where(too_large, delta * float(step_cap) / (norms + 1e-12), delta)
        flat_new = flat + delta
    return remove_center_of_mass(unflatten_states(flat_new))


def levy_score_integral(
    states: np.ndarray,
    temperature: float,
    lam: float,
    sigma_L: float,
    multipliers: np.ndarray,
    pm: np.ndarray,
    rng: np.random.Generator,
    r_min: float,
    n_dir_score: int,
    n_theta: int,
    score_clip: float | None = None,
) -> np.ndarray:
    """
    Monte Carlo estimate of the stationary Lévy-score correction.

    Manuscript formula (Eq. 27):
        S_L^s(x) = -λ ∫₀¹ ∫ r exp{-(E(x-θr)-E(x))/T*} ν_norm(dr) dθ

    where ν_norm is the normalised jump measure (prob over multipliers × uniform on sphere).

    SIGN CONVENTION — returned value and how it enters the drift:
    This function returns `code_score`, which satisfies `code_score = -S_L^s`.
    The LSB-MC SDE drift is `-∇V + S_L^s = -∇V - code_score`.
    In step_lsbmc this appears as `drift - score`, where `drift = -∇V`.
    Therefore `drift - score` is mathematically correct.

    SYMMETRISED ESTIMATOR:
    For each randomly sampled direction r from the full sphere, the identity
        E_r[r · exp(-(E(x-θr)-E(x))/T*)] = E_r[0.5*(ratio_minus - ratio_plus)*r]
    holds because ratio_plus(r) = ratio_minus(-r), making the raw estimator
    antisymmetric in r while the symmetrised version is equivalent in expectation.
    The symmetrised form halves the estimator variance without introducing bias.
    """
    states = remove_center_of_mass(states)
    n_samples = states.shape[0]
    energies_0 = total_energy(states, r_min=r_min)
    theta = np.linspace(0.0, 1.0, int(n_theta))
    weights = np.ones_like(theta)
    weights[0] = 0.5
    weights[-1] = 0.5
    weights /= np.sum(weights)

    directions = sample_com_free_directions(rng, int(n_dir_score))
    score = np.zeros_like(states)
    for mult, prob in zip(np.asarray(multipliers, dtype=float), np.asarray(pm, dtype=float)):
        jump = float(sigma_L) * float(mult) * directions
        for direction in jump:
            accum = np.zeros_like(states)
            for th, w_th in zip(theta, weights):
                plus = remove_center_of_mass(states + float(th) * direction[None, :, :])
                minus = remove_center_of_mass(states - float(th) * direction[None, :, :])
                e_plus = total_energy(plus, r_min=r_min)
                e_minus = total_energy(minus, r_min=r_min)
                ratio_plus = np.exp(np.clip(-(e_plus - energies_0) / float(temperature), -80.0, 80.0))
                ratio_minus = np.exp(np.clip(-(e_minus - energies_0) / float(temperature), -80.0, 80.0))
                term = 0.5 * (ratio_minus - ratio_plus)
                accum += float(w_th) * term[:, None, None] * direction[None, :, :]
            score += float(prob) / float(n_dir_score) * accum

    score *= float(lam)
    if score_clip is not None:
        score = np.clip(score, -float(score_clip), float(score_clip))
    return remove_center_of_mass(score)


def step_lsbmc(
    states: np.ndarray,
    dt: float,
    temperature: float,
    sigma_noise: float,
    lam: float,
    sigma_L: float,
    multipliers: np.ndarray,
    pm: np.ndarray,
    r_min: float,
    rng: np.random.Generator,
    n_dir_score: int = 4,
    n_theta: int = 5,
    jump_cap: float | None = 1.25,
    score_clip: float | None = 25.0,
) -> np.ndarray:
    drift = -grad_energy(states, r_min=r_min)
    score = levy_score_integral(
        states,
        temperature=temperature,
        lam=lam,
        sigma_L=sigma_L,
        multipliers=multipliers,
        pm=pm,
        rng=rng,
        r_min=r_min,
        n_dir_score=n_dir_score,
        n_theta=n_theta,
        score_clip=score_clip,
    )
    proposal = states + tamed_increment(drift - score, dt) + brownian_noise(
        rng,
        n_samples=states.shape[0],
        sigma_noise=sigma_noise,
        dt=dt,
    )

    jump_counts = rng.poisson(float(lam) * float(dt), size=states.shape[0])
    active = np.where(jump_counts > 0)[0]
    if active.size > 0:
        for idx in active:
            total_jump = np.zeros((N_ATOMS, SPATIAL_DIM), dtype=float)
            for _ in range(int(jump_counts[idx])):
                mult = float(rng.choice(multipliers, p=pm))
                direction = sample_com_free_directions(rng, 1)[0]
                total_jump += float(sigma_L) * mult * direction
            if jump_cap is not None:
                norm = np.linalg.norm(total_jump.reshape(-1))
                if norm > float(jump_cap):
                    total_jump *= float(jump_cap) / (norm + 1e-12)
            proposal[idx] += total_jump

    return remove_center_of_mass(proposal)


def aggregate_histories(histories: list[dict], keys: list[str]) -> tuple[dict, dict]:
    stacked = {k: np.stack([np.asarray(h[k], dtype=float) for h in histories], axis=0) for k in keys}
    mean = {k: np.nanmean(stacked[k], axis=0) for k in keys}
    std = {k: np.nanstd(stacked[k], axis=0) for k in keys}
    return mean, std


def build_benchmark_eval_steps(total_steps: int, num_checkpoints: int) -> set[int]:
    num_checkpoints = max(2, int(num_checkpoints))
    return set(np.unique(np.linspace(0, int(total_steps), num_checkpoints, dtype=int)).tolist())


def canonical_cluster_configuration() -> np.ndarray:
    bond = 2.0 ** (1.0 / 6.0)
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    outer = bond * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    center = np.zeros((1, SPATIAL_DIM), dtype=float)
    return remove_center_of_mass(np.concatenate([center, outer], axis=0))


def random_rotate_states(states: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    batch, squeeze = _ensure_state_batch(states)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=batch.shape[0])
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    rotated = batch.copy()
    x = batch[:, :, 0]
    y = batch[:, :, 1]
    rotated[:, :, 0] = cos_a[:, None] * x - sin_a[:, None] * y
    rotated[:, :, 1] = sin_a[:, None] * x + cos_a[:, None] * y
    return rotated[0] if squeeze else rotated


def initial_ensemble(
    n_samples: int,
    rng: np.random.Generator,
    init_scale: float = 1.35,
    init_noise: float = 0.08,
) -> np.ndarray:
    base = canonical_cluster_configuration()
    scales = float(init_scale) * np.exp(0.18 * rng.standard_normal((int(n_samples), 1, 1)))
    states = scales * np.repeat(base[None, :, :], int(n_samples), axis=0)
    states += float(init_noise) * rng.standard_normal(states.shape)
    states = random_rotate_states(states, rng)
    return remove_center_of_mass(states)


def load_reference_bundle(output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    paths = get_lj7_paths(output_dir)
    with open(paths.reference_metadata, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return {
        "paths": paths,
        "descriptors": np.load(paths.reference_descriptors),
        "metadata": metadata,
    }


def generate_reference_descriptors(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    seed: int = 2027,
    temperature: float = DEFAULT_T_STAR,
    dt: float = 2.0e-4,
    burn_in_steps: int = 8000,
    thin: int = 20,
    reference_size: int = 2048,
    n_chains: int = 128,
    init_scale: float = 1.10,
    init_noise: float = 0.08,
    r_min: float = DEFAULT_R_MIN,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    paths = ensure_output_dir(output_dir)
    if not force and os.path.exists(paths.reference_descriptors) and os.path.exists(paths.reference_metadata):
        return load_reference_bundle(output_dir)

    rng = np.random.default_rng(int(seed))
    sigma_noise = float(np.sqrt(2.0 * float(temperature)))
    n_chains = max(1, min(int(n_chains), int(reference_size)))
    samples_per_chain = int(np.ceil(float(reference_size) / float(n_chains)))
    total_steps = int(burn_in_steps) + samples_per_chain * int(thin)
    states = initial_ensemble(
        n_samples=n_chains,
        rng=rng,
        init_scale=init_scale,
        init_noise=init_noise,
    )
    collected = []
    acceptance = []

    for step_idx in range(total_steps):
        states, acc = step_mala(
            states,
            dt=dt,
            temperature=temperature,
            sigma_noise=sigma_noise,
            r_min=r_min,
            rng=rng,
        )
        acceptance.append(acc)
        if step_idx >= int(burn_in_steps) and (step_idx - int(burn_in_steps)) % int(thin) == 0:
            collected.append(sorted_pair_distance_descriptor(states, r_min=r_min))

    reference_descriptors = np.concatenate(collected, axis=0)[: int(reference_size)]
    np.save(paths.reference_descriptors, reference_descriptors)
    metadata = {
        "benchmark": "LJ7(2d)",
        "reference_method": "multi-chain MALA from random rotated noisy compact LJ7 configurations",
        "seed": int(seed),
        "temperature_reduced": float(temperature),
        "sigma_noise": float(sigma_noise),
        "dt": float(dt),
        "burn_in_steps": int(burn_in_steps),
        "thin": int(thin),
        "reference_size": int(reference_descriptors.shape[0]),
        "n_chains": int(n_chains),
        "init_scale": float(init_scale),
        "init_noise": float(init_noise),
        "r_min": float(r_min),
        "descriptor_dim": int(reference_descriptors.shape[1]),
        "mean_acceptance_rate": float(np.mean(acceptance)) if acceptance else float("nan"),
        "caveat": (
            "No finite mode set is imposed for LJ7(2d), so EMC is disabled. "
            "The reference cloud is an approximate descriptor-space equilibrium proxy "
            "built from many MALA chains started from compact configurations."
        ),
    }
    with open(paths.reference_metadata, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    if verbose:
        print(f"[reference] saved {reference_descriptors.shape[0]} descriptor samples")
    return {
        "paths": paths,
        "descriptors": reference_descriptors,
        "metadata": metadata,
    }


def run_simulation(
    seed: int,
    temperature: float,
    dt: float,
    total_time: float,
    n_samples: int,
    r_min: float,
    alpha: float,
    lam: float,
    sigma_L: float,
    jump_multipliers: tuple[float, ...],
    jump_weights: tuple[float, ...],
    init_scale: float,
    init_noise: float,
    benchmark_ref_samples: np.ndarray | None = None,
    benchmark_config: dict | None = None,
    n_dir_score: int = 4,
    n_theta: int = 5,
    jump_cap: float | None = 1.25,
    score_clip: float | None = 25.0,
    flmc_step_cap: float | None = 1.25,
    return_benchmark_history: bool = False,
    # EMC parameters (optional; no-op when emc_enabled=False)
    minima_descriptors: np.ndarray | None = None,
    cluster_max_pair_distance: float = DEFAULT_CLUSTER_MAX_PAIR_DISTANCE,
    assignment_radius: float | None = None,
    emc_enabled: bool = False,
    emc_hard_coverage_threshold: float = 0.01,
    emc_assignment_mode: str = "quench",
    emc_quench_maxiter: int = 400,
    emc_quench_gtol: float = 1e-6,
) -> tuple:
    rng = np.random.default_rng(int(seed))
    sigma_noise = float(np.sqrt(2.0 * float(temperature)))
    multipliers = np.asarray(jump_multipliers, dtype=float)
    weights = np.asarray(jump_weights, dtype=float)
    weights /= np.sum(weights)

    initial = initial_ensemble(
        n_samples=int(n_samples),
        rng=rng,
        init_scale=init_scale,
        init_noise=init_noise,
    )
    states = {
        "ula": initial.copy(),
        "mala": initial.copy(),
        "flmc": initial.copy(),
        "lsbmc": initial.copy(),
    }

    sinkhorn_mmd_metric_names = [
        "pairdist_benchmark_sinkhorn_ot_cost",
        "pairdist_benchmark_sinkhorn_divergence",
        "pairdist_benchmark_mmd_squared",
        "pairdist_benchmark_mmd",
    ]
    _run_emc = emc_enabled and minima_descriptors is not None
    emc_metric_names: list[str] = list(_EMC_METRIC_NAMES) if _run_emc else []
    benchmark_metric_names = sinkhorn_mmd_metric_names + emc_metric_names

    benchmark_history = None
    benchmark_eval_steps = set()
    if benchmark_ref_samples is not None and benchmark_config is not None:
        benchmark_history = init_benchmark_history(benchmark_metric_names)
        benchmark_eval_steps = build_benchmark_eval_steps(
            int(np.round(float(total_time) / float(dt))),
            benchmark_config.get("benchmark_num_checkpoints", 21),
        )

    steps = int(np.round(float(total_time) / float(dt)))
    mala_acceptance = []

    for step_idx in range(steps + 1):
        if benchmark_history is not None and step_idx in benchmark_eval_steps:
            t_val = float(step_idx) * float(dt)
            benchmark_history["t"].append(t_val)
            descriptors = {
                slug: sorted_pair_distance_descriptor(state, r_min=r_min)
                for slug, state in states.items()
            }
            for method_idx, (slug, _, _) in enumerate(BENCHMARK_METHODS):
                # --- Sinkhorn / MMD metrics (unchanged) ---
                metric_rng = make_metric_rng(benchmark_config["metric_seed"], seed, step_idx, method_idx)
                values = compute_benchmark_metrics(
                    descriptors[slug],
                    benchmark_ref_samples,
                    benchmark_config,
                    metric_rng,
                    metric_prefix="pairdist_benchmark",
                    context=f"lennard_jones seed={seed} t={t_val:.5f} method={slug}",
                )
                for metric_name in sinkhorn_mmd_metric_names:
                    benchmark_history[f"{metric_name}_{slug}"].append(values.get(metric_name, float("nan")))

                # --- EMC metrics (additional; quench-based or raw descriptor) ---
                if _run_emc:
                    if emc_assignment_mode == "quench":
                        assignment = quench_and_assign_states_to_minima(
                            states[slug],
                            minima_descriptors,
                            r_min=float(r_min),
                            cluster_max_pair_distance=float(cluster_max_pair_distance),
                            assignment_radius=assignment_radius,
                            local_maxiter=int(emc_quench_maxiter),
                            local_gtol=float(emc_quench_gtol),
                        )
                    else:  # "raw"
                        assignment = assign_descriptors_to_minima(
                            descriptors[slug],
                            minima_descriptors,
                            cluster_max_pair_distance=float(cluster_max_pair_distance),
                            assignment_radius=assignment_radius,
                        )
                    labels = assignment["labels"]
                    benchmark_history[f"lj7_emc_{slug}"].append(
                        compute_emc_from_labels(labels)
                    )
                    benchmark_history[f"lj7_assigned_fraction_{slug}"].append(
                        assignment["assigned_fraction"]
                    )
                    benchmark_history[f"lj7_hard_mode_coverage_{slug}"].append(
                        compute_hard_mode_coverage(labels, min_fraction=float(emc_hard_coverage_threshold))
                    )
                    benchmark_history[f"lj7_conditional_mode_entropy_{slug}"].append(
                        compute_conditional_mode_entropy(labels)
                    )

        if step_idx == steps:
            break

        states["ula"] = step_ula(
            states["ula"],
            dt=dt,
            sigma_noise=sigma_noise,
            r_min=r_min,
            rng=rng,
        )
        states["mala"], acc = step_mala(
            states["mala"],
            dt=dt,
            temperature=temperature,
            sigma_noise=sigma_noise,
            r_min=r_min,
            rng=rng,
        )
        mala_acceptance.append(acc)
        states["flmc"] = step_flmc(
            states["flmc"],
            dt=dt,
            temperature=temperature,
            alpha=alpha,
            r_min=r_min,
            rng=rng,
            step_cap=flmc_step_cap,
        )
        states["lsbmc"] = step_lsbmc(
            states["lsbmc"],
            dt=dt,
            temperature=temperature,
            sigma_noise=sigma_noise,
            lam=lam,
            sigma_L=sigma_L,
            multipliers=multipliers,
            pm=weights,
            r_min=r_min,
            rng=rng,
            n_dir_score=n_dir_score,
            n_theta=n_theta,
            jump_cap=jump_cap,
            score_clip=score_clip,
        )

    acc_rate = float(np.mean(mala_acceptance)) if mala_acceptance else float("nan")
    if return_benchmark_history:
        return benchmark_history, states["ula"], states["lsbmc"], states["flmc"], states["mala"], acc_rate
    return states["ula"], states["lsbmc"], states["flmc"], states["mala"], acc_rate


def plot_emc_metrics_figure(
    t: np.ndarray,
    mean: dict,
    std: dict,
    output_base: str,
    title: str,
    methods: list | None = None,
) -> None:
    """
    Save a three-panel figure showing EMC, hard mode coverage, and assigned
    fraction over time, one curve per sampler method.

    Parameters
    ----------
    t : np.ndarray
        Evaluation time points.
    mean : dict
        Keys of the form "{metric_name}_{slug}", values are mean arrays.
    std : dict
        Same structure as mean but for standard deviations.
    output_base : str
        Output path without extension; saves both .png and .pdf.
    title : str
        Figure title.
    methods : list or None
        List of (slug, label, color) tuples.  Defaults to BENCHMARK_METHODS.
    """
    import matplotlib.pyplot as plt

    if methods is None:
        methods = BENCHMARK_METHODS

    panel_keys = [
        ("lj7_emc", "EMC", [0.0, 1.0]),
        ("lj7_conditional_mode_entropy", "Cond. Mode Entropy", [0.0, 1.0]),
        ("lj7_hard_mode_coverage", "Hard Mode Coverage", [0.0, 1.0]),
        ("lj7_assigned_fraction", "Assigned Fraction", [0.0, 1.0]),
    ]
    n_panels = len(panel_keys)
    t = np.asarray(t, dtype=float)

    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.2))
    if n_panels == 1:
        axes = [axes]

    for ax, (metric_name, ylabel, ylim) in zip(axes, panel_keys):
        for slug, label, color in methods:
            key = f"{metric_name}_{slug}"
            if key not in mean:
                continue
            mu = np.asarray(mean[key], dtype=float)
            sigma = np.asarray(std[key], dtype=float)
            ax.plot(t, mu, label=label, color=color, linewidth=1.6)
            ax.fill_between(t, mu - sigma, mu + sigma, alpha=0.18, color=color)
        ax.set_xlabel("time")
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{output_base}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LJ7(2d) benchmark with pair-distance metrics and Sinkhorn/MMD.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--T-star", type=float, default=DEFAULT_T_STAR)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--total-time", type=float, default=1.0)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--reference-size", type=int, default=2048)
    parser.add_argument("--reference-chains", type=int, default=128)
    parser.add_argument("--reference-burn-in", type=int, default=3000)
    parser.add_argument("--reference-thin", type=int, default=10)
    parser.add_argument("--reference-force", action="store_true")
    parser.add_argument("--init-scale", type=float, default=1.35)
    parser.add_argument("--init-noise", type=float, default=0.08)
    parser.add_argument("--reference-init-scale", type=float, default=1.10)
    parser.add_argument("--reference-init-noise", type=float, default=0.08)
    parser.add_argument("--r-min", type=float, default=DEFAULT_R_MIN)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--lam", type=float, default=1.5)
    parser.add_argument("--sigma-L", type=float, default=0.60)
    parser.add_argument("--n-dir-score", type=int, default=4)
    parser.add_argument("--n-theta", type=int, default=5)
    parser.add_argument("--jump-cap", type=float, default=1.25)
    parser.add_argument("--score-clip", type=float, default=25.0)
    parser.add_argument("--flmc-step-cap", type=float, default=1.25)
    parser.add_argument("--eval-checkpoints", type=int, default=21)
    parser.add_argument("--metric-seed", type=int, default=2029)
    parser.add_argument("--benchmark-num-repeats", type=int, default=2)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.08)
    parser.add_argument("--sinkhorn-max-iter", type=int, default=300)
    parser.add_argument("--sinkhorn-tol", type=float, default=1e-5)
    parser.add_argument("--sinkhorn-subsample-size", type=int, default=192)
    parser.add_argument("--mmd-num-kernels", type=int, default=10)
    parser.add_argument("--mmd-subsample-size", type=int, default=256)
    parser.add_argument("--mmd-bandwidth-base", type=float, default=1.0)
    parser.add_argument("--mmd-bandwidth-multiplier", type=float, default=2.0)
    # --- EMC flags ---
    parser.add_argument(
        "--emc-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable EMC (Entropic Mode Coverage) metric for LJ7(2d). Default: True.",
    )
    parser.add_argument(
        "--emc-assignment-radius",
        type=float,
        default=None,
        help=(
            "Hard radius in descriptor space for well assignment. "
            "Default: None (auto = 0.35 * min inter-minimum separation)."
        ),
    )
    parser.add_argument(
        "--emc-hard-coverage-threshold",
        type=float,
        default=0.01,
        help=(
            "Minimum fraction of all samples required for a mode to count as "
            "covered in the hard mode coverage metric. Default: 0.01."
        ),
    )
    parser.add_argument(
        "--minima-force",
        action="store_true",
        default=False,
        help="Force rediscovery of the four LJ7(2d) minima even if cached files exist.",
    )
    parser.add_argument(
        "--emc-assignment-mode",
        choices=["quench", "raw"],
        default="quench",
        help=(
            "EMC well assignment mode.  'quench' runs local_minimize() from each "
            "sample and assigns the quenched descriptor to the nearest minimum "
            "(basin-of-attraction assignment; scientifically preferred).  'raw' "
            "assigns the raw sample descriptor using a nearest-ball test.  "
            "Default: quench."
        ),
    )
    parser.add_argument(
        "--emc-quench-maxiter",
        type=int,
        default=400,
        help="Maximum L-BFGS-B iterations per sample quench (quench mode only). Default: 400.",
    )
    parser.add_argument(
        "--emc-quench-gtol",
        type=float,
        default=1e-6,
        help="Gradient tolerance for per-sample quench (quench mode only). Default: 1e-6.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    out_dir = os.path.abspath(args.output_dir)
    paths = ensure_output_dir(out_dir)

    # -----------------------------------------------------------------------
    # A. EMC preprocessing: load or discover the four compact LJ7(2d) minima
    # -----------------------------------------------------------------------
    emc_enabled = bool(args.emc_enabled)
    minima_bundle = None
    minima_descriptors_arr = None
    emc_assignment_radius: float | None = None
    assignment_radius_mode = "disabled"
    assignment_radius_value: float | None = None

    if emc_enabled:
        print("[emc] loading or discovering four LJ7(2d) compact minima ...")
        minima_bundle = load_or_discover_lj7_minima(
            output_dir=out_dir,
            cluster_max_pair_distance=DEFAULT_CLUSTER_MAX_PAIR_DISTANCE,
            force=bool(args.minima_force),
            verbose=True,
        )
        minima_descriptors_arr = minima_bundle["descriptors"]
        n_modes = minima_descriptors_arr.shape[0]

        # Resolve assignment radius
        if args.emc_assignment_radius is not None:
            emc_assignment_radius = float(args.emc_assignment_radius)
            assignment_radius_mode = "manual"
            assignment_radius_value = emc_assignment_radius
        else:
            # Auto: 0.35 × minimum pairwise descriptor separation
            sep = float("inf")
            for a in range(n_modes):
                for b in range(n_modes):
                    if a != b:
                        d = float(
                            np.linalg.norm(minima_descriptors_arr[a] - minima_descriptors_arr[b])
                        )
                        if d < sep:
                            sep = d
            emc_assignment_radius = 0.35 * sep
            assignment_radius_mode = "auto"
            assignment_radius_value = emc_assignment_radius
        print(
            f"[emc] assignment_radius = {emc_assignment_radius:.6f} ({assignment_radius_mode}), "
            f"n_modes = {n_modes}"
        )

        # Save minima_summary.csv
        with open(paths.minima_summary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["minimum_index", "energy", "max_pair_distance", "descriptor_norm"])
            for idx in range(n_modes):
                desc = minima_descriptors_arr[idx]
                writer.writerow([
                    idx,
                    float(minima_bundle["energies"][idx]),
                    float(np.max(desc)),
                    float(np.linalg.norm(desc)),
                ])
        print(f"[emc] saved minima summary to {paths.minima_summary}")

    # -----------------------------------------------------------------------
    # Benchmark configuration (Sinkhorn / MMD; emc_enabled flag passed through)
    # -----------------------------------------------------------------------
    benchmark_config = get_default_benchmark_config({
        "metric_seed": int(args.metric_seed),
        "benchmark_num_checkpoints": int(args.eval_checkpoints),
        "benchmark_num_repeats": int(args.benchmark_num_repeats),
        "sinkhorn_method": "sinkhorn_stabilized",
        "sinkhorn_epsilon": float(args.sinkhorn_epsilon),
        "sinkhorn_max_iter": int(args.sinkhorn_max_iter),
        "sinkhorn_tol": float(args.sinkhorn_tol),
        "sinkhorn_subsample_size": int(args.sinkhorn_subsample_size),
        "mmd_num_kernels": int(args.mmd_num_kernels),
        "mmd_subsample_size": int(args.mmd_subsample_size),
        "mmd_bandwidth_base": float(args.mmd_bandwidth_base),
        "mmd_bandwidth_multiplier": float(args.mmd_bandwidth_multiplier),
        "emc_enabled": emc_enabled,
    })

    reference_size = max(
        int(args.reference_size),
        int(benchmark_config["sinkhorn_subsample_size"]),
        int(benchmark_config["mmd_subsample_size"]),
    )
    reference = generate_reference_descriptors(
        output_dir=out_dir,
        seed=int(args.seed) + 1,
        temperature=float(args.T_star),
        dt=float(args.dt),
        burn_in_steps=int(args.reference_burn_in),
        thin=int(args.reference_thin),
        reference_size=reference_size,
        n_chains=int(args.reference_chains),
        init_scale=float(args.reference_init_scale),
        init_noise=float(args.reference_init_noise),
        r_min=float(args.r_min),
        force=bool(args.reference_force),
        verbose=True,
    )
    reference_descriptors = reference["descriptors"]

    # -----------------------------------------------------------------------
    # D. Run simulations (with EMC computed at every checkpoint)
    # -----------------------------------------------------------------------
    emc_assignment_mode = str(args.emc_assignment_mode)
    benchmark_histories = []
    acc_rates = []
    # Collect final states from every seed for multi-seed occupancy reporting.
    final_states_all_seeds: list[dict] = []

    for seed_offset in range(int(args.num_seeds)):
        (
            benchmark_history,
            states_ula,
            states_lsbmc,
            states_flmc,
            states_mala,
            acc_rate,
        ) = run_simulation(
            seed=int(args.seed) + seed_offset,
            temperature=float(args.T_star),
            dt=float(args.dt),
            total_time=float(args.total_time),
            n_samples=int(args.n_samples),
            r_min=float(args.r_min),
            alpha=float(args.alpha),
            lam=float(args.lam),
            sigma_L=float(args.sigma_L),
            jump_multipliers=(1.0, 1.7, 2.4),
            jump_weights=(0.70, 0.22, 0.08),
            init_scale=float(args.init_scale),
            init_noise=float(args.init_noise),
            benchmark_ref_samples=reference_descriptors,
            benchmark_config=benchmark_config,
            n_dir_score=int(args.n_dir_score),
            n_theta=int(args.n_theta),
            jump_cap=float(args.jump_cap),
            score_clip=float(args.score_clip),
            flmc_step_cap=float(args.flmc_step_cap),
            return_benchmark_history=True,
            # EMC
            minima_descriptors=minima_descriptors_arr,
            cluster_max_pair_distance=DEFAULT_CLUSTER_MAX_PAIR_DISTANCE,
            assignment_radius=emc_assignment_radius,
            emc_enabled=emc_enabled,
            emc_hard_coverage_threshold=float(args.emc_hard_coverage_threshold),
            emc_assignment_mode=emc_assignment_mode,
            emc_quench_maxiter=int(args.emc_quench_maxiter),
            emc_quench_gtol=float(args.emc_quench_gtol),
        )
        benchmark_histories.append(benchmark_history)
        acc_rates.append(acc_rate)
        final_states_all_seeds.append({
            "seed": int(args.seed) + seed_offset,
            "ula": states_ula,
            "mala": states_mala,
            "flmc": states_flmc,
            "lsbmc": states_lsbmc,
        })
        print(f"[benchmark] completed seed {int(args.seed) + seed_offset}")

    # -----------------------------------------------------------------------
    # Aggregate histories
    # -----------------------------------------------------------------------
    sinkhorn_mmd_metric_names = [
        "pairdist_benchmark_sinkhorn_ot_cost",
        "pairdist_benchmark_sinkhorn_divergence",
        "pairdist_benchmark_mmd_squared",
        "pairdist_benchmark_mmd",
    ]
    emc_metric_names: list[str] = (
        list(_EMC_METRIC_NAMES) if (emc_enabled and minima_descriptors_arr is not None) else []
    )
    all_metric_names = sinkhorn_mmd_metric_names + emc_metric_names

    all_keys = [
        f"{metric_name}_{slug}"
        for metric_name in all_metric_names
        for slug, _, _ in BENCHMARK_METHODS
    ]
    benchmark_mean, benchmark_std = aggregate_histories(benchmark_histories, all_keys)
    benchmark_t = np.asarray(benchmark_histories[0]["t"], dtype=float)

    benchmark_base = resolve_unique_output_base(paths.benchmark_base)
    benchmark_metadata_path = resolve_unique_json_path(paths.benchmark_metadata)

    # -----------------------------------------------------------------------
    # Main benchmark figure: Sinkhorn / MMD / EMC (3-panel when EMC enabled)
    # -----------------------------------------------------------------------
    main_metric_names = [
        "pairdist_benchmark_sinkhorn_divergence",
        "pairdist_benchmark_mmd",
    ]
    main_metric_labels = {
        "pairdist_benchmark_sinkhorn_divergence": "Pair-Distance Sinkhorn",
        "pairdist_benchmark_mmd": "Pair-Distance MMD",
    }
    if emc_enabled and emc_metric_names:
        main_metric_names.append("lj7_emc")
        main_metric_labels["lj7_emc"] = "EMC"
    plot_benchmark_metrics_figure(
        benchmark_t,
        benchmark_mean,
        benchmark_std,
        main_metric_names,
        benchmark_base,
        "Lennard-Jones LJ7(2d): Benchmark Metrics",
        metric_labels=main_metric_labels,
    )
    save_benchmark_metrics_csv(
        f"{benchmark_base}.csv",
        benchmark_t,
        benchmark_mean,
        benchmark_std,
        all_metric_names,
    )

    # -----------------------------------------------------------------------
    # F. EMC figure (additional; only when EMC is enabled)
    # -----------------------------------------------------------------------
    emc_figure_base: str | None = None
    if emc_enabled and emc_metric_names:
        emc_figure_base = resolve_unique_output_base(paths.emc_benchmark_base)
        plot_emc_metrics_figure(
            benchmark_t,
            benchmark_mean,
            benchmark_std,
            emc_figure_base,
            "Lennard-Jones LJ7(2d): EMC Metrics",
        )
        print(f"[emc] saved EMC figure to {emc_figure_base}.png/.pdf")

    # -----------------------------------------------------------------------
    # C/E. Save per-seed and summary mode occupancy CSVs across all seeds
    # -----------------------------------------------------------------------
    if emc_enabled and minima_descriptors_arr is not None and final_states_all_seeds:
        # Helper: compute assignment for one method's final states
        def _final_assignment(final_states_dict: dict, slug: str) -> dict:
            if emc_assignment_mode == "quench":
                return quench_and_assign_states_to_minima(
                    final_states_dict[slug],
                    minima_descriptors_arr,
                    r_min=float(args.r_min),
                    cluster_max_pair_distance=DEFAULT_CLUSTER_MAX_PAIR_DISTANCE,
                    assignment_radius=emc_assignment_radius,
                    local_maxiter=int(args.emc_quench_maxiter),
                    local_gtol=float(args.emc_quench_gtol),
                )
            else:
                descs = sorted_pair_distance_descriptor(
                    final_states_dict[slug], r_min=float(args.r_min)
                )
                return assign_descriptors_to_minima(
                    descs,
                    minima_descriptors_arr,
                    cluster_max_pair_distance=DEFAULT_CLUSTER_MAX_PAIR_DISTANCE,
                    assignment_radius=emc_assignment_radius,
                )

        # Accumulate per-(seed, method) rows
        by_seed_rows: list[dict] = []
        for fs in final_states_all_seeds:
            seed_val = fs["seed"]
            for slug, _, _ in BENCHMARK_METHODS:
                assignment = _final_assignment(fs, slug)
                labels = assignment["labels"]
                occ = compute_mode_occupancy(labels)
                row = {
                    "seed": seed_val,
                    "method": slug,
                    "mode_0": float(occ[0]),
                    "mode_1": float(occ[1]),
                    "mode_2": float(occ[2]),
                    "mode_3": float(occ[3]),
                    "assigned_fraction": float(assignment["assigned_fraction"]),
                    "emc": float(compute_emc_from_labels(labels)),
                    "hard_mode_coverage": float(
                        compute_hard_mode_coverage(labels, min_fraction=float(args.emc_hard_coverage_threshold))
                    ),
                    "conditional_mode_entropy": float(compute_conditional_mode_entropy(labels)),
                }
                by_seed_rows.append(row)

        # Write mode_occupancy_final_by_seed.csv
        by_seed_cols = [
            "seed", "method",
            "mode_0", "mode_1", "mode_2", "mode_3",
            "assigned_fraction", "emc", "hard_mode_coverage", "conditional_mode_entropy",
        ]
        with open(paths.mode_occupancy_final_by_seed, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=by_seed_cols)
            writer.writeheader()
            writer.writerows(by_seed_rows)
        print(f"[emc] saved per-seed mode occupancy to {paths.mode_occupancy_final_by_seed}")

        # Write mode_occupancy_final_summary.csv (mean ± std over seeds per method)
        scalar_cols = ["mode_0", "mode_1", "mode_2", "mode_3",
                       "assigned_fraction", "emc", "hard_mode_coverage", "conditional_mode_entropy"]
        summary_cols = ["method"]
        for c in scalar_cols:
            summary_cols += [f"{c}_mean", f"{c}_std"]

        with open(paths.mode_occupancy_final_summary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_cols)
            writer.writeheader()
            for slug, _, _ in BENCHMARK_METHODS:
                slug_rows = [r for r in by_seed_rows if r["method"] == slug]
                summary_row: dict = {"method": slug}
                for c in scalar_cols:
                    vals = np.array([r[c] for r in slug_rows], dtype=float)
                    summary_row[f"{c}_mean"] = float(np.mean(vals))
                    summary_row[f"{c}_std"] = float(np.std(vals))
                writer.writerow(summary_row)
        print(f"[emc] saved summary mode occupancy to {paths.mode_occupancy_final_summary}")

    # -----------------------------------------------------------------------
    # Benchmark metadata JSON
    # -----------------------------------------------------------------------
    emc_extra: dict = {
        "emc_enabled": emc_enabled,
        "emc_definition": "assigned_fraction * exp(H(p_assigned)) / n_modes",
        "emc_note_exploration": (
            "EMC is an exploration metric, not an equilibrium-fidelity metric. "
            "It measures diversity of basin visits during the finite-time run "
            "from the compact canonical initialization.  Use Sinkhorn/MMD for "
            "distribution-fidelity assessment."
        ),
        "emc_note_near_one": (
            "EMC near 1 requires BOTH a high assigned fraction (most samples "
            "land in a bonded well) AND near-uniform occupancy across all four "
            "wells.  Either condition alone is insufficient."
        ),
        "conditional_mode_entropy_definition": "exp(H(p_assigned)) / n_modes",
        "hard_mode_coverage_threshold": float(args.emc_hard_coverage_threshold),
        "emc_assignment_mode": emc_assignment_mode,
    }
    if emc_enabled and minima_descriptors_arr is not None:
        emc_extra.update({
            "minima_descriptors_path": os.path.relpath(paths.minima_descriptors, paths.output_dir),
            "cluster_max_pair_distance": float(DEFAULT_CLUSTER_MAX_PAIR_DISTANCE),
            "assignment_radius_mode": assignment_radius_mode,
            "assignment_radius_value": assignment_radius_value,
            "n_modes": int(minima_descriptors_arr.shape[0]),
            "emc_applicable": True,
            "emc_assignment_mode_note": (
                "quench: each sample is locally minimized before assignment — "
                "approximates basin-of-attraction assignment.  "
                "raw: raw sample descriptor is assigned using a nearest-ball test."
            ) if emc_assignment_mode == "quench" else (
                "raw: raw sample descriptor assigned via nearest-ball in descriptor space."
            ),
        })
        if emc_assignment_mode == "quench":
            emc_extra["emc_quench_maxiter"] = int(args.emc_quench_maxiter)
            emc_extra["emc_quench_gtol"] = float(args.emc_quench_gtol)
    else:
        emc_extra["emc_applicable"] = False

    save_benchmark_metadata_json(
        benchmark_metadata_path,
        "lennard_jones_n7_d2",
        benchmark_config,
        all_metric_names,
        mode_descriptors=None,
        extra_metadata={
            "benchmark": "LJ7(2d)",
            "temperature_reduced": float(args.T_star),
            "sigma_noise": float(2.0 * float(args.T_star)) ** 0.5,
            "sigma_noise_squared": 2.0 * float(args.T_star),
            "sigma_noise_squared_relation": "sigma_noise^2 = 2 * T_star",
            "r_min": float(args.r_min),
            "descriptor_dim": int(DESCRIPTOR_DIM),
            "descriptor_definition": "sorted pair distances of the seven-atom 2D cluster",
            "representation": "pair-distance descriptor benchmark",
            "benchmark_reference_size": int(reference_descriptors.shape[0]),
            "benchmark_num_checkpoints": int(benchmark_t.size),
            "num_seeds": int(args.num_seeds),
            "n_samples_per_method": int(args.n_samples),
            "dt": float(args.dt),
            "total_time": float(args.total_time),
            "alpha": float(args.alpha),
            "lam": float(args.lam),
            "sigma_L": float(args.sigma_L),
            "n_dir_score": int(args.n_dir_score),
            "n_theta": int(args.n_theta),
            "jump_cap": float(args.jump_cap),
            "score_clip": float(args.score_clip),
            "reference_descriptors_path": os.path.relpath(
                paths.reference_descriptors, paths.output_dir
            ),
            "reference_method": reference["metadata"]["reference_method"],
            "mala_mean_acceptance_rate": float(np.mean(acc_rates)) if acc_rates else float("nan"),
            **emc_extra,
        },
    )

    print(f"Saved benchmark metrics to: {benchmark_base}.png/.pdf/.csv")
    print(f"Saved benchmark metadata to: {benchmark_metadata_path}")
    result: dict = {
        "paths": paths,
        "benchmark_base": benchmark_base,
        "benchmark_metadata_path": benchmark_metadata_path,
    }
    if emc_figure_base is not None:
        result["emc_figure_base"] = emc_figure_base
    return result


if __name__ == "__main__":
    main()
