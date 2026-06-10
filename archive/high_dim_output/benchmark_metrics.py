#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import subprocess
import sys
import warnings
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np


BENCHMARK_METHODS = [
    ("ula", "ULA", "C0"),
    ("mala", "MALA", "C2"),
    ("flmc", "FLMC", "tab:orange"),
    ("lsbmc", "LSBMC", "C3"),
]


DEFAULT_BENCHMARK_CONFIG = {
    "metric_seed": 2026,
    "benchmark_num_checkpoints": 7,
    "benchmark_num_repeats": 1,
    "sinkhorn_backend": "auto",
    "sinkhorn_method": "sinkhorn_log",
    "sinkhorn_epsilon": 0.1,
    "sinkhorn_max_iter": 500,
    "sinkhorn_tol": 1e-6,
    "sinkhorn_subsample_size": 256,
    "mmd_num_kernels": 10,
    "mmd_bandwidth_base": 1.0,
    "mmd_bandwidth_multiplier": 2.0,
    "mmd_subsample_size": 384,
    "emc_tau": 0.5,
    "emc_enabled": True,
    "emc_max_modes": 256,
}


def get_default_benchmark_config(overrides=None):
    config = dict(DEFAULT_BENCHMARK_CONFIG)
    if overrides:
        config.update(overrides)
    return config


def benchmark_history_keys(metric_names, methods=BENCHMARK_METHODS):
    return [f"{metric_name}_{slug}" for metric_name in metric_names for slug, _, _ in methods]


def init_benchmark_history(metric_names, methods=BENCHMARK_METHODS):
    history = {"t": []}
    for key in benchmark_history_keys(metric_names, methods=methods):
        history[key] = []
    return history


def make_metric_rng(metric_seed, *tokens):
    ints = [int(metric_seed)]
    ints.extend(int(tok) for tok in tokens)
    return np.random.default_rng(np.random.SeedSequence(ints))


def sample_from_2d_grid_density(rng, pi, gx, gy, n_samples):
    pi = np.asarray(pi, dtype=float)
    gx = np.asarray(gx, dtype=float)
    gy = np.asarray(gy, dtype=float)
    if pi.ndim != 2:
        raise ValueError("Expected a 2D density grid.")

    dx = gx[1] - gx[0]
    dy = gy[1] - gy[0]
    weights = (pi * dx * dy).ravel()
    weights /= np.sum(weights)
    idx = rng.choice(weights.size, size=int(n_samples), replace=True, p=weights)
    iy, ix = np.unravel_index(idx, pi.shape)
    samples = np.stack(
        [
            gx[ix] + (rng.random(n_samples) - 0.5) * dx,
            gy[iy] + (rng.random(n_samples) - 0.5) * dy,
        ],
        axis=1,
    )
    return samples


def save_figure_both(fig, out_base, dpi=200):
    fig.savefig(f"{out_base}.png", dpi=dpi)
    fig.savefig(f"{out_base}.pdf")
    plt.close(fig)


def clip_samples_to_box(samples, lower, upper):
    arr = np.asarray(samples, dtype=float)
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    return np.clip(arr, lower_arr, upper_arr)


def plot_benchmark_metrics_figure(
    t,
    mean,
    std,
    metric_names,
    out_base,
    title,
    metric_labels=None,
    methods=BENCHMARK_METHODS,
):
    metric_labels = dict(metric_labels or {})
    n_metrics = len(metric_names)
    t = np.asarray(t, dtype=float)
    max_abs_t = float(np.max(np.abs(t))) if t.size else 0.0
    fig, axes = plt.subplots(1, n_metrics, figsize=(6.0 * n_metrics, 4.2))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, metric_names):
        for slug, label, color in methods:
            key = f"{metric_name}_{slug}"
            ax.errorbar(
                t,
                mean[key],
                yerr=std[key],
                fmt="-",
                color=color,
                label=label,
                alpha=0.9,
                capsize=2,
            )
        ax.set_title(metric_labels.get(metric_name, metric_name))
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if max_abs_t > 0.0 and (max_abs_t < 1.0e-2 or max_abs_t >= 1.0e3):
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        if metric_name.endswith("emc"):
            emc_vals = []
            for slug, _, _ in methods:
                key = f"{metric_name}_{slug}"
                vals = np.asarray(mean[key], dtype=float)
                errs = np.asarray(std[key], dtype=float)
                mask = np.isfinite(vals) & np.isfinite(errs)
                if np.any(mask):
                    emc_vals.append(vals[mask] - errs[mask])
                    emc_vals.append(vals[mask] + errs[mask])
            if emc_vals:
                emc_vals = np.concatenate(emc_vals)
                ymin = max(0.0, float(np.min(emc_vals)))
                ymax = min(1.05, float(np.max(emc_vals)))
                if ymax <= 0.20:
                    pad = max(0.01, 0.15 * max(ymax - ymin, 1e-3))
                    ax.set_ylim(max(0.0, ymin - pad), min(1.05, ymax + pad))
                else:
                    ax.set_ylim(0.0, 1.05)
            else:
                ax.set_ylim(0.0, 1.05)

    fig.suptitle(title)
    fig.tight_layout()
    save_figure_both(fig, out_base)


def save_benchmark_metrics_csv(
    out_path,
    t,
    mean,
    std,
    metric_names,
    methods=BENCHMARK_METHODS,
):
    headers = ["t"]
    for metric_name in metric_names:
        for slug, _, _ in methods:
            headers.append(f"{metric_name}_{slug}_mean")
            headers.append(f"{metric_name}_{slug}_std")

    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for idx, time_value in enumerate(np.asarray(t, dtype=float)):
            row = [float(time_value)]
            for metric_name in metric_names:
                for slug, _, _ in methods:
                    key = f"{metric_name}_{slug}"
                    row.append(float(mean[key][idx]))
                    row.append(float(std[key][idx]))
            writer.writerow(row)


def save_benchmark_metadata_json(
    out_path,
    target_name,
    config,
    metric_names,
    mode_descriptors=None,
    extra_metadata=None,
):
    payload = {
        "target": target_name,
        "benchmark_metric_names": list(metric_names),
        "benchmark_config": dict(config),
        "mode_descriptors": _mode_descriptors_to_json(mode_descriptors),
    }
    if any("sinkhorn_ot_cost" in metric_name for metric_name in metric_names):
        payload["sinkhorn_ot_cost_definition"] = (
            "Regularized OT objective with entropy term, matching the configured benchmark formula."
        )
        payload["sinkhorn_divergence_definition"] = (
            "Debiased Sinkhorn divergence built from the regularized OT objective."
        )
    if any("emc" in metric_name for metric_name in metric_names):
        payload["emc_definition"] = (
            "Normalized entropy of the aggregate soft mode-occupancy probabilities, "
            "so 0 indicates single-mode collapse and 1 indicates uniform coverage "
            "across the predefined mode descriptors."
        )
    if extra_metadata:
        payload.update(extra_metadata)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def hypercube_mode_descriptors(dim):
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive.")
    grids = np.array(np.meshgrid(*([[-1.0, 1.0]] * dim), indexing="ij"))
    return grids.reshape(dim, -1).T


def wasserstein2_1d_exact(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size <= 1 or y.size <= 1:
        return float("nan")

    x = np.sort(x)
    y = np.sort(y)
    n = max(x.size, y.size)
    q = (np.arange(n, dtype=float) + 0.5) / n
    qx = (np.arange(x.size, dtype=float) + 0.5) / x.size
    qy = (np.arange(y.size, dtype=float) + 0.5) / y.size
    xq = np.interp(q, qx, x)
    yq = np.interp(q, qy, y)
    return float(np.sqrt(np.mean((xq - yq) ** 2)))


def compute_benchmark_metrics(
    samples,
    reference_samples,
    config,
    rng,
    metric_prefix="benchmark",
    mode_descriptors=None,
    context="benchmark",
):
    num_repeats = max(1, int(config.get("benchmark_num_repeats", 1)))
    if num_repeats == 1:
        return _compute_benchmark_metrics_once(
            samples,
            reference_samples,
            config,
            rng,
            metric_prefix=metric_prefix,
            mode_descriptors=mode_descriptors,
            context=context,
        )

    accum = {}
    for repeat_idx in range(num_repeats):
        repeat_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        values = _compute_benchmark_metrics_once(
            samples,
            reference_samples,
            config,
            repeat_rng,
            metric_prefix=metric_prefix,
            mode_descriptors=mode_descriptors,
            context=f"{context} repeat={repeat_idx}",
        )
        for key, value in values.items():
            accum.setdefault(key, []).append(float(value))

    return {key: float(np.mean(vals)) for key, vals in accum.items()}


def _compute_benchmark_metrics_once(
    samples,
    reference_samples,
    config,
    rng,
    metric_prefix="benchmark",
    mode_descriptors=None,
    context="benchmark",
):
    results = {}

    try:
        sinkhorn_ot_cost, sinkhorn_divergence = compute_sinkhorn_metrics(
            samples,
            reference_samples,
            config,
            rng,
        )
    except Exception as exc:
        warnings.warn(
            f"{context}: Sinkhorn metric failed with {exc}. Recording NaN values.",
            RuntimeWarning,
        )
        sinkhorn_ot_cost = float("nan")
        sinkhorn_divergence = float("nan")
    results[f"{metric_prefix}_sinkhorn_ot_cost"] = sinkhorn_ot_cost
    results[f"{metric_prefix}_sinkhorn_divergence"] = sinkhorn_divergence

    try:
        mmd_squared, mmd = compute_mmd_metrics(samples, reference_samples, config, rng)
    except Exception as exc:
        warnings.warn(
            f"{context}: MMD failed with {exc}. Recording NaN values.",
            RuntimeWarning,
        )
        mmd_squared = float("nan")
        mmd = float("nan")
    results[f"{metric_prefix}_mmd_squared"] = mmd_squared
    results[f"{metric_prefix}_mmd"] = mmd

    if mode_descriptors is not None and bool(config.get("emc_enabled", True)):
        try:
            results[f"{metric_prefix}_emc"] = compute_emc(
                samples,
                mode_descriptors=mode_descriptors,
                tau=float(config["emc_tau"]),
            )
        except Exception as exc:
            warnings.warn(
                f"{context}: EMC failed with {exc}. Recording NaN values.",
                RuntimeWarning,
            )
            results[f"{metric_prefix}_emc"] = float("nan")

    return results


def compute_sinkhorn_metrics(samples, reference_samples, config, rng):
    x = _subsample_rows(_ensure_2d(samples), config.get("sinkhorn_subsample_size"), rng)
    y = _subsample_rows(_ensure_2d(reference_samples), config.get("sinkhorn_subsample_size"), rng)
    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("Need at least one finite sample in each measure.")

    reg = float(config["sinkhorn_epsilon"])
    if reg <= 0.0:
        raise ValueError("sinkhorn_epsilon must be positive.")

    a = np.full(x.shape[0], 1.0 / x.shape[0], dtype=float)
    b = np.full(y.shape[0], 1.0 / y.shape[0], dtype=float)
    backend = _resolve_sinkhorn_backend(
        config.get("sinkhorn_backend", "auto"),
        x.shape[0],
        y.shape[0],
    )
    method = str(config.get("sinkhorn_method", "sinkhorn_log"))
    max_iter = int(config["sinkhorn_max_iter"])
    tol = float(config["sinkhorn_tol"])

    ot_xy = _regularized_ot_cost(x, y, a, b, reg, backend, method, max_iter, tol)
    ot_xx = _regularized_ot_cost(x, x, a, a, reg, backend, method, max_iter, tol)
    ot_yy = _regularized_ot_cost(y, y, b, b, reg, backend, method, max_iter, tol)
    sinkhorn_divergence = ot_xy - 0.5 * ot_xx - 0.5 * ot_yy
    if sinkhorn_divergence < 0.0 and abs(sinkhorn_divergence) < 1e-10:
        sinkhorn_divergence = 0.0
    return float(ot_xy), float(sinkhorn_divergence)


def compute_mmd_metrics(samples, reference_samples, config, rng):
    x = _subsample_rows(_ensure_2d(samples), config.get("mmd_subsample_size"), rng)
    y = _subsample_rows(_ensure_2d(reference_samples), config.get("mmd_subsample_size"), rng)
    n = x.shape[0]
    m = y.shape[0]
    if n <= 1 or m <= 1:
        raise ValueError("Need at least two samples in each measure for unbiased MMD.")

    bandwidths = _build_mmd_bandwidths(x, y, config, rng)
    num_kernels = float(len(bandwidths))
    kxx_sum = _chunked_kernel_sum(x, x, bandwidths)
    kyy_sum = _chunked_kernel_sum(y, y, bandwidths)
    kxy_sum = _chunked_kernel_sum(x, y, bandwidths)

    diag_x = n * num_kernels
    diag_y = m * num_kernels
    mmd_squared = (
        (kxx_sum - diag_x) / (n * (n - 1))
        + (kyy_sum - diag_y) / (m * (m - 1))
        - 2.0 * kxy_sum / (n * m)
    )
    mmd = np.sqrt(max(mmd_squared, 0.0))
    return float(mmd_squared), float(mmd)


def compute_emc(samples, mode_descriptors, tau):
    x = _ensure_2d(samples)
    modes = _ensure_2d(mode_descriptors)
    if modes.shape[0] < 2:
        raise ValueError("EMC requires at least two explicit mode descriptors.")

    tau = float(tau)
    if tau <= 0.0:
        raise ValueError("emc_tau must be positive.")

    probs = compute_mode_assignment_probabilities(x, modes, tau)
    mode_probs = np.mean(probs, axis=0)
    mode_probs /= np.sum(mode_probs) + 1e-300
    entropy = -np.sum(mode_probs * np.log(mode_probs + 1e-300))
    entropy /= np.log(float(modes.shape[0]))
    return float(entropy)


def compute_mode_assignment_probabilities(samples, mode_descriptors, tau):
    x = _ensure_2d(samples)
    modes = _ensure_2d(mode_descriptors)
    tau = float(tau)
    if tau <= 0.0:
        raise ValueError("emc_tau must be positive.")

    d2 = _safe_squared_distances(x, modes)
    logits = -d2 / tau
    logits -= np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= np.sum(probs, axis=1, keepdims=True) + 1e-300
    return probs


def _ensure_2d(samples):
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("Expected a 1D or 2D sample array.")
    mask = np.all(np.isfinite(arr), axis=1)
    return arr[mask]


def _subsample_rows(samples, max_points, rng):
    samples = np.asarray(samples, dtype=float)
    if max_points is None:
        return samples
    max_points = int(max_points)
    if max_points <= 0 or samples.shape[0] <= max_points:
        return samples
    idx = rng.choice(samples.shape[0], size=max_points, replace=False)
    return samples[idx]


def _safe_squared_distances(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True).T
    d2 = x_norm + y_norm - 2.0 * (x @ y.T)
    return np.maximum(d2, 0.0)


def _chunked_kernel_sum(x, y, bandwidths, chunk_size=256):
    y = np.asarray(y, dtype=float)
    y_norm = np.sum(y * y, axis=1)
    total = 0.0
    for start in range(0, x.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), x.shape[0])
        x_chunk = x[start:stop]
        x_norm = np.sum(x_chunk * x_chunk, axis=1, keepdims=True)
        d2 = x_norm + y_norm[None, :] - 2.0 * (x_chunk @ y.T)
        d2 = np.maximum(d2, 0.0)
        kernel_vals = np.zeros_like(d2)
        for bw in bandwidths:
            kernel_vals += np.exp(-d2 / max(bw * bw, 1e-12))
        total += float(np.sum(kernel_vals))
    return total


def _build_mmd_bandwidths(x, y, config, rng):
    z = np.concatenate([x, y], axis=0)
    z = _subsample_rows(z, min(256, z.shape[0]), rng)
    if z.shape[0] <= 1:
        return np.array([1.0], dtype=float)

    d2 = _safe_squared_distances(z, z)
    tri_upper = d2[np.triu_indices(z.shape[0], k=1)]
    tri_upper = tri_upper[tri_upper > 1e-12]
    median_sqdist = float(np.median(tri_upper)) if tri_upper.size > 0 else 1.0
    anchor = np.sqrt(max(median_sqdist, 1e-12))

    num_kernels = max(1, int(config["mmd_num_kernels"]))
    center = anchor * max(float(config["mmd_bandwidth_base"]), 1e-12)
    multiplier = float(config["mmd_bandwidth_multiplier"])
    if num_kernels == 1 or multiplier <= 1.0:
        return np.array([center], dtype=float)

    offsets = np.arange(num_kernels, dtype=float) - 0.5 * (num_kernels - 1)
    bandwidths = center * (multiplier ** offsets)
    bandwidths = np.maximum(bandwidths, 1e-12)
    return bandwidths.astype(float)


def _regularized_ot_cost(x, y, a, b, reg, backend, method, max_iter, tol):
    cost = _safe_squared_distances(x, y)
    plan = _sinkhorn_plan(cost, a, b, reg, backend, method, max_iter, tol)
    positive = plan > 0.0
    entropy = np.sum(plan[positive] * (np.log(plan[positive]) - 1.0))
    return float(np.sum(plan * cost) + reg * entropy)


def _sinkhorn_plan(cost, a, b, reg, backend, method, max_iter, tol):
    if backend == "pot":
        return _sinkhorn_plan_pot(cost, a, b, reg, method, max_iter, tol)
    return _sinkhorn_plan_numpy(cost, a, b, reg, max_iter, tol)


def _sinkhorn_plan_pot(cost, a, b, reg, method, max_iter, tol):
    import ot

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plan = np.asarray(
            ot.sinkhorn(
                a,
                b,
                cost,
                reg=reg,
                method=method,
                numItermax=max_iter,
                stopThr=tol,
                log=False,
                verbose=False,
                warn=True,
            ),
            dtype=float,
        )

    if not np.all(np.isfinite(plan)):
        return _sinkhorn_plan_numpy(cost, a, b, reg, max_iter, tol)

    for caught_warning in caught:
        message = str(caught_warning.message).lower()
        if "sinkhorn did not converge" in message:
            warnings.warn(
                "POT Sinkhorn did not converge; falling back to the internal log-domain solver.",
                RuntimeWarning,
            )
            return _sinkhorn_plan_numpy(cost, a, b, reg, max_iter, tol)

    return plan


def _sinkhorn_plan_numpy(cost, a, b, reg, max_iter, tol):
    log_a = np.log(np.asarray(a, dtype=float) + 1e-300)
    log_b = np.log(np.asarray(b, dtype=float) + 1e-300)
    log_k = -np.asarray(cost, dtype=float) / reg
    log_u = np.zeros_like(log_a)
    log_v = np.zeros_like(log_b)

    plan = None
    for it in range(int(max_iter)):
        log_u = log_a - _logsumexp(log_k + log_v[None, :], axis=1)
        log_v = log_b - _logsumexp(log_k.T + log_u[None, :], axis=1)

        if it % 20 == 0 or it == int(max_iter) - 1:
            log_plan = log_u[:, None] + log_k + log_v[None, :]
            plan = np.exp(log_plan)
            row_err = np.max(np.abs(np.sum(plan, axis=1) - a))
            col_err = np.max(np.abs(np.sum(plan, axis=0) - b))
            if max(row_err, col_err) < tol:
                break

    if plan is None:
        log_plan = log_u[:, None] + log_k + log_v[None, :]
        plan = np.exp(log_plan)
    return plan


def _logsumexp(arr, axis):
    arr = np.asarray(arr, dtype=float)
    max_vals = np.max(arr, axis=axis, keepdims=True)
    stabilized = np.exp(arr - max_vals)
    out = max_vals + np.log(np.sum(stabilized, axis=axis, keepdims=True) + 1e-300)
    return np.squeeze(out, axis=axis)


def _mode_descriptors_to_json(mode_descriptors):
    if mode_descriptors is None:
        return None
    arr = np.asarray(mode_descriptors, dtype=float)
    if arr.ndim == 1:
        return arr.tolist()
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0].tolist()
    return arr.tolist()


@lru_cache(maxsize=None)
def _safe_import_available(module_name):
    try:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except Exception:
        return False
    return proc.returncode == 0


def _resolve_sinkhorn_backend(requested_backend, n_x, n_y):
    backend = str(requested_backend).lower()
    if backend == "pot":
        return "pot" if _safe_import_available("ot") else "numpy"
    if backend == "numpy":
        return "numpy"
    if backend == "auto":
        if max(int(n_x), int(n_y)) <= 2048 and _safe_import_available("ot"):
            return "pot"
        return "numpy"
    return "numpy"
