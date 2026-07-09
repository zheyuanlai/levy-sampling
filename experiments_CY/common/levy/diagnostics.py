"""Small diagnostic helpers for jump and score outputs."""

from __future__ import annotations

import numpy as np


def jump_count_summary(counts):
    counts = np.asarray(counts, dtype=float)
    return {
        "mean_jump_count": float(np.mean(counts)) if counts.size else 0.0,
        "total_jump_count": float(np.sum(counts)) if counts.size else 0.0,
        "max_jump_count": float(np.max(counts)) if counts.size else 0.0,
    }


def merge_score_diagnostics(diags):
    if not diags:
        return {}
    keys = sorted({k for d in diags for k in d})
    out = {}
    for key in keys:
        vals = [d[key] for d in diags if key in d and np.isfinite(d[key])]
        out[key] = float(np.mean(vals)) if vals else np.nan
    return out


def energy_before_after(energy_fn, before, after):
    return {
        "mean_energy_before_jump": float(np.mean(energy_fn(before))),
        "mean_energy_after_jump": float(np.mean(energy_fn(after))),
    }
