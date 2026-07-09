"""Finite-dimensional spectral reference curves for time-decay diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from .plot_style import REFERENCE_STYLES, method_color


def metric_rate_factor(metric: str) -> int:
    name = str(metric).lower().replace("-", "_")
    return 2 if "chi2" in name or "chi_square" in name else 1


def first_valid_anchor(df: pd.DataFrame, metric: str, method: str) -> Optional[Tuple[float, float]]:
    sub = df[df["method"].astype(str) == str(method)][["time", metric]].copy()
    sub = sub[np.isfinite(sub["time"]) & np.isfinite(sub[metric]) & (sub[metric] > 0)]
    if sub.empty:
        return None
    grouped = sub.groupby("time", as_index=False)[metric].mean().sort_values("time")
    row = grouped.iloc[0]
    return float(row["time"]), float(row[metric])


def add_spectral_reference_lines(
    ax,
    df: pd.DataFrame,
    metric: str,
    method: str,
    *,
    form_rate: Optional[float],
    abscissa_rate: Optional[float],
    records: list[dict[str, object]],
    experiment: str,
    figure: str,
    tmax: float | None = None,
    logy: bool = True,
) -> None:
    """Plot anchored form-gap and validated-abscissa comparisons.

    The amplitude rule is the first valid positive mean value of the same
    method and metric. Chi-square-type metrics use twice the spectral rate.
    """

    anchor = first_valid_anchor(df, metric, method)
    if anchor is None:
        return
    anchor_time, anchor_value = anchor
    if tmax is None:
        tmax = float(np.nanmax(df["time"].to_numpy(dtype=float)))
    if not np.isfinite(tmax) or tmax <= anchor_time:
        return
    times = np.linspace(anchor_time, tmax, 300)
    factor = metric_rate_factor(metric)
    color = method_color(method)

    for reference, rate in (("form_gap", form_rate), ("abscissa", abscissa_rate)):
        if rate is None or not np.isfinite(rate) or rate <= 0:
            continue
        effective_rate = factor * float(rate)
        values = np.maximum(anchor_value * np.exp(-effective_rate * (times - anchor_time)), 1e-14)
        style = REFERENCE_STYLES[reference]
        label = f"{method} {'form' if reference == 'form_gap' else 'abscissa'} ref."
        ax.plot(times, values, color=color, label=label, **style)
        records.append(
            {
                "experiment": experiment,
                "figure": figure,
                "metric": metric,
                "method": method,
                "reference": reference,
                "spectral_rate": float(rate),
                "metric_rate_factor": factor,
                "effective_decay_rate": effective_rate,
                "amplitude_rule": "anchor at first valid positive mean metric value",
                "anchor_time": anchor_time,
                "anchor_value": anchor_value,
                "validated_finite_dimensional_reference": True,
            }
        )
    if logy:
        ax.set_yscale("log")


def write_reference_registry(records: list[dict[str, object]], path: Union[str, Path]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "experiment",
        "figure",
        "metric",
        "method",
        "reference",
        "spectral_rate",
        "metric_rate_factor",
        "effective_decay_rate",
        "amplitude_rule",
        "anchor_time",
        "anchor_value",
        "validated_finite_dimensional_reference",
    ]
    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame(columns=columns)
    else:
        frame = frame.drop_duplicates(
            subset=["experiment", "figure", "metric", "method", "reference"], keep="last"
        ).sort_values(["figure", "metric", "method", "reference"])
        frame = frame.reindex(columns=columns)
    frame.to_csv(out, index=False)
    return out
