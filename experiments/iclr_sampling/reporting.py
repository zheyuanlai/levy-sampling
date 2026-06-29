"""Result aggregation: raw CSV, per-config summary CSV, and LaTeX tables."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Metrics aggregated in summary_by_config.csv (only those present are used).
SUMMARY_METRICS = [
    "w2_or_sliced_w2", "mmd", "mode_coverage", "n_covered_modes", "mode_kl",
    "emc", "emc_notebook", "count_mode_kl", "count_emc", "block_marginal_kl",
    "overall_deep_frac", "acceptance_rate", "swap_acceptance_rate",
    "runtime_sec", "grad_evals", "pot_evals", "nonfinite_count",
]


def write_raw_csv(rows: List[Dict], path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def summarize(df: pd.DataFrame, group_cols: List[str],
              path: Optional[str] = None) -> pd.DataFrame:
    """mean / std / standard-error / n_seeds per config for every metric."""
    metrics = [m for m in SUMMARY_METRICS if m in df.columns]
    records = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        n = len(sub)
        rec["n_seeds"] = n
        for m in metrics:
            vals = pd.to_numeric(sub[m], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                rec[f"{m}_mean"] = np.nan
                rec[f"{m}_std"] = np.nan
                rec[f"{m}_se"] = np.nan
            else:
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                rec[f"{m}_mean"] = mean
                rec[f"{m}_std"] = std
                rec[f"{m}_se"] = std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        records.append(rec)
    summary = pd.DataFrame(records).sort_values(group_cols).reset_index(drop=True)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(path, index=False)
    return summary


def _fmt(mean, se, prec=3):
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "--"
    if se is None or (isinstance(se, float) and math.isnan(se)) or se == 0:
        return f"{mean:.{prec}f}"
    return f"{mean:.{prec}f}$\\pm${se:.{prec}f}"


def latex_table(summary: pd.DataFrame, config_col: str, config_val,
                methods: List[str], metric_cols: List[str],
                metric_headers: Dict[str, str], caption: str, label: str,
                method_order: Optional[List[str]] = None) -> str:
    """Build a booktabs LaTeX table for one config value (e.g. one dimension)."""
    sub = summary[summary[config_col] == config_val]
    metric_cols = [m for m in metric_cols if f"{m}_mean" in summary.columns]
    method_order = method_order or methods
    header = " & ".join(["Method"] + [metric_headers.get(m, m) for m in metric_cols])
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "r" * len(metric_cols) + "}",
        "\\toprule", header + " \\\\", "\\midrule",
    ]
    for m in method_order:
        row = sub[sub["method"] == m]
        if row.empty:
            continue
        r = row.iloc[0]
        cells = [m]
        for mc in metric_cols:
            prec = 0 if mc in ("grad_evals", "pot_evals", "n_covered_modes") else 3
            if prec == 0:
                mean = r.get(f"{mc}_mean", np.nan)
                cells.append("--" if (isinstance(mean, float) and math.isnan(mean))
                             else f"{mean:.0f}")
            else:
                cells.append(_fmt(r.get(f"{mc}_mean", np.nan),
                                  r.get(f"{mc}_se", np.nan), prec))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def scaling_latex_table(summary: pd.DataFrame, config_col: str,
                        metric: str, methods: List[str], config_vals: List,
                        caption: str, label: str,
                        config_header: str = "dim") -> str:
    """A table with methods as rows and scaling values as columns for one metric."""
    head = " & ".join([config_header] + [f"{config_header}={v}" for v in config_vals])
    head = "Method & " + " & ".join([str(v) for v in config_vals])
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "r" * len(config_vals) + "}",
        "\\toprule", head + " \\\\", "\\midrule",
    ]
    for m in methods:
        cells = [m]
        for v in config_vals:
            row = summary[(summary["method"] == m) & (summary[config_col] == v)]
            if row.empty:
                cells.append("--")
            else:
                r = row.iloc[0]
                cells.append(_fmt(r.get(f"{metric}_mean", np.nan),
                                  r.get(f"{metric}_se", np.nan)))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def write_text(path, text: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
