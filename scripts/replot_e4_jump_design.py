#!/usr/bin/env python
"""Figures and tables for the E4 jump-design study. Reads CSVs only, no GPU.

Three questions, three figures:

1. Does an alpha-stable nu that knows nothing about the phase square still let
   the score correction work?  Terminal accuracy of the corrected and
   uncorrected dynamics under each nu, against the manuscript's phase-edge law.
2. How much of the correction survives the realised-displacement estimator as
   the bank size A changes, and what does it cost?
3. Is any of that an artifact of one choice of scale or truncation?
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent.parent
METRICS = (("W2", r"SW$_2$"), ("MMD", "MMD"), ("TV", "basin TV"))
Q_THETA = 32
Q_U = 16


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def terminal_stats(rows: list[dict]) -> dict[str, dict[str, tuple[float, float]]]:
    """{method: {metric: (mean over seeds, standard deviation)}} at the last step."""
    last = max(int(r["step"]) for r in rows)
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for row in rows:
        if int(row["step"]) != last:
            continue
        bucket = out.setdefault(row["method"], {})
        for key, _ in METRICS:
            value = row.get(key)
            if value in (None, ""):
                continue
            bucket.setdefault(key, []).append(float(value))
    return {
        method: {
            key: (statistics.fmean(vals),
                  statistics.stdev(vals) if len(vals) > 1 else 0.0)
            for key, vals in metrics.items()
        }
        for method, metrics in out.items()
    }


def load_study(root: Path) -> dict[str, dict]:
    study = {}
    for config_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        csv_path = config_dir / "metrics_timeseries.csv"
        manifest_path = config_dir / "manifest.json"
        if not csv_path.exists() or not manifest_path.exists():
            continue
        rows = read_rows(csv_path)
        study[config_dir.name] = {
            "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
            "rows": rows,
            "terminal": terminal_stats(rows),
        }
    return study


def load_baseline(repo: Path) -> dict:
    """Terminal values of the manuscript's frozen E4 run, not a rerun."""
    path = repo / "results" / "coupled_phi4" / "metrics_timeseries.csv"
    if not path.exists():
        return {}
    return terminal_stats(read_rows(path))


def bank_size(arm: str) -> int | None:
    if arm.startswith("LSC-CP-RA-"):
        return int(arm.rsplit("-", 1)[1])
    return None


def score_cost(arm: str) -> int:
    """Chord energies per particle per step -- the study's cost axis."""
    A = bank_size(arm)
    if A is not None:
        return A * Q_THETA
    if arm == "LSC-CP":
        return Q_U * Q_U * Q_THETA
    return 0


def _reference_keys(study):
    return [k for k, v in study.items()
            if v["manifest"]["configuration"]["is_reference"]]


def figure_design_comparison(study, baseline, out_dir, formats):
    """Corrected vs uncorrected under each nu, against the phase-edge law."""
    keys = _reference_keys(study)
    if not keys:
        return
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.6))
    labels, groups = [], []
    if baseline:
        labels.append("phase-edge\n(manuscript)")
        groups.append({
            "uncorrected": baseline.get("CP"),
            "corrected": baseline.get("LSC-CP"),
            "bank": baseline.get("LSC-CP-MA"),
        })
    for key in sorted(keys):
        entry = study[key]
        terminal = entry["terminal"]
        design = entry["manifest"]["configuration"]["design"]
        exact = terminal.get("LSC-CP")
        labels.append(r"$\nu$-2 composed" if design == "nu2"
                      else r"$\nu$-24 (FLA-matched)")
        groups.append({
            "uncorrected": terminal.get("CP"),
            "corrected": exact,
            "bank": terminal.get("LSC-CP-RA-8"),
        })
    series = (("uncorrected", "Raw-CP", "#b0b0b0"),
              ("corrected", "LSC-CP (exact)", "#1f4e79"),
              ("bank", "LSC-CP-RA (8)", "#c0504d"))
    width = 0.26
    for ax, (key, title) in zip(axes, METRICS):
        for si, (field, label, colour) in enumerate(series):
            xs, ys, es = [], [], []
            for gi, group in enumerate(groups):
                stats = group.get(field)
                if not stats or key not in stats:
                    continue
                xs.append(gi + (si - 1) * width)
                ys.append(stats[key][0])
                es.append(stats[key][1])
            if xs:
                ax.bar(xs, ys, width, yerr=es, capsize=2, label=label,
                       color=colour)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(title)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("E4 terminal accuracy under three jump measures "
                 "(lower is better)", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, "e4_jump_design_comparison", formats)


def figure_bank_sweep(study, baseline, out_dir, formats):
    """Accuracy and cost as a function of the i.i.d. bank size A."""
    keys = sorted(_reference_keys(study))
    if not keys:
        return
    fig, axes = plt.subplots(2, len(METRICS),
                             figsize=(4.2 * len(METRICS), 6.4))
    colours = {"nu2": "#1f4e79", "nu24": "#c0504d"}
    names = {"nu2": r"$\nu$-2 composed", "nu24": r"$\nu$-24 (FLA-matched)"}
    for key in keys:
        entry = study[key]
        design = entry["manifest"]["configuration"]["design"]
        terminal = entry["terminal"]
        banks = sorted((bank_size(a), a) for a in terminal
                       if bank_size(a) is not None)
        for row, (metric, title) in enumerate(METRICS):
            ax = axes[0][row]
            xs = [A for A, arm in banks if metric in terminal[arm]]
            ys = [terminal[arm][metric][0] for A, arm in banks
                  if metric in terminal[arm]]
            es = [terminal[arm][metric][1] for A, arm in banks
                  if metric in terminal[arm]]
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=2,
                        color=colours[design], label=names[design])
            # Ground truth: the deterministic quadrature for nu-2, the
            # converged large bank for nu-24, which has no quadrature.
            truth = terminal.get("LSC-CP")
            truth_label = "exact quadrature"
            if truth is None:
                big = max((bank_size(a) for a in terminal
                           if bank_size(a) is not None), default=None)
                if big is not None and big >= 64:
                    truth = terminal.get(f"LSC-CP-RA-{big}")
                    truth_label = f"converged bank (A={big})"
            if truth and metric in truth:
                ax.axhline(truth[metric][0], ls="--", lw=1,
                           color=colours[design], alpha=0.7)
            raw = terminal.get("CP")
            if raw and metric in raw:
                ax.axhline(raw[metric][0], ls=":", lw=1,
                           color=colours[design], alpha=0.7)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("bank size $A$")
            ax.set_ylabel(title)
            ax.grid(alpha=0.3)

            axc = axes[1][row]
            cxs = [score_cost(arm) for A, arm in banks if metric in terminal[arm]]
            axc.errorbar(cxs, ys, yerr=es, marker="o", capsize=2,
                         color=colours[design], label=names[design])
            axc.set_xscale("log", base=2)
            axc.set_yscale("log")
            axc.set_xlabel("chord energies per particle per step")
            axc.set_ylabel(title)
            axc.grid(alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    seen, uniq = set(), []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            uniq.append((handle, label))
    axes[0][0].legend([h for h, _ in uniq], [l for _, l in uniq], fontsize=8)
    fig.suptitle("Bank size is an estimator knob: the jump law is identical "
                 "for every $A$\n(dashed: ground truth,  dotted: uncorrected "
                 "Raw-CP with the same $\\nu$)", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "e4_jump_design_bank_sweep", formats)


def figure_sensitivity(study, out_dir, formats):
    """Does the ranking survive a change of scale or truncation?"""
    fig, axes = plt.subplots(2, len(METRICS),
                             figsize=(4.2 * len(METRICS), 6.4))
    colours = {"nu2": "#1f4e79", "nu24": "#c0504d"}
    names = {"nu2": r"$\nu$-2 composed", "nu24": r"$\nu$-24 (FLA-matched)"}
    axis_specs = ((0, "scale", "truncation_mass", "jump-length multiplier $L$"),
                  (1, "truncation_mass", "scale", "retained tail mass $q$"))
    for arm, style in (("LSC-CP-RA-8", "-o"), ("CP", ":s")):
        for design in ("nu2", "nu24"):
            for row, varying, fixed, xlabel in axis_specs:
                entries = []
                for entry in study.values():
                    config = entry["manifest"]["configuration"]
                    if config["design"] != design:
                        continue
                    # The box-control configurations sit at the reference (q, L),
                    # so they would land on top of the reference point of both
                    # axes. They are reported separately by report_box_control.
                    if config.get("box_reach_multiplier", 1.0) != 1.0:
                        continue
                    reference_value = (1.0 if fixed == "scale" else 0.99)
                    if config[fixed] != reference_value:
                        continue
                    if arm not in entry["terminal"]:
                        continue
                    entries.append((config[varying], entry["terminal"][arm]))
                entries.sort()
                if len(entries) < 2:
                    continue
                for col, (metric, title) in enumerate(METRICS):
                    ax = axes[row][col]
                    xs = [x for x, t in entries if metric in t]
                    ys = [t[metric][0] for x, t in entries if metric in t]
                    es = [t[metric][1] for x, t in entries if metric in t]
                    ax.errorbar(xs, ys, yerr=es, fmt=style, capsize=2,
                                color=colours[design],
                                label=f"{names[design]}, {arm}")
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(title)
                    ax.set_yscale("log")
                    ax.grid(alpha=0.3)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        axes[0][0].legend(fontsize=7)
    fig.suptitle("Sensitivity to jump length and to how much of the "
                 "alpha-stable tail is retained (bank $A=8$)", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "e4_jump_design_sensitivity", formats)


def _save(fig, out_dir, stem, formats):
    for extension in formats:
        directory = out_dir / extension
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"{stem}.{extension}", dpi=600,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}." + "/".join(formats))


def write_table(study, baseline, path: Path):
    """Terminal table, the numbers the supplementary text quotes."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["design", "truncation_mass", "scale", "arm",
                         "bank_size", "chord_energies_per_particle_per_step",
                         *[f"{k}_mean" for k, _ in METRICS],
                         *[f"{k}_sd" for k, _ in METRICS]])
        for method, stats in sorted(baseline.items()):
            writer.writerow(["phase_edge_frozen", "", "", method, "", "",
                             *[stats.get(k, (math.nan, 0.0))[0] for k, _ in METRICS],
                             *[stats.get(k, (math.nan, 0.0))[1] for k, _ in METRICS]])
        for key in sorted(study):
            entry = study[key]
            config = entry["manifest"]["configuration"]
            for arm, stats in sorted(entry["terminal"].items()):
                writer.writerow([
                    config["design"], config["truncation_mass"], config["scale"],
                    arm, bank_size(arm) or "", score_cost(arm) or "",
                    *[stats.get(k, (math.nan, 0.0))[0] for k, _ in METRICS],
                    *[stats.get(k, (math.nan, 0.0))[1] for k, _ in METRICS]])
    print(f"  wrote {path}")


# The repository's own fail-closed thresholds, so the study is judged by the
# same standard as the manuscript runs rather than a bespoke one.
GATES = (
    ("score_clip_fraction_cumulative", 0.01),
    ("state_box_clip_fraction_cumulative", 0.01),
    ("jump_boundary_clip_fraction_per_applied_jump_cumulative", 0.01),
    ("basin_map_outside_mass", 0.001),
)


def report_box_control(study) -> None:
    """Reference point against the same point with a doubled box allowance.

    The sampling box is a numerical guard, not part of the model, so a result
    that moves when it widens is a result about the wall rather than about nu.
    A heavy-tailed nu keeps a real population of particles out in excursions
    where a second jump can reach that wall, which is why this control exists.
    """
    pairs = []
    for key, entry in study.items():
        config = entry["manifest"]["configuration"]
        if config.get("box_reach_multiplier", 1.0) == 1.0:
            continue
        base_key = key.rsplit("_box", 1)[0]
        if base_key in study:
            pairs.append((base_key, key))
    if not pairs:
        return
    print("\nBox-sensitivity control (reference box vs doubled jump allowance):")
    for base_key, wide_key in sorted(pairs):
        base, wide = study[base_key], study[wide_key]
        half = base["manifest"]["sampling_box_design"]["sampling_box_half_width"]
        half_wide = wide["manifest"]["sampling_box_design"]["sampling_box_half_width"]
        print(f"  {base_key}: box +/-{half:g} -> +/-{half_wide:g}")
        for arm in sorted(set(base["terminal"]) & set(wide["terminal"])):
            parts = []
            for metric, label in METRICS:
                a = base["terminal"][arm].get(metric)
                b = wide["terminal"][arm].get(metric)
                if a is None or b is None:
                    continue
                change = (b[0] - a[0]) / a[0] * 100.0 if a[0] else float("nan")
                parts.append(f"{metric} {a[0]:.4f} -> {b[0]:.4f} ({change:+.1f}%)")
            print(f"    {arm:<15} " + "  ".join(parts))


def report_gates(study) -> bool:
    """Print the worst value of each gated diagnostic per configuration and arm.

    Heavy tails are exactly what these gates exist to catch, so this is the
    first thing to read after a run.
    """
    print("\nGate diagnostics (worst value over checkpoints and seeds):")
    clean = True
    for key in sorted(study):
        entry = study[key]
        worst: dict[str, dict[str, float]] = {}
        for row in entry["rows"]:
            bucket = worst.setdefault(row["method"], {})
            for name, _ in GATES:
                value = row.get(name)
                if value in (None, ""):
                    continue
                bucket[name] = max(bucket.get(name, 0.0), float(value))
        print(f"  {key}")
        for arm in sorted(worst):
            flags = []
            for name, limit in GATES:
                value = worst[arm].get(name)
                if value is None:
                    continue
                mark = "" if value <= limit else "  ** OVER **"
                if mark:
                    clean = False
                flags.append(f"{name.split('_cumulative')[0]}={value:.2e}{mark}")
            print(f"    {arm:<15} " + "  ".join(flags))
    print("  all gated diagnostics within the repository thresholds"
          if clean else "  SOME DIAGNOSTICS EXCEED THEIR THRESHOLD")
    return clean


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="production")
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--figure-root", default=None)
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--gates-only", action="store_true",
                        help="print the gate diagnostics and stop")
    args = parser.parse_args(argv)

    root = Path(args.results_root) if args.results_root else (
        HERE / "results" / "e4_jump_design" / args.stage)
    if not root.exists():
        print(f"no results at {root}", file=sys.stderr)
        return 1
    out_dir = Path(args.figure_root) if args.figure_root else (
        HERE / "figures" / "e4_jump_design" / args.stage)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    study = load_study(root)
    baseline = load_baseline(HERE)
    print(f"{len(study)} configurations from {root}")
    report_gates(study)
    report_box_control(study)
    if args.gates_only:
        return 0
    figure_design_comparison(study, baseline, out_dir, formats)
    figure_bank_sweep(study, baseline, out_dir, formats)
    figure_sensitivity(study, out_dir, formats)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(study, baseline, out_dir / "e4_jump_design_terminal.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
