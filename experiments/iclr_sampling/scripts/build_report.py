"""Populate the executive summary + appendix from the summary CSVs and compile.

    python -m experiments.iclr_sampling.scripts.build_report

Finds the latest run directory for each experiment, writes
``reports/iclr_sampling_report/{exec_summary,appendix_configs}.tex`` with real
numbers, ensures the figures/tables are copied in, and compiles with tectonic.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPORT = Path("reports/iclr_sampling_report")
RESULTS = "results/iclr_sampling"


def latest(tag: str):
    ds = sorted(glob.glob(f"{RESULTS}/*_{tag}"))
    return Path(ds[-1]) if ds else None


def load_summary(run_dir):
    if run_dir is None:
        return None
    p = run_dir / "summary_by_config.csv"
    return pd.read_csv(p) if p.exists() else None


def _row(summary, col, val, method):
    sub = summary[(summary[col] == val) & (summary["method"] == method)]
    return sub.iloc[0] if not sub.empty else None


def _g(row, key, default=float("nan")):
    if row is None or key not in row:
        return default
    v = row[key]
    return float(v) if v == v else default


def _fmt(x, nd=2):
    if x is None or x != x:
        return "n/a"
    return f"{x:.{nd}f}"


def _num(summary, col, val, method, key, nd=2):
    """Formatted single value pulled from a summary CSV (mean column)."""
    return _fmt(_g(_row(summary, col, val, method), key), nd)


def copy_artifacts(run_dir):
    if run_dir is None:
        return
    # The report prefers tables over plots: only the time-evolution
    # (convergence) figures are carried into the report; the rest of the
    # information lives in the tables below.
    for p in (run_dir / "figures").glob("*_convergence.*"):
        shutil.copy(p, REPORT / "figures" / p.name)
    for p in (run_dir / "report_artifacts").glob("*.tex"):
        shutil.copy(p, REPORT / "tables" / p.name)


def build_exec_summary(mw, mog, bg):
    """Three short plain-language paragraphs with live headline numbers."""
    P = []
    P.append(
        "Across all three studies the picture is the same. The local samplers "
        "(ULA, MALA, FLMC, ULD, HMC) stay trapped near wherever they were "
        "started and never discover the rest of the distribution; parallel "
        "tempering (PT), the strongest generic baseline, recovers partial "
        "coverage but degrades steadily as the problem grows; and "
        "\\textbf{LSB-MC reaches essentially the whole distribution}, attaining "
        "the best mode coverage and EMC and the smallest mode-weight KL, "
        "typically sitting at the finite-sample reference floor.")

    if mog is not None:
        P.append(
            "\\textbf{Discovering many modes (MoG).} With $K{=}40$ equally "
            "weighted Gaussian modes and all particles started near one of "
            f"them, LSB-MC covers {_num(mog,'n_modes',40,'LSBMC','mode_coverage_mean')} "
            "of the modes with EMC "
            f"{_num(mog,'n_modes',40,'LSBMC','emc_mean')} (1.0 is perfect), "
            "while the best baseline (PT) reaches only coverage "
            f"{_num(mog,'n_modes',40,'PT','mode_coverage_mean')} and EMC "
            f"{_num(mog,'n_modes',40,'PT','emc_mean')}; the corresponding "
            f"Wasserstein distance is {_num(mog,'n_modes',40,'LSBMC','w2_or_sliced_w2_mean')} "
            f"for LSB-MC versus {_num(mog,'n_modes',40,'PT','w2_or_sliced_w2_mean')} for PT.")

    if mw is not None:
        P.append(
            "\\textbf{Scaling to high dimension (ManyWell).} At $d{=}64$ "
            "(32 independent double wells), LSB-MC keeps a count-EMC of "
            f"{_num(mw,'n_blocks',32,'LSBMC','count_emc_mean')} and a per-block "
            f"marginal KL of {_num(mw,'n_blocks',32,'LSBMC','block_marginal_kl_mean',3)}, "
            "i.e.\\ each well is filled in almost exactly the right proportion, "
            "whereas PT and HMC have collapsed to count-EMC "
            f"$\\approx{_num(mw,'n_blocks',32,'PT','count_emc_mean')}$; the "
            f"sliced-Wasserstein distance is {_num(mw,'n_blocks',32,'LSBMC','w2_or_sliced_w2_mean')} "
            f"for LSB-MC versus {_num(mw,'n_blocks',32,'PT','w2_or_sliced_w2_mean')} for PT.")

    if bg is not None:
        P.append(
            "\\textbf{A real Bayesian model (label switching).} The posterior "
            "of a 3-component Gaussian mixture has six identical, relabelled "
            "copies. Every baseline finds only the one copy it started in "
            f"(coverage {_num(bg,'K',3,'PT','mode_coverage_mean')}, 1 of 6 modes, "
            f"EMC {_num(bg,'K',3,'PT','emc_mean',3)}), whereas LSB-MC populates "
            f"all six uniformly (coverage {_num(bg,'K',3,'LSBMC','mode_coverage_mean')}, "
            f"EMC {_num(bg,'K',3,'LSBMC','emc_mean',3)}) using a jump bank read "
            "directly off the label-permutation symmetry.")

    P.append(
        "Compute is accounted for fairly (wall-clock time and gradient "
        "evaluations, with PT and HMC charged their extra work and LSB-MC "
        "charged its Lévy-score quadrature), so the advantage is not an "
        "artefact of spending more compute --- though, as Section~\\ref{sec:compute} "
        "shows, that quadrature does add a real and problem-dependent per-step cost.")
    return "\n\n".join(P)


# Live macros consumed by main.tex prose so the narrative never goes stale.
NUMBER_KEYS = [
    "MogLSBcovK", "MogLSBemcK", "MogPTcovK", "MogPTemcK", "MogHMCcovK",
    "MogLSBwtwoK", "MogPTwtwoK", "MogLSBemcSmall", "MogLSBemcBig",
    "MogLSBruntimeBig", "MogPTruntimeBig", "MogLSBpotBig",
    "MwLSBcemcBig", "MwPTcemcBig", "MwLSBcemcSmall", "MwPTcemcSmall",
    "MwLSBblockklBig", "MwPTblockklBig", "MwLSBswtwoBig", "MwPTswtwoBig",
    "MwLSBruntimeBig", "MwPTruntimeBig",
    "BgLSBcov", "BgLSBmodes", "BgPTcov", "BgLSBemc", "BgPTemc",
    "BgLSBwtwo", "BgBaseWtwo", "BgLSBruntime",
]


def build_numbers(mw, mog, bg):
    """Emit \\newcommand macros with live headline numbers for main.tex."""
    v = {k: "n/a" for k in NUMBER_KEYS}

    if mog is not None:
        v["MogLSBcovK"] = _num(mog, "n_modes", 40, "LSBMC", "mode_coverage_mean")
        v["MogLSBemcK"] = _num(mog, "n_modes", 40, "LSBMC", "emc_mean")
        v["MogPTcovK"] = _num(mog, "n_modes", 40, "PT", "mode_coverage_mean")
        v["MogPTemcK"] = _num(mog, "n_modes", 40, "PT", "emc_mean")
        v["MogHMCcovK"] = _num(mog, "n_modes", 40, "HMC", "mode_coverage_mean")
        v["MogLSBwtwoK"] = _num(mog, "n_modes", 40, "LSBMC", "w2_or_sliced_w2_mean")
        v["MogPTwtwoK"] = _num(mog, "n_modes", 40, "PT", "w2_or_sliced_w2_mean")
        v["MogLSBemcSmall"] = _num(mog, "n_modes", 10, "LSBMC", "emc_mean")
        v["MogLSBemcBig"] = _num(mog, "n_modes", 80, "LSBMC", "emc_mean")
        v["MogLSBruntimeBig"] = _num(mog, "n_modes", 80, "LSBMC", "runtime_sec_mean", 1)
        v["MogPTruntimeBig"] = _num(mog, "n_modes", 80, "PT", "runtime_sec_mean", 1)
        pot = _g(_row(mog, "n_modes", 80, "LSBMC"), "pot_evals_mean")
        v["MogLSBpotBig"] = "n/a" if pot != pot else f"{pot / 1e6:.1f}"

    if mw is not None:
        v["MwLSBcemcBig"] = _num(mw, "n_blocks", 32, "LSBMC", "count_emc_mean")
        v["MwPTcemcBig"] = _num(mw, "n_blocks", 32, "PT", "count_emc_mean")
        v["MwLSBcemcSmall"] = _num(mw, "n_blocks", 4, "LSBMC", "count_emc_mean", 3)
        v["MwPTcemcSmall"] = _num(mw, "n_blocks", 4, "PT", "count_emc_mean")
        v["MwLSBblockklBig"] = _num(mw, "n_blocks", 32, "LSBMC", "block_marginal_kl_mean", 3)
        v["MwPTblockklBig"] = _num(mw, "n_blocks", 32, "PT", "block_marginal_kl_mean", 3)
        v["MwLSBswtwoBig"] = _num(mw, "n_blocks", 32, "LSBMC", "w2_or_sliced_w2_mean")
        v["MwPTswtwoBig"] = _num(mw, "n_blocks", 32, "PT", "w2_or_sliced_w2_mean")
        v["MwLSBruntimeBig"] = _num(mw, "n_blocks", 32, "LSBMC", "runtime_sec_mean", 1)
        v["MwPTruntimeBig"] = _num(mw, "n_blocks", 32, "PT", "runtime_sec_mean", 1)

    if bg is not None:
        v["BgLSBcov"] = _num(bg, "K", 3, "LSBMC", "mode_coverage_mean")
        m = _g(_row(bg, "K", 3, "LSBMC"), "n_covered_modes_mean")
        v["BgLSBmodes"] = "n/a" if m != m else f"{m:.0f}"
        v["BgPTcov"] = _num(bg, "K", 3, "PT", "mode_coverage_mean")
        v["BgLSBemc"] = _num(bg, "K", 3, "LSBMC", "emc_mean", 3)
        v["BgPTemc"] = _num(bg, "K", 3, "PT", "emc_mean", 3)
        v["BgLSBwtwo"] = _num(bg, "K", 3, "LSBMC", "w2_or_sliced_w2_mean")
        v["BgBaseWtwo"] = _num(bg, "K", 3, "MALA", "w2_or_sliced_w2_mean")
        v["BgLSBruntime"] = _num(bg, "K", 3, "LSBMC", "runtime_sec_mean", 0)

    return "\n".join(f"\\newcommand{{\\{k}}}{{{v[k]}}}" for k in NUMBER_KEYS)


METHODS_ORDER = ["ULA", "MALA", "FLMC", "LSBMC", "ULD", "HMC", "PT"]
# (title, summary, scaling column, largest value) for the headline settings
def _largest_blocks(mw, mog, bg):
    return [("MoG, $K=80$", mog, "n_modes", 80),
            ("ManyWell, $d=64$", mw, "n_blocks", 32),
            ("Bayesian GMM, $K=3$", bg, "K", 3)]


def build_compute_table(mw, mog, bg):
    """Per-method compute at the largest setting of each study."""
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Compute at the largest setting of each study (mean over 5 "
             r"seeds): wall-clock time, gradient evaluations, and L\'evy-score "
             r"potential evaluations (millions). HMC is charged $L{+}1$ gradients "
             r"per proposal, PT $T$ kernels per step, LSB-MC its quadrature.}",
             r"\label{tab:compute}",
             r"\begin{tabular}{lrrr}", r"\toprule",
             r"Method & time (s) & grad evals & pot.\ evals (M) \\"]
    for title, summ, col, val in _largest_blocks(mw, mog, bg):
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{4}}{{l}}{{\textit{{{title}}}}}\\")
        if summ is None:
            continue
        for m in METHODS_ORDER:
            row = _row(summ, col, val, m)
            t, g, p = (_g(row, "runtime_sec_mean"), _g(row, "grad_evals_mean"),
                       _g(row, "pot_evals_mean"))
            ts = "--" if t != t else f"{t:.2f}"
            gs = "--" if g != g else f"{g:.0f}"
            ps = "--" if p != p else f"{p / 1e6:.3f}"
            lines.append(rf"{m} & {ts} & {gs} & {ps} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_acceptance_table(mw, mog, bg):
    """Metropolis acceptance and PT swap rates at the largest setting."""
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\caption{Metropolis--Hastings acceptance and PT swap-acceptance "
             r"rates at the largest setting of each study (mean over 5 seeds); "
             r"``--'' = not applicable to the method.}",
             r"\label{tab:acceptance}",
             r"\begin{tabular}{lrr}", r"\toprule",
             r"Method & MH acc. & PT swap \\"]
    for title, summ, col, val in _largest_blocks(mw, mog, bg):
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{3}}{{l}}{{\textit{{{title}}}}}\\")
        if summ is None:
            continue
        for m in METHODS_ORDER:
            row = _row(summ, col, val, m)
            a, s = _g(row, "acceptance_rate_mean"), _g(row, "swap_acceptance_rate_mean")
            asr = "--" if (a != a or a == 0.0) else f"{a:.2f}"
            ssr = "--" if (s != s or s == 0.0) else f"{s:.2f}"
            lines.append(rf"{m} & {asr} & {ssr} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def build_appendix(dirs):
    lines = ["\\small"]
    for tag, run_dir in dirs.items():
        if run_dir is None:
            continue
        cfgp = run_dir / "configs" / "experiment_config.json"
        if not cfgp.exists():
            continue
        cfg = json.load(open(cfgp))
        run = cfg.get("run", {})
        lines.append(f"\\paragraph{{{tag.replace('_',' ')}}}")
        lines.append("\\begin{itemize}")
        var = cfg.get('scaling', {}).get('var')
        lines.append(f"\\item scaling: \\texttt{{\\detokenize{{{var}}}}} "
                     f"$\\in$ {cfg.get('scaling',{}).get('values')}")
        lines.append(f"\\item methods: {', '.join(cfg.get('methods',[]))}")
        lines.append(f"\\item particles {run.get('n_particles')}, steps "
                     f"{run.get('n_steps')}, dt {run.get('dt')}, seeds "
                     f"{run.get('seeds')}")
        mc = cfg.get("method_cfgs", {})
        if mc:
            mcs = "; ".join(f"{k}: {v}" for k, v in mc.items())
            lines.append(f"\\item method configs: \\texttt{{\\detokenize{{{mcs}}}}}")
        tc = cfg.get("target_cfg", {})
        if tc:
            lines.append(f"\\item target: \\texttt{{\\detokenize{{{tc}}}}}")
        lines.append(f"\\item output: \\texttt{{\\detokenize{{{run_dir}}}}}")
        lines.append("\\end{itemize}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manywell", default=None)
    ap.add_argument("--mog", default=None)
    ap.add_argument("--bayes", default=None)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    mw_dir = Path(args.manywell) if args.manywell else latest("manywell_scaling")
    mog_dir = Path(args.mog) if args.mog else latest("mog_scaling")
    bg_dir = Path(args.bayes) if args.bayes else latest("bayes_gmm_label_switching")
    print(f"manywell: {mw_dir}\nmog: {mog_dir}\nbayes: {bg_dir}")

    for d in (mw_dir, mog_dir, bg_dir):
        copy_artifacts(d)

    mw, mog, bg = load_summary(mw_dir), load_summary(mog_dir), load_summary(bg_dir)
    (REPORT / "exec_summary.tex").write_text(build_exec_summary(mw, mog, bg))
    (REPORT / "numbers.tex").write_text(build_numbers(mw, mog, bg))
    (REPORT / "tables" / "compute_table.tex").write_text(build_compute_table(mw, mog, bg))
    (REPORT / "tables" / "acceptance_table.tex").write_text(build_acceptance_table(mw, mog, bg))
    (REPORT / "appendix_configs.tex").write_text(build_appendix(
        {"manywell_scaling": mw_dir, "mog_scaling": mog_dir,
         "bayes_gmm_label_switching": bg_dir}))
    print("wrote exec_summary.tex, numbers.tex, compute/acceptance tables and appendix_configs.tex")

    if not args.no_compile and shutil.which("tectonic"):
        out = subprocess.run(["tectonic", "main.tex"], cwd=REPORT,
                             capture_output=True, text=True)
        ok = (REPORT / "main.pdf").exists() and out.returncode == 0
        print("tectonic:", "OK" if ok else "FAILED")
        if not ok:
            print(out.stderr[-1500:])
    print(f"PDF: {REPORT / 'main.pdf'}")


if __name__ == "__main__":
    main()
