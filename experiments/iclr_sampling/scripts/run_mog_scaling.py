"""MoG mode-count-scaling experiment.

    python -m experiments.iclr_sampling.scripts.run_mog_scaling \
        --config experiments/iclr_sampling/configs/mog_scaling.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from experiments.iclr_sampling.scripts import _common as C
from experiments.iclr_sampling import plotting, reporting

REPORT_FIG = "reports/iclr_sampling_report/figures"
REPORT_TAB = "reports/iclr_sampling_report/tables"

METRIC_LABELS = {
    "w2_or_sliced_w2": "$W_2$ (exact OT)", "mmd": "MMD",
    "mode_coverage": "mode coverage", "n_covered_modes": "# covered modes",
    "emc": "EMC = $e^{-KL}$", "mode_kl": "mode-weight KL",
    "emc_notebook": "EMC (notebook)", "runtime_sec": "runtime (s)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir, log, env = C.setup_run(cfg, args.config, "mog_scaling")

    raw_rows, curves, finals, refs, targets = C.run_scaling(cfg, run_dir, log, device)
    df, summary = C.finalize(run_dir, raw_rows, "n_modes", log)

    methods = cfg["methods"]
    fig_dir = run_dir / "figures"
    figs = []

    figs.append(C.safe(log, plotting.scaling_plot,
        summary, "n_modes",
        ["w2_or_sliced_w2", "mmd", "mode_coverage", "n_covered_modes",
         "emc", "mode_kl"],
        METRIC_LABELS, methods, fig_dir, "mog_scaling_metrics",
        "MoG mode-count scaling", x_label="number of modes $K$"))

    figs.append(C.safe(log, plotting.compute_quality_plot,
        summary, "runtime_sec_mean", "emc", "EMC", methods, fig_dir,
        "mog_compute_quality", "MoG: EMC vs wall-clock", group_col="n_modes",
        compute_label="runtime (s)"))

    figs.append(C.safe(log, plotting.acceptance_plot,
        summary, "n_modes", methods, fig_dir, "mog_acceptance",
        "MoG: acceptance / swap rates", x_label="number of modes $K$"))

    figs.append(C.safe(log, plotting.convergence_plot,
        curves, "coverage", "mode coverage", methods, fig_dir, "mog_convergence",
        "MoG: mode coverage over time"))

    # scatter panels + mode-weight bars for the headline K=40 (and any present)
    headline = 40 if 40 in finals else cfg["scaling"]["values"][-1]
    tgt = targets[headline]
    mu = tgt.mu.detach().cpu().numpy()
    ref = refs[headline].detach().cpu().numpy()
    figs.append(C.safe(log, plotting.mog_scatter_panels,
        finals[headline], mu, ref, fig_dir, f"mog_scatter_K{headline}",
        f"MoG (K={headline}): final samples", methods))

    weights = {}
    for m, X in finals[headline].items():
        Xt = torch.tensor(X, device=device)
        labels = tgt.assign_modes(Xt)
        cnt = torch.bincount(labels.reshape(-1), minlength=tgt.n_modes).float()
        weights[m] = (cnt / cnt.sum()).cpu().numpy()
    figs.append(C.safe(log, plotting.mode_weight_bars,
        weights, 1.0 / tgt.n_modes, fig_dir, f"mog_modeweights_K{headline}",
        f"MoG (K={headline}): per-mode empirical weights vs uniform", methods))

    # ---- LaTeX tables ---- #
    tab_dir = run_dir / "report_artifacts"
    table_paths = []
    metric_cols = ["w2_or_sliced_w2", "mmd", "mode_coverage", "emc", "mode_kl",
                   "runtime_sec", "grad_evals"]
    headers = {"w2_or_sliced_w2": "$W_2$", "mmd": "MMD", "mode_coverage": "cov.",
               "emc": "EMC", "mode_kl": "KL", "runtime_sec": "time(s)",
               "grad_evals": "grads"}
    for K in cfg["scaling"]["values"]:
        tex = reporting.latex_table(
            summary, "n_modes", K, methods, metric_cols, headers,
            caption=f"MoG results at $K={K}$ modes (mean$\\pm$SE over "
                    f"{len(cfg['run']['seeds'])} seeds).",
            label=f"tab:mog_K{K}")
        p = tab_dir / f"mog_table_K{K}.tex"
        reporting.write_text(p, tex)
        table_paths.append(p)
    tex = reporting.scaling_latex_table(
        summary, "n_modes", "emc", methods, cfg["scaling"]["values"],
        caption="MoG EMC across mode counts (mean$\\pm$SE).",
        label="tab:mog_emc_scaling", config_header="K")
    p = tab_dir / "mog_emc_scaling.tex"
    reporting.write_text(p, tex)
    table_paths.append(p)

    C.copy_to_report(run_dir, figs, REPORT_FIG, table_paths, REPORT_TAB)
    write_readme(run_dir, cfg, env, args.config)
    log(f"\nDONE. Outputs in {run_dir}")
    print(str(run_dir))


def write_readme(run_dir, cfg, env, cfg_path):
    lines = [
        "# MoG mode-count-scaling run", "",
        f"- Config: `{cfg_path}`", f"- Methods: {cfg['methods']}",
        f"- Mode counts K: {cfg['scaling']['values']}",
        f"- Seeds: {cfg['run']['seeds']}",
        f"- GPU: CUDA_VISIBLE_DEVICES={env.get('cuda_visible_devices')} "
        f"({env.get('gpu_name','?')})", "",
        "## Command", "```",
        "python -m experiments.iclr_sampling.scripts.run_mog_scaling "
        f"--config {cfg_path}", "```", "",
        "## Outputs",
        "- `raw_runs.csv`, `summary_by_config.csv`",
        "- `figures/` (png+pdf), `report_artifacts/` (tables, ref npz), "
        "`logs/run.log`",
    ]
    reporting.write_text(run_dir / "README.md", "\n".join(lines))


if __name__ == "__main__":
    main()
