"""ManyWell dimension-scaling experiment.

    python -m experiments.iclr_sampling.scripts.run_manywell_scaling \
        --config experiments/iclr_sampling/configs/manywell_scaling.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.iclr_sampling.scripts import _common as C
from experiments.iclr_sampling import plotting, reporting

REPORT_FIG = "reports/iclr_sampling_report/figures"
REPORT_TAB = "reports/iclr_sampling_report/tables"

METRIC_LABELS = {
    "w2_or_sliced_w2": "Sliced $W_2$", "mmd": "MMD",
    "count_emc": "count-EMC", "count_mode_kl": "count-mode KL",
    "block_marginal_kl": "block marginal KL", "mode_coverage": "count coverage",
    "emc_notebook": "config-EMC (notebook)", "runtime_sec": "runtime (s)",
    "grad_evals": "grad evals",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir, log, env = C.setup_run(cfg, args.config, "manywell_scaling")

    raw_rows, curves, finals, refs, targets = C.run_scaling(cfg, run_dir, log, device)
    df, summary = C.finalize(run_dir, raw_rows, "n_blocks", log)
    summary["dimension"] = (2 * summary["n_blocks"]).astype(int)

    methods = cfg["methods"]
    fig_dir = run_dir / "figures"
    figs = []

    figs.append(C.safe(log, plotting.scaling_plot,
        summary, "dimension",
        ["w2_or_sliced_w2", "mmd", "count_emc", "count_mode_kl",
         "block_marginal_kl", "mode_coverage"],
        METRIC_LABELS, methods, fig_dir, "manywell_scaling_metrics",
        "ManyWell dimension scaling", x_label="dimension $d$"))

    figs.append(C.safe(log, plotting.compute_quality_plot,
        summary, "runtime_sec_mean", "count_emc", "count-EMC", methods,
        fig_dir, "manywell_compute_quality",
        "ManyWell: count-EMC vs wall-clock", group_col="dimension",
        compute_label="runtime (s)"))

    figs.append(C.safe(log, plotting.compute_quality_plot,
        summary, "grad_evals_mean", "w2_or_sliced_w2", "Sliced $W_2$", methods,
        fig_dir, "manywell_gradevals_quality",
        "ManyWell: Sliced $W_2$ vs gradient evaluations", group_col="dimension",
        compute_label="gradient evaluations"))

    figs.append(C.safe(log, plotting.acceptance_plot,
        summary, "dimension", methods, fig_dir, "manywell_acceptance",
        "ManyWell: acceptance / swap rates", x_label="dimension $d$"))

    figs.append(C.safe(log, plotting.convergence_plot,
        curves, "emc", "count-EMC", methods, fig_dir, "manywell_convergence",
        "ManyWell: count-EMC over time"))

    # ---- LaTeX tables ---- #
    tab_dir = run_dir / "report_artifacts"
    table_paths = []
    metric_cols = ["w2_or_sliced_w2", "mmd", "count_emc", "count_mode_kl",
                   "block_marginal_kl", "runtime_sec", "grad_evals"]
    headers = {"w2_or_sliced_w2": "Sliced $W_2$", "mmd": "MMD",
               "count_emc": "count-EMC", "count_mode_kl": "count-KL",
               "block_marginal_kl": "block-KL", "runtime_sec": "time(s)",
               "grad_evals": "grads"}
    for nb in cfg["scaling"]["values"]:
        tex = reporting.latex_table(
            summary, "n_blocks", nb, methods, metric_cols, headers,
            caption=f"ManyWell results at $d={2*nb}$ (mean$\\pm$SE over "
                    f"{len(cfg['run']['seeds'])} seeds).",
            label=f"tab:manywell_d{2*nb}")
        p = tab_dir / f"manywell_table_d{2*nb}.tex"
        reporting.write_text(p, tex)
        table_paths.append(p)
    # scaling table: count-EMC across dimensions
    tex = reporting.scaling_latex_table(
        summary, "dimension", "count_emc", methods,
        [2 * nb for nb in cfg["scaling"]["values"]],
        caption="ManyWell count-EMC across dimensions (mean$\\pm$SE).",
        label="tab:manywell_countemc_scaling", config_header="d")
    p = tab_dir / "manywell_countemc_scaling.tex"
    reporting.write_text(p, tex)
    table_paths.append(p)

    C.copy_to_report(run_dir, figs, REPORT_FIG, table_paths, REPORT_TAB)
    write_readme(run_dir, cfg, env, args.config, log)
    log(f"\nDONE. Outputs in {run_dir}")
    print(str(run_dir))


def write_readme(run_dir, cfg, env, cfg_path, log):
    lines = [
        "# ManyWell dimension-scaling run", "",
        f"- Config: `{cfg_path}`",
        f"- Methods: {cfg['methods']}",
        f"- Dimensions: {[2*v for v in cfg['scaling']['values']]} "
        f"(n_blocks {cfg['scaling']['values']})",
        f"- Seeds: {cfg['run']['seeds']}",
        f"- GPU: CUDA_VISIBLE_DEVICES={env.get('cuda_visible_devices')} "
        f"({env.get('gpu_name','?')})",
        "",
        "## Command",
        "```",
        "python -m experiments.iclr_sampling.scripts.run_manywell_scaling "
        f"--config {cfg_path}",
        "```",
        "",
        "## Outputs",
        "- `raw_runs.csv` — per-seed metrics", "- `summary_by_config.csv` — mean/std/SE",
        "- `figures/` — png+pdf", "- `report_artifacts/` — LaTeX tables, reference npz",
        "- `logs/run.log` — full log",
    ]
    reporting.write_text(run_dir / "README.md", "\n".join(lines))


if __name__ == "__main__":
    main()
