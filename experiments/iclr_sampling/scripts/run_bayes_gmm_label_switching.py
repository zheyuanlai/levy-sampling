"""Bayesian Gaussian-mixture label-switching posterior experiment.

    python -m experiments.iclr_sampling.scripts.run_bayes_gmm_label_switching \
        --config experiments/iclr_sampling/configs/bayes_gmm_label_switching.yaml
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
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
    "runtime_sec": "runtime (s)",
}


def energy_histogram(target, finals, ref, device, fig_dir, methods):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    with torch.no_grad():
        Vref = target.potential(ref).detach().cpu().numpy()
        ax.hist(Vref, bins=60, density=True, color="0.6", alpha=0.5,
                label="reference")
        for m in methods:
            if m not in finals:
                continue
            X = torch.tensor(finals[m], device=device)
            V = target.potential(X).detach().cpu().numpy()
            ax.hist(V, bins=60, density=True, histtype="step", lw=1.6,
                    color=plotting.METHOD_COLORS.get(m), label=m)
    ax.set_xlabel("posterior energy $V(\\theta)$"); ax.set_ylabel("density")
    ax.set_title("Bayesian GMM: posterior energy distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return plotting.save_fig(fig, fig_dir, "bayes_gmm_energy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir, log, env = C.setup_run(cfg, args.config, "bayes_gmm_label_switching")

    raw_rows, curves, finals_all, refs, targets = C.run_scaling(cfg, run_dir, log, device)
    df, summary = C.finalize(run_dir, raw_rows, "K", log)

    methods = cfg["methods"]
    K = cfg["scaling"]["values"][0]
    tgt = targets[K]
    ref = refs[K]
    finals = finals_all[K]
    fig_dir = run_dir / "figures"

    # save the synthetic data, MAP, and permutation modes
    np.savez(run_dir / "report_artifacts" / "bayes_gmm_setup.npz",
             y=tgt.y.cpu().numpy(), true_mu=tgt.true_mu,
             map_theta=tgt.map_theta.cpu().numpy(),
             mode_centers=tgt.mode_centers.cpu().numpy(),
             perms=np.array(tgt.perms))
    log(f"  MAP energy = {getattr(tgt, 'map_energy', float('nan')):.3f}; "
        f"K! = {tgt.n_modes} modes; jump bank = {tgt.jump_bank_size}")

    figs = []
    figs.append(C.safe(log, plotting.scaling_plot,
        summary, "K",
        ["w2_or_sliced_w2", "mmd", "mode_coverage", "emc", "mode_kl"],
        METRIC_LABELS, methods, fig_dir, "bayes_gmm_metrics",
        f"Bayesian GMM (K={K}) metrics", x_label="K"))

    # mode-weight bars over the K! permutation modes
    weights = {}
    for m, X in finals.items():
        Xt = torch.tensor(X, device=device)
        labels = tgt.assign_modes(Xt)
        cnt = torch.bincount(labels.reshape(-1), minlength=tgt.n_modes).float()
        weights[m] = (cnt / cnt.sum()).cpu().numpy()
    figs.append(C.safe(log, plotting.mode_weight_bars,
        weights, 1.0 / tgt.n_modes, fig_dir, "bayes_gmm_modeweights",
        f"Bayesian GMM: empirical weight on each of the {tgt.n_modes} "
        f"label-permutation modes vs uniform", methods))

    figs.append(C.safe(log, energy_histogram, tgt, finals, ref, device, fig_dir, methods))

    figs.append(C.safe(log, plotting.convergence_plot,
        curves, "coverage", "mode coverage", methods, fig_dir,
        "bayes_gmm_convergence", "Bayesian GMM: mode coverage over time"))

    figs.append(C.safe(log, plotting.acceptance_plot,
        summary, "K", methods, fig_dir, "bayes_gmm_acceptance",
        "Bayesian GMM: acceptance / swap rates", x_label="K"))

    # ---- LaTeX table ---- #
    tab_dir = run_dir / "report_artifacts"
    metric_cols = ["mode_coverage", "n_covered_modes", "emc", "mode_kl",
                   "w2_or_sliced_w2", "mmd", "runtime_sec", "grad_evals"]
    headers = {"mode_coverage": "cov.", "n_covered_modes": "\\#modes",
               "emc": "EMC", "mode_kl": "KL", "w2_or_sliced_w2": "$W_2$",
               "mmd": "MMD", "runtime_sec": "time(s)", "grad_evals": "grads"}
    tex = reporting.latex_table(
        summary, "K", K, methods, metric_cols, headers,
        caption=f"Bayesian GMM label-switching posterior (K={K}, dim={tgt.dim}, "
                f"{tgt.n_modes} modes; mean$\\pm$SE over "
                f"{len(cfg['run']['seeds'])} seeds).",
        label="tab:bayes_gmm")
    p = tab_dir / "bayes_gmm_table.tex"
    reporting.write_text(p, tex)

    C.copy_to_report(run_dir, figs, REPORT_FIG, [p], REPORT_TAB)
    write_readme(run_dir, cfg, env, args.config, tgt)
    log(f"\nDONE. Outputs in {run_dir}")
    print(str(run_dir))


def write_readme(run_dir, cfg, env, cfg_path, tgt):
    lines = [
        "# Bayesian GMM label-switching run", "",
        f"- Config: `{cfg_path}`", f"- Methods: {cfg['methods']}",
        f"- K={tgt.K}, p={tgt.p}, posterior dim={tgt.dim}, modes (K!)={tgt.n_modes}",
        f"- Jump bank (permutation differences): {tgt.jump_bank_size}",
        f"- Seeds: {cfg['run']['seeds']}",
        f"- GPU: CUDA_VISIBLE_DEVICES={env.get('cuda_visible_devices')} "
        f"({env.get('gpu_name','?')})", "",
        "## Reference",
        "High-quality reference = a long MALA chain confined to one mode, then "
        "replicated across all K! label permutations (the posterior is exactly "
        "label-symmetric). Cached in `report_artifacts/ref_*.npz`.", "",
        "## Command", "```",
        "python -m experiments.iclr_sampling.scripts.run_bayes_gmm_label_switching "
        f"--config {cfg_path}", "```", "",
        "## Outputs",
        "- `raw_runs.csv`, `summary_by_config.csv`",
        "- `figures/` (png+pdf), `report_artifacts/` (tables, setup npz, ref npz)",
        "- `logs/run.log`",
    ]
    reporting.write_text(run_dir / "README.md", "\n".join(lines))


if __name__ == "__main__":
    main()
