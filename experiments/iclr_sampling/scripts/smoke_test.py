"""Smoke test: tiny runs of every target x sampler, plus a tectonic compile check.

    python -m experiments.iclr_sampling.scripts.smoke_test

Verifies: no NaNs in final samples, every metric executes, CSV + a figure are
written, and the LaTeX report skeleton compiles with tectonic (if installed).
"""
from __future__ import annotations

import math
import shutil
import subprocess
import sys

import numpy as np
import torch

from experiments.iclr_sampling.scripts import _common as C
from experiments.iclr_sampling import plotting, reporting
from experiments.iclr_sampling.experiment import run_target_experiment


TINY_RUN = {
    "n_particles": 128, "n_steps": 30, "dt": 0.005, "metric_every": 10,
    "seeds": [0, 1], "base_seed": 0, "n_ref": 1000,
    "w2_sub": 64, "mmd_sub": 128, "n_proj": 50,
}
METHODS = ["ULA", "MALA", "FLMC", "LSBMC", "ULD", "HMC", "PT"]
METHOD_CFGS = {
    "FLMC": {"alpha": 1.5}, "ULD": {"dt": 0.01, "friction": 2.0},
    "HMC": {"dt": 0.03, "n_leapfrog": 5},
    "PT": {"n_temps": 4, "beta_min": 0.1, "swap_interval": 3},
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {"experiment_name": "smoke_test", "target": "smoke",
           "scaling": {"var": "case", "values": ["all"]},
           "methods": METHODS, "run": TINY_RUN}
    run_dir, log, env = C.setup_run(cfg, "(in-code)", "smoke_test")

    from experiments.iclr_sampling.targets import (
        ManyWellTarget, MoGTarget, BayesGMMTarget)
    targets = {
        "manywell_d8": ManyWellTarget(n_blocks=4, device=device),
        "mog_K10": MoGTarget(n_modes=10, device=device, n_theta=4, k_neighbors=4),
        "bayes_gmm": BayesGMMTarget(K=3, p=2, n_data=120, device=device, n_theta=4),
    }

    all_rows = []
    failures = []
    for tname, target in targets.items():
        log(f"\n--- smoke: {tname} (dim={target.dim}, "
            f"bank={target.jump_bank_size}) ---")
        ref = target.sample_reference(TINY_RUN["n_ref"],
                                      TINY_RUN.get("ref_seed", 1), device)
        res = run_target_experiment(target, METHODS, METHOD_CFGS, TINY_RUN,
                                    device, log_fn=log, ref=ref)
        for row in res["rows"]:
            row["experiment_name"] = "smoke_test"
            row["target_name"] = tname
            all_rows.append(row)
            # validation: metrics finite where applicable
            for key in ("w2_or_sliced_w2", "mmd"):
                v = row.get(key, float("nan"))
                if v != v:
                    failures.append(f"{tname}/{row['method']}: {key} is NaN")
            if row.get("nonfinite_count", 0) > 0:
                failures.append(
                    f"{tname}/{row['method']}: {row['nonfinite_count']} nonfinite states")

    df = reporting.write_raw_csv(all_rows, run_dir / "raw_runs.csv")
    summary = reporting.summarize(df, ["target_name", "method"],
                                  run_dir / "summary_by_config.csv")
    log(f"\nwrote raw_runs.csv ({len(df)} rows) and summary_by_config.csv")

    # one figure to exercise the plotting + save path
    fig = plotting.scaling_plot(
        summary.assign(idx=range(len(summary))), "idx", ["emc", "mmd"],
        {"emc": "EMC", "mmd": "MMD"}, ["LSBMC"], run_dir / "figures",
        "smoke_figure", "smoke")

    # tectonic compile of a skeleton report
    tectonic_ok = compile_skeleton(run_dir, log)

    log("\n================ SMOKE SUMMARY ================")
    log(f"rows: {len(df)} | targets: {list(targets)} | methods: {METHODS}")
    log(f"tectonic compile: {'OK' if tectonic_ok else 'FAILED/absent'}")
    if failures:
        log(f"FAILURES ({len(failures)}):")
        for f in failures:
            log("  - " + f)
        log("SMOKE TEST FAILED")
        print(str(run_dir))
        sys.exit(1)
    log("ALL SMOKE CHECKS PASSED")
    print(str(run_dir))


def compile_skeleton(run_dir, log) -> bool:
    if shutil.which("tectonic") is None:
        log("tectonic not found on PATH — skipping compile check.")
        return False
    tex = r"""\documentclass{article}
\usepackage{graphicx,booktabs,amsmath}
\title{ICLR Sampling Suite --- Skeleton Compile Check}
\begin{document}\maketitle
\section{Check}
The Lévy-score correction is
$S_L(x) = -\int_0^1 \sum_j \lambda_j r_j \exp(-2(V(x-\theta r_j)-V(x))/\sigma^2)\,d\theta$.
\end{document}
"""
    tex_path = run_dir / "report_artifacts" / "skeleton.tex"
    reporting.write_text(tex_path, tex)
    try:
        out = subprocess.run(
            ["tectonic", str(tex_path)], capture_output=True, text=True, timeout=180)
        ok = out.returncode == 0 and (tex_path.with_suffix(".pdf")).exists()
        if not ok:
            log("tectonic stderr:\n" + out.stderr[-1000:])
        return ok
    except Exception as e:
        log(f"tectonic compile error: {e}")
        return False


if __name__ == "__main__":
    main()
