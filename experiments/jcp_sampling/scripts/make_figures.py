from __future__ import annotations

import argparse
from pathlib import Path

from experiments.jcp_sampling.core.plotting import generate_manuscript_figures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--report-fig-dir", default="reports/jcp_sampling_report/figures")
    ap.add_argument("--summary-csv", default=None)
    args = ap.parse_args()
    summary = args.summary_csv or str(Path(args.results_root) / "aggregate" / "all_summary.csv")
    figs = generate_manuscript_figures(summary, args.report_fig_dir, results_root=args.results_root)
    print(f"generated {len(figs)} figure files")


if __name__ == "__main__":
    main()
