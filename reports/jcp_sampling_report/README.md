# JCP sampling report

This report is generated from `results/jcp_sampling/<timestamp>` by:

```bash
python -m experiments.jcp_sampling.scripts.aggregate_results --results-root results/jcp_sampling/<timestamp>
python -m experiments.jcp_sampling.scripts.make_figures --results-root results/jcp_sampling/<timestamp>
python -m experiments.jcp_sampling.scripts.make_report_assets --results-root results/jcp_sampling/<timestamp>
cd reports/jcp_sampling_report && tectonic main.tex
```

Do not manually type numerical results into the report; generate tables and summaries from CSV/JSON result files.
