# LSB-MC report

Theory-consistent experimental study of Lévy-Score-Based Monte Carlo for
multimodal sampling. The compiled write-up is [`main.pdf`](main.pdf).

## Build

```bash
tectonic main.tex
```

## Files

- `main.tex` — the manuscript (hand-written: introduction, theory + proof of
  stationarity, samplers, metrics, per-experiment discussion, limitations).
- `numbers.tex`, `exec_summary.tex`, `appendix_configs.tex` — **auto-generated**
  by `experiments/iclr_sampling/scripts/build_report.py` from the run summary
  CSVs, so every headline number in the prose stays in sync with the data. Do not
  edit by hand; rerun `build_report` instead.
- `recommendations.tex` — static guidance for the main paper.
- `figures/`, `tables/` — figures (PDF/PNG) and LaTeX tables copied in by
  `build_report.py`. Committed so the report builds without re-running experiments.

Regenerate the auto files and recompile in one step:

```bash
python -m experiments.iclr_sampling.scripts.build_report
```
