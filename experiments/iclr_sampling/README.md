# ICLR sampling experiment suite

A self-contained suite for stronger experiments on **Lévy-Score-Based Monte
Carlo (LSB-MC)** for multimodal Boltzmann sampling. It does **not** modify the
showcase notebooks or `archive/`.

The write-up — theory, a proof that the Lévy-score drift keeps `p` invariant, and
all results — is in [`reports/iclr_sampling_report/`](../../reports/iclr_sampling_report/)
(compiled `main.pdf`).

## Conventions
- Target density `p(x) ∝ exp(-2V/σ²)` with `σ²=2`, so `p ∝ exp(-V)`.
- Overdamped Langevin `dX = -∇V dt + σ dB` has `p` as its stationary law, hence
  `force = -∇V`, diffusion noise `σ·√dt·N(0,I)`.
- LSB-MC dynamics `dZ = (-∇V + S_L) dt + σ dB + dL`, with `L` a compound-Poisson
  process on a finite, target-informed jump bank and `S_L` the Lévy-score
  correction that preserves `p`. The correction is never removed.

## Modules
| file | contents |
|------|----------|
| `targets.py`   | `ManyWellTarget`, `MoGTarget`, `BayesGMMTarget`; generic finite-bank Lévy score |
| `samplers.py`  | `ULA`, `MALA`, `FLMC`, `LSBMC` |
| `baselines.py` | `ParallelTempering`, `HMC`, `UnderdampedLangevin` (BAOAB) |
| `metrics.py`   | MMD, exact/sliced/Sinkhorn W2, mode coverage, mode-weight KL, EMC, ManyWell count metrics |
| `experiment.py`| run loop, step-size tuning, compute accounting |
| `plotting.py` / `reporting.py` | figures (png+pdf), summary CSVs, LaTeX tables |
| `scripts/`     | smoke test + the three experiment entry points |

## GPU
One GPU only. `utils.select_gpu()` pins `CUDA_VISIBLE_DEVICES` (default 4) before
torch initialises CUDA. Override with `ICLR_GPU=<id>` or by exporting
`CUDA_VISIBLE_DEVICES`.

## Run
```bash
conda activate iclr-sampling
python -m experiments.iclr_sampling.scripts.smoke_test
python -m experiments.iclr_sampling.scripts.run_bayes_gmm_label_switching --config experiments/iclr_sampling/configs/bayes_gmm_label_switching.yaml
python -m experiments.iclr_sampling.scripts.run_mog_scaling            --config experiments/iclr_sampling/configs/mog_scaling.yaml
python -m experiments.iclr_sampling.scripts.run_manywell_scaling       --config experiments/iclr_sampling/configs/manywell_scaling.yaml

# collect the latest runs, regenerate the report's numbers/tables, compile the PDF
python -m experiments.iclr_sampling.scripts.build_report   # needs `tectonic`
```
Outputs go to timestamped `results/iclr_sampling/YYYYMMDD_HHMMSS_<tag>/`. These
are regenerable and **git-ignored**; `build_report.py` copies the figures and
LaTeX tables it needs into the (committed) report directory, so the report builds
without re-running the experiments.
