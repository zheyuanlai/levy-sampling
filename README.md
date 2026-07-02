# Lévy-score-corrected nonlocal Boltzmann sampling

A reproducible pipeline for **target-preserving nonlocal jump diffusions**: finite-activity
compound-Poisson (CP) jumps added to overdamped Langevin dynamics, made Boltzmann-preserving by a
**stationary Lévy-score drift correction** determined by the target density and a fixed jump law.

For a Gibbs target `π(x) ∝ exp(-V(x)/ε)` and a finite jump measure `ν`, the raw CP jump operator
generally biases the invariant law. The stationary Lévy score

```
S_{ν,ε}(x) = -∫₀¹ ∫ r · [ p_ε(x - θr) / p_ε(x) ] ν(dr) dθ
```

is the drift that cancels this weak imbalance, giving the **LSC-CP** generator that preserves the
Boltzmann distribution while keeping the same nonlocal jumps. The comparison spine is
**local Langevin / raw CP / LSC-CP**: raw CP uses the identical jump law as LSC-CP (sharing one
random-number stream) and differs from it *only* by the stationary drift, so it isolates the
correction.

## Repository layout

```
experiments/jcp_sampling/      # the sampling pipeline
  core/                        #   potentials, jump banks, Lévy score, samplers, metrics, plotting, I/O
  configs/                     #   experiment YAMLs (smoke + core suites)
  scripts/                     #   run_experiment, launch_sweep, aggregate_results, make_figures, make_report_assets
  tests/                       #   unit tests (pytest)
reports/jcp_sampling_report/   # comprehensive LaTeX report + generated figures/tables (main.pdf)
JCP_EXPERIMENT_PLAN.md         # experiment roster, design, and results summary
results/                       # timestamped run outputs (git-ignored; regenerable)
paper/                         # manuscript sources (git-ignored)
```

## Examples

The three main examples (following the manuscript) plus additional landscapes:

| Target | Dim | What it probes |
|---|---|---|
| Double well `V=x⁴/4−x²/2` | 1 | target fidelity vs. relaxation; timestep bias |
| Triple well (Gaussian mixture) | 1 | jump-support design (adjacent vs. overlong) |
| Transformed Müller–Brown | 10 | basin communication in mixed coordinates |
| Four-well / ManyWell | 2 / 8–64 | basin-graph geometry; high-dimensional scaling |
| Lennard-Jones (LJ7, LJ38) | 14 / 114 | scope boundary for rigid clusters |
| Alanine dipeptide (Ramachandran) | 2 (torus) | conformational sampling with wrapped jumps |

## Environment

Conda env `ddlpm` (PyTorch 2.12+cu130). Runs need the repo root on `PYTHONPATH`:

```bash
export PYTHONPATH=$(pwd)
```

## Reproduce

```bash
# unit tests
conda run -n ddlpm python -m pytest experiments/jcp_sampling/tests -q

# smoke suite (tiny, exercises every code path), then core suites, with bounded 2-GPU concurrency
conda run -n ddlpm python -m experiments.jcp_sampling.scripts.launch_sweep \
  --suite experiments/jcp_sampling/configs/suites/smoke.yaml --gpus 0,1 --max-concurrent 2

# aggregate a run, build the report
conda run -n ddlpm python -m experiments.jcp_sampling.scripts.aggregate_results \
  --results-root results/jcp_sampling --manifest <launcher-manifest.jsonl>
conda run -n ddlpm python -m experiments.jcp_sampling.scripts.make_figures      --results-root results/jcp_sampling
conda run -n ddlpm python -m experiments.jcp_sampling.scripts.make_report_assets --results-root results/jcp_sampling
cd reports/jcp_sampling_report && tectonic main.tex
```

All headline numbers, tables, and figures in the report are generated from the CSV/JSON artifacts of
the runs, not typed by hand.

## Key findings

- **LSC-CP preserves the target where raw CP does not.** On the double well the CDF-sup error drops
  from `0.109` (raw CP) to `0.020` (LSC-CP); the raw-CP defect is **timestep-independent** (a wrong
  invariant law, not a discretization artifact), and starting at equilibrium LSC-CP stays put while
  raw CP drifts off.
- **Geometry matters.** Jump support/direction must match the metastable geometry: geometry-matched
  jumps beat random-direction controls of the same length, and jumps that are too short behave like
  local noise.
- **Efficiency regime.** On smooth low-dimensional targets LSC-CP is the most accurate but not the
  fastest per wall-clock (efficiency metrics are always reported alongside a bias column); its wins
  are the trapped and high-dimensional regimes (four-well, ManyWell 40–600× lower block-KL).
- **Scope boundary.** The additive correction works when basins connect through low-energy corridors
  and provably fails when the direct displacement crosses a high-energy barrier (rigid clusters); for
  LJ38 a Metropolis-corrected geometry-matched darting hybrid still solves the global-funnel-discovery
  problem. See the report Discussion.

See `JCP_EXPERIMENT_PLAN.md` and `reports/jcp_sampling_report/main.pdf` for the full account.
