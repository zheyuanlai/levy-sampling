# JCP experiments — LSC-CP compound-Poisson sampling benchmark

This standalone experiment tree supports academic discussion of
**Lévy-score-corrected compound-Poisson (LSC-CP)** equilibrium sampling. The
current examples are the quartic double well, MoG40, the depth-balanced
Müller–Brown-type system in 10 dimensions, and the coupled two-component
\(\phi^4\) chain. The exact continuous-time generator preserves the Boltzmann
target. The random-atomic and multi-atom implementations pair each realized
atom bank with the Poisson jumps generated from that same bank.

Raw CP uses the same nonlocal geometry without the score correction. It is a
mechanism/invariant-law diagnostic, not a fair target-preserving efficiency
baseline.

## Quick start

```bash
conda activate jcp-exp
cd /home/zheyuanlai/levy-sampling/JCP_experiments

# Required preflight. The GPU tests use one selected device; CPU regression
# tests include metrics, run gates, reference handling, and stationary traces.
JCP_GPU=4 python -m pytest tests tests_cpu -q

cd notebooks
python build_notebooks.py

# Bounded launcher: one GPU per child, at most two children concurrently.
cd ..

# Safe gate first: regenerate, test, and run all selected smokes, then stop.
./run_production.sh --gpus 0,1 --max-concurrent 2 --smoke-only

# Launch the long full-notebook campaign only after reviewing the smoke
# artifacts and status. Each notebook writes resolved model/cache provenance,
# then its measured certificate JSON before asserting the certificate gate.
./run_production.sh --gpus 0,1 --max-concurrent 2
```

The generated notebooks are `01_double_well.ipynb`, `02_mog40.ipynb`,
`03_mb3well_10d.ipynb`, and `04_coupled_phi4.ipynb`. Do not run production
cells merely to regenerate or inspect notebooks.

## Immutable run layout

The launcher and notebooks refuse to overwrite an existing run. The default
layout is:

```text
results/jcp_sampling/<run-id>/
  launch_plan.json
  status.json
  smoke/
    <experiment>/
      stdout.log
      stderr.log
      smoke_plan.json
      status.json
      artifacts/
        original_config.yaml
        resolved_config.json
        smoke_metrics.csv
        smoke_manifest.json
  <experiment>/
    stdout.log
    stderr.log
    run_plan.json
    status.json
    notebook_status.json
    executed_notebook.ipynb
    artifacts/
      original_config.yaml
      resolved_preflight_config.json
      certificate_result.json
      resolved_config.json
      manifest.json
      metrics_timeseries.csv
      summary.csv
      figures/
      stationarity/
        <method>_summary.csv
        <method>_traces.npz
        all_methods_summary.csv
```

Set `JCP_RUN_ID` to deliberately share one run ID across notebooks. Set
`JCP_RESULTS_ROOT` only when using a different immutable output root. Cached
basin maps remain under `JCP_experiments/cache/`; they are reusable numerical
cache data, not run results. The source YAML is written once before model
construction and is never appended. `resolved_preflight_config.json` freezes
the actual model, jump bank, numerical box, builder parameters, cache paths and
SHA256 values before certification. `certificate_result.json` is written before
the pass assertion; `resolved_config.json` is reserved for post-refinement
choices such as timestep, quadrature, PT ladder, and trace protocol. A failed
notebook status records which of these stages was preserved.

## Coupled-phi4 numerical domain

The thermodynamic target is unbounded; the sampling box is only a numerical
overflow guard. For E4 its half-width is derived, not tuned to make a smoke
pass. A simultaneous Laplace-mixture coordinate envelope with union-bound tail
budget `1e-8` is padded by one maximum componentwise shell-jump reach; the
hottest PT envelope is also covered, and the maximum is rounded outward. The
current values are `B_beta=1.91041`, `B_beta_min=3.56877`, and
`R_infinity=2.20000`, giving `ceil(max(B_beta+R_infinity, B_beta_min)) = 5`.
This is a high-probability one-jump envelope, not compact support: Gaussian
noise, optional jitter, and multiple jumps remain unbounded.

Generic state clipping and CP-specific jump-boundary clipping are reported
separately. The latter is normalized by the exact number of applied jumps and
is gated at zero by default. E4 also reports `basin_map_outside_mass` before
its two-dimensional basin lookup clamps to the map edge; only target-preserving
methods gate on excessive outside mass, so raw-CP/FLA negative results are not
filtered out. Before using E4 for a manuscript claim, run a short box
sensitivity comparison at half-width 5 versus 6 or 7 and require phase/FES/
energy results to agree within uncertainty.

## Stationary efficiency protocol

Sparse ensemble-relaxation checkpoints are not used to estimate IAT, ESS, or
R-hat. Each notebook runs separate small-chain trajectories with uniformly
spaced post-step observations and records basin indicators, energy, collective
variables, and positions. Every observable row reports target, signed and
absolute bias, IAT, ESS, R-hat, MCSE, ESS/sec, ESS/gradient evaluation,
ESS/potential evaluation, and ESS/Lévy-score quadrature evaluation.

Only an explicitly certified exact draw for the box-restricted MALA kernel
could justify a zero-burn MALA trace. None of the current numerical-grid,
inverse-CDF, unbounded-mixture, or finite-SIR references makes that exact
claim, so every current method receives a charged settling run. ULA, BAOAB,
and the split LSC-CP integrators also have finite-step invariant laws that need
not equal the target. PT requires settling because its cold draw is replicated
over a non-equilibrium temperature ladder. The coupled \(\phi^4\) unweighted
reference cloud is finite SIR, hence approximate and not i.i.d.; this
provenance is stored. Direct weighted SNIS supplies its basin, energy, and
collective-variable targets.

By default, the settling period is one production horizon \(T\), followed by
approximately one \(T\)-long recorded trajectory. Thus approximate kernels run
about twice the stationary-trace simulation steps of exact-start MALA. The
small trace batch defaults to four seeds and eight chains per seed; adjust
`JCP_TRACE_SEEDS`, `JCP_TRACE_CHAINS`, `JCP_TRACE_DRAWS`,
`JCP_TRACE_SETTLING_BURN_FRACTION`, or `JCP_PT_TRACE_BURN_FRACTION` explicitly
when performing timing studies.

## Source layout

```text
src/
  experiments.py    experiment builders and sampler factories
  potentials.py     potentials and evaluation counters
  jumps.py           jump laws and compound-Poisson updates
  score.py           exact, random-atomic, and paired multi-atom scores
  samplers.py        ULA, MALA, FLA, BAOAB, PT, raw CP, and LSC-CP
  metrics.py         distributional, thermodynamic, and convergence metrics
  stationarity.py    uniform trace collection, IAT/ESS/R-hat, CSV/NPZ output
  certificate.py     weak generator-level stationarity certificates
  references.py      direct references, finite SIR, and weighted SNIS
  runner.py          timing, refinement gates, manifests, and passage diagnostics
notebooks/            generated experiment notebooks
standalone_experiment_note/  separate LaTeX academic experiment discussion
tests/                GPU/scientific implementation tests
tests_cpu/            CPU-only regression tests
```

## Interpretation constraints

LSC-CP is evaluated for equilibrium sampling and observables, not physical
reaction kinetics. Generator-level invariance, finite-step target bias, and
stationary sampling efficiency are separate claims and are reported
separately. Negative, unstable, censored, or failed runs must be retained in
their immutable run directories.
