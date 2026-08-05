# AGENTS.md — levy-sampling

## Project goal

This repository studies Lévy-score-corrected jump diffusions for accelerated equilibrium sampling of Boltzmann distributions. The current goal is to reframe the project as a Journal of Chemical Physics enhanced-sampling paper, with reproducible experiments and a manuscript-ready LaTeX report.

The repository-root pipeline (`src/`, `configs/`, `notebooks/`, `scripts/`,
`tests/`) was restructured around a run/plot split. `README.md` describes the
current architecture and supersedes any older plan document on these points:

- eight notebooks, one run and one plot per experiment; notebooks are source
  files and nothing regenerates them;
- configuration in `configs/` is the actual run input, with exactly one default
  per experiment and no smoke/dev/production profiles;
- `configs/registry.yaml` is the single source of truth for method identity,
  display names, styling, and per-experiment enablement;
- results are per-variant directories written atomically, with `catalog.csv` a
  derived index rebuilt by scanning manifests;
- cost is force-equivalent evaluations from measured oracle counters;
  whole-algorithm wall-clock is not a reported scientific metric.

## Non-negotiable directory constraints

Never edit or overwrite the old ICLR experiment outputs:

- `doublewell_output/`
- `manywell_output/`
- `mog40_output/`
- `reports/iclr_sampling_report/`

Old notebooks and old experiment code may be read for formulas, validation, and
design context, but they must not be reintroduced into the execution path. There
is no dual "new format first, fall back to old format" path anywhere: a one-off
migration tool may read old files, but it must not sit in the daily run, plot,
or release flow.

Do not overwrite an existing run directory. Every variant writes its own
directory and retires rather than deletes: mark a superseded run invalid, keep
its evidence.

## Scientific framing

Frame the paper as a JCP enhanced-sampling paper, not an ICLR benchmark paper.

Core story:

- Target: Boltzmann distribution, typically `pi(x) ∝ exp(-beta V(x))`.
- Problem: metastability and exponentially slow inter-basin mixing for local dynamics.
- Method: add finite-activity jumps plus a stationary Lévy-score drift correction.
- Mechanism: jumps provide nonlocal inter-basin equilibrium transport while the Lévy score preserves the target invariant distribution.
- Caveat: the method is for equilibrium sampling and observables, not physical reaction kinetics.

Do not claim success based only on plots. Compute quantitative metrics. Do not hide negative results; if a method fails or is unstable, report it.

## Jump design principles

Jump laws are first-class objects in `src/jumps.py`, each exposing one sampling
primitive that draws **iid from the full law**: for a finite mixture
`rho = sum_k w_k q_k`, draw a component index from `w` and then a displacement
from that component, independently for every bank slot. One fixed draw per
component is component stratification, a different estimator, and is not part of
the default method set.

For a finite jump bank,

```text
nu(dr) = Lambda sum_e w_e delta_{r_e}(dr).
```

The stationary Lévy-score correction is

```text
S_nu(x)
= - Lambda sum_e w_e r_e int_0^1 exp[-beta (V(x - theta r_e) - V(x))] d theta.
```

Use vectorized quadrature over particles, jump vectors, and theta nodes. Use chunking to avoid GPU OOM. Stabilize exponentials and record clipping/nonfinite diagnostics.

The no-score jump process changes the invariant distribution, so Raw-CP is the
uncorrected-transport diagnostic rather than a fair baseline. FLA plays the same
role for continuous nonlocal dynamics.

`LSC-CP-RA` is a single iid estimator family whose parameter `A` is a Monte
Carlo bank size. The same bank drives the score and the compound-Poisson
increment and is refreshed every step, so score and noise always see the
identical random empirical Lévy measure.

## Device policy

CPU and CUDA are both fully supported execution paths. Device selection is
`auto` by default: CUDA when available, CPU otherwise, and an explicit
`cpu`/`cuda`/`cuda:N` is honoured. Do not reintroduce a GPU allow-list, a pinned
index, an environment variable that can forbid a run, or an assertion that some
particular device must be used. The device is recorded as provenance and enters
the FEE cost calibration; it is never a precondition for running.

This is a shared node, so still do not launch uncontrolled sweeps.

## The four experiments

E1 double well (1D), E2 40-mode Gaussian mixture (2D), E3 extended
Müller–Brown (10D), E4 one-dimensional two-component coupled quartic chain
(24D). Each has one default full configuration in `configs/experiments/`. There
is no separate smoke profile: to debug locally, temporarily shrink the particle
count or the horizon as an explicit edit, and do not commit a second profile.

E4 is a two-component coupled quartic chain, not a scalar phi^4 model, and its
order parameter is a two-component magnetization.

## Required saved artifacts

Every run directory carries `resolved_config.yaml` (fully expanded, including
values derived at build time), `calibration.json`, `metrics_timeseries.csv`,
`cost_timeseries.csv`, `terminal_samples.npz`, `sample_snapshots/`,
`diagnostics.json`, `manifest.json`, and `COMPLETE`, plus an optional
`stationarity.npz`. The manifest records git provenance, device and dtype
provenance, every hash (target, reference, calibration, variant, RNG pairing
group, FEE calibration), the seeds, and the file hashes.

## Required metrics

Every official metric is computed at run time and written to
`metrics_timeseries.csv`. Plot notebooks read those numbers and never recompute
them.

Cost is force-equivalent evaluations, `N_FEE = N_F + rho * N_V_eq^extra`, from
counters that record what the sampler actually asked the target for. Never infer
a cost from `steps x particles`. Whole-algorithm wall-clock is not a scientific
metric. The extra-potential axis is an LSC-only score-cost diagnostic and never
a claim about total computational cost.

Main-text metric sets are fixed per experiment: E1 exact `W_2`, `MMD^2`, and KS
distance; E2 EMC, mode-weight Jensen-Shannon, and the per-mode occupancy
profile; E3 CV-SW2 and CV-MMD^2 on the latent CV; E4 phase-weight
Jensen-Shannon, order-parameter SW2 and MMD^2, energy/site absolute error, and
susceptibility relative Frobenius error. E4 reports static equilibrium
observables only: no first-passage, transition count, round trip, relay path, or
kinetic transition matrix.

EMC is the normalized entropy of the hard-assignment occupancy vector. The
quantity `exp(H)/K` is different and must be named the effective mode fraction.

## Workflow and stop conditions

Before a full run: run `python -m pytest tests/ -q` and
`python scripts/validate_release.py`. Both must pass.

A variant that has no admissible timestep is recorded as `uncalibratable` with a
diagnosis, and the remaining variants keep running. A reference that fails its
frozen acceptance gates exits nonzero and is not promoted. Do not delete failed
or negative results, and do not soften a gate to get past it: extend the run
instead. Do not claim success from a plot; quote the computed metric.

## Reporting

The final report should be LaTeX and manuscript-ready under `reports/jcp_sampling_report/`.

It must include:

- method summary
- jump design section
- experiment setup
- baselines
- metrics and compute accounting
- results
- ablations
- limitations
- reproducibility appendix
- tables generated from CSV/JSON, not manually typed

Compile the report to PDF before stopping; `tectonic` is acceptable.
