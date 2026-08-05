# LSC-CP: Lévy-score-corrected compound-Poisson sampling

Code and configuration for the four benchmark examples used in the paper on
Lévy-score-corrected compound-Poisson (LSC-CP) sampling of Boltzmann
distributions.

Open a run notebook, run all cells, and the experiment executes from zero on
whatever hardware you have. There is one default configuration per experiment,
no profile to choose, no GPU to pin, and no pre-existing results required.

## Run and plot are separate

Each experiment has two notebooks.

| Experiment | Run | Plot |
|---|---|---|
| E1 double well (1D) | `notebooks/E1_double_well_run.ipynb` | `notebooks/E1_double_well_plot.ipynb` |
| E2 40-mode Gaussian mixture (2D) | `notebooks/E2_mog40_run.ipynb` | `notebooks/E2_mog40_plot.ipynb` |
| E3 extended Müller–Brown (10D) | `notebooks/E3_muller_brown_run.ipynb` | `notebooks/E3_muller_brown_plot.ipynb` |
| E4 coupled quartic chain (24D) | `notebooks/E4_coupled_quartic_chain_run.ipynb` | `notebooks/E4_coupled_quartic_chain_plot.ipynb` |

The **run** notebook builds or reuses the reference, calibrates each variant,
runs it, computes every official metric, and saves the result. It does no
typesetting.

The **plot** notebook only reads saved results. It never calls a sampler, a
tuner, a refinement, or a reference builder, and it never recomputes an
official metric. Scatter, CDF, histogram, and KDE panels are display renderings
of saved snapshots; they cannot override a number in `metrics_timeseries.csv`.

A notebook cell is thin — it calls `src/`:

```python
from src.pipeline import load_experiment, run_variants_and_save

experiment = load_experiment("E3")
experiment.ensure_reference()

run_variants_and_save(experiment=experiment, method="FLA", variants=[
    {"alpha": 1.6}, {"alpha": 1.7}, {"alpha": 1.8},
])
```

Notebooks are source files. Edit them directly; nothing regenerates them.

## Methods

`configs/registry.yaml` is the single source of truth for method identity,
display names, colours, markers, and which methods each experiment enables.
This README, the configuration, the notebooks, the plot legends, and the
validator all read it.

| Internal name | Displayed as | Notes |
|---|---|---|
| `ULA` | ULA | overdamped Langevin, Euler–Maruyama |
| `MALA` | MALA | Metropolis-adjusted, correct tamed proposal density |
| `FLA` | FLA | fractional Langevin; the *uncorrected* nonlocal comparator |
| `ULD` | ULD | underdamped Langevin. BAOAB is its integrator, never its name |
| `PT` | PT | parallel tempering, tamed MALA within each replica |
| `Raw-CP` | Raw-CP | compound Poisson with no score correction |
| `LSC-CP` | LSC-CP | full deterministic score quadrature |
| `LSC-CP-RA` | LSC-CP-RA, LSC-CP-RA (A=4), … | iid random-atomic estimator family |

`LSC-CP-RA` is **one** estimator family. Its parameter `A` is the size of a
Monte Carlo bank drawn iid from the full normalized jump law
`rho = nu / lambda`; the same bank builds the score and the compound-Poisson
increment, and it is refreshed every step. `A = 1, 4, 8` are variants of that
one method, run from a single notebook cell and saved separately.

### Canonical and tamed

Every method that supports taming runs two variants by default: canonical
(`tame: false`) and tamed (`tame: true`). MALA and PT implement the actual
tamed proposal density, with the reverse drift recomputed at the proposal
point, rather than switching taming off to sidestep the question.

The two variants are calibrated separately — separate timestep, separate
acceptance, and for PT a separately tuned ladder. They share named random
streams, which is common-random-number pairing and **not** pathwise coupling:
when they calibrate to different timesteps they are not two discretisations of
one continuous-time path.

A variant with no admissible timestep is recorded as `uncalibratable` with a
diagnosis, and the remaining variants keep running. That is a result about the
method, and it is not hidden.

## Results layout

```text
results/E3_muller_brown/
  reference/                     built once, reused by every method
  protocols/<target-hash>/...    calibration cache
  fee_calibration/               measured per-configuration oracle costs
  runs/
    FLA/
      <run-id>/
        resolved_config.yaml     the fully expanded config actually used
        calibration.json
        metrics_timeseries.csv   every official seed-level metric
        cost_timeseries.csv      raw oracle counters and derived FEE
        terminal_samples.npz
        sample_snapshots/
        diagnostics.json
        stationarity.npz         optional
        manifest.json
        COMPLETE
  catalog.csv                    derived index, rebuildable at any time
```

Each variant writes its own directory, atomically: everything goes to a
temporary directory, then the manifest carrying file hashes, then an atomic
rename, then `COMPLETE`. Workers never touch a shared index, so variants can run
concurrently without coordinating, and a reader never sees a half-written run.

`catalog.csv` is derived. Rebuild it whenever you like:

```bash
python scripts/build_catalog.py results/E3_muller_brown/
```

Only runs with a manifest, a `COMPLETE` marker, a known schema version, matching
file hashes, and no invalid marker are admitted.

## Cost accounting

Cost is measured, not assumed. Every sampler reaches the potential only through
a counted oracle (`value`, `force`, `value_and_force`), and each call updates a
counter as it happens. Nothing reconstructs a cost from `steps × particles`, so
a caching change moves the recorded numbers by itself.

The reported cost is force-equivalent evaluations:

```text
N_FEE = N_F + rho * N_V_eq^extra,        rho = C_V / C_F
```

`C_V` and `C_F` are both *amortized wall time per configuration*, measured by a
microbenchmark frozen per device, dtype, software version, and target
implementation. Runs may share a FEE axis only when their
`fee_calibration_hash` matches; the plotter refuses to merge mismatched
calibrations without an explicit compatibility record.

FEE is an oracle-cost proxy, deliberately not a full cost model: it excludes
communication, host–device transfers, framework overhead, random number
generation, accept/reject logic, and parallel efficiency. Whole-algorithm
wall-clock is not a reported scientific metric.

Every official curve is produced against both simulation time and FEE. The
extra-potential axis is a *diagnostic restricted to the LSC estimators* — full
LSC-CP against LSC-CP-RA(A) — titled "LSC score potential-evaluation cost". It
is never a claim about total computational cost, and methods with no extra
potential evaluations are not admitted to it.

Distribution snapshots (scatter, CDF, histogram) are compared at matched
simulation time only, always from an actually saved checkpoint. Sample positions
are never interpolated in time or in budget.

## Reproducibility

Each `(experiment, method family, pairing group, seed, stream)` owns its own
generator, seeded by a stable keyed hash. Draws are produced one seed block at a
time with a per-seed shape that does not depend on the batch, so running seed 3
alone is bitwise identical to seed 3 inside an eight-seed campaign, and adding
or removing seeds or whole methods leaves existing streams untouched.

## Running without Jupyter

```bash
python scripts/run_experiment.py E3
python scripts/run_experiment.py E3 --methods FLA,LSC-CP-RA --device cpu
python scripts/build_reference.py E4          # exits nonzero if gates fail
python scripts/build_catalog.py --all results/
python scripts/validate_release.py            # source-package checks
```

CPU and CUDA are both supported execution paths. `--device auto` (the default)
picks CUDA when it is available and CPU otherwise. There is no GPU allow-list,
no pinned index, and no environment variable that can forbid a run; the device
is recorded as provenance only.

## Source package versus frozen release

The **source** package is what you need to run everything from zero: `src/`,
`configs/`, `notebooks/`, `scripts/`, `tests/`, this README, and the environment
lock. It does not require `results/`, `figures/`, or `cache/` to exist, and
nothing checks for them before running.

The **frozen release** additionally carries `results/`, `figures/`,
`resolved_configs/`, `manifests/`, and `executed_notebooks/`. Only release
validation requires those:

```bash
python scripts/build_release.py --source dist/source.zip
python scripts/build_release.py --frozen dist/release.zip
python scripts/validate_release.py --release
```

## References

Each experiment builds its reference once and reuses it for every method and
every parameter value.

- **E1** — high-precision one-dimensional inverse-CDF reference, with a one-off
  validation over grid refinement, a widened box, moments, partition mass, and
  a reference-versus-reference sampling floor.
- **E2** — exact mixture sampling. The mode descriptor is a hard assignment by
  component log-density, `a(x) = argmax_k log N(x; mu_k, I)`. Under that
  descriptor the true masses `p*_k` need not equal `1/40`, so they are estimated
  from a large frozen bank and frozen, together with the reference coverage line
  `EMC*`. Entropic mode coverage is the normalized entropy of the occupancy
  vector, `EMC = -sum_k p_k log p_k / log K`; `exp(H)/K` is a different quantity
  and is reported separately as the effective mode fraction.
- **E3** — the collective variable is the *latent* pair
  `z_{1:2} = (x B^{-T})_{1:2}`, never the first two sampling coordinates. The
  primary reference is the 2D CV grid density; reference CV samples drawn from
  it are used for two-sample metrics and scatter panels.
- **E4** — multi-start long-run PT-MALA is the primary reference, cross-checked
  against a Laplace-mixture self-normalized importance sampler. Acceptance gates
  are frozen in `configs/experiments/E4_reference_acceptance.yaml` and produce a
  per-gate `reference_validation.json`. A failing gate exits nonzero and the
  result is not promoted; the escalation order is to extend the PT run, then
  improve the proposal, and only then consider the optional annealed SMC
  fallback. Averaging the two references, or picking whichever looks better, is
  not an option.

## Environment

```bash
conda env create -f environment.yml
conda activate jcp-levy-release
python -m pytest tests/ -q
```

## Scope

This is equilibrium sampling. The jumps provide nonlocal inter-basin transport
and the Lévy score preserves the target; the resulting trajectories are not
physical reaction kinetics. E4 reports static equilibrium observables only — no
first-passage times, transition counts, round trips, or kinetic transition
matrices.
