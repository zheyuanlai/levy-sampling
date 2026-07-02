# JCP Experiment Plan for Lévy-Score-Corrected Jump Diffusions

## Objective

Build a reproducible Journal of Chemical Physics enhanced-sampling experiment pipeline for Lévy-score-corrected jump diffusions. The pipeline will run smoke tests and core experiments, generate quantitative CSV/JSON summaries, produce manuscript-ready figures and tables, and compile a LaTeX report to PDF.

This is a conversion from an ICLR-style multimodal sampling benchmark into a JCP-style equilibrium Boltzmann-sampling study. The emphasis is physical/statistical-mechanical: metastability, equilibrium observables, basin populations, free-energy recovery, jump-measure design, and compute-normalized efficiency.

## Manuscript alignment (2026-07 revision)

The senior's rewritten manuscript ("A stationary Lévy-score correction for target-preserving nonlocal Boltzmann sampling") is the authoritative scope. It renames the method **LSC-CP** (Lévy-score-corrected compound-Poisson) and pins the paper on **three main examples** compared with a clean three-method spine — **local Langevin (ULA) / raw CP / LSC-CP**:

1. **Double well** `V(x)=x⁴/4−x²/2` (wells at ±1, ΔV=1/4), ε=0.125, Δt=0.003, T=4.8, plus a temperature sweep for the well-TV threshold-relaxation time. Diagnostics: CDF-sup error, density-L1 error, threshold time.
2. **Triple well**: normalized Gaussian mixture, modes (−3,0,3), scales (0.50,0.75,0.50), weights (5/21,3/7,1/3), ε=0.08. Adjacent vs overlong shell support. Diagnostics: mode-TV, middle-mass error.
3. **Transformed Müller–Brown (10D)**: `U₁₀(z)=U_MB(z₁,z₂)+½σ⁻²Σ_{ℓ≥3}z_ℓ²`, σ=0.75, ε=0.5, evolved in mixed coordinates `x=zBᵀ`. Diagnostics: basin-TV, first all-basin coverage time.

The pipeline was extended to serve this: `TripleWell1D` and `TransformedMuellerBrown10D` potentials (`core/potentials.py`), a **raw CP** first-class method (`use_score=False` flag on `LevyScoreJumpDiffusion`, sharing the jump RNG stream with LSC-CP via common random numbers so the two differ only by the stationary drift), and the manuscript diagnostics in `core/metrics.py` (`cdf_sup_error`, `density_l1_error`, `basin_tv_series`, `threshold_time`, `first_all_basin_coverage_time`). Raw CP is the manuscript's central diagnostic (it exposes the invariant-measure defect of uncorrected jumps) and is *not* merely a forbidden baseline.

Reproduction status: the double-well example reproduces the manuscript to within seed noise (CDF-sup: raw 0.109 vs 0.108, LSC-CP 0.0199 vs 0.019). Triple-well and 10D-MB reproduction in progress.

Per the approved plan, the paper keeps the local/raw/LSC-CP spine but **also** adds a full efficiency benchmark (ULA/MALA/BAOAB/HMC/PT + ESS/compute, ESS always paired with a bias column) and four additional landscapes (ManyWell high-D, four-well graph ablation, Lennard-Jones — gated by a known rotational-symmetry obstruction, alanine dipeptide — gated). The additional Tier-1 ablations that defend the three main examples are: **timestep-bias** (E1), **jump scale×intensity sweep** (E2), **triple-well support ablation** (E3), **random-direction matched-length control** (E4), and **start-at-equilibrium invariance** (E5). See `/home/zheyuanlai/.claude/plans/jcp-experiment-plan-md-now-the-problem-floating-petal.md` for the full experiment roster and prioritization.

## Results achieved (2026-07-02)

All experiments in the approved plan have been run; report at `reports/jcp_sampling_report/main.pdf`.

- **Reproduction gate (passed).** Double well CDF-sup raw 0.109 -> LSC-CP 0.020 (manuscript 0.108/0.019); triple-well mode-TV overlong 0.015 / adjacent 0.047 / raw ~0.17-0.28; 10D MB basin-TV LSC-CP 0.012 / raw 0.181 / local 0.110 (manuscript 0.017/0.190/0.075).
- **E1 timestep bias (decisive).** Raw CP residual error is ~0.10 and *independent of the timestep* (0.101->0.109 across h=0.006..0.00075); LSC-CP sits at the ~0.015 finite-sample floor with jump-free ULA. The raw-CP defect is a wrong invariant law, not a discretization artifact.
- **E2 scale x intensity.** Started at equilibrium, target fidelity is flat across all (c, lambda) (the correction preserves the target for any bank); off-equilibrium, too-short jumps (c=0.5) relax slowest and matched jumps (c~1) relax fastest.
- **E3/E4 support and direction.** Triple-well overlong < adjacent < too-short; 10D MB lifted jumps (0.012) beat a random-direction matched-length control (0.124).
- **E5 invariance.** From equilibrium, LSC-CP stays put (triple basin-TV 0.011, MB 0.012) while raw CP drifts off (0.165, 0.191).
- **E6 benchmark (honest positioning).** On the smooth double well, LSC-CP is the most accurate (CDF-sup 0.020) but slowest per wall-clock (ESS/sec 3.3k); local Langevin posts 17k ESS/sec at 0.40 bias -- ESS must be read with the bias column.
- **E7 ManyWell (high-D win).** LSC-CP block-marginal KL ~1e-3 across d=8..64 vs baseline 0.3-0.6.
- **E8 four-well.** Geometry-matched edge graph gives the best basin KL; structured beats the random matched-length control.
- **E9 Lennard-Jones (documented limitation).** Fixed-bank jump productivity collapses to 2.7% under cluster rotation; a rotationally-augmented gated bank restores it to 100%. LJ7-2D has no trapped-multi-isomer window (MALA interconverts ~30x/trajectory even at beta=8), so no method can win there.
- **E9b LJ38-3D double funnel (barrier theorem + Metropolis-darting hybrid win).** Recovered the FCC global minimum (-173.928, exact literature value) and an icosahedral minimum (-170.99). The straight-line inter-funnel displacement crosses a +87.7 eps barrier of atomic overlaps, giving beta*barrier ~700 at the solid-solid transition, so the rejection-free additive score integrand vanishes -- the additive correction *provably cannot cross the barrier*. The same geometry-matched displacement, applied as a Metropolis-corrected endpoint dart, escapes the icosahedral funnel to find the FCC global (FCC-funnel fraction from the icosahedral start: local MC 0.00 vs darting 1.00) that local Monte Carlo never reaches. Geometry-matched nonlocal jumps therefore solve LJ38, but via the MH-corrected hybrid, not the pure additive score; entropy-balanced population recovery needs full smart-darting (future work).
- **E10 alanine dipeptide (positive torus case).** Wrapped basin-to-basin jumps give LSC-CP the best basin-population fidelity (basin-TV 0.013, ~0.98 all-basin coverage), beating a random-direction control (0.027), raw CP (0.174), and trapped local baselines; PT is the only competitive baseline.

## Repository boundaries

Do not modify or overwrite old ICLR artifacts:

```text
doublewell_output/
manywell_output/
mog40_output/
reports/iclr_sampling_report/
```

The notebooks and old code may be read for formulas and validation, but the new implementation should live in the JCP namespace:

```text
experiments/jcp_sampling/
results/jcp_sampling/<timestamp>/
reports/jcp_sampling_report/
```

The new JCP pipeline should not write into `doublewell_output/`, `manywell_output/`, `mog40_output/`, or `reports/iclr_sampling_report/`.

## Scientific framing

The report should be framed as a JCP enhanced-sampling paper, not as an ICLR benchmark paper.

Core message:

1. The target is a Boltzmann distribution, `pi(x) ∝ exp(-beta V(x))`.
2. The sampling problem is metastability: local diffusions mix within basins but exchange mass between basins on exponentially slow time scales.
3. The method adds finite-activity nonlocal jumps and a stationary Lévy-score drift correction.
4. Geometry-matched jumps provide a nonlocal equilibrium transport channel between basins.
5. The Lévy score preserves the target invariant distribution in continuous time.
6. The method is for equilibrium sampling and free-energy/observable estimation, not for reproducing physical reaction kinetics.
7. Negative or unstable results must be reported rather than hidden.

## Directory and file plan

Create the following package and support files.

```text
experiments/jcp_sampling/
  __init__.py
  configs/
    suites/
      smoke.yaml
      core.yaml
    smoke/
      double_well_smoke.yaml
      four_well_smoke.yaml
      muller_brown_smoke.yaml
      manywell_smoke.yaml
      lj_smoke.yaml
    double_well/
      scale_intensity_timestep.yaml
    four_well/
      graph_ablation.yaml
    muller_brown/
      free_energy_recovery.yaml
    lj/
      lj7_isomer_observables.yaml
    manywell/
      dimension_scaling.yaml
  core/
    __init__.py
    potentials.py
    jump_banks.py
    levy_score.py
    samplers.py
    baselines.py
    metrics.py
    observables.py
    references.py
    experiment.py
    io_utils.py
    plotting.py
  scripts/
    __init__.py
    run_experiment.py
    launch_sweep.py
    aggregate_results.py
    make_figures.py
    make_report_assets.py
    smoke_test.py
  tests/
    test_jump_banks.py
    test_levy_score.py
    test_metrics.py
    test_smoke_configs.py
```

Create or maintain:

```text
results/jcp_sampling/.gitkeep
```

Create the report tree:

```text
reports/jcp_sampling_report/
  README.md
  main.tex
  sections/
    01_introduction.tex
    02_method.tex
    03_jump_design.tex
    04_experimental_protocol.tex
    05_double_well.tex
    06_four_well.tex
    07_muller_brown.tex
    08_lennard_jones.tex
    09_manywell.tex
    10_discussion_limitations.tex
    A_reproducibility.tex
  figures/
  tables/
  numbers.tex
  exec_summary.tex
  appendix_configs.tex
```

## Core module design

### `core/potentials.py`

Define a common potential interface:

```python
class BasePotential:
    name: str
    dim: int
    beta: float
    state_clip: float

    def potential(self, x): ...
    def gradient(self, x): ...
    def force(self, x): return -self.gradient(x)
    def minima(self): ...
    def basin_labels(self, x): ...
    def observables(self, x): ...
    def reference(self, n, seed, device): ...
    def metadata(self) -> dict: ...
```

Concrete potentials:

- `DoubleWell1D`: `V(x)=1/4 (x^2-1)^2`, minima at `-1,+1`, basin split at `0`.
- `FourWell2D`: `V(x,y)=(x^2-1)^2+(y^2-1)^2`, minima `(±1,±1)`, quadrant basins.
- `MuellerBrown2D`: standard Müller–Brown potential with documented scaling and grid domain; grid reference density.
- `ManyWell`: independent double-well blocks following the existing notebook/ICLR design, with blockwise basin labels and exact/inverse-CDF block reference.
- `LennardJonesCluster2D`: start with LJ7 in 2D if feasible; remove center of mass, use pair-distance descriptors, quench for inherent-structure assignment. If LJ7 is too slow in smoke/full runs, fall back to LJ5 with the failure/reason documented.

### `core/jump_banks.py`

Implement jump banks as first-class objects.

```python
@dataclass
class FiniteJumpBank:
    name: str
    vectors: torch.Tensor        # (n_edges, dim)
    weights: torch.Tensor        # nonnegative, normalized
    intensity: float             # Lambda
    metadata: dict

    def validate(self): ...
    def sample(self, shape, generator, dt): ...
    def to(self, device=None, dtype=None): ...
```

Required factories:

1. `double_well_shell(minima, scale, intensity)`
2. `minima_complete_graph(minima, intensity, symmetric=True)`
3. `minima_edge_graph(minima, edges, intensity, symmetric=True)`
4. `random_matched_length_control(reference_bank, seed, preserve_lengths=True)`
5. `manywell_block_flip(n_blocks, displacement, intensity_per_block)`
6. optional `lj_isomer_aligned(isomer_catalogue, intensity)`
7. optional `torsion_wrapped_graph(...)`

For a finite bank,

```text
nu(dr) = Lambda sum_e w_e delta_{r_e}(dr).
```

The random matched-length control is the key fair ablation for jump geometry. It should preserve the empirical jump-length distribution of the matched graph while randomizing directions/orientations in the same dimension. Do not use “no-score jumps” as a main baseline because it changes the invariant distribution.

### `core/levy_score.py`

Implement the stationary Lévy score:

```text
S_nu(x) = - Lambda sum_e w_e r_e int_0^1 exp[-beta (V(x - theta r_e) - V(x))] d theta.
```

Functions:

```python
def gauss_legendre_01(n, device, dtype): ...

def stationary_levy_score(
    potential_fn,
    x,
    bank: FiniteJumpBank,
    beta: float,
    n_theta: int,
    theta_nodes=None,
    theta_weights=None,
    particle_chunk: int | None = None,
    jump_chunk: int = 64,
    exponent_clip: float = 60.0,
    score_clip: float | None = None,
    return_diagnostics: bool = False,
): ...

def count_levy_quadrature_evals(n_particles, n_theta, n_jumps): ...
```

Implementation requirements:

- Vectorize over particles, jump vectors, and theta nodes when memory allows.
- Support chunking over particles and jump vectors to avoid GPU OOM.
- Clamp exponent differences and record clipping diagnostics.
- Record mean/max score norm, exponent min/max, nonfinite counts, and quadrature evaluation counts.
- Tests must verify chunked and unchunked results agree within tolerance.
- Tests must check the 1D stationarity identity numerically on a grid for a finite bank.

### `core/samplers.py` and `core/baselines.py`

Samplers should share a common interface:

```python
class Sampler:
    name: str
    def init_state(self, n_chains, seed, device): ...
    def step(self, state, generator): ...
    def final_samples(self, state): ...
    def diagnostics(self) -> dict: ...
```

Core method:

- `LevyScoreJumpDiffusion`: explicit/tamed Euler update with drift `-grad V + S_nu`, Brownian noise `sqrt(2/beta) sqrt(dt) xi`, and compound-Poisson finite-bank jumps.

Baselines:

- `OverdampedLangevin` / ULA
- `MALA`
- `BAOAB` underdamped Langevin
- `HMC`
- `ParallelTempering`

All methods must count wall-clock time, potential evaluations, gradient evaluations, and, for LSB, Lévy-score quadrature evaluations. PT must be charged for every replica. HMC must be charged for every leapfrog gradient.

### `core/metrics.py`

Required metrics where applicable:

- basin population error
- basin KL
- free-energy RMSE
- observable bias
- integrated autocorrelation time (IAT)
- ESS
- ESS/sec
- ESS/gradient evaluation
- ESS/potential evaluation
- ESS/Lévy-score quadrature evaluation
- wall-clock time
- potential evaluations
- gradient evaluations
- Lévy-score quadrature evaluations
- jump occurrence counts/rates
- transition matrix between basins
- nonfinite count and clipped-state fraction

Functions:

```python
def basin_population_metrics(labels, target_probs): ...
def transition_matrix(labels, n_basins): ...
def autocorr_fft(x): ...
def integrated_autocorrelation_time(x, method="initial_positive"): ...
def ess_from_iat(n_samples, iat): ...
def observable_bias(samples, reference_values, observables): ...
def free_energy_rmse(sample_hist, ref_density, beta, floor=...): ...
def compute_metric_bundle(...): ...
```

### `core/observables.py`

Define target-specific observables:

- Double well: `x`, `x^2`, right-well indicator, energy.
- Four well: `x`, `y`, `x^2+y^2`, basin label, energy.
- Müller–Brown: `V`, `x`, `y`, `|x|^2`, basin label, free-energy surface on grid.
- LJ: energy, energy variance, sorted pair-distance descriptor, inherent-structure/isomer label, pair-distance histogram.
- ManyWell: deep-well indicators, deep-count distribution, per-block marginals.

### `core/references.py`

Reference generation:

- Double well: high-resolution grid quadrature/inverse CDF.
- Four well: analytic/product grid reference or separable reference when using the separable four-well potential.
- Müller–Brown: normalized grid density on documented domain.
- ManyWell: exact product reference via inverse CDF per block.
- LJ: PT reference and/or archived/discovered isomer catalogue; save the reference source and commands.

### `core/experiment.py`

Define:

```python
@dataclass
class RunSpec: ...
@dataclass
class RunResult: ...

def build_potential(cfg): ...
def build_jump_bank(cfg, potential): ...
def build_sampler(cfg, potential, bank): ...
def run_method(spec, method_cfg): ...
def run_single_config(config_path, output_dir, device): ...
def validate_run_artifacts(run_dir): ...
```

`run_single_config` must save raw per-seed metrics, time-series metrics, final samples or compressed summaries, method diagnostics, and failure status.

### `core/io_utils.py`

Responsibilities:

- Create `results/jcp_sampling/<timestamp>_<tag>/`.
- Save the original YAML config and resolved config JSON.
- Save git commit hash, branch, and dirty status.
- Save environment info, Python/package versions, GPU info, and `CUDA_VISIBLE_DEVICES`.
- Tee stdout/stderr into logs.
- Write `raw_metrics.csv`, `summary_by_method.csv`, `timeseries.csv`, `run_manifest.json`, `run_status.json`, and `README.md`.
- Never overwrite an existing run directory.

### `core/plotting.py`

Figures:

- Potential contours/densities with jump arrows.
- Sample density overlays.
- Basin population bars.
- Transition matrices.
- Metric vs wall-clock / gradient eval / quadrature eval curves.
- Double-well heatmap over jump scale and intensity.
- Timestep-bias plot.
- Müller–Brown free-energy surface and RMSE curves.
- LJ energy and pair-distance histograms.
- ManyWell scaling curves.

Save every figure as both `.pdf` and `.png`.

## Script design

### `scripts/run_experiment.py`

Run one YAML config on one device:

```bash
python -m experiments.jcp_sampling.scripts.run_experiment \
  --config experiments/jcp_sampling/configs/double_well/scale_intensity_timestep.yaml \
  --output-root results/jcp_sampling \
  --tag double_well_scale
```

### `scripts/launch_sweep.py`

Launch a suite of configs with controlled GPU use:

```bash
python -m experiments.jcp_sampling.scripts.launch_sweep \
  --suite experiments/jcp_sampling/configs/suites/core.yaml \
  --gpus 0,1 \
  --max-concurrent 2
```

Requirements:

- Accept `--gpus 0,1` and `--max-concurrent 2`.
- Assign one GPU per process via `CUDA_VISIBLE_DEVICES`.
- Maintain a queue and never exceed two concurrent processes.
- Write a launcher manifest with command, config, GPU, PID, start/end time, return code, and output directory.
- On failure, preserve logs and continue independent jobs unless the failure is a global validation failure.

### `scripts/aggregate_results.py`

Collect completed run directories into cross-experiment summaries. It should never ignore failed runs; failures should appear in a status table.

### `scripts/make_figures.py`

Read figure-ready CSV/JSON summaries and create figures under both the timestamped result directory and `reports/jcp_sampling_report/figures/`.

### `scripts/make_report_assets.py`

Generate:

```text
reports/jcp_sampling_report/numbers.tex
reports/jcp_sampling_report/exec_summary.tex
reports/jcp_sampling_report/appendix_configs.tex
reports/jcp_sampling_report/tables/*.tex
```

Tables must be generated from CSV/JSON, not manually typed.

### `scripts/smoke_test.py`

Run all smoke configs, verify finite outputs and required artifacts, and optionally compile a minimal report skeleton.

## Experiment suites

### Experiment 1: double well

Potential:

```text
V(x)=1/4 (x^2-1)^2.
```

Purpose: establish the method in the simplest metastable Boltzmann system, quantify well-population recovery, IAT, ESS/sec, and finite-step bias.

Jump banks:

- `double_well_shell` with minima at `-1,+1` and natural displacement `D=2`.
- Scale sweep: `cD`, with `c in {0.5, 0.75, 1.0, 1.25, 1.5}`.
- Intensity sweep: `Lambda in {0, 0.05, 0.1, 0.3, 1, 3}`.
- Timestep refinement for selected best/unstable settings.

Baselines: ULA, MALA, BAOAB, HMC, PT.

Metrics:

- left/right population error
- basin KL
- density L1/L2 error on grid
- IAT of right-well indicator
- ESS/sec and ESS/gradient
- mean energy and `E[x^2]` bias
- wall-clock/eval counts

Figures:

- target density and final histograms
- well population vs time
- heatmap over scale/intensity
- timestep-bias plot
- ESS/sec vs configuration

### Experiment 2: four well graph ablation

Potential:

```text
V(x,y)=(x^2-1)^2+(y^2-1)^2.
```

Purpose: test whether graph design matters for a known basin network.

Jump banks:

- `minima_edge_graph`: edges between nearest-neighbor minima: horizontal/vertical jumps `(±2,0),(0,±2)`.
- `minima_complete_graph`: all ordered/symmetric pairs among four minima.
- `random_matched_length_control`: same length distribution as complete or edge graph, randomized directions.
- no-jump local baselines.

Metrics:

- basin population error and KL
- transition matrix between four basins
- IAT/ESS for basin indicators
- observable bias for `x`, `y`, `x^2+y^2`, energy
- ESS/sec and ESS/eval

Figures:

- contour plot with jump graph overlay
- basin population bars
- transition matrices
- graph-ablation table
- metric-vs-compute plot

### Experiment 3: Müller–Brown

Purpose: demonstrate free-energy and basin-population recovery on a standard molecular enhanced-sampling landscape.

Implementation:

- Use the standard Müller–Brown potential with documented scaling.
- Define a grid domain, grid resolution, and normalized reference density.
- Locate minima by optimization or use a documented known set after verifying with local minimization.
- Define basins by nearest minimum or gradient-flow/quench assignment; document choice.

Jump banks:

- `minima_complete_graph`
- `minima_edge_graph` / saddle-network graph if robustly identified
- `random_matched_length_control`

Baselines: ULA, MALA, BAOAB, HMC, PT.

Metrics:

- basin population error/KL
- free-energy RMSE on grid: `F=-beta^{-1} log rho`
- observable bias for `V`, `x`, `y`, `|x|^2`
- IAT/ESS of basin indicators and energy
- ESS/sec and ESS/eval

Figures:

- potential contours with minima and jump arrows
- reference density/free energy
- sampled density/free energy by method
- basin population error vs compute
- free-energy RMSE vs compute

### Experiment 4: Lennard–Jones cluster

Purpose: test physical observables and inherent-structure/isomer populations beyond toy 2D landscapes.

Start with LJ7 in 2D, using the archived `archive/LJ7.py` as a reference for formulas and descriptors. If LJ7 is too slow or unstable under smoke/full constraints, switch to LJ5 and document the reason.

Potential:

```text
V(R)=sum_{i<j} 4 epsilon [(sigma/r_ij)^12 - (sigma/r_ij)^6].
```

Implementation requirements:

- Remove center of mass.
- Use sorted pair-distance descriptors for alignment/isomer assignment.
- Discover or load a low-energy isomer catalogue.
- For isomer jumps, align structures before taking displacement vectors.
- Save assumptions and catalogue metadata.

Jump banks:

- `lj_isomer_aligned`: aligned displacement between low-energy inherent structures.
- `random_matched_length_control`.
- optional small local perturbation bank only as diagnostic.

Baselines: BAOAB, MALA if feasible, HMC if stable, PT reference, LSB-MC.

Metrics:

- energy histogram
- mean and variance of potential energy
- pair-distance distribution error
- inherent-structure/isomer population error vs PT/reference
- isomer population KL
- IAT/ESS for energy and isomer labels
- ESS/sec and ESS/eval

Figures:

- isomer structures if possible
- energy histograms
- pair-distance distributions
- isomer population table/bar plot
- compute-normalized observable error

### Experiment 5: ManyWell scaling

Purpose: stress-test dimension scaling while retaining exact reference metrics.

Use the existing ManyWell design but keep it in the JCP framing as a high-dimensional metastable Boltzmann stress test.

Jump bank:

- `manywell_block_flip` only.
- Do not use an exponential full-mode jump bank.

Dimensions:

- `d in {8,16,32,64}` initially; optional larger dimensions only after core runs are stable.

Metrics:

- per-block marginal KL
- deep-well count distribution KL
- count-EMC / entropic mode coverage
- observable bias for energy and deep-count
- IAT/ESS for per-block indicators and deep-count
- ESS/sec and ESS/eval

Figures:

- count-EMC vs dimension
- per-block KL vs dimension
- error vs wall-clock/gradient/quadrature evals

## Config policy

Every full config must have a smoke config first. Smoke configs should use tiny particle counts, short horizons, reduced quadrature, and limited methods but still exercise the same code paths.

Each full YAML should include:

- `experiment_name`
- `target`
- `target_cfg`
- `reference_cfg`
- `jump_banks`
- `methods`
- `method_cfgs`
- `run`
- `metrics`
- `plotting`
- `failure_policy`

Each run must save:

- original config YAML
- resolved config JSON
- git commit hash, branch, and dirty status
- environment info
- stdout/stderr log
- metrics JSON/CSV
- time-series metrics
- figure-ready summaries
- run status/failure manifest

## GPU and launch policy

This is a shared H200 node. Use one GPU per experiment process unless there is a documented reason to do otherwise. Use at most two GPUs concurrently.

The launcher must accept:

```bash
--gpus 0,1
--max-concurrent 2
```

and set `CUDA_VISIBLE_DEVICES` for each child process. Do not launch uncontrolled sweeps. Do not run full experiments directly in a way that bypasses the launcher/concurrency limit.

## Smoke and full-run commands

Run tests first:

```bash
python -m pytest experiments/jcp_sampling/tests -q
```

Run smoke suite:

```bash
python -m experiments.jcp_sampling.scripts.launch_sweep \
  --suite experiments/jcp_sampling/configs/suites/smoke.yaml \
  --gpus 0,1 \
  --max-concurrent 2
```

Run core suite only after smoke passes:

```bash
python -m experiments.jcp_sampling.scripts.launch_sweep \
  --suite experiments/jcp_sampling/configs/suites/core.yaml \
  --gpus 0,1 \
  --max-concurrent 2
```

Then aggregate, plot, and build report assets:

```bash
python -m experiments.jcp_sampling.scripts.aggregate_results \
  --results-root results/jcp_sampling/<timestamp>

python -m experiments.jcp_sampling.scripts.make_figures \
  --results-root results/jcp_sampling/<timestamp>

python -m experiments.jcp_sampling.scripts.make_report_assets \
  --results-root results/jcp_sampling/<timestamp>

cd reports/jcp_sampling_report && tectonic main.tex
```

## Stop conditions and failure handling

Stop before full runs if:

- unit tests fail;
- any smoke config fails;
- a required artifact is missing from smoke;
- GPU assignment/concurrency validation fails;
- metrics are NaN/nonfinite without an explicit documented reason.

Abort an individual run and mark it failed if:

- states become nonfinite above threshold;
- CUDA OOM occurs;
- exponent clipping saturates beyond configured threshold;
- sampler acceptance collapses and remains below configured threshold;
- required metrics cannot be computed;
- runtime exceeds configured wall-clock limit.

Failure handling requirements:

- Preserve logs and partial metrics.
- Write `run_status.json` with failure reason.
- Continue independent jobs only if safe.
- Do not delete failed/negative results.
- Include failures or unstable settings in the report tables/limitations.
- Do not claim success based only on plots; quantitative metrics are required.

## Report plan

The final report should be manuscript-ready LaTeX under `reports/jcp_sampling_report/` and compile to `main.pdf`.

Required sections:

1. Introduction and JCP framing
2. Method summary and stationarity of the Lévy score
3. Jump design principles
4. Experiment setup and baselines
5. Metrics and compute accounting
6. Double-well results
7. Four-well graph ablation
8. Müller–Brown free-energy recovery
9. Lennard–Jones observables/isomers
10. ManyWell scaling
11. Discussion and limitations
12. Reproducibility appendix

All headline numbers and tables must be generated from CSV/JSON through `make_report_assets.py`, not typed by hand.

## Definition of done

The conversion is complete only when:

- JCP code exists under `experiments/jcp_sampling/`.
- JCP results are under `results/jcp_sampling/<timestamp>/`.
- JCP report files are under `reports/jcp_sampling_report/`.
- Unit tests pass.
- Smoke configs pass.
- Core experiments have completed or failed with documented reasons.
- Figures/tables are generated from result files.
- The LaTeX report compiles to PDF.
- Positive and negative findings are both summarized.
- A status/reproducibility note lists exact commands to reproduce the results.
