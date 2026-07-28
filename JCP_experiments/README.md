# JCP E1--E4 reproducibility release

This directory contains the manuscript code and frozen numerical outputs for
the four examples used in the JCP paper on Lévy-score-corrected
compound-Poisson (LSC-CP) sampling.

The release has three independent workflows:

1. **Review the code and frozen results on CPU.**
2. **Regenerate every manuscript figure from the frozen results on CPU.**
3. **Rerun the full stochastic experiments on CUDA GPUs.**

The first two workflows are the recommended starting point for a collaborator.
They do not rerun any sampler.

## Manuscript experiment matrix

Internal name `BAOAB` is the practical integrator for underdamped Langevin
dynamics and is displayed as **ULD** in the paper. Internal name `CP` is
displayed as **Raw-CP**.

| Example | Notebook | Methods shown in the manuscript |
|---|---|---|
| E1 double well | `notebooks/01_double_well.ipynb` | ULA, ULD, PT, FLA, Raw-CP, LSC-CP, LSC-CP-RA |
| E2 MoG40 | `notebooks/02_mog40.ipynb` | ULA, ULD, PT, FLA, LSC-CP, LSC-CP-RA |
| E3 Müller--Brown 10D | `notebooks/03_mb3well_10d.ipynb` | ULA, ULD, PT, FLA, LSC-CP, LSC-CP-RA (4) |
| E4 coupled two-component phi4 | `notebooks/04_coupled_phi4.ipynb` | ULA, ULD, PT, FLA, LSC-CP, LSC-CP-RA (8) |

The reported metrics are:

- \(W_2\);
- MMD;
- basin total variation;
- worst-basin ESS.

The first three are lower-is-better. Worst-basin ESS is higher-is-better.
Because FLA does not preserve the target, target ESS is deliberately shown as
not applicable for FLA. E1 Raw-CP ESS is reported as a mixing diagnostic but
must always be read together with its distributional bias metrics.

## Directory guide

```text
configs/                         human-readable E1--E4 run specifications
src/                             shared algorithms and experiment builders
notebooks/
  00_environment_check.ipynb     CPU-safe installation/release checks
  01_double_well.ipynb           E1, all manuscript methods
  02_mog40.ipynb                 E2, all manuscript methods
  03_mb3well_10d.ipynb           E3, all manuscript methods
  04_coupled_phi4.ipynb          E4, all manuscript methods
  05_manuscript_plotting.ipynb   plots only; never runs a sampler
scripts/                         validation, plotting, and archive tools
results/                         frozen CSV/JSON results and samples
figures/png/                     manuscript PNG files
figures/pdf/                     manuscript PDF files
tests_cpu/                       CPU reproducibility tests
```

The experiment notebooks are generated from
`notebooks/build_notebooks.py`. Algorithm implementations live under `src/`;
the notebooks orchestrate them and record provenance rather than maintaining
four copies of each method.

## 1. Create the environment

The tested release baseline is Python 3.12 with PyTorch 2.6 or newer. A newer
CUDA build of PyTorch may be used for production runs.

From this directory:

```bash
conda env create -f environment.yml
conda activate jcp-levy-release
python -m pip install -e .
```

If an existing environment is used instead, install the package and development
dependencies with:

```bash
python -m pip install -e '.[dev]'
```

## 2. Validate the unpacked release on CPU

Run the structural and frozen-result validator:

```bash
python scripts/validate_release.py --require-figures
```

Then run the focused CPU tests:

```bash
python -m pytest -q \
  tests_cpu/test_release_structure.py \
  tests_cpu/test_manuscript_replot.py \
  tests_cpu/test_runner_controls.py \
  tests_cpu/test_sampler_diagnostics.py \
  tests_cpu/test_stationarity.py
```

The same checks are available interactively in
`notebooks/00_environment_check.ipynb`.

To execute it non-interactively while preserving a clean source notebook:

```bash
mkdir -p executed_notebooks
python notebooks/run_notebook.py \
  notebooks/00_environment_check.ipynb \
  --output-notebook executed_notebooks/00_environment_check.ipynb \
  --status-path executed_notebooks/00_environment_check.status.json \
  --timeout 1800
```

## 3. Regenerate all manuscript figures on CPU

The plotting notebook reads only committed files below `results/`.

```bash
mkdir -p executed_notebooks
python notebooks/run_notebook.py \
  notebooks/05_manuscript_plotting.ipynb \
  --output-notebook executed_notebooks/05_manuscript_plotting.ipynb \
  --status-path executed_notebooks/05_manuscript_plotting.status.json \
  --timeout 7200
```

Equivalent direct commands are:

```bash
python scripts/replot_manuscript_figures.py --no-clean
python scripts/replot_generated_samples.py --overwrite
python scripts/validate_release.py --require-figures
```

Outputs are written to `figures/png/` and `figures/pdf/`. For every example,
the metric script writes individual \(W_2\), MMD, basin-TV, and worst-basin-ESS
figures and a 2-by-2 combined figure against physical time, NFE, and wall-clock
time. The generated-sample script writes density and scatter comparisons. E1
uses the manuscript display range \(x\in[-2,2]\).

## 4. Regenerate the notebook source files

Notebook regeneration does not execute any experiment:

```bash
python notebooks/build_notebooks.py
python scripts/validate_release.py
```

Do not edit generated `.ipynb` files without making the corresponding change in
`notebooks/build_notebooks.py`, or the next regeneration will overwrite it.

## 5. Full CUDA production rerun

Full experiment runs are expensive and require Linux, CUDA, and sufficient GPU
memory. Use the bounded launcher instead of executing production notebooks
directly. The launcher restricts every child to one visible GPU, records the
resolved method matrix and environment, runs the explicit E1--E4 unit-test
allow-list, and requires a real dynamics smoke before starting a full notebook.

First run only the preflight and smoke stage:

```bash
./run_production.sh --gpus 0 --max-concurrent 1 --smoke-only
```

Review the generated status, logs, and smoke artifacts. Then launch the full
E1--E4 campaign:

```bash
./run_production.sh --gpus 0,1 --max-concurrent 2
```

GPU indices are examples; replace them with devices assigned to you. Passing
`--gpus` explicitly opts those devices into the repository GPU guard.

The launcher creates immutable run directories under:

```text
../results/jcp_sampling/<run-id>/
```

It refuses to overwrite an existing run ID. Use `--run-id` to assign a stable
name and `--experiments` to run a comma-separated subset. Use
`python launch_production.py --help` for the complete interface.

Production stochastic results are expected to agree within Monte Carlo
uncertainty, not bit-for-bit across different GPU architectures or PyTorch
versions. Every run records package, hardware, seed, configuration, and Git
provenance in its manifest.

## 6. Build the collaborator archive

The archive builder validates both the research tree and the staged standalone
copy, excludes caches and macOS metadata, and writes `SHA256SUMS` inside the
archive:

```bash
python scripts/build_collaborator_zip.py \
  --output dist/JCP_levy_sampler_code.zip
```

The output path must not already exist. This fail-closed behavior prevents an
old review package from being silently replaced.

After unpacking the zip, the collaborator should start with:

```bash
conda env create -f environment.yml
conda activate jcp-levy-release
python -m pip install -e .
python scripts/validate_release.py --require-figures
```

## Source of truth and provenance

- `src/manuscript.py` is the single source of truth for the E1--E4 method
  matrix and plot labels.
- `configs/*.yaml` records the human-readable production protocol.
- Each `results/<experiment>/manifest.json` records the resolved parameters and
  provenance of the frozen result set.
- `results/*/metrics_timeseries.csv` contains the metric trajectories.
- `results/*/stationarity/*_summary.csv` contains worst-basin ESS diagnostics.
- `results/*/positions.csv` is the only sampler output used by the generated
  density/scatter figures.

Raw-CP is present only in E1 because it does not follow the target geometry.
Free-energy metrics may remain in historical CSV files for provenance, but the
manuscript plotting workflow uses only the four declared metrics above.
