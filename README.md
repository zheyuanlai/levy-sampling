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

The **internal** method matrix (what every run computes and what the result
files contain) is wider than the **manuscript display** matrix (what the
figures draw). Nothing is deleted from `results/`; the display matrix simply
selects the arms that answer each example's question.

| Example | Notebook | Methods displayed in the manuscript | Scientific role |
|---|---|---|---|
| E1 double well | `notebooks/01_double_well.ipynb` | ULA, ULD, Raw-CP, LSC-CP, LSC-CP-RA | isolate the raw-jump stationarity bias |
| E2 MoG40 | `notebooks/02_mog40.ipynb` | ULA, ULD, PT, LSC-CP, LSC-CP-RA | generic multimode transport |
| E3 Müller--Brown 10D | `notebooks/03_mb3well_10d.ipynb` | ULA, ULD, PT, LSC-CP, LSC-CP-RA (4) | relay geometry after embedding |
| E4 coupled two-component phi4 | `notebooks/04_coupled_phi4.ipynb` | ULA, ULD, PT, LSC-CP, LSC-CP-RA (8) | the coupled \(\phi^4\) chain |

The internal matrix additionally runs FLA everywhere, PT in E1, and MALA; those
curves stay in the CSV files and in `src/manuscript.py` but are not drawn.
`scripts/replot_manuscript_figures.py:REPORT_METHODS` is the display matrix.

The reported metrics are:

- \(W_2\) in E1 (exact, one-dimensional) and \(\mathrm{SW}_2\) in E2--E4
  (fixed-projection sliced Wasserstein-2). Both live in the CSV column named
  `W2`; the figures are labeled with the metric actually computed;
- MMD;
- basin total variation;
- worst-basin ESS.

The first three are lower-is-better. Worst-basin ESS is higher-is-better.
FLA and E1 Raw-CP do not preserve the target, so their worst-basin ESS values
are mixing diagnostics rather than target ESS. They must always be read
together with their distributional bias metrics.

### Wall-clock protocol

Wall-clock is a reported axis. Every method of an experiment is timed under one
protocol:

- one dedicated GPU per experiment, with no other compute process on that
  device — each run manifest records
  `hardware.gpu_compute_apps_on_own_device_at_start`, and
  `scripts/validate_release.py` refuses a run that was contended;
- one process, one visible device (`gpu_count_visible == 1`);
- the identical batched ensemble shape (seeds × particles) for every method;
- CUDA-synchronised timers around sampler work only — metrics, reference
  sampling, plotting and I/O are outside the timed region;
- 20 untimed warm-up steps on a throwaway sampler.

Because seeds are batched, wall-clock is one number per method rather than a
per-seed distribution, so the wall-clock axis carries no seed error bars by
construction. The physical-time and NFE views are unaffected.

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
figures/png/                     manuscript PNG files (600 dpi)
figures/tiff/                    manuscript TIFF files (600 dpi, LZW)
figures/svg/                     manuscript SVG files (vector)
figures/pdf/                     manuscript PDF files (vector)
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

Run the structural, frozen-result, and figure validator:

```bash
python scripts/validate_release.py --require-figures
```

The same validation is available interactively in
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

Outputs are written to `figures/png/`, `figures/tiff/`, `figures/svg/`, and
`figures/pdf/`. For every example, the metric script writes individual
\(W_2\)/\(\mathrm{SW}_2\), MMD, basin-TV, and worst-basin-ESS figures and a
2-by-2 combined figure against physical time, NFE, and wall-clock time. The
generated-sample script writes density and scatter comparisons in PNG and PDF.
E1 uses the manuscript display range \(x\in[-2,2]\).

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
resolved method matrix and environment, and validates the release before
starting the full notebooks. Launch the E1--E4 campaign with:

```bash
./run_production.sh --gpus 0 --max-concurrent 1
```

GPU indices are examples; replace them with devices assigned to you. Passing
`--gpus` explicitly opts those devices into the repository GPU guard.

Because wall-clock is a reported axis, the published campaign is run with
`--max-concurrent 1` on a single **idle** GPU, so no two experiments and no
foreign process ever share a device with a timed sampler. Running on several
GPUs at once is faster but produces timings that `scripts/validate_release.py`
will reject.

The launcher creates immutable run directories under:

```text
results/jcp_sampling/<run-id>/
```

It refuses to overwrite an existing run ID. Use `--run-id` to assign a stable
name and `--experiments` to run a comma-separated subset. Use
`python launch_production.py --help` for the complete interface.

Production stochastic results are expected to agree within Monte Carlo
uncertainty, not bit-for-bit across different GPU architectures or PyTorch
versions. Every run records package, hardware, seed, configuration, and Git
provenance in its manifest.

### 5a. Bootstrapping a new release contract

The launcher gates on `scripts/validate_release.py --require-figures` before it
starts, so it cannot run when the frozen results predate a change to the
release contract itself — as when the wall-clock columns and figures were
reinstated. `scripts/run_wallclock_campaign.py` is the driver for that case. It
issues the same `notebooks/run_notebook.py` command with the same
one-visible-GPU child environment, strictly one experiment at a time:

```bash
python scripts/run_wallclock_campaign.py --gpu 0 --run-id <run-id>
```

Its `RUN_MATRIX` pins the full released method matrix, which is wider than the
manuscript display matrix — the released CSV files also carry MALA in every
experiment, Raw-CP in E2--E4, and the single-atom LSC-CP-RA arm alongside the
multi-atom arm in E3/E4. Use the launcher for ordinary reruns.

### 5b. Promote a run into the frozen release

Runs write immutable artifacts; `results/<experiment>/` is the published tree.
The promotion step is fail-closed — it refuses a run that did not finish
successfully, that is missing any released file, or whose manifest shows a
contended GPU:

```bash
python scripts/promote_run.py --run-id <run-id>
python scripts/replot_manuscript_figures.py
python scripts/validate_release.py --require-figures
```

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

- `src/manuscript.py` is the single source of truth for the E1--E4 **internal**
  method matrix, plot labels, resource axes, and export formats.
- `scripts/replot_manuscript_figures.py:REPORT_METHODS` is the single source of
  truth for the **manuscript display** matrix — which of those methods each
  figure draws.
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
