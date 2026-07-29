"""Generate the four experiment notebooks (01-04) with nbformat.

Run from the repository's ``notebooks/`` directory: ``python build_notebooks.py``.
Markdown is kept minimal by design: target introduction + jump law / Levy
score only; hyperparameters and protocol details live in the code cells.
"""
from __future__ import annotations

import nbformat as nbf


def md(src: str):
    return nbf.v4.new_markdown_cell(src)


def code(src: str):
    return nbf.v4.new_code_cell(src)


def _release_notebook(cells) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.12"}
    return nb


def build_environment_nb() -> nbf.NotebookNode:
    return _release_notebook([
        md(r"""# E1--E4 release environment check

This notebook performs the CPU-safe checks that should be run immediately
after unpacking the collaborator archive. It validates package versions,
the four experiment/config/notebook contracts, frozen result completeness,
and manuscript figure completeness. It does **not** launch a sampler or a
production GPU experiment."""),
        code(r"""from pathlib import Path
import os
import platform
import subprocess
import sys

HERE = Path.cwd().resolve()
ROOT = HERE.parent if HERE.name == "notebooks" else HERE
if not (ROOT / "src" / "manuscript.py").is_file():
    raise RuntimeError("Run this notebook from the repository's notebooks/ directory")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
sys.path.insert(0, str(ROOT))

import matplotlib
import nbclient
import nbformat
import numpy
import pandas
import scipy
import torch
import yaml

print("Project root:", ROOT)
print("Python:", sys.version.split()[0], platform.platform())
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("SciPy:", scipy.__version__)
print("Matplotlib:", matplotlib.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())"""),
        code(r"""from scripts.validate_release import validate_release

report = validate_release(
    ROOT,
    check_results=True,
    require_figures=True,
)
print("Release validation:", report["status"])
for key, item in report["experiments"].items():
    print(key, item["results"]["release_methods"])"""),
        code(r"""print("CPU release checks passed.")"""),
    ])


def build_plotting_nb() -> nbf.NotebookNode:
    return _release_notebook([
        md(r"""# E1--E4 manuscript plotting

This notebook reads the frozen `results/` artifacts and regenerates every
manuscript metric, density, and scatter figure. It never reruns a sampler.

Outputs are written in both PNG and PDF under `figures/png/` and
`figures/pdf/`. BAOAB is labelled **ULD** in all figures."""),
        code(r"""from pathlib import Path
import os
import subprocess
import sys

HERE = Path.cwd().resolve()
ROOT = HERE.parent if HERE.name == "notebooks" else HERE
if not (ROOT / "src" / "manuscript.py").is_file():
    raise RuntimeError("Run this notebook from the repository's notebooks/ directory")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
sys.path.insert(0, str(ROOT))

from src.manuscript import EXPERIMENTS, METRICS, RESOURCE_AXES

print("Project root:", ROOT)
for key, spec in EXPERIMENTS.items():
    print(spec.number, key, "->", [spec.display_labels[m] for m in spec.methods])
print("Metrics:", METRICS)
print("Resource axes:", RESOURCE_AXES)"""),
        code(r"""from scripts.validate_release import validate_release

validate_release(ROOT, check_results=True, require_figures=False)
print("Frozen inputs are complete.")"""),
        code(r"""metric_command = [
    sys.executable,
    str(ROOT / "scripts" / "replot_manuscript_figures.py"),
    "--results-dir", str(ROOT / "results"),
    "--figures-dir", str(ROOT / "figures"),
    "--no-clean",
]
print("Running:", " ".join(metric_command))
subprocess.run(metric_command, cwd=ROOT, env=os.environ.copy(), check=True)"""),
        code(r"""sample_command = [
    sys.executable,
    str(ROOT / "scripts" / "replot_generated_samples.py"),
    "--results-root", str(ROOT / "results"),
    "--output-root", str(ROOT / "figures"),
    "--cache-root", str(ROOT / "cache" / "generated_samples"),
    "--manifest-path",
    str(ROOT / "cache" / "generated_samples" / "generated_sample_plots_manifest.json"),
    "--overwrite",
]
print("Running:", " ".join(sample_command))
subprocess.run(sample_command, cwd=ROOT, env=os.environ.copy(), check=True)"""),
        code(r"""report = validate_release(
    ROOT,
    check_results=True,
    require_figures=True,
)
png_files = sorted((ROOT / "figures" / "png").glob("*.png"))
pdf_files = sorted((ROOT / "figures" / "pdf").glob("*.pdf"))
print("Final validation:", report["status"])
print(f"Generated {len(png_files)} PNG and {len(pdf_files)} PDF files.")"""),
    ])


# ======================================================================
# shared code cells
# ======================================================================
def cell_setup(exp_name: str, builder: str, extra: str = "") -> str:
    return f'''EXPERIMENT = "{exp_name}"
import os, sys, math, time, json, re, hashlib
sys.path.insert(0, os.path.abspath(".."))
from src.gpu_guard import select_gpu
select_gpu(int(os.environ.get("JCP_GPU", "4")))
import torch
assert torch.cuda.device_count() == 1
torch.set_default_dtype(torch.float64)
import numpy as np
import pandas as pd

from src import config as C
from src.manuscript import manuscript_methods
from src.experiments import ({builder}, make_sampler_factory,
                             make_batched_factory, make_metrics)
from src.runner import (run_experiment_batched, run_one, refine_dt,
                        quadrature_refinement, write_timeseries_csv,
                        write_summary_csv, write_manifest, write_positions_csv,
                        mirror_into_repo,
                        ula_first_passage, hardware_manifest)
from src.samplers import tune_ladder
from src.certificate import make_phi_family, certificate_grid, certificate_importance
from src.plotting import metric_grid
from src.stationarity import (collect_stationary_trajectories,
                              flat_summary_rows, write_stationarity_csv,
                              write_stationarity_npz)

DEV = "cuda"
RUN_ID = os.environ.get("JCP_RUN_ID")
if not RUN_ID:
    RUN_ID = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-p{{os.getpid()}}"
if (RUN_ID in (".", "..")
        or re.fullmatch(r"[A-Za-z0-9_.-]+", RUN_ID) is None):
    raise ValueError(
        "JCP_RUN_ID must be a safe single component using letters, digits, "
        "dot, underscore, or hyphen")
RESULTS_ROOT = os.path.abspath(os.environ.get(
    "JCP_RESULTS_ROOT", os.path.join("..", "..", "results", "jcp_sampling")))
RUN_ROOT = os.path.join(RESULTS_ROOT, RUN_ID)
EXPERIMENT_ROOT = os.path.join(RUN_ROOT, EXPERIMENT)
RESULTS = os.path.join(EXPERIMENT_ROOT, "artifacts")
FIGURES = os.path.join(RESULTS, "figures")
CACHE = os.path.abspath(os.path.join("..", "cache", EXPERIMENT))
# The launcher owns the experiment directory (logs/status/executed notebook);
# this notebook exclusively creates its immutable artifacts/ leaf beneath it.
# Reusing a run ID across experiments is allowed; artifacts never overwrite.
os.makedirs(RESULTS, exist_ok=False)
os.makedirs(FIGURES, exist_ok=False)
os.makedirs(CACHE, exist_ok=True)
print("RUN_ID:", RUN_ID, "RUN_ROOT:", RUN_ROOT, "RESULTS:", RESULTS)

# Write the requested source configuration before constructing the experiment:
# expensive reference/basin construction failures must still leave provenance.
source_config_path = os.path.join(RESULTS, "original_config.yaml")
_prebuild_config = dict(
    schema_version=1,
    experiment=EXPERIMENT,
    builder_function="{builder}",
    builder_invocation={extra!r},
    requested_methods=os.environ.get(
        "JCP_METHODS", ",".join(manuscript_methods(EXPERIMENT))).split(","),
    requested_trace_environment=dict(
        JCP_TRACE_SEEDS=os.environ.get("JCP_TRACE_SEEDS", "4"),
        JCP_TRACE_CHAINS=os.environ.get("JCP_TRACE_CHAINS", "8"),
        JCP_TRACE_DRAWS=os.environ.get("JCP_TRACE_DRAWS", "1000"),
        JCP_TRACE_SETTLING_BURN_FRACTION=os.environ.get(
            "JCP_TRACE_SETTLING_BURN_FRACTION", "1.0"),
        JCP_PT_TRACE_BURN_FRACTION=os.environ.get(
            "JCP_PT_TRACE_BURN_FRACTION",
            os.environ.get("JCP_TRACE_SETTLING_BURN_FRACTION", "1.0"))),
    cache_directory=CACHE,
    cache_policy="builder-validated production cache; path and SHA256 recorded after build",
    requested_failure_threshold_environment=dict(
        JCP_MAX_SCORE_CLIP_FRACTION=os.environ.get(
            "JCP_MAX_SCORE_CLIP_FRACTION", "0.01"),
        JCP_MAX_STATE_BOX_CLIP_FRACTION=os.environ.get(
            "JCP_MAX_STATE_BOX_CLIP_FRACTION", "0.01"),
        JCP_MAX_JUMP_BOUNDARY_CLIP_FRACTION=os.environ.get(
            "JCP_MAX_JUMP_BOUNDARY_CLIP_FRACTION", "0.0"),
        JCP_MAX_BASIN_MAP_OUTSIDE_FRACTION=os.environ.get(
            "JCP_MAX_BASIN_MAP_OUTSIDE_FRACTION", "0.001"),
        JCP_MAX_JUMP_CAP_HITS=os.environ.get("JCP_MAX_JUMP_CAP_HITS", "0"),
        JCP_MIN_MALA_ACCEPTANCE=os.environ.get(
            "JCP_MIN_MALA_ACCEPTANCE", "0.05"),
        JCP_MIN_PT_SWAP_ACCEPTANCE=os.environ.get(
            "JCP_MIN_PT_SWAP_ACCEPTANCE", "0.10")),
)
with open(source_config_path, "x", encoding="utf-8") as config_handle:
    config_handle.write("# YAML 1.2 source configuration; created before experiment build\\n")
    for config_key, config_value in _prebuild_config.items():
        config_handle.write(
            config_key + ": " + json.dumps(config_value, sort_keys=True,
                                             allow_nan=False) + "\\n")
{extra}
cfg = exp.cfg
_basin_cache_provenance = exp.extras.get("basin_cache_provenance")
if (_basin_cache_provenance is not None
        and _basin_cache_provenance.get("validation_status")
        not in ("validated", "created_validated")):
    raise RuntimeError(
        "production requires a fully metadata-validated basin cache; got "
        + repr(_basin_cache_provenance))
# Method matrix: seven configured methods by default; override with JCP_METHODS
# (comma-separated). The production launcher uses exact+RA for E1/E2 and the
# paired multi-atom estimator for E3/E4.
RUN_METHODS = os.environ.get(
    "JCP_METHODS", ",".join(manuscript_methods(EXPERIMENT))).split(",")

# Declared fail-closed numerical thresholds. Environment overrides are recorded
# in both source and resolved configs; they cannot silently change a run.
FAIL_THRESHOLDS = dict(
    max_score_clip_fraction=float(os.environ.get(
        "JCP_MAX_SCORE_CLIP_FRACTION", "0.01")),
    max_state_box_clip_fraction=float(os.environ.get(
        "JCP_MAX_STATE_BOX_CLIP_FRACTION", "0.01")),
    # Tolerance for pi-targeting LSC-CP methods only (raw CP is recorded, not
    # gated). Zero is unsatisfiable: state-independent jumps give every
    # CP-family sampler a small boundary-contact rate on any finite box
    # (measured LSC-CP rates 2e-4..2e-3 per applied jump at production
    # dynamics config, 2026-07-17 probe).
    max_jump_boundary_clip_fraction=float(os.environ.get(
        "JCP_MAX_JUMP_BOUNDARY_CLIP_FRACTION", "0.01")),
    max_basin_map_outside_fraction=float(os.environ.get(
        "JCP_MAX_BASIN_MAP_OUTSIDE_FRACTION", "0.001")),
    max_jump_cap_hits=int(os.environ.get("JCP_MAX_JUMP_CAP_HITS", "0")),
    min_mala_acceptance=float(os.environ.get(
        "JCP_MIN_MALA_ACCEPTANCE", "0.05")),
    min_pt_swap_acceptance=float(os.environ.get(
        "JCP_MIN_PT_SWAP_ACCEPTANCE", "0.10")),
)
if (not 0.0 <= FAIL_THRESHOLDS["max_score_clip_fraction"] <= 1.0
        or not 0.0 <= FAIL_THRESHOLDS["max_state_box_clip_fraction"] <= 1.0
        or not 0.0 <= FAIL_THRESHOLDS["max_jump_boundary_clip_fraction"] <= 1.0
        or not 0.0 <= FAIL_THRESHOLDS["max_basin_map_outside_fraction"] <= 1.0
        or not 0.0 <= FAIL_THRESHOLDS["min_mala_acceptance"] <= 1.0
        or not 0.0 <= FAIL_THRESHOLDS["min_pt_swap_acceptance"] <= 1.0
        or FAIL_THRESHOLDS["max_jump_cap_hits"] < 0):
    raise ValueError("invalid JCP numerical failure threshold")

# Capture the requested trace settings before any certificate/refinement gate.
# The final resolved config also records the bounded/derived trace protocol.
TRACE_REQUEST = dict(
    seed_count=int(os.environ.get("JCP_TRACE_SEEDS", "4")),
    chains_per_seed=int(os.environ.get("JCP_TRACE_CHAINS", "8")),
    draws=int(os.environ.get("JCP_TRACE_DRAWS", "1000")),
    settling_burn_fraction=float(os.environ.get(
        "JCP_TRACE_SETTLING_BURN_FRACTION", "1.0")),
    pt_burn_fraction=float(os.environ.get(
        "JCP_PT_TRACE_BURN_FRACTION",
        os.environ.get("JCP_TRACE_SETTLING_BURN_FRACTION", "1.0"))),
)
# Freeze the actual model/bank/domain immediately after construction, before
# expensive certificates, reference refinement, or dynamics can fail.  The
# prebuild source file above remains unchanged; all resolved builder/model/
# cache details belong in a separate immutable preflight JSON artifact.
_jump_config = dict(type=type(exp.law).__name__)
if hasattr(exp.law, "atoms"):
    _jump_config.update(
        atoms=exp.law.atoms.detach().cpu().tolist(),
        weights=exp.law.weights.detach().cpu().tolist(),
        shell_half_width=exp.law.h.detach().cpu().tolist())
    if hasattr(exp.law, "jitter_sigma"):
        _jump_config["jitter_sigma"] = float(exp.law.jitter_sigma)
elif hasattr(exp.law, "a") and hasattr(exp.law, "b"):
    _jump_config.update(
        annulus_inner_radius=float(exp.law.a),
        annulus_outer_radius=float(exp.law.b))
_box_config = dict(
    type=type(exp.box).__name__,
    coordinates=("latent" if type(exp.box).__name__ == "LatentRectBox"
                 else "sampling"),
    lower=exp.box.lo.detach().cpu().tolist(),
    upper=exp.box.hi.detach().cpu().tolist())
_cache_artifacts = []
for _cache_name in sorted(os.listdir(CACHE)):
    _cache_path = os.path.join(CACHE, _cache_name)
    if os.path.isfile(_cache_path):
        with open(_cache_path, "rb") as _cache_handle:
            _cache_sha256 = hashlib.sha256(_cache_handle.read()).hexdigest()
        _cache_artifacts.append(dict(
            path=os.path.abspath(_cache_path), sha256=_cache_sha256))
preflight_config = dict(
    schema_version=1,
    experiment=EXPERIMENT,
    source="resolved immediately after experiment/model/cache build",
    source_config_file=os.path.abspath(source_config_path),
    builder=dict(
        function="{builder}", invocation={extra!r},
        parameters=exp.extras.get("builder_reference_parameters", dict())),
    model=dict(
        experiment_name=exp.name,
        potential_type=type(exp.pot).__name__,
        dimension=cfg.d,
        beta=cfg.beta),
    run_config=dict(
        d=cfg.d, n_particles=cfg.n_particles, T=cfg.T, dt=cfg.dt,
        beta=cfg.beta, eps=cfg.eps, lam=cfg.lam,
        seeds=list(cfg.seeds), n_checkpoints=cfg.n_checkpoints,
        q_theta=cfg.q_theta, q_rho=cfg.q_rho),
    jump_law=_jump_config,
    sampling_box=_box_config,
    sampling_box_design=exp.extras.get("sampling_box_design"),
    cp_drift_cap=float(exp.cp_drift_cap),
    pt_beta_min=float(exp.pt_beta_min),
    reference_and_cache=dict(
        sample_method=exp.extras.get("reference_sample_method", "unspecified"),
        scalar_method=exp.extras.get(
            "reference_scalar_method",
            exp.extras.get("reference_sample_method", "unspecified")),
        builder_parameters=exp.extras.get(
            "builder_reference_parameters", dict()),
        cache_directory=os.path.abspath(CACHE),
        cache_artifacts=_cache_artifacts,
        basin_cache_provenance=_basin_cache_provenance,
        diagnostics=exp.extras.get("reference_diagnostics")),
    requested_methods=list(RUN_METHODS),
    trace_request=TRACE_REQUEST,
    failure_thresholds=FAIL_THRESHOLDS,
    hardware_and_git=hardware_manifest(),
)
preflight_config_path = os.path.join(RESULTS, "resolved_preflight_config.json")
write_manifest(preflight_config_path, **preflight_config)
with open(preflight_config_path, "rb") as _preflight_handle:
    _preflight_config_sha256 = hashlib.sha256(_preflight_handle.read()).hexdigest()
source_config = dict(_prebuild_config)
_reference_basin_map_outside_mass = float(
    exp.extras.get("reference_diagnostics", dict()).get(
        "weighted_basin_map_outside_mass", 0.0))
if (_reference_basin_map_outside_mass
        > FAIL_THRESHOLDS["max_basin_map_outside_fraction"]):
    raise RuntimeError(
        "reference basin-map outside mass %g exceeds declared threshold %g"
        % (_reference_basin_map_outside_mass,
           FAIL_THRESHOLDS["max_basin_map_outside_fraction"]))

CERTIFICATE_TOLERANCE = 1e-6
certificate_result_path = os.path.join(RESULTS, "certificate_result.json")
def persist_certificate_result(report, settings):
    """Persist the measured deployed-quadrature certificate before asserting."""
    _max_residual = float(report["max_residual"])
    _payload = dict(
        schema_version=1,
        experiment=EXPERIMENT,
        stage="final gate immediately before production dynamics",
        settings=dict(settings),
        tolerance=CERTIFICATE_TOLERANCE,
        max_residual=_max_residual,
        passed=bool(_max_residual < CERTIFICATE_TOLERANCE),
        report=report,
        resolved_preflight_config_file=os.path.abspath(preflight_config_path),
        resolved_preflight_config_sha256=_preflight_config_sha256,
    )
    write_manifest(certificate_result_path, **_payload)
    return _payload

print("wrote immutable source config before build:", source_config_path)
print("wrote resolved preflight config after model/cache build:",
      preflight_config_path, "sha256", _preflight_config_sha256)
print(f"experiment={{cfg.name}}  d={{cfg.d}}  N={{cfg.n_particles}}  T={{cfg.T}}  dt0={{cfg.dt}}")
print(f"beta={{cfg.beta}}  eps={{cfg.eps}}  lambda={{cfg.lam}}  seeds={{cfg.seeds}}")
print("RUN_METHODS:", RUN_METHODS)
print(preflight_config["hardware_and_git"])'''


CELL_LADDER = '''# PT: geometric ladder beta_k = beta * r^(k-1); K tuned so the post-burn-in
# swap acceptance lands in [0.2, 0.4]
gen = torch.Generator(device=DEV); gen.manual_seed(0)
x0_pilot = exp.init_fn(min(512, cfg.n_particles), gen)
pt_betas, ladder_info = tune_ladder(exp.pot, x0_pilot, cfg.dt, exp.box,
                                    cfg.beta, exp.pt_beta_min, pilot_steps=20_000)
print(f"PT ladder: K={ladder_info['K']}  r={ladder_info['r']:.4f}  "
      f"beta_K={pt_betas[-1].item():.4f}  swap acceptance={ladder_info['swap_acceptance']:.3f}"
      f"  band_attained={ladder_info['band_attained']}")'''


CELL_REFERENCE = '''# frozen reference sample (size N), frozen sliced-W2 projections, frozen MMD
# bandwidth (median heuristic on the reference); bias floors from 20
# independent reference pairs. EMC convention: exp(H(p_hat))/K for uniform
# p*, 1 - EJS(p_hat, p*) otherwise -- near 1 is better in both cases.
metrics_fn, floors, aux = make_metrics(exp, cfg.n_particles)
emc_target = exp.emc_target
print("p_star:", np.round(exp.p_star.cpu().numpy(), 6),
      " uniform:", exp.uniform_target)
print("MMD bandwidth:", round(aux["bandwidth"], 4))
print("reference sample:", aux["sample_reference_method"],
      " scalar reference:", aux["scalar_reference_method"],
      " energy histogram:", aux["energy_reference_method"])
print("FES:", aux["fes_dim"], "D,", aux["fes_bins_per_dim"],
      "bins/dim, support threshold", aux["fes_pi_min"],
      "reference-weighted RMSE in kBT; reference:",
      aux["fes_reference_method"])
if aux.get("reference_diagnostics"):
    print("reference diagnostics:", aux["reference_diagnostics"])
for k, v in floors.items():
    print(f"  floor {k:>12s}: {v['mean']:.5f} +- {v['std']:.5f}")'''


def cell_dt_production(main_metrics: str) -> str:
    return f'''# dt rule: largest dyadic dt at which every PI-TARGETING method's terminal
# metrics agree with dt/2 (5% / floor-band / 4-sigma noise guards); FLA and
# raw CP have invariant laws != pi and are recorded but do not gate.
# Production: all seeds batched into one (S*N)-particle ensemble per method.
MAIN_METRICS = {main_metrics}

def run_terminal_all(dt_):
    n_ = int(round(cfg.T / dt_))
    factory = make_sampler_factory(exp, dt_, pt_betas, score_kwargs=CHOSEN_QUAD)
    out = {{}}
    for m in RUN_METHODS:
        rows_, _ = run_one(m, 0, factory, n_, n_, dt_, metrics_fn, exp.pot, quiet=True)
        out[m] = {{k: rows_[-1][k] for k in MAIN_METRICS}}
    print(f"  refine_dt: finished pass at dt={{dt_}}", flush=True)
    return out

dt_final, dt_table = refine_dt(run_terminal_all, cfg.dt, floors,
                               exclude=("FLA", "CP", "CP-RA"))
print("chosen dt:", dt_final)
for row in dt_table:
    print(row)

n_steps = int(round(cfg.T / dt_final))
steps_per_ck = max(1, n_steps // cfg.n_checkpoints)
# dense-early checkpoint schedule: the nonlocal transient lives in the
# first ~5% of the run; 60 dense + 160 sparse points, identical across
# methods (measurement cadence only -- no protocol change)
from src.runner import checkpoint_schedule
ck_steps = checkpoint_schedule(n_steps)
bfactory = make_batched_factory(exp, dt_final, pt_betas, cfg.seeds,
                                score_kwargs=CHOSEN_QUAD)
t0 = time.time()
rows, method_info = run_experiment_batched(RUN_METHODS, cfg.seeds, bfactory,
                                           n_steps, steps_per_ck, dt_final,
                                           metrics_fn, exp.pot,
                                           cfg.n_particles,
                                           checkpoint_steps=ck_steps)
print(f"production total: {{time.time()-t0:.0f}}s")

def _diagnostic_max(name, method=None):
    return max((float(row[name]) for row in rows
                if name in row and (method is None or row["method"] == method)),
               default=0.0)

def _method_info_max(name):
    return max((float(info[name]) for info in method_info.values()
                if name in info), default=0.0)

def _method_info_value(method, name, default):
    return float(method_info.get(method, {{}}).get(name, default))

def _targeting_method_info_max(name):
    return max((float(info[name]) for method, info in method_info.items()
                if method not in ("FLA", "CP", "CP-RA") and name in info),
               default=0.0)

def _targeting_diagnostic_max(name):
    return max((float(row[name]) for row in rows
                if row["method"] not in ("FLA", "CP", "CP-RA")
                and name in row), default=0.0)

def _lsc_cp_method_info_max(name):
    return max((float(info[name]) for method, info in method_info.items()
                if method.startswith("LSC-CP") and name in info), default=0.0)

def _raw_cp_method_info_max(name):
    return max((float(info[name]) for method, info in method_info.items()
                if method.startswith("CP") and name in info), default=0.0)

observed_failure_diagnostics = dict(
    nonfinite_fraction=_diagnostic_max("nonfinite_frac"),
    nonfinite_proposal_count=_method_info_max(
        "nonfinite_proposal_count_cumulative"),
    # Gate the final lifetime ratios saved once per batched method. Taking a
    # max over checkpoint prefixes is not the same estimand because cumulative
    # fractions may decrease as more proposals arrive.
    score_clip_fraction=_method_info_max(
        "score_clip_fraction_cumulative"),
    state_box_clip_fraction=_method_info_max(
        "state_box_clip_fraction_cumulative"),
    # Jumps fire state-independently, so rare multi-jump excursions contact
    # any finite box; a hard-zero gate is unsatisfiable at production scale.
    # Gate pi-targeting LSC-CP methods against the declared tolerance. Raw
    # CP's invariant law != pi and its (larger) boundary-contact rate is part
    # of the documented defect: recorded, not gated, exactly like the dt and
    # basin-map gates.
    jump_boundary_clip_fraction_cp=(
        _lsc_cp_method_info_max(
            "jump_boundary_clip_fraction_per_applied_jump_cumulative")),
    jump_boundary_clip_fraction_raw_cp=(
        _raw_cp_method_info_max(
            "jump_boundary_clip_fraction_per_applied_jump_cumulative")),
    basin_map_outside_fraction_targeting=(
        _targeting_diagnostic_max("basin_map_outside_mass")),
    reference_basin_map_outside_mass=_reference_basin_map_outside_mass,
    jump_cap_hits=_method_info_max("jump_cap_hit_count_cumulative"),
    # Use exact accepted/proposed lifetime ratios from the intended method.
    # PT's within-replica MALA counts remain reported but do not enter the
    # standalone MALA gate.
    mala_acceptance_mean=_method_info_value(
        "MALA", "mala_accept_fraction_cumulative", -1.0),
    pt_swap_acceptance_mean=_method_info_value(
        "PT", "pt_swap_accept_fraction_cumulative", -1.0),
)
if observed_failure_diagnostics["nonfinite_fraction"] != 0.0:
    raise RuntimeError("nonfinite production state/metric fraction is nonzero")
if observed_failure_diagnostics["nonfinite_proposal_count"] != 0.0:
    raise RuntimeError("nonfinite production proposal count is nonzero")
for observed_name, threshold_name in (
        ("score_clip_fraction", "max_score_clip_fraction"),
        ("state_box_clip_fraction", "max_state_box_clip_fraction"),
        ("jump_boundary_clip_fraction_cp",
         "max_jump_boundary_clip_fraction"),
        ("basin_map_outside_fraction_targeting",
         "max_basin_map_outside_fraction"),
        ("reference_basin_map_outside_mass",
         "max_basin_map_outside_fraction"),
        ("jump_cap_hits", "max_jump_cap_hits")):
    observed = observed_failure_diagnostics[observed_name]
    threshold = FAIL_THRESHOLDS[threshold_name]
    if observed > threshold:
        raise RuntimeError("%s=%g exceeds declared %s=%g" %
                           (observed_name, observed, threshold_name, threshold))
# Acceptance floors gate a method's own mixing, so they apply only when that
# method is in this run's matrix. The -1.0 sentinel means "method absent"; when
# the production method set is sharded across GPUs, a shard can legitimately
# lack MALA or PT,
# and its acceptance floor must be skipped rather than fail on the sentinel. A
# method that IS present but produced no acceptance statistic still fails, since
# then the value is a real (nonnegative) ratio below the floor rather than -1.
for observed_name, threshold_name, guard_method in (
        ("mala_acceptance_mean", "min_mala_acceptance", "MALA"),
        ("pt_swap_acceptance_mean", "min_pt_swap_acceptance", "PT")):
    if guard_method not in RUN_METHODS:
        continue
    observed = observed_failure_diagnostics[observed_name]
    threshold = FAIL_THRESHOLDS[threshold_name]
    if observed < threshold:
        raise RuntimeError("%s=%g is below declared %s=%g" %
                           (observed_name, observed, threshold_name, threshold))
print("fail-closed diagnostics:", observed_failure_diagnostics,
      "thresholds:", FAIL_THRESHOLDS)'''


CELL_FIGURES = '''from src.plotting import metric_single
from src.runner import convergence_report
# Plotted-method policy: BOTH LSC arms are shown in every experiment -- the
# exact deterministic-quadrature score "LSC-CP" (black) and one realised-
# displacement estimator (purple): single-atom RA on E1/E2, atom-stratified MA
# on E3/E4. The MA arm reads as "LSC-CP-RA (A)" with A the atom count.
PLOT_RAW = "CP" if "CP" in RUN_METHODS else (
    "CP-RA" if "CP-RA" in RUN_METHODS else None)
# Realised LSC arms are one family "LSC-CP-RA (k)" by atoms-per-step k: genuine
# single-atom RA is k=1, multi-atom MA is k=A. E1/E2 (continuous laws) keep the
# plain "LSC-CP-RA".
_ARM_ATOMS = {"mb3well_10d": 4, "coupled_phi4": 8}
def _realised_label(arm):
    A = _ARM_ATOMS.get(EXPERIMENT)
    if arm == "LSC-CP-RA":
        return "LSC-CP-RA (1)" if A else "LSC-CP-RA"
    if arm == "LSC-CP-MA":
        return f"LSC-CP-RA ({A})" if A else "LSC-CP-MA"
    return arm
PLOT_METHODS = [m for m in ("ULA", "MALA", "FLA", "BAOAB", "PT")
                if m in RUN_METHODS]
PLOT_LABELS = {}
if PLOT_RAW is not None:
    PLOT_METHODS.append(PLOT_RAW)
    PLOT_LABELS[PLOT_RAW] = "Raw-CP"
if "LSC-CP" in RUN_METHODS:
    PLOT_METHODS.append("LSC-CP")
    PLOT_LABELS["LSC-CP"] = "LSC-CP"
for _ra in ("LSC-CP-RA", "LSC-CP-MA"):
    if _ra in RUN_METHODS:
        PLOT_METHODS.append(_ra)
        PLOT_LABELS[_ra] = _realised_label(_ra)
print("plotted methods:", PLOT_METHODS, " labels:", PLOT_LABELS)

fig = metric_grid(rows, os.path.join(FIGURES, EXPERIMENT + "_metrics"),
                  metrics=("W2", "MMD", "EMC"), floors=floors,
                  emc_target=emc_target, methods=PLOT_METHODS,
                  label_overrides=PLOT_LABELS)
print("saved:", os.path.join(FIGURES, EXPERIMENT + "_metrics") + ".{png,pdf}")

# per-metric log-y single figures on t / NFE axes (all curves start
# at the shared n=0 point; linear x so t=0 / NFE=0 is representable; the cost
# axes are truncated at the largest non-LSC terminal x -- metric_single's
# xmax_mode="baselines" -- so the 10-30x-per-step LSC-CP curve cannot squeeze
# every baseline against the y-axis)
_present = set().union(*[set(r) for r in rows])
# every per-metric figure on disk is regenerated here, so none can go stale
# with old data/axes when the notebook re-runs
_single = [m for m in ("W2", "TV", "TV_density", "MMD",
                       "FES_RMSE_kBT", "FES_outside_mass", "basin_KL_target",
                       "e_F", "basin_rel_max", "KSD", "W1_cdf", "CDF_sup",
                       "pdf_L1", "KDE_chi2", "W2_10d")
           if m in _present and not (m == "e_F" and "FES_RMSE_kBT" in _present)]
for _m in _single:
    for _axis in ("t", "nfe"):
        metric_single(rows, _m, os.path.join(FIGURES, f"{EXPERIMENT}_{_m}_{_axis}"),
                      xaxis=_axis, floors=floors, methods=PLOT_METHODS,
                      emc_target=emc_target, label_overrides=PLOT_LABELS,
                      show=False)
print("saved per-metric log-y figures:", _single, "x {t, nfe}")

# ---- sample-space figures. Terminal samples are persisted FIRST and then read
# ---- back, so these figures are exactly what a CSV-only replot reproduces.
# ---- metric_space maps to the plane the metrics live on (identity for E1/E2,
# ---- latent for E3, qbar for E4).
from src.runner import write_positions_csv
from src.plotting import (load_positions_csv, density_overlay, fes_ceiling,
                          fes_profile_1d, fes_map_2d, REFERENCE_KEY)
_pos = {m: exp.metric_space(method_info[m]["final_positions_all"])[:, :2]
        for m in RUN_METHODS}
_pos[REFERENCE_KEY] = exp.metric_space(aux["ref_x"])[:, :2]
_pos_csv = os.path.join(RESULTS, "positions.csv")
_n_pos = write_positions_csv(_pos, _pos_csv)
print(f"positions.csv: {_n_pos} rows over {len(_pos)} blocks "
      f"({', '.join(_pos)})")
_P = load_positions_csv(_pos_csv)
_dim = _P[REFERENCE_KEY].shape[1]
_plot_pos = [m for m in PLOT_METHODS if m in _P]

for _m in _plot_pos:
    density_overlay(_P, _m, os.path.join(FIGURES, f"{EXPERIMENT}_density_{_m}"),
                    label_overrides=PLOT_LABELS, show=False)
print(f"saved density overlays vs reference: {_plot_pos}")

if _dim == 1:
    fes_profile_1d(_P, os.path.join(FIGURES, f"{EXPERIMENT}_FES_profile"),
                   beta=cfg.beta, methods=_plot_pos,
                   label_overrides=PLOT_LABELS, show=False)
    print("saved 1-D FES profile (true + every method, one figure)")
else:
    _fmax = fes_ceiling(_P, beta=cfg.beta)
    for _m in [REFERENCE_KEY] + _plot_pos:
        fes_map_2d(_P, _m, os.path.join(FIGURES, f"{EXPERIMENT}_FES_{_m}"),
                   beta=cfg.beta, fmax=_fmax, label_overrides=PLOT_LABELS,
                   show=False)
    print(f"saved 2-D FES maps (shared F ceiling {_fmax:.2f} kT): "
          f"{[REFERENCE_KEY] + _plot_pos}")

# R-hat/IAT are deliberately NOT computed from the sparse, nonstationary
# relaxation checkpoints above.  The separate reference-start trace cell
# reports them from uniform post-step scalar trajectories only.
print("stationary diagnostics are in", os.path.join(RESULTS, "stationarity"))'''


CELL_STATIONARITY = r"""# Uniform scalar trajectories for IAT/ESS/R-hat.
# These are separate from the sparse, nonstationary relaxation checkpoints.
# FLA and E1 Raw-CP are retained as explicitly non-targeting mixing diagnostics:
# their ESS must be interpreted together with (never instead of) target bias.
STATIONARY_METHODS = [m for m in RUN_METHODS
                      if m not in ("CP", "CP-RA")]
if EXPERIMENT == "double_well" and "CP" in RUN_METHODS:
    STATIONARY_METHODS.append("CP")
TRACE_SEED_COUNT = min(len(cfg.seeds), int(os.environ.get("JCP_TRACE_SEEDS", "4")))
TRACE_CHAINS_PER_SEED = int(os.environ.get("JCP_TRACE_CHAINS", "8"))
TRACE_DRAWS_REQUESTED = int(os.environ.get("JCP_TRACE_DRAWS", "1000"))
if TRACE_SEED_COUNT < 1 or TRACE_CHAINS_PER_SEED < 1 or n_steps < 2:
    raise ValueError("stationary traces require >=1 seed/chain and >=2 production steps")
TRACE_SEEDS = tuple(cfg.seeds[:TRACE_SEED_COUNT])
TRACE_DRAWS = min(max(2, TRACE_DRAWS_REQUESTED), n_steps)
TRACE_STEPS_PER_DRAW = max(1, n_steps // TRACE_DRAWS)
TRACE_SETTLING_BURN_FRACTION = float(os.environ.get(
    "JCP_TRACE_SETTLING_BURN_FRACTION", "1.0"))
TRACE_PT_BURN_FRACTION = float(os.environ.get(
    "JCP_PT_TRACE_BURN_FRACTION", str(TRACE_SETTLING_BURN_FRACTION)))
if TRACE_SETTLING_BURN_FRACTION < 0 or TRACE_PT_BURN_FRACTION < 0:
    raise ValueError("stationary settling-burn fractions must be non-negative")
TRACE_SETTLING_BURN = int(round(TRACE_SETTLING_BURN_FRACTION * n_steps))
TRACE_PT_BURN = int(round(TRACE_PT_BURN_FRACTION * n_steps))

trace_factory = make_sampler_factory(
    exp, dt_final, pt_betas, n_particles=TRACE_CHAINS_PER_SEED,
    score_kwargs=CHOSEN_QUAD, reference_init=True)
reference_cv_means = aux["reference_cv_means"]
cv_names = ["x"] if len(reference_cv_means) == 1 else [
    f"cv_{j}" for j in range(len(reference_cv_means))]
reference_method = aux["sample_reference_method"]
# Zero burn for MALA is an explicit opt-in, never inferred from a friendly
# sampler name. The present inverse-CDF/grid/SIR references are numerical or
# approximate; E2's exact unbounded MoG draw is not exactly the box-restricted
# MALA target. Therefore every current experiment receives charged settling.
reference_is_exact_target_draw = bool(
    exp.extras.get("reference_exact_for_mala", False))

_common = dict(
    sampler_factory=trace_factory, seeds=TRACE_SEEDS,
    n_draws=TRACE_DRAWS, steps_per_draw=TRACE_STEPS_PER_DRAW, dt=dt_final,
    labels_fn=exp.labels_fn, energy_fn=exp.pot.V, cv_fn=exp.metric_space,
    counter_source=exp.pot, warmup_steps=C.N_WARMUP_STEPS,
    basin_ids=list(range(exp.p_star.numel())), cv_names=cv_names,
    basin_target_probabilities=exp.p_star.cpu().tolist(),
    reference_energy_mean=aux["reference_energy_mean"],
    reference_cv_means=reference_cv_means,
)
stationarity = {"schema_version": 1, "methods": {}, "collection": {}}

def _collect_stationary_group(key, methods, burn_steps, exact_stationary_start,
                              initialization_method):
    if not methods:
        return
    collected = collect_stationary_trajectories(
        methods=methods, burn_in_steps=burn_steps,
        equilibrium_initialized=exact_stationary_start,
        initialization_method=initialization_method, **_common)
    stationarity["methods"].update(collected["methods"])
    stationarity["collection"][key] = collected["collection"]

# Only target-start MALA begins stationary for its exact discrete-time kernel.
# ULA, BAOAB, and split LSC-CP have finite-step invariant laws that need not be
# pi, so their target draw is NOT a stationary kernel draw: charge a T-length
# settling run before measuring IAT/ESS. With an approximate SIR reference,
# MALA is settled as well. Bias against pi remains beside every ESS value.
_exact_mala = (["MALA"] if reference_is_exact_target_draw
               and "MALA" in STATIONARY_METHODS else [])
_settled_nonpt = [m for m in STATIONARY_METHODS
                  if m != "PT" and m not in _exact_mala]
_collect_stationary_group(
    "exact_target_start_mala", _exact_mala, 0, True,
    reference_method + ":exact_target_draw_for_invariant_MALA")
_collect_stationary_group(
    "charged_settling_non_pt", _settled_nonpt, TRACE_SETTLING_BURN, False,
    reference_method + ":reference_draw_then_charged_kernel_settling")
if "PT" in STATIONARY_METHODS:
    # PT replicates one cold reference draw over the ladder, not the joint
    # tempered equilibrium. Its declared burn is timed and charged.
    _collect_stationary_group(
        "charged_settling_pt", ["PT"], TRACE_PT_BURN, False,
        reference_method + ":cold_draw_replicated_over_ladder_then_charged_settling")

_stationary_rows = []
if stationarity["methods"]:
    _stationary_dir = os.path.join(RESULTS, "stationarity")
    os.makedirs(_stationary_dir, exist_ok=False)
    for _method, _result in stationarity["methods"].items():
        _summary = _result["summary"]
        _raw = _result["raw"]
        write_stationarity_csv(
            os.path.join(_stationary_dir, f"{_method}_summary.csv"), _summary)
        write_stationarity_npz(
            os.path.join(_stationary_dir, f"{_method}_traces.npz"),
            trace_times=_raw["trace_times"], positions_t=_raw["positions_t"],
            labels_t=_raw["labels_t"], energy_t=_raw["energy_t"], cv_t=_raw["cv_t"],
            seed_index=_raw["seed_index"],
            chain_index_within_seed=_raw["chain_index_within_seed"],
            summary=_summary,
            metadata={"experiment": EXPERIMENT, "run_id": RUN_ID, "method": _method})
        _stationary_rows.extend(flat_summary_rows(_summary))
    _stationary_table = pd.DataFrame(_stationary_rows)
    _stationary_table.to_csv(
        os.path.join(_stationary_dir, "all_methods_summary.csv"),
        index=False, mode="x")
else:
    _stationary_dir = None
    _stationary_table = pd.DataFrame()

stationarity_manifest = {
    "methods": {m: r["summary"] for m, r in stationarity["methods"].items()},
    "collection": stationarity["collection"],
    "protocol": {
        "exact_zero_burn_methods": _exact_mala,
        "charged_settling_non_pt_methods": _settled_nonpt,
        "settling_burn_steps": TRACE_SETTLING_BURN,
        "pt_burn_steps": TRACE_PT_BURN,
        "recorded_steps": TRACE_DRAWS * TRACE_STEPS_PER_DRAW,
        "reference_sample_method": reference_method,
        "basin_target_method": aux["scalar_reference_method"],
        "energy_target_method": aux["scalar_reference_method"],
        "cv_target_method": aux["scalar_reference_method"],
        "reference_exact_for_mala": reference_is_exact_target_draw,
        "reference_requires_settling": not reference_is_exact_target_draw,
        "sir_reference_is_approximate": (
            reference_method == "sampling_importance_resampling"),
    },
    "excluded_non_targeting_methods": [m for m in RUN_METHODS
                                         if m == "CP-RA"
                                         or (m == "CP" and EXPERIMENT != "double_well")],
    "non_targeting_mixing_diagnostics": {
        m: {
            "interpretation": "biased-kernel mixing diagnostic; not target ESS",
            "must_report_with_target_bias": True,
        }
        for m in STATIONARY_METHODS if m in ("FLA", "CP")
    },
}
print("stationary trace protocol:", TRACE_SEEDS, TRACE_CHAINS_PER_SEED,
      "chains/seed,", TRACE_DRAWS, "draws, stride", TRACE_STEPS_PER_DRAW,
      "recorded steps", TRACE_DRAWS * TRACE_STEPS_PER_DRAW,
      "settling burn", TRACE_SETTLING_BURN, "PT burn", TRACE_PT_BURN)
if not _stationary_table.empty:
    display(_stationary_table[
        ["method", "kind", "name", "mean", "target", "absolute_bias", "iat",
         "ess", "rhat", "ess_per_second", "ess_per_gradient_eval",
         "ess_per_potential_eval", "ess_per_score_quadrature_eval"]
    ].round(5))"""


def cell_csv(extra_manifest: str = "") -> str:
    return f'''ts_path = os.path.join(RESULTS, "metrics_timeseries.csv")
write_timeseries_csv(rows, ts_path)
summary_metrics = MAIN_METRICS + ["nonfinite_frac", "basin_map_outside_mass"]
summary = write_summary_csv(rows, RUN_METHODS, cfg.seeds, summary_metrics,
                            method_info, floors, os.path.join(RESULTS, "summary.csv"))

write_manifest(
    os.path.join(RESULTS, "resolved_config.json"),
    schema_version=1,
    experiment=EXPERIMENT,
    run_id=RUN_ID,
    source_config_file=os.path.abspath(source_config_path),
    resolved_preflight_config_file=os.path.abspath(preflight_config_path),
    resolved_preflight_config_sha256=_preflight_config_sha256,
    certificate_result_file=os.path.abspath(certificate_result_path),
    results_directory=RESULTS,
    resolved_dt=dt_final,
    resolved_n_steps=n_steps,
    resolved_quadrature=CHOSEN_QUAD,
    resolved_pt_ladder=dict(
        betas=[float(v) for v in pt_betas.detach().cpu()],
        tuning=ladder_info),
    selected_methods=list(RUN_METHODS),
    requested_trace_settings=TRACE_REQUEST,
    stationarity_protocol=stationarity_manifest["protocol"],
    failure_thresholds=FAIL_THRESHOLDS,
    observed_failure_diagnostics=observed_failure_diagnostics,
)

# plot policy recomputed here (not inherited from the figures cell) so the
# manifest is self-contained even if cells are run out of order; it makes
# the immutable timestamped experiment directory sufficient to regenerate
# every figure with no GPU.
# BOTH LSC arms are recorded, matching CELL_FIGURES. The plotting workflow
# reads this manifest block, so a stale list here would silently drop an arm.
_plot_raw = "CP" if "CP" in RUN_METHODS else (
    "CP-RA" if "CP-RA" in RUN_METHODS else None)
_arm_atoms = {{"mb3well_10d": 4, "coupled_phi4": 8}}
def _realised_label(arm):
    A = _arm_atoms.get(EXPERIMENT)
    if arm == "LSC-CP-RA":
        return "LSC-CP-RA (1)" if A else "LSC-CP-RA"
    if arm == "LSC-CP-MA":
        return f"LSC-CP-RA ({{A}})" if A else "LSC-CP-MA"
    return arm
_plot_methods = [m for m in ("ULA", "MALA", "FLA", "BAOAB", "PT")
                 if m in RUN_METHODS]
_plot_labels = {{}}
if _plot_raw is not None:
    _plot_methods.append(_plot_raw)
    _plot_labels[_plot_raw] = "Raw-CP"
if "LSC-CP" in RUN_METHODS:
    _plot_methods.append("LSC-CP")
    _plot_labels["LSC-CP"] = "LSC-CP"
for _ra in ("LSC-CP-RA", "LSC-CP-MA"):
    if _ra in RUN_METHODS:
        _plot_methods.append(_ra)
        _plot_labels[_ra] = _realised_label(_ra)
manifest = dict(
    experiment=EXPERIMENT,
    run_id=RUN_ID,
    results_directory=RESULTS,
    config=dict(d=cfg.d, N=cfg.n_particles, T=cfg.T, dt0=cfg.dt, dt=dt_final,
                beta=cfg.beta, eps=cfg.eps, lam=cfg.lam, seeds=list(cfg.seeds),
                n_checkpoints=cfg.n_checkpoints, warmup_steps=C.N_WARMUP_STEPS,
                batched_seeds=True),
    emc_target=float(emc_target),
    p_star=[float(v) for v in exp.p_star.cpu()],
    plot=dict(methods=_plot_methods, label_overrides=_plot_labels),
    quadrature=dict(chosen=CHOSEN_QUAD, table=quad_table),
    dt_refinement=[{{k: (str(v) if isinstance(v, tuple) else v) for k, v in row.items()}}
                   for row in dt_table],
    pt_ladder={{k: v for k, v in ladder_info.items()}},
    certificate=cert_report,
    provenance=dict(
        source_config_file=os.path.abspath(source_config_path),
        resolved_preflight_config_file=os.path.abspath(preflight_config_path),
        resolved_preflight_config_sha256=_preflight_config_sha256,
        certificate_result_file=os.path.abspath(certificate_result_path)),
    bias_floors=floors,
    reference={{k: v for k, v in aux.items() if k != "ref_x"}},
    stationarity=stationarity_manifest,
    failure_thresholds=FAIL_THRESHOLDS,
    observed_failure_diagnostics=observed_failure_diagnostics,
    barrier_verification=barrier_report,
    method_info={{m: {{k: v for k, v in mi.items() if isinstance(v, (int, float))}}
                 for m, mi in method_info.items()}},
    hardware=hardware_manifest(),
    {extra_manifest}
)
write_manifest(os.path.join(RESULTS, "manifest.json"), **manifest)
print("wrote", ts_path)

# Mirror the evidence and figures into the repository tree. The immutable run
# directory stays the source of truth; this keeps root results/ and figures/
# on the latest run instead of silently going stale,
# which previously required copying every artifact across by hand.
_mirror = mirror_into_repo(RESULTS, EXPERIMENT, os.path.abspath(".."))
print(f"mirrored -> {{_mirror['results_dir']}} ({{len(_mirror['files'])}} files), "
      f"{{_mirror['figures_dir']}} ({{_mirror['figure_files']}} figures)")

from IPython.display import display
display(pd.read_csv(os.path.join(RESULTS, "summary.csv")).round(5))'''


# ======================================================================
# E1 double well
# ======================================================================
def build_e1_nb() -> nbf.NotebookNode:
    cells = [
        md(r"""# E1 — 1D double well

**Target.** $\pi(x)\propto e^{-\beta V(x)}$ at $\beta=8$ ($\varepsilon=1/\beta$) with $V(x)=(x^2-1)^2$: minima $\pm1$, saddle $0$, $\beta\Delta V=8$, Kramers time $\tau=\tfrac{2\pi}{\sqrt{32}}e^{8}\approx3.3\times10^3$ — local samplers started in the left well essentially never equilibrate within $T=100$. Seven methods (ULA, MALA, FLA, BAOAB, PT, Raw-CP, LSC-CP) share one tamed drift map, one $\Delta t$, one metric cadence and per-seed initial conditions $x_0\sim\mathcal N(-1,0.05^2)$."""),
        code(cell_setup("double_well", "build_e1", "exp = build_e1(device=DEV)")),
        code('''# model asserts + barrier verification (committed arrival in the right-well
# core x > 0.7; exposure/events exponential MLE and KM restricted mean)
V = lambda x: (x**2 - 1.0)**2
assert V(1.0) == 0.0 and V(-1.0) == 0.0 and V(0.0) == 1.0
g = torch.Generator(device=DEV); g.manual_seed(0)
barrier_report = ula_first_passage(exp.pot, exp.box, exp.init_fn(cfg.n_particles, g),
                                   exp.exit_committed, cfg.dt, int(cfg.T/cfg.dt),
                                   cfg.eps, g)
barrier_report["kramers_tau"] = exp.kramers_tau
print(f"ULA committed events {barrier_report['event_count']}/"
      f"{barrier_report['n_particles']}; exponential waiting-time MLE "
      f"{barrier_report['exponential_waiting_time_mle']:.0f}, "
      f"KM RMST@T {barrier_report['kaplan_meier_rmst_at_horizon']:.0f}, "
      f"Kramers time {exp.kramers_tau:.0f}")'''),
        md(r"""## Jump law and Lévy score

Two-atom symmetric shell: $r = \pm2 + \rho\,u$, $\rho\sim\mathrm{Unif}(-h,h)$, $h=0.2$, $w=(\tfrac12,\tfrac12)$, $\lambda=1$ — a $\pm2$ jump maps minimum to minimum. The stationary correction
$$S_{\nu,\beta}(x) = -\lambda\!\int\!\nu(dr)\,r\!\int_0^1\! e^{-\beta[V(x-\theta r)-V(x)]}d\theta$$
makes $\pi$ invariant for the jump diffusion *exactly at generator level, for any $\nu$*. It is approximated with Gauss–Legendre probability weights for expectations under the same declared continuous $\nu$ used by the jump sampler; finite quadrature is not literally that continuous measure, and the refinement/certificate gates control its error. We use **log-space accumulation**: the per-direction integrals span hundreds of orders of magnitude at $\beta=8$, so we assemble $\log I$ by log-sum-exp, extract the max exponent $M(x)$, form the $O(1)$ direction vector $v(x)$, and return $S = -\lambda e^{\min(M,600)}v$ — the drift is tamed, so only the direction matters when $\|S\|$ is astronomical. The weak stationarity residual $\mathcal R(\varphi)$ (drift term assembled in log space; domain one jump length beyond the support) certifies the correction; a deliberately tight box fails it."""),
        code('''DEFAULT_QUAD = dict(q_theta=C.Q_THETA, q_rho=C.Q_RHO)
phis = make_phi_family(1, [0.0], 1.0, DEV)

def cert_e1(q_theta, q_rho, lo=-5.2, hi=5.2):
    score = exp.make_score(q_theta=q_theta, q_rho=q_rho)
    shifts, logw = exp.law.quadrature_shifts(64)   # fine continuous-nu J side
    return certificate_grid(exp.pot, score, shifts, logw, cfg.lam, cfg.beta,
                            phis, [lo], [hi], n_panels=120, nodes_per_panel=8)

cert_report = cert_e1(**DEFAULT_QUAD)
print(f"max R at default orders = {cert_report['max_residual']:.3e}")
if cert_report["max_residual"] >= CERTIFICATE_TOLERANCE:
    print("default quadrature is not certified; refinement must select a passing order")
tight = cert_e1(**DEFAULT_QUAD, lo=-1.3, hi=1.3)
print(f"deliberately TIGHT box: max R = {tight['max_residual']:.3e}")'''),
        code(CELL_LADDER),
        code(CELL_REFERENCE),
        code('''# quadrature refinement: smallest (Q_theta, Q_rho) with R < 1e-6 and
# terminal LSC-CP metrics converged against the finest setting
def run_terminal_lsc(**quad):
    f = make_sampler_factory(exp, cfg.dt, pt_betas, score_kwargs=quad)
    n_ = int(round(cfg.T / cfg.dt))
    r_, _ = run_one("LSC-CP", 0, f, n_, n_, cfg.dt, metrics_fn, exp.pot, quiet=True)
    return {k: r_[-1][k] for k in ("W2", "TV", "TV_density", "MMD", "EMC")}

settings = [dict(q_theta=qt, q_rho=qr) for qt in (8, 16, 32) for qr in (4, 8, 16)]
CHOSEN_QUAD, quad_table = quadrature_refinement(
    settings, run_terminal_lsc, lambda **s: cert_e1(**s)["max_residual"], floors)
print("chosen quadrature:", CHOSEN_QUAD)
cert_report = cert_e1(**CHOSEN_QUAD)
certificate_result = persist_certificate_result(cert_report, CHOSEN_QUAD)
print("final certificate result:", certificate_result)
assert certificate_result["passed"], certificate_result
display(pd.DataFrame(quad_table).round(6))'''),
        code(cell_dt_production('["W2", "TV", "TV_density", "MMD", "EMC"]')),
        code(CELL_STATIONARITY),
        code(CELL_FIGURES),
        code('''# terminal-sample CDF of every method vs the true CDF (single plot;
# all 5 seed blocks pooled -> 20k points per method)
from src.plotting import cdf_comparison
ref = exp.extras["ref"]
samples = {m: method_info[m]["final_positions_all"].reshape(-1).cpu().numpy()
           for m in RUN_METHODS}
cdf_fig = cdf_comparison(samples, ref.x.cpu().numpy(), ref.cdf.cpu().numpy(),
                         os.path.join(FIGURES, EXPERIMENT + "_cdf"))
print("saved:", os.path.join(FIGURES, EXPERIMENT + "_cdf") + ".{png,pdf}")'''),
        md(r"""## Raw-CP forensic — does raw CP converge to the *predicted* biased law?

Raw CP does not target $\pi$ (no score correction); the paper premise is that it converges to a *biased* stationary law $\rho_\infty^{\rm raw}$ solving the linear stationary equation $0=\varepsilon\rho''+\partial_x[V'\rho]+\lambda\sum_{a,q}w_{a,q}[\rho(x-r_{a,q})-\rho]$. We solve $\rho_\infty^{\rm raw}$ exactly on a fine grid (`src/stationary.py`; Chang–Cooper conservative flux, $\lambda\to0$ recovers $e^{-\beta V}$ to machine precision) and overlay its CDF on the empirical raw-CP CDF. If $W_1(\text{empirical},\rho_\infty^{\rm raw})\ll W_1(\text{empirical},\pi)$ the raw-CP code is correct and its bias matches theory — and the LSC-CP score removes *exactly* this bias."""),
        code('''# solve the predicted raw-CP stationary law and compare CDFs
from src.stationary import doublewell_rawcp_forensic, w1_from_samples
import matplotlib.pyplot as _plt
fen = doublewell_rawcp_forensic(exp.law, cfg.beta, cfg.lam, lo=-5.2, hi=5.2)
xg, cdf_pred, cdf_gibbs = fen["x"], fen["cdf_pred"], fen["cdf_gibbs"]
# empirical raw-CP terminal ensemble (all seeds pooled). NB: production runs
# use the same [-5.2,5.2] box as the stationary solve; boundary mass is
# therefore represented rather than silently clipped away.
emp_cp = method_info["CP"]["final_positions_all"].reshape(-1).cpu().numpy()
w1_ep = w1_from_samples(emp_cp, xg, cdf_pred)
w1_et = w1_from_samples(emp_cp, xg, cdf_gibbs)
print(f"W1(empirical raw-CP, predicted rho_inf^raw) = {w1_ep:.4f}")
print(f"W1(empirical raw-CP, true Gibbs pi)         = {w1_et:.4f}")
print(f"ratio pred/true = {w1_ep/w1_et:.3f}  "
      f"(<1 => raw CP tracks the predicted biased law, not pi)")
_ts = np.sort(emp_cp); _F = np.arange(1, _ts.size + 1) / _ts.size
_fig, _ax = _plt.subplots(figsize=(4.8, 3.4))
_ax.plot(xg, cdf_gibbs, color="#888888", lw=2.4, label=r"true $\\pi\\propto e^{-\\beta V}$")
_ax.plot(xg, cdf_pred, color="#D55E00", lw=1.8, ls="--",
         label=r"predicted raw-CP $\\rho_\\infty^{\\rm raw}$")
_ax.step(_ts, _F, where="post", color="#000000", lw=1.1, label="empirical raw-CP")
_ax.set_xlim(-3, 3); _ax.set_xlabel(r"$x$"); _ax.set_ylabel("CDF")
_ax.legend(fontsize=8, loc="lower right", frameon=False)
_fig.tight_layout()
_base = os.path.join(FIGURES, "rawcp_stationary_forensic")
_fig.savefig(_base + ".png", dpi=600, bbox_inches="tight")
_fig.savefig(_base + ".pdf", bbox_inches="tight")
print("saved:", _base + ".{png,pdf}")'''),
        code(cell_csv()),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


# ======================================================================
# E2 MoG40
# ======================================================================
def build_e2_nb() -> nbf.NotebookNode:
    cells = [
        md(r"""# E2 — MoG40 (2D)

**Target.** $V(x) = -\tfrac1\beta\log\sum_{k=1}^{40} e^{-\|x-\mu_k\|^2/2}$, so $\pi\propto e^{-\beta V}$ is an equal-weight mixture of $\mathcal N(\mu_k, I_2)$ (the $1/\beta$ prefactor is what makes the barriers right: $\beta\Delta V = d^2/8-\log 2$ between modes at distance $d$). Modes $\mu_k\sim\mathrm{Unif}([-40,40]^2)$, frozen from `default_rng(0)`. All particles start at $\mu_0+0.5\,\xi$; partition = nearest-mode Voronoi, $p^\star_k = 1/40$."""),
        code(cell_setup("mog40", "build_e2", "exp = build_e2(device=DEV)")),
        code('''np.savetxt(os.path.join(RESULTS, "modes.csv"), exp.pot.mu.cpu().numpy(),
           delimiter=",", header="mu_x,mu_y", comments="")
dists = torch.cdist(exp.pot.mu, exp.pot.mu); dists.fill_diagonal_(float("inf"))
nn = dists.min(dim=1).values.cpu().numpy()
print(f"NN distances: min {nn.min():.2f} median {np.median(nn):.2f} max {nn.max():.2f}"
      "  -> jump radii Unif[4, 15] chosen from this histogram alone")
g = torch.Generator(device=DEV); g.manual_seed(0)
barrier_report = ula_first_passage(exp.pot, exp.box, exp.init_fn(cfg.n_particles, g),
                                   exp.exit_committed, cfg.dt, int(cfg.T/cfg.dt), cfg.eps, g)
barrier_report["kramers_tau_mode0"] = exp.kramers_tau
print(f"ULA committed events {barrier_report['event_count']}/"
      f"{barrier_report['n_particles']}; exponential waiting-time MLE "
      f"{barrier_report['exponential_waiting_time_mle']:.0f}, "
      f"KM RMST@T {barrier_report['kaplan_meier_rmst_at_horizon']:.0f}, "
      f"mode-0 Kramers time {barrier_report['kramers_tau_mode0']:.0f}")'''),
        md(r"""## Jump law and Lévy score (closed form)

Deliberately generic annulus law — $r=\rho u_\phi$, $\rho\sim\mathrm{Unif}[4,15]$, $\phi\sim\mathrm{Unif}[0,2\pi)$ — **neither PT nor LSC-CP receives mode locations**. For the Gaussian mixture the $\theta$ and $\rho$ integrals of the score are analytic: with $m_{k\ell}=u_\ell\cdot(x-\mu_k)$,
$$S(x) = -\frac{\lambda}{M_\phi(b-a)}\sum_\ell u_\ell \sum_k e^{\log\omega_k + m_{k\ell}^2/2}\,\sqrt{\tfrac\pi2}\,\mathcal B(m_{k\ell}),$$
with $\mathcal B(m) = F(b\!-\!m)-F(a\!-\!m)+(b\!-\!a)\,\mathrm{erf}(m/\sqrt2)>0$ and $F(z)=z\,\mathrm{erf}(z/\sqrt2)+\sqrt{2/\pi}e^{-z^2/2}$, evaluated by outer-regime branches in which the $O(m)$ parts cancel *analytically* (the naive form has 100% error at $m=30$; the branched form is validated against 3000-digit mpmath and a brute-force 3-D quadrature at $10^{-8}$). Only the periodic $\phi$-trapezoid ($M_\phi$ directions) is numerical — **zero potential evaluations**."""),
        code('''DEFAULT_QUAD = dict(m_phi=C.M_PHI)
phis = make_phi_family(2, [0.0, 0.0], 30.0, DEV)

def cert_e2(m_phi):
    score = exp.make_score(m_phi=m_phi)
    shifts, logw = exp.law.quadrature_shifts(16, 64)
    return certificate_grid(exp.pot, score, shifts, logw, cfg.lam, cfg.beta,
                            phis, [-60.0, -60.0], [60.0, 60.0],
                            n_panels=120, nodes_per_panel=6, chunk=8192)

cert_report = cert_e2(**DEFAULT_QUAD)
print(f"max R at default orders = {cert_report['max_residual']:.3e}")
if cert_report["max_residual"] >= CERTIFICATE_TOLERANCE:
    print("default quadrature is not certified; refinement must select a passing order")'''),
        code(CELL_LADDER),
        code(CELL_REFERENCE + '''
assert aux["bandwidth"] > 3.0   # bandwidth reflects mode spacing, not width'''),
        code('''def run_terminal_lsc(**quad):
    f = make_sampler_factory(exp, cfg.dt, pt_betas, score_kwargs=quad)
    n_ = int(round(cfg.T / cfg.dt))
    r_, _ = run_one("LSC-CP", 0, f, n_, n_, cfg.dt, metrics_fn, exp.pot, quiet=True)
    return {k: r_[-1][k] for k in ("W2", "TV", "MMD", "EMC")}

settings = [dict(m_phi=m) for m in (16, 32, 64)]
CHOSEN_QUAD, quad_table = quadrature_refinement(
    settings, run_terminal_lsc, lambda **s: cert_e2(**s)["max_residual"], floors)
print("chosen quadrature:", CHOSEN_QUAD)
cert_report = cert_e2(**CHOSEN_QUAD)
certificate_result = persist_certificate_result(cert_report, CHOSEN_QUAD)
print("final certificate result:", certificate_result)
assert certificate_result["passed"], certificate_result
display(pd.DataFrame(quad_table).round(6))'''),
        code(cell_dt_production('["W2", "TV", "MMD", "EMC"]')),
        code(CELL_STATIONARITY),
        code(CELL_FIGURES),
        code('''# terminal exact-W2 spot check (Hungarian, 500-point subsample, 2D only)
gen_h = torch.Generator(device=DEV); gen_h.manual_seed(202)
ref_sub = exp.ref_sample(2500, gen_h)
from src.metrics import hungarian_w2
hungarian = {m: hungarian_w2(method_info[m]["final_positions_seed0"], ref_sub, m=500)
             for m in RUN_METHODS}
print("Hungarian W2:", {k: round(v, 3) for k, v in hungarian.items()})
''' + cell_csv("hungarian_w2_terminal=hungarian,")),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


# ======================================================================
# E3 4-well modified Mueller-Brown 10D
# ======================================================================
CELL_E3_QUAD = '''# the run's LSC estimator (exact LSC-CP, or a practical RA/MA variant)
_LSC = next((m for m in RUN_METHODS if m.startswith("LSC-CP")), "LSC-CP")
def run_terminal_lsc(**quad):
    f = make_sampler_factory(exp, cfg.dt, pt_betas, score_kwargs=quad)
    n_ = int(round(cfg.T / cfg.dt))
    r_, _ = run_one(_LSC, 0, f, n_, n_, cfg.dt, metrics_fn, exp.pot, quiet=True)
    return {k: r_[-1][k] for k in ("W2", "TV", "MMD", "EMC", "W2_10d")}

settings = [dict(q_theta=qt, q_rho=qr) for qt in (8, 16, 32) for qr in (4, 8, 16)]
CHOSEN_QUAD, quad_table = quadrature_refinement(
    settings, run_terminal_lsc, lambda **s: cert_e3(**s)["max_residual"], floors)
print("chosen quadrature:", CHOSEN_QUAD)
cert_report = cert_e3(**CHOSEN_QUAD)
certificate_result = persist_certificate_result(cert_report, CHOSEN_QUAD)
print("final certificate result:", certificate_result)
assert certificate_result["passed"], certificate_result
display(pd.DataFrame(quad_table).round(6))'''

MD_E3_TITLE_MB3 = r"""# E3 — depth-retuned 3-well Müller–Brown (10D)

**Target.** Standard Müller–Brown functional form with depth parameters retuned to equal-depth wells, embedded in 10D ($U(z)=V_3(z_1,z_2)+\|z_{3:10}\|^2/(2\cdot0.4^2)$, $x=zB^\top$), at $\beta=24$. The standard MB is multimodal OR metastable but never both (its depth gap $0.659$ exceeds every barrier except A's own exit $0.401$); retuning $(D_1,D_3)$ to equal depths ($V=-0.7957$) decouples depth from barrier and makes $\beta$ a free dial. At $\beta=24$ the target is genuinely **trimodal AND metastable with two timescales**: $A\leftrightarrow B$ slow ($\beta b=11.1$, local methods cannot cross in $T=200$), $B\leftrightarrow C$ moderate ($\beta b=4.0$). Masses $p^\star\approx(0.32,0.42,0.26)$. Runs start in well $C$; only nonlocal relay jumps populate the far well $A$. Metrics in latent 2D $z_{1:2}$ (full-10D sliced $W_2$ in the CSV)."""

CELL_E3_ASSERTS_MB3 = '''from src.potentials import MB3_CRITICAL, mb3_2d, mb3_2d_grad, newton_refine
for key, (z_tab, V_tab) in MB3_CRITICAL.items():
    z = newton_refine(mb3_2d_grad, torch.tensor(z_tab, device=DEV))
    Vv = mb3_2d(z.unsqueeze(0))[0].item()
    assert abs(Vv - V_tab) < 5e-4, (key, Vv)
    print(f"{key:5s}: ({z[0].item():+.4f}, {z[1].item():+.4f})  V = {Vv:.4f}")
b_AB, b_BC = exp.extras["b_AB"], exp.extras["b_BC"]
print(f"p_star (A,B,C): {np.round(exp.p_star.cpu().numpy(), 4)} | "
      f"beta*b(A<->B) = {cfg.beta*b_AB:.1f} (slow) | "
      f"beta*b(B<->C) = {cfg.beta*b_BC:.1f} (moderate)")

# barrier structure: ULA from C reaches B (B<->C moderate) but NOT A (A<->B slow)
g = torch.Generator(device=DEV); g.manual_seed(0)
rep_A = ula_first_passage(exp.pot, exp.box, exp.init_fn(cfg.n_particles, g),
                          exp.exit_committed, cfg.dt, int(cfg.T/cfg.dt), cfg.eps, g)
g2 = torch.Generator(device=DEV); g2.manual_seed(0)
rep_B = ula_first_passage(exp.pot, exp.box, exp.init_fn(cfg.n_particles, g2),
                          exp.extras["exit_to_B"], cfg.dt, int(cfg.T/cfg.dt), cfg.eps, g2)
# Kramers estimate of the expected number of A-crossings over the N*T budget:
# a handful is fine (rare-event fluctuation); what matters is that ULA does NOT
# populate A at its equilibrium mass p*_A -- committed-exit fraction << p*_A.
_exp_cross = rep_A['n_particles'] * (1.0 - math.exp(-cfg.T / max(exp.kramers_tau, 1e-30)))
print(f"ULA T={cfg.T}: committed C->A (far well) = {rep_A['n_exits']}/{rep_A['n_particles']} "
      f"(frac {rep_A['exit_fraction']:.4f}; Kramers expects ~{_exp_cross:.1f}, "
      f"<< equilibrium p*_A={float(exp.p_star[0]):.2f}) | "
      f"C->B (reachable) = {rep_B['n_exits']}/{rep_B['n_particles']} (expect > 0)")
assert rep_A['exit_fraction'] < 0.2 * float(exp.p_star[0]), \
    "local ULA must stay far from equilibrating the far well A"
assert rep_B['n_exits'] > 0, "B<->C must be reachable (two-timescale structure)"
# barrier_report is consumed by the shared manifest cell (far-well C->A passage)
barrier_report = {**rep_A, "exit_to_B_fraction": rep_B["exit_fraction"],
                  "kramers_tau": exp.kramers_tau}
print(f"Kramers tau(A<->B) = {exp.kramers_tau:.2e} time units >> T={cfg.T}")'''

MD_E3_TARGET_MB3 = r"""## Target density: explicit form and visualization

The target is $\pi(x)\propto e^{-\beta U(x)}$ on $\mathbb R^{10}$ at $\beta=24$. In latent coordinates $z = xB^{-\top}$ ($x = zB^\top$, $B = Q\,\mathrm{diag}(0.75,\dots,1.45)$, $Q$ a frozen Haar rotation from `default_rng(12345)`):

$$U(z) = V_3(z_1,z_2) + \frac{\|z_{3:10}\|^2}{2\sigma_{\mathrm{aux}}^2},\qquad \sigma_{\mathrm{aux}}=0.4,$$

$$V_3(z_1,z_2) = \sum_{k=1}^{4} D_k\, \exp\!\big(a_k(z_1-\bar x_k)^2 + b_k(z_1-\bar x_k)(z_2-\bar y_k) + c_k(z_2-\bar y_k)^2\big),$$

the **standard Müller–Brown functional form** with depths $(D_1,D_3)$ retuned so the three deep wells are equal-depth:

| $k$ | $D_k$ | $a_k$ | $b_k$ | $c_k$ | $\bar x_k$ | $\bar y_k$ |
|---|---|---|---|---|---|---|
| 1 | $-1.6607$ | $-1$   | $0$  | $-10$  | $1$    | $0$    |
| 2 | $-1.0$    | $-1$   | $0$  | $-10$  | $0$    | $0.5$  |
| 3 | $-1.0218$ | $-6.5$ | $11$ | $-6.5$ | $-0.5$ | $1.5$  |
| 4 | $+0.15$   | $0.7$  | $0.6$| $0.7$  | $-1$   | $1$    |

**Why retuned.** The standard MB's depth gap $V_B-V_A=0.659$ exceeds every barrier but A's own exit ($0.401$): any $\beta$ equalising the masses ($\beta\lesssim4.6$) kills metastability, any $\beta$ creating metastability ($\beta\gtrsim25$) makes the mass ratio $e^{-16}$ (effectively unimodal). Retuning to equal-depth wells decouples the two exponentials, so $\beta$ becomes a free dial. Since $x=zB^\top$ is invertible linear, $\pi$ factorises exactly: $\pi(z)=\frac{e^{-\beta V_3}}{Z_2}\prod_{j=3}^{10}\mathcal N(z_j;0,\sigma_{\mathrm{aux}}^2)$ — all multimodal structure is in the 2D marginal below."""

CELL_E3_TARGET_VIZ_MB3 = r'''# Target visualization (self-contained, CPU-safe; touches no run state).
import os, sys, math
sys.path.insert(0, os.path.abspath(".."))
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.potentials import mb3_2d, MB3_CRITICAL

beta_viz = float(cfg.beta)
zg1 = torch.linspace(-1.3, 1.3, 701, dtype=torch.float64)
zg2 = torch.linspace(-0.6, 1.9, 701, dtype=torch.float64)
ZZ = torch.stack(torch.meshgrid(zg1, zg2, indexing="ij"), dim=-1)
Vg = mb3_2d(ZZ)
cell_area = float((zg1[1] - zg1[0]) * (zg2[1] - zg2[0]))
logp2 = -beta_viz * Vg
logp2 = logp2 - (torch.logsumexp(logp2.reshape(-1), 0) + math.log(cell_area))

wells = ["A", "B", "C"]
zw = torch.tensor([MB3_CRITICAL[k][0] for k in wells], dtype=torch.float64)
lab_g = torch.cdist(ZZ.reshape(-1, 2), zw).argmin(1)
p2_flat = torch.exp(logp2.reshape(-1))
mass = torch.tensor([float(p2_flat[lab_g == i].sum()) for i in range(3)])
mass = mass / mass.sum()
print("latent-2D grid well masses (A,B,C):", np.round(mass.numpy(), 4),
      " (cf. p_star printed above)")

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
lv = np.linspace(-0.85, 0.2, 22)
cf0 = axes[0].contourf(zg1, zg2, Vg.T, levels=lv, cmap="viridis")
axes[0].contour(zg1, zg2, Vg.T, levels=lv, colors="k", linewidths=0.25, alpha=0.4)
for k in wells:
    (zx, zy), Vk = MB3_CRITICAL[k]
    axes[0].plot(zx, zy, "o", ms=5, mfc="w", mec="k")
    axes[0].annotate(f"{k}  V={Vk:.3f}", (zx, zy), textcoords="offset points",
                     xytext=(7, 5), fontsize=8, color="w")
for s in ("S_AB", "S_BC"):
    (sx, sy), Vs = MB3_CRITICAL[s]
    axes[0].plot(sx, sy, "x", ms=7, color="r")
    axes[0].annotate(f"{s}  V={Vs:.3f}", (sx, sy), textcoords="offset points",
                     xytext=(-7, -3), ha="right", fontsize=8, color="r")
axes[0].set(title=r"latent potential $V_3(z_1, z_2)$ (depth-retuned)",
            xlabel="$z_1$", ylabel="$z_2$")
fig.colorbar(cf0, ax=axes[0], label="$V_3$")

l10 = (logp2 / math.log(10.0)).numpy()
lv1 = np.linspace(l10.max() - 8.0, l10.max(), 33)
cf1 = axes[1].contourf(zg1, zg2, np.maximum(l10.T, lv1[0]), levels=lv1, cmap="magma")
for k, m in zip(wells, mass):
    (zx, zy), _ = MB3_CRITICAL[k]
    axes[1].annotate(f"{k}: {float(m):.3f}", (zx, zy), textcoords="offset points",
                     xytext=(7, 5), fontsize=8, color="w")
axes[1].set(title=r"2D marginal $\log_{10}\pi_2$ at $\beta=24$ (labels: basin masses)",
            xlabel="$z_1$", ylabel="$z_2$")
fig.colorbar(cf1, ax=axes[1], label=r"$\log_{10}\pi_2$")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIGURES, "mb3well_10d_target." + ext), dpi=200,
                bbox_inches="tight")
plt.show()'''

MD_E3_JUMP_MB3 = r"""## Jump law and Lévy score

**Relay design.** Four atoms $\{\pm r_{BA},\ \pm r_{BC}\}$ through the middle hub $B$ ($r_{BA}=z_A-z_B$, $r_{BC}=z_C-z_B$ in latent coords, then $r=(\Delta z,0_8)B^\top$), $w_a=\tfrac14$, shell $h=0.1\min\|r_a\|$, $\lambda=1$; the CP pair's drift step is capped at $2h$ (the $B$–$A$ chord overshoots $S_{AB}$, so the detailed-balance return flux is integrated with small in-tube steps). **No direct $A$–$C$ atom** — the two-hop relay through $B$ covers it. The atom set is FIXED and every atom fires from every state (never gated on the current basin — a state-dependent selection law would break the RA invariance argument). Score: generic shell with log-space accumulation; certificate on the exact latent-2D reduction (per-atom residuals cover the RA estimator)."""

CELL_E3_CERT_MB3 = '''from src.potentials import MB3Latent2D
from src.jumps import ShellJumpLaw
from src.score import ShellScore
potr = MB3Latent2D()
dz = exp.extras["atoms_z"][:, :2]
h_z = exp.extras["h"] * dz.norm(dim=1) / exp.law.atoms.norm(dim=1)
law_r = ShellJumpLaw(dz, exp.law.weights.clone(), h_z)
clo, chi = exp.extras["cert_lo"], exp.extras["cert_hi"]   # generous box (>= 1 jump)
print("relay atoms (latent):", np.round(dz.cpu().numpy(), 3).tolist())
print("h =", round(exp.extras["h"], 4), " drift cap (CP pair) =",
      round(exp.cp_drift_cap, 4))
DEFAULT_QUAD = dict(q_theta=C.Q_THETA, q_rho=C.Q_RHO)
phis = make_phi_family(2, [0.0, 0.5], 0.8, DEV)

def cert_e3(q_theta, q_rho):
    score = ShellScore(potr, law_r, cfg.lam, cfg.beta, q_theta, q_rho)
    shifts, logw = law_r.quadrature_shifts(64)
    return certificate_grid(potr, score, shifts, logw, cfg.lam, cfg.beta, phis,
                            list(clo), list(chi),
                            n_panels=200, nodes_per_panel=10, chunk=8192)

cert_report = cert_e3(**DEFAULT_QUAD)
print(f"max R at default orders = {cert_report['max_residual']:.3e}")
if cert_report["max_residual"] >= CERTIFICATE_TOLERANCE:
    print("default quadrature is not certified; refinement must select a passing order")'''

CELL_E3_PISTART_MB3 = '''# pi-start hold test: initialise at the reference and measure any stationary
# drift of the discretised LSC-CP chain
K = exp.p_star.shape[0]
g_h = torch.Generator(device=DEV); g_h.manual_seed(777)
x_pi = exp.ref_sample(4000, g_h)
from src.metrics import occupancy as _occ
bf = make_batched_factory(exp, dt_final, pt_betas, (0,), n_particles=4000,
                          score_kwargs=CHOSEN_QUAD)
s_h = bf(_LSC)                       # the run's LSC estimator (exact / RA / MA)
s_h.x = x_pi.clone()
p0 = _occ(exp.labels_fn(s_h.positions()), K)
for _i in range(int(round(100.0 / dt_final))):
    s_h.step()
p1 = _occ(exp.labels_fn(s_h.positions()), K)
hold_tv = 0.5 * float((p1 - exp.p_star).abs().sum())
pi_start_hold = {"init": [round(float(v), 4) for v in p0],
                 "after_T100": [round(float(v), 4) for v in p1],
                 "TV_vs_pstar": round(hold_tv, 4)}
print("pi-start hold:", pi_start_hold)'''


def build_e3_nb() -> nbf.NotebookNode:
    cells = [
        md(MD_E3_TITLE_MB3),
        code(cell_setup("mb3well_10d", "build_e3",
                        'exp = build_e3(device=DEV, basin_cache=os.path.join(CACHE, "basin_map_v2.npz"))')),
        code(CELL_E3_ASSERTS_MB3),
        md(MD_E3_TARGET_MB3),
        code(CELL_E3_TARGET_VIZ_MB3),
        md(MD_E3_JUMP_MB3),
        code(CELL_E3_CERT_MB3),
        code(CELL_LADDER),
        code(CELL_REFERENCE),
        code(CELL_E3_QUAD),
        code(cell_dt_production('["W2", "TV", "MMD", "EMC", "W2_10d"]')),
        code(CELL_STATIONARITY),
        code(CELL_E3_PISTART_MB3),
        code(CELL_FIGURES),
        code(cell_csv("pi_start_hold=pi_start_hold,")),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb


# ======================================================================
# E4 coupled phi4
# ======================================================================
MD_E4_TARGET = r"""## Target density: explicit form and visualization

The state is $q=(q_1,\dots,q_{12})\in\mathbb R^{24}$ with sites $q_i=(x_i,y_i)\in\mathbb R^2$, periodic ($q_{13}\equiv q_1$). With $\delta=1/12$, $\kappa=2.5$, $\beta=8$ the target density is, explicitly,

$$\pi(q)\;\propto\;\exp\Big[-\beta\Big(\underbrace{\tfrac{\kappa}{2\delta}}_{=\,15}\sum_{i=1}^{12}\|q_{i+1}-q_i\|^2\;+\;\underbrace{\tfrac{1}{12}}_{=\,\delta}\sum_{i=1}^{12}W(q_i)\Big)\Big],$$
$$W(x,y)=(x^2-1)^2+(y^2-1)^2-0.0125\,xy+0.0075\,x+0.015\,y$$

(un-normalised; $Z$ has no closed form). Structure:

- **Four phases.** $W$ has four minima at $v\approx(\pm1,\pm1)$ (tilt-shifted in the 3rd decimal) with $W$-values $-0.0351\,(--)$, $+0.0050\,(+-)$, $+0.0100\,(++)$, $+0.0200\,(-+)$: inter-phase asymmetries $\beta\Delta W\le0.44$, escape barriers $\beta b\approx7.8$–$8.2$. The global minima of $V$ are the four coherent fields $\mathbf 1\otimes v$.
- **Stiff coupling.** A bond deviation costs $\beta\kappa/(2\delta)=120$ per unit squared distance, so $\pi$ concentrates in four narrow tubes around the coherent fields; the cheapest non-coherent excursion (a kink pair, cost $5.96$) is far above the coherent barrier $\approx1.0$, so phase changes proceed coherently.
- **Slice vs marginal.** On the homogeneous slice $q_i\equiv v$ the coupling term vanishes and $V(\mathbf 1\otimes v)=W(v)$ exactly, so the right panel below ($e^{-\beta W}/Z_W$) is the exact restriction of $\pi$ to the coherent 2-plane. It is a slice, **not** a marginal: the phase masses $p^\star=(0.325,\,0.211,\,0.237,\,0.227)$ are 24D basin integrals estimated by direct weighted SNIS below that also count transverse fluctuations, and agree with the Laplace (harmonic) prediction to $\sim10^{-2}$."""

CELL_E4_TARGET_VIZ = r'''# Target visualization (self-contained, CPU-safe; touches no run state).
# Left: site potential W whose four minima define the coherent phases.
# Right: exact coherent-slice density exp(-beta W)/Z_W -- V(1 (x) v) = W(v),
# so this is pi restricted to the homogeneous 2-plane (a slice, NOT a
# marginal; the exact 24D phase masses also count transverse fluctuations).
import os, sys, math
sys.path.insert(0, os.path.abspath(".."))
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.potentials import phi4_W, PHI4_MINIMA, PHI4_LAPLACE_MASSES

beta_viz = 8.0
vg = torch.linspace(-1.9, 1.9, 601, dtype=torch.float64)
VX, VY = torch.meshgrid(vg, vg, indexing="ij")
Wg = phi4_W(torch.stack([VX, VY], dim=-1))
dens_slice = torch.exp(-beta_viz * (Wg - Wg.min()))
dens_slice = dens_slice / (dens_slice.sum() * float(vg[1] - vg[0]) ** 2)

W0 = PHI4_MINIMA["--"][1]
for ph, (v, Wv) in PHI4_MINIMA.items():
    print(f"  phase {ph}: v = ({v[0]:+.4f}, {v[1]:+.4f})   W = {Wv:+.4f}   "
          f"beta*dW vs '--' = {beta_viz*(Wv - W0):.2f}   "
          f"Laplace mass = {PHI4_LAPLACE_MASSES[ph]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
lv = np.linspace(-0.05, 3.0, 25)
cf0 = axes[0].contourf(vg, vg, Wg.T.clamp(max=3.0), levels=lv, cmap="viridis")
axes[0].contour(vg, vg, Wg.T.clamp(max=3.0), levels=lv, colors="k",
                linewidths=0.25, alpha=0.4)
for ph, (v, Wv) in PHI4_MINIMA.items():
    axes[0].plot(v[0], v[1], "o", ms=5, mfc="w", mec="k")
    axes[0].annotate(f"{ph}: W={Wv:+.3f}", (v[0], v[1]),
                     textcoords="offset points", xytext=(-14, 9),
                     fontsize=8, color="w")
axes[0].set(title=r"site potential $W(x,y)$ (clipped at 3)",
            xlabel="$x$", ylabel="$y$")
fig.colorbar(cf0, ax=axes[0], label="$W$")

cf1 = axes[1].contourf(vg, vg, dens_slice.T, levels=30, cmap="magma")
for ph, (v, _) in PHI4_MINIMA.items():
    axes[1].annotate(f"{ph}: $p^\\star$={PHI4_LAPLACE_MASSES[ph]:.3f}",
                     (v[0], v[1]), textcoords="offset points",
                     xytext=(-18, 10), fontsize=8, color="w")
axes[1].set(title=r"coherent-slice density $e^{-\beta W}/Z_W$ at $\beta=8$"
                  r" (labels: 24D phase masses)",
            xlabel="$x$", ylabel="$y$")
fig.colorbar(cf1, ax=axes[1], label="slice density")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(FIGURES, "coupled_phi4_target." + ext), dpi=200,
                bbox_inches="tight")
plt.show()'''


def build_e4_nb() -> nbf.NotebookNode:
    cells = [
        md(r"""# E4 — Coupled $\phi^4$ chain (24D)

**Target.** $q_i\in\mathbb R^2$, $N_s=12$ periodic sites, $\delta=1/N_s$, $\kappa=2.5$:
$$V(q) = \frac{\kappa}{2\delta}\sum_i\|q_{i+1}-q_i\|^2 + \delta\sum_i W(q_i),\qquad W(x,y) = (x^2-1)^2+(y^2-1)^2-0.0125\,xy+0.0075\,x+0.015\,y.$$
Four coherent phases $(\pm1,\pm1)$; for homogeneous fields $V(\mathbf 1\otimes v)=W(v)$, so the coherent barrier equals the barrier of $W$ ($\beta\cdot\min$ barrier $=7.8$), and the kink-pair cost $5.96\gg1$ makes **the coherent flip the minimum-energy path**. The tilt terms are chosen so the inter-phase asymmetry is $\beta\Delta W = 0.44 \le 0.5$: inside the regime where the tamed fixed-step integrator realises the correction's detailed-balance return flux (at $\beta\Delta W\approx1.8$ a measured, $\Delta t$-independent occupancy offset of order $10\%$ appears — a deliberate benchmark-design choice, recorded here). Phase masses remain distinguishably non-uniform, $p^\star \approx (0.323, 0.212, 0.238, 0.227)$. Init at the $--$ phase; partition = basin of $W$ at $\bar q$. **Reference.** Unweighted $W_2$/MMD clouds use finite sampling-importance-resampling (SIR), which is approximate and not i.i.d.; phase masses and scalar energy moments use direct weighted SNIS with proposal diagnostics. A long PT chain provides an independent phase-mass cross-check."""),
        code(cell_setup("coupled_phi4", "build_e4",
                        'exp = build_e4(device=DEV, basin_cache=os.path.join(CACHE, "basin_map_v2.npz"))')),
        code('''from src.potentials import (PHI4_MINIMA, PHI4_ESCAPE_BARRIERS,
                            PHI4_LAPLACE_MASSES, phi4_W, phi4_W_grad, newton_refine)
V2 = exp.extras["minima_2d"]
for i, ph in enumerate(exp.extras["phases"]):
    v_tab, W_tab = PHI4_MINIMA[ph]
    W = phi4_W(V2[i].unsqueeze(0))[0].item()
    assert abs(V2[i][0].item()-v_tab[0]) < 5e-5 and abs(W - W_tab) < 5e-4
    # p_star is a direct weighted-SNIS estimate; the Laplace table should agree to
    # the anharmonic correction scale (~1e-2)
    assert abs(exp.p_star[i].item() - PHI4_LAPLACE_MASSES[ph]) < 2e-2
print("minima / Laplace masses verified; kink pair cost",
      round(2*exp.pot.kink_energy(), 2), ">> 1.0 coherent barrier")
g = torch.Generator(device=DEV); g.manual_seed(0)
barrier_report = ula_first_passage(exp.pot, exp.box, exp.init_fn(cfg.n_particles, g),
                                   exp.exit_committed, cfg.dt, int(cfg.T/cfg.dt), cfg.eps, g)
barrier_report["kramers_tau_langer"] = exp.kramers_tau
print(f"ULA committed events {barrier_report['event_count']}/"
      f"{barrier_report['n_particles']}; exponential waiting-time MLE "
      f"{barrier_report['exponential_waiting_time_mle']:.0f}, "
      f"KM RMST@T {barrier_report['kaplan_meier_rmst_at_horizon']:.0f}, "
      f"24D Langer time {barrier_report['kramers_tau_langer']:.0f}")'''),
        md(MD_E4_TARGET),
        code(CELL_E4_TARGET_VIZ),
        md(r"""## Jump law and Lévy score (moment-exact)

Homogeneous phase shifts on the **8 edge atoms** of the phase square ($r_a=\mathbf 1_{N_s}\otimes(v_j-v_i)$ for the four edges $\{--\!\leftrightarrow\!-+,\ --\!\leftrightarrow\!+-,\ -+\!\leftrightarrow\!++,\ +-\!\leftrightarrow\!++\}$ in both directions, $w_a=1/8$, shell $h=0.1\min\|r_a\|$). The two **diagonal** pairs $--\!\leftrightarrow\!++$ and $-+\!\leftrightarrow\!+-$ are dropped: their coherent chords pass through the field-zero hilltop at the centre; diagonal transitions relay through a mixed phase in two hops. The gradient energy is exactly invariant under homogeneous shifts, so $V(q-r)-V(q)=\delta\sum_i[W(q_i-d)-W(q_i)]$ is a fixed polynomial in $d$ whose coefficients are the per-particle moments $\sum x_i, \sum x_i^2, \sum x_i^3$ (and $y$ analogues): moments once per step in $O(N_s)$, then every quadrature energy delta is $O(1)$ — **no lattice sweeps** (validated to $10^{-13}$). In 24D the certificate uses the shifted-form identity with importance sampling from the Laplace mixture (equivalent to the deployed score; the $M$ cap never fires on the sampled region). *Optional isotropic jitter* $r=\mathbf 1_{N_s}\otimes(v_j-v_i)+\sigma\xi$ (default $\sigma=0$) is supported only by sampled-bank RA/MA estimators, which use the realised displacement directly; deterministic quadrature and its certificate do not apply when $\sigma>0$."""),
        code('''from src.jumps import gauss_legendre_01
from src.certificate import TanhRidgeProduct
DEFAULT_QUAD = dict(q_theta=C.Q_THETA, q_rho=C.Q_RHO)
phis = make_phi_family(24, exp.extras["means24"][0].tolist(), 1.5, DEV, n_phi=4)
# jump-ALIGNED test functions: in 24D random ridge directions have
# a.r_hat ~ 1/sqrt(24) and are blind to variation along the coherent path,
# which is exactly where the theta-quadrature acts; add ridges along the
# first atoms at three sharpness scales.
for a_idx, sc in ((0, 0.5), (0, 1.0), (2, 0.5)):
    r0 = exp.law.atoms[a_idx]
    rhat = (r0 / r0.norm()).unsqueeze(0)
    mid = exp.extras["means24"][0] + 0.5 * r0
    phis.append(TanhRidgeProduct(rhat, (rhat @ mid.unsqueeze(1)).reshape(1),
                                 torch.tensor([sc], device=DEV)))

def cert_e4(q_theta, q_rho):
    theta, w_theta = gauss_legendre_01(q_theta, DEV)
    shifts, logw = exp.law.quadrature_shifts(q_rho)
    shifts_j, logw_j = exp.law.quadrature_shifts(64)
    return certificate_importance(exp.pot, shifts, logw, theta, w_theta,
                                  cfg.lam, cfg.beta, phis, exp.extras["laplace"],
                                  n_samples=200_000,
                                  nu_shifts_jump=shifts_j, nu_logw_jump=logw_j)

cert_report = cert_e4(**DEFAULT_QUAD)
print(f"max R at default orders = {cert_report['max_residual']:.3e}")
print("NOTE: the jump-aligned phis expose the theta-quadrature defect along "
      "the coherent path at Q_theta=16; the refinement gate below selects "
      "the smallest orders with R < 1e-6 (and re-asserts).")
gm = torch.Generator(device=DEV); gm.manual_seed(11)
Mv, _ = exp.make_score(**DEFAULT_QUAD).log_parts(exp.extras["laplace"].sample(100_000, gm))
print(f"max log score magnitude on support: {Mv.max().item():.1f} << 600")
cert_report["max_log_magnitude_on_support"] = float(Mv.max().item())'''),
        code(CELL_LADDER),
        code(CELL_REFERENCE + '''

# SNIS proposal quality + PT cross-check of the direct-SNIS phase masses
g_ess = torch.Generator(device=DEV); g_ess.manual_seed(555)
ess = exp.extras["laplace"].snis_ess_fraction(exp.pot, cfg.beta, g_ess)
print(f"SNIS proposal ESS fraction: {ess:.3f}")
gen_x = torch.Generator(device=DEV); gen_x.manual_seed(4242)
from src.samplers import ParallelTempering
from src.metrics import occupancy
pt_x = ParallelTempering(exp.pot, exp.init_fn(1000, gen_x), cfg.dt, pt_betas,
                         gen_x, exp.box)
for _ in range(int(round(300.0 / cfg.dt))):
    pt_x.step()
p_pt = occupancy(exp.labels_fn(pt_x.positions()), 4).cpu().numpy()
print("long-PT phase masses:", np.round(p_pt, 3),
      " vs direct-SNIS p*:", np.round(exp.p_star.cpu().numpy(), 3))
pt_crosscheck = p_pt.tolist()'''),
        code('''def run_terminal_lsc(**quad):
    f = make_sampler_factory(exp, cfg.dt, pt_betas, score_kwargs=quad)
    n_ = int(round(cfg.T / cfg.dt))
    r_, _ = run_one("LSC-CP", 0, f, n_, n_, cfg.dt, metrics_fn, exp.pot, quiet=True)
    return {k: r_[-1][k] for k in ("W2", "TV", "MMD", "EMC")}

settings = [dict(q_theta=qt, q_rho=qr) for qt in (8, 16, 32) for qr in (4, 8, 16)]
CHOSEN_QUAD, quad_table = quadrature_refinement(
    settings, run_terminal_lsc, lambda **s: cert_e4(**s)["max_residual"], floors)
print("chosen quadrature:", CHOSEN_QUAD)
cert_report = cert_e4(**CHOSEN_QUAD)
certificate_result = persist_certificate_result(cert_report, CHOSEN_QUAD)
print("final certificate result:", certificate_result)
assert certificate_result["passed"], certificate_result
display(pd.DataFrame(quad_table).round(6))'''),
        code(cell_dt_production('["W2", "TV", "MMD", "EMC"]')),
        code(CELL_STATIONARITY),
        code(CELL_FIGURES),
        code(cell_csv("pt_phase_mass_crosscheck=pt_crosscheck,")),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    return nb



if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for name, builder in [("00_environment_check", build_environment_nb),
                          ("01_double_well", build_e1_nb),
                          ("02_mog40", build_e2_nb),
                          ("03_mb3well_10d", build_e3_nb),          # main E3 (mb3)
                          ("04_coupled_phi4", build_e4_nb),
                          ("05_manuscript_plotting", build_plotting_nb)]:
        nb = builder()
        # Stable IDs make source regeneration byte-for-byte reproducible.
        for index, cell in enumerate(nb.cells):
            cell["id"] = f"{name}-{index:03d}"
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                     "language": "python"}
        path = os.path.join(here, f"{name}.ipynb")
        with open(path, "w") as f:
            nbf.write(nb, f)
        print("wrote", path)
