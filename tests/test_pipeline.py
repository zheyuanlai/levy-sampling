"""The run/plot split, atomic results, and the derived catalog.

These tests execute a genuinely small but complete experiment so the artifacts
under test are the real ones, written by the real writer.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from src import catalog as catalog_module, pipeline as pipeline_module
from src.calibration import CalibrationError
from src.catalog import scan, select_runs, write_catalog
from src.config import (checkpoint_steps, default_variants, expand_variants,
                        load_experiment, load_method_configs, load_registry,
                        snapshot_checkpoints)
from src.pipeline import run_variants_and_save
from src.results import (COMPLETE_MARKER, MANIFEST_NAME, RunPaths, RunWriter,
                         mark_invalid, read_manifest, verify_run)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

#: A complete E1 run, shrunk so the suite stays fast. Nothing about the code
#: path differs from production; only the sizes do.
SMALL = {
    "protocol": {"particles": 120, "seeds": 2, "final_time": 1.0},
    "checkpoints": {"dense": {"count": 3, "fraction": 0.25},
                    "sparse": {"count": 4}},
    "reference": {"n_grid": 8001, "sample_bank_size": 2000,
                  "validation": {"grid_sizes": [4001, 8001],
                                 "self_w2_replicates": 2}},
    "plot_snapshots": {"time_values": [0.5, 1.0], "max_points_per_seed": 40},
    "calibration": {"dt": {"max_halvings": 2}},
}


@pytest.fixture(scope="module")
def completed_experiment(tmp_path_factory):
    """One real experiment directory with two saved variants."""
    root = tmp_path_factory.mktemp("results")
    experiment = load_experiment("E1", device="cpu", results_root=root,
                                 overrides=SMALL)
    experiment.ensure_reference()
    experiment.ensure_fee_calibration()
    reports = run_variants_and_save(experiment=experiment, method="ULA",
                                    variants=[{}])
    assert all(report["status"] == "complete" for report in reports), reports
    return experiment, reports


# -------------------------------------------------------- saved artifacts
def test_a_completed_run_writes_every_required_artifact(completed_experiment):
    experiment, reports = completed_experiment
    for report in reports:
        run_dir = Path(report["run_directory"])
        for name in ("resolved_config.yaml", "calibration.json",
                     "metrics_timeseries.csv", "cost_timeseries.csv",
                     "terminal_samples.npz", "diagnostics.json",
                     MANIFEST_NAME, COMPLETE_MARKER):
            assert (run_dir / name).is_file(), f"{name} missing from {run_dir}"
        assert list((run_dir / "sample_snapshots").glob("*.npz"))


def test_canonical_and_tamed_are_saved_as_separate_variants(
        completed_experiment):
    _, reports = completed_experiment
    labels = {report["variant_label"] for report in reports}
    assert labels == {"ULA, canonical", "ULA, tamed"}
    directories = {report["run_directory"] for report in reports}
    assert len(directories) == 2


def test_manifest_records_the_provenance_a_reader_needs(completed_experiment):
    _, reports = completed_experiment
    manifest = read_manifest(Path(reports[0]["run_directory"]))
    for key in ("run_id", "schema_version", "method", "variant_label",
                "parameters", "tame", "tame_cap", "dt", "target_hash",
                "reference_hash", "calibration_hash", "variant_hash",
                "metric_definition_hash",
                "rng_pair_group", "particles", "seeds", "device_provenance",
                "fee_calibration_hash", "fee_cost_unit", "files", "status"):
        assert key in manifest, f"manifest is missing {key}"
    assert manifest["fee_cost_unit"] == "amortized_time_per_configuration"
    assert manifest["rng"]["per_seed_generator"] is True
    assert manifest["rng"]["seed_execution_order"] == list(manifest["seeds"])


def test_cost_timeseries_comes_from_counters_not_a_step_formula(
        completed_experiment):
    """The recorded force count must match the counters, and the derived FEE
    must be reconstructible from the raw counters and rho."""
    import csv

    experiment, reports = completed_experiment
    run_dir = Path(reports[0]["run_directory"])
    with (run_dir / "cost_timeseries.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    for row in rows:
        n_force = int(row["n_force"])
        assert n_force == (int(row["n_force_only"])
                           + int(row["n_value_and_force"]))
        assert int(row["n_extra_potential"]) == (
            int(row["n_potential_only"]) - int(row["n_potential_baseline"]))
        expected = n_force + float(row["rho"]) * float(
            row["n_extra_potential_equivalent"])
        assert float(row["n_fee"]) == pytest.approx(expected, rel=1e-9)
    # The counters accumulate: a later checkpoint costs more than an earlier one.
    assert int(rows[-1]["n_force"]) > int(rows[0]["n_force"])


def test_metrics_are_recorded_per_seed_at_every_checkpoint(
        completed_experiment):
    import csv

    experiment, reports = completed_experiment
    run_dir = Path(reports[0]["run_directory"])
    with (run_dir / "metrics_timeseries.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    seeds = {row["seed"] for row in rows}
    steps = {row["step"] for row in rows}
    assert len(seeds) == len(experiment.seeds)
    assert len(rows) == len(seeds) * len(steps)
    for column in ("W2_exact_1d", "MMD2_biased", "KS", "n_fee_per_particle"):
        assert column in rows[0], f"{column} is not in the metrics table"

def test_snapshots_and_resolved_config_carry_complete_cost_identity(
        completed_experiment):
    _, reports = completed_experiment
    run_dir = Path(reports[0]["run_directory"])
    snapshot_path = next((run_dir / "sample_snapshots").glob("*.npz"))
    with np.load(snapshot_path, allow_pickle=False) as snapshot:
        for key in ("n_fee", "n_fee_per_particle", "n_force",
                    "n_force_per_particle", "n_extra_potential",
                    "n_extra_potential_equivalent",
                    "n_extra_potential_equivalent_per_particle", "rho",
                    "fee_calibration_hash", "fee_cost_unit"):
            assert key in snapshot.files
    resolved = yaml.safe_load(
        (run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    runtime = resolved["resolved"]
    assert runtime["checkpoint_costs"]
    assert runtime["fee_calibration"]["fee_calibration_hash"]
    assert runtime["metric_definition_hash"]


def test_stale_metric_definition_hash_is_rejected(completed_experiment):
    _, reports = completed_experiment
    source = Path(reports[0]["run_directory"])
    copy = source.parent / "probe-stale-metric"
    shutil.copytree(source, copy)
    manifest_path = copy / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metric_definition_hash"] = "stale-definition"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    admissible, reason = verify_run(copy)
    assert not admissible
    assert "stale metric definition" in reason
    shutil.rmtree(copy)




# ------------------------------------------------------------ atomicity
def test_an_incomplete_run_is_not_admitted(tmp_path):
    paths = RunPaths(tmp_path, "E9_probe").ensure()
    writer = RunWriter(paths, method="ULA", run_id="half-written")
    writer.__enter__()
    writer.write_text("resolved_config.yaml", "partial: true\n")
    # Abandon it the way a crash would, leaving the temporary directory behind.
    temp_dir = writer.temp_dir
    assert temp_dir.is_dir()
    rows, rejections = scan(paths.experiment_dir)
    assert rows == []
    assert not any(str(temp_dir) == rejection["run_directory"]
                   for rejection in rejections), (
        "a .tmp- directory must be skipped outright, not merely rejected")
    shutil.rmtree(temp_dir)


def test_a_failed_run_leaves_nothing_behind(tmp_path):
    paths = RunPaths(tmp_path, "E9_probe").ensure()
    with pytest.raises(RuntimeError, match="boom"):
        with RunWriter(paths, method="ULA", run_id="doomed") as writer:
            writer.write_text("resolved_config.yaml", "x: 1\n")
            raise RuntimeError("boom")
    assert not (paths.method_dir("ULA") / "doomed").exists()
    assert not list(paths.method_dir("ULA").glob(".tmp-*"))

def test_uncalibratable_variant_is_persisted_and_not_selected_for_plots(
        tmp_path, monkeypatch):
    experiment = load_experiment("E1", device="cpu", results_root=tmp_path,
                                 overrides=SMALL)

    def fail_calibration(*args, **kwargs):
        raise CalibrationError(
            "timestep",
            [{"candidate": 0.001, "passed": False}],
            next_candidate=0.0005,
            diagnosis="no timestep passed the frozen stability gates")

    monkeypatch.setattr(pipeline_module, "run_variant", fail_calibration)
    reports = run_variants_and_save(
        experiment=experiment, method="ULA",
        variants=[{"tame": False}])
    assert len(reports) == 1
    report = reports[0]
    assert report["status"] == "uncalibratable"

    run_dir = Path(report["run_directory"])
    for name in ("resolved_config.yaml", "calibration.json",
                 "diagnostics.json", MANIFEST_NAME, COMPLETE_MARKER):
        assert (run_dir / name).is_file()
    admissible, reason = verify_run(run_dir)
    assert admissible, reason

    rows, rejections = scan(experiment.paths.experiment_dir)
    assert rejections == []
    assert len(rows) == 1 and rows[0]["status"] == "uncalibratable"
    assert select_runs(experiment.paths.experiment_dir,
                       from_manifests=True) == []
    outcomes = select_runs(experiment.paths.experiment_dir, status=None,
                           from_manifests=True)
    assert len(outcomes) == 1 and outcomes[0]["status"] == "uncalibratable"



def test_a_run_without_its_manifest_or_marker_is_rejected(completed_experiment):
    _, reports = completed_experiment
    source = Path(reports[0]["run_directory"])
    for missing in (MANIFEST_NAME, COMPLETE_MARKER):
        copy = source.parent / f"probe-{missing}"
        shutil.copytree(source, copy)
        (copy / missing).unlink()
        admissible, reason = verify_run(copy)
        assert not admissible, f"{missing} removal should disqualify a run"
        assert missing.lower().split(".")[0] in reason.lower()
        shutil.rmtree(copy)


def test_a_tampered_file_is_caught_by_its_hash(completed_experiment):
    _, reports = completed_experiment
    source = Path(reports[0]["run_directory"])
    copy = source.parent / "probe-tampered"
    shutil.copytree(source, copy)
    (copy / "metrics_timeseries.csv").write_text("method,seed\nfake,0\n",
                                                 encoding="utf-8")
    admissible, reason = verify_run(copy)
    assert not admissible
    assert "hash mismatch" in reason
    shutil.rmtree(copy)


def test_a_run_can_be_retired_without_deleting_it(completed_experiment):
    _, reports = completed_experiment
    source = Path(reports[0]["run_directory"])
    copy = source.parent / "probe-invalid"
    shutil.copytree(source, copy)
    mark_invalid(copy, "superseded by a corrected calibration")
    admissible, reason = verify_run(copy)
    assert not admissible and "invalid" in reason
    assert (copy / "metrics_timeseries.csv").is_file(), (
        "retiring a run must not delete its evidence")
    shutil.rmtree(copy)


# -------------------------------------------------------------- catalog
def test_a_worker_never_writes_the_shared_catalog(tmp_path):
    """Running a variant must not create or touch catalog.csv."""
    experiment = load_experiment("E1", device="cpu", results_root=tmp_path,
                                 overrides=SMALL)
    experiment.ensure_reference()
    experiment.ensure_fee_calibration()
    catalog_path = experiment.paths.catalog_path
    assert not catalog_path.exists()
    run_variants_and_save(experiment=experiment, method="ULA",
                          variants=[{"tame": False}])
    assert not catalog_path.exists(), (
        "the worker wrote the shared index; only the scanner may")


def test_the_catalog_is_fully_rebuildable_from_manifests(completed_experiment):
    experiment, reports = completed_experiment
    report = write_catalog(experiment.paths.experiment_dir)
    assert report["n_runs"] == len(reports)
    first = experiment.paths.catalog_path.read_text(encoding="utf-8")
    experiment.paths.catalog_path.unlink()
    write_catalog(experiment.paths.experiment_dir)
    assert experiment.paths.catalog_path.read_text(encoding="utf-8") == first


def test_plot_selection_agrees_whether_it_scans_or_reads_the_catalog(
        completed_experiment):
    experiment, _ = completed_experiment
    write_catalog(experiment.paths.experiment_dir)
    from_catalog = select_runs(experiment.paths.experiment_dir, method="ULA")
    from_manifests = select_runs(experiment.paths.experiment_dir, method="ULA",
                                 from_manifests=True)
    assert ({row["run_id"] for row in from_catalog}
            == {row["run_id"] for row in from_manifests})


def test_tame_and_parameter_filters_select_the_right_variant(
        completed_experiment):
    experiment, _ = completed_experiment
    canonical = select_runs(experiment.paths.experiment_dir, method="ULA",
                            tame=False, from_manifests=True)
    tamed = select_runs(experiment.paths.experiment_dir, method="ULA",
                        tame=True, from_manifests=True)
    assert len(canonical) == 1 and len(tamed) == 1
    assert canonical[0]["run_id"] != tamed[0]["run_id"]

def test_plot_loader_rejects_a_missing_requested_method(completed_experiment):
    from src.plotting import load_runs

    experiment, _ = completed_experiment
    with pytest.raises(ValueError, match="requires missing methods"):
        load_runs(
            experiment.paths.experiment_dir,
            {"methods": ["ULA", "ULD"], "tame_view": "canonical_only"})


def test_plot_loader_annotates_an_uncalibratable_method_instead_of_failing(
        completed_experiment, monkeypatch):
    """A method that was tried and had no admissible timestep is a result.

    E3's canonical FLA is uncalibratable at every alpha, so the canonical view
    of its figures has no FLA curve. Omitting it silently would misrepresent
    the campaign, and raising would make the negative result unpublishable, so
    the loader reports it and the figure annotates it.
    """
    from src.plotting import load_runs

    experiment, _ = completed_experiment

    def fail_calibration(*args, **kwargs):
        raise CalibrationError(
            "timestep", [{"candidate": 0.001, "passed": False}],
            next_candidate=0.0005,
            diagnosis="boundary rejection stays above the frozen gate")

    monkeypatch.setattr(pipeline_module, "run_variant", fail_calibration)
    reports = run_variants_and_save(experiment=experiment, method="ULD",
                                    variants=[{"tame": False}])
    assert reports[0]["status"] == "uncalibratable"

    runs = load_runs(experiment.paths.experiment_dir,
                     {"methods": ["ULA", "ULD"], "tame_view": "canonical_only"})
    assert [run.method for run in runs] == ["ULA"]
    assert "ULD" in runs.uncalibratable
    assert "boundary rejection" in (
        runs.uncalibratable["ULD"][0]["diagnosis"])

    # A method that was never run at all is still a specification error: the
    # tolerance is for recorded negative evidence, not for absent evidence.
    with pytest.raises(ValueError, match="specification error"):
        load_runs(
            experiment.paths.experiment_dir,
            {"methods": ["ULA", "PT"], "tame_view": "canonical_only"})




# ----------------------------------------------------- hashes and caching
def test_changing_a_metric_definition_invalidates_the_reference_hash(tmp_path):
    """A run's reference hash must move when the reference definition moves."""
    first = load_experiment("E1", device="cpu", results_root=tmp_path,
                            overrides=SMALL)
    first.ensure_reference()
    changed = {**SMALL, "reference": {**SMALL["reference"], "n_grid": 6001}}
    second = load_experiment("E1", device="cpu", results_root=tmp_path / "b",
                             overrides=changed)
    second.ensure_reference()
    assert first.reference_hash != second.reference_hash


def test_calibration_is_reused_when_nothing_relevant_changed(tmp_path):
    from src.calibration import calibrate

    experiment = load_experiment("E1", device="cpu", results_root=tmp_path,
                                 overrides=SMALL)
    variant = [v for v in default_variants(experiment.registry,
                                           experiment.method_configs, "E1",
                                           "ULA") if not v.tame][0]
    first = calibrate(experiment, variant)
    stored = list((experiment.paths.protocols_dir).rglob("*.json"))
    assert stored, "the calibration was not cached"
    second = calibrate(experiment, variant)
    assert first["calibration_hash"] == second["calibration_hash"]
    assert first["dt"] == second["dt"]


def test_canonical_and_tamed_calibrate_to_separate_records(tmp_path):
    from src.calibration import calibration_key, PILOT_DEFAULTS

    experiment = load_experiment("E1", device="cpu", results_root=tmp_path,
                                 overrides=SMALL)
    variants = default_variants(experiment.registry,
                                experiment.method_configs, "E1", "ULA")
    keys = {variant.tame: calibration_key(experiment, variant, PILOT_DEFAULTS)
            for variant in variants}
    assert keys[False] != keys[True], (
        "canonical and tamed must not share a calibration record")


# ------------------------------------------------------- variant expansion
def test_an_unpinned_variant_expands_into_canonical_and_tamed():
    registry = load_registry()
    method_configs = load_method_configs()
    expanded = expand_variants(registry, method_configs, "FLA",
                               [{"alpha": 1.7}])
    assert [variant.tame for variant in expanded] == [False, True]
    assert expanded[0].rng_pair_group == expanded[1].rng_pair_group, (
        "paired variants must share their random-stream identity")
    assert expanded[0].hash != expanded[1].hash, (
        "but they are distinct variants")


def test_the_bank_size_appears_in_the_label_only_when_it_is_not_the_default():
    registry = load_registry()
    method_configs = load_method_configs()
    labels = {
        variant.parameters["A"]: variant.label
        for variant in expand_variants(registry, method_configs, "LSC-CP-RA",
                                       [{"A": 1}, {"A": 4}, {"A": 8}])
        if not variant.tame}
    assert labels[1] == "LSC-CP-RA, canonical"
    assert labels[4] == "LSC-CP-RA (A=4), canonical"
    assert labels[8] == "LSC-CP-RA (A=8), canonical"


def test_a_method_that_cannot_be_tamed_is_not_expanded():
    registry = load_registry()
    method_configs = load_method_configs()
    registry = {**registry, "methods": {
        **registry["methods"],
        "ULA": {**registry["methods"]["ULA"], "supports_tame": False}}}
    expanded = expand_variants(registry, method_configs, "ULA", [{}])
    assert len(expanded) == 1 and expanded[0].tame is False


# ----------------------------------------------------------- checkpoints
def test_the_checkpoint_schedule_is_dense_early_and_covers_both_ends():
    steps = checkpoint_steps(10_000, dense_count=60, dense_fraction=0.05,
                             sparse_count=160)
    assert steps[0] == 0 and steps[-1] == 10_000
    assert steps == sorted(set(steps))
    dense = [step for step in steps if 0 < step <= 500]
    assert len(dense) >= 55, "the early window is not densely covered"


def test_snapshots_use_a_saved_checkpoint_at_or_below_the_request():
    steps = [0, 10, 50, 100, 500]
    dt = 0.01
    chosen = snapshot_checkpoints(steps, dt, [0.2, 5.0])
    assert chosen == [10, 500]
    for step, requested in zip(chosen, [0.2, 5.0]):
        assert step * dt <= requested + 1e-12, (
            "a snapshot must never be taken from after the requested time")


# --------------------------------------------- the plot side reads only
def test_the_plotting_module_never_imports_the_run_side():
    """A structural check on the import graph, not a naming convention."""
    source = (REPOSITORY_ROOT / "src" / "plotting.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            if node.level:
                imported.add(f".{node.module}")
    forbidden = {"src.samplers", "src.factory", "src.pipeline",
                 "src.calibration", "src.references", "src.score",
                 "src.targets", "src.measurements", "src.observables",
                 "samplers", "factory", "pipeline", "calibration",
                 "references", "score", "targets", "measurements"}
    offenders = {name for name in imported
                 if name.lstrip(".") in {f.split(".")[-1] for f in forbidden}
                 or name in forbidden}
    assert not offenders, f"src/plotting.py imports the run side: {offenders}"


@pytest.mark.parametrize("notebook", sorted(
    (REPOSITORY_ROOT / "notebooks").glob("*_plot.ipynb")))
def test_a_plot_notebook_never_runs_a_sampler(notebook):
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", []))
                       for cell in payload["cells"]
                       if cell["cell_type"] == "code")
    for forbidden in ("run_variants_and_save", "build_sampler", "src.pipeline",
                      "src.samplers", "src.calibration", "src.references",
                      "src.factory", "ensure_reference", "calibrate",
                      "tune_pt_ladder"):
        assert forbidden not in source, (
            f"{notebook.name} calls {forbidden}; a plot notebook only reads "
            "saved results")


@pytest.mark.parametrize("notebook", sorted(
    (REPOSITORY_ROOT / "notebooks").glob("*.ipynb")))
def test_notebooks_carry_no_machine_specific_paths_or_outputs(notebook):
    text = notebook.read_text(encoding="utf-8")
    for marker in ("/home/", "/Users/", "C:\\\\Users\\\\", "/mnt/data/",
                   "CUDA_VISIBLE_DEVICES", "JCP_GPU", "nvidia-smi"):
        assert marker not in text, f"{notebook.name} contains {marker!r}"
    payload = json.loads(text)
    for cell in payload["cells"]:
        assert not cell.get("outputs"), f"{notebook.name} has stored outputs"
        assert cell.get("execution_count") is None


# ------------------------------------------------------- source package
def test_the_source_package_needs_no_results_directory(tmp_path, monkeypatch):
    """A clean checkout with no results must still configure and calibrate."""
    experiment = load_experiment("E1", device="cpu",
                                 results_root=tmp_path / "fresh",
                                 overrides=SMALL)
    assert experiment.paths.experiment_dir.is_dir()
    assert not any(experiment.paths.runs_dir.iterdir())
    assert experiment.particles == 120
