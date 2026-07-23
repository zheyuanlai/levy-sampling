from __future__ import annotations

import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys
import time

import nbformat
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import launch_production as launcher  # noqa: E402
from src.runner import (  # noqa: E402
    _kaplan_meier_rmst,
    hardware_manifest,
    write_manifest,
    write_summary_csv,
    write_timeseries_csv,
)


def _load_notebook_runner():
    path = ROOT / "notebooks" / "run_notebook.py"
    spec = importlib.util.spec_from_file_location("jcp_run_notebook", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_csv_and_json_writers_refuse_overwrite_by_default(tmp_path):
    rows = [{"method": "A", "seed": 0, "step": 1, "t": 1.0,
             "wallclock_s": 0.1, "nfe": 1, "TV": 0.2}]
    timeseries = tmp_path / "metrics.csv"
    write_timeseries_csv(rows, timeseries)
    with pytest.raises(FileExistsError):
        write_timeseries_csv(rows, timeseries)
    write_timeseries_csv(rows, timeseries, overwrite=True)

    summary = tmp_path / "summary.csv"
    write_summary_csv(rows, ["A"], [0], ["TV"], {},
                      {"TV": {"mean": 0.1, "std": 0.0}}, summary)
    with pytest.raises(FileExistsError):
        write_summary_csv(rows, ["A"], [0], ["TV"], {},
                          {"TV": {"mean": 0.1, "std": 0.0}}, summary)
    write_summary_csv(rows, ["A"], [0], ["TV"], {},
                      {"TV": {"mean": 0.1, "std": 0.0}}, summary,
                      overwrite=True)

    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, value=1)
    with pytest.raises(FileExistsError):
        write_manifest(manifest, value=2)
    write_manifest(manifest, overwrite=True, value=3,
                   nonfinite=[float("nan"), float("inf"), -float("inf")])
    payload = json.loads(
        manifest.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(
            AssertionError(f"non-standard JSON constant: {value}")),
    )
    assert payload["value"] == 3
    assert payload["nonfinite"] == ["nan", "inf", "-inf"]


def test_hardware_manifest_is_cpu_safe_and_has_git_provenance(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = hardware_manifest()
    assert result["cuda_available"] is False
    assert result["gpu"] is None
    assert result["gpu_count_visible"] == 0
    assert result["gpu_compute_apps_at_start"] == []
    assert result["git_commit"]
    assert result["git_branch"]
    assert "git_dirty" in result
    assert set(result["python_packages"]) == {
        "numpy", "scipy", "pandas", "matplotlib", "nbformat"}
    assert all(value is None or isinstance(value, str)
               for value in result["python_packages"].values())


def test_kaplan_meier_rmst_and_censored_exponential_mean_are_distinct():
    # Two events and one right-censoring at the common horizon. With no early
    # censoring, KM-RMST equals the mean observed time: (1 + 2 + 4) / 3.
    rmst = _kaplan_meier_rmst(
        times=[1.0, 2.0, 4.0], events=[True, True, False], horizon=4.0)
    assert rmst == pytest.approx(7.0 / 3.0)
    exponential_mle = (1.0 + 2.0 + 4.0) / 2.0
    assert exponential_mle == pytest.approx(3.5)
    assert rmst != exponential_mle
    assert _kaplan_meier_rmst([4.0, 4.0], [False, False], 4.0) == 4.0


def test_launcher_dry_run_and_hard_concurrency_cap(tmp_path):
    code = launcher.main([
        "--gpus", "0,1", "--max-concurrent", "2",
        "--experiments", "double_well,mog40",
        "--run-id", "cpu-dry-run", "--output-root", str(tmp_path),
        "--dry-run", "--smoke-only", "--no-regen", "--skip-tests",
    ])
    assert code == 0
    run_dir = tmp_path / "cpu-dry-run"
    plan = json.loads((run_dir / "launch_plan.json").read_text())
    assert plan["gpus"] == ["0", "1"]
    assert plan["max_concurrent_effective"] == 2
    assert plan["smoke_only"] is True
    assert json.loads((run_dir / "double_well" / "status.json").read_text())["gpu"] == "0"
    assert json.loads((run_dir / "mog40" / "status.json").read_text())["gpu"] == "1"

    with pytest.raises(SystemExit):
        launcher.main([
            "--gpus", "0,1", "--max-concurrent", "3",
            "--run-id", "invalid", "--output-root", str(tmp_path),
            "--dry-run", "--no-regen", "--skip-tests",
        ])
    with pytest.raises(SystemExit):
        launcher.main([
            "--gpus", "0", "--max-concurrent", "1",
            "--run-id", "cannot-skip-tests", "--output-root", str(tmp_path),
            "--no-regen", "--skip-tests",
        ])


def test_launcher_masks_each_child_to_one_selected_physical_gpu():
    env = launcher.job_environment(
        {"JCP_EXTRA_GPUS": "7"}, gpu="1", selected_gpus=("0", "1"),
        run_id="run", methods="A,B", results_root=Path("/tmp/jcp-results"))
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["JCP_GPU"] == "1"
    assert set(env["JCP_EXTRA_GPUS"].split(",")) == {"0", "1", "7"}
    assert env["JCP_METHODS"] == "A,B"
    assert env["JCP_RESULTS_ROOT"] == "/tmp/jcp-results"


def test_logged_command_preserves_separate_stdout_and_stderr(tmp_path):
    stdout = tmp_path / "job" / "stdout.log"
    stderr = tmp_path / "job" / "stderr.log"
    launcher._initialize_logs(stdout, stderr, "test")
    code = launcher.run_logged(
        [sys.executable, "-c",
         "import sys; print('OUT'); print('ERR', file=sys.stderr); raise SystemExit(3)"],
        cwd=ROOT, env=dict(**__import__("os").environ),
        stdout_path=stdout, stderr_path=stderr, timeout=10, phase="test",
    )
    assert code == 3
    assert "OUT" in stdout.read_text()
    assert "ERR" in stderr.read_text()


def test_failed_job_keeps_logs_and_stage_aware_failure_manifest(tmp_path,
                                                                 monkeypatch):
    calls = []
    def fail_after_certificate(command, **kwargs):
        calls.append((command, kwargs))
        artifacts = tmp_path / "double_well" / "artifacts"
        artifacts.mkdir()
        (artifacts / "original_config.yaml").write_text("experiment: double_well\n")
        (artifacts / "resolved_preflight_config.json").write_text("{}\n")
        (artifacts / "certificate_result.json").write_text(json.dumps({
            "passed": False, "max_residual": 2e-6,
            "tolerance": 1e-6, "settings": {"q_theta": 16}}))
        return 9
    monkeypatch.setattr(launcher, "run_logged", fail_after_certificate)
    args = SimpleNamespace(
        gpus=("0",), run_id="failed-job", notebook_timeout=5,
        wall_timeout=5, output_root=tmp_path.parent,
    )
    result = launcher.run_experiment_job("double_well", "0", tmp_path, args)
    status = json.loads((tmp_path / "double_well" / "status.json").read_text())
    assert len(calls) == 1
    assert Path(calls[0][0][1]).name == "run_notebook.py"
    assert result["status"] == status["status"] == "failed"
    assert status["failure_phase"] == "notebook"
    assert status["returncode"] == 9
    assert status["last_preserved_artifact_stage"] == "certificate_measured"
    assert status["certificate_result_summary"]["passed"] is False
    assert status["artifact_presence_after_notebook"][
        "resolved_preflight_config.json"] is True
    assert (tmp_path / "double_well" / "stdout.log").exists()
    assert (tmp_path / "double_well" / "stderr.log").exists()


def test_notebook_runner_writes_immutable_success_and_failure_artifacts(tmp_path,
                                                                        monkeypatch):
    module = _load_notebook_runner()
    source = tmp_path / "source.ipynb"
    nbformat.write(nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("1 + 1")]),
                   source)

    class SuccessfulClient:
        def __init__(self, notebook, **kwargs):
            self.notebook = notebook

        def execute(self):
            self.notebook.cells.append(nbformat.v4.new_markdown_cell("executed"))

    monkeypatch.setattr(module, "NotebookClient", SuccessfulClient)
    output = tmp_path / "success.ipynb"
    status_path = tmp_path / "success.json"
    status = module.execute_notebook(source, output_notebook=output,
                                     status_path=status_path, timeout=2)
    assert status["status"] == "success"
    assert output.exists() and status_path.exists()
    with pytest.raises(FileExistsError):
        module.execute_notebook(source, output_notebook=output,
                                status_path=tmp_path / "other.json", timeout=2)

    class FailingClient(SuccessfulClient):
        def execute(self):
            self.notebook.cells.append(nbformat.v4.new_markdown_cell("partial"))
            raise RuntimeError("intentional failure")

    monkeypatch.setattr(module, "NotebookClient", FailingClient)
    failed_output = tmp_path / "failed.ipynb"
    failed_status = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="intentional failure"):
        module.execute_notebook(source, output_notebook=failed_output,
                                status_path=failed_status, timeout=2)
    assert failed_output.exists()
    payload = json.loads(failed_status.read_text())
    assert payload["status"] == "failed"
    assert payload["error_type"] == "RuntimeError"


def _load_script_module(filename: str, module_name: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generated_notebooks_emit_exclusive_original_and_resolved_configs():
    for filename in (
        "01_double_well.ipynb", "02_mog40.ipynb",
        "03_mb3well_10d.ipynb", "04_coupled_phi4.ipynb",
    ):
        notebook = nbformat.read(ROOT / "notebooks" / filename, as_version=4)
        source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code")
        assert '"original_config.yaml"' in source
        assert 'open(source_config_path, "x"' in source
        assert 'open(source_config_path, "a"' not in source
        assert '"resolved_preflight_config.json"' in source
        assert 'write_manifest(preflight_config_path, **preflight_config)' in source
        assert '"certificate_result.json"' in source
        assert 'write_manifest(certificate_result_path, **_payload)' in source
        assert '"resolved_config.json"' in source
        assert "resolved_dt=dt_final" in source
        assert "resolved_quadrature=CHOSEN_QUAD" in source
        assert 'stationarity_protocol=stationarity_manifest["protocol"]' in source
        assert "jump_law=_jump_config" in source
        assert "sampling_box=_box_config" in source
        assert "cp_drift_cap=float(exp.cp_drift_cap)" in source
        assert "trace_request=TRACE_REQUEST" in source
        assert "failure_thresholds=FAIL_THRESHOLDS" in source
        assert "observed_failure_diagnostics" in source
        assert "min_mala_acceptance" in source
        assert "min_pt_swap_acceptance" in source
        assert "max_jump_boundary_clip_fraction" in source
        assert "max_basin_map_outside_fraction" in source
        assert "jump_boundary_clip_fraction_cp" in source
        assert "basin_map_outside_fraction_targeting" in source
        assert 'score_clip_fraction=_method_info_max(' in source
        assert 'state_box_clip_fraction=_method_info_max(' in source
        assert '"MALA", "mala_accept_fraction_cumulative"' in source
        assert '"PT", "pt_swap_accept_fraction_cumulative"' in source
        # Source config is written in the setup cell, before certificate cells.
        setup_index = next(i for i, cell in enumerate(notebook.cells)
                           if cell.cell_type == "code"
                           and '"original_config.yaml"' in cell.source)
        preflight_index = next(i for i, cell in enumerate(notebook.cells)
                               if cell.cell_type == "code"
                               and '"resolved_preflight_config.json"' in cell.source)
        certificate_index = next(i for i, cell in enumerate(notebook.cells)
                                 if cell.cell_type == "code"
                                 and "persist_certificate_result" in cell.source
                                 and "CHOSEN_QUAD" in cell.source)
        production_index = next(i for i, cell in enumerate(notebook.cells)
                                if cell.cell_type == "code"
                                and "production total:" in cell.source)
        resolved_index = next(i for i, cell in enumerate(notebook.cells)
                              if cell.cell_type == "code"
                              and '"resolved_config.json"' in cell.source)
        assert setup_index == preflight_index < certificate_index < production_index
        assert production_index < resolved_index
        cert_source = notebook.cells[certificate_index].source
        assert cert_source.index("persist_certificate_result") < cert_source.index(
            'assert certificate_result["passed"]')


def test_replot_uses_explicit_immutable_artifact_and_output_paths(tmp_path,
                                                                  monkeypatch):
    replot_module = _load_script_module("replot_figures.py", "jcp_replot")
    artifacts = tmp_path / "run" / "double_well" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "manifest.json").write_text(json.dumps({
        "experiment": "double_well",
        "bias_floors": {"W2": {"mean": 0.1, "std": 0.01}},
        "emc_target": 1.0,
        "plot": {"methods": ["ULA"], "label_overrides": {}},
    }))
    (artifacts / "metrics_timeseries.csv").write_text(
        "method,seed,step,t,nfe,wallclock_s,W2,MMD,EMC\n"
        "ULA,0,1,0.1,10,0.01,0.2,0.3,0.9\n"
    )

    import src.plotting as plotting
    single_calls = []
    grid_calls = []
    monkeypatch.setattr(
        plotting, "metric_single",
        lambda rows, metric, out_base, **kwargs:
        single_calls.append((metric, Path(out_base), kwargs)),
    )
    monkeypatch.setattr(
        plotting, "metric_grid",
        lambda rows, out_base, **kwargs:
        grid_calls.append((Path(out_base), kwargs)),
    )

    output = tmp_path / "derived" / "replot"
    result = replot_module.replot("double_well", artifacts, output)
    assert output.is_dir()
    assert result["artifacts_dir"] == str(artifacts.resolve())
    assert result["output_dir"] == str(output.resolve())
    assert {call[0] for call in single_calls} == {"W2", "MMD"}
    assert all(call[1].parent == output.resolve() for call in single_calls)
    assert len(single_calls) == 6
    assert len(grid_calls) == 1 and grid_calls[0][0].parent == output.resolve()
    with pytest.raises(FileExistsError):
        replot_module.replot("double_well", artifacts, output)


def test_replot_rejects_incomplete_manifest_without_gpu_fallback(tmp_path):
    replot_module = _load_script_module("replot_figures.py", "jcp_replot_incomplete")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "manifest.json").write_text(
        json.dumps({"experiment": "double_well"}))
    (artifacts / "metrics_timeseries.csv").write_text(
        "method,seed,step,t,nfe,wallclock_s,W2\nULA,0,1,0.1,1,0.1,0.2\n")
    with pytest.raises(ValueError, match="GPU fallback"):
        replot_module.replot("double_well", artifacts, tmp_path / "output")


def test_smoke_config_must_be_subset_of_full_method_matrix():
    """A method shard may narrow the matrix but never widen it.

    A shard smokes exactly the methods it runs, so "everything you run, you
    smoked" still holds; whole-matrix coverage across shards is enforced
    separately by scripts/merge_method_shards.py.
    """
    smoke = _load_script_module("smoke_experiment.py", "jcp_smoke")
    methods = smoke._parse_methods(smoke.EXPERIMENT_NOTEBOOK_METHODS["double_well"])
    args = SimpleNamespace(
        experiment="double_well", methods=methods, particles=64, steps=64,
        q_theta=4, q_rho=2, pt_replicas=4,
        basin_n_grid=48, basin_flow_steps=800, basin_mass_n_quad=96,
        reference_grid_size=128, snis_proposals=2_048,
        max_score_clip_fraction=0.01,
        max_state_box_clip_fraction=0.01,
        max_jump_boundary_clip_fraction=0.0,
        max_basin_map_outside_fraction=0.0,
        max_jump_cap_hits=0,
        min_mala_acceptance=0.01, min_pt_swap_acceptance=0.01,
    )
    smoke._validate_args(args)                 # the full matrix
    args.methods = methods[:-1]
    smoke._validate_args(args)                 # a proper shard is allowed
    args.methods = []
    with pytest.raises(ValueError, match="at least one method"):
        smoke._validate_args(args)
    args.methods = list(methods) + ["NOT-A-METHOD"]
    with pytest.raises(ValueError, match="subset"):
        smoke._validate_args(args)


def test_launcher_and_smoke_method_tables_agree():
    """The launcher and the smoke script keep independent copies of the
    production method matrix. A divergence between them is invisible until it
    aborts a launched production run at the smoke gate, so pin them together.
    """
    smoke = _load_script_module("smoke_experiment.py", "jcp_smoke_tables")
    assert set(smoke.EXPERIMENT_NOTEBOOK_METHODS) == set(launcher.EXPERIMENTS)
    for name, (_notebook, methods) in launcher.EXPERIMENTS.items():
        assert methods == smoke.EXPERIMENT_NOTEBOOK_METHODS[name], name


def test_infinite_box_bounds_survive_strict_json():
    """An unbounded coordinate serialises as a sentinel, not a crash.

    A periodic coordinate has no box boundary, so TorusBox sets its limits to
    +-inf by construction. Writing that with allow_nan=False raises, which took
    down two launched E5 production runs at the smoke gate. json_safe maps the
    non-finite values onto the repo's "inf"/"-inf" sentinels while NaN, which is
    never legitimate here, stays visible as "nan" rather than being silently
    dropped.
    """
    import json
    import math

    from src.runner import json_safe

    lo = torch.tensor([-math.inf, -1.5, 0.0])
    hi = torch.tensor([math.inf, 1.5, math.inf])
    payload = {"lower": json_safe(lo), "upper": json_safe(hi)}
    text = json.dumps(payload, allow_nan=False)          # must not raise
    assert json.loads(text) == {"lower": ["-inf", -1.5, 0.0],
                                "upper": ["inf", 1.5, "inf"]}
    assert json_safe(float("nan")) == "nan"
    # the raw form the bug used is still rejected, so the sentinel is doing the
    # work rather than allow_nan having been quietly relaxed somewhere
    with pytest.raises(ValueError):
        json.dumps({"lower": lo.tolist()}, allow_nan=False)


def test_every_deployed_lsc_arm_charges_score_quadrature():
    """No deployed LSC arm may integrate the Levy score in closed form."""
    smoke = _load_script_module("smoke_experiment.py", "jcp_smoke_analytic")
    for experiment in smoke.EXPERIMENT_NOTEBOOK_METHODS:
        for method in smoke.EXPERIMENT_NOTEBOOK_METHODS[experiment].split(","):
            expected = method.startswith("LSC-CP")
            assert smoke._requires_score_quadrature(experiment, method) is expected, (
                f"{experiment}/{method}")


# An experiment may substitute an OFFLINE exact-vs-realised cross-validation for
# a DEPLOYED exact arm only where deploying it at the production ensemble is
# prohibitive. E5 is the sole case: the exact arm was measured at 1.67 s/step at
# its best block size (66 GiB peak; 2^18 gives 1.99 s/step, so the cost is
# FLOP/memory bound, not launch bound), i.e. ~18.5 h for 40k steps against ~3 h
# for the realised-measure arm. This is not a weakening of the invariant: an
# experiment on this list must instead ship cross-validation evidence, which
# test_offline_validated_exact_arms_ship_cross_validation_evidence checks.
EXACT_ARM_VALIDATED_OFFLINE = {"alanine_dipeptide": "results/e5_exact_vs_ma.json"}


def test_production_method_matrix_carries_two_lsc_arms_everywhere():
    """Each experiment reports an exact arm and one realised-displacement arm.

    The exact arm may be validated offline instead of deployed, but only for an
    experiment on EXACT_ARM_VALIDATED_OFFLINE, and never both ways at once.
    """
    for name, (_notebook, methods) in launcher.EXPERIMENTS.items():
        arms = [m for m in methods.split(",") if m.startswith("LSC-CP")]
        realised = [a for a in arms if a in ("LSC-CP-RA", "LSC-CP-MA")]
        assert len(realised) == 1, f"{name} needs one realised arm: {arms}"
        if name in EXACT_ARM_VALIDATED_OFFLINE:
            assert "LSC-CP" not in arms, (
                f"{name} both deploys the exact arm and claims offline "
                f"validation; do one or the other: {arms}")
            continue
        assert "LSC-CP" in arms, f"{name} is missing the exact arm: {arms}"
        assert len(arms) == 2, f"{name} must carry exactly two LSC arms: {arms}"


def _write_shard(root, rid, methods, registered, experiment="alanine_dipeptide",
                 commit="abc123"):
    art = root / rid / experiment / "artifacts"
    (art / "stationarity").mkdir(parents=True)
    (root / rid / "launch_plan.json").write_text(json.dumps({
        "registered_methods": {experiment: registered},
        "methods": {experiment: ",".join(methods)},
        "experiments": [experiment], "smoke_config": {"particles": 64},
        "git": {"commit": commit}}))
    (root / rid / "status.json").write_text(json.dumps({"status": "ok"}))
    import csv as _csv
    for name, cols in (("summary.csv", ["method", "TV_mean"]),
                       ("metrics_timeseries.csv", ["method", "step", "TV"]),
                       ("positions.csv", ["method", "particle", "cv0", "cv1"])):
        with (art / name).open("w", newline="") as handle:
            writer = _csv.DictWriter(handle, fieldnames=cols)
            writer.writeheader()
            for m in methods:
                writer.writerow({c: (m if c == "method" else "1") for c in cols})
            if name == "positions.csv":            # shard-independent block
                writer.writerow({c: ("reference" if c == "method" else "9")
                                 for c in cols})
    (art / "stationarity" / "all_methods_summary.csv").write_text(
        "method,worst_basin_ess\nX,1\n")


def test_method_shard_merge_is_complete_and_deduplicates_shared_blocks(tmp_path):
    """The merge is what turns gated shards back into one experiment.

    Two things it must get right, both of which bit in practice: positions.csv
    carries a `reference` pseudo-method block that every shard writes, so naive
    concatenation triplicates the reference AND makes each shard look like it
    covered a method it never ran; and a shard set that does not cover the
    registered matrix must abort rather than emit a table with a hole in it.
    """
    merge = _load_script_module("merge_method_shards.py", "jcp_merge")
    registered = "ULA,MALA,FLA,BAOAB,PT,CP,LSC-CP-MA"
    shards = {"r-A": ["LSC-CP-MA"], "r-B": ["PT", "MALA", "FLA"],
              "r-C": ["CP", "BAOAB", "ULA"]}
    for rid, methods in shards.items():
        _write_shard(tmp_path, rid, methods, registered)

    prov = merge.merge("alanine_dipeptide", list(shards), "merged", tmp_path)
    assert prov["coverage_complete"] is True
    assert prov["source_run_ids"] == list(shards)

    import csv as _csv
    from collections import Counter
    out = tmp_path / "merged" / "alanine_dipeptide" / "artifacts"
    counts = Counter(r["method"] for r in
                     _csv.DictReader((out / "positions.csv").open()))
    assert counts["reference"] == 1, counts          # not once per shard
    assert set(counts) == set(registered.split(",")) | {"reference"}
    # every shard's non-CSV artifacts survive, attributed to their shard
    assert {p.parent.parent.name for p in
            out.glob("per_shard/*/stationarity/*.csv")} == set(shards)

    # incomplete coverage must abort
    with pytest.raises(ValueError, match="union of shards"):
        merge.merge("alanine_dipeptide", ["r-A", "r-B"], "merged2", tmp_path)


def test_method_shard_merge_across_commits_gated_on_numeric_engine(tmp_path,
                                                                   monkeypatch):
    """A shard at a different commit is mergeable iff the numerical engine is
    byte-identical across the commits -- this is what lets a shard survive a
    later notebook/launcher/doc fix without a full recompute, while still
    refusing a merge whose rows came from different numerics.
    """
    merge = _load_script_module("merge_method_shards.py", "jcp_merge_commits")
    registered = "ULA,MALA,FLA,BAOAB,PT,CP,LSC-CP-MA"
    _write_shard(tmp_path, "r-A", ["LSC-CP-MA"], registered, commit="c_new")
    _write_shard(tmp_path, "r-B", ["PT", "MALA", "FLA"], registered, commit="c_new")
    _write_shard(tmp_path, "r-C", ["CP", "BAOAB", "ULA"], registered, commit="c_old")

    # engine byte-identical at both commits -> different commits are allowed
    monkeypatch.setattr(merge, "_git_blob",
                        lambda commit, path: b"identical-engine-bytes")
    prov = merge.merge("alanine_dipeptide", ["r-A", "r-B", "r-C"], "m_ok", tmp_path)
    assert prov["coverage_complete"] is True
    assert prov["shard_commits"]["r-C"] == "c_old"

    # one engine file differs between the commits -> refuse
    def _blob(commit, path):
        if path.endswith("samplers.py"):
            return b"A" if commit == "c_old" else b"B"
        return b"same"
    monkeypatch.setattr(merge, "_git_blob", _blob)
    with pytest.raises(ValueError, match="numerical engine differs"):
        merge.merge("alanine_dipeptide", ["r-A", "r-B", "r-C"], "m_bad", tmp_path)


def test_offline_validated_exact_arms_ship_cross_validation_evidence():
    """Trading a deployed exact arm for offline validation requires the evidence.

    Two independent claims must be on file: that the realised-measure estimator
    is pointwise unbiased for the exact score, and that substituting it does not
    change the end-to-end answer. Without this artifact the substitution above is
    an unsupported assertion, so its absence must fail the suite.
    """
    for name, rel in EXACT_ARM_VALIDATED_OFFLINE.items():
        path = ROOT / rel
        assert path.exists(), f"{name}: missing cross-validation artifact {rel}"
        data = json.loads(path.read_text(encoding="utf-8"))
        # (A) pointwise: E_bank[S_MA] -> S_exact
        pw = data["pointwise"]
        assert pw["median_rel_err"] < 1e-2, pw
        assert pw["p90_rel_err"] < 5e-2, pw
        assert pw["corr_phi"] > 0.999, pw
        # (B) end-to-end, run at the production step count and ensemble size
        cfg = data["config"]
        assert cfg["steps"] == 40_000 and cfg["n_particles"] == 1_000, cfg
        for arm in ("ma", "exact"):
            assert data[arm]["nonfinite"] == 0, (arm, data[arm])
            assert data[arm]["basin_L1"] < 0.1, (arm, data[arm])


def test_smoke_builders_receive_coarse_settings_and_run_local_cache(tmp_path,
                                                                    monkeypatch):
    smoke = _load_script_module("smoke_experiment.py", "jcp_smoke_builders")
    import src.experiments as experiments
    calls = []
    sentinel = object()

    def fake_e3(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(experiments, "build_e3", fake_e3)
    args = SimpleNamespace(
        basin_n_grid=48, basin_flow_steps=800, basin_mass_n_quad=96,
        reference_grid_size=128, snis_proposals=2_048,
    )
    cache_dir = tmp_path / "run" / "smoke_cache"
    assert smoke._build_experiment(
        "mb3well_10d", "cuda", cache_dir, args) is sentinel
    kwargs = calls[0]
    assert kwargs["basin_n_grid"] == 48
    assert kwargs["basin_flow_steps"] == 800
    assert kwargs["basin_mass_n_quad"] == 96
    assert kwargs["reference_grid_shape"] == (128, 128)
    assert Path(kwargs["basin_cache"]).parent == cache_dir
    assert str(ROOT / "cache") not in kwargs["basin_cache"]


def test_smoke_job_requires_success_artifacts_and_records_real_command(tmp_path,
                                                                       monkeypatch):
    commands = []

    def fake_logged(command, **kwargs):
        commands.append((command, kwargs))
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True)
        for name in ("original_config.yaml", "resolved_config.json",
                     "smoke_metrics.csv", "smoke_manifest.json"):
            (output / name).write_text("{}\n")
        return 0

    monkeypatch.setattr(launcher, "run_logged", fake_logged)
    args = SimpleNamespace(
        gpus=("0",), run_id="smoke-job", output_root=tmp_path,
        smoke_timeout=5,
    )
    result = launcher.run_smoke_job("double_well", "0", tmp_path / "run", args)
    assert result["status"] == "success"
    command, kwargs = commands[0]
    assert Path(command[1]).name == "smoke_experiment.py"
    assert command[command.index("--experiment") + 1] == "double_well"
    assert command[command.index("--methods") + 1] == launcher.DUAL_RA
    assert kwargs["phase"] == "dynamics_smoke"
    status = json.loads(
        (tmp_path / "run" / "smoke" / "double_well" / "status.json").read_text())
    assert status["status"] == "success"


def test_launcher_stops_before_all_full_jobs_on_any_smoke_failure(tmp_path,
                                                                   monkeypatch):
    monkeypatch.setattr(
        launcher, "_run_preflight",
        lambda *args, **kwargs: {"status": "success", "returncode": 0},
    )
    monkeypatch.setattr(
        launcher, "run_smoke_job",
        lambda name, gpu, run_dir, args: {
            "experiment": name, "gpu": gpu, "status": "failed",
            "failure_phase": "dynamics_smoke",
        },
    )
    full_calls = []
    monkeypatch.setattr(
        launcher, "run_experiment_job",
        lambda *args, **kwargs: full_calls.append(args) or {
            "experiment": args[0], "gpu": args[1], "status": "success"},
    )
    code = launcher.main([
        "--gpus", "0", "--max-concurrent", "1",
        "--experiments", "double_well",
        "--run-id", "smoke-must-pass", "--output-root", str(tmp_path),
        "--no-regen",
    ])
    assert code == 1
    assert full_calls == []
    status = json.loads((tmp_path / "smoke-must-pass" / "status.json").read_text())
    assert status["failure_phase"] == "smoke"
    assert status["full_experiments_started"] is False
    assert status["failed_smoke_experiments"] == ["double_well"]



def test_notebook_builder_output_is_independent_of_caller_cwd():
    source = (ROOT / "notebooks" / "build_notebooks.py").read_text()
    assert 'here = os.path.dirname(os.path.abspath(__file__))' in source
    assert 'path = os.path.join(here, f"{name}.ipynb")' in source
    assert 'steps_per_ck = max(1, n_steps // cfg.n_checkpoints)' in source
    assert 'C.N_CHECKPOINTS' not in source
    assert 'basin_map_v2.npz' in source


def test_smoke_persists_request_before_build_and_declares_acceptance_gates():
    smoke = _load_script_module("smoke_experiment.py", "jcp_smoke_static")
    source = inspect.getsource(smoke.run_smoke)
    assert source.index('original_config.yaml') < source.index('_build_experiment')
    assert launcher.SMOKE_CONFIG["min_mala_acceptance"] == 0.01
    assert launcher.SMOKE_CONFIG["min_pt_swap_acceptance"] == 0.01
    assert launcher.SMOKE_CONFIG["max_jump_boundary_clip_fraction"] == 0.0
    assert launcher.SMOKE_CONFIG["max_basin_map_outside_fraction"] == 0.0
    assert 'method == "MALA"' in source and 'method == "PT"' in source
    assert 'nonfinite_proposal_count_cumulative' in source


def test_run_logged_timeout_kills_descendant_process_group(tmp_path):
    stdout = tmp_path / "timeout" / "stdout.log"
    stderr = tmp_path / "timeout" / "stderr.log"
    launcher._initialize_logs(stdout, stderr, "timeout")
    child_pid_path = tmp_path / "child.pid"
    parent_code = (
        "import pathlib,subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid)); "
        "time.sleep(30)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        launcher.run_logged(
            [sys.executable, "-c", parent_code], cwd=ROOT,
            env=dict(os.environ), stdout_path=stdout, stderr_path=stderr,
            timeout=0.3, phase="timeout-test")
    deadline = time.monotonic() + 3.0
    while not child_pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert child_pid_path.exists()
    child_pid = int(child_pid_path.read_text())

    def alive_non_zombie(pid):
        stat = Path(f"/proc/{pid}/stat")
        if not stat.exists():
            return False
        try:
            return stat.read_text().split()[2] != "Z"
        except (FileNotFoundError, IndexError):
            return False

    while alive_non_zombie(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not alive_non_zombie(child_pid)


def test_full_launcher_uses_only_notebook_certificate_gate(tmp_path):
    plan = launcher._job_plan(
        "double_well", "0", "run", tmp_path / "double_well", 60)
    assert "inside notebook" in plan["certificate_gate_execution"]
    assert any(path.endswith("resolved_preflight_config.json")
               for path in plan["expected_preflight_artifacts"])
    assert any(path.endswith("certificate_result.json")
               for path in plan["expected_success_artifacts"])
    source = inspect.getsource(launcher.run_experiment_job)
    assert "certificate_gate.py" not in source
    assert "gate_timeout" not in source


def test_phi4_box_is_laplace_tail_and_one_jump_safe(tmp_path):
    from src.experiments import build_e4
    exp = build_e4(
        device="cpu", basin_cache=str(tmp_path / "phi4_basin.npz"),
        basin_n_grid=8, basin_flow_steps=2, snis_proposals=128)
    design = exp.extras["sampling_box_design"]
    assert design["tail_probability_union_bound"] == pytest.approx(1e-8)
    assert design["phase_component_extent"] == pytest.approx(1.0034238, rel=1e-6)
    assert design["max_component_std_beta"] == pytest.approx(0.1403836, rel=1e-6)
    assert design["max_componentwise_jump_reach"] == pytest.approx(
        2.2000015, rel=1e-6)
    assert design["one_jump_target_required_half_width"] == pytest.approx(
        design["target_phase_envelope_half_width"]
        + design["max_componentwise_jump_reach"])
    assert design["required_half_width_before_rounding"] < 5.0
    assert design["sampling_box_half_width"] == 5.0
    assert torch.all(exp.box.lo == -5.0) and torch.all(exp.box.hi == 5.0)

    endpoints = (exp.law.atoms.unsqueeze(1)
                 + torch.tensor([-1.0, 1.0], dtype=torch.float64).view(1, 2, 1)
                 * exp.law.h.view(-1, 1, 1)
                 * exp.law.units.unsqueeze(1))
    landings = (exp.extras["means24"].view(4, 1, 1, 24)
                + endpoints.view(1, 8, 2, 24)).reshape(-1, 24)
    assert bool(exp.box.contains(landings).all())
    diagnostics = exp.extras["reference_diagnostics"]
    assert "weighted_outside_jump_safe_core_mass" in diagnostics
    assert "weighted_basin_map_outside_mass" in diagnostics
    # Bounds are derived from the basin map object; the domain must cover the
    # jump-reachable order-parameter set (double-jump reach ~3.3), not just
    # the phase minima.
    assert exp.extras["basin_map_metric_bounds"] == [[-4.0, -4.0], [4.0, 4.0]]
    basins = exp.extras["basins"]
    assert exp.extras["basin_map_metric_bounds"] == [
        basins.lo.tolist(), basins.hi.tolist()]


def test_e4_energy_reference_is_declared_direct_snis():
    source = (ROOT / "src" / "experiments.py").read_text()
    assert '"reference_energy_values": p_energy.detach()' in source
    assert 'energy_reference_method = "direct_snis_weighted_histogram"' in source



def test_replot_plot_policy_shows_both_lsc_arms_and_labels_the_atom_count():
    """Every experiment plots the exact arm AND the realised-displacement arm."""
    module = _load_script_module("replot_figures.py", "jcp_replot_policy")

    low, low_labels = module._plot_policy(
        "double_well", {"ULA", "CP", "LSC-CP", "LSC-CP-RA"})
    assert low == ["ULA", "CP", "LSC-CP", "LSC-CP-RA"]
    assert low_labels["LSC-CP"] == "LSC-CP"
    assert low_labels["LSC-CP-RA"] == "LSC-CP-RA"

    for experiment, atoms in (("mb3well_10d", 4), ("coupled_phi4", 8)):
        high, high_labels = module._plot_policy(
            experiment, {"ULA", "CP", "LSC-CP", "LSC-CP-MA"})
        assert high == ["ULA", "CP", "LSC-CP", "LSC-CP-MA"]
        assert high_labels["LSC-CP"] == "LSC-CP"
        assert high_labels["LSC-CP-MA"] == f"LSC-CP-RA ({atoms})"

    both, _ = module._plot_policy(
        "double_well", {"LSC-CP", "LSC-CP-RA", "LSC-CP-MA"})
    assert {"LSC-CP", "LSC-CP-RA", "LSC-CP-MA"} <= set(both)


def test_notebook_and_replot_plot_policies_agree_on_labels():
    """Generator and CSV-only replot duplicate the policy; drift breaks replots."""
    module = _load_script_module("replot_figures.py", "jcp_replot_policy")
    generator = (ROOT / "notebooks" / "build_notebooks.py").read_text()
    for experiment, label in module._RA_LABEL.items():
        assert f'"{experiment}": "{label}"' in generator, (
            f"{experiment} -> {label} missing from the notebook generator")


def test_mirror_into_repo_refreshes_results_and_figures(tmp_path):
    """The in-repo mirror must track the latest run, including deletions.

    JCP_experiments/results/ and figures/ were previously copied across by hand
    and silently went several runs stale; a method dropped from the matrix must
    not leave its old per-method stationarity file behind either.
    """
    from src.runner import mirror_into_repo

    src = tmp_path / "run" / "artifacts"
    (src / "stationarity").mkdir(parents=True)
    (src / "figures").mkdir()
    (src / "summary.csv").write_text("method\nLSC-CP\n")
    (src / "manifest.json").write_text("{}")
    (src / "positions.csv").write_text("method,cv0\nreference,0.0\n")
    (src / "stationarity" / "LSC-CP_summary.csv").write_text("x\n1\n")
    (src / "figures" / "e_LSC-CP.png").write_bytes(b"png")
    (src / "figures" / "e_LSC-CP.pdf").write_bytes(b"pdf")

    repo = tmp_path / "repo"
    # pre-existing stale mirror content that must be replaced, not merged
    (repo / "results" / "e" / "stationarity").mkdir(parents=True)
    (repo / "results" / "e" / "stationarity" / "DROPPED_summary.csv").write_text("old\n")
    (repo / "figures" / "e").mkdir(parents=True)
    (repo / "figures" / "e" / "stale.png").write_bytes(b"old")

    report = mirror_into_repo(src, "e", repo)

    results = repo / "results" / "e"
    figures = repo / "figures" / "e"
    assert (results / "summary.csv").read_text() == "method\nLSC-CP\n"
    assert (results / "positions.csv").is_file()
    assert (results / "stationarity" / "LSC-CP_summary.csv").is_file()
    # the dropped method's stale file is gone, not merged in
    assert not (results / "stationarity" / "DROPPED_summary.csv").exists()
    # stale figures are cleared before the new ones land
    assert not (figures / "stale.png").exists()
    assert (figures / "e_LSC-CP.png").is_file() and (figures / "e_LSC-CP.pdf").is_file()
    assert report["figure_files"] == 2
    assert "summary.csv" in report["files"]
    # absent optional inputs are skipped rather than erroring
    assert "modes.csv" not in report["files"]


def test_every_notebook_mirrors_into_the_repo_tree():
    """Each generated notebook ends by refreshing the in-repo mirror."""
    for name in ("01_double_well", "02_mog40", "03_mb3well_10d",
                 "04_coupled_phi4", "05_alanine_dipeptide"):
        nb = nbformat.read(ROOT / "notebooks" / f"{name}.ipynb", as_version=4)
        sources = ["".join(c["source"]) for c in nb.cells]
        assert any("mirror_into_repo(RESULTS, EXPERIMENT" in s for s in sources), name
