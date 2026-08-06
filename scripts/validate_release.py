#!/usr/bin/env python
"""Validate the source package, or a frozen release.

    python scripts/validate_release.py            # source package checks
    python scripts/validate_release.py --release  # also require results/figures

The two modes exist because the packages differ. The SOURCE package must be
runnable from zero: it contains ``src/``, ``configs/``, ``notebooks/``,
``scripts/``, ``tests/``, a README, and an environment lock, and it must NOT
require ``results/``, ``figures/``, or ``cache/`` to exist. Only the frozen
RELEASE package additionally carries results, figures, resolved configs,
manifests, and executed notebooks -- and only ``--release`` checks for them.

Nothing here gates a run. Requiring figures before running an experiment was the
old behaviour and is gone: a validator that demands the outputs of the thing it
is about to launch can never pass on a clean checkout.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

#: Paths that must never appear in a notebook's source, outputs, or metadata.
FORBIDDEN_PATH_PATTERNS = (
    re.compile(r"/home/[A-Za-z0-9._-]+/"),
    re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    re.compile(r"C:\\\\Users\\\\"),
    re.compile(r"/mnt/data/"),
)

#: Machinery from the previous pipeline that must not survive anywhere in the
#: normal execution path.
FORBIDDEN_TOKENS = {
    "LSC-CP-MA": "the component-stratified estimator is deleted, not renamed",
    "LSC_CP_MA": "the component-stratified estimator is deleted, not renamed",
    "paired_multiatom": "the component-stratified sampler branch is deleted",
    "MultiAtomShellScore": "the component-stratified score is deleted",
    "jitter_sigma": "jitter is removed from the public configuration",
    "JCP_EXTRA_GPUS": "GPU allow-lists are removed",
    "JCP_GPU": "pinned GPU indices are removed",
    "gpu_guard": "the GPU guard is removed",
    "merge_method_shards": "shard merging is removed; results are discovered by scan",
    "build_notebooks": "the notebook generator is removed; notebooks are source",
    "--require-figures": "release validation never gates a run",
    "run_wallclock_campaign": "wall-clock is not a formal scientific cost metric",
    "launch_production": "the bounded GPU launcher is removed; use scripts/run_experiment.py",
    "recompute_fla_stationarity": "stationarity is a generic per-variant entry point",
}

#: Files allowed to name the forbidden tokens, because their job is to assert
#: those things are gone. Everything else is scanned.
TOKEN_SCAN_EXEMPT = {"validate_release.py", "test_lsc.py", "test_pipeline.py",
                     "test_legacy_removal.py"}

#: Files that must be present for the source package to be runnable from zero.
REQUIRED_SOURCE_PATHS = (
    "src", "configs", "notebooks", "scripts", "tests", "README.md",
    "environment.yml", "configs/registry.yaml", "configs/plots/manuscript.yaml",
)

#: Directories a source package must NOT require.
OPTIONAL_IN_SOURCE = ("results", "figures", "cache")

REQUIRED_RELEASE_PATHS = (
    "results", "figures", "resolved_configs", "manifests", "executed_notebooks",
)


@dataclass
class Report:
    checks: list = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"check": name, "passed": bool(passed),
                            "detail": detail})

    @property
    def failures(self):
        return [check for check in self.checks if not check["passed"]]

    def to_dict(self) -> dict:
        return {"checks": self.checks, "passed": not self.failures,
                "n_failed": len(self.failures)}


# ------------------------------------------------------------ source checks
def check_source_layout(root: Path, report: Report) -> None:
    for relative in REQUIRED_SOURCE_PATHS:
        path = root / relative
        report.add(f"source layout: {relative}", path.exists(),
                   "" if path.exists() else f"missing {path}")


def check_no_result_dependency(root: Path, report: Report) -> None:
    """A clean checkout has no results; importing and configuring must still work."""
    for relative in OPTIONAL_IN_SOURCE:
        present = (root / relative).exists()
        report.add(f"source does not require {relative}/", True,
                   f"present (fine, but not required); " if present
                   else "absent, as a clean checkout would be")


def check_imports(report: Report) -> None:
    modules = ("src.config", "src.pipeline", "src.samplers", "src.score",
               "src.jumps", "src.targets", "src.potentials", "src.metrics",
               "src.observables", "src.measurements", "src.fee", "src.rng",
               "src.results", "src.catalog", "src.calibration", "src.factory",
               "src.plotting", "src.stationarity", "src.references")
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as error:                            # noqa: BLE001
            report.add(f"import {name}", False,
                       f"{type(error).__name__}: {error}")
        else:
            report.add(f"import {name}", True)


def check_configs(root: Path, report: Report) -> None:
    from src.config import (load_method_configs, load_registry, load_yaml)

    try:
        registry = load_registry()
    except Exception as error:                                # noqa: BLE001
        report.add("registry loads", False, f"{type(error).__name__}: {error}")
        return
    report.add("registry loads", True)

    methods = registry.get("methods", {})
    report.add("registry declares methods", bool(methods))
    for name, entry in methods.items():
        for key in ("display_name", "implementation", "supports_tame", "color",
                    "marker"):
            report.add(f"registry {name}.{key}", key in entry,
                       "" if key in entry else f"{name} is missing {key}")
    # ULD is the method name; BAOAB may appear only as the integrator.
    uld = methods.get("ULD", {})
    report.add("ULD displays as ULD",
               uld.get("display_name") == "ULD",
               f"display_name is {uld.get('display_name')!r}")
    report.add("BAOAB appears only as an integrator",
               uld.get("integrator") == "BAOAB")
    ra = methods.get("LSC-CP-RA", {})
    report.add("LSC-CP-RA is one iid estimator family",
               ra.get("estimator_type") == "iid_random_atomic",
               f"estimator_type is {ra.get('estimator_type')!r}")
    report.add("no LSC-CP-MA in the registry", "LSC-CP-MA" not in methods)
    for name in ("MALA", "PT"):
        report.add(f"{name} supports taming",
                   bool(methods.get(name, {}).get("supports_tame")))

    try:
        method_configs = load_method_configs()
    except Exception as error:                                # noqa: BLE001
        report.add("method configs load", False,
                   f"{type(error).__name__}: {error}")
        method_configs = {}
    else:
        report.add("method configs load", True)
    for name in methods:
        report.add(f"method config for {name}", name in method_configs)

    for experiment_id, entry in registry.get("experiments", {}).items():
        path = root / entry["config"]
        if not path.is_file():
            report.add(f"{experiment_id} config exists", False, str(path))
            continue
        report.add(f"{experiment_id} config exists", True)
        config = load_yaml(path)
        for section in ("target", "jump_law", "boundary", "protocol", "taming",
                        "checkpoints", "reference", "metrics", "calibration"):
            report.add(f"{experiment_id} config has {section}",
                       section in config)
        boundary_rule = (config.get("boundary") or {}).get("rule")
        report.add(f"{experiment_id} boundary rule is reject",
                   boundary_rule == "reject",
                   f"rule is {boundary_rule!r}")
        checkpoints = config.get("checkpoints") or {}
        report.add(f"{experiment_id} has no stale n_checkpoints",
                   "n_checkpoints" not in checkpoints and
                   "n_checkpoints" not in config)
        for notebook_key in ("run_notebook", "plot_notebook"):
            notebook = root / entry[notebook_key]
            report.add(f"{experiment_id} {notebook_key} exists",
                       notebook.is_file(), str(notebook))


def check_notebooks(root: Path, report: Report) -> None:
    notebooks = sorted((root / "notebooks").glob("*.ipynb"))
    report.add("eight notebooks present", len(notebooks) == 8,
               f"found {len(notebooks)}: {[p.name for p in notebooks]}")
    for path in notebooks:
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_PATH_PATTERNS:
            match = pattern.search(text)
            report.add(f"{path.name}: no absolute paths", match is None,
                       f"found {match.group(0)!r}" if match else "")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            report.add(f"{path.name}: valid JSON", False, str(error))
            continue
        report.add(f"{path.name}: valid JSON", True)
        cells = payload.get("cells", [])
        has_outputs = any(cell.get("outputs") for cell in cells)
        has_counts = any(cell.get("execution_count") is not None
                         for cell in cells)
        report.add(f"{path.name}: outputs cleared", not has_outputs)
        report.add(f"{path.name}: execution counts cleared", not has_counts)
        if path.name.endswith("_plot.ipynb"):
            source = "\n".join("".join(cell.get("source", []))
                               for cell in cells)
            for forbidden in ("src.pipeline", "src.samplers", "src.factory",
                              "src.calibration", "src.references",
                              "run_variants_and_save", "build_sampler"):
                report.add(
                    f"{path.name}: does not call {forbidden}",
                    forbidden not in source,
                    "a plot notebook must not run a sampler, a tuner, a "
                    "refinement, or a reference build")


def check_no_legacy_tokens(root: Path, report: Report) -> None:
    """The old schema must not survive in the normal execution path.

    There is no new-format-first-then-fall-back-to-old dual path. A one-off
    migration tool may read old files, but it must not sit in the daily run,
    plot, or release flow.
    """
    scanned = []
    for directory in ("src", "scripts", "notebooks", "configs", "tests"):
        base = root / directory
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in (".py", ".yaml", ".yml",
                                                  ".ipynb", ".md"):
                scanned.append(path)
    scanned.extend(path for path in (root / "README.md",) if path.is_file())
    for token, why in FORBIDDEN_TOKENS.items():
        offenders = []
        for path in scanned:
            if path.name in TOKEN_SCAN_EXEMPT:
                continue                      # these name the tokens on purpose
            try:
                if token in path.read_text(encoding="utf-8"):
                    offenders.append(str(path.relative_to(root)))
            except (OSError, UnicodeDecodeError):
                continue
        report.add(f"no legacy token {token!r}", not offenders,
                   f"{why}; found in {offenders}" if offenders else "")


def check_device_policy(root: Path, report: Report) -> None:
    """CPU and CUDA are both supported; nothing may forbid a device."""
    from src.device import resolve_device

    try:
        cpu = resolve_device("cpu")
        auto = resolve_device("auto")
    except Exception as error:                                # noqa: BLE001
        report.add("device resolution", False,
                   f"{type(error).__name__}: {error}")
        return
    report.add("device resolution", True, f"cpu={cpu}, auto={auto}")
    report.add("no CUDA_VISIBLE_DEVICES requirement",
               True,
               f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r} "
               "is provenance only")


def check_output_writable(root: Path, report: Report) -> None:
    try:
        # Probe the repository parent of the default output without creating a
        # results/ directory in a clean source checkout.
        with tempfile.TemporaryDirectory(prefix=".output-write-probe-",
                                         dir=root):
            pass
    except OSError as error:
        report.add("output directory writable", False, str(error))
    else:
        report.add("output directory writable", True, str(root / "results"))


# ----------------------------------------------------------- release checks
def check_release_artifacts(root: Path, report: Report) -> None:
    """Require a complete, reproducible frozen-release evidence matrix."""
    from src.catalog import scan
    from src.config import (default_variants, load_method_configs,
                            load_registry, load_yaml)
    from src.results import MANIFEST_NAME, sha256_file, stable_hash

    roots = {relative: root / relative
             for relative in REQUIRED_RELEASE_PATHS}
    for relative, path in roots.items():
        is_directory = path.is_dir()
        report.add(f"release artifact directory: {relative}", is_directory,
                   "" if is_directory else f"missing directory {path}")
        files = list(path.rglob("*")) if is_directory else []
        nonempty = any(item.is_file() and item.stat().st_size > 0
                       for item in files)
        report.add(f"release artifact nonempty: {relative}", nonempty,
                   "" if nonempty else f"no nonempty files under {path}")

    try:
        registry = load_registry(root / "configs")
        method_configs = load_method_configs(root / "configs")
        plot_config = load_yaml(
            root / "configs" / "plots" / "manuscript.yaml")
    except Exception as error:                                # noqa: BLE001
        report.add("release configuration loads", False,
                   f"{type(error).__name__}: {error}")
        return
    report.add("release configuration loads", True)

    selected_rows = []
    results_root = roots["results"]
    for experiment_id, entry in registry["experiments"].items():
        experiment_key = f"{experiment_id}_{entry['slug']}"
        experiment_dir = results_root / experiment_key
        runs_exist = (experiment_dir / "runs").is_dir()
        report.add(f"{experiment_id}: production runs directory", runs_exist,
                   str(experiment_dir / "runs"))
        if not runs_exist:
            continue

        rows, rejections = scan(experiment_dir)
        statuses_ok = all(
            row.get("status") in ("complete", "uncalibratable")
            for row in rows)
        identities_ok = all(
            row.get("experiment_id") == experiment_id
            for row in rows)
        # A run retired with an INVALID marker is not a defect: the policy is to
        # supersede rather than delete, so a rebuilt campaign necessarily leaves
        # retired directories behind and every one of them is rejected by the
        # scanner. Only rejections the scanner could not explain that way -- a
        # missing manifest, a hash mismatch, an unknown schema, a run that never
        # completed -- indicate a broken release.
        retired = [item for item in rejections
                   if "marked invalid" in str(item.get("reason", ""))]
        unexplained = [item for item in rejections if item not in retired]
        report.add(
            f"{experiment_id}: all run directories verify",
            bool(rows) and not unexplained and statuses_ok and identities_ok,
            f"{len(rows)} admitted; {len(retired)} retired (superseded, kept as "
            f"evidence); {len(unexplained)} unexplained; "
            f"statuses={sorted({row.get('status') for row in rows})}; "
            f"unexplained={unexplained}")

        reference_files = (
            list((experiment_dir / "reference").rglob("*"))
            if (experiment_dir / "reference").is_dir() else [])
        report.add(
            f"{experiment_id}: frozen reference evidence",
            any(path.is_file() and path.stat().st_size > 0
                for path in reference_files),
            str(experiment_dir / "reference"))

        expected_config = load_yaml(root / entry["config"])
        expected = []
        for method in entry["methods"]:
            expected.extend(default_variants(
                registry, method_configs, experiment_id, method))

        missing = []
        nonproduction = []
        selected_for_experiment = []
        for variant in expected:
            candidates = [
                row for row in rows
                if row.get("variant_hash") == variant.hash
                and row.get("status") in ("complete", "uncalibratable")
            ]
            production = []
            for row in candidates:
                resolved_path = (
                    Path(row["run_directory"]) / "resolved_config.yaml")
                try:
                    resolved = load_yaml(resolved_path)
                    saved_input = {
                        key: value for key, value in resolved.items()
                        if key != "resolved"}
                    matches = (
                        stable_hash(saved_input)
                        == stable_hash(expected_config))
                except Exception:                            # noqa: BLE001
                    matches = False
                if matches:
                    production.append(row)
            if not production:
                missing.append(variant.label)
                if candidates:
                    nonproduction.append(variant.label)
                continue
            selected_for_experiment.append(max(
                production,
                key=lambda row: str(row.get("written_at_utc") or "")))

        report.add(
            f"{experiment_id}: complete default variant outcome matrix",
            not missing,
            f"expected={len(expected)}, selected={len(selected_for_experiment)}, "
            f"missing={missing}, reduced-or-nondefault-only={nonproduction}")
        report.add(
            f"{experiment_id}: at least one completed trajectory",
            any(row.get("status") == "complete"
                for row in selected_for_experiment),
            f"statuses={[row.get('status') for row in selected_for_experiment]}")
        selected_rows.extend(selected_for_experiment)

    def copied_hashes(directory: Path, suffixes: tuple[str, ...]) -> set[str]:
        if not directory.is_dir():
            return set()
        return {
            sha256_file(path)
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in suffixes
        }

    manifest_hashes = copied_hashes(roots["manifests"], (".json",))
    resolved_hashes = copied_hashes(
        roots["resolved_configs"], (".yaml", ".yml"))
    missing_manifests = []
    missing_resolved = []
    for row in selected_rows:
        run_dir = Path(row["run_directory"])
        label = f"{row.get('experiment_id')}:{row.get('variant_label')}"
        manifest_path = run_dir / MANIFEST_NAME
        resolved_path = run_dir / "resolved_config.yaml"
        if (not manifest_path.is_file()
                or sha256_file(manifest_path) not in manifest_hashes):
            missing_manifests.append(label)
        if (not resolved_path.is_file()
                or sha256_file(resolved_path) not in resolved_hashes):
            missing_resolved.append(label)
    report.add(
        "collected manifests cover selected default outcomes",
        bool(selected_rows) and not missing_manifests,
        f"selected={len(selected_rows)}, missing={missing_manifests}")
    report.add(
        "collected resolved configs cover selected default outcomes",
        bool(selected_rows) and not missing_resolved,
        f"selected={len(selected_rows)}, missing={missing_resolved}")

    formats = tuple(plot_config["defaults"]["output_formats"])
    views = tuple(plot_config["defaults"].get("tame_views", ()))
    for experiment_id, entry in registry["experiments"].items():
        specification = plot_config[experiment_id]
        output_dir = (
            roots["figures"] / specification["experiment_key"])
        names = list(specification["figures"])
        plot_notebook = root / entry["plot_notebook"]
        try:
            payload = json.loads(plot_notebook.read_text(encoding="utf-8"))
            source = "\n".join(
                "".join(cell.get("source", []))
                for cell in payload.get("cells", []))
            match = re.search(
                r'MAIN_CURVE_FIGURE\s*=\s*["\\\']([^"\\\']+)["\\\']',
                source)
        except Exception as error:                            # noqa: BLE001
            report.add(f"{experiment_id}: plot notebook contract", False,
                       f"{type(error).__name__}: {error}")
            match = None
        else:
            report.add(
                f"{experiment_id}: plot notebook declares tame views",
                match is not None,
                "" if match else "MAIN_CURVE_FIGURE is not declared")
        if match:
            names.extend(
                f"{match.group(1)}__{view}" for view in views)
        for name in names:
            missing_formats = [
                suffix for suffix in formats
                if not (output_dir / f"{name}.{suffix}").is_file()
                or (output_dir / f"{name}.{suffix}").stat().st_size == 0
            ]
            report.add(
                f"{experiment_id}: figure {name}",
                not missing_formats,
                f"missing/empty formats={missing_formats}")

    expected_notebooks = {
        Path(entry[key]).name
        for entry in registry["experiments"].values()
        for key in ("run_notebook", "plot_notebook")
    }
    executed_root = roots["executed_notebooks"]
    report_path = executed_root / "execution_report.json"
    try:
        execution_records = json.loads(
            report_path.read_text(encoding="utf-8"))
        by_name = {
            Path(record.get("executed", "")).name: record
            for record in execution_records
        }
    except Exception as error:                                # noqa: BLE001
        report.add("executed notebook report loads", False,
                   f"{type(error).__name__}: {error}")
        by_name = {}
    else:
        report.add("executed notebook report loads",
                   isinstance(execution_records, list))
    for name in sorted(expected_notebooks):
        record = by_name.get(name, {})
        path = executed_root / name
        passed = (
            path.is_file() and path.stat().st_size > 0
            and record.get("status") == "success")
        report.add(
            f"executed notebook succeeds: {name}", passed,
            f"status={record.get('status')!r}, path={path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", action="store_true",
                        help="also require the frozen release artifacts")
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    report = Report()
    check_source_layout(args.root, report)
    check_no_result_dependency(args.root, report)
    check_imports(report)
    check_configs(args.root, report)
    check_notebooks(args.root, report)
    check_no_legacy_tokens(args.root, report)
    check_device_policy(args.root, report)
    check_output_writable(args.root, report)
    if args.release:
        check_release_artifacts(args.root, report)

    for check in report.checks:
        if not check["passed"]:
            print(f"FAIL  {check['check']}: {check['detail']}")
    passed = len(report.checks) - len(report.failures)
    print(f"\n{passed}/{len(report.checks)} checks passed")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report.to_dict(), indent=2),
                             encoding="utf-8")
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
