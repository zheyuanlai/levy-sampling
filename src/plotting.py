"""Read-only manuscript figures.

This module reads saved run artifacts and draws them. It never runs a sampler,
never builds a reference, never tunes a step size, and never recomputes an
official metric: every number that reaches an axis already exists in a run's
``metrics_timeseries.csv``, ``cost_timeseries.csv``, ``manifest.json``, or a
saved ``sample_snapshots/*.npz``.

IMPORT RULE (a test asserts the import graph): ``src.plotting`` must not import
or call ``src.samplers``, ``src.factory``, ``src.pipeline``, ``src.calibration``,
``src.references``, ``src.score``, or ``src.targets`` -- directly or through a
helper. Permitted imports are the standard library, numpy, matplotlib, yaml,
``src.catalog``, and ``src.results``. ``src.config`` is permitted by policy for
``load_registry``/``load_yaml`` only, but is deliberately NOT imported here:
importing it pulls ``src.targets`` into the process and would break the rule
above. The registry and the plot configuration are plain YAML, so
:func:`load_registry` and :func:`load_plot_config` read them directly.

Scatter, CDF, histogram, and KDE panels are display-only renderings of saved
snapshots. They never override or restate the numbers in
``metrics_timeseries.csv``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import json
import math
import warnings

import numpy as np
import yaml

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .catalog import select_runs
from .results import json_safe, read_manifest

__all__ = [
    "RunData", "Snapshot",
    "apply_style", "load_plot_config", "load_registry", "figure_spec",
    "load_runs", "load_reference_artifacts", "method_style", "seed_aggregate",
    "check_fee_comparability", "check_extra_potential_eligibility",
    "select_snapshot", "tame_view_filters",
    "shared_limits", "points_per_panel", "assert_panels_consistent",
    "curve_figure", "snapshot_figure", "twin_axis_cdf_figure",
    "contour_scatter_grid", "snapshot_matrix", "mode_metric_panel",
    "supplement_panels", "save_figure", "figure_provenance",
]

#: Sentinel telling an argument apart from an explicitly passed ``None``.
_UNSET = object()

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = REPOSITORY_ROOT / "configs"

#: Which saved column each x axis reads. A figure plots simulation time, the
#: force-equivalent cost, or the LSC score potential-evaluation count, and each
#: of those is a column some run already wrote.
X_AXIS_COLUMNS = {
    "simulation_time": "t",
    "fee": "n_fee_per_particle",
    "extra_potential": "n_extra_potential_equivalent_per_particle",
}

DEFAULT_X_AXIS_LABELS = {
    "simulation_time": r"simulation time $t$",
    "fee": r"force-equivalent cost $N_{\mathrm{FEE}}$ / particle",
    "extra_potential": r"$N_{V,\mathrm{eq}}^{\mathrm{extra}}$ / particle",
}

DEFAULT_UNCERTAINTY = {
    "unit": "seed",
    "estimator": "seed_bootstrap",
    "interval": 0.95,
    "bootstrap_replicates": 2000,
    "bootstrap_seed": 5150,
}

#: Axes every official curve is produced against when a spec names none.
DEFAULT_CURVE_X_AXES = ("simulation_time", "fee")

#: Quantities a supplement panel may take from a saved snapshot array by
#: slicing it. These are display-only views of a saved array, never a
#: recomputed observable.
_DERIVED_SNAPSHOT_QUANTITIES = {
    "mx": ("order_parameter", 0),
    "my": ("order_parameter", 1),
    "m_norm": ("order_parameter", "norm"),
}

#: Saved metric-column families a supplement panel may read as a vector.
_METRIC_FAMILY_PREFIXES = {
    "two_point_correlation": "correlation_r",
    "phase_occupancy": "phase_occupancy_",
    "susceptibility": "susceptibility_",
    "occupancy_ratio": "mode_occupancy_ratio_",
}

#: The extra-potential axis counts LSC score potential evaluations only. The
#: title is fixed so no figure can quietly promote it to a total cost.
EXTRA_POTENTIAL_TITLE = "LSC score potential-evaluation cost"

#: Phrases an extra-potential axis label may not contain.
_FULL_COST_CLAIMS = ("total cost", "total computational", "full cost",
                     "full computational", "wall clock", "wall-clock",
                     "computational cost")

#: The integrator name that must never reach a reader. ULD is the method.
_FORBIDDEN_DISPLAY_TOKEN = "BAOAB"

_MARKER_SIZE = 3.2
_SCATTER_SIZE = 2.0
_SCATTER_ALPHA = 0.35
_BACKGROUND_ALPHA = 0.35
_BAND_ALPHA = 0.18


# ------------------------------------------------------------------- style
def apply_style(*, backend: str | None = "Agg", **overrides) -> dict:
    """Apply the manuscript rcParams. Notebooks call this; importing does not.

    Returns the parameters that were set, so a caller can restore them.
    """
    if backend is not None:
        matplotlib.use(backend, force=False)
    params = {
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.3,
        "lines.markersize": _MARKER_SIZE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "image.cmap": "Greys",
    }
    params.update(overrides)
    matplotlib.rcParams.update(params)
    return params


def _new_figure(width: float, height: float) -> Figure:
    """A Figure with an Agg canvas, built without touching pyplot's registry."""
    figure = Figure(figsize=(width, height), constrained_layout=True)
    FigureCanvasAgg(figure)
    return figure


# ------------------------------------------------------- configuration input
def load_plot_config(path) -> dict:
    """Read a figure specification such as ``configs/plots/manuscript.yaml``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"plot configuration not found: {path}")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"plot configuration {path} is not a mapping")
    return config


def load_registry(config_dir=DEFAULT_CONFIG_DIR) -> dict:
    """Read ``registry.yaml``: the one source of method identity and style."""
    path = Path(config_dir) / "registry.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"registry not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def figure_spec(plot_config: dict, experiment: str, figure: str) -> dict:
    """One figure's specification with the file-level defaults folded in."""
    if experiment not in plot_config:
        raise KeyError(
            f"plot configuration has no experiment {experiment!r}; it defines "
            f"{sorted(k for k in plot_config if isinstance(plot_config[k], dict) and 'figures' in plot_config[k])}")
    figures = plot_config[experiment].get("figures") or {}
    if figure not in figures:
        raise KeyError(f"{experiment} defines no figure {figure!r}; it defines "
                       f"{sorted(figures)}")
    spec = dict(figures[figure])
    spec.setdefault("name", figure)
    spec.setdefault("experiment", experiment)
    spec.setdefault("experiment_key",
                    plot_config[experiment].get("experiment_key"))
    for key, value in (plot_config.get("defaults") or {}).items():
        spec.setdefault(key, value)
    return spec


# ---------------------------------------------------------------- csv input
def _column_array(values) -> np.ndarray:
    """One CSV column as a typed numpy array. No pandas: it may be absent."""
    cleaned = ["" if value is None else str(value).strip() for value in values]
    tokens = {value.lower() for value in cleaned}
    if (tokens - {""}) and not (tokens - {"", "true", "false"}):
        return np.array([value.lower() == "true" for value in cleaned],
                        dtype=bool)
    try:
        return np.array([np.nan if value == "" else float(value)
                         for value in cleaned], dtype=float)
    except ValueError:
        return np.array(cleaned, dtype=object)


def _read_csv_columns(path: Path) -> dict:
    """Read a saved timeseries CSV into a dict of numpy arrays."""
    if not path.is_file():
        raise FileNotFoundError(f"missing saved timeseries: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        names = list(reader.fieldnames or [])
        raw = {name: [] for name in names}
        for row in reader:
            for name in names:
                raw[name].append(row.get(name))
    return {name: _column_array(values) for name, values in raw.items()}


# ------------------------------------------------------------------ run data
@dataclass
class Snapshot:
    """One saved ``sample_snapshots/checkpoint_*.npz``, exactly as written.

    ``t``, ``checkpoint_step``, and ``n_fee`` are the REALISED values of the
    checkpoint that was saved, never the time a figure asked for.
    """

    run_id: str
    method: str
    variant_label: str
    path: Path
    requested_time: float
    t: float
    checkpoint_step: int
    n_fee: float
    n_fee_per_particle: float
    arrays: dict = field(repr=False)

    def coordinates(self, name: str = "x") -> np.ndarray:
        if name not in self.arrays:
            raise KeyError(f"{self.path} saved no array {name!r}; it saved "
                           f"{sorted(self.arrays)}")
        return np.asarray(self.arrays[name])

    def describe(self) -> dict:
        return json_safe({
            "run_id": self.run_id, "method": self.method,
            "variant_label": self.variant_label,
            "requested_time": self.requested_time, "realised_t": self.t,
            "checkpoint_step": self.checkpoint_step, "n_fee": self.n_fee,
            "n_fee_per_particle": self.n_fee_per_particle,
            "source": str(self.path),
        })


@dataclass
class RunData:
    """One saved run directory, read once and held in memory.

    Snapshots and terminal samples load lazily: a curve figure never opens a
    sample file at all.
    """

    run_dir: Path
    manifest: dict
    metrics: dict = field(repr=False)
    cost: dict = field(repr=False)
    resolved_config: dict = field(repr=False)
    _snapshot_cache: dict = field(default_factory=dict, repr=False)
    _terminal: dict = field(default=None, repr=False)

    # -- identity ---------------------------------------------------------
    @property
    def run_id(self) -> str:
        return str(self.manifest.get("run_id", self.run_dir.name))

    @property
    def method(self) -> str:
        return str(self.manifest.get("method"))

    @property
    def variant_label(self) -> str:
        return str(self.manifest.get("variant_label", self.method))

    @property
    def variant_hash(self) -> str:
        return str(self.manifest.get("variant_hash", ""))

    @property
    def parameters(self) -> dict:
        return dict(self.manifest.get("parameters") or {})

    @property
    def tame(self) -> bool:
        return bool(self.parameters.get("tame",
                                        self.manifest.get("tame", False)))

    @property
    def seeds(self) -> tuple:
        return tuple(self.manifest.get("seeds") or ())

    @property
    def fee_calibration_hash(self):
        value = self.manifest.get("fee_calibration_hash")
        return None if value is None else str(value)

    @property
    def fee_cost_unit(self):
        return self.manifest.get("fee_cost_unit")

    @property
    def experiment_dir(self) -> Path:
        """``<results root>/<experiment key>``, from the run's own location.

        The layout is fixed by ``src.results.RunPaths``:
        ``<experiment dir>/runs/<method>/<run id>``.
        """
        return self.run_dir.parents[2]

    @property
    def reference_dir(self) -> Path:
        return self.experiment_dir / "reference"

    # -- lazy sample accessors --------------------------------------------
    def snapshot_paths(self) -> list:
        directory = self.run_dir / "sample_snapshots"
        if not directory.is_dir():
            return []
        return sorted(directory.glob("checkpoint_*.npz"))

    def snapshot_times(self) -> list:
        return [float(self._snapshot_payload(path)["t"])
                for path in self.snapshot_paths()]

    def _snapshot_payload(self, path: Path) -> dict:
        key = str(path)
        if key not in self._snapshot_cache:
            with np.load(path, allow_pickle=False) as handle:
                self._snapshot_cache[key] = {name: handle[name]
                                             for name in handle.files}
        return self._snapshot_cache[key]

    def snapshot(self, t, *, policy: str = "nearest_below") -> Snapshot:
        """The saved snapshot matched to simulation time ``t``."""
        return select_snapshot(self, t, policy=policy)

    def terminal(self) -> dict:
        """The saved ``terminal_samples.npz`` as a dict of arrays."""
        if self._terminal is None:
            path = self.run_dir / "terminal_samples.npz"
            if not path.is_file():
                raise FileNotFoundError(f"missing terminal samples: {path}")
            with np.load(path, allow_pickle=False) as handle:
                self._terminal = {name: handle[name] for name in handle.files}
        return self._terminal

    def has_extra_potential(self) -> bool:
        """Whether this run ever paid an LSC score potential evaluation."""
        for source, column in (
                (self.cost, "n_extra_potential_equivalent"),
                (self.cost, "n_extra_potential_equivalent_per_particle"),
                (self.metrics, "n_extra_potential_equivalent_per_particle")):
            values = source.get(column)
            if values is None:
                continue
            values = np.asarray(values, dtype=float)
            if values.size and np.nanmax(np.abs(values)) > 0.0:
                return True
        return False


def _read_run(run_dir) -> RunData:
    run_dir = Path(run_dir)
    manifest = read_manifest(run_dir)
    if manifest is None:
        raise FileNotFoundError("missing or unreadable manifest: "
                                f"{run_dir / 'manifest.json'}")
    if not (run_dir / "COMPLETE").is_file():
        raise FileNotFoundError(f"run is not marked COMPLETE: {run_dir}")
    resolved = {}
    resolved_path = run_dir / "resolved_config.yaml"
    if resolved_path.is_file():
        with resolved_path.open(encoding="utf-8") as handle:
            resolved = yaml.safe_load(handle) or {}
    return RunData(
        run_dir=run_dir,
        manifest=manifest,
        metrics=_read_csv_columns(run_dir / "metrics_timeseries.csv"),
        cost=_read_csv_columns(run_dir / "cost_timeseries.csv"),
        resolved_config=resolved,
    )


def _matches_parameters(row: dict, parameters: dict) -> bool:
    for key, wanted in (parameters or {}).items():
        if key == "tame":
            got = str(row.get("tame", "")).strip().lower() in ("true", "1",
                                                               "yes")
            if bool(got) != bool(wanted):
                return False
            continue
        got = row.get(f"param_{key}")
        if got is None or str(got) == "":
            return False
        try:
            if not math.isclose(float(got), float(wanted), rel_tol=1e-9,
                                abs_tol=1e-12):
                return False
        except (TypeError, ValueError):
            if str(got) != str(wanted):
                return False
    return True


def _variant_filter(row: dict, variants) -> bool:
    """Apply the plot config's ``variants:`` block to one catalog row."""
    if variants is None:
        return True
    if isinstance(variants, dict):
        wanted = variants.get(row.get("method"))
        if wanted is None:
            return False
        return any(_matches_parameters(row, entry or {}) for entry in wanted)
    labels = [item for item in variants if isinstance(item, str)]
    dicts = [item for item in variants if isinstance(item, dict)]
    if labels and row.get("variant_label") in labels:
        return True
    return any(_matches_parameters(row, entry) for entry in dicts)


def _spec_methods(spec: dict):
    """The method list a figure specification asks for.

    Different figure kinds name their methods in different places: a metric
    grid at the top level, a snapshot matrix under ``rows``, and the twin-axis
    CDF under ``left_axis`` because its right axis carries the potential rather
    than a method. Missing any of these would silently load every run in the
    directory and draw methods the figure never asked for.
    """
    for key in ("methods", "rows"):
        if spec.get(key):
            return list(spec[key])
    for section in ("left_axis", "right_axis"):
        nested = spec.get(section) or {}
        if nested.get("methods"):
            return list(nested["methods"])
    for panel in spec.get("panels") or ():
        if isinstance(panel, dict) and panel.get("methods"):
            return list(panel["methods"])
    return None


class RunSelection(list):
    """The loaded runs, plus the methods that were requested but uncalibratable.

    A plain list of runs everywhere it is used, so nothing downstream has to
    know about this type. The extra attribute carries the negative outcomes so
    a figure can say a method was uncalibratable instead of quietly omitting it.
    """

    def __init__(self, runs=(), *, uncalibratable=None):
        super().__init__(runs)
        self.uncalibratable = dict(uncalibratable or {})


def load_runs(experiment_dir, spec=None, *, methods=_UNSET, variants=_UNSET,
              tame=_UNSET, latest_only=_UNSET) -> list:
    """Load every completed run of one experiment that passes the filters.

    A figure specification may be passed positionally, which is what the plot
    notebooks do: ``methods``, ``variants``, and the tame filter implied by
    ``tame_view`` are then read from the spec. An explicit keyword argument
    always wins over the spec.

    Manifests are scanned directly rather than through ``catalog.csv``: this
    module is read-only and must not create or refresh a derived index.

    ``variants`` may be a list of variant labels, a list of parameter dicts, or
    the plot config's ``{method: [parameters, ...]}`` mapping.
    """
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.is_dir():
        raise FileNotFoundError(
            f"experiment directory not found: {experiment_dir}")
    spec = dict(spec or {})
    if methods is _UNSET:
        methods = _spec_methods(spec)
    if variants is _UNSET:
        variants = spec.get("variants")
    if tame is _UNSET:
        view = spec.get("tame_view")
        tame = None if view is None else tame_view_filters(view).tame
    if latest_only is _UNSET:
        latest_only = bool(spec.get("latest_run_only", True))
    if methods is None:
        rows = select_runs(experiment_dir, tame=tame, latest_only=latest_only,
                           from_manifests=True)
    else:
        rows = []
        for method in methods:
            rows.extend(select_runs(experiment_dir, method=method, tame=tame,
                                    latest_only=latest_only,
                                    from_manifests=True))
    rows = [row for row in rows if _variant_filter(row, variants)]
    uncalibratable: dict[str, list] = {}
    if methods is not None:
        loaded_methods = {str(row.get("method")) for row in rows}
        missing_methods = [name for name in methods
                           if name not in loaded_methods]
        # A method with no completed run is only tolerated when the campaign
        # actually recorded it as uncalibratable under the same filters. That
        # is a result about the method and belongs on the figure. A method that
        # was simply never run is still an error: silently dropping it would
        # turn a misconfigured plot specification into a plausible figure.
        still_missing = []
        for name in missing_methods:
            negative = [
                row for row in select_runs(
                    experiment_dir, method=name, tame=tame,
                    status="uncalibratable", latest_only=latest_only,
                    from_manifests=True)
                if _variant_filter(row, variants)]
            if negative:
                # The scan projection does not carry the diagnosis, so read it
                # from the manifest the row points at. Still read-only, and
                # still no derived index is created or refreshed.
                entries = []
                for row in negative:
                    try:
                        manifest = read_manifest(Path(row["run_directory"]))
                    except Exception:                         # noqa: BLE001
                        manifest = {}
                    entries.append({
                        "variant_label": row.get("variant_label"),
                        "diagnosis": (manifest.get("diagnosis")
                                      or manifest.get("calibration_kind")
                                      or "uncalibratable")})
                uncalibratable[name] = entries
            else:
                still_missing.append(name)
        if still_missing:
            raise ValueError(
                f"plot specification requires missing methods {still_missing}; "
                f"loaded {sorted(loaded_methods)} after tame/variant filters. "
                "No uncalibratable outcome was recorded for them either, so "
                "this is a specification error rather than a result.")
    if not rows:
        raise ValueError(
            "no completed runs matched: experiment_dir="
            f"{experiment_dir}, methods={methods}, variants={variants}, "
            f"tame={tame}, latest_only={latest_only}. Check that the runs "
            "finished (COMPLETE marker present) and that the method names "
            "match configs/registry.yaml.")
    order = {name: index for index, name in enumerate(methods or [])}
    runs = [_read_run(row["run_directory"]) for row in rows]
    runs.sort(key=lambda run: (order.get(run.method, len(order)), run.method,
                               run.variant_label, run.tame))
    return RunSelection(runs, uncalibratable=uncalibratable)


# ------------------------------------------------------- reference artifacts
class _MissingArtifact:
    """Placeholder for a reference file that is not on disk.

    Held in the artifact dict so a figure that needs the value fails naming the
    file, while a figure that does not need it still draws.
    """

    def __init__(self, message: str) -> None:
        self.message = message

    def __repr__(self) -> str:                       # pragma: no cover
        return f"<missing: {self.message}>"


def _unwrap_artifact(value, what: str):
    if isinstance(value, _MissingArtifact):
        raise FileNotFoundError(f"{what}: {value.message}")
    return value


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as handle:
        return {name: handle[name] for name in handle.files}


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


_REFERENCE_CACHE: dict = {}


def load_reference_artifacts(experiment_dir, *, refresh: bool = False) -> dict:
    """Read one experiment's saved reference directory.

    Plain numpy and json reads only: the reference builders are never imported,
    so this stays inside the module's import rule. Files that are absent become
    :class:`_MissingArtifact` placeholders, so a figure that needs one fails
    naming the file instead of drawing nothing.
    """
    experiment_dir = Path(experiment_dir)
    key = str(experiment_dir.resolve())
    if not refresh and key in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[key]
    directory = experiment_dir / "reference"
    if not directory.is_dir():
        raise FileNotFoundError(
            f"no saved reference directory at {directory}; the run notebook "
            "builds it, and the plot notebook only reads it")

    record = _load_json(directory / "reference.json") \
        if (directory / "reference.json").is_file() else {}
    experiment_id = str(record.get("experiment_id")
                        or experiment_dir.name.split("_")[0])
    artifacts = {
        "experiment_id": experiment_id,
        "reference_dir": str(directory),
        "record": record,
        "files": sorted(path.name for path in directory.iterdir()
                        if path.is_file()),
    }
    builder = {"E1": _reference_e1, "E2": _reference_e2, "E3": _reference_e3,
               "E4": _reference_e4}.get(experiment_id)
    if builder is None:
        raise KeyError(
            f"no reference reader for experiment {experiment_id!r}; "
            f"{directory} holds {artifacts['files']}")
    builder(directory, record, artifacts)
    _REFERENCE_CACHE[key] = artifacts
    return artifacts


def _missing(directory: Path, name: str) -> _MissingArtifact:
    return _MissingArtifact(f"{directory / name} is not on disk")


def _reference_e1(directory: Path, record: dict, out: dict) -> None:
    """E1: the inverse-CDF grid, the potential behind it, and the W2 floor."""
    grid_file = directory / "reference_grid.npz"
    if grid_file.is_file():
        arrays = _load_npz(grid_file)
        x = np.asarray(arrays["grid"], dtype=float).ravel()
        out["cdf"] = {"x": x, "F": np.asarray(arrays["cdf"], dtype=float).ravel()}
        out["pdf"] = {"x": x, "p": np.asarray(arrays["pdf"], dtype=float).ravel()}
        beta = float((record.get("provenance") or {}).get("beta", 1.0))
        # The saved reference density is exp(-beta V) normalised over the box,
        # so the potential behind the CDFs is a monotone transform of a saved
        # array, up to the additive normalisation constant. No potential is
        # evaluated here.
        density = out["pdf"]["p"]
        with np.errstate(divide="ignore"):
            potential = -np.log(np.where(density > 0.0, density, np.nan)) / beta
        out["potential"] = {"x": x, "V": potential - np.nanmin(potential),
                            "source": "saved reference density"}
    else:
        for key in ("cdf", "pdf", "potential"):
            out[key] = _missing(directory, "reference_grid.npz")

    validation_file = directory / "reference_validation.json"
    if validation_file.is_file():
        validation = _load_json(validation_file)
        out["validation"] = validation
        saved_floors = validation.get("sampling_floors") or {}
        floor_entries = {}
        for metric in ("W2_exact_1d", "MMD2_biased", "KS"):
            record = saved_floors.get(metric) or {}
            if "mean" not in record:
                continue
            mean = float(record["mean"])
            sd = float(record.get("sd", 0.0))
            floor_entries[metric] = {
                "lo": mean - sd, "hi": mean + sd,
                "mean": mean, "sd": sd,
                "source": str(validation_file),
            }
        # Compatibility with pre-generalisation E1 references.
        self_w2 = validation.get("self_w2") or {}
        if "W2_exact_1d" not in floor_entries and "mean" in self_w2:
            mean, sd = float(self_w2["mean"]), float(self_w2.get("sd", 0.0))
            floor_entries["W2_exact_1d"] = {
                "lo": mean - sd, "hi": mean + sd,
                "mean": mean, "sd": sd,
                "source": str(validation_file),
            }
        if floor_entries:
            out["sampling_floor"] = floor_entries
        else:
            out["sampling_floor"] = _MissingArtifact(
                f"{validation_file} records no primary-metric sampling floors")
    else:
        out["validation"] = _missing(directory, "reference_validation.json")
        out["sampling_floor"] = _missing(directory,
                                         "reference_validation.json")


def _reference_e2(directory: Path, record: dict, out: dict) -> None:
    """E2: frozen descriptor masses, EMC*, and a background from the bank."""
    masses_file = directory / "descriptor_masses.npz"
    if masses_file.is_file():
        arrays = _load_npz(masses_file)
        p_star = np.asarray(arrays["p_star"], dtype=float).ravel()
        out["p_star"] = p_star
        # One fixed mode order, taken from the frozen reference masses and used
        # unchanged for every method.
        out["mode_order"] = [f"{index:03d}"
                             for index in np.argsort(-p_star, kind="stable")]
        out["mode_order_rule"] = "descending reference descriptor mass p*"
    else:
        out["p_star"] = _missing(directory, "descriptor_masses.npz")
        out["mode_order"] = None

    emc = record.get("emc_star")
    if emc is None:
        diagnostics = directory / "diagnostics.json"
        if diagnostics.is_file():
            emc = _load_json(diagnostics).get("emc_star")
    out["emc_star"] = (float(emc) if emc is not None
                       else _MissingArtifact(
                           f"neither {directory / 'reference.json'} nor "
                           f"{directory / 'diagnostics.json'} records emc_star"))

    bank_file = directory / "reference_samples.npz"
    if bank_file.is_file():
        bank = np.asarray(_load_npz(bank_file)["sample_bank"], dtype=float)
        out["background"] = _density_background(
            bank, label="exact reference log-density",
            source=str(bank_file))
    else:
        out["background"] = _missing(directory, "reference_samples.npz")


def _reference_e3(directory: Path, record: dict, out: dict) -> None:
    """E3: the CV grid and the reference free-energy surface on it."""
    grid_file = directory / "cv_grid.npz"
    fes_file = directory / "fes_grid.npz"
    if grid_file.is_file() and fes_file.is_file():
        grid = _load_npz(grid_file)
        axis_1 = np.asarray(grid["axis_1"], dtype=float).ravel()
        axis_2 = np.asarray(grid["axis_2"], dtype=float).ravel()
        # The surfaces are stored with "ij" indexing (axis_1 first), while a
        # contour plot wants (len(y), len(x)).
        out["background"] = {
            "x": axis_1, "y": axis_2,
            "z": np.asarray(_load_npz(fes_file)["fes_grid"], dtype=float).T,
            "label": "reference CV free-energy surface",
            "is_estimate": False, "source": str(fes_file)}
    else:
        out["background"] = _missing(
            directory, "cv_grid.npz / fes_grid.npz")
    density_file = directory / "density_grid.npz"
    out["density_grid"] = (
        np.asarray(_load_npz(density_file)["density_grid"], dtype=float).T
        if density_file.is_file() else _missing(directory, "density_grid.npz"))
    diagnostics = directory / "diagnostics.json"
    out["validation"] = (_load_json(diagnostics).get("validation", {})
                         if diagnostics.is_file()
                         else _missing(directory, "diagnostics.json"))


def _reference_e4(directory: Path, record: dict, out: dict) -> None:
    """E4: the order-parameter free-energy estimate and the validation record."""
    grid_file = directory / "reference_order_parameter_grid.npz"
    if grid_file.is_file():
        grid = _load_npz(grid_file)
        out["background"] = {
            "x": np.asarray(grid["m_x_centers"], dtype=float).ravel(),
            "y": np.asarray(grid["m_y_centers"], dtype=float).ravel(),
            "z": np.asarray(grid["free_energy"], dtype=float).T,
            "label": "reference free-energy estimate",
            # A binned estimate from the PT bank, not an exact surface.
            "is_estimate": True, "source": str(grid_file)}
        out["p_star"] = np.asarray(grid["p_star"], dtype=float)
    else:
        out["background"] = _missing(
            directory, "reference_order_parameter_grid.npz")
        out["p_star"] = _missing(directory,
                                 "reference_order_parameter_grid.npz")
    validation_file = directory / "reference_validation.json"
    out["validation"] = (_load_json(validation_file)
                         if validation_file.is_file()
                         else _missing(directory, "reference_validation.json"))


def _density_background(samples, *, label: str, source: str,
                        bins: int = 120) -> dict:
    """Display-only log-density surface of a saved exact reference bank.

    A binned rendering of samples that were already saved. It is background
    decoration and never stands in for a metric.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] < 2:
        raise ValueError("a density background needs (N, 2) saved samples, got "
                         f"shape {samples.shape}")
    counts, x_edges, y_edges = np.histogram2d(samples[:, 0], samples[:, 1],
                                              bins=bins, density=True)
    with np.errstate(divide="ignore"):
        z = np.log(np.where(counts > 0.0, counts, np.nan))
    floor = np.nanpercentile(z, 1.0)
    return {"x": 0.5 * (x_edges[1:] + x_edges[:-1]),
            "y": 0.5 * (y_edges[1:] + y_edges[:-1]),
            "z": np.nan_to_num(z, nan=floor).T,
            "label": label, "is_estimate": True, "source": source}


def _resolve_reference(runs, reference):
    """The caller's reference, or the one saved next to these runs."""
    if reference is not None:
        return dict(reference)
    return load_reference_artifacts(runs[0].experiment_dir)


def _resolve_background(runs, spec, background, reference):
    """The saved background surface named by the spec."""
    if background is not None:
        return dict(background)
    artifacts = _resolve_reference(runs, reference)
    surface = artifacts.get("background")
    if surface is None:
        raise KeyError(
            f"the saved reference at {artifacts.get('reference_dir')} carries "
            f"no background surface for {spec.get('background', 'this figure')}")
    return dict(_unwrap_artifact(
        surface, f"figure {spec.get('name', spec.get('kind'))} needs the "
                 f"background {spec.get('background', '')}".strip()))


# --------------------------------------------------------------- run styling
def _guard_display_text(text) -> str:
    """Bar the integrator name from any reader-visible string.

    ULD is the method; BAOAB is only its integrator and must never reach a
    legend, a tick label, or a title.
    """
    if _FORBIDDEN_DISPLAY_TOKEN.lower() in str(text).lower():
        raise ValueError(
            f"{_FORBIDDEN_DISPLAY_TOKEN!r} reached a reader-visible string "
            f"({text!r}). BAOAB is the integrator; the method is ULD. Fix the "
            "label at its source (configs/registry.yaml display_name).")
    return str(text)


def _display_name(registry: dict, method: str) -> str:
    """Registry display name for a method, with the integrator name barred."""
    entry = (registry.get("methods") or {}).get(method)
    if entry is None:
        raise KeyError(f"registry has no method {method!r}; it knows "
                       f"{sorted(registry.get('methods') or {})}")
    return _guard_display_text(entry.get("display_name", method))


def _format_number(value):
    return f"{value:g}" if isinstance(value, float) else value


def _variant_label(registry: dict, method: str, parameters: dict) -> str:
    """Legend label for one variant, following the registry's own rules."""
    entry = registry["methods"][method]
    base = _display_name(registry, method)
    template = entry.get("variant_label_template")
    defaults = entry.get("variant_label_default_when") or {}
    hyperparameters = {key: value for key, value in (parameters or {}).items()
                       if key != "tame"}
    if template is not None:
        if not all(hyperparameters.get(key) == value
                   for key, value in defaults.items()):
            base = template.format(**hyperparameters)
    else:
        extras = ", ".join(f"{key}={_format_number(value)}"
                           for key, value in sorted(hyperparameters.items()))
        if extras:
            base = f"{base} {extras}"
    tame = bool((parameters or {}).get("tame"))
    return f"{base}, {'tamed' if tame else 'canonical'}"


def method_style(registry: dict, method: str, *, tame: bool,
                 hyperparameter_index: int = 0, parameters=None,
                 variant_label=None) -> dict:
    """Colour, line style, marker, and legend label for one variant.

    The colour belongs to the method and is identical in every experiment; the
    tame flag chooses the line style; the marker cycles through
    ``style.hyperparameter_markers`` so several hyperparameter values of one
    method stay one colour. The label is the variant label, hyperparameter
    value included.
    """
    entry = (registry.get("methods") or {}).get(method)
    if entry is None:
        raise KeyError(f"registry has no method {method!r}; it knows "
                       f"{sorted(registry.get('methods') or {})}")
    style = registry.get("style") or {}
    markers = list(style.get("hyperparameter_markers")
                   or [entry.get("marker", "o")])
    if variant_label is not None:
        label = _guard_display_text(variant_label)
    elif parameters is not None:
        label = _guard_display_text(
            _variant_label(registry, method, {**parameters, "tame": tame}))
    else:
        label = _guard_display_text(
            f"{_display_name(registry, method)}, "
            f"{'tamed' if tame else 'canonical'}")
    return {
        "color": entry.get("color", "#333333"),
        "linestyle": (style.get("tamed_linestyle", "--") if tame
                      else style.get("canonical_linestyle", "-")),
        "marker": markers[int(hyperparameter_index) % len(markers)],
        "label": label,
    }


def _hyperparameter_key(run: RunData) -> tuple:
    return tuple(sorted((key, str(value))
                        for key, value in run.parameters.items()
                        if key != "tame"))


def _hyperparameter_indices(runs) -> dict:
    """Stable marker index per variant, ordered by hyperparameter value.

    Canonical and tamed runs of the same hyperparameters share an index, so a
    paired view shows one marker per hyperparameter value rather than two.
    """
    by_method = {}
    for run in runs:
        keys = by_method.setdefault(run.method, [])
        key = _hyperparameter_key(run)
        if key not in keys:
            keys.append(key)
    for keys in by_method.values():
        keys.sort()
    return {run.run_id: by_method[run.method].index(_hyperparameter_key(run))
            for run in runs}


def _run_style(registry: dict, run: RunData, indices: dict) -> dict:
    return method_style(registry, run.method, tame=run.tame,
                        hyperparameter_index=indices.get(run.run_id, 0),
                        variant_label=run.variant_label)


def tame_view_filters(view: str):
    """The run filter for one tame view.

    Returns a predicate over :class:`RunData`. Its ``.tame`` attribute is the
    matching ``load_runs(tame=...)`` argument (``None`` for the paired view).
    """
    wanted = {"canonical_only": False, "tamed_only": True,
              "paired": None}.get(view, "missing")
    if wanted == "missing":
        raise ValueError(f"unknown tame view {view!r}; expected one of "
                         "canonical_only, tamed_only, paired")

    def predicate(run) -> bool:
        return wanted is None or bool(run.tame) is wanted

    predicate.tame = wanted
    predicate.view = view
    return predicate


# ------------------------------------------------------------ seed statistics
def seed_aggregate(rows, metric: str, x_column: str, *, uncertainty=None):
    """Aggregate one metric across seeds; return ``(x, centre, lo, hi)``.

    The statistical unit is the SEED. Particles inside one seed share an
    initial ensemble and a random stream, so they are not independent
    experiment repeats and are never treated as such: the run has already
    reduced each seed's particles to one number per checkpoint, and this
    function resamples those per-seed numbers.

    The interval is the seed bootstrap of ``defaults.uncertainty``: seeds are
    resampled with replacement ``bootstrap_replicates`` times from a frozen
    ``bootstrap_seed``, and the band is the percentile interval of the
    resampled seed means. One resample matrix is drawn and reused at every x,
    so the band is coherent along a curve and identical between runs.
    """
    columns = rows.metrics if isinstance(rows, RunData) else rows
    if not columns:
        raise ValueError("seed_aggregate got no rows; nothing to aggregate")
    for name in (metric, x_column, "seed"):
        if name not in columns:
            raise KeyError(f"saved rows have no column {name!r}; they have "
                           f"{sorted(columns)}")
    settings = dict(DEFAULT_UNCERTAINTY)
    settings.update(uncertainty or {})
    if str(settings.get("unit", "seed")) != "seed":
        raise ValueError(
            f"uncertainty unit is {settings.get('unit')!r}; the seed is the "
            "only admissible statistical unit")

    x_all = np.asarray(columns[x_column], dtype=float)
    y_all = np.asarray(columns[metric], dtype=float)
    seed_all = np.asarray(columns["seed"])
    finite = np.isfinite(x_all)
    if not finite.any():
        raise ValueError(f"column {x_column!r} holds no finite values")
    seeds = np.unique(seed_all[finite])
    xs = np.unique(x_all[finite])
    x_index = {value: position for position, value in enumerate(xs)}
    seed_index = {value: position for position, value in enumerate(seeds)}
    table = np.full((xs.size, seeds.size), np.nan)
    for position in np.flatnonzero(finite):
        table[x_index[x_all[position]],
              seed_index[seed_all[position]]] = y_all[position]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmean(table, axis=1)
        estimator = str(settings.get("estimator", "seed_bootstrap"))
        if estimator == "none" or seeds.size < 2:
            return xs, centre, centre.copy(), centre.copy()
        if estimator != "seed_bootstrap":
            raise ValueError(f"unknown uncertainty estimator {estimator!r}; "
                             "this module implements seed_bootstrap only")
        replicates = int(settings.get("bootstrap_replicates", 2000))
        generator = np.random.default_rng(
            int(settings.get("bootstrap_seed", 0)))
        draw = generator.integers(0, seeds.size, size=(replicates, seeds.size))
        resampled = np.nanmean(table[:, draw], axis=2)
        tail = 100.0 * (1.0 - float(settings.get("interval", 0.95))) / 2.0
        lo = np.nanpercentile(resampled, tail, axis=1)
        hi = np.nanpercentile(resampled, 100.0 - tail, axis=1)
    return xs, centre, lo, hi


# ------------------------------------------------------------------- gates
def check_fee_comparability(runs, spec=None) -> str:
    """Refuse to share a FEE axis between runs calibrated differently.

    A force-equivalent cost is a common currency only when every run priced its
    oracles with the same calibration. Runs with different
    ``fee_calibration_hash`` may share an axis only when the spec supplies a
    compatibility record that names exactly those hashes and asserts both
    ``cost_unit_comparable`` and ``workload_comparable``.
    """
    runs = list(runs or [])
    if not runs:
        raise ValueError("check_fee_comparability got no runs; a FEE axis "
                         "needs at least one run with a fee_calibration_hash")
    hashes = {}
    for run in runs:
        value = run.fee_calibration_hash
        if value is None:
            raise ValueError(
                f"run {run.run_id} ({run.variant_label}) has no "
                "fee_calibration_hash in its manifest; it cannot go on a FEE "
                "axis")
        hashes.setdefault(value, []).append(run.variant_label)
    if len(hashes) == 1:
        return next(iter(hashes))

    record = None
    for holder in ((spec or {}).get("fee_axis") or {}, spec or {}):
        if isinstance(holder, dict) and holder.get("compatibility_record"):
            record = holder["compatibility_record"]
            break
    detail = "; ".join(f"{key}: {sorted(set(value))}"
                       for key, value in sorted(hashes.items()))
    if not isinstance(record, dict):
        raise ValueError(
            "runs on one FEE axis have different fee_calibration_hash values "
            f"({detail}) and the spec supplies no compatibility record. Either "
            "recalibrate onto one hash, or add fee_axis.compatibility_record "
            "naming these hashes and asserting cost_unit_comparable and "
            "workload_comparable.")
    named = set(map(str, record.get("hashes")
                    or record.get("calibration_hashes") or ()))
    if named != set(hashes):
        raise ValueError(
            f"the FEE compatibility record names {sorted(named)} but the runs "
            f"carry {sorted(hashes)}; a record must name exactly the hashes it "
            "certifies")
    for claim in ("cost_unit_comparable", "workload_comparable"):
        if record.get(claim) is not True:
            raise ValueError(
                f"the FEE compatibility record does not assert {claim}; "
                "without it these runs do not share a cost axis")
    return "|".join(sorted(hashes))


def check_extra_potential_eligibility(runs) -> list:
    """The extra-potential axis is LSC-only.

    A method that never evaluates the LSC score potential has an all-zero
    counter, so putting it on this axis would stack its whole curve on x = 0
    and invite a comparison that does not exist.
    """
    runs = list(runs or [])
    if not runs:
        raise ValueError("check_extra_potential_eligibility got no runs; the "
                         "extra-potential axis needs at least one LSC run")
    ineligible = sorted({f"{run.method} ({run.variant_label})" for run in runs
                         if not run.has_extra_potential()})
    if ineligible:
        raise ValueError(
            "the extra-potential axis counts LSC score potential evaluations "
            "only, but these runs have an all-zero "
            f"n_extra_potential_equivalent counter: {ineligible}. Restrict the "
            "figure to the LSC estimators.")
    return runs


def _check_extra_potential_label(label: str) -> str:
    lowered = str(label).lower()
    for claim in _FULL_COST_CLAIMS:
        if claim in lowered:
            raise ValueError(
                f"the extra-potential axis label {label!r} claims a full "
                "computational cost. It counts LSC score potential evaluations "
                "only.")
    return str(label)


# ---------------------------------------------------------------- snapshots
def select_snapshot(run: RunData, requested_time, *,
                    policy: str = "nearest_below") -> Snapshot:
    """The saved snapshot matched to ``requested_time`` by simulation time.

    Nothing is interpolated in time or in budget: the returned object carries
    the realised ``t``, ``checkpoint_step``, and ``n_fee`` of the checkpoint
    that was actually written.
    """
    if policy not in ("nearest_below", "exact"):
        raise ValueError(f"unknown snapshot policy {policy!r}; expected "
                         "nearest_below or exact")
    paths = run.snapshot_paths()
    if not paths:
        raise FileNotFoundError(
            f"run {run.run_id} ({run.variant_label}) saved no snapshots under "
            f"{run.run_dir / 'sample_snapshots'}")
    requested = float(requested_time)
    best = None
    for path in paths:
        payload = run._snapshot_payload(path)
        t = float(payload["t"])
        if policy == "exact":
            if math.isclose(t, requested, rel_tol=1e-9, abs_tol=1e-9):
                best = (t, path, payload)
                break
            continue
        if t <= requested + 1e-9 and (best is None or t > best[0]):
            best = (t, path, payload)
    if best is None:
        available = sorted(float(run._snapshot_payload(p)["t"]) for p in paths)
        raise ValueError(
            f"run {run.run_id} ({run.variant_label}) has no snapshot at or "
            f"below t={requested:g}; it saved {available}. Snapshots are "
            "matched by simulation time and never interpolated.")
    t, path, payload = best
    return Snapshot(
        run_id=run.run_id, method=run.method, variant_label=run.variant_label,
        path=path, requested_time=requested, t=t,
        checkpoint_step=int(payload["checkpoint_step"]),
        n_fee=float(payload["n_fee"]),
        n_fee_per_particle=float(payload["n_fee_per_particle"]),
        arrays={name: value for name, value in payload.items()
                if isinstance(value, np.ndarray)})


# ------------------------------------------------------- small-multiple rules
def points_per_panel(n_available: int, n_requested: int) -> np.ndarray:
    """Deterministic subsample indices, identical in every panel.

    Evenly spaced indices, the rule the run itself used when it downsampled a
    snapshot, so every panel shows the same number of points chosen the same
    way whatever the method.
    """
    n_available = int(n_available)
    if n_available <= 0:
        raise ValueError("points_per_panel needs at least one saved point")
    take = min(int(n_requested), n_available)
    if take <= 0:
        raise ValueError(f"points_per_panel got n_requested={n_requested}")
    return np.linspace(0, n_available - 1, take).round().astype(int)


def shared_limits(arrays, *, margin: float = 0.04):
    """Common ``(xlim, ylim)`` over every panel's points.

    Small multiples are comparable only when they share their axes, so the
    limits are computed once over all panels and applied to all of them.
    """
    clouds = []
    for item in arrays:
        if item is None:
            continue
        item = np.asarray(item, dtype=float)
        if item.size == 0:
            continue
        clouds.append(item.reshape(-1, item.shape[-1]) if item.ndim > 1
                      else item.reshape(-1, 1))
    if not clouds:
        raise ValueError("shared_limits got no points; nothing to bound")
    stacked = np.concatenate(clouds, axis=0)
    if stacked.shape[1] == 1:
        stacked = np.column_stack([stacked[:, 0], stacked[:, 0]])
    limits = []
    for axis in (0, 1):
        values = stacked[:, axis]
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError("shared_limits got no finite points")
        lo, hi = float(values.min()), float(values.max())
        pad = margin * (hi - lo) if hi > lo else max(abs(hi), 1.0) * margin
        limits.append((lo - pad, hi + pad))
    return limits[0], limits[1]


def assert_panels_consistent(panels) -> dict:
    """Refuse small multiples whose panels are not drawn the same way.

    A grid of scatter panels is a comparison only when the axes, contour
    levels, point count, marker size, alpha, and subsampling rule are identical
    everywhere.
    """
    panels = list(panels)
    if not panels:
        raise ValueError("assert_panels_consistent got no panels")
    keys = ("xlim", "ylim", "levels", "n_points", "marker_size", "alpha",
            "subsample_rule")
    reference = panels[0]
    for key in keys:
        if key not in reference:
            raise KeyError(f"panel record is missing {key!r}; it has "
                           f"{sorted(reference)}")
        expected = reference[key]
        for panel in panels[1:]:
            found = panel.get(key)
            if isinstance(expected, (list, tuple, np.ndarray)):
                same = (found is not None
                        and np.shape(expected) == np.shape(found)
                        and np.allclose(np.asarray(expected, dtype=float),
                                        np.asarray(found, dtype=float)))
            else:
                same = expected == found
            if not same:
                raise ValueError(
                    f"small multiples disagree on {key!r}: "
                    f"{reference.get('name')} has {expected!r} but "
                    f"{panel.get('name')} has {found!r}")
    return {"n_panels": len(panels),
            **{key: reference[key] for key in keys}}


# --------------------------------------------------------------- draw helpers
def _require_runs(runs, what: str) -> list:
    # Preserve the uncalibratable record across the guard: every figure helper
    # funnels through here, and losing the attribute would turn an annotated
    # negative result back into a silent omission.
    uncalibratable = getattr(runs, "uncalibratable", None)
    runs = RunSelection(runs or (), uncalibratable=uncalibratable)
    if not runs:
        raise ValueError(
            f"{what} got no runs; nothing to draw. Load runs with load_runs() "
            "and check the method, variant, and tame filters.")
    return runs


def _x_column(x_axis: str) -> str:
    if x_axis not in X_AXIS_COLUMNS:
        raise ValueError(f"unknown x axis {x_axis!r}; expected one of "
                         f"{sorted(X_AXIS_COLUMNS)}")
    return X_AXIS_COLUMNS[x_axis]


def _x_label(spec: dict, x_axis: str) -> str:
    labels = dict(DEFAULT_X_AXIS_LABELS)
    labels.update((spec or {}).get("x_axis_labels") or {})
    label = labels[x_axis]
    if x_axis == "extra_potential":
        _check_extra_potential_label(label)
    return label


def _apply_gate(runs, spec: dict, x_axis: str) -> None:
    """Every FEE axis passes the comparability gate before anything is drawn."""
    if x_axis == "fee":
        check_fee_comparability(runs, spec)
    elif x_axis == "extra_potential":
        check_extra_potential_eligibility(runs)


def _set_title(axes, text) -> None:
    axes.set_title(_guard_display_text(text))


def _figure_legend(figure: Figure, handles, labels, *, ncol: int = 4) -> None:
    if not handles:
        return
    for label in labels:
        _guard_display_text(label)
    # Variant labels carry the hyperparameter value and the tame state, so they
    # run long. A fixed column count then pushes the outer entries past the
    # figure edge and they are cropped on save. Budget columns by the widest
    # label against the figure width instead.
    longest = max(len(label) for label in labels)
    width_inches = float(figure.get_size_inches()[0])
    # 8 pt text runs about 18 characters per inch; the line handle and the
    # inter-column gap cost roughly another eight characters per entry.
    characters_per_inch = 18.0
    affordable = max(1, int(width_inches * characters_per_inch / (longest + 8)))
    figure.legend(handles, labels,
                  loc="outside lower center",
                  ncol=max(1, min(ncol, affordable, len(labels))),
                  frameon=False)


def _collect(handles: list, labels: list, handle, label: str) -> None:
    if label not in labels:
        handles.append(handle)
        labels.append(_guard_display_text(label))


def _collect_uncalibratable(handles: list, labels: list, registry: dict,
                            uncalibratable: dict) -> list:
    """Legend entries for requested methods that had no admissible timestep.

    Drawn in the method's own colour so it reads as that method, but with no
    line and an open marker, so it cannot be mistaken for a curve. The point is
    that the reader sees the method was tried and failed rather than silently
    finding it absent from the panel.
    """
    from matplotlib.lines import Line2D

    recorded = []
    for method in sorted(uncalibratable):
        entries = uncalibratable[method] or []
        try:
            colour = method_style(registry, method, tame=False)["color"]
        except KeyError:
            colour = "0.4"
        display = (registry.get("methods", {}).get(method, {})
                   .get("display_name", method))
        handle = Line2D([], [], color=colour, linestyle="none", marker="x",
                        markersize=_MARKER_SIZE, markeredgewidth=1.4)
        _collect(handles, labels, handle, f"{display}: uncalibratable")
        recorded.append({"method": method,
                         "variants": [entry.get("variant_label")
                                      for entry in entries],
                         "diagnoses": sorted({str(entry.get("diagnosis"))
                                              for entry in entries})})
    return recorded


def _reference_style(registry: dict) -> dict:
    style = registry.get("style") or {}
    return {"color": style.get("reference_color", "#000000"),
            "linestyle": style.get("reference_linestyle", "-")}


def _background_cmap(registry: dict) -> str:
    return (registry.get("style") or {}).get("target_background_cmap", "Greys")


def _empirical_cdf(values):
    """Display-only empirical CDF of saved samples. Not an official metric."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cannot draw a CDF of zero finite samples")
    ordered = np.sort(values)
    return ordered, np.arange(1, ordered.size + 1) / ordered.size


def _gaussian_kde_1d(values, grid, *, bandwidth=None):
    """Display-only fixed-bandwidth KDE of saved samples. Not a metric."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cannot draw a KDE of zero finite samples")
    if bandwidth is None:
        bandwidth = max(1.06 * float(np.std(values)) * values.size ** (-0.2),
                        1e-9)
    grid = np.asarray(grid, dtype=float)
    z = (grid[:, None] - values[None, :]) / bandwidth
    return (np.exp(-0.5 * z ** 2).sum(axis=1)
            / (values.size * bandwidth * math.sqrt(2.0 * math.pi)))


def _levels_from_background(background: dict) -> np.ndarray:
    grid_z = np.asarray(background["z"], dtype=float)
    finite = grid_z[np.isfinite(grid_z)]
    if finite.size == 0:
        raise ValueError("background surface has no finite values")
    return np.linspace(float(finite.min()), float(finite.max()), 12)


def _background_levels(background: dict) -> np.ndarray:
    supplied = (background or {}).get("levels")
    return np.asarray(supplied if supplied is not None
                      else _levels_from_background(background), dtype=float)


def _draw_background(axes, background: dict, registry: dict, *, levels=None):
    """Target or reference surface: greyscale, faint, and behind the samples."""
    for key in ("x", "y", "z"):
        if key not in (background or {}):
            raise KeyError(
                f"background is missing {key!r}; pass the saved reference "
                "surface as {'x': ..., 'y': ..., 'z': ...}")
    grid_x = np.asarray(background["x"], dtype=float)
    grid_y = np.asarray(background["y"], dtype=float)
    grid_z = np.asarray(background["z"], dtype=float)
    if grid_x.ndim == 1 and grid_y.ndim == 1:
        grid_x, grid_y = np.meshgrid(grid_x, grid_y, indexing="xy")
    levels = _background_levels(background) if levels is None else \
        np.asarray(levels, dtype=float)
    axes.contourf(grid_x, grid_y, grid_z, levels=levels,
                  cmap=_background_cmap(registry),
                  alpha=float(background.get("alpha", _BACKGROUND_ALPHA)),
                  zorder=0)
    axes.contour(grid_x, grid_y, grid_z, levels=levels, colors="0.55",
                 linewidths=0.4, alpha=0.6, zorder=1)
    return levels


def _snapshot_points(snapshot: Snapshot, coordinates: str,
                     n_points: int) -> np.ndarray:
    points = np.asarray(snapshot.coordinates(coordinates), dtype=float)
    if points.ndim == 1:
        points = points[:, None]
    return points[points_per_panel(points.shape[0], n_points)]


def _scatter(axes, points, color):
    y = points[:, 1] if points.shape[1] > 1 else points[:, 0]
    axes.scatter(points[:, 0], y, s=_SCATTER_SIZE, c=color,
                 alpha=_SCATTER_ALPHA, linewidths=0, zorder=2)


def figure_provenance(figure: Figure) -> dict:
    """What a figure actually drew: realised snapshot times, gates, panels."""
    return dict(getattr(figure, "provenance", {}) or {})


def _record_provenance(figure: Figure, payload: dict) -> None:
    figure.provenance = json_safe(payload)


# ------------------------------------------------------------ figure builders
def curve_figure(runs, spec: dict, registry: dict = None, *, x_axis=None,
                 reference=None) -> Figure:
    """``kind: metric_grid`` -- rows are metrics, columns are the x axes.

    With no ``x_axis`` the axes come from ``spec['columns']``, so one call
    produces the metric-by-axis grid the plot config describes: every official
    curve against simulation time and against force-equivalent cost. Passing
    ``x_axis`` restricts the figure to that single axis. A ``mode_metric_panel``
    spec is routed to :func:`mode_metric_panel`.
    """
    uncalibratable = dict(getattr(runs, "uncalibratable", {}) or {})
    runs = _require_runs(runs, "curve_figure")
    registry = load_registry() if registry is None else registry
    spec = _with_defaults(spec)
    kind = spec.get("kind", "metric_grid")
    if kind == "mode_metric_panel":
        return mode_metric_panel(
            RunSelection(runs, uncalibratable=uncalibratable), spec, registry,
            reference=reference)
    if kind != "metric_grid":
        raise ValueError(
            f"curve_figure draws metric_grid and mode_metric_panel specs, not "
            f"{kind!r}; use snapshot_figure() for {kind!r}")
    metrics = list(spec.get("metrics") or [])
    if not metrics:
        raise ValueError("curve_figure needs spec['metrics']; none were given")
    axes_names = ([x_axis] if x_axis is not None
                  else list(spec.get("columns") or DEFAULT_CURVE_X_AXES))
    if not axes_names:
        raise ValueError("curve_figure needs at least one x axis; spec"
                         "['columns'] is empty")

    title = spec.get("title")
    if "extra_potential" in axes_names:
        if len(axes_names) > 1:
            raise ValueError(
                "the extra-potential axis counts LSC score potential "
                "evaluations only, so it is its own figure; it cannot share a "
                f"grid with {axes_names}")
        if title not in (None, EXTRA_POTENTIAL_TITLE):
            raise ValueError("the extra-potential figure title must be "
                             f"{EXTRA_POTENTIAL_TITLE!r}, not {title!r}")
        title = EXTRA_POTENTIAL_TITLE
    for name in axes_names:
        _apply_gate(runs, spec, name)

    floors = _sampling_floors(runs, spec, reference)
    uncertainty = spec.get("uncertainty")
    indices = _hyperparameter_indices(runs)
    figure = _new_figure(3.4 * len(axes_names) + 0.6,
                         2.2 * len(metrics) + 0.9)
    axes_grid = figure.subplots(len(metrics), len(axes_names), squeeze=False)
    handles, labels, drawn = [], [], 0
    floors_drawn, floors_absent = [], []

    for row, metric in enumerate(metrics):
        metric_column = metric["column"]
        for position, x_name in enumerate(axes_names):
            axes = axes_grid[row][position]
            column = _x_column(x_name)
            for run in runs:
                if metric_column not in run.metrics:
                    continue
                style = _run_style(registry, run, indices)
                x, centre, lo, hi = seed_aggregate(run.metrics, metric_column,
                                                   column,
                                                   uncertainty=uncertainty)
                keep = np.isfinite(centre)
                if not keep.any():
                    continue
                drawn += 1
                line, = axes.plot(x[keep], centre[keep], color=style["color"],
                                  linestyle=style["linestyle"],
                                  marker=style["marker"],
                                  markevery=max(1, int(keep.sum()) // 12),
                                  markersize=_MARKER_SIZE, label=style["label"])
                axes.fill_between(x[keep], lo[keep], hi[keep],
                                  color=style["color"], alpha=_BAND_ALPHA,
                                  linewidth=0)
                _collect(handles, labels, line, style["label"])
            if floors is not None:
                (floors_drawn if _draw_sampling_floor(axes, floors,
                                                      metric_column)
                 else floors_absent).append(metric_column)
            if position == 0:
                axes.set_ylabel(_guard_display_text(
                    metric.get("label", metric_column)))
            if metric.get("log_y"):
                axes.set_yscale("log")
            if spec.get("log_x"):
                axes.set_xscale("log")
            if row == len(metrics) - 1:
                axes.set_xlabel(_x_label(spec, x_name))
    if drawn == 0:
        raise ValueError(
            "curve_figure drew nothing: none of the loaded runs carry the "
            f"columns {[metric['column'] for metric in metrics]}. The first "
            f"run saved {sorted(runs[0].metrics)}")

    if title:
        figure.suptitle(_guard_display_text(title))
    uncalibratable_record = _collect_uncalibratable(
        handles, labels, registry, uncalibratable)
    _figure_legend(figure, handles, labels)
    _record_provenance(figure, {
        "kind": "metric_grid", "x_axes": axes_names,
        "x_columns": [_x_column(name) for name in axes_names],
        "metrics": [metric["column"] for metric in metrics],
        "runs": [{"run_id": run.run_id, "variant_label": run.variant_label}
                 for run in runs],
        "fee_calibration_hashes": sorted({run.fee_calibration_hash
                                          for run in runs}),
        "sampling_floor_drawn": sorted(set(floors_drawn)),
        "sampling_floor_absent": sorted(set(floors_absent)),
        "uncalibratable": uncalibratable_record,
    })
    return figure


def _with_defaults(spec) -> dict:
    """One figure spec with the plot config's file-level defaults folded in.

    The notebooks pass the raw ``figures[...]`` entry, so the defaults have to
    be supplied here rather than in a notebook cell.
    """
    spec = dict(spec or {})
    spec.setdefault("uncertainty", DEFAULT_UNCERTAINTY)
    spec.setdefault("x_axis_labels", DEFAULT_X_AXIS_LABELS)
    spec.setdefault("snapshot_time_policy", "nearest_below")
    return spec


def _sampling_floors(runs, spec: dict, reference):
    """The saved reference-versus-reference floors, when a spec asks for them."""
    if not spec.get("show_sampling_floor"):
        return None
    floors = spec.get("sampling_floor")
    if floors is None:
        artifacts = _resolve_reference(runs, reference)
        floors = _unwrap_artifact(
            artifacts.get("sampling_floor",
                          _MissingArtifact(
                              "the saved reference records no sampling floor")),
            "this figure sets show_sampling_floor")
    if not isinstance(floors, dict) or not floors:
        raise ValueError(
            "show_sampling_floor is set but the saved reference records no "
            "reference-versus-reference floor; this module never recomputes "
            "one")
    return floors


def _draw_sampling_floor(axes, floors: dict, column: str) -> bool:
    """Grey band for one metric's saved floor. False when none was recorded."""
    entry = floors.get(column)
    if entry is None:
        return False
    if isinstance(entry, dict):
        lo, hi = float(entry["lo"]), float(entry["hi"])
    else:
        lo = hi = float(entry)
    axes.axhspan(lo, hi, color="0.75", alpha=0.35, linewidth=0, zorder=0)
    return True


def snapshot_figure(runs, spec: dict, registry: dict = None, *, reference=None,
                    background=None) -> Figure:
    """Draw whichever snapshot-based figure ``spec['kind']`` names.

    The reference artifacts and the background surface are read from the saved
    reference directory beside the runs unless the caller passes them.
    """
    runs = _require_runs(runs, "snapshot_figure")
    registry = load_registry() if registry is None else registry
    spec = _with_defaults(spec)
    kind = spec.get("kind")
    if kind == "twin_axis_cdf":
        return twin_axis_cdf_figure(runs, spec, registry,
                                    reference=reference)
    if kind == "contour_scatter_grid":
        return contour_scatter_grid(runs, spec, registry,
                                    background=background, reference=reference)
    if kind == "snapshot_matrix":
        return snapshot_matrix(runs, spec, registry, background=background,
                               reference=reference)
    if kind == "supplement_panels":
        return supplement_panels(runs, spec, registry, reference=reference,
                                 background=background)
    if kind == "mode_metric_panel":
        return mode_metric_panel(runs, spec, registry, reference=reference)
    raise ValueError(
        f"snapshot_figure does not know figure kind {kind!r}; it draws "
        "twin_axis_cdf, contour_scatter_grid, snapshot_matrix, "
        "supplement_panels, and mode_metric_panel")


def twin_axis_cdf_figure(runs, spec: dict, registry: dict = None, *,
                         reference=None) -> Figure:
    """``kind: twin_axis_cdf`` -- CDFs on the left, the potential on the right.

    One figure with two y axes, not two panels. The CDFs are display-only
    renderings of the snapshot saved at the matched simulation time.
    """
    runs = _require_runs(runs, "twin_axis_cdf_figure")
    registry = load_registry() if registry is None else registry
    spec = _with_defaults(spec)
    reference = _resolve_reference(runs, reference)
    if not reference:
        raise ValueError(
            "twin_axis_cdf_figure needs the saved reference: pass "
            "reference={'cdf': {'x': ..., 'F': ...} or 'samples': ..., "
            "'potential': {'x': ..., 'V': ...}}")
    left = dict(spec.get("left_axis") or {})
    right = dict(spec.get("right_axis") or {})
    requested = _requested_snapshot_time(runs, spec)
    policy = str(spec.get("snapshot_time_policy", "nearest_below"))

    indices = _hyperparameter_indices(runs)
    figure = _new_figure(4.8, 3.3)
    axes = figure.subplots()
    twin = axes.twinx()

    potential = reference.get("potential")
    if isinstance(potential, _MissingArtifact):
        potential = None
    if potential is not None:
        style = dict(right.get("style") or {})
        twin.plot(np.asarray(potential["x"], dtype=float),
                  np.asarray(potential["V"], dtype=float),
                  color=style.get("color", "0.55"),
                  alpha=float(style.get("alpha", 0.35)),
                  linewidth=float(style.get("linewidth", 1.2)),
                  zorder=int(style.get("zorder", 0)))
        twin.set_ylabel("potential $V$", color="0.45")
        twin.tick_params(axis="y", colors="0.45", labelsize=7)
        if right.get("minimal_ticks", True):
            twin.locator_params(axis="y", nbins=3)
    twin.spines["right"].set_visible(True)

    handles, labels, realised = [], [], []
    for run in runs:
        snapshot = select_snapshot(run, requested, policy=policy)
        realised.append(snapshot.describe())
        values = snapshot.coordinates(left.get("coordinates", "x"))
        grid, cdf = _empirical_cdf(values)
        style = _run_style(registry, run, indices)
        line, = axes.plot(grid, cdf, color=style["color"],
                          linestyle=style["linestyle"], linewidth=1.2,
                          zorder=3, label=style["label"])
        _collect(handles, labels, line, style["label"])

    if left.get("include_exact_target", True):
        exact = dict(left.get("exact_style") or {})
        base = _reference_style(registry)
        saved_cdf = reference.get("cdf")
        if isinstance(saved_cdf, _MissingArtifact) and "samples" in reference:
            saved_cdf = None
        if saved_cdf is not None:
            saved_cdf = _unwrap_artifact(
                saved_cdf, "this figure draws the exact target CDF")
            grid = np.asarray(saved_cdf["x"], dtype=float)
            cdf = np.asarray(saved_cdf["F"], dtype=float)
        elif "samples" in reference:
            grid, cdf = _empirical_cdf(reference["samples"])
        else:
            raise KeyError("reference has neither 'cdf' nor 'samples'; the "
                           "exact target curve cannot be drawn")
        label = exact.get("label", "exact target")
        line, = axes.plot(grid, cdf, color=exact.get("color", base["color"]),
                          linestyle=exact.get("linestyle", base["linestyle"]),
                          linewidth=1.4, zorder=4, label=label)
        _collect(handles, labels, line, label)

    axes.set_xlabel("$x$")
    axes.set_ylabel("CDF")
    axes.set_ylim(-0.02, 1.02)
    _set_title(axes, spec.get(
        "title", f"matched simulation time $t={realised[0]['realised_t']:g}$"))
    _collect_uncalibratable(handles, labels, registry,
                            getattr(runs, "uncalibratable", {}) or {})
    _figure_legend(figure, handles, labels, ncol=3)
    _record_provenance(figure, {"kind": "twin_axis_cdf",
                                "requested_time": requested,
                                "snapshots": realised})
    return figure


def contour_scatter_grid(runs, spec: dict, registry: dict = None, *,
                         background=None, reference=None) -> Figure:
    """``kind: contour_scatter_grid`` -- one panel per method, sharing everything."""
    runs = _require_runs(runs, "contour_scatter_grid")
    registry = load_registry() if registry is None else registry
    spec = _with_defaults(spec)
    background = _resolve_background(runs, spec, background, reference)
    coordinates = str(spec.get("coordinates", "x"))
    if spec.get("snapshot_time") is None:
        raise ValueError("contour_scatter_grid needs spec['snapshot_time']")
    requested = float(spec["snapshot_time"])
    policy = str(spec.get("snapshot_time_policy", "nearest_below"))
    n_points = int(spec.get("points_per_panel", 3000))

    snapshots = [select_snapshot(run, requested, policy=policy) for run in runs]
    clouds = [_snapshot_points(s, coordinates, n_points) for s in snapshots]
    n_points = min(cloud.shape[0] for cloud in clouds)
    clouds = [cloud[points_per_panel(cloud.shape[0], n_points)]
              for cloud in clouds]
    xlim, ylim = (shared_limits(clouds) if spec.get("shared_limits", True)
                  else (None, None))
    levels = _background_levels(background)

    layout = dict(spec.get("layout") or {})
    columns = int(layout.get("columns", min(3, len(runs))))
    rows = int(layout.get("rows", math.ceil(len(runs) / columns)))
    figure = _new_figure(2.4 * columns, 2.5 * rows + 0.4)
    axes_grid = figure.subplots(rows, columns, squeeze=False).ravel()
    indices = _hyperparameter_indices(runs)
    panels, realised = [], []

    for axes, run, snapshot, cloud in zip(axes_grid, runs, snapshots, clouds):
        _draw_background(axes, background, registry, levels=levels)
        style = _run_style(registry, run, indices)
        _scatter(axes, cloud, style["color"])
        centers = background.get("component_centers")
        marks = spec.get("mark_component_centers")
        if centers is not None and marks:
            centers = np.asarray(centers, dtype=float)
            axes.scatter(centers[:, 0], centers[:, 1],
                         marker=marks.get("marker", "+"),
                         c=marks.get("color", "black"),
                         s=float(marks.get("size", 8)), linewidths=0.6,
                         zorder=3)
        if xlim is not None:
            axes.set_xlim(*xlim)
            axes.set_ylim(*ylim)
        _set_title(axes, style["label"])
        axes.tick_params(labelsize=7)
        panels.append({"name": style["label"], "xlim": axes.get_xlim(),
                       "ylim": axes.get_ylim(), "levels": levels,
                       "n_points": cloud.shape[0], "marker_size": _SCATTER_SIZE,
                       "alpha": _SCATTER_ALPHA,
                       "subsample_rule": "evenly spaced saved indices"})
        realised.append(snapshot.describe())
    for axes in axes_grid[len(runs):]:
        axes.set_visible(False)

    consistency = assert_panels_consistent(panels)
    caption = spec.get("title")
    estimate = bool(spec.get("background_is_estimate")
                    or background.get("is_estimate"))
    if caption is None:
        caption = (f"matched simulation time $t={realised[0]['realised_t']:g}$"
                   + (" (reference estimate background)" if estimate else ""))
    figure.suptitle(_guard_display_text(caption))
    _record_provenance(figure, {"kind": "contour_scatter_grid",
                                "requested_time": requested,
                                "snapshots": realised,
                                "panel_consistency": consistency})
    return figure


def snapshot_matrix(runs, spec: dict, registry: dict = None, *,
                    background=None, reference=None) -> Figure:
    """``kind: snapshot_matrix`` -- methods down the rows, matched times across."""
    runs = _require_runs(runs, "snapshot_matrix")
    registry = load_registry() if registry is None else registry
    spec = _with_defaults(spec)
    if spec.get("annotate_critical_points"):
        raise ValueError(
            "annotate_critical_points is refused on purpose: this figure shows "
            "the CV distribution, not a kinetic story, so it carries no minima, "
            "saddle, basin-boundary, jump-arrow, or transition-path marks")
    background = _resolve_background(runs, spec, background, reference)
    times = list(spec.get("snapshot_times") or [])
    if not times:
        raise ValueError("snapshot_matrix needs spec['snapshot_times']")
    row_methods = list(spec.get("rows")
                       or dict.fromkeys(run.method for run in runs))
    coordinates = str(spec.get("coordinates", "x"))
    policy = str(spec.get("snapshot_time_policy", "nearest_below"))
    n_points = int(spec.get("points_per_panel", 3000))

    selected = []
    for method in row_methods:
        matches = [run for run in runs if run.method == method]
        if not matches:
            raise ValueError(
                f"snapshot_matrix wants a row for {method!r} but no loaded run "
                f"has that method; loaded {sorted({r.method for r in runs})}")
        selected.append(matches[0])

    grid = [[select_snapshot(run, t, policy=policy) for t in times]
            for run in selected]
    clouds = [[_snapshot_points(s, coordinates, n_points) for s in row]
              for row in grid]
    n_points = min(cloud.shape[0] for row in clouds for cloud in row)
    clouds = [[cloud[points_per_panel(cloud.shape[0], n_points)]
               for cloud in row] for row in clouds]
    xlim, ylim = (shared_limits([c for row in clouds for c in row])
                  if spec.get("shared_limits", True) else (None, None))
    levels = _background_levels(background)

    figure = _new_figure(2.2 * len(times) + 0.5, 2.2 * len(selected) + 0.5)
    axes_grid = figure.subplots(len(selected), len(times), squeeze=False)
    indices = _hyperparameter_indices(selected)
    panels, realised = [], []

    for row, (run, row_snapshots, row_clouds) in enumerate(
            zip(selected, grid, clouds)):
        style = _run_style(registry, run, indices)
        for column, (snapshot, cloud) in enumerate(zip(row_snapshots,
                                                       row_clouds)):
            axes = axes_grid[row][column]
            _draw_background(axes, background, registry, levels=levels)
            _scatter(axes, cloud, style["color"])
            if xlim is not None:
                axes.set_xlim(*xlim)
                axes.set_ylim(*ylim)
            if row == 0:
                _set_title(axes, f"$t={snapshot.t:g}$")
            if column == 0:
                axes.set_ylabel(_display_name(registry, run.method))
            axes.tick_params(labelsize=7)
            panels.append({"name": f"{style['label']} t={snapshot.t:g}",
                           "xlim": axes.get_xlim(), "ylim": axes.get_ylim(),
                           "levels": levels, "n_points": cloud.shape[0],
                           "marker_size": _SCATTER_SIZE,
                           "alpha": _SCATTER_ALPHA,
                           "subsample_rule": "evenly spaced saved indices"})
            realised.append(snapshot.describe())

    consistency = assert_panels_consistent(panels)
    _record_provenance(figure, {"kind": "snapshot_matrix",
                                "requested_times": [float(t) for t in times],
                                "snapshots": realised,
                                "panel_consistency": consistency})
    return figure


def mode_metric_panel(runs, spec: dict, registry: dict = None, *,
                      reference=None) -> Figure:
    """``kind: mode_metric_panel`` -- mode coverage, mode weights, occupancy.

    The coverage panel's reference line is the frozen ``EMC*`` the caller read
    from the saved reference descriptor; it is never a hard-coded 1.0. The
    occupancy panel plots a ratio, so its reference line is exactly 1.0, and
    the mode order is the reference's, fixed for every method.
    """
    runs = _require_runs(runs, "mode_metric_panel")
    registry = load_registry() if registry is None else registry
    spec = _with_defaults(spec)
    reference = _resolve_reference(runs, reference)
    panels = list(spec.get("panels") or [])
    if not panels:
        raise ValueError("mode_metric_panel needs spec['panels']")
    uncertainty = spec.get("uncertainty")
    indices = _hyperparameter_indices(runs)

    widths = [len(panel.get("columns") or ["simulation_time"])
              if panel.get("kind", "curve") == "curve"
              else len(panel.get("snapshot_times") or [0.0])
              for panel in panels]
    columns = max(widths)
    figure = _new_figure(3.2 * columns, 2.5 * len(panels) + 0.9)
    axes_grid = figure.subplots(len(panels), columns, squeeze=False)
    handles, labels, realised = [], [], []

    for row, panel in enumerate(panels):
        kind = panel.get("kind", "curve")
        if kind == "curve":
            used = _draw_metric_row(axes_grid[row], panel, runs, registry,
                                    reference, indices, uncertainty, spec,
                                    handles, labels)
        elif kind == "occupancy_profile":
            used = len(panel.get("snapshot_times") or [])
            realised.extend(_draw_occupancy_profile(
                axes_grid[row], panel, runs, registry, reference, indices,
                uncertainty))
        else:
            raise ValueError(
                f"mode_metric_panel does not know panel kind {kind!r}")
        for axes in axes_grid[row][used:]:
            axes.set_visible(False)

    _collect_uncalibratable(handles, labels, registry,
                            getattr(runs, "uncalibratable", {}) or {})
    _figure_legend(figure, handles, labels)
    _record_provenance(figure, {"kind": "mode_metric_panel",
                                "reference_keys": sorted(reference),
                                "occupancy_snapshots": realised})
    return figure


def _draw_metric_row(axes_row, panel, runs, registry, reference, indices,
                     uncertainty, spec, handles, labels) -> int:
    metric = panel["metric"]
    axis_names = list(panel.get("columns") or ["simulation_time"])
    for position, x_axis in enumerate(axis_names):
        axes = axes_row[position]
        x_column = _x_column(x_axis)
        _apply_gate(runs, spec, x_axis)
        for run in runs:
            if metric["column"] not in run.metrics:
                continue
            style = _run_style(registry, run, indices)
            x, centre, lo, hi = seed_aggregate(run.metrics, metric["column"],
                                               x_column,
                                               uncertainty=uncertainty)
            keep = np.isfinite(centre)
            if not keep.any():
                continue
            line, = axes.plot(x[keep], centre[keep], color=style["color"],
                              linestyle=style["linestyle"],
                              marker=style["marker"],
                              markevery=max(1, int(keep.sum()) // 12),
                              markersize=_MARKER_SIZE, label=style["label"])
            axes.fill_between(x[keep], lo[keep], hi[keep], color=style["color"],
                              alpha=_BAND_ALPHA, linewidth=0)
            _collect(handles, labels, line, style["label"])
        line_spec = panel.get("reference_line")
        if line_spec is not None:
            value = _resolve_reference_value(line_spec, reference)
            base = _reference_style(registry)
            label = (line_spec.get("label")
                     if isinstance(line_spec, dict) else None)
            axes.axhline(value, color=base["color"], linestyle="--",
                         linewidth=1.0, zorder=1, label=label)
            if label:
                _collect(handles, labels, axes.lines[-1], label)
        axes.set_xlabel(_x_label(spec, x_axis))
        if position == 0:
            axes.set_ylabel(_guard_display_text(
                metric.get("label", metric["column"])))
        if metric.get("log_y"):
            axes.set_yscale("log")
    return len(axis_names)


def _resolve_reference_value(line_spec, reference: dict) -> float:
    """A reference line's value, resolved from the caller's saved reference."""
    if isinstance(line_spec, (int, float)) and not isinstance(line_spec, bool):
        return float(line_spec)
    if "value" in line_spec:
        return float(line_spec["value"])
    source = line_spec.get("source")
    if source is None:
        raise KeyError("reference_line has neither 'value' nor 'source'")
    key = str(source)
    if key.startswith("reference."):
        key = key[len("reference."):]
    node = reference
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(
                f"the saved reference has no {source!r} (looked for {part!r} "
                f"in {sorted(node) if isinstance(node, dict) else node}). This "
                "line must be the frozen reference value; it is never defaulted "
                "to 1.0.")
        node = node[part]
    return float(_unwrap_artifact(node, f"the reference line {source!r}"))


def _draw_occupancy_profile(axes_row, panel: dict, runs, registry, reference,
                            indices, uncertainty) -> list:
    """Per-mode occupancy ratios in fixed mode order, reference line at 1.0."""
    times = list(panel.get("snapshot_times") or [])
    if not times:
        raise ValueError("occupancy_profile needs snapshot_times")
    order = reference.get("mode_order")
    if isinstance(order, _MissingArtifact):
        order = None
    realised = []
    for position, requested in enumerate(times):
        axes = axes_row[position]
        realised_here = None
        for run in runs:
            prefix, names = _metric_family(run, panel.get("quantity",
                                                          "occupancy_ratio"),
                                           panel.get("column_prefix"))
            if not names:
                raise KeyError(
                    f"run {run.run_id} saved no per-mode occupancy columns "
                    f"(looked for a {prefix}* family); this panel plots saved "
                    "occupancy ratios and never recomputes them")
            if order is not None:
                # One ordering, taken from the reference and applied to every
                # method; the modes are never re-sorted per method.
                lookup = {_mode_key(name[len(prefix):]): name for name in names}
                missing = [str(mode) for mode in order
                           if _mode_key(mode) not in lookup]
                if missing:
                    raise KeyError(
                        f"the reference mode order names {missing} but run "
                        f"{run.run_id} saved {sorted(lookup)}")
                names = [lookup[_mode_key(mode)] for mode in order]
            t_values = np.asarray(run.metrics["t"], dtype=float)
            grid = np.unique(t_values[np.isfinite(t_values)])
            candidates = grid[grid <= float(requested) + 1e-9]
            if candidates.size == 0:
                raise ValueError(f"run {run.run_id} has no checkpoint at or "
                                 f"below t={float(requested):g}")
            realised_here = float(candidates.max())
            slot = int(np.argmin(np.abs(grid - realised_here)))
            values = [float(seed_aggregate(run.metrics, name, "t",
                                           uncertainty=uncertainty)[1][slot])
                      for name in names]
            style = _run_style(registry, run, indices)
            axes.plot(np.arange(len(values)), values, color=style["color"],
                      linestyle=style["linestyle"], marker=style["marker"],
                      markersize=_MARKER_SIZE, linewidth=1.0,
                      label=style["label"])
            realised.append({"run_id": run.run_id,
                             "requested_time": float(requested),
                             "realised_t": realised_here,
                             "n_modes": len(values)})
        # The ratio's reference is exactly one, and the mode order comes from
        # the reference: it is never re-sorted per method.
        axes.axhline(float(panel.get("reference_line", 1.0)), color="0.2",
                     linestyle="--", linewidth=1.0, zorder=1)
        axes.set_xlabel("mode index (reference order, fixed)")
        if position == 0:
            axes.set_ylabel("occupancy ratio")
        _set_title(axes, f"$t={realised_here:g}$")
    return realised


def supplement_panels(runs, spec: dict, registry: dict = None, *,
                      reference=None, background=None) -> Figure:
    """``kind: supplement_panels`` -- the mixed supplementary grid.

    Every panel renders a saved snapshot array, a saved metric column, or a
    value from the saved reference. A quantity that was not measured at run
    time is an error, not something this module computes.
    """
    runs = _require_runs(runs, "supplement_panels")
    registry = load_registry() if registry is None else registry
    spec = _with_defaults(spec)
    reference = _resolve_reference(runs, reference)
    panels = list(spec.get("panels") or [])
    if not panels:
        raise ValueError("supplement_panels needs spec['panels']")
    requested = _requested_snapshot_time(runs, spec)
    policy = str(spec.get("snapshot_time_policy", "nearest_below"))
    n_points = int(spec.get("points_per_panel", 3000))
    uncertainty = spec.get("uncertainty")
    indices = _hyperparameter_indices(runs)
    snapshots = {run.run_id: select_snapshot(run, requested, policy=policy)
                 for run in runs}

    columns = int((spec.get("layout") or {}).get("columns", 3))
    rows = math.ceil(len(panels) / columns)
    figure = _new_figure(3.1 * columns, 2.6 * rows + 0.9)
    axes_grid = figure.subplots(rows, columns, squeeze=False).ravel()
    handles, labels = [], []

    for axes, panel in zip(axes_grid, panels):
        kind = panel.get("kind")
        quantity = panel.get("quantity")
        if kind in ("cdf", "marginal_cdf"):
            for run in runs:
                style = _run_style(registry, run, indices)
                grid, cdf = _empirical_cdf(
                    _snapshot_quantity(snapshots[run.run_id], quantity))
                line, = axes.plot(grid, cdf, color=style["color"],
                                  linestyle=style["linestyle"], linewidth=1.1,
                                  label=style["label"])
                _collect(handles, labels, line, style["label"])
            axes.set_ylabel("CDF")
            axes.set_xlabel(_guard_display_text(quantity))
        elif kind == "histogram":
            for run in runs:
                style = _run_style(registry, run, indices)
                values = np.asarray(_snapshot_quantity(
                    snapshots[run.run_id], quantity), dtype=float).ravel()
                axes.hist(values, bins=int(panel.get("bins", 40)),
                          histtype="step", density=True, color=style["color"],
                          linestyle=style["linestyle"], label=style["label"])
            axes.set_xlabel(_guard_display_text(quantity))
            axes.set_ylabel("density")
        elif kind == "kde":
            pooled = np.concatenate([
                np.asarray(_snapshot_quantity(snapshots[run.run_id], quantity),
                           dtype=float).ravel() for run in runs])
            grid = np.linspace(float(np.nanmin(pooled)),
                               float(np.nanmax(pooled)), 200)
            for run in runs:
                style = _run_style(registry, run, indices)
                density = _gaussian_kde_1d(
                    _snapshot_quantity(snapshots[run.run_id], quantity), grid,
                    bandwidth=panel.get("bandwidth"))
                line, = axes.plot(grid, density, color=style["color"],
                                  linestyle=style["linestyle"], linewidth=1.1,
                                  label=style["label"])
                _collect(handles, labels, line, style["label"])
            axes.set_xlabel(_guard_display_text(quantity))
            axes.set_ylabel("density (fixed-bandwidth KDE)")
        elif kind == "radial_distribution":
            for run in runs:
                style = _run_style(registry, run, indices)
                values = np.asarray(_snapshot_quantity(
                    snapshots[run.run_id], quantity), dtype=float).ravel()
                axes.hist(values, bins=int(panel.get("bins", 40)),
                          histtype="step", density=True, color=style["color"],
                          linestyle=style["linestyle"], label=style["label"])
            axes.set_xlabel(_guard_display_text(quantity))
            axes.set_ylabel("density")
        elif kind == "correlation_profile":
            for run in runs:
                style = _run_style(registry, run, indices)
                names, profile = _metric_vector(run, quantity, requested,
                                                uncertainty,
                                                panel.get("column_prefix"))
                line, = axes.plot(np.arange(len(profile)), profile,
                                  color=style["color"],
                                  linestyle=style["linestyle"],
                                  marker=style["marker"],
                                  markersize=_MARKER_SIZE,
                                  label=style["label"])
                _collect(handles, labels, line, style["label"])
            axes.set_xlabel("separation")
            axes.set_ylabel(_guard_display_text(quantity))
        elif kind == "occupancy_bars":
            width = 0.8 / max(len(runs), 1)
            names = []
            for position, run in enumerate(runs):
                style = _run_style(registry, run, indices)
                names, values = _metric_vector(run, quantity, requested,
                                               uncertainty,
                                               panel.get("column_prefix"))
                axes.bar(np.arange(len(values)) + position * width, values,
                         width=width, color=style["color"], alpha=0.85,
                         label=style["label"])
            axes.set_xticks(np.arange(len(names)) + 0.4 - width / 2.0)
            axes.set_xticklabels([_guard_display_text(name) for name in names],
                                 fontsize=7, rotation=90)
        elif kind == "scalar_bars":
            names = (list(quantity) if isinstance(quantity, (list, tuple))
                     else [quantity])
            width = 0.8 / max(len(runs), 1)
            for position, run in enumerate(runs):
                style = _run_style(registry, run, indices)
                values = [_metric_scalar(run, name, requested, uncertainty)
                          for name in names]
                axes.bar(np.arange(len(names)) + position * width, values,
                         width=width, color=style["color"], alpha=0.85,
                         label=style["label"])
            axes.set_xticks(np.arange(len(names)) + 0.4 - width / 2.0)
            axes.set_xticklabels([_guard_display_text(name) for name in names],
                                 fontsize=7)
        elif kind == "matrix_heatmap":
            run = runs[0]
            names, values = _metric_vector(run, quantity, requested,
                                           uncertainty,
                                           panel.get("column_prefix"))
            side = int(round(math.sqrt(len(values))))
            matrix = (np.asarray(values, dtype=float).reshape(side, side)
                      if side * side == len(values)
                      else np.asarray(values, dtype=float)[None, :])
            image = axes.imshow(matrix, cmap=_background_cmap(registry),
                                aspect="auto")
            figure.colorbar(image, ax=axes, fraction=0.046)
            _set_title(axes,
                       f"{quantity} ({_display_name(registry, run.method)})")
        elif kind == "contour_scatter":
            surface = _resolve_background(runs, {**spec, **panel}, background,
                                          reference)
            levels = _draw_background(axes, surface, registry)
            clouds = []
            for run in runs:
                points = np.asarray(_snapshot_quantity(
                    snapshots[run.run_id], quantity), dtype=float)
                if points.ndim == 1:
                    points = points[:, None]
                clouds.append(points[points_per_panel(points.shape[0],
                                                      n_points)])
            take = min(cloud.shape[0] for cloud in clouds)
            records = []
            for run, cloud in zip(runs, clouds):
                cloud = cloud[points_per_panel(cloud.shape[0], take)]
                style = _run_style(registry, run, indices)
                _scatter(axes, cloud, style["color"])
                records.append({"name": style["label"], "levels": levels,
                                "n_points": cloud.shape[0],
                                "marker_size": _SCATTER_SIZE,
                                "alpha": _SCATTER_ALPHA,
                                "subsample_rule":
                                    "evenly spaced saved indices"})
            xlim, ylim = shared_limits(clouds)
            axes.set_xlim(*xlim)
            axes.set_ylim(*ylim)
            for record in records:
                record["xlim"], record["ylim"] = xlim, ylim
            assert_panels_consistent(records)
        elif kind == "reference_validation":
            table = _resolve_reference_table(panel.get("source"), reference)
            names = list(table)
            axes.barh(np.arange(len(names)),
                      [float(table[name]) for name in names], color="0.55")
            axes.set_yticks(np.arange(len(names)))
            axes.set_yticklabels([_guard_display_text(name) for name in names],
                                 fontsize=7)
            _set_title(axes, "reference validation")
        else:
            raise ValueError(
                f"supplement_panels does not know panel kind {kind!r}")
        if not axes.get_title():
            _set_title(axes, ", ".join(map(str, quantity))
                       if isinstance(quantity, (list, tuple)) else quantity)

    for axes in axes_grid[len(panels):]:
        axes.set_visible(False)
    _collect_uncalibratable(handles, labels, registry,
                            getattr(runs, "uncalibratable", {}) or {})
    _figure_legend(figure, handles, labels)
    _record_provenance(figure, {
        "kind": "supplement_panels", "requested_time": requested,
        "snapshots": [snapshot.describe()
                      for snapshot in snapshots.values()]})
    return figure


def _snapshot_quantity(snapshot: Snapshot, name):
    """A per-particle quantity from a saved snapshot.

    Either an array the measurement suite saved, or a display-only slice of one
    (``mx`` is column 0 of the saved order parameter). Nothing is recomputed
    from the sample positions.
    """
    if name is None:
        raise ValueError("a supplement panel needs a 'quantity'")
    name = str(name)
    if name in snapshot.arrays:
        return snapshot.coordinates(name)
    derived = _DERIVED_SNAPSHOT_QUANTITIES.get(name)
    if derived is not None and derived[0] in snapshot.arrays:
        source = np.asarray(snapshot.coordinates(derived[0]), dtype=float)
        if derived[1] == "norm":
            return np.linalg.norm(source, axis=-1)
        return source[:, int(derived[1])]
    raise KeyError(
        f"{snapshot.path} saved no array {name!r}; it saved "
        f"{sorted(snapshot.arrays)} and this module can slice "
        f"{sorted(_DERIVED_SNAPSHOT_QUANTITIES)} out of them. The measurement "
        "suite has to save the quantity at run time; plotting never recomputes "
        "it.")


def _mode_key(token):
    text = str(token)
    return int(text) if text.isdigit() else text


def _metric_family(run: RunData, quantity, explicit_prefix=None):
    """The saved metric-column family for one vector quantity.

    Only numeric suffixes count, so ``susceptibility_`` picks up
    ``susceptibility_00`` but not ``susceptibility_relative_frobenius``.
    """
    candidates = []
    if explicit_prefix:
        candidates.append(str(explicit_prefix))
    known = _METRIC_FAMILY_PREFIXES.get(str(quantity))
    if known:
        candidates.append(known)
    candidates += [f"mode_{quantity}_", f"{quantity}_"]
    for prefix in candidates:
        names = sorted(name for name in run.metrics
                       if name.startswith(prefix)
                       and name[len(prefix):].isdigit())
        if names:
            return prefix, names
    return candidates[0], []


def _metric_slot(run: RunData, requested_time: float):
    """The saved checkpoint at or below ``requested_time`` and its row index."""
    t_values = np.asarray(run.metrics["t"], dtype=float)
    grid = np.unique(t_values[np.isfinite(t_values)])
    candidates = grid[grid <= float(requested_time) + 1e-9]
    if candidates.size == 0:
        raise ValueError(f"run {run.run_id} has no checkpoint at or below "
                         f"t={float(requested_time):g}")
    realised = float(candidates.max())
    return grid, int(np.argmin(np.abs(grid - realised))), realised


def _metric_vector(run: RunData, quantity, requested_time, uncertainty,
                   explicit_prefix=None):
    """``(labels, values)`` of a saved metric-column family at a matched time."""
    prefix, names = _metric_family(run, quantity, explicit_prefix)
    if not names:
        raise KeyError(
            f"run {run.run_id} saved no {prefix}* metric columns for "
            f"{quantity!r}; this panel plots saved numbers and never "
            f"recomputes them. It saved {sorted(run.metrics)}")
    _, slot, _ = _metric_slot(run, requested_time)
    values = [float(seed_aggregate(run.metrics, name, "t",
                                   uncertainty=uncertainty)[1][slot])
              for name in names]
    return [name[len(prefix):] for name in names], values


def _metric_scalar(run: RunData, quantity, requested_time,
                   uncertainty) -> float:
    """One saved scalar metric at a matched time, by name or ``<name>_error``."""
    for column in (str(quantity), f"{quantity}_error", f"{quantity}_mean"):
        if column in run.metrics:
            _, slot, _ = _metric_slot(run, requested_time)
            return float(seed_aggregate(run.metrics, column, "t",
                                        uncertainty=uncertainty)[1][slot])
    raise KeyError(
        f"run {run.run_id} saved no metric column for {quantity!r} (tried "
        f"{quantity!r}, {quantity}_error, {quantity}_mean); this panel plots "
        "saved numbers and never recomputes them")


def _requested_snapshot_time(runs, spec: dict) -> float:
    """The spec's snapshot time, or the latest time every run saved."""
    for key in ("snapshot_time",):
        if spec.get(key) is not None:
            return float(spec[key])
    times = spec.get("snapshot_times")
    if times:
        return float(times[-1])
    common = None
    for run in runs:
        available = set(run.snapshot_times())
        common = available if common is None else (common & available)
    if not common:
        raise ValueError(
            "this figure names no snapshot_time and the loaded runs share no "
            "saved checkpoint; snapshots are matched by simulation time and "
            "never interpolated")
    return float(max(common))


def _resolve_reference_table(source, reference: dict) -> dict:
    key = str(source or "")
    if key.startswith("reference."):
        key = key[len("reference."):]
    node = reference
    for part in [item for item in key.split(".") if item]:
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"the saved reference has no {source!r}")
        node = _unwrap_artifact(node[part], f"the panel source {source!r}")
    node = _unwrap_artifact(node, f"the panel source {source!r}")
    table = _flatten_scalars(node)
    if not table:
        raise ValueError(
            f"the saved reference entry {source!r} holds no numeric values to "
            "draw")
    return table


def _flatten_scalars(node, prefix: str = "", depth: int = 1) -> dict:
    """Numeric leaves of a saved validation record, one nesting level deep."""
    out = {}
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[f"{prefix}{key}"] = float(value)
            elif depth > 0 and isinstance(value, (dict, list)):
                out.update(_flatten_scalars(value, f"{key}.", depth - 1))
    elif isinstance(node, list):
        for item in node:
            if not isinstance(item, dict):
                continue
            name = (item.get("check") or item.get("name") or item.get("gate")
                    or item.get("statistic"))
            for field in ("value", "observed", "measured"):
                value = item.get(field)
                if isinstance(value, (int, float)) and not isinstance(value,
                                                                      bool):
                    out[f"{prefix}{name}"] = float(value)
                    break
    return out


# ------------------------------------------------------------------- output
def save_figure(figure: Figure, name: str, output_dir,
                formats=("png", "pdf", "svg", "tiff"), dpi: int = 400) -> dict:
    """Write one figure in every requested format and return the paths."""
    if figure is None:
        raise ValueError("save_figure got no figure")
    formats = tuple(formats or ())
    if not formats:
        raise ValueError("save_figure got an empty format list")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for suffix in formats:
        path = output_dir / f"{name}.{suffix}"
        # TIFF goes through Pillow; LZW keeps a 400 dpi page lossless and small
        # enough for a submission system.
        options = ({"pil_kwargs": {"compression": "tiff_lzw"}}
                   if suffix in ("tif", "tiff") else {})
        # bbox_inches="tight" so an outside legend is included in the saved
        # page rather than cropped at the figure edge.
        figure.savefig(path, format=suffix, dpi=dpi, bbox_inches="tight",
                       **options)
        if not path.is_file() or path.stat().st_size == 0:
            raise IOError(f"save_figure wrote an empty file: {path}")
        written[suffix] = str(path)
    return written
