"""Configuration is the run input, not documentation.

Every builder reads its parameters from YAML. There is no second, hard-coded
parameter table anywhere in ``src``, and every run writes back the fully
expanded ``resolved_config.yaml`` it actually used -- including values derived
at build time such as the shell half-width, the drift cap, and the checkpoint
times.

There is exactly one default configuration per experiment. There are no
smoke/dev/production/reference profiles: opening a run notebook and running all
cells executes the full default configuration. Lowering the particle count for
local debugging is an explicit temporary edit, not a second committed profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
from pathlib import Path

import numpy as np
import torch
import yaml

from . import experiments as experiment_components
from .device import DTYPE, device_provenance, resolve_device
from .results import RunPaths, slugify, stable_hash
from .rng import EnsembleStreams
from .targets import TARGET_BUILDERS

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = REPOSITORY_ROOT / "configs"
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "results"


# --------------------------------------------------------------- YAML input
def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_registry(config_dir: str | Path = DEFAULT_CONFIG_DIR) -> dict:
    """The single source of truth for method identity, styling, and matrices."""
    return load_yaml(Path(config_dir) / "registry.yaml")


def load_method_configs(config_dir: str | Path = DEFAULT_CONFIG_DIR) -> dict:
    """Method YAML keyed by internal method name.

    A file may describe one method (``method:``) or a family (``methods:``).
    """
    out: dict[str, dict] = {}
    method_dir = Path(config_dir) / "methods"
    for path in sorted(method_dir.glob("*.yaml")):
        payload = load_yaml(path)
        names = payload.get("methods") or [payload["method"]]
        for name in names:
            merged = {key: value for key, value in payload.items()
                      if key not in ("method", "methods")}
            merged.update(payload.get(name, {}) or {})
            merged["source"] = str(path.relative_to(REPOSITORY_ROOT))
            out[name] = merged
    return out


# ------------------------------------------------------------------ variants
@dataclass(frozen=True)
class Variant:
    """One runnable configuration of one method.

    ``parameters`` is the full identity, ``tame`` included. ``rng_pair_group``
    is the pairing identity and deliberately excludes ``tame``, so canonical and
    tamed variants of the same method at the same hyperparameters share their
    named random streams.
    """

    method: str
    family: str
    parameters: dict
    tame: bool
    label: str
    slug: str
    rng_pair_group: dict

    @property
    def hash(self) -> str:
        return stable_hash({"method": self.method,
                            "parameters": self.parameters})

    @property
    def rng_pair_group_hash(self) -> str:
        return stable_hash(self.rng_pair_group)

    def describe(self) -> dict:
        return {
            "method": self.method,
            "family": self.family,
            "parameters": dict(self.parameters),
            "tame": self.tame,
            "variant_label": self.label,
            "variant_slug": self.slug,
            "variant_hash": self.hash,
            "rng_pair_group": dict(self.rng_pair_group),
            "rng_pair_group_hash": self.rng_pair_group_hash,
        }


def variant_display_label(registry: dict, method: str, parameters: dict) -> str:
    """Legend label for one variant, from the registry alone."""
    entry = registry["methods"][method]
    base = entry.get("display_name", method)
    template = entry.get("variant_label_template")
    defaults = entry.get("variant_label_default_when") or {}
    hyperparameters = {key: value for key, value in parameters.items()
                       if key != "tame"}
    if template is not None:
        is_default = all(hyperparameters.get(key) == value
                         for key, value in defaults.items())
        if not is_default:
            base = template.format(**hyperparameters)
    else:
        extras = ", ".join(f"{key}={_format_number(value)}"
                           for key, value in sorted(hyperparameters.items()))
        if extras:
            base = f"{base} {extras}"
    return f"{base}, {'tamed' if parameters.get('tame') else 'canonical'}"


def _format_number(value):
    if isinstance(value, float):
        return f"{value:g}"
    return value


def make_variant(registry: dict, method_configs: dict, method: str,
                 parameters: dict) -> Variant:
    """Expand one parameter dict into a fully identified variant."""
    if method not in registry["methods"]:
        raise KeyError(
            f"unknown method {method!r}; registry knows "
            f"{sorted(registry['methods'])}")
    entry = registry["methods"][method]
    resolved = dict(entry.get("defaults") or {})
    resolved.update({key: value for key, value in parameters.items()
                     if key != "tame"})
    tame = bool(parameters.get("tame", False))
    if tame and not entry.get("supports_tame", False):
        raise ValueError(f"{method} does not support taming")
    resolved["tame"] = tame

    pairing_keys = (method_configs.get(method, {}) or {}).get(
        "rng_pair_group_keys")
    if pairing_keys is None:
        pairing_keys = sorted(key for key in resolved if key != "tame")
    pair_group = {"method_family": entry.get("family", method)}
    pair_group.update({key: resolved[key] for key in pairing_keys
                       if key in resolved})

    label = variant_display_label(registry, method, resolved)
    hyperparameters = "-".join(
        f"{key}{_format_number(value)}"
        for key, value in sorted(resolved.items()) if key != "tame")
    slug_parts = [slugify(method)]
    if hyperparameters:
        slug_parts.append(slugify(hyperparameters))
    slug_parts.append("tamed" if tame else "canonical")
    return Variant(method=method, family=entry.get("family", method),
                   parameters=resolved, tame=tame, label=label,
                   slug="-".join(slug_parts), rng_pair_group=pair_group)


def expand_variants(registry: dict, method_configs: dict, method: str,
                    variants) -> list[Variant]:
    """Expand a notebook's variant list into canonical and tamed runs.

    An entry that pins ``tame`` explicitly is taken as written. An entry that
    does not is expanded into both variants when the method supports taming,
    because every taming-capable method runs both by default.
    """
    entry = registry["methods"][method]
    supports_tame = bool(entry.get("supports_tame", False))
    out: list[Variant] = []
    for parameters in variants:
        parameters = dict(parameters or {})
        if "tame" in parameters or not supports_tame:
            out.append(make_variant(registry, method_configs, method,
                                    parameters))
            continue
        for tame in (False, True):
            out.append(make_variant(registry, method_configs, method,
                                    {**parameters, "tame": tame}))
    return out


def default_variants(registry: dict, method_configs: dict, experiment_id: str,
                     method: str) -> list[Variant]:
    """The experiment's default variant grid for one method."""
    entry = registry["experiments"][experiment_id]["methods"]
    if method not in entry:
        raise KeyError(f"{method!r} is not enabled for {experiment_id}")
    return expand_variants(registry, method_configs, method, entry[method])


# -------------------------------------------------------------- checkpoints
def checkpoint_steps(n_steps: int, *, dense_count: int, dense_fraction: float,
                     sparse_count: int, include_initial: bool = True,
                     include_terminal: bool = True) -> list[int]:
    """One shared checkpoint schedule, identical across every method.

    Dense early coverage, because the nonlocal methods equilibrate within about
    one ``lambda^-1``, then sparse coverage to the end. Step 0 is included when
    ``include_initial`` is set; every method starts from the same initial
    ensemble, so those rows are identical by construction and anchor the curves.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if dense_count < 1 or sparse_count < 1:
        raise ValueError("dense_count and sparse_count must be positive")
    if not 0.0 <= dense_fraction <= 1.0:
        raise ValueError("dense_fraction must lie in [0, 1]")
    dense_end = min(n_steps, max(dense_count, int(round(n_steps * dense_fraction))))
    dense = np.linspace(1, dense_end, min(dense_count, dense_end))
    remaining = n_steps - dense_end
    sparse = (np.linspace(dense_end + 1, n_steps, min(sparse_count, remaining))
              if remaining > 0 else np.empty(0))
    steps = sorted({int(round(value))
                    for value in np.concatenate([dense, sparse])})
    if include_terminal and steps[-1] != n_steps:
        steps.append(n_steps)
    if include_initial:
        steps = [0] + steps
    return steps


def snapshot_checkpoints(steps: list[int], dt: float,
                         time_values) -> list[int]:
    """Map requested snapshot times to actually saved checkpoints.

    Nothing is interpolated. When a requested time has no exactly matching
    checkpoint the nearest checkpoint at or below it is used, and the caller
    records the realised time so a figure never claims a time it did not save.
    """
    chosen = []
    for requested in time_values:
        candidates = [step for step in steps if step * dt <= requested + 1e-12]
        chosen.append(max(candidates) if candidates else steps[0])
    return sorted(set(chosen))


# -------------------------------------------------------------- experiment
@dataclass
class ExperimentContext:
    """Everything shared by every variant of one experiment."""

    experiment_id: str
    slug: str
    config: dict
    registry: dict
    method_configs: dict
    target: object
    law: object
    box: object
    init_fn: object
    device: torch.device
    dtype: torch.dtype
    paths: RunPaths
    cp_cap: float
    default_cap: float
    pt_beta_min: float
    n_steps: int
    checkpoint_steps: list[int] = field(default_factory=list)
    extras: dict = field(default_factory=dict)
    _reference: object | None = field(default=None, repr=False)
    _fee_calibration: object | None = field(default=None, repr=False)

    # -- identity ----------------------------------------------------------
    @property
    def key(self) -> str:
        return f"{self.experiment_id}_{self.slug}"

    @property
    def beta(self) -> float:
        return float(self.target.beta)

    @property
    def particles(self) -> int:
        return int(self.config["protocol"]["particles"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(int(self.config["protocol"]["seeds"])))

    @property
    def final_time(self) -> float:
        return float(self.config["protocol"]["final_time"])

    @property
    def intensity(self) -> float:
        return float(self.config["jump_law"]["intensity"])

    @property
    def target_hash(self) -> str:
        target_config = copy.deepcopy(self.config["target"])
        return stable_hash({"target": target_config,
                            "jump_law": self.extras["resolved"]["jump_law"],
                            "boundary": self.extras["resolved"]["boundary"]})

    def checkpoint_times(self, dt: float) -> list[float]:
        return [step * dt for step in self.checkpoint_steps]

    def steps_for(self, dt: float) -> int:
        return int(round(self.final_time / float(dt)))

    def schedule_for(self, dt: float) -> list[int]:
        checkpoints = self.config["checkpoints"]
        return checkpoint_steps(
            self.steps_for(dt),
            dense_count=int(checkpoints["dense"]["count"]),
            dense_fraction=float(checkpoints["dense"]["fraction"]),
            sparse_count=int(checkpoints["sparse"]["count"]),
            include_initial=bool(checkpoints.get("include_initial", True)),
            include_terminal=bool(checkpoints.get("include_terminal", True)))

    def tame_cap_for(self, variant: Variant) -> float | None:
        """Resolved taming cap: ``None`` for canonical, a number for tamed."""
        if not variant.tame:
            return None
        family = self.registry["methods"][variant.method].get("family",
                                                              variant.method)
        return self.cp_cap if family in ("CP", "LSC-CP", "LSC-CP-RA") else \
            self.default_cap

    def streams_for(self, variant: Variant, seeds=None) -> EnsembleStreams:
        return EnsembleStreams(
            self.experiment_id, variant.family, variant.rng_pair_group,
            self.seeds if seeds is None else tuple(seeds), self.device,
            self.dtype)

    # -- lazily-built shared inputs ---------------------------------------
    def ensure_reference(self, *, rebuild: bool = False):
        """Build the reference once, or reuse the stored one.

        The reference does not depend on any method parameter, so it is built at
        most once per experiment and reused by every variant.
        """
        if self._reference is not None and not rebuild:
            return self._reference
        from . import references

        self._reference = references.build_or_load(
            self.experiment_id, self.config, self.target,
            self.paths.reference_dir, device=self.device, rebuild=rebuild)
        return self._reference

    @property
    def reference_hash(self) -> str:
        reference = self.ensure_reference()
        return stable_hash(reference.describe())

    def ensure_fee_calibration(self, *, refresh: bool = False):
        """Measure the per-configuration oracle costs once per device."""
        if self._fee_calibration is not None and not refresh:
            return self._fee_calibration
        from . import fee

        chord_counts = self.extras.get("fee_chord_counts", (16, 64, 128))
        self._fee_calibration = fee.load_or_calibrate(
            self.target, self.paths.fee_cache_dir, refresh=refresh,
            chord_counts=tuple(chord_counts))
        return self._fee_calibration

    def resolved_config(self, *, variant: Variant | None = None,
                        dt: float | None = None,
                        calibration: dict | None = None) -> dict:
        """The fully expanded configuration a run actually used."""
        resolved = copy.deepcopy(self.config)
        resolved["resolved"] = copy.deepcopy(self.extras["resolved"])
        resolved["resolved"]["device"] = device_provenance(self.device,
                                                           self.dtype)
        resolved["resolved"]["seeds"] = list(self.seeds)
        if variant is not None:
            resolved["resolved"]["variant"] = variant.describe()
            resolved["resolved"]["tame_cap"] = self.tame_cap_for(variant)
        if dt is not None:
            steps = self.schedule_for(dt)
            resolved["resolved"]["dt"] = float(dt)
            resolved["resolved"]["n_steps"] = self.steps_for(dt)
            resolved["resolved"]["checkpoint_steps"] = steps
            resolved["resolved"]["checkpoint_times"] = [
                step * float(dt) for step in steps]
        if calibration is not None:
            resolved["resolved"]["calibration"] = calibration
        return resolved


def load_experiment(experiment_id: str, *, device: str = "auto",
                    config_dir: str | Path = DEFAULT_CONFIG_DIR,
                    results_root: str | Path = DEFAULT_RESULTS_ROOT,
                    overrides: dict | None = None) -> ExperimentContext:
    """Build the shared context for one experiment.

    This does the parts every method needs: read the configuration, construct
    the target, the jump law, the numerical box, and the initial condition, and
    fix the checkpoint schedule. It does NOT run any calibration -- those are
    conditional on which method actually runs.
    """
    registry = load_registry(config_dir)
    if experiment_id not in registry["experiments"]:
        raise KeyError(
            f"unknown experiment {experiment_id!r}; registry knows "
            f"{sorted(registry['experiments'])}")
    entry = registry["experiments"][experiment_id]
    config = load_yaml(REPOSITORY_ROOT / entry["config"])
    if overrides:
        config = _deep_update(copy.deepcopy(config), overrides)

    resolved_device = resolve_device(device)
    torch.set_default_dtype(DTYPE)

    builder = TARGET_BUILDERS[config["target"]["builder"]]
    target = builder(config, resolved_device)
    components = experiment_components.build_components(
        experiment_id, config, target, resolved_device)

    paths = RunPaths(Path(results_root),
                     f"{experiment_id}_{entry['slug']}").ensure()
    context = ExperimentContext(
        experiment_id=experiment_id,
        slug=entry["slug"],
        config=config,
        registry=registry,
        method_configs=load_method_configs(config_dir),
        target=target,
        law=components["law"],
        box=components["box"],
        init_fn=components["init_fn"],
        device=resolved_device,
        dtype=DTYPE,
        paths=paths,
        cp_cap=components["cp_cap"],
        default_cap=components["default_cap"],
        pt_beta_min=components["pt_beta_min"],
        n_steps=0,
        extras={key: value for key, value in components.items()
                if key not in ("law", "box", "init_fn")},
    )
    initial_dt = float(config["protocol"]["initial_dt"])
    context.n_steps = context.steps_for(initial_dt)
    context.checkpoint_steps = context.schedule_for(initial_dt)
    return context


def _deep_update(base: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base
