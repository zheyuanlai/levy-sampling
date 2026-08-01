"""Single source of truth for the E1--E4 manuscript release.

Internal sampler names are kept stable for compatibility with the numerical
code and frozen CSV files.  ``display_labels`` contains the names used in the
JCP manuscript and figures (in particular BAOAB -> ULD).
"""
from __future__ import annotations

from dataclasses import dataclass


METRICS: tuple[str, ...] = ("W2", "MMD", "TV", "worst_basin_ESS")
# Wall-clock is reported again: every method in E1--E4 is now timed by one
# protocol -- a single dedicated GPU with no co-tenants, the same batched
# ensemble shape, CUDA-synchronized timers around sampler work only, and
# untimed warm-up steps. See the timing policy in src/runner.py.
RESOURCE_AXES: tuple[str, ...] = ("t", "nfe", "wallclock")
# Publication export formats; one figures/<format>/ directory per entry.
FIGURE_FORMATS: tuple[str, ...] = ("png", "tiff", "svg", "pdf")


@dataclass(frozen=True)
class ManuscriptExperiment:
    number: str
    key: str
    title: str
    notebook: str
    config: str
    methods: tuple[str, ...]
    display_labels: dict[str, str]

    @property
    def methods_csv(self) -> str:
        return ",".join(self.methods)


_COMMON_LABELS = {
    "ULA": "ULA",
    "BAOAB": "ULD",
    "PT": "PT",
    "FLA": "FLA",
    "CP": "Raw-CP",
    "LSC-CP": "LSC-CP",
    "LSC-CP-RA": "LSC-CP-RA",
    "LSC-CP-MA": "LSC-CP-RA",
}


EXPERIMENTS: dict[str, ManuscriptExperiment] = {
    "double_well": ManuscriptExperiment(
        number="E1",
        key="double_well",
        title="Double well",
        notebook="01_double_well.ipynb",
        config="E1_double_well.yaml",
        methods=("ULA", "BAOAB", "PT", "FLA", "CP", "LSC-CP", "LSC-CP-RA"),
        display_labels={
            **_COMMON_LABELS,
            "LSC-CP-RA": "LSC-CP-RA",
        },
    ),
    "mog40": ManuscriptExperiment(
        number="E2",
        key="mog40",
        title="MoG40",
        notebook="02_mog40.ipynb",
        config="E2_mog40.yaml",
        methods=("ULA", "BAOAB", "PT", "FLA", "LSC-CP", "LSC-CP-RA"),
        display_labels={
            **_COMMON_LABELS,
            "LSC-CP-RA": "LSC-CP-RA",
        },
    ),
    "mb3well_10d": ManuscriptExperiment(
        number="E3",
        key="mb3well_10d",
        title="Müller--Brown three-well system (10D)",
        notebook="03_mb3well_10d.ipynb",
        config="E3_mb3well_10d.yaml",
        methods=("ULA", "BAOAB", "PT", "FLA", "LSC-CP", "LSC-CP-MA"),
        display_labels={
            **_COMMON_LABELS,
            "LSC-CP-MA": "LSC-CP-RA (4)",
        },
    ),
    "coupled_phi4": ManuscriptExperiment(
        number="E4",
        key="coupled_phi4",
        title="Coupled two-component phi4 chain",
        notebook="04_coupled_phi4.ipynb",
        config="E4_coupled_phi4.yaml",
        methods=("ULA", "BAOAB", "PT", "FLA", "LSC-CP", "LSC-CP-MA"),
        display_labels={
            **_COMMON_LABELS,
            "LSC-CP-MA": "LSC-CP-RA (8)",
        },
    ),
}


def experiment_spec(key: str) -> ManuscriptExperiment:
    """Return a manuscript experiment specification or fail with a useful key."""
    try:
        return EXPERIMENTS[key]
    except KeyError as exc:
        raise KeyError(
            f"unknown manuscript experiment {key!r}; "
            f"choose from {tuple(EXPERIMENTS)}"
        ) from exc


def manuscript_methods(key: str) -> tuple[str, ...]:
    return experiment_spec(key).methods


def display_label(key: str, method: str) -> str:
    spec = experiment_spec(key)
    return spec.display_labels.get(method, method)
