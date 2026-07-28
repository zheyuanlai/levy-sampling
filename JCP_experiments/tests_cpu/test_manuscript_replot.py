from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "replot_manuscript_figures.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("manuscript_replot", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manuscript_method_matrix_and_labels():
    module = _load_module()
    assert module.EXPERIMENTS["double_well"].methods == (
        "ULA", "BAOAB", "PT", "FLA", "CP", "LSC-CP", "LSC-CP-RA",
    )
    assert module.EXPERIMENTS["mog40"].methods == (
        "ULA", "BAOAB", "PT", "FLA", "LSC-CP", "LSC-CP-RA",
    )
    for experiment, atoms in (("mb3well_10d", 4), ("coupled_phi4", 8)):
        spec = module.EXPERIMENTS[experiment]
        assert spec.methods == (
            "ULA", "BAOAB", "PT", "FLA", "LSC-CP", "LSC-CP-MA",
        )
        assert spec.labels["LSC-CP-MA"] == f"LSC-CP-RA ({atoms})"
    assert all(
        spec.labels["BAOAB"] == "ULD"
        for spec in module.EXPERIMENTS.values()
    )


def test_method_styles_are_stable_across_experiments():
    module = _load_module()
    assert set().union(*(set(spec.methods) for spec in module.EXPERIMENTS.values())) <= (
        set(module.METHOD_STYLE)
    )
    assert module.METHOD_STYLE["LSC-CP-RA"] == module.METHOD_STYLE["LSC-CP-MA"]


def test_stationarity_summary_uses_combined_nfe(tmp_path):
    module = _load_module()
    path = tmp_path / "summary.csv"
    path.write_text(
        "method,worst_basin_ess,worst_basin_ess_per_second,"
        "gradient_evals,potential_evals,score_quadrature_evals\n"
        "LSC-CP,100,2,10,30,60\n",
        encoding="utf-8",
    )
    summary = module._stationarity_summary(path, ["LSC-CP"])
    assert summary["LSC-CP"]["worst_basin_ess_per_nfe"] == pytest.approx(1.0)


def test_full_replot_writes_only_png_and_pdf_directories(tmp_path):
    module = _load_module()
    figures = tmp_path / "figures"
    result = module.regenerate(ROOT / "results", figures)
    assert {path.name for path in figures.iterdir()} == {"png", "pdf"}
    assert result["png_count"] == 60
    assert result["pdf_count"] == 60
    assert len(list((figures / "png").glob("*.png"))) == 60
    assert len(list((figures / "pdf").glob("*.pdf"))) == 60


def test_no_clean_allows_generated_sample_figures_to_coexist(tmp_path):
    module = _load_module()
    figures = tmp_path / "figures"
    for extension in ("png", "pdf"):
        directory = figures / extension
        directory.mkdir(parents=True)
        (directory / f"existing_generated_density.{extension}").write_text("x")
    result = module.regenerate(ROOT / "results", figures, clean=False)
    assert result["png_count"] == 60
    assert result["pdf_count"] == 60
    assert (figures / "png" / "existing_generated_density.png").exists()
    assert (figures / "pdf" / "existing_generated_density.pdf").exists()


def test_replot_refuses_broad_output_directory():
    module = _load_module()
    with pytest.raises(ValueError, match="unsafe figures directory"):
        module.regenerate(ROOT / "results", ROOT)
