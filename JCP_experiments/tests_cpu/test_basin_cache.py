from __future__ import annotations

import numpy as np
import pytest
import torch

from src.references import GradientFlowBasinMap2D


def _zero_grad(z):
    return torch.zeros_like(z)


def _build(cache, **kwargs):
    return GradientFlowBasinMap2D(
        _zero_grad,
        torch.tensor([[-0.5, 0.0], [0.5, 0.0]], dtype=torch.float64),
        (-1.0, -1.0), (1.0, 1.0), n_grid=5, device="cpu",
        cache=str(cache), dt_flow=0.1, n_flow=1, **kwargs)


def test_basin_cache_writes_and_validates_full_construction_metadata(tmp_path):
    cache = tmp_path / "basin.npz"
    created = _build(cache)
    assert created.cache_validation_status == "created_validated"
    assert created.cache_sha256 and len(created.cache_sha256) == 64
    with np.load(cache, allow_pickle=False) as data:
        assert {"labels", "cache_schema_version", "n_grid", "lo", "hi",
                "minima", "dt_flow", "n_flow"}.issubset(data.files)

    loaded = _build(cache)
    assert loaded.cache_validation_status == "validated"
    assert loaded.cache_sha256 == created.cache_sha256
    assert torch.equal(loaded.labels, created.labels)


def test_basin_cache_rejects_metadata_mismatch_without_overwrite(tmp_path):
    cache = tmp_path / "basin.npz"
    _build(cache)
    before = cache.read_bytes()
    with pytest.raises(ValueError, match="metadata mismatch"):
        GradientFlowBasinMap2D(
            _zero_grad,
            torch.tensor([[-0.5, 0.0], [0.5, 0.0]], dtype=torch.float64),
            (-1.0, -1.0), (1.0, 1.0), n_grid=5, device="cpu",
            cache=str(cache), dt_flow=0.2, n_flow=1)
    assert cache.read_bytes() == before


def test_legacy_basin_cache_requires_explicit_unverified_opt_in(tmp_path):
    cache = tmp_path / "legacy.npz"
    np.savez(cache, labels=np.zeros((5, 5), dtype=np.int64))
    before = cache.read_bytes()
    with pytest.raises(ValueError, match="legacy/incomplete"):
        _build(cache)
    with pytest.warns(RuntimeWarning, match="legacy basin cache"):
        legacy = _build(cache, allow_legacy_unverified=True)
    assert legacy.cache_validation_status == "legacy_unverified"
    assert legacy.cache_sha256 and cache.read_bytes() == before
