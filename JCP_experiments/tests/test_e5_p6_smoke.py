"""E5 P6 gate: build_e5_alanine + TorusBox + the seven-method smoke.

Gate (task S P6):
  * no non-finite values (nonfinite_count == 0 per method);
  * LSC-CP (RA and MA) reach the positive-phi island while ULA/MALA/BAOAB stay
    trapped (island occupancy ~ 0);
  * raw CP reaches the island but with measurably larger basin-mass error.

Sizing note: the per-step cost of this potential is dominated by kernel-launch
overhead, not by the ensemble (ULA measured 31 ms/step at N = 64, 1000 and
4000 alike), so the smoke buys its jump statistics with a LARGE ensemble and few
steps rather than the reverse. Expected jumps ~ N * lam * dt * n_steps.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from tests.conftest import CACHE_DIR  # noqa: F401  (selects the GPU)
from src.e5_alanine.build_reference import CACHE_DIR as E5_CACHE

DEV = "cuda"

SMOKE_N = 2000
SMOKE_STEPS = 600
LOCAL_METHODS = ("ULA", "MALA", "BAOAB")
LSC_METHODS = ("LSC-CP-RA", "LSC-CP-MA")
ALL_METHODS = ("ULA", "MALA", "FLA", "BAOAB", "PT", "CP",
               "CP-RA", "LSC-CP-RA", "LSC-CP-MA")


def _exp():
    if not os.path.exists(os.path.join(E5_CACHE, "reference.npz")):
        pytest.skip("E5 reference cache not generated")
    from src.experiments import build_e5_alanine
    return build_e5_alanine(device=DEV, n_particles=SMOKE_N, seeds=(0,))


def _run_smoke(exp, methods=ALL_METHODS, n_steps=SMOKE_STEPS):
    from src.experiments import make_batched_factory
    from src.samplers import geometric_ladder
    pt_betas = geometric_ladder(exp.cfg.beta, exp.pt_beta_min, 6, DEV)
    factory = make_batched_factory(exp, exp.cfg.dt, pt_betas, seeds=(0,),
                                   n_particles=SMOKE_N)
    island = set(exp.extras["positive_phi_basins"])
    out = {}
    for m in methods:
        s = factory(m)
        for _ in range(n_steps):
            s.step()
        pos = s.positions()
        labels = exp.labels_fn(pos)
        occ = float(sum((labels == k).to(torch.float64).mean().item()
                        for k in island))
        p_hat = torch.stack([(labels == k).to(torch.float64).mean()
                             for k in range(exp.p_star.shape[0])])
        basin_err = float((p_hat - exp.p_star).abs().sum().item())
        out[m] = {
            "nonfinite": int((~torch.isfinite(pos)).any(dim=-1).sum().item()),
            "island_occupancy": occ,
            "basin_L1": basin_err,
            "finite_energy": bool(torch.isfinite(exp.pot.V(pos)).all().item()),
            "in_box": bool(exp.box.contains(pos).all().item()),
        }
    return out


@pytest.fixture(scope="module")
def smoke():
    exp = _exp()
    res = _run_smoke(exp)
    for m, r in res.items():
        print(f"{m:11s} island_occ={r['island_occupancy']:.4f} "
              f"basin_L1={r['basin_L1']:.3f} nonfinite={r['nonfinite']}")
    return exp, res


def test_no_nonfinite_values(smoke):
    _, res = smoke
    for m, r in res.items():
        assert r["nonfinite"] == 0, (m, r)
        assert r["finite_energy"], m
        assert r["in_box"], m


def test_locals_stay_trapped(smoke):
    """ULA/MALA/BAOAB cannot cross the phi barrier in the smoke horizon."""
    _, res = smoke
    for m in LOCAL_METHODS:
        assert res[m]["island_occupancy"] == 0.0, (m, res[m])


def test_lsc_cp_reaches_the_island(smoke):
    """Both practical estimators cross; so does raw CP (same jump geometry)."""
    _, res = smoke
    for m in LSC_METHODS:
        assert res[m]["island_occupancy"] > 0.0, (m, res[m])
    assert (res["CP"]["island_occupancy"] > 0.0
            or res["CP-RA"]["island_occupancy"] > 0.0), res


def test_raw_cp_is_more_biased_than_lsc_cp(smoke):
    """Raw CP reaches the island but misplaces basin mass: its uncorrected jump
    kernel does not preserve the target."""
    _, res = smoke
    best_lsc = min(res[m]["basin_L1"] for m in LSC_METHODS)
    raw = max(res["CP"]["basin_L1"], res["CP-RA"]["basin_L1"])
    print(f"basin L1: best LSC-CP = {best_lsc:.3f}, raw CP = {raw:.3f}")
    assert raw > best_lsc, (raw, best_lsc)


def test_experiment_wiring(smoke):
    exp, _ = smoke
    assert exp.name == "alanine_dipeptide"
    assert exp.cfg.d == 60
    assert abs(exp.cfg.beta * exp.cfg.eps - 1.0) < 1e-12
    # local beta, not the global config.BETA = 8
    from src.config import BETA
    assert abs(exp.cfg.beta - BETA) > 1.0
    assert exp.metric_space(exp.extras["reference"].qt[:8]).shape == (8, 2)
    assert exp.kramers_tau > 0.0 and exp.pt_beta_min > 0.0
