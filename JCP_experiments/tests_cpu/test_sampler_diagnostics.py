from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jumps import ShellJumpLaw  # noqa: E402
from src.metrics import nonfinite_count, nonfinite_frac  # noqa: E402
from src.runner import CSV_BASE_COLUMNS, write_manifest, write_summary_csv  # noqa: E402
from src.samplers import (  # noqa: E402
    BAOAB,
    CompoundPoisson,
    MALA,
    ParallelTempering,
    RectBox,
    SamplerBase,
    ULA,
)
from src.score import (  # noqa: E402
    MultiAtomShellScore,
    RandomAtomicShellScore,
    _finalize,
)


class _ZeroPotential:
    def V(self, x):
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)

    def grad(self, x):
        return torch.zeros_like(x)


class _PushPotential(_ZeroPotential):
    def grad(self, x):
        return -torch.ones_like(x)


def _generators(seed: int = 1):
    gd = torch.Generator(device="cpu"); gd.manual_seed(seed)
    gj = torch.Generator(device="cpu"); gj.manual_seed(seed + 1)
    return gd, gj


def _assert_jump_accounting(diag, n: int, steps: int, dt: float):
    sampled = diag["jump_count_cumulative"]
    applied = diag["jump_count_applied_cumulative"]
    assert sampled >= applied >= 0
    assert diag["jump_rate_per_particle_time_cumulative"] == pytest.approx(
        sampled / (n * steps * dt))
    assert diag["jump_applied_rate_per_particle_time_cumulative"] == pytest.approx(
        applied / (n * steps * dt))


def test_all_compound_poisson_modes_report_lifetime_counts_and_rates():
    pot = _ZeroPotential()
    n, steps, dt = 96, 3, 0.2
    x0 = torch.zeros(n, 1, dtype=torch.float64)
    box = RectBox([-1e6], [1e6], "cpu")
    law = ShellJumpLaw(
        torch.tensor([[1.0], [-1.0]], dtype=torch.float64),
        torch.tensor([0.4, 0.6], dtype=torch.float64), h=0.0)

    gd, gj = _generators(10)
    full = CompoundPoisson(
        pot, x0, dt, 0.0, 2.0, law, gd, gj, box, jump_mode="full")
    for _ in range(steps):
        full.step()
    full_diag = full.pop_diagnostics()
    _assert_jump_accounting(full_diag, n, steps, dt)
    assert full_diag["jump_count_cumulative"] == full_diag["jump_count_applied_cumulative"]
    assert full_diag["jump_cap_k"] == 8
    assert full_diag["jump_cap_hit_count_cumulative"] == 0
    assert full_diag["jump_cap_excess_count_cumulative"] == 0

    gd, gj = _generators(20)
    atomic = CompoundPoisson(
        pot, x0, dt, 0.0, 2.0, law, gd, gj, box, jump_mode="atomic")
    for _ in range(steps):
        atomic.step()
    atomic_diag = atomic.pop_diagnostics()
    _assert_jump_accounting(atomic_diag, n, steps, dt)
    assert atomic_diag["jump_count_cumulative"] == atomic_diag["jump_count_applied_cumulative"]
    assert "jump_cap_k" not in atomic_diag

    score = MultiAtomShellScore(
        pot, law, lam=2.0, beta=1.0, q_theta=2, m_max=float("inf"))
    gd, gj = _generators(30)
    paired = CompoundPoisson(
        pot, x0, dt, 1.0, 2.0, law, gd, gj, box,
        score=score, jump_mode="paired_multiatom")
    for _ in range(steps):
        paired.step()
    paired_diag = paired.pop_diagnostics()
    _assert_jump_accounting(paired_diag, n, steps, dt)
    assert paired_diag["jump_count_cumulative"] == paired_diag["jump_count_applied_cumulative"]
    assert paired_diag["score_clip_count_cumulative"] == 0
    assert paired_diag["score_clip_fraction_cumulative"] == 0.0


@pytest.mark.parametrize("mode", ["full", "atomic", "paired_multiatom"])
def test_cp_modes_report_boundary_clips_per_applied_jump(monkeypatch, mode):
    # Force exactly one Poisson occurrence for every particle/atom.  A tiny dt
    # keeps the MA diffusion/score pre-jump state inside [-1,1], while the
    # deterministic +2 jump makes every post-jump candidate leave it.
    monkeypatch.setattr(
        torch, "poisson",
        lambda rates, generator=None: torch.ones_like(rates))
    n, dt = 7, 1e-12
    pot = _ZeroPotential()
    x0 = torch.zeros(n, 1, dtype=torch.float64)
    law = ShellJumpLaw(
        torch.tensor([[2.0]], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64), h=0.0)
    gd, gj = _generators(40)
    kwargs = {}
    eps = 0.0
    if mode == "paired_multiatom":
        eps = 1.0
        kwargs["score"] = MultiAtomShellScore(
            pot, law, lam=1.0, beta=1.0, q_theta=2)
    sampler = CompoundPoisson(
        pot, x0, dt, eps, 1.0, law, gd, gj,
        RectBox([-1.0], [1.0], "cpu"), jump_mode=mode, **kwargs)
    sampler.step()
    diag = sampler.pop_diagnostics()
    assert diag["jump_boundary_applied_count_cumulative"] == n
    assert diag["jump_boundary_clip_count_cumulative"] == n
    assert diag["jump_boundary_clip_fraction_per_applied_jump_cumulative"] == 1.0
    assert diag["state_box_clip_count_cumulative"] == n


def test_zero_applied_jumps_report_finite_zero_boundary_fraction(monkeypatch):
    monkeypatch.setattr(
        torch, "poisson",
        lambda rates, generator=None: torch.zeros_like(rates))
    pot = _ZeroPotential()
    law = ShellJumpLaw(
        torch.tensor([[2.0]], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64), h=0.0)
    gd, gj = _generators(43)
    sampler = CompoundPoisson(
        pot, torch.zeros(3, 1), 0.1, 0.0, 1.0, law, gd, gj,
        RectBox([-1.0], [1.0], "cpu"), jump_mode="full")
    sampler.step()
    diag = sampler.pop_diagnostics()
    assert diag["jump_boundary_applied_count_cumulative"] == 0
    assert diag["jump_boundary_clip_count_cumulative"] == 0
    assert diag["jump_boundary_clip_fraction_per_applied_jump_cumulative"] == 0.0


def test_full_mode_preserves_exact_pre_cap_count_and_cap_diagnostics():
    pot = _ZeroPotential()
    n, dt = 128, 1.0
    x0 = torch.zeros(n, 1, dtype=torch.float64)
    law = ShellJumpLaw(torch.tensor([[1.0]], dtype=torch.float64),
                       torch.tensor([1.0], dtype=torch.float64), h=0.0)
    gd, gj = _generators(44)
    sampler = CompoundPoisson(
        pot, x0, dt, 0.0, 20.0, law, gd, gj,
        RectBox([-1e6], [1e6], "cpu"), jump_mode="full")
    sampler.step()
    diag = sampler.pop_diagnostics()
    assert diag["jump_cap_hit_count_cumulative"] > 0
    assert diag["jump_cap_hit_fraction_cumulative"] > 0
    assert diag["jump_count_cumulative"] > diag["jump_count_applied_cumulative"]
    assert diag["jump_cap_excess_count_cumulative"] == (
        diag["jump_count_cumulative"] - diag["jump_count_applied_cumulative"])


def test_state_clipping_and_mala_pt_outside_rejection_are_cumulative():
    n, steps = 12, 3
    x0 = torch.zeros(n, 1, dtype=torch.float64)
    zero_box = RectBox([0.0], [0.0], "cpu")
    gen = torch.Generator(device="cpu"); gen.manual_seed(50)
    ula = ULA(_PushPotential(), x0, 1.0, 0.0, gen, zero_box)
    for _ in range(steps):
        ula.step()
    first = ula.pop_diagnostics()
    assert first["state_box_clip_count_cumulative"] == n * steps
    assert first["state_box_clip_fraction_cumulative"] == 1.0
    ula.step()
    second = ula.pop_diagnostics()
    assert second["state_box_clip_count_cumulative"] == n * (steps + 1)

    gen = torch.Generator(device="cpu"); gen.manual_seed(51)
    mala = MALA(_ZeroPotential(), x0, 0.1, 1.0, gen, zero_box)
    for _ in range(steps):
        mala.step()
    md = mala.pop_diagnostics()
    assert md["outside_proposal_reject_count_cumulative"] == n * steps
    assert md["outside_proposal_reject_fraction_cumulative"] == 1.0
    assert md["mala_accept"] == 0.0
    assert md["mala_accept_count_cumulative"] == 0
    assert md["mala_proposal_count_cumulative"] == n * steps
    assert md["mala_accept_fraction_cumulative"] == 0.0

    gen = torch.Generator(device="cpu"); gen.manual_seed(52)
    betas = torch.tensor([2.0, 1.0], dtype=torch.float64)
    pt = ParallelTempering(_ZeroPotential(), x0, 0.1, betas, gen, zero_box)
    for _ in range(steps):
        pt.step()
    pd = pt.pop_diagnostics()
    assert pd["outside_proposal_reject_count_cumulative"] == 2 * n * steps
    assert pd["outside_proposal_reject_fraction_cumulative"] == 1.0


def test_nonfinite_count_and_output_wiring(tmp_path):
    x = torch.tensor([[0.0], [float("nan")], [float("inf")], [2.0]])
    assert nonfinite_count(x) == 2
    assert nonfinite_frac(x) == 0.5

    required = {
        "nonfinite_count", "state_box_clip_fraction_cumulative",
        "outside_proposal_reject_fraction_cumulative",
        "nonfinite_proposal_count_cumulative",
        "nonfinite_proposal_fraction_cumulative",
        "score_clip_count_cumulative", "score_clip_fraction_cumulative",
        "mala_accept_count_cumulative", "mala_proposal_count_cumulative",
        "mala_accept_fraction_cumulative",
        "pt_swap_accept_count_cumulative", "pt_swap_proposal_count_cumulative",
        "pt_swap_accept_fraction_cumulative",
        "jump_count_cumulative", "jump_rate_per_particle_time_cumulative",
        "jump_boundary_clip_count_cumulative",
        "jump_boundary_applied_count_cumulative",
        "jump_boundary_clip_fraction_per_applied_jump_cumulative",
        "jump_cap_hit_count_cumulative",
    }
    assert required.issubset(CSV_BASE_COLUMNS)
    row = {
        "method": "CP", "seed": 0, "step": 2, "t": 0.2,
        "wallclock_s": 0.1, "nfe": 2, "TV": 0.1,
        "nonfinite_count": 0, "nonfinite_frac": 0.0,
        "jump_count_cumulative": 7,
        "jump_rate_per_particle_time_cumulative": 1.75,
        "state_box_clip_fraction_cumulative": 0.0,
        "jump_cap_hit_count_cumulative": 0,
    }
    info = {"CP": {key: row[key] for key in required if key in row}}
    out = tmp_path / "summary.csv"
    write_summary_csv([row], ["CP"], [0], ["TV"], info,
                      {"TV": {"mean": 0.01}}, out)
    summary = next(csv.DictReader(out.open()))
    assert summary["nonfinite_count_mean"] == "0.0"
    assert summary["jump_count_cumulative"] == "7"
    assert summary["jump_rate_per_particle_time_cumulative"] == "1.75"

    manifest = tmp_path / "manifest.json"
    write_manifest(manifest, method_info=info)
    payload = json.loads(manifest.read_text())
    assert payload["method_info"]["CP"]["jump_count_cumulative"] == 7


def test_step_paths_contain_no_host_sync_calls():
    functions = [
        SamplerBase._clip_state, SamplerBase._record_outside_proposals,
        CompoundPoisson._record_jump_counts,
        CompoundPoisson._record_jump_boundary, CompoundPoisson._step_full,
        CompoundPoisson._step_atomic, CompoundPoisson._step_paired_multiatom,
        MALA.step, ParallelTempering.step, ULA.step,
    ]
    for function in functions:
        source = inspect.getsource(function)
        assert ".item(" not in source
        assert ".cpu(" not in source


class _QuadraticPotential(_ZeroPotential):
    def grad(self, x):
        return x


def test_clip_state_preserves_nonfinite_rows_and_separates_diagnostics():
    sampler = SamplerBase()
    sampler.box = RectBox([-1.0], [1.0], "cpu")
    candidate = torch.tensor(
        [[float("inf")], [float("nan")], [2.0], [0.0]],
        dtype=torch.float64)
    result = sampler._clip_state(candidate)
    assert torch.isinf(result[0]).all()
    assert torch.isnan(result[1]).all()
    assert result[2, 0] == 1.0 and result[3, 0] == 0.0
    diag = sampler.pop_diagnostics()
    assert diag["nonfinite_proposal_count_cumulative"] == 2
    assert diag["nonfinite_proposal_fraction_cumulative"] == 0.5
    assert diag["state_box_clip_count_cumulative"] == 1
    assert diag["state_box_clip_fraction_cumulative"] == 0.25

    mh = SamplerBase()
    mh.box = sampler.box
    inside = mh.box.contains(candidate)
    mh._record_outside_proposals(candidate, inside)
    mh_diag = mh.pop_diagnostics()
    assert mh_diag["nonfinite_proposal_count_cumulative"] == 2
    assert mh_diag["outside_proposal_reject_count_cumulative"] == 1


def test_baoab_caches_force_at_clipped_stored_position():
    x0 = torch.zeros(1, 1, dtype=torch.float64)
    gen = torch.Generator(device="cpu"); gen.manual_seed(91)
    sampler = BAOAB(
        _QuadraticPotential(), x0, dt=1.0, eps=0.0, gen=gen,
        box=RectBox([-1.0], [1.0], "cpu"))
    sampler.p.fill_(10.0)
    sampler.step()
    assert sampler.positions()[0, 0] == 1.0
    assert torch.allclose(
        sampler.f, -sampler.positions() / (1.0 + sampler.positions().abs()))
    assert sampler.pop_diagnostics()["state_box_clip_count_cumulative"] == 1


def test_shell_geometry_and_multiatom_units_fail_closed():
    weights = torch.tensor([1.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="nonzero norm"):
        ShellJumpLaw(torch.tensor([[0.0]], dtype=torch.float64), weights, h=0.0)
    with pytest.raises(ValueError, match="finite"):
        ShellJumpLaw(torch.tensor([[float("nan")]], dtype=torch.float64), weights, h=0.0)
    with pytest.raises(ValueError, match="weights"):
        ShellJumpLaw(torch.tensor([[1.0]], dtype=torch.float64),
                     torch.tensor([-1.0], dtype=torch.float64), h=0.0)
    with pytest.raises(ValueError, match="half-width"):
        ShellJumpLaw(torch.tensor([[1.0]], dtype=torch.float64), weights, h=-0.1)

    law = ShellJumpLaw(torch.tensor([[1.0]], dtype=torch.float64), weights, h=0.0)
    law.units = torch.tensor([[2.0]], dtype=torch.float64)
    with pytest.raises(ValueError, match="normalized and aligned"):
        MultiAtomShellScore(_ZeroPotential(), law, lam=1.0, beta=1.0)


def test_paired_multiatom_rejects_zero_diffusion_temperature():
    pot = _ZeroPotential()
    law = ShellJumpLaw(
        torch.tensor([[1.0]], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64), h=0.0)
    score = MultiAtomShellScore(pot, law, lam=1.0, beta=1.0, q_theta=2)
    gd, gj = _generators(101)
    with pytest.raises(ValueError, match="eps = 1"):
        CompoundPoisson(
            pot, torch.zeros(2, 1), 0.1, 0.0, 1.0, law, gd, gj,
            RectBox([-10.0], [10.0], "cpu"), score=score,
            jump_mode="paired_multiatom")



def _assert_exact_clip_diagnostics(diag, expected_count, expected_total):
    assert diag["m_clip_count"].dtype == torch.int64
    assert diag["m_clip_total"].dtype == torch.int64
    assert diag["m_clip_count"].device == diag["m_clip_fraction"].device
    assert diag["m_clip_total"].device == diag["m_clip_fraction"].device
    assert int(diag["m_clip_count"]) == expected_count
    assert int(diag["m_clip_total"]) == expected_total
    assert float(diag["m_clip_fraction"]) == pytest.approx(
        expected_count / expected_total)


def test_score_paths_emit_exact_device_integer_clip_counts():
    M = torch.tensor([-1.0, 2.0, 3.0], dtype=torch.float64)
    _, exact_diag = _finalize(M, torch.ones(3, 1), m_max=1.0)
    _assert_exact_clip_diagnostics(exact_diag, 2, 3)

    pot = _ZeroPotential()
    law = ShellJumpLaw(
        torch.tensor([[1.0], [-1.0]], dtype=torch.float64),
        torch.tensor([0.5, 0.5], dtype=torch.float64), h=0.0)
    x = torch.zeros(3, 1, dtype=torch.float64)
    shifts = torch.ones_like(x)
    ra = RandomAtomicShellScore(
        pot, law, lam=1.0, beta=1.0, q_theta=2, m_max=-1.0)
    _, ra_diag = ra.score_for_shift(x, shifts)
    _assert_exact_clip_diagnostics(ra_diag, 3, 3)

    ma = MultiAtomShellScore(
        pot, law, lam=1.0, beta=1.0, q_theta=2, m_max=-1.0)
    bank = law.atoms.unsqueeze(0).expand(3, -1, -1)
    _, ma_diag = ma.score_for_bank(x, bank)
    _assert_exact_clip_diagnostics(ma_diag, 3, 3)



def test_pt_swap_acceptance_uses_exact_lifetime_counts():
    n, steps = 5, 4
    x0 = torch.zeros(n, 1, dtype=torch.float64)
    gen = torch.Generator(device="cpu"); gen.manual_seed(120)
    pt = ParallelTempering(
        _ZeroPotential(), x0, 0.01,
        torch.tensor([2.0, 1.0], dtype=torch.float64), gen,
        RectBox([-100.0], [100.0], "cpu"), n_swap=1)
    for _ in range(steps):
        pt.step()
    diag = pt.pop_diagnostics()
    # With identical zero energies every attempted adjacent swap accepts. For
    # K=2, alternating passes attempt swaps on two of the four steps.
    assert diag["pt_swap_accept_count_cumulative"] == 2 * n
    assert diag["pt_swap_proposal_count_cumulative"] == 2 * n
    assert diag["pt_swap_accept_fraction_cumulative"] == 1.0
    assert diag["mala_proposal_count_cumulative"] == 2 * n * steps
