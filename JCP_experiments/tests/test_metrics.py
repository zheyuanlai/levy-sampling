"""Metric validation (spec section 9.4) + EPS round-trip (spec section 10)."""
import math
import os
import shutil
import subprocess
import tempfile

import torch

from src import metrics as M
from src.plotting import apply_style, plot_metric

DEV = "cuda"


def test_mmd_nonnegative_and_zero_on_self():
    gen = torch.Generator(device=DEV)
    gen.manual_seed(0)
    x = torch.randn(800, 3, generator=gen, device=DEV)
    y = torch.randn(800, 3, generator=gen, device=DEV) + 0.3
    bw = M.median_heuristic(x)
    assert M.mmd_biased(x, y, bw) >= 0.0
    assert M.mmd_biased(x, x, bw) < 1e-7
    # biased V-statistic is ||mu_X - mu_Y||_H^2, strictly positive here
    assert M.mmd_biased(x, y, bw) > 0.01


def test_sliced_w2_self_and_floor():
    gen = torch.Generator(device=DEV)
    gen.manual_seed(1)
    proj = M.make_projections(4, 200, device=DEV)
    x = torch.randn(2000, 4, generator=gen, device=DEV)
    assert M.sliced_w2(x, x, proj) < 1e-12
    # independent same-law samples sit at the ~N^{-1/2} bias floor, not 0
    y = torch.randn(2000, 4, generator=gen, device=DEV)
    floor = M.sliced_w2(x, y, proj)
    assert 1e-3 < floor < 0.2


def test_w2_exact_1d():
    gen = torch.Generator(device=DEV)
    gen.manual_seed(2)
    x = torch.randn(5000, 1, generator=gen, device=DEV)
    assert M.w2_exact_1d(x, x) < 1e-12
    assert abs(M.w2_exact_1d(x, x + 1.0) - 1.0) < 1e-9


def test_ejs_bounds():
    p = torch.tensor([0.25, 0.25, 0.5], device=DEV)
    assert M.ejs(p, p) == 0.0
    disj_a = torch.tensor([1.0, 0.0], device=DEV)
    disj_b = torch.tensor([0.0, 1.0], device=DEV)
    assert abs(M.ejs(disj_a, disj_b) - 1.0) < 1e-12
    q = torch.tensor([0.3, 0.3, 0.4], device=DEV)
    assert 0.0 < M.ejs(p, q) < 1.0


def test_emc():
    K = 8
    uni = torch.full((K,), 1.0 / K, device=DEV)
    assert abs(M.emc(uni) - 1.0) < 1e-12
    conc = torch.zeros(K, device=DEV)
    conc[0] = 1.0
    assert abs(M.emc(conc) - 1.0 / K) < 1e-12


def test_occupancy_tv():
    p = torch.tensor([1.0, 0.0], device=DEV)
    q = torch.tensor([0.0, 1.0], device=DEV)
    assert M.occupancy_tv(p, q) == 1.0
    assert M.occupancy_tv(p, p) == 0.0


def test_eps_roundtrip_ghostscript():
    """EPS must contain no transparency; verify by round-tripping one saved
    EPS through Ghostscript (exit code 0, non-empty output)."""
    if shutil.which("gs") is None:
        raise RuntimeError("Ghostscript required for EPS verification")
    apply_style()
    rows = []
    for seed in (0, 1, 2):
        for step in (10, 20, 30):
            rows.append({"method": "ULA", "seed": seed, "step": step,
                         "t": step * 0.01, "wallclock_s": step * 0.002,
                         "W2": 1.0 / step + 0.01 * seed})
            rows.append({"method": "LSC-CP", "seed": seed, "step": step,
                         "t": step * 0.01, "wallclock_s": step * 0.004,
                         "W2": 0.5 / step + 0.01 * seed})
    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "W2_vs_time")
        plot_metric(rows, "W2", "t", base, floor=0.01,
                    methods=("ULA", "LSC-CP"))
        eps = base + ".eps"
        assert os.path.exists(eps) and os.path.getsize(eps) > 0
        out = os.path.join(td, "roundtrip.pdf")
        r = subprocess.run(["gs", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pdfwrite",
                            f"-sOutputFile={out}", eps],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert os.path.getsize(out) > 0
        assert os.path.exists(base + ".pdf") and os.path.exists(base + ".png")
        assert os.path.exists(base + ".txt")


def test_bias_floors_shape():
    def sample_ref(n, gen):
        return torch.randn(n, 2, generator=gen, device=DEV)

    proj = M.make_projections(2, 50, device=DEV)
    floors = M.bias_floors(sample_ref,
                           {"W2": lambda a, b: M.sliced_w2(a, b, proj)},
                           {"absmean": lambda a: a.mean().abs().item()},
                           n=500, replicates=5, device=DEV)
    assert set(floors) == {"W2", "absmean"}
    for v in floors.values():
        assert v["mean"] >= 0.0 and v["std"] >= 0.0
