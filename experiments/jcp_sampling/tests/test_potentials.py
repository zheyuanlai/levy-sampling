
import math

import torch

from experiments.jcp_sampling.core.potentials import (
    DoubleWellBarrier1D, TripleWell1D, TransformedMuellerBrown10D, build_potential)


def test_double_well_barrier_height_and_symmetry():
    for H in (0.25, 1.0, 4.0):
        pot = DoubleWellBarrier1D(H=H, eps=0.5)
        # barrier height is exactly H: V(0)=H, wells V(+-1)=0
        assert abs(float(pot.potential(torch.zeros(1, 1))) - H) < 1e-6
        assert float(pot.potential(torch.ones(1, 1)).abs()) < 1e-6
        assert pot.beta == 2.0  # beta = 1/eps
        assert pot.basin_labels(torch.tensor([[-0.5], [0.5]])).tolist() == [0, 1]
        # 50/50 reference for the symmetric well
        ref = pot.reference(200000, 0)
        frac = torch.bincount(pot.basin_labels(ref), minlength=2).float() / ref.shape[0]
        assert torch.allclose(frac, torch.tensor([0.5, 0.5]), atol=0.01)


def test_double_well_barrier_gradient_matches_fd():
    pot = DoubleWellBarrier1D(H=2.0, eps=0.25)
    x = torch.tensor([[-1.3], [0.2], [0.9]], dtype=torch.float64)
    g = pot.gradient(x)[:, 0]
    h = 1e-4
    gn = (pot.potential(x + h) - pot.potential(x - h)) / (2 * h)
    assert float((g - gn).abs().max()) < 1e-4


def test_double_well_barrier_build_dispatch():
    pot = build_potential({"kind": "double_well_barrier", "target_cfg": {"H": 3.0, "eps": 0.25}})
    assert isinstance(pot, DoubleWellBarrier1D) and pot.H == 3.0 and pot.eps == 0.25


def test_triplewell_reference_matches_target_probs():
    tw = TripleWell1D(eps=0.08)
    ref = tw.reference(200000, 0)
    frac = torch.bincount(tw.basin_labels(ref), minlength=3).float() / ref.shape[0]
    assert torch.allclose(frac, tw.target_basin_probs(), atol=0.01)


def test_triplewell_gradient_matches_fd():
    tw = TripleWell1D(eps=0.08)
    x = torch.tensor([[-2.7], [0.1], [2.9]])
    g = tw.gradient(x)[:, 0]
    h = 1e-3
    gn = (tw.potential(x + h) - tw.potential(x - h)) / (2 * h)
    assert float((g - gn).abs().max()) < 1e-3


def test_alanine_fes_interpolant_and_populations(tmp_path):
    """AlanineFES2D ingests a real-format FES npz: analytic Fourier gradient matches FD, the
    interpolant reproduces a band-limited FES, and basin populations follow the grid Boltzmann
    weight. Uses a band-limited synthetic FES so the interpolant is (near) exact."""
    import numpy as np
    from experiments.jcp_sampling.core.potentials import AlanineFES2D
    Ng = 64
    ax = -math.pi + 2 * math.pi * np.arange(Ng) / Ng
    PHI, PSI = np.meshgrid(ax, ax, indexing="ij")
    # band-limited FES with two basins (minima near (-pi/2, +pi/2) and (+pi/2, -pi/2))
    F = -np.cos(PHI + math.pi / 2) - np.cos(PSI - math.pi / 2) + 0.5 * np.cos(PHI + PSI)
    F = F - F.min()
    minima = np.array([[-math.pi / 2, math.pi / 2], [math.pi / 2, -math.pi / 2]], dtype=np.float32)
    p = tmp_path / "fes.npz"
    np.savez(p, phi=ax, psi=ax, F=F, kT=0.6, minima=minima)
    pot = AlanineFES2D(fes_path=str(p), n_modes=8)
    assert pot.beta == 1.0 / 0.6
    # analytic gradient matches finite differences of the interpolant
    x = torch.tensor([[-1.4, 1.3], [1.2, -1.1], [0.2, 0.3]], dtype=torch.float64)
    g = pot.gradient(x)
    gn = torch.zeros_like(g)
    for i in range(2):
        e = torch.zeros(1, 2, dtype=torch.float64); e[0, i] = 1e-4
        gn[:, i] = (pot.potential(x + e) - pot.potential(x - e)) / 2e-4
    assert float((g - gn).abs().max()) < 1e-4
    # reference honors the grid Boltzmann populations
    ref = pot.reference(200000, 0)
    frac = torch.bincount(pot.basin_labels(ref), minlength=2).float() / ref.shape[0]
    assert torch.allclose(frac, pot.target_basin_probs(), atol=0.02)


def test_muller10d_reference_lifting_and_multibasin():
    mb = TransformedMuellerBrown10D(eps=0.5, scale=0.02)
    ref = mb.reference(200000, 1)
    frac = torch.bincount(mb.basin_labels(ref), minlength=3).float() / ref.shape[0]
    assert torch.allclose(frac, mb.target_basin_probs(), atol=0.02)
    # each lifted minimum projects back into its own basin
    assert mb.basin_labels(mb.minima()).tolist() == [0, 1, 2]
    # genuinely multi-basin at eps=0.5 (not the degenerate near-unimodal regime)
    assert float(mb.target_basin_probs().min()) > 0.02


def test_muller10d_gradient_matches_fd():
    mb = TransformedMuellerBrown10D(eps=0.5, scale=0.02)
    x = torch.randn(4, 10) * 0.4
    g = mb.gradient(x)
    gn = torch.zeros_like(g)
    for i in range(10):
        e = torch.zeros(1, 10); e[0, i] = 1e-3
        gn[:, i] = (mb.potential(x + e) - mb.potential(x - e)) / (2e-3)
    assert float((g - gn).abs().max()) < 5e-3
