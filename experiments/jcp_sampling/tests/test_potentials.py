
import torch

from experiments.jcp_sampling.core.potentials import TripleWell1D, TransformedMuellerBrown10D


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
