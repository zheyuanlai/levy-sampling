
import torch

from experiments.jcp_sampling.core.jump_banks import (
    double_well_shell,
    minima_complete_graph,
    random_matched_length_control,
    manywell_block_flip,
    build_jump_bank,
)


def test_double_well_shell_weights_and_vectors():
    bank = double_well_shell(scale=1.0, intensity=0.3)
    assert bank.size == 2
    assert torch.allclose(bank.weights.sum(), torch.tensor(1.0))
    assert torch.allclose(bank.vectors[:, 0].abs(), torch.tensor([2.0, 2.0]))
    assert bank.intensity == 0.3


def test_complete_graph_and_random_control_lengths():
    minima = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    bank = minima_complete_graph(minima, intensity=1.0)
    ctrl = random_matched_length_control(bank, seed=123)
    assert bank.size == 6
    assert ctrl.size == bank.size
    assert torch.allclose(torch.sort(ctrl.vectors.norm(dim=1)).values, torch.sort(bank.vectors.norm(dim=1)).values, atol=1e-5)


def test_manywell_block_flip_rates():
    bank = manywell_block_flip(n_blocks=3, displacement=3.46, intensity_per_block=0.1)
    assert bank.size == 6
    assert bank.dim == 6
    assert abs(bank.intensity - 0.3) < 1e-8
    assert torch.allclose(bank.rates, torch.full((6,), 0.05))


def test_build_jump_bank_keeps_non_intensity_based_banks():
    from experiments.jcp_sampling.core.potentials import FourWell2D, ManyWell

    four = FourWell2D()
    ctrl = build_jump_bank(
        "random_matched_length_control",
        four,
        {
            "kind": "random_matched_length_control",
            "reference": {"kind": "minima_edge_graph", "edges": [[0, 1], [0, 2]], "intensity": 0.5},
            "seed": 7,
        },
    )
    assert ctrl.name == "random_matched_length_control"
    assert ctrl.intensity == 0.5
    assert ctrl.size == 4

    many = ManyWell(n_blocks=3)
    bank = build_jump_bank(
        "manywell_block_flip",
        many,
        {"kind": "manywell_block_flip", "intensity_per_block": 0.2},
    )
    assert bank.name == "manywell_block_flip"
    assert abs(bank.intensity - 0.6) < 1e-8
    assert bank.size == 6
