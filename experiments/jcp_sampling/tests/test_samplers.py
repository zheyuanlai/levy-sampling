
import torch

from experiments.jcp_sampling.core.jump_banks import double_well_shell
from experiments.jcp_sampling.core.levy_score import stationary_levy_score
from experiments.jcp_sampling.core.potentials import DoubleWell1D
from experiments.jcp_sampling.core.samplers import LevyScoreJumpDiffusion, torch_generator


class _FlatDoubleWell(DoubleWell1D):
    """Flat (zero) potential: force and (for a symmetric bank) stationary score both vanish."""

    def potential(self, x):
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)

    def gradient(self, x):
        return torch.zeros_like(x)


def test_rawcp_and_lscp_share_jump_stream():
    """Raw CP and LSC-CP use one common RNG stream: on a flat symmetric target where both the
    force and the stationary score vanish, identical seeds must give bit-identical trajectories,
    proving the Brownian noise and jump increments are drawn identically (common random numbers).
    """
    flat = _FlatDoubleWell(beta=1.0)
    bank = double_well_shell((-1.0, 1.0), scale=1.0, intensity=1.0)  # symmetric => sum_e w_e r_e = 0
    x0 = torch.zeros(500, 1)
    ga = torch_generator(7, torch.device("cpu"))
    gb = torch_generator(7, torch.device("cpu"))
    sa = LevyScoreJumpDiffusion(flat, 0.01, bank=bank, use_score=True, n_theta=6)
    sb = LevyScoreJumpDiffusion(flat, 0.01, bank=bank, use_score=False, n_theta=6)
    xa, xb = x0.clone(), x0.clone()
    for _ in range(30):
        xa, _ = sa.step(xa, ga)
        xb, _ = sb.step(xb, gb)
    assert float((xa - xb).abs().max()) == 0.0


def test_use_score_changes_drift():
    """use_score=True actually injects the nonzero stationary correction on a real target."""
    pot = DoubleWell1D(beta=4.0)
    bank = double_well_shell(scale=1.0, intensity=1.0)
    x = torch.linspace(-1.5, 1.5, 11).reshape(-1, 1)
    S = stationary_levy_score(pot.potential, x, bank, pot.beta, n_theta=8)
    assert float(S.abs().max()) > 1e-3
