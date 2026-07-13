"""RA-LSC state independence (rho independent of x).

The simple invariance argument (formulation note 17.4) requires the realised
displacement R ~ rho to be drawn independently of the current state X. This is
enforced structurally: the jump laws' `.sample` take no state, the estimator
RECEIVES R (never draws it from x), and the atomic sampler draws R from the
state-free jump stream at the start of the step. These tests pin those
guarantees so a future refactor cannot silently make atom selection
state-dependent.
"""
import inspect

import torch

from tests.conftest import CACHE_DIR  # noqa: F401  (selects the GPU)
from src.jumps import ShellJumpLaw, AnnulusJumpLaw
from src.score import RandomAtomicShellScore
from src.samplers import CompoundPoisson

DEV = "cuda"


def test_jump_laws_sample_takes_no_state():
    assert list(inspect.signature(ShellJumpLaw.sample).parameters) == ["self", "n", "gen"]
    assert list(inspect.signature(AnnulusJumpLaw.sample).parameters) == ["self", "n", "gen"]


def test_estimator_receives_shift_not_draws_it():
    # score_for_shift takes R as an argument; it must not sample R from state
    assert list(inspect.signature(RandomAtomicShellScore.score_for_shift).parameters) \
        == ["self", "x", "R"]
    src = inspect.getsource(RandomAtomicShellScore.score_for_shift)
    assert "law.sample" not in src, "estimator must not draw its own displacement"


def test_atomic_sampler_draws_state_free_shift():
    src = inspect.getsource(CompoundPoisson._step_atomic)
    assert "self.law.sample(n, self.gen_jump)" in src


def test_sample_is_deterministic_given_seed():
    law = ShellJumpLaw(torch.tensor([[2.0], [-2.0]], dtype=torch.float64, device=DEV),
                       torch.tensor([0.5, 0.5], dtype=torch.float64, device=DEV), 0.2)
    g1 = torch.Generator(device=DEV); g1.manual_seed(7)
    g2 = torch.Generator(device=DEV); g2.manual_seed(7)
    assert torch.equal(law.sample(64, g1), law.sample(64, g2))
