"""Shared fixtures: small analytic targets the tests can check against."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.potentials import Potential            # noqa: E402
from src.rng import EnsembleStreams             # noqa: E402
from src.samplers import RectBox, UnboundedBox  # noqa: E402
from src.targets import Target                  # noqa: E402

torch.set_default_dtype(torch.float64)


class IsotropicGaussian(Potential):
    """``V(x) = ||x||^2 / (2 sigma^2)``, so ``pi = N(0, sigma^2 / beta)``."""

    name = "isotropic_gaussian"

    def __init__(self, d: int = 1, sigma: float = 1.0) -> None:
        self.d = int(d)
        self.sigma = float(sigma)

    def V(self, x: torch.Tensor) -> torch.Tensor:
        return (x * x).sum(-1) / (2.0 * self.sigma ** 2)

    def grad_V(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.sigma ** 2

    def stationary_std(self, beta: float) -> float:
        return self.sigma / math.sqrt(beta)


class QuarticWell(Potential):
    """``V(x) = (x^2 - 1)^2`` in one dimension: a nonlinear force for chord tests."""

    d = 1
    name = "quartic_well"

    def V(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0]
        return (x1 * x1 - 1.0) ** 2

    def grad_V(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0]
        return (4.0 * x1 * (x1 * x1 - 1.0)).unsqueeze(-1)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def gaussian_target(device) -> Target:
    return Target(IsotropicGaussian(d=1, sigma=1.0), beta=2.0,
                  name="gaussian", device=device)


@pytest.fixture
def gaussian_target_2d(device) -> Target:
    return Target(IsotropicGaussian(d=2, sigma=1.0), beta=2.0,
                  name="gaussian2d", device=device)


@pytest.fixture
def quartic_target(device) -> Target:
    return Target(QuarticWell(), beta=1.5, name="quartic", device=device)


@pytest.fixture
def unbounded_box() -> UnboundedBox:
    return UnboundedBox()


def make_streams(seeds=(0,), device=torch.device("cpu"), *,
                 experiment="TEST", family="TEST", pair_group=None
                 ) -> EnsembleStreams:
    return EnsembleStreams(experiment, family, pair_group or {}, tuple(seeds),
                           device)


def tight_box(device, half_width: float, d: int) -> RectBox:
    return RectBox([-half_width] * d, [half_width] * d, device)
