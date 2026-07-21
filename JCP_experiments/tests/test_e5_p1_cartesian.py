"""E5 P1 gate: batched torch Cartesian force field matches OpenMM.

Gate (task S P1) over >= 1000 configs (reference conformer + Gaussian
perturbations at several scales + distorted high-energy ones):
  * energy matches OpenMM to rel 1e-6 (abs 1e-4 kJ/mol near zero);
  * forces match to rel 1e-5 in L2;
  * grad matches central finite differences to 1e-6;
  * units identical to OpenMM (energies compared directly in kJ/mol).
"""
from __future__ import annotations

import numpy as np
import torch

from tests.conftest import CACHE_DIR  # noqa: F401  (selects the GPU)
from src.e5_alanine.system import (build_alanine_system,
                                   openmm_energy_force,
                                   openmm_reference_context)
from src.e5_alanine.cartesian import AlanineDipeptideCartesian

DEV = "cuda"


def _make_configs(p0: np.ndarray, n: int = 1024) -> np.ndarray:
    """Reference + Gaussian perturbations across scales + distorted configs."""
    rng = np.random.default_rng(20260721)
    scales = np.array([0.0, 0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 0.05])
    hi = np.array([0.08, 0.12])                     # a few distorted high-energy
    configs = [p0.copy()]
    for k in range(n - 1):
        s = hi[k % 2] if k % 64 == 0 else scales[k % len(scales)]
        configs.append(p0 + s * rng.standard_normal(p0.shape))
    return np.stack(configs)


def test_torch_ff_matches_openmm_energy_force():
    ala = build_alanine_system()
    ctx = openmm_reference_context(ala.system)
    pot = AlanineDipeptideCartesian(device=DEV)

    configs = _make_configs(ala.positions_nm, 1024)
    assert configs.shape[0] >= 1000

    x = torch.tensor(configs.reshape(configs.shape[0], -1), device=DEV)
    E_t = pot.V(x).cpu().numpy()
    F_t = (-pot.grad(x)).reshape(configs.shape[0], ala.system.getNumParticles(),
                                 3).cpu().numpy()

    max_relE, max_relF = 0.0, 0.0
    for k in range(configs.shape[0]):
        E_omm, F_omm = openmm_energy_force(ctx, configs[k])
        relE = abs(E_t[k] - E_omm) / (abs(E_omm) + 1e-9)
        # combined tolerance handles the near-zero-energy crossings
        assert abs(E_t[k] - E_omm) < 1e-6 * abs(E_omm) + 1e-4, (k, E_t[k], E_omm)
        relF = (np.linalg.norm(F_t[k] - F_omm)
                / (np.linalg.norm(F_omm) + 1e-30))
        assert relF < 1e-5, (k, relF)
        max_relE, max_relF = max(max_relE, relE), max(max_relF, relF)
    # far inside tolerance (double vs double); record the worst case
    assert max_relE < 1e-6 and max_relF < 1e-5, (max_relE, max_relF)


def test_grad_matches_finite_differences():
    ala = build_alanine_system()
    pot = AlanineDipeptideCartesian(device=DEV)
    p0 = ala.positions_nm
    rng = np.random.default_rng(7)
    h = 1e-6
    d = pot.d
    for scale in (0.0, 0.01, 0.03):
        p = p0 + scale * rng.standard_normal(p0.shape)
        x = torch.tensor(p.reshape(-1), device=DEV)
        g = pot.grad(x.unsqueeze(0))[0]
        # central FD: batch the 2d perturbed points
        eye = torch.eye(d, device=DEV, dtype=torch.float64)
        xp = x.unsqueeze(0) + h * eye
        xm = x.unsqueeze(0) - h * eye
        with pot.no_count():
            fd = (pot._V_raw(xp) - pot._V_raw(xm)) / (2.0 * h)
        rel = (g - fd).norm() / (fd.norm() + 1e-30)
        assert float(rel) < 1e-6, (scale, float(rel))


def test_counters_track_evaluations():
    pot = AlanineDipeptideCartesian(device=DEV)
    pot.reset_counters()
    x = torch.zeros(7, pot.d, device=DEV)
    pot.V(x)
    pot.grad(x)
    assert pot.n_V == 7 and pot.n_grad == 7
    with pot.no_count():
        pot.V(x)
        pot.grad(x)
    assert pot.n_V == 7 and pot.n_grad == 7          # no_count restores
    assert pot.nfe() == 14
