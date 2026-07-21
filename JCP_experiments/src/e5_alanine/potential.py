"""P3: ``AlanineDipeptideBAT(Potential)`` = internal-coordinate U_eff + whitening.

The sampler runs isotropic overdamped Langevin in a whitened internal coordinate
q_tilde = D^{-1} q (fixed diagonal D), which equals preconditioned Langevin in q.
The potential the sampler sees is

    V(q_tilde) = U_eff(D q_tilde),
    U_eff(q)   = U(x(q)) - (1/beta) J(q),

with U the Cartesian force field (``cartesian.py``), x(q) the BAT reconstruction
(``bat.py``), and J the analytic log|Jacobian| (S1.5).  Since q_tilde = D^{-1} q is
a constant linear change of variables, the invariant of the whitened Langevin is
mu_q_tilde ∝ e^{-beta U_eff(D q_tilde)}, i.e. mapping q_tilde -> q -> x yields the
Cartesian Boltzmann measure -- the same measure the OpenMM reference samples.

Whitening D (S1.7): D_ii = 1 for torsions (phi, psi stay affine coords of q_tilde);
for bonds/angles D_ii = the thermal std sqrt(1/(beta H_ii)) from the diagonal of
the U_eff Hessian at the reference conformer, which equalizes the stiff bond/angle
timescales with the soft torsions so a single dt is comfortable.

For a pure-torsion shift the Jacobian is constant along the chord (S1.6): every
bond/angle is unchanged, so J cancels and the score integrand reduces to the
physical energy difference.  P3 verifies this residual is machine-zero.
"""
from __future__ import annotations

import numpy as np
import scipy.constants
import torch

from ..potentials import Potential
from .bat import BATTransform
from .cartesian import AlanineDipeptideCartesian
from .system import load_params

E5_TEMPERATURE = 300.0                       # kelvin


def e5_beta(temperature: float = E5_TEMPERATURE) -> float:
    """Inverse temperature beta = 1/(kB T) in mol/kJ, derived from constants."""
    kB_kJ_per_mol_K = scipy.constants.R / 1000.0    # 0.00831446... kJ/(mol K)
    return 1.0 / (kB_kJ_per_mol_K * temperature)


class AlanineDipeptideBAT(Potential):
    """Whitened internal-coordinate potential; state q_tilde is (..., 60)."""

    name = "alanine_bat"

    def __init__(self, params: dict | None = None, beta: float | None = None,
                 device: str | torch.device = "cuda",
                 whitening: torch.Tensor | None = None) -> None:
        super().__init__()
        if params is None:
            params = load_params()
        self.device = torch.device(device)
        self.beta = float(beta) if beta is not None else e5_beta()
        self.cart = AlanineDipeptideCartesian(params, device)
        self.bat = BATTransform(params, device)
        self.d = self.bat.n_internal            # 60
        self.phi_slot = self.bat.phi_slot
        self.psi_slot = self.bat.psi_slot
        self.torsion_slots_t = self.bat.torsion_slots_t
        self.bond_slots_t = self.bat.bond_slots_t
        self.angle_slots_t = self.bat.angle_slots_t

        ref = torch.as_tensor(params["ref_positions_nm"], dtype=torch.float64,
                              device=self.device).reshape(-1)
        self.q_ref = self.bat.to_bat(ref)       # (60,)

        if whitening is None:
            whitening, self.whitening_provenance = self._compute_whitening()
        else:
            whitening = torch.as_tensor(whitening, dtype=torch.float64,
                                        device=self.device)
            self.whitening_provenance = {"method": "supplied"}
        self.D = whitening                      # (60,)
        self.Dinv = 1.0 / self.D

    # -- whitening -----------------------------------------------------------
    def _compute_whitening(self):
        """Fixed diagonal D from the U_eff Hessian at the reference conformer."""
        H = torch.autograd.functional.hessian(
            lambda q: self._U_eff_from_q(q), self.q_ref)     # (60, 60)
        Hdiag = torch.diagonal(H).clamp(min=1e-8)
        D = torch.ones(self.d, dtype=torch.float64, device=self.device)
        ba = torch.cat([self.bond_slots_t, self.angle_slots_t])
        std = torch.sqrt(1.0 / (self.beta * Hdiag[ba]))
        # torsions stay at 1; bonds/angles cannot be scaled above the torsion
        D[ba] = std.clamp(max=1.0)
        prov = {
            "method": "hessian_diag_at_reference_conformer",
            "temperature_K": E5_TEMPERATURE, "beta": self.beta,
            "D_bond_min": float(D[self.bond_slots_t].min()),
            "D_bond_max": float(D[self.bond_slots_t].max()),
            "D_angle_min": float(D[self.angle_slots_t].min()),
            "D_angle_max": float(D[self.angle_slots_t].max()),
            "D_torsion": 1.0,
        }
        return D, prov

    # -- energy --------------------------------------------------------------
    def _U_eff_from_q(self, q: torch.Tensor) -> torch.Tensor:
        """U_eff(q) = U(x(q)) - (1/beta) J(q) in UN-whitened internal coords."""
        x = self.bat.to_cartesian(q)
        return self.cart._V_raw(x) - (1.0 / self.beta) * self.bat.log_jacobian(q)

    def _U_cart_from_qt(self, qt: torch.Tensor) -> torch.Tensor:
        """Physical Cartesian energy U(x(D q_tilde)) only (no Jacobian term)."""
        return self.cart._V_raw(self.bat.to_cartesian(qt * self.D))

    def _V_raw(self, qt: torch.Tensor) -> torch.Tensor:
        return self._U_eff_from_q(qt * self.D)

    def V(self, qt: torch.Tensor) -> torch.Tensor:
        self.n_V += int(np.prod(qt.shape[:-1]))
        return self._V_raw(qt)

    def grad(self, qt: torch.Tensor) -> torch.Tensor:
        self.n_grad += int(np.prod(qt.shape[:-1]))
        with torch.enable_grad():
            q = qt.detach().requires_grad_(True)
            E = self._V_raw(q).sum()
            (g,) = torch.autograd.grad(E, q)
        return g

    # -- collective variables ------------------------------------------------
    def to_cv(self, qt: torch.Tensor) -> torch.Tensor:
        """(phi, psi) in (-pi, pi] from the whitened state."""
        return self.bat.cv(qt * self.D)
