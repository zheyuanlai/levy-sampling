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
            whitening, self.whitening_provenance = self._cached_whitening()
        else:
            whitening = torch.as_tensor(whitening, dtype=torch.float64,
                                        device=self.device)
            self.whitening_provenance = {"method": "supplied"}
        self.D = whitening                      # (60,)
        self.Dinv = 1.0 / self.D

    # -- whitening -----------------------------------------------------------
    # -- whitening cache -----------------------------------------------------
    def _cached_whitening(self):
        """Load the whitening from disk, else compute (basin minimisation +
        Hessian, ~45 s) and cache it. Keyed on beta so a temperature change
        invalidates the cache."""
        import json
        import os

        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "cache", "e5_alanine",
            "whitening.npz")
        # signature guards against a stale cache after a coordinate-convention
        # change (the D vector is meaningless in a different BAT layout)
        sig = np.array([self.d, int(self.phi_slot), int(self.psi_slot),
                        len(self.bat.leader_torsion_slots),
                        len(self.bat.offset_torsion_slots)], dtype=np.int64)
        if os.path.exists(path):
            try:
                with np.load(path, allow_pickle=False) as data:
                    if (abs(float(data["beta"]) - self.beta) < 1e-12
                            and "signature" in data
                            and np.array_equal(data["signature"], sig)):
                        self.q_min = torch.as_tensor(
                            data["q_min"], dtype=torch.float64, device=self.device)
                        return (torch.as_tensor(data["D"], dtype=torch.float64,
                                                device=self.device),
                                json.loads(str(data["provenance"])))
            except Exception:
                pass                      # fall through and recompute
        D, prov = self._compute_whitening()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, D=D.cpu().numpy(), q_min=self.q_min.cpu().numpy(),
                 beta=np.float64(self.beta), signature=sig,
                 provenance=np.array(json.dumps(prov)))
        return D, prov

    # -- basin minimisation --------------------------------------------------
    def _minimise(self, q_start: torch.Tensor, D0: torch.Tensor,
                  n_steps: int = 1500, lr: float = 0.05) -> torch.Tensor:
        """Deterministic descent to the nearest U_eff minimum, in D0-whitened
        coordinates (well conditioned, so plain Adam converges quickly)."""
        x = (q_start * (1.0 / D0)).clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=lr)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999)
        for _ in range(n_steps):
            opt.zero_grad()
            E = self._U_eff_from_q(x * D0)
            E.backward()
            opt.step()
            sched.step()
        return (x.detach() * D0)

    def _compute_whitening(self):
        """Fixed diagonal D from the U_eff Hessian at the reference conformer.

        phi and psi keep D = 1 so they stay affine, unit-scale coordinates of
        q_tilde (the jump atoms are pure phi/psi shifts and must read directly as
        rotations).  EVERY other coordinate -- bonds, angles, the stiff sibling
        torsion offsets and the remaining proper torsions -- is scaled to the
        common thermal curvature, which is what makes a single dt comfortable.
        The cap at 1 prevents amplifying a coordinate that is already softer
        than the phi/psi scale.
        """
        keep = {int(self.phi_slot), int(self.psi_slot)}
        idx = torch.tensor([i for i in range(self.d) if i not in keep],
                           dtype=torch.long, device=self.device)

        def _diag_to_D(Hdiag):
            D = torch.ones(self.d, dtype=torch.float64, device=self.device)
            D[idx] = torch.sqrt(
                1.0 / (self.beta * Hdiag[idx].clamp(min=1e-8))).clamp(max=1.0)
            return D

        # bootstrap D0 at the (unminimised) reference conformer, descend to the
        # basin minimum, then take the Hessian THERE: the supplied conformer is
        # not a stationary point (|grad U_eff| ~ 105, almost all of it in phi),
        # which inflates the raw diagonal curvature ~7x.
        H0 = torch.autograd.functional.hessian(
            lambda q: self._U_eff_from_q(q), self.q_ref)
        D0 = _diag_to_D(torch.diagonal(H0))
        self.q_min = self._minimise(self.q_ref, D0)
        H = torch.autograd.functional.hessian(
            lambda q: self._U_eff_from_q(q), self.q_min)     # (60, 60)
        Hdiag = torch.diagonal(H)
        D = _diag_to_D(Hdiag)
        with torch.no_grad():
            gmin = torch.autograd.functional.jacobian(
                lambda q: self._U_eff_from_q(q), self.q_min)
        prov = {
            "method": "hessian_diag_at_minimised_basin_conformer",
            "grad_norm_at_minimum": float(gmin.norm()),
            "U_eff_at_minimum": float(self._U_eff_from_q(self.q_min)),
            "max_curvature_at_minimum": float(Hdiag.max()),
            "curvature_phi": float(Hdiag[self.phi_slot]),
            "curvature_psi": float(Hdiag[self.psi_slot]),
            "temperature_K": E5_TEMPERATURE, "beta": self.beta,
            "unwhitened_slots": {"phi": int(self.phi_slot),
                                 "psi": int(self.psi_slot)},
            "D_bond_min": float(D[self.bond_slots_t].min()),
            "D_bond_max": float(D[self.bond_slots_t].max()),
            "D_angle_min": float(D[self.angle_slots_t].min()),
            "D_angle_max": float(D[self.angle_slots_t].max()),
            "D_torsion_min": float(D[self.torsion_slots_t].min()),
            "D_torsion_max": float(D[self.torsion_slots_t].max()),
            "D_phi": 1.0, "D_psi": 1.0,
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
