"""``TorusBox``: periodic-in-torsion, generous-in-stiff numerical box for E5.

The sampler state is the whitened internal coordinate q_tilde.  Its torsion slots
are angles on a torus, so they have no boundary: ``clip`` WRAPS them to (-pi, pi]
(exact, because U_eff is 2pi-periodic in every torsion -- the BAT reconstruction is
periodic and the Jacobian is torsion-independent), and ``contains`` reports them
as always inside so that wrapping is never miscounted as a boundary clip.

The bond and angle slots do have a boundary, and it is a *physical* one rather
than a mere overflow guard: the Jacobian carries ln b and ln sin a, so the box
must keep b > 0 and a in (0, pi).  The limits are set in physical units (a bond
may range over [lo_frac, hi_frac] x its reference length; an angle over
[a_min, pi - a_min]) and converted to whitened coordinates.  They sit many
thermal standard deviations away from equilibrium, so in normal sampling the box
never binds -- exactly like the generous boxes of E1/E3/E4.
"""
from __future__ import annotations

import numpy as np
import torch


class TorusBox:
    """Box in whitened internal coordinates: torsions wrap, stiff DOF clamp."""

    def __init__(self, potential, *, bond_lo_frac: float = 0.3,
                 bond_hi_frac: float = 3.0, angle_margin: float = 0.10) -> None:
        self.pot = potential
        dev = potential.D.device
        d = potential.d
        self.torsion_slots = potential.torsion_slots_t
        self.bond_slots = potential.bond_slots_t
        self.angle_slots = potential.angle_slots_t

        q_ref = potential.q_ref                      # unwhitened reference internal
        lo_q = torch.full((d,), -float("inf"), dtype=torch.float64, device=dev)
        hi_q = torch.full((d,), float("inf"), dtype=torch.float64, device=dev)
        lo_q[self.bond_slots] = bond_lo_frac * q_ref[self.bond_slots]
        hi_q[self.bond_slots] = bond_hi_frac * q_ref[self.bond_slots]
        lo_q[self.angle_slots] = angle_margin
        hi_q[self.angle_slots] = np.pi - angle_margin
        # whitened limits (D > 0 elementwise, so the map is order preserving)
        self.lo = lo_q * potential.Dinv
        self.hi = hi_q * potential.Dinv
        # A torsion is 2pi-periodic in PHYSICAL units, so in whitened units its
        # period is 2pi / D. Only phi and psi have D = 1; wrapping the other
        # torsion slots as if they had period 2pi would destroy the state.
        self.tor_D = potential.D[self.torsion_slots]
        self.tor_Dinv = potential.Dinv[self.torsion_slots]
        # mask of coordinates that actually carry a boundary (bonds + angles)
        self.bounded = torch.zeros(d, dtype=torch.bool, device=dev)
        self.bounded[self.bond_slots] = True
        self.bounded[self.angle_slots] = True
        self.limits_physical = {
            "bond_lo_frac": bond_lo_frac, "bond_hi_frac": bond_hi_frac,
            "angle_lo_rad": angle_margin, "angle_hi_rad": float(np.pi - angle_margin),
        }

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        """Torsions are always inside (torus); stiff DOF must lie in [lo, hi]."""
        ok = (x >= self.lo) & (x <= self.hi)
        ok = ok | (~self.bounded)                    # unbounded slots always inside
        return ok.all(dim=-1)

    def clip(self, x: torch.Tensor) -> torch.Tensor:
        """Wrap torsions to (-pi, pi] in PHYSICAL units; clamp bonds/angles."""
        y = torch.clamp(x, self.lo, self.hi)         # inf limits leave slots untouched
        tor = y[..., self.torsion_slots] * self.tor_D          # -> radians
        y = y.index_copy(-1, self.torsion_slots, _wrap(tor) * self.tor_Dinv)
        return y


def _wrap(t: torch.Tensor) -> torch.Tensor:
    """Wrap angle(s) to (-pi, pi]."""
    return -(torch.remainder(np.pi - t, 2.0 * np.pi) - np.pi)
