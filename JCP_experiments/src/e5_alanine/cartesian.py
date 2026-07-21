"""P1: batched torch Cartesian force field for alanine dipeptide (vacuum).

``AlanineDipeptideCartesian`` reproduces the OpenMM potential energy of the
flexible ``AlanineDipeptideVacuum`` system (see ``system.py`` / ``params.npz``)
in pure float64 torch, batched over arbitrary leading dimensions, on GPU. It is
the inner Cartesian energy used by the BAT potential (``potential.py``) and is
validated against OpenMM to rel 1e-6 (energy) / 1e-5 (force) in P1.

Energy = harmonic bonds + harmonic angles + periodic torsions + nonbonded, with
the OpenMM ``NoCutoff`` vacuum convention:

  * bond:    0.5 k (r - r0)^2
  * angle:   0.5 k (theta - theta0)^2
  * torsion: k (1 + cos(n phi - phase))
  * nonbonded (i<j, NOT an exception):
        ke q_i q_j / r + 4 eps_ij [(sig_ij/r)^12 - (sig_ij/r)^6],
        sig_ij = (sig_i + sig_j)/2,  eps_ij = sqrt(eps_i eps_j)   (Lorentz-Berthelot)
  * nonbonded exceptions (excluded 1-2/1-3 and scaled 1-4):
        ke chargeProd / r + 4 eps_exc [(sig_exc/r)^12 - (sig_exc/r)^6]

The exception pairs are removed from the standard sum and replaced by their
stored (chargeProd, sig_exc, eps_exc); excluded pairs carry chargeProd = eps = 0.
"""
from __future__ import annotations

import numpy as np
import torch

from ..potentials import Potential
from .system import load_params


def _dihedral_torch(pos: torch.Tensor, quartet: torch.Tensor) -> torch.Tensor:
    """Signed dihedral(s) for atom quartets, atan2 form (S1.8).

    pos: (..., N, 3); quartet: (T, 4) long -> (..., T) radians in (-pi, pi].
    """
    p1 = pos[..., quartet[:, 0], :]
    p2 = pos[..., quartet[:, 1], :]
    p3 = pos[..., quartet[:, 2], :]
    p4 = pos[..., quartet[:, 3], :]
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    b2n = b2 / b2.norm(dim=-1, keepdim=True)
    x = (n1 * n2).sum(-1)
    y = (torch.cross(n1, n2, dim=-1) * b2n).sum(-1)
    return torch.atan2(y, x)


class AlanineDipeptideCartesian(Potential):
    """Batched float64 Cartesian force field; state x is (..., 66) = (..., 22, 3)."""

    d = 66
    name = "alanine_cartesian"

    def __init__(self, params: dict | None = None,
                 device: str | torch.device = "cuda") -> None:
        super().__init__()
        if params is None:
            params = load_params()
        self.device = torch.device(device)
        dt = torch.float64
        self.n_atoms = int(params["n_atoms"])
        self.ke = float(params["one_4pi_eps0"])

        def _t(a, dtype=dt):
            return torch.as_tensor(np.asarray(a), dtype=dtype, device=self.device)

        # -- bonds -----------------------------------------------------------
        self.bond_idx = _t(params["bond_idx"], torch.long)          # (Nb, 2)
        self.bond_r0 = _t(params["bond_r0"])                        # (Nb,)
        self.bond_k = _t(params["bond_k"])
        # -- angles ----------------------------------------------------------
        self.angle_idx = _t(params["angle_idx"], torch.long)        # (Na, 3)
        self.angle_t0 = _t(params["angle_theta0"])
        self.angle_k = _t(params["angle_k"])
        # -- torsions --------------------------------------------------------
        self.tor_idx = _t(params["torsion_idx"], torch.long)        # (Nt, 4)
        self.tor_n = _t(params["torsion_periodicity"])              # float (Nt,)
        self.tor_phase = _t(params["torsion_phase"])
        self.tor_k = _t(params["torsion_k"])

        # -- nonbonded: partition all i<j into standard + exception pairs ----
        q = np.asarray(params["nb_charge"], dtype=np.float64)
        sig = np.asarray(params["nb_sigma"], dtype=np.float64)
        eps = np.asarray(params["nb_eps"], dtype=np.float64)
        exc_idx = np.asarray(params["exc_idx"], dtype=np.int64)
        exc_set = {frozenset((int(i), int(j))) for i, j in exc_idx}

        std_pairs, std_qq, std_sig, std_eps = [], [], [], []
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                if frozenset((i, j)) in exc_set:
                    continue
                std_pairs.append((i, j))
                std_qq.append(q[i] * q[j])
                std_sig.append(0.5 * (sig[i] + sig[j]))            # Lorentz
                std_eps.append(np.sqrt(eps[i] * eps[j]))           # Berthelot
        self.std_idx = _t(np.array(std_pairs, dtype=np.int64), torch.long)  # (Ns, 2)
        self.std_qq = _t(np.array(std_qq))
        self.std_sig = _t(np.array(std_sig))
        self.std_eps = _t(np.array(std_eps))

        self.exc_idx = _t(exc_idx, torch.long)                     # (Ne, 2)
        self.exc_qq = _t(params["exc_chargeProd"])
        self.exc_sig = _t(params["exc_sigma"])
        self.exc_eps = _t(params["exc_eps"])

    # -- energy components ---------------------------------------------------
    def _bond_energy(self, pos: torch.Tensor) -> torch.Tensor:
        d = pos[..., self.bond_idx[:, 0], :] - pos[..., self.bond_idx[:, 1], :]
        r = d.norm(dim=-1)
        return (0.5 * self.bond_k * (r - self.bond_r0) ** 2).sum(-1)

    def _angle_energy(self, pos: torch.Tensor) -> torch.Tensor:
        v1 = pos[..., self.angle_idx[:, 0], :] - pos[..., self.angle_idx[:, 1], :]
        v2 = pos[..., self.angle_idx[:, 2], :] - pos[..., self.angle_idx[:, 1], :]
        cos = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1))
        cos = cos.clamp(-1.0 + 1e-12, 1.0 - 1e-12)
        theta = torch.acos(cos)
        return (0.5 * self.angle_k * (theta - self.angle_t0) ** 2).sum(-1)

    def _torsion_energy(self, pos: torch.Tensor) -> torch.Tensor:
        phi = _dihedral_torch(pos, self.tor_idx)                   # (..., Nt)
        return (self.tor_k * (1.0 + torch.cos(self.tor_n * phi - self.tor_phase))).sum(-1)

    def _pair_energy(self, pos: torch.Tensor, idx: torch.Tensor,
                     qq: torch.Tensor, sig: torch.Tensor,
                     eps: torch.Tensor) -> torch.Tensor:
        d = pos[..., idx[:, 0], :] - pos[..., idx[:, 1], :]
        r = d.norm(dim=-1)
        inv_r = 1.0 / r
        coul = self.ke * qq * inv_r
        sr6 = (sig * inv_r) ** 6
        lj = 4.0 * eps * (sr6 * sr6 - sr6)
        return (coul + lj).sum(-1)

    def _nonbonded_energy(self, pos: torch.Tensor) -> torch.Tensor:
        std = self._pair_energy(pos, self.std_idx, self.std_qq,
                                self.std_sig, self.std_eps)
        exc = self._pair_energy(pos, self.exc_idx, self.exc_qq,
                                self.exc_sig, self.exc_eps)
        return std + exc

    # -- Potential interface -------------------------------------------------
    def _V_raw(self, x: torch.Tensor) -> torch.Tensor:
        pos = x.reshape(*x.shape[:-1], self.n_atoms, 3)
        return (self._bond_energy(pos) + self._angle_energy(pos)
                + self._torsion_energy(pos) + self._nonbonded_energy(pos))

    def V(self, x: torch.Tensor) -> torch.Tensor:
        self.n_V += int(np.prod(x.shape[:-1]))
        return self._V_raw(x)

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        self.n_grad += int(np.prod(x.shape[:-1]))
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            E = self._V_raw(xx).sum()
            (g,) = torch.autograd.grad(E, xx)
        return g
