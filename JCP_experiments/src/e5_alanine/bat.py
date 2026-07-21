"""P2: differentiable, batched BAT (bond-angle-torsion) internal coordinates.

We run the LSC-CP sampler in internal coordinates so that the backbone torsions
phi, psi are literally two coordinates.  The transform is a Z-matrix / NeRF
(Natural Extension Reference Frame) reconstruction anchored on three backbone
atoms, which fixes the 6 external DOF; the 22-atom molecule then has
D - 6 = 60 internal DOF.

Layout of the internal vector q (length 60):
    q[0] = r12  (root bond A1-A2)
    q[1] = r23  (root bond A2-A3)
    q[2] = a123 (root angle A1-A2-A3)
    then, for each of the 19 non-root atoms in build order:
        (b_D, a_D, tau_D)  =  (bond |D-C|, angle(B,C,D), dihedral(A,B,C,D)).

Root = (ACE:C=4, ALA:N=6, ALA:CA=8).  The first two non-root atoms are forced to
be ALA:C=14 and NME:N=16 so that
    tau_14 = dihedral(4,6,8,14)  = phi   (q index 5)
    tau_16 = dihedral(6,8,14,16) = psi   (q index 8).

Analytic log|Jacobian| of the reduced free-Cartesian -> internal map (S1.5):
    log|det| = ln(r23) + sum_{non-root D} [ 2 ln b_D + ln sin a_D ].
The root-atom-3 term ln(r23) is the frame boundary term (a 2-D polar element,
one power of the bond, no sine).  Every term depends only on bonds and angles --
never on a torsion -- which is the torsion-independence checked in P2 and the
premise of the Jacobian-free chord in P3 (S1.6).
"""
from __future__ import annotations

import numpy as np
import torch

from .system import load_params

# root atoms (ACE:C, ALA:N, ALA:CA); phi/psi pinned as the first non-root torsions
ROOT = (4, 6, 8)
FORCED_FIRST = (14, 16)


def _adjacency(bond_idx: np.ndarray, n_atoms: int) -> list[list[int]]:
    adj = [[] for _ in range(n_atoms)]
    for i, j in bond_idx:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    return adj


def build_zmatrix(bond_idx: np.ndarray, n_atoms: int,
                  root=ROOT, forced_first=FORCED_FIRST):
    """Deterministic build order + (A,B,C) dihedral/angle/bond references.

    Returns (order, refA, refB, refC): ``order`` lists atoms in placement order
    (root first); for each non-root atom, C is the bond parent, B the angle
    parent, A the dihedral parent, all placed earlier.  References prefer the
    earliest-placed neighbours (the backbone), giving well-defined dihedrals.
    """
    adj = _adjacency(bond_idx, n_atoms)
    order = list(root)
    placed = {a: k for k, a in enumerate(order)}   # atom -> placement rank

    def _pick_refs(d: int):
        # C = earliest-placed bonded neighbour of D
        cand_c = sorted((a for a in adj[d] if a in placed), key=placed.get)
        if not cand_c:
            return None
        C = cand_c[0]
        # B = earliest-placed neighbour of C, != D
        cand_b = sorted((a for a in adj[C] if a in placed and a != d),
                        key=placed.get)
        if not cand_b:
            return None
        B = cand_b[0]
        # A = earliest-placed neighbour of B (!= C, D); else of C (!= B, D)
        cand_a = [a for a in sorted((x for x in adj[B] if x in placed),
                                    key=placed.get) if a not in (C, d)]
        if not cand_a:
            cand_a = [a for a in sorted((x for x in adj[C] if x in placed),
                                        key=placed.get) if a not in (B, d)]
        if not cand_a:
            return None
        return C, B, cand_a[0]

    refA = {}
    refB = {}
    refC = {}

    def _place(d: int):
        refs = _pick_refs(d)
        if refs is None:
            raise RuntimeError(f"no valid Z-matrix refs for atom {d}")
        C, B, A = refs
        refC[d], refB[d], refA[d] = C, B, A
        placed[d] = len(order)
        order.append(d)

    for d in forced_first:
        _place(d)
    # greedy: repeatedly place any atom that now has valid references
    remaining = [a for a in range(n_atoms) if a not in placed]
    while remaining:
        progressed = False
        for d in list(remaining):
            if _pick_refs(d) is not None:
                _place(d)
                remaining.remove(d)
                progressed = True
        if not progressed:
            raise RuntimeError(f"cannot place atoms {remaining}")

    nonroot = order[3:]
    refA_a = np.array([refA[d] for d in nonroot], dtype=np.int64)
    refB_a = np.array([refB[d] for d in nonroot], dtype=np.int64)
    refC_a = np.array([refC[d] for d in nonroot], dtype=np.int64)

    # -- correlated (leader/offset) torsions ------------------------------
    # Atoms sharing a bond/angle parent pair (C, B) rotate about the SAME axis.
    # If each kept an absolute torsion, moving one while holding its siblings
    # fixed would distort the local (e.g. tetrahedral) geometry instead of
    # rotating the fragment: measured curvature of such a coordinate is ~800-1300
    # kJ/mol/rad^2, versus ~50 for the true soft rotation. So the first atom
    # placed about each axis is the LEADER and carries the proper torsion; its
    # siblings carry the (near-constant, stiff) OFFSET from the leader. Moving a
    # leader torsion then rotates the whole fragment, which is the physical
    # phi/psi motion. The reparametrisation is unit-triangular in the torsion
    # block, so the Jacobian of S1.5 is unchanged.
    leader_of = np.full(len(nonroot), -1, dtype=np.int64)
    first_of_axis: dict = {}
    for k, d in enumerate(nonroot):
        key = (int(refC[d]), int(refB[d]))
        if key in first_of_axis:
            leader_of[k] = first_of_axis[key]
        else:
            first_of_axis[key] = k
    return (np.array(order, dtype=np.int64), refA_a, refB_a, refC_a, leader_of)


class BATTransform:
    """Batched float64 BAT transform + analytic log|Jacobian|.

    ``to_bat(x)`` and ``to_cartesian(q)`` are differentiable and batched over any
    leading dimensions.  ``to_cartesian`` emits Cartesian coordinates in the
    canonical frame (A1 at origin, A2 on +x, A3 in the +xy half-plane); since the
    force field is roto-translation invariant the canonical embedding carries the
    correct energy.
    """

    def __init__(self, params: dict | None = None,
                 device: str | torch.device = "cuda") -> None:
        if params is None:
            params = load_params()
        self.device = torch.device(device)
        self.n_atoms = int(params["n_atoms"])
        order, refA, refB, refC, leader_of = build_zmatrix(
            np.asarray(params["bond_idx"]), self.n_atoms)
        self.order = order                       # (22,) placement order
        self.nonroot = order[3:]                 # (19,) atoms with (b,a,tau)
        self.root = tuple(int(a) for a in order[:3])
        self.n_internal = 3 * self.n_atoms - 6   # 60

        lt = lambda a: torch.as_tensor(a, dtype=torch.long, device=self.device)
        self.refA, self.refB, self.refC = lt(refA), lt(refB), lt(refC)
        self.order_t = lt(order)
        self.nonroot_t = lt(self.nonroot)
        # Python-int reference lists for the sequential NeRF loop. Indexing the
        # position list with a CUDA tensor element (int(self.refA[k])) would force
        # a device synchronisation on every atom placement -- 66 syncs per
        # to_cartesian call, which dominated the step cost.
        self.refA_i = [int(v) for v in refA]
        self.refB_i = [int(v) for v in refB]
        self.refC_i = [int(v) for v in refC]
        self.nonroot_i = [int(v) for v in self.nonroot]
        # torsion leader/offset bookkeeping (see build_zmatrix)
        self.leader_of = leader_of                      # (19,) -1 if leader
        self.leader_of_i = [int(v) for v in leader_of]
        self.is_sibling_t = torch.as_tensor(leader_of >= 0, device=self.device)
        self.leader_idx_t = lt(np.where(leader_of >= 0, leader_of,
                                        np.arange(len(leader_of))))

        # q layout indices
        self.i_r12, self.i_r23, self.i_a123 = 0, 1, 2
        self.bond_slots = [0, 1] + [3 + 3 * k for k in range(len(self.nonroot))]
        self.angle_slots = [2] + [4 + 3 * k for k in range(len(self.nonroot))]
        self.torsion_slots = [5 + 3 * k for k in range(len(self.nonroot))]
        # phi = tau of the 1st non-root atom (14); psi = tau of the 2nd (16)
        self.phi_slot = self.torsion_slots[0]
        self.psi_slot = self.torsion_slots[1]
        # proper (leader) vs offset (sibling) torsion slots
        self.leader_torsion_slots = [5 + 3 * k for k in range(len(self.nonroot))
                                     if leader_of[k] < 0]
        self.offset_torsion_slots = [5 + 3 * k for k in range(len(self.nonroot))
                                     if leader_of[k] >= 0]
        if (self.phi_slot not in self.leader_torsion_slots
                or self.psi_slot not in self.leader_torsion_slots):
            raise RuntimeError("phi/psi must be proper (leader) torsions")
        self.bond_slots_t = lt(np.array(self.bond_slots))
        self.angle_slots_t = lt(np.array(self.angle_slots))
        self.torsion_slots_t = lt(np.array(self.torsion_slots))

        # free Cartesian coordinate indices (66 minus the 6 frame-fixed ones)
        a1, a2, a3 = self.root
        fixed = {3 * a1, 3 * a1 + 1, 3 * a1 + 2, 3 * a2 + 1, 3 * a2 + 2, 3 * a3 + 2}
        self.free_cart_idx = torch.tensor(
            [c for c in range(3 * self.n_atoms) if c not in fixed],
            dtype=torch.long, device=self.device)

    # -- Cartesian -> internal ----------------------------------------------
    def to_bat(self, x: torch.Tensor) -> torch.Tensor:
        pos = x.reshape(*x.shape[:-1], self.n_atoms, 3)
        A1, A2, A3 = self.root
        p1, p2, p3 = pos[..., A1, :], pos[..., A2, :], pos[..., A3, :]
        r12 = (p2 - p1).norm(dim=-1)
        r23 = (p3 - p2).norm(dim=-1)
        a123 = _angle(p1, p2, p3)
        pA = pos[..., self.refA, :]
        pB = pos[..., self.refB, :]
        pC = pos[..., self.refC, :]
        pD = pos[..., self.nonroot_t, :]
        b = (pD - pC).norm(dim=-1)                                  # (..., 19)
        a = _angle(pB, pC, pD)
        tau_abs = _dihedral(pA, pB, pC, pD)                         # (..., 19)
        # siblings store the offset from their axis leader (unit-triangular map)
        tau = torch.where(self.is_sibling_t,
                          _wrap(tau_abs - tau_abs[..., self.leader_idx_t]),
                          tau_abs)
        # interleave (b, a, tau) per non-root atom into q[3:]
        bat = torch.stack([b, a, tau], dim=-1)                      # (..., 19, 3)
        bat = bat.reshape(*bat.shape[:-2], 3 * bat.shape[-2])       # (..., 57)
        head = torch.stack([r12, r23, a123], dim=-1)                # (..., 3)
        return torch.cat([head, bat], dim=-1)                      # (..., 60)

    # -- internal -> Cartesian (NeRF) ---------------------------------------
    def to_cartesian(self, q: torch.Tensor) -> torch.Tensor:
        batch = q.shape[:-1]
        dev, dt = q.device, q.dtype
        A1, A2, A3 = self.root
        r12, r23, a123 = q[..., 0], q[..., 1], q[..., 2]
        zeros = torch.zeros(batch, device=dev, dtype=dt)
        pos = [None] * self.n_atoms
        pos[A1] = torch.stack([zeros, zeros, zeros], dim=-1)
        pos[A2] = torch.stack([r12, zeros, zeros], dim=-1)
        # A3 in the xy-plane: A2 + r23*(-cos a123, sin a123, 0)
        pos[A3] = pos[A2] + torch.stack(
            [-r23 * torch.cos(a123), r23 * torch.sin(a123), zeros], dim=-1)
        tau_abs = [None] * len(self.nonroot_i)
        for k, d in enumerate(self.nonroot_i):
            b = q[..., 3 + 3 * k]
            a = q[..., 4 + 3 * k]
            tau = q[..., 5 + 3 * k]
            lead = self.leader_of_i[k]
            # a sibling's stored coordinate is its offset from the axis leader,
            # which is always placed earlier in the build order
            tau_abs[k] = tau if lead < 0 else tau_abs[lead] + tau
            pos[d] = _nerf(pos[self.refA_i[k]], pos[self.refB_i[k]],
                           pos[self.refC_i[k]], b, a, tau_abs[k])
        return torch.stack(pos, dim=-2).reshape(*batch, 3 * self.n_atoms)

    # -- analytic log|Jacobian| ---------------------------------------------
    def log_jacobian(self, q: torch.Tensor) -> torch.Tensor:
        r23 = q[..., self.i_r23]
        b = q[..., self.bond_slots_t[2:]]        # non-root bonds (19,)
        a = q[..., self.angle_slots_t[1:]]       # non-root angles (19,)
        return (torch.log(r23) + (2.0 * torch.log(b)).sum(-1)
                + torch.log(torch.sin(a)).sum(-1))

    def cv(self, q: torch.Tensor) -> torch.Tensor:
        """(phi, psi) collective variables (wrapped to (-pi, pi])."""
        phi = _wrap(q[..., self.phi_slot])
        psi = _wrap(q[..., self.psi_slot])
        return torch.stack([phi, psi], dim=-1)


# ----------------------------------------------------------------- helpers
def _angle(pi, pj, pk):
    """angle at pj between (pi-pj) and (pk-pj)."""
    v1 = pi - pj
    v2 = pk - pj
    cos = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1))
    return torch.acos(cos.clamp(-1.0 + 1e-12, 1.0 - 1e-12))


def _dihedral(pA, pB, pC, pD):
    b1, b2, b3 = pB - pA, pC - pB, pD - pC
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    b2n = b2 / b2.norm(dim=-1, keepdim=True)
    x = (n1 * n2).sum(-1)
    y = (torch.cross(n1, n2, dim=-1) * b2n).sum(-1)
    return torch.atan2(y, x)


def _nerf(A, B, C, b, theta, chi):
    """Place D given refs A,B,C and (bond b, angle=theta at C, dihedral=chi)."""
    bc = C - B
    bc = bc / bc.norm(dim=-1, keepdim=True)
    ab = B - A
    n = torch.cross(ab, bc, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True)
    m = torch.cross(n, bc, dim=-1)                 # in-plane, perp to bc
    st = torch.sin(theta)
    d = (-bc * torch.cos(theta).unsqueeze(-1)
         + m * (st * torch.cos(chi)).unsqueeze(-1)
         + n * (st * torch.sin(chi)).unsqueeze(-1))
    return C + b.unsqueeze(-1) * d


def _wrap(t):
    """wrap angle(s) to (-pi, pi]."""
    return -(torch.remainder(np.pi - t, 2.0 * np.pi) - np.pi)
