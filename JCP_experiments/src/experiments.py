"""Experiment builders E1-E4: single source of truth shared by the tests and
the notebooks. Each builder wires potential + jump law + box + init +
reference + partition + metric space + certificate inputs.

Global protocol: beta = 8, eps = 0.125, lam = 1, 5 seeds, shared x0 per
seed, metric cadence fixed in t and identical across methods.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .config import (BETA, EPS, LAMBDA, M_PHI, Q_RHO, Q_THETA, RunConfig,
                     diffusion_seed, init_seed, jump_seed)
from .jumps import AnnulusJumpLaw, ShellJumpLaw, gauss_legendre_01
from .potentials import (CoupledPhi4, DoubleWell1D, MB3_CRITICAL, MB4_CRITICAL,
                         MB_CRITICAL, MB3Latent2D, MB4Latent2D, MoG40,
                         MuellerBrownLatent2D, PHI4_MINIMA,
                         TransformedMB3Well10D, TransformedMB4Well10D,
                         TransformedMuellerBrown10D,
                         mb3_2d, mb3_2d_grad, mb4_2d, mb4_2d_grad,
                         muller_brown_2d, muller_brown_2d_grad, newton_refine,
                         phi4_W, phi4_W_grad)
from .references import (GradientFlowBasinMap2D, Grid1DInverseCDF,
                         LaplaceMixture, Latent2DGaussianReference,
                         MB10DReference)
from .samplers import (BAOAB, FLA, MALA, ULA, CompoundPoisson, LatentRectBox,
                       ParallelTempering, RectBox)
from .score import MoG40Score, RandomAtomicShellScore, ShellScore
from . import metrics as M


@dataclass
class Experiment:
    name: str
    cfg: RunConfig
    pot: object
    law: object
    box: object
    init_fn: Callable                     # (n, gen) -> x0
    ref_sample: Callable                  # (n, gen) -> reference draws
    make_score: Callable                  # (q_theta, q_rho) -> score object
    labels_fn: Callable                   # x -> partition labels
    p_star: torch.Tensor
    metric_space: Callable                # x -> coords used for W2/MMD
    pt_beta_min: float
    # barrier verification: COMMITTED exit event (arrival in another basin's
    # core), not first touch of a partition boundary -- boundary-touch counts
    # half-committed excursions and overstates the escape rate several-fold
    exit_committed: Callable              # x -> bool
    kramers_tau: float
    cp_drift_cap: float = 1.0             # drift-step cap for the CP pair
    extras: dict = field(default_factory=dict)

    @property
    def uniform_target(self) -> bool:
        K = self.p_star.shape[0]
        return bool(torch.allclose(self.p_star,
                                   torch.full_like(self.p_star, 1.0 / K),
                                   atol=1e-6))

    @property
    def emc_target(self) -> float:
        # EMC is defined so that 1 is optimal for every experiment:
        # uniform p*: EMC = exp(H(p_hat))/K; non-uniform p*: EMC = 1 - EJS.
        return 1.0


# ===================================================================== E1
def build_e1(device="cuda") -> Experiment:
    pot = DoubleWell1D()
    cfg = RunConfig(name="double_well", d=1, n_particles=4000, T=100.0, dt=0.005)
    atoms = torch.tensor([[2.0], [-2.0]], dtype=torch.float64, device=device)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64, device=device)
    law = ShellJumpLaw(atoms, weights, h=0.2)     # +-2 maps minimum to minimum
    # generous box = the certificate domain: pi has ~no mass beyond +-2, so
    # LSC-CP never hits the boundary, but raw CP injects tail/barrier mass out to
    # ~+-3.5 -- a tight [-3,3] clip would pile that mass at the edge and confound
    # the raw-CP CDF. Widening removes the clip artifact (LSC-CP unaffected).
    box = RectBox([-5.2], [5.2], device)
    ref = Grid1DInverseCDF(lambda x: -BETA * (x * x - 1.0) ** 2, -5.2, 5.2,
                           device=device)

    def init_fn(n, gen):
        return -1.0 + 0.05 * torch.randn(n, 1, generator=gen, device=device,
                                         dtype=torch.float64)

    def make_score(q_theta=Q_THETA, q_rho=Q_RHO):
        return ShellScore(pot, law, LAMBDA, BETA, q_theta, q_rho)

    p_star = torch.tensor([0.5, 0.5], dtype=torch.float64, device=device)
    return Experiment(
        name="double_well", cfg=cfg, pot=pot, law=law, box=box,
        init_fn=init_fn, ref_sample=lambda n, g: ref.sample(n, g),
        make_score=make_score,
        labels_fn=lambda x: (x[:, 0] > 0).long(),
        p_star=p_star, metric_space=lambda x: x,
        pt_beta_min=1.0,
        exit_committed=lambda x: x[:, 0] > 0.7,     # right-well core arrival
        kramers_tau=DoubleWell1D.kramers_time(BETA),
        # density-TV bins span the widened box; bump count to keep ~0.03 width
        extras={"ref": ref, "density_tv_box": (-5.2, 5.2), "density_tv_bins": 350},
    )


# ===================================================================== E2
def build_e2(device="cuda") -> Experiment:
    pot = MoG40(beta=BETA, device=device)
    cfg = RunConfig(name="mog40", d=2, n_particles=2500, T=100.0, dt=0.01)
    # deliberately generic law: [4, 15] set from the NN-distance histogram
    # alone; neither PT nor LSC-CP receives mode locations.
    law = AnnulusJumpLaw(4.0, 15.0, device)
    box = RectBox([-65.0, -65.0], [65.0, 65.0], device)

    def init_fn(n, gen):
        return pot.mu[0] + 0.5 * torch.randn(n, 2, generator=gen, device=device,
                                             dtype=torch.float64)

    def make_score(q_theta=None, q_rho=None, m_phi=M_PHI):
        # closed form: theta and rho integrals are analytic; only m_phi matters
        return MoG40Score(pot.mu, 4.0, 15.0, LAMBDA, m_phi=m_phi)

    def labels_fn(x):
        d2 = ((x.unsqueeze(1) - pot.mu.unsqueeze(0)) ** 2).sum(-1)
        return d2.argmin(dim=1)

    p_star = torch.full((40,), 1.0 / 40.0, dtype=torch.float64, device=device)

    # nearest-neighbour gap of mode 0, for the Kramers estimate
    dists = torch.cdist(pot.mu, pot.mu)
    dists.fill_diagonal_(float("inf"))
    d0 = float(dists[0].min().item())
    beta_dV = d0 ** 2 / 8.0 - math.log(2.0)
    # 1D Kramers along the inter-mode line: omega_min = omega_saddle ~ 1
    # (unit-variance components), tau ~ 2 pi e^{beta dV}
    kramers = 2.0 * math.pi * math.exp(beta_dV)
    return Experiment(
        name="mog40", cfg=cfg, pot=pot, law=law, box=box,
        init_fn=init_fn,
        ref_sample=lambda n, g: pot.sample_exact(n, g),
        make_score=make_score, labels_fn=labels_fn, p_star=p_star,
        metric_space=lambda x: x,
        # PT's bottleneck here is hot-chain DIFFUSION across the mode cloud,
        # not swap acceptance: in d=2 the V-overlap stays large, so even K=2
        # lands in the acceptance band while the hot replica needs
        # t ~ L^2 beta_min / 4 >> T to traverse the box. Diffusive-traversal
        # criterion: beta_min = 4T/L^2 with L = 130, T = 100 -> 0.025; the
        # acceptance band then forces the dense ladder PT actually needs.
        pt_beta_min=0.025,
        exit_committed=_mog40_committed_exit(pot),
        kramers_tau=kramers,
        extras={"nn_dist_mode0": d0, "beta_dV_mode0": beta_dV},
    )


# ===================================================================== E3
def build_e3(device="cuda", basin_cache: str | None = None,
             beta: float = 24.0) -> Experiment:
    """E3: depth-retuned 3-well Mueller-Brown, embedded in 10D (x = z B^T).

    Standard MB geometry with depths retuned to equal (V = -0.7957), so the
    temperature is a free dial (the standard MB is multimodal OR metastable, but
    never both). At beta = 24 the target is trimodal AND metastable with TWO
    timescales: A<->B slow (beta*b = 11.1, local methods cannot cross in T=200),
    B<->C moderate (beta*b = 4.0). Basin masses ~ (0.32, 0.42, 0.26). Init in
    well C; only nonlocal relay jumps through the middle hub B populate A.

    Jump law = 4 relay atoms {+-r_BA, +-r_BC} (through the middle hub B; NO
    direct A-C atom, whose chord overshoots the field-zero region), uniform
    weights, shell h = 0.1 min||r_a||, CP-pair drift step capped at 2h (the B-A
    chord overshoots S_AB, so integrate the return flux with small in-tube
    steps). beta is threaded locally (config.BETA stays 8 for E1/E2/E4).
    """
    E3_BETA = float(beta)          # <-- E3 temperature (switch to 32.0 here)
    pot = TransformedMB3Well10D(device=device)
    cfg = RunConfig(name="mb3well_10d", d=10, n_particles=2000, T=200.0,
                    dt=0.005, beta=E3_BETA)

    mins = {}
    for key in ("A", "B", "C"):
        z0 = torch.tensor(MB3_CRITICAL[key][0], dtype=torch.float64, device=device)
        mins[key] = newton_refine(mb3_2d_grad, z0)
    zA, zB, zC = mins["A"], mins["B"], mins["C"]
    Z3 = torch.stack([zA, zB, zC])

    # relay atoms through the middle hub B: {+-r_BA, +-r_BC}. No direct A-C.
    dz_list = [zA - zB, zB - zA, zC - zB, zB - zC]
    atoms_z = torch.stack([torch.cat([dz, torch.zeros(8, dtype=torch.float64,
                                                      device=device)])
                           for dz in dz_list])
    atoms_x = pot.from_latent(atoms_z)                       # (4, 10)
    weights = torch.full((4,), 0.25, dtype=torch.float64, device=device)
    h = 0.1 * float(atoms_x.norm(dim=1).min().item())
    law = ShellJumpLaw(atoms_x, weights, h=h)
    drift_cap = 2.0 * h

    # latent box: >= one max jump length (~1.07) beyond the three minima
    lo2d, hi2d = (-2.0, -1.3), (1.9, 2.6)
    lo_lat = [lo2d[0], lo2d[1]] + [-2.0] * 8
    hi_lat = [hi2d[0], hi2d[1]] + [2.0] * 8
    box = LatentRectBox(lo_lat, hi_lat, pot)

    ref = Latent2DGaussianReference(pot, lambda z: -E3_BETA * mb3_2d(z),
                                    lo2d, hi2d, E3_BETA)

    basins = GradientFlowBasinMap2D(mb3_2d_grad, Z3, lo2d, hi2d,
                                    n_grid=600, device=device, cache=basin_cache)
    p_star = basins.p_star(lambda z: -E3_BETA * mb3_2d(z))

    def init_fn(n, gen):
        z = torch.zeros(n, 10, dtype=torch.float64, device=device)
        z[:, :2] = zC                                        # init in well C
        z += 0.05 * torch.randn(n, 10, generator=gen, device=device,
                                dtype=torch.float64)
        return pot.from_latent(z)

    def make_score(q_theta=Q_THETA, q_rho=Q_RHO):
        return ShellScore(pot, law, LAMBDA, E3_BETA, q_theta, q_rho)

    def labels_fn(x):
        return basins.assign(pot.to_latent(x)[:, :2])

    def metric_space(x):
        return pot.to_latent(x)[:, :2]

    # slow timescale: A<->B barrier (Kramers tau, astronomically beyond T)
    bAB = MB3_CRITICAL["S_AB"][1] - MB3_CRITICAL["B"][1]     # 0.4634
    kramers = 2.0 * math.pi * math.exp(min(E3_BETA * bAB, 700.0))

    return Experiment(
        name="mb3well_10d", cfg=cfg, pot=pot, law=law, box=box,
        init_fn=init_fn, ref_sample=lambda n, g: ref.sample(n, g),
        make_score=make_score, labels_fn=labels_fn, p_star=p_star,
        metric_space=metric_space,
        # hot replica must cross b(A<->B): beta_min * b(A<->B) ~ 2
        pt_beta_min=2.0 / bAB,
        # committed exit = arrival in the FAR well A's core (the slow crossing);
        # B<->C is designed to be reachable, so gate specifically on A.
        exit_committed=lambda x: (
            torch.cdist(pot.to_latent(x)[:, :2], zA.unsqueeze(0)).squeeze(1) < 0.25),
        kramers_tau=kramers,
        cp_drift_cap=drift_cap,
        extras={"minima_latent": mins, "atoms_z": atoms_z, "h": h,
                "basins": basins, "ref": ref, "Z3": Z3, "beta": E3_BETA,
                "lo2d": lo2d, "hi2d": hi2d, "b_AB": bAB,
                "b_BC": MB3_CRITICAL["S_BC"][1] - MB3_CRITICAL["B"][1],
                # generous certificate box (>= one jump beyond support; the
                # tighter sampling box would read a large residual): the
                # order-one identity mass lives where pi is tiny and S enormous.
                "cert_lo": [-3.2, -2.4], "cert_hi": [3.0, 3.7],
                # secondary barrier check: arrival in B core from C (reachable)
                "exit_to_B": lambda x: (
                    torch.cdist(pot.to_latent(x)[:, :2], zB.unsqueeze(0)).squeeze(1) < 0.25)},
    )


# ============================================= E3 (archived 4-well variant)
def build_e3_mb4well(device="cuda", basin_cache: str | None = None) -> Experiment:
    """E3: 4-well modified Mueller-Brown (archive/mueller.py precedent),
    embedded in 10D exactly as before (x = z B^T). At beta = 8 the target is
    genuinely multimodal AND metastable: masses ~ (0.617, 0.338, 0.027,
    0.018); W1<->W2 share a beta*16.5 saddle; W3/W4 are islands behind
    plateau-level walls (beta*b ~ 80) that only nonlocal transport can
    populate. Init on island W3. ARCHIVED appendix stress test (plateau-walled
    islands); the mb3 build_e3 above is the main E3."""
    pot = TransformedMB4Well10D(device=device)
    cfg = RunConfig(name="mb4well_10d", d=10, n_particles=2000, T=200.0,
                    dt=0.005)

    mins = {}
    for key in ("W1", "W2", "W3", "W4"):
        z0 = torch.tensor(MB4_CRITICAL[key][0], dtype=torch.float64, device=device)
        mins[key] = newton_refine(mb4_2d_grad, z0)
    Z4 = torch.stack([mins[k] for k in ("W1", "W2", "W3", "W4")])

    # jump law by the measured E3 design rules: complete graph over the four
    # latent minima (every well has >= 1.8% mass -- no negligible relay
    # targets), O(1) uniform weights (mass-ratio skew measurably backfires),
    # and the CP pair's drift step capped at 2h (shell resolution scale).
    dz_list = [Z4[j] - Z4[i] for i in range(4) for j in range(4) if i != j]
    atoms_z = torch.stack([torch.cat([dz, torch.zeros(8, dtype=torch.float64,
                                                      device=device)])
                           for dz in dz_list])
    atoms_x = pot.from_latent(atoms_z)                       # (12, 10)
    weights = torch.full((12,), 1.0 / 12.0, dtype=torch.float64, device=device)
    h = 0.1 * float(atoms_x.norm(dim=1).min().item())
    law = ShellJumpLaw(atoms_x, weights, h=h)
    drift_cap = 2.0 * h

    lo_lat = [-2.0, -1.7] + [-2.0] * 8
    hi_lat = [2.2, 2.7] + [2.0] * 8
    box = LatentRectBox(lo_lat, hi_lat, pot)

    ref = Latent2DGaussianReference(pot, lambda z: -BETA * mb4_2d(z),
                                    (-2.0, -1.7), (2.2, 2.7), BETA)

    basins = GradientFlowBasinMap2D(mb4_2d_grad, Z4, (-2.0, -1.7), (2.2, 2.7),
                                    n_grid=600, device=device, cache=basin_cache)
    p_star = basins.p_star(lambda z: -BETA * mb4_2d(z))

    def init_fn(n, gen):
        z = torch.zeros(n, 10, dtype=torch.float64, device=device)
        z[:, :2] = mins["W3"]
        z += 0.05 * torch.randn(n, 10, generator=gen, device=device,
                                dtype=torch.float64)
        return pot.from_latent(z)

    def make_score(q_theta=Q_THETA, q_rho=Q_RHO):
        return ShellScore(pot, law, LAMBDA, BETA, q_theta, q_rho)

    def labels_fn(x):
        return basins.assign(pot.to_latent(x)[:, :2])

    def metric_space(x):
        return pot.to_latent(x)[:, :2]

    # crude escape estimate from the W3 island: the wall is the V ~ 0
    # plateau, so tau ~ 2 pi e^{beta * (0 - V(W3))} -- astronomically beyond
    # any simulation; committed local exits should be ZERO.
    barrier_W3 = float(-mb4_2d(mins["W3"].unsqueeze(0)).item())
    kramers = 2.0 * math.pi * math.exp(min(BETA * barrier_W3, 700.0))

    zW1, zW2, zW4 = mins["W1"], mins["W2"], mins["W4"]
    return Experiment(
        name="mb4well_10d", cfg=cfg, pot=pot, law=law, box=box,
        init_fn=init_fn, ref_sample=lambda n, g: ref.sample(n, g),
        make_score=make_score, labels_fn=labels_fn, p_star=p_star,
        metric_space=metric_space,
        # hot replica must climb the beta*b~80 plateau walls: beta_min such
        # that beta_min * 10.4 ~ 2
        pt_beta_min=0.2,
        # committed exit from island W3: arrival in any other well core
        exit_committed=lambda x: (
            torch.cdist(pot.to_latent(x)[:, :2],
                        torch.stack([zW1, zW2, zW4])).min(dim=1).values < 0.25),
        kramers_tau=kramers,
        cp_drift_cap=drift_cap,
        extras={"minima_latent": mins, "atoms_z": atoms_z, "h": h,
                "basins": basins, "barrier_W3": barrier_W3, "ref": ref,
                "Z4": Z4},
    )


# ===================================================================== E4
def build_e4(device="cuda", basin_cache: str | None = None,
             jitter_sigma: float = 0.0) -> Experiment:
    pot = CoupledPhi4()
    cfg = RunConfig(name="coupled_phi4", d=24, n_particles=1000, T=100.0,
                    dt=0.002)

    phases = ["--", "-+", "+-", "++"]                        # idx 0,1,2,3
    vs = []
    for ph in phases:
        v0 = torch.tensor(PHI4_MINIMA[ph][0], dtype=torch.float64, device=device)
        vs.append(newton_refine(phi4_W_grad, v0))
    V2 = torch.stack(vs)                                     # (4, 2)
    # coherent states 1_{Ns} (x) v; flat layout is (x0,y0,x1,y1,...), i.e.
    # sites = flat.reshape(Ns, 2), so tile v per site:
    means24 = V2.unsqueeze(1).expand(4, pot.Ns, 2).reshape(4, 24).contiguous()

    # 8 EDGE atoms of the phase square: drop the two diagonal pairs
    # (-- <-> ++) = {0,3} and (-+ <-> +-) = {1,2}, whose coherent chords cross
    # the field-zero hilltop at the centre. Diagonal transitions relay through
    # a mixed phase in two hops instead. Coherent tiling 1_{Ns} (x) (v_j - v_i).
    _DIAGONALS = ({0, 3}, {1, 2})
    atom_list, edge_pairs = [], []
    for i in range(4):
        for j in range(4):
            if i != j and {i, j} not in _DIAGONALS:
                dv = V2[j] - V2[i]
                atom_list.append(dv.unsqueeze(0).expand(pot.Ns, 2).reshape(24))
                edge_pairs.append((phases[i], phases[j]))
    atoms = torch.stack(atom_list)                           # (8, 24)
    A_e4 = atoms.shape[0]
    weights = torch.full((A_e4,), 1.0 / A_e4, dtype=torch.float64, device=device)
    h = 0.1 * float(atoms.norm(dim=1).min().item())
    # optional per-site transverse jitter (RA-LSC only; free because the RA
    # score needs no closed-form quadrature over the jump law). Off by default.
    if jitter_sigma > 0.0:
        from .jumps import JitteredShellJumpLaw
        law = JitteredShellJumpLaw(atoms, weights, h, jitter_sigma)
    else:
        law = ShellJumpLaw(atoms, weights, h=h)
    # drift cap = max ||r_a||: the coherent paths cross no foreign basin, so
    # the detailed-balance return flow is best integrated by steps that may
    # retrace a full jump (measured: pi-start TV 0.052 at cap=1 vs 0.023 at
    # cap=||r||, floor 0.018). Contrast E3, whose chords cross a foreign
    # basin and need small in-tube steps (cap=2h).
    drift_cap_e4 = float(atoms.norm(dim=1).max().item())

    box = RectBox([-2.0] * 24, [2.0] * 24, device)

    # Laplace reference: 24x24 Hessians at the coherent minima (autograd)
    from torch.autograd.functional import hessian as _th_hessian
    Hs = []
    for k in range(4):
        Hk = _th_hessian(lambda q: pot._V_raw(q.unsqueeze(0))[0],
                         means24[k].clone())
        Hs.append(0.5 * (Hk + Hk.T))
    H24 = torch.stack(Hs)
    energies = phi4_W(V2)                                    # V(1 (x) v) = W(v)
    laplace = LaplaceMixture(means24, H24, energies, BETA)

    basins = GradientFlowBasinMap2D(phi4_W_grad, V2, (-2.0, -2.0), (2.0, 2.0),
                                    n_grid=400, device=device, cache=basin_cache)

    def qbar(x):
        return x.reshape(-1, pot.Ns, 2).mean(dim=1)          # (N, 2)

    def labels_fn(x):
        return basins.assign(qbar(x))

    # reference: EXACT pi via SNIS resampling from the Laplace mixture
    # (Laplace itself is kept in extras as the proposal / cross-check);
    # p_star from a large fixed-seed exact draw.
    def ref_sample(n, gen):
        return laplace.sample_exact_snis(n, gen, pot, BETA)

    g_p = torch.Generator(device=device)
    g_p.manual_seed(31337)
    from .metrics import occupancy as _occ
    p_star = _occ(labels_fn(laplace.sample_exact_snis(200_000, g_p, pot, BETA)), 4)

    def init_fn(n, gen):
        return means24[0] + 0.05 * torch.randn(n, 24, generator=gen,
                                               device=device, dtype=torch.float64)

    def make_score(q_theta=Q_THETA, q_rho=Q_RHO):
        return ShellScore(pot, law, LAMBDA, BETA, q_theta, q_rho)

    # Kramers for the coherent -- escape: homogeneous soft modes dominate;
    # use full-24D overdamped Kramers with the coherent saddle of W
    sad0 = newton_refine(phi4_W_grad, torch.tensor([-1.0, 0.0], dtype=torch.float64,
                                                   device=device))
    sad24 = sad0.unsqueeze(0).expand(pot.Ns, 2).reshape(24)
    Hsad = _th_hessian(lambda q: pot._V_raw(q.unsqueeze(0))[0], sad24.clone())
    Hsad = 0.5 * (Hsad + Hsad.T)
    ev_s = torch.linalg.eigvalsh(Hsad)
    lam_neg = float(-ev_s[0].item())
    ev_b = torch.linalg.eigvalsh(H24[0])
    log_pref = 0.5 * (torch.log(ev_s.abs()).sum() - torch.log(ev_b).sum())
    barrier = float((phi4_W(sad0.unsqueeze(0)) - energies[0]).item())
    kramers = (2.0 * math.pi / lam_neg) * math.exp(float(log_pref.item())) \
        * math.exp(BETA * barrier)

    return Experiment(
        name="coupled_phi4", cfg=cfg, pot=pot, law=law, box=box,
        init_fn=init_fn, ref_sample=ref_sample,
        make_score=make_score, labels_fn=labels_fn, p_star=p_star,
        metric_space=qbar,
        pt_beta_min=1.0,
        # committed exit from --: mean order parameter within 0.35 of another
        # coherent minimum
        exit_committed=lambda x: (
            (qbar(x).unsqueeze(1) - V2[1:].unsqueeze(0)).norm(dim=2).min(dim=1)
            .values < 0.35),
        kramers_tau=kramers,
        cp_drift_cap=drift_cap_e4,
        extras={"minima_2d": V2, "means24": means24, "hessians": H24,
                "laplace": laplace, "basins": basins, "h": h,
                "barrier_minus_minus": barrier, "phases": phases,
                "edge_pairs": edge_pairs, "jitter_sigma": jitter_sigma},
    )


def _mog40_committed_exit(pot):
    def fn(x):
        d = torch.cdist(x, pot.mu)                 # (N, 40)
        lab = d.argmin(dim=1)
        return (lab != 0) & (d.min(dim=1).values < 2.0)
    return fn


# ============================================================ sampler wiring
def make_sampler_factory(exp: Experiment, dt: float, pt_betas: torch.Tensor,
                         n_particles: int | None = None,
                         score_kwargs: dict | None = None):
    """Factory (method, seed) -> fresh sampler; x0 shared across methods per
    seed; CP and LSC-CP share the jump stream (pathwise coupling)."""
    dev = exp.p_star.device
    N = n_particles or exp.cfg.n_particles
    eps = exp.cfg.eps
    beta = exp.cfg.beta
    lam = exp.cfg.lam
    score_kwargs = score_kwargs or {}

    def factory(method: str, seed: int):
        g_init = torch.Generator(device=dev)
        g_init.manual_seed(init_seed(seed))
        x0 = exp.init_fn(N, g_init)
        gen = torch.Generator(device=dev)
        gen.manual_seed(diffusion_seed(method, seed))
        if method == "ULA":
            return ULA(exp.pot, x0, dt, eps, gen, exp.box)
        if method == "MALA":
            return MALA(exp.pot, x0, dt, beta, gen, exp.box)
        if method == "FLA":
            return FLA(exp.pot, x0, dt, beta, gen, exp.box)
        if method == "BAOAB":
            return BAOAB(exp.pot, x0, dt, eps, gen, exp.box)
        if method == "PT":
            return ParallelTempering(exp.pot, x0, dt, pt_betas, gen, exp.box)
        if method in ("CP", "LSC-CP"):
            g_jump = torch.Generator(device=dev)
            g_jump.manual_seed(jump_seed(seed))               # SHARED stream
            score = exp.make_score(**score_kwargs) if method == "LSC-CP" else None
            return CompoundPoisson(exp.pot, x0, dt, eps, lam, exp.law,
                                   gen, g_jump, exp.box, score=score,
                                   name=method, drift_cap=exp.cp_drift_cap)
        if method in ("CP-RA", "LSC-CP-RA"):
            g_jump = torch.Generator(device=dev)
            g_jump.manual_seed(jump_seed(seed))               # SHARED (RA pair)
            q_theta = score_kwargs.get("q_theta", Q_THETA)
            score = (RandomAtomicShellScore(exp.pot, exp.law, lam, beta, q_theta)
                     if method == "LSC-CP-RA" else None)
            return CompoundPoisson(exp.pot, x0, dt, eps, lam, exp.law,
                                   gen, g_jump, exp.box, score=score,
                                   name=method, drift_cap=exp.cp_drift_cap,
                                   jump_mode="atomic")
        raise ValueError(method)

    return factory


def make_batched_factory(exp: Experiment, dt: float, pt_betas: torch.Tensor,
                         seeds, n_particles: int | None = None,
                         score_kwargs: dict | None = None):
    """All seeds batched into one (S*N)-particle ensemble per method (the
    wall-clock axis is no longer reported, so sequential per-seed timing is
    unnecessary and the GPU is far better utilised). Per-seed x0 blocks are
    generated exactly as in the sequential path; CP and LSC-CP still share
    one jump stream."""
    dev = exp.p_star.device
    N = n_particles or exp.cfg.n_particles
    eps, beta, lam = exp.cfg.eps, exp.cfg.beta, exp.cfg.lam
    score_kwargs = score_kwargs or {}

    def factory(method: str):
        blocks = []
        for seed in seeds:
            g = torch.Generator(device=dev)
            g.manual_seed(init_seed(seed))
            blocks.append(exp.init_fn(N, g))
        x0 = torch.cat(blocks, dim=0)
        gen = torch.Generator(device=dev)
        gen.manual_seed(diffusion_seed(method, 0))
        if method == "ULA":
            return ULA(exp.pot, x0, dt, eps, gen, exp.box)
        if method == "MALA":
            return MALA(exp.pot, x0, dt, beta, gen, exp.box)
        if method == "FLA":
            return FLA(exp.pot, x0, dt, beta, gen, exp.box)
        if method == "BAOAB":
            return BAOAB(exp.pot, x0, dt, eps, gen, exp.box)
        if method == "PT":
            return ParallelTempering(exp.pot, x0, dt, pt_betas, gen, exp.box)
        if method in ("CP", "LSC-CP"):
            g_jump = torch.Generator(device=dev)
            g_jump.manual_seed(jump_seed(0))                 # SHARED stream
            score = exp.make_score(**score_kwargs) if method == "LSC-CP" else None
            return CompoundPoisson(exp.pot, x0, dt, eps, lam, exp.law,
                                   gen, g_jump, exp.box, score=score,
                                   name=method, drift_cap=exp.cp_drift_cap)
        if method in ("CP-RA", "LSC-CP-RA"):
            g_jump = torch.Generator(device=dev)
            g_jump.manual_seed(jump_seed(0))                 # SHARED (RA pair)
            q_theta = score_kwargs.get("q_theta", Q_THETA)
            score = (RandomAtomicShellScore(exp.pot, exp.law, lam, beta, q_theta)
                     if method == "LSC-CP-RA" else None)
            return CompoundPoisson(exp.pot, x0, dt, eps, lam, exp.law,
                                   gen, g_jump, exp.box, score=score,
                                   name=method, drift_cap=exp.cp_drift_cap,
                                   jump_mode="atomic")
        raise ValueError(method)

    return factory


# ============================================================ metric wiring
def make_metrics(exp: Experiment, n: int, ref_seed: int = 424242,
                 device="cuda", floor_replicates: int = 20):
    """Frozen reference sample (size n = run's N), frozen projections,
    frozen MMD bandwidth, bias floors. Returns (metrics_fn, floors)."""
    g_ref = torch.Generator(device=device)
    g_ref.manual_seed(ref_seed)
    ref_x = exp.ref_sample(n, g_ref)
    ref_m = exp.metric_space(ref_x)
    d_m = ref_m.shape[1]
    K = exp.p_star.shape[0]

    proj = M.make_projections(d_m, 200, seed=777, device=device) if d_m > 1 else None
    bw = M.median_heuristic(ref_m)

    # ---- reference quantities for the chemistry-native + KSD metrics --------
    beta_m = exp.cfg.beta
    with exp.pot.no_count():
        ref_V = exp.pot.V(ref_x)                              # (n,)
    ref_mean_V = float(ref_V.mean().item())
    ref_var_V = float(ref_V.var(unbiased=True).item())
    # free-energy CV = first metric-space coordinate (z1 for E3, x for E1, ...)
    ref_cv = ref_m[:, 0]
    cv_lo, cv_hi = float(ref_cv.min().item()), float(ref_cv.max().item())
    cv_pad = 0.05 * (cv_hi - cv_lo + 1e-9)
    cv_edges = torch.linspace(cv_lo - cv_pad, cv_hi + cv_pad, 41,
                              dtype=torch.float64, device=device)
    ref_F, ref_p = M.free_energy_profile(ref_cv, cv_edges, beta_m)
    pi_min = 5.0 / n                                          # >= ~5 ref counts/bin
    e_lo, e_hi = float(ref_V.min().item()), float(ref_V.max().item())
    e_pad = 0.05 * (e_hi - e_lo + 1e-9)
    e_edges = torch.linspace(e_lo - e_pad, e_hi + e_pad, 41,
                             dtype=torch.float64, device=device)
    _ei = torch.bucketize(ref_V.reshape(-1), e_edges[1:-1])
    ref_ehist = torch.bincount(_ei, minlength=e_edges.shape[0] - 1).to(torch.float64)
    ref_ehist = ref_ehist / ref_ehist.sum()

    is_e1 = exp.name == "double_well"
    if is_e1:
        lo, hi = exp.extras["density_tv_box"]
        nb = exp.extras["density_tv_bins"]
        edges = torch.linspace(lo, hi, nb + 1, dtype=torch.float64, device=device)
        centers = 0.5 * (edges[1:] + edges[:-1])
        logp = -BETA * (centers * centers - 1.0) ** 2
        mass = torch.exp(logp - logp.max())
        target_mass = mass / mass.sum()

    proj10 = None
    if exp.cfg.d == 10:      # latent-metric experiments also report full-10D W2
        proj10 = M.make_projections(10, 200, seed=778, device=device)

    def w2_fn(a, b):
        return M.w2_exact_1d(a, b) if d_m == 1 else M.sliced_w2(a, b, proj)

    uniform = exp.uniform_target

    def emc_fn(p_hat):
        # near 1 = better in BOTH cases: exp(H)/K for uniform targets,
        # 1 - EJS(p_hat, p*) for non-uniform targets
        return M.emc(p_hat) if uniform else 1.0 - M.ejs(p_hat, exp.p_star)

    def metrics_fn(x):
        xm = exp.metric_space(x)
        labels = exp.labels_fn(x)
        p_hat = M.occupancy(labels, K)
        out = {
            "W2": w2_fn(xm, ref_m),
            "TV": M.occupancy_tv(p_hat, exp.p_star),
            "MMD": M.mmd_biased(xm, ref_m, bw),
            "EMC": emc_fn(p_hat),
            "nonfinite_frac": M.nonfinite_frac(x),
        }
        if is_e1:
            out["TV_density"] = M.density_tv_1d(x, edges, target_mass)
        if proj10 is not None:
            out["W2_10d"] = M.sliced_w2(x, ref_x, proj10)
        # ---- chemistry-native + KSD (potential evals excluded from NFE) ----
        cv = xm[:, 0]
        out["e_F"] = M.free_energy_profile_error(cv, cv_edges, beta_m,
                                                 ref_F, ref_p, pi_min)
        brel, bL1 = M.basin_rel_mass_error(p_hat, exp.p_star)
        out["basin_rel_max"] = brel
        out["basin_L1"] = bL1
        out["occ0"] = float(p_hat[0].item())     # slow-mode scalar (basin 0)
        with exp.pot.no_count():
            V = exp.pot.V(x)
            gV = exp.pot.grad(x)
        eV, eVar = M.observable_error(V, ref_mean_V, ref_var_V)
        out["V_mean_err"] = eV
        out["V_var_err"] = eVar
        out["E_overlap_deficit"] = 1.0 - M.energy_hist_overlap(V, e_edges, ref_ehist)
        out["KSD"] = M.ksd_imq(x, -beta_m * gV)
        return out

    two = {"W2": w2_fn, "MMD": lambda a, b: M.mmd_biased(a, b, bw)}

    def sample_ref_metric(nn, gen):
        return exp.metric_space(exp.ref_sample(nn, gen))

    floors = M.bias_floors(sample_ref_metric, two, {}, n,
                           replicates=floor_replicates, device=device)
    # occupancy-type floors need raw x, not metric space
    floors_occ = M.bias_floors(exp.ref_sample, {}, _occ_floor_fns(exp, K),
                               n, replicates=floor_replicates, device=device)
    floors.update(floors_occ)
    if proj10 is not None:
        floors_10 = M.bias_floors(exp.ref_sample,
                                  {"W2_10d": lambda a, b: M.sliced_w2(a, b, proj10)},
                                  {}, n, replicates=floor_replicates, device=device)
        floors.update(floors_10)
    return metrics_fn, floors, {"bandwidth": bw, "ref_x": ref_x}


def _occ_floor_fns(exp: Experiment, K: int):
    uniform = exp.uniform_target

    def _tv(x):
        return M.occupancy_tv(M.occupancy(exp.labels_fn(x), K), exp.p_star)

    def _emc(x):
        p_hat = M.occupancy(exp.labels_fn(x), K)
        return M.emc(p_hat) if uniform else 1.0 - M.ejs(p_hat, exp.p_star)

    out = {"TV": _tv, "EMC": _emc}
    if exp.name == "double_well":
        # density TV floor: fresh reference sample against the exact bins
        lo, hi = exp.extras["density_tv_box"]
        nb = exp.extras["density_tv_bins"]
        dev = exp.p_star.device
        edges = torch.linspace(lo, hi, nb + 1, dtype=torch.float64, device=dev)
        centers = 0.5 * (edges[1:] + edges[:-1])
        logp = -BETA * (centers * centers - 1.0) ** 2
        mass = torch.exp(logp - logp.max())
        target_mass = mass / mass.sum()
        out["TV_density"] = lambda x: M.density_tv_1d(x, edges, target_mass)
    return out
