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
from .potentials import (CoupledPhi4, DoubleWell1D, MB4_CRITICAL,
                         MB_CRITICAL, MB4Latent2D, MoG40,
                         MuellerBrownLatent2D, PHI4_MINIMA,
                         TransformedMB4Well10D, TransformedMuellerBrown10D,
                         mb4_2d, mb4_2d_grad, muller_brown_2d,
                         muller_brown_2d_grad, newton_refine, phi4_W,
                         phi4_W_grad)
from .references import (GradientFlowBasinMap2D, Grid1DInverseCDF,
                         LaplaceMixture, Latent2DGaussianReference,
                         MB10DReference)
from .samplers import (BAOAB, FLA, MALA, ULA, CompoundPoisson, LatentRectBox,
                       ParallelTempering, RectBox)
from .score import MoG40Score, ShellScore
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
    box = RectBox([-3.0], [3.0], device)
    ref = Grid1DInverseCDF(lambda x: -BETA * (x * x - 1.0) ** 2, -3.0, 3.0,
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
        extras={"ref": ref, "density_tv_box": (-3.0, 3.0), "density_tv_bins": 200},
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
def build_e3(device="cuda", basin_cache: str | None = None) -> Experiment:
    """E3: 4-well modified Mueller-Brown (archive/mueller.py precedent),
    embedded in 10D exactly as before (x = z B^T). At beta = 8 the target is
    genuinely multimodal AND metastable: masses ~ (0.617, 0.338, 0.027,
    0.018); W1<->W2 share a beta*16.5 saddle; W3/W4 are islands behind
    plateau-level walls (beta*b ~ 80) that only nonlocal transport can
    populate. Init on island W3."""
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
def build_e4(device="cuda", basin_cache: str | None = None) -> Experiment:
    pot = CoupledPhi4()
    cfg = RunConfig(name="coupled_phi4", d=24, n_particles=1000, T=100.0,
                    dt=0.002)

    phases = ["--", "-+", "+-", "++"]
    vs = []
    for ph in phases:
        v0 = torch.tensor(PHI4_MINIMA[ph][0], dtype=torch.float64, device=device)
        vs.append(newton_refine(phi4_W_grad, v0))
    V2 = torch.stack(vs)                                     # (4, 2)
    # coherent states 1_{Ns} (x) v; flat layout is (x0,y0,x1,y1,...), i.e.
    # sites = flat.reshape(Ns, 2), so tile v per site:
    means24 = V2.unsqueeze(1).expand(4, pot.Ns, 2).reshape(4, 24).contiguous()

    # complete graph over the 4 minima: 12 directed homogeneous atoms
    atom_list = []
    for i in range(4):
        for j in range(4):
            if i != j:
                dv = V2[j] - V2[i]
                atom_list.append(dv.unsqueeze(0).expand(pot.Ns, 2).reshape(24))
    atoms = torch.stack(atom_list)                           # (12, 24)
    weights = torch.full((12,), 1.0 / 12.0, dtype=torch.float64, device=device)
    h = 0.1 * float(atoms.norm(dim=1).min().item())
    law = ShellJumpLaw(atoms, weights, h=h)

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
        extras={"minima_2d": V2, "means24": means24, "hessians": H24,
                "laplace": laplace, "basins": basins, "h": h,
                "barrier_minus_minus": barrier, "phases": phases},
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
    if exp.name == "muller_brown_10d":
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
