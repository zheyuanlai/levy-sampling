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

from .config import (M_PHI, Q_RHO, Q_THETA, RunConfig,
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
from .score import (MoG40Score, MultiAtomShellScore, RandomAtomicShellScore,
                    ShellScore)
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
    # N=16000 / 24 seeds: the production metrics plateau at their MC-noise
    # floor, and at N=4000 x 5 seeds the tail band CV reached 0.6-0.7 for the
    # histogram TV (350 bins ~ 11 particles/bin) and ~0.5 for KSD -- visually
    # dominant on a log axis. 1D is cheap; 4x particles + ~5x seeds cut the
    # plateau noise ~4-5x. (Verified not a bug: reference sample, projections
    # and MMD bandwidth are all frozen in make_metrics.)
    cfg = RunConfig(name="double_well", d=1, n_particles=16000, T=100.0,
                    dt=0.005, seeds=tuple(range(24)))
    atoms = torch.tensor([[2.0], [-2.0]], dtype=torch.float64, device=device)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64, device=device)
    law = ShellJumpLaw(atoms, weights, h=0.2)     # +-2 maps minimum to minimum
    # generous box = the certificate domain: pi has ~no mass beyond +-2, so
    # LSC-CP never hits the boundary, but raw CP injects tail/barrier mass out to
    # ~+-3.5 -- a tight [-3,3] clip would pile that mass at the edge and confound
    # the raw-CP CDF. Widening removes the clip artifact (LSC-CP unaffected).
    box = RectBox([-5.2], [5.2], device)
    reference_bounds = (-5.2, 5.2)
    reference_n_grid = 200_001
    ref = Grid1DInverseCDF(
        lambda x: -cfg.beta * (x * x - 1.0) ** 2,
        *reference_bounds, n_grid=reference_n_grid, device=device)

    def init_fn(n, gen):
        return -1.0 + 0.05 * torch.randn(n, 1, generator=gen, device=device,
                                         dtype=torch.float64)

    def make_score(q_theta=Q_THETA, q_rho=Q_RHO):
        return ShellScore(pot, law, cfg.lam, cfg.beta, q_theta, q_rho)

    p_star = torch.tensor([0.5, 0.5], dtype=torch.float64, device=device)
    return Experiment(
        name="double_well", cfg=cfg, pot=pot, law=law, box=box,
        init_fn=init_fn, ref_sample=lambda n, g: ref.sample(n, g),
        make_score=make_score,
        labels_fn=lambda x: (x[:, 0] > 0).long(),
        p_star=p_star, metric_space=lambda x: x,
        pt_beta_min=1.0,
        exit_committed=lambda x: x[:, 0] > 0.7,     # right-well core arrival
        kramers_tau=DoubleWell1D.kramers_time(cfg.beta),
        # density-TV bins span the widened box; bump count to keep ~0.03 width
        extras={"ref": ref, "density_tv_box": (-5.2, 5.2),
                "density_tv_bins": 350,
                "reference_sample_method": "numerical_inverse_cdf",
                "builder_reference_parameters": {
                    "inverse_cdf_bounds": list(reference_bounds),
                    "inverse_cdf_n_grid": reference_n_grid,
                }},
    )


# ===================================================================== E2
def build_e2(device="cuda") -> Experiment:
    # 24 seeds (N kept at 2500: the exact ShellScore's per-step quadrature cost
    # scales with the batched ensemble, and E2 runs the exact+RA dual matrix)
    cfg = RunConfig(name="mog40", d=2, n_particles=2500, T=100.0, dt=0.01,
                    seeds=tuple(range(24)))
    pot = MoG40(beta=cfg.beta, device=device)
    # deliberately generic law: [4, 15] set from the NN-distance histogram
    # alone; neither PT nor LSC-CP receives mode locations.
    # The band is narrow ([10, 14], width 4 rather than the former [4, 15]) so
    # the radial rule needs few nodes, while b = 14 still keeps the mode graph
    # connected -- measured 1 component at both the 1e-3 and 3e-3 mode-hit
    # thresholds (diameter 12 / 14), matching the former band's 11 / 13.
    law = AnnulusJumpLaw(10.0, 14.0, device)
    box = RectBox([-65.0, -65.0], [65.0, 65.0], device)

    def init_fn(n, gen):
        return pot.mu[0] + 0.5 * torch.randn(n, 2, generator=gen, device=device,
                                             dtype=torch.float64)

    # E2-specific quadrature defaults, NOT the global Q_THETA/Q_RHO/M_PHI. The
    # annulus rule is limited by its angular order: measured against a fine
    # analytic comparator, q_rho saturates by 3-4 (8 is no better than 4) while
    # m_phi 32 -> 48 improves the median relative score error 67x, from 2.3e-3
    # to 3.5e-5. Inheriting the global M_PHI = 32 would silently under-resolve.
    def make_score(q_theta=16, q_rho=4, m_phi=64, **kw):
        # Numerical integration only. The analytic MoG40Score is retained in
        # score.py as an exactness comparator (tests, certificate_gate) but is
        # NOT deployed: it needs the mixture means, so it does not generalise.
        # Annulus quadrature is q_theta x q_rho x m_phi chord energies.
        return ShellScore(pot, law, cfg.lam, cfg.beta, q_theta, q_rho,
                          m_phi=m_phi, **kw)

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
        extras={"nn_dist_mode0": d0, "beta_dV_mode0": beta_dV,
                "reference_sample_method": "exact_gaussian_mixture",
                "builder_reference_parameters": {
                    "exact_mixture_components": int(pot.mu.shape[0]),
                    "component_covariance": "identity",
                }},
    )


# ===================================================================== E3
def build_e3(device="cuda", basin_cache: str | None = None,
             beta: float = 24.0, *,
             basin_n_grid: int = 600,
             basin_flow_steps: int = 40_000,
             basin_flow_dt: float = 1.5e-4,
             basin_mass_n_quad: int = 1200,
             reference_grid_shape: tuple[int, int] = (2400, 2400)) -> Experiment:
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
                    dt=0.005, beta=E3_BETA, seeds=tuple(range(16)))

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

    ref = Latent2DGaussianReference(
        pot, lambda z: -E3_BETA * mb3_2d(z), lo2d, hi2d, E3_BETA,
        shape=reference_grid_shape)

    basins = GradientFlowBasinMap2D(
        mb3_2d_grad, Z3, lo2d, hi2d, n_grid=basin_n_grid,
        device=device, cache=basin_cache, dt_flow=basin_flow_dt,
        n_flow=basin_flow_steps)
    p_star = basins.p_star(
        lambda z: -E3_BETA * mb3_2d(z), n_quad=basin_mass_n_quad)

    def init_fn(n, gen):
        z = torch.zeros(n, 10, dtype=torch.float64, device=device)
        z[:, :2] = zC                                        # init in well C
        z += 0.05 * torch.randn(n, 10, generator=gen, device=device,
                                dtype=torch.float64)
        return pot.from_latent(z)

    def make_score(q_theta=Q_THETA, q_rho=Q_RHO):
        return ShellScore(pot, law, cfg.lam, cfg.beta, q_theta, q_rho)

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
                "basin_cache_provenance": basins.cache_provenance(),
                "lo2d": lo2d, "hi2d": hi2d, "b_AB": bAB,
                "b_BC": MB3_CRITICAL["S_BC"][1] - MB3_CRITICAL["B"][1],
                "reference_sample_method": "numerical_grid_times_gaussian",
                "builder_reference_parameters": {
                    "latent_reference_bounds": [list(lo2d), list(hi2d)],
                    "latent_reference_grid_shape": list(reference_grid_shape),
                    "basin_n_grid": int(basin_n_grid),
                    "basin_flow_steps": int(basin_flow_steps),
                    "basin_flow_dt": float(basin_flow_dt),
                    "basin_mass_n_quad": int(basin_mass_n_quad),
                },
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

    ref = Latent2DGaussianReference(pot, lambda z: -cfg.beta * mb4_2d(z),
                                    (-2.0, -1.7), (2.2, 2.7), cfg.beta)

    basins = GradientFlowBasinMap2D(mb4_2d_grad, Z4, (-2.0, -1.7), (2.2, 2.7),
                                    n_grid=600, device=device, cache=basin_cache)
    p_star = basins.p_star(lambda z: -cfg.beta * mb4_2d(z))

    def init_fn(n, gen):
        z = torch.zeros(n, 10, dtype=torch.float64, device=device)
        z[:, :2] = mins["W3"]
        z += 0.05 * torch.randn(n, 10, generator=gen, device=device,
                                dtype=torch.float64)
        return pot.from_latent(z)

    def make_score(q_theta=Q_THETA, q_rho=Q_RHO):
        return ShellScore(pot, law, cfg.lam, cfg.beta, q_theta, q_rho)

    def labels_fn(x):
        return basins.assign(pot.to_latent(x)[:, :2])

    def metric_space(x):
        return pot.to_latent(x)[:, :2]

    # crude escape estimate from the W3 island: the wall is the V ~ 0
    # plateau, so tau ~ 2 pi e^{beta * (0 - V(W3))} -- astronomically beyond
    # any simulation; committed local exits should be ZERO.
    barrier_W3 = float(-mb4_2d(mins["W3"].unsqueeze(0)).item())
    kramers = 2.0 * math.pi * math.exp(min(cfg.beta * barrier_W3, 700.0))

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


def _phi4_sampling_box_design(means: torch.Tensor, atoms: torch.Tensor,
                               h: float, hessians: torch.Tensor, *,
                               beta: float, pt_beta_min: float,
                               tail_probability: float = 1e-8,
                               jitter_sigma: float = 0.0) -> dict:
    """Return a conservative high-probability numerical box for E4.

    The target is unbounded, so no finite box is an exact support bound.  Use a
    simultaneous Laplace-mixture component envelope with declared union-bound
    tail budget, pad the target-temperature envelope by one maximum
    *componentwise* displacement from the shell law, also cover the hottest PT
    envelope, and round the half-width upward.  The default gives +/-5.
    """
    if (means.ndim != 2 or atoms.ndim != 2 or hessians.ndim != 3
            or means.shape[1] != atoms.shape[1]
            or hessians.shape[1:] != (means.shape[1], means.shape[1])):
        raise ValueError("incompatible phi4 means/atoms/Hessians")
    if not (0.0 < tail_probability < 1.0):
        raise ValueError("tail_probability must lie strictly between zero and one")
    if beta <= 0.0 or pt_beta_min <= 0.0:
        raise ValueError("beta and pt_beta_min must be positive")
    atom_norms = atoms.norm(dim=1, keepdim=True)
    if bool((atom_norms <= 0).any().item()):
        raise ValueError("phi4 jump atoms must be nonzero")
    units = atoms / atom_norms
    max_componentwise_jump_reach = float(
        (atoms.abs() + float(h) * units.abs()).amax().item())

    n_modes, dimension = means.shape
    tail_quantile_probability = 1.0 - tail_probability / (
        2.0 * n_modes * dimension)
    normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=means.dtype, device=means.device),
        torch.tensor(1.0, dtype=means.dtype, device=means.device))
    normal_quantile = float(normal.icdf(torch.tensor(
        tail_quantile_probability, dtype=means.dtype,
        device=means.device)).item())
    inverse_hessians = torch.linalg.inv(hessians)
    max_component_std_target = float(torch.sqrt(
        torch.diagonal(inverse_hessians / float(beta), dim1=-2, dim2=-1)
    ).amax().item())
    max_component_std_hottest_pt = float(torch.sqrt(
        torch.diagonal(inverse_hessians / float(pt_beta_min), dim1=-2, dim2=-1)
    ).amax().item())
    phase_component_extent = float(means.abs().amax().item())
    target_phase_envelope = (
        phase_component_extent + normal_quantile * max_component_std_target)
    hottest_pt_envelope = (
        phase_component_extent + normal_quantile * max_component_std_hottest_pt)
    one_jump_target_requirement = (
        target_phase_envelope + max_componentwise_jump_reach)
    required_half_width = max(one_jump_target_requirement, hottest_pt_envelope)
    sampling_half_width = float(math.ceil(required_half_width))
    if sampling_half_width < required_half_width:
        raise AssertionError("rounded phi4 sampling box is not conservative")
    return {
        "formula": (
            "ceil(max(B_beta(alpha)+R_infinity, B_beta_min(alpha))); "
            "B_b(alpha)=max|mu|+Phi^{-1}(1-alpha/(2Kd))*"
            "max sqrt(diag(H_k^{-1})/b); "
            "R_infinity=max(|r_ac|+h_a|u_ac|)"),
        "tail_probability_union_bound": float(tail_probability),
        "normal_quantile": normal_quantile,
        "n_phase_modes": int(n_modes),
        "dimension": int(dimension),
        "phase_component_extent": phase_component_extent,
        "max_component_std_beta": max_component_std_target,
        "max_component_std_beta_min": max_component_std_hottest_pt,
        "beta": float(beta),
        "pt_beta_min": float(pt_beta_min),
        "target_phase_envelope_half_width": target_phase_envelope,
        "hottest_pt_envelope_half_width": hottest_pt_envelope,
        "max_componentwise_jump_reach": max_componentwise_jump_reach,
        "one_jump_target_required_half_width": one_jump_target_requirement,
        "required_half_width_before_rounding": required_half_width,
        "sampling_box_half_width": sampling_half_width,
        "jump_safe_core_half_width": (
            sampling_half_width - max_componentwise_jump_reach),
        "jitter_sigma": float(jitter_sigma),
        "guaranteed_default_shell_jumps_from_target_envelope": (
            float(jitter_sigma) == 0.0),
        "jitter_caveat": (
            "Gaussian jitter has unbounded support; the finite box guarantee "
            "applies only to the default sigma=0 shell law"
            if float(jitter_sigma) > 0.0 else None),
    }


# ===================================================================== E4
def build_e4(device="cuda", basin_cache: str | None = None,
             jitter_sigma: float = 0.0, *,
             basin_n_grid: int = 800,
             basin_flow_steps: int = 40_000,
             basin_flow_dt: float = 1.5e-4,
             snis_proposals: int = 200_000) -> Experiment:
    pot = CoupledPhi4()
    cfg = RunConfig(name="coupled_phi4", d=24, n_particles=1000, T=100.0,
                    dt=0.002, seeds=tuple(range(16)))

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
    # Optional isotropic jitter, supported by sampled-bank RA/MA scores because
    # they use the realised displacements directly. Off by default; the
    # deterministic exact-quadrature score/certificate cannot use this law.
    if jitter_sigma > 0.0:
        from .jumps import JitteredShellJumpLaw
        law = JitteredShellJumpLaw(atoms, weights, h, jitter_sigma)
    else:
        law = ShellJumpLaw(atoms, weights, h=h)
    # drift cap = 2h (one shell width), same rule as E3. An earlier design used
    # cap = max||r_a|| to let raw-CP retrace full jumps (raw-CP pi-start return
    # flow prefers it), but the Levy score is astronomically large here (beta=8
    # over 24 coupled dims), so under tame(b,dt,cap) the score step saturates at
    # length ~cap toward the deepest well and OVERSHOOTS: at cap=||r||=6.93,
    # LSC-CP over-concentrated the "--" phase to occ=0.50 (target 0.325) and lost
    # to raw-CP on every metric. raw-CP is cap-insensitive on the production
    # metrics (W2/MMD/EMC/basin identical at 2h vs ||r||), so tightening the cap
    # costs raw-CP nothing and fixes the score overshoot: at cap=2h LSC-CP lands
    # occ(--)=0.324 and beats raw-CP W2 0.11 vs 0.27, basin 0.08 vs 0.22. Both
    # exact and multi-atom estimators agree at this cap (it is a deterministic
    # taming-saturation effect, not estimator variance).
    drift_cap_e4 = 2.0 * float(h)

    # Laplace reference: 24x24 Hessians at the coherent minima (autograd)
    from torch.autograd.functional import hessian as _th_hessian
    Hs = []
    for k in range(4):
        Hk = _th_hessian(lambda q: pot._V_raw(q.unsqueeze(0))[0],
                         means24[k].clone())
        Hs.append(0.5 * (Hk + Hk.T))
    H24 = torch.stack(Hs)
    energies = phi4_W(V2)                                    # V(1 (x) v) = W(v)
    laplace = LaplaceMixture(means24, H24, energies, cfg.beta)

    pt_beta_min_e4 = 1.0
    box_design = _phi4_sampling_box_design(
        means24, atoms, h, H24, beta=cfg.beta,
        pt_beta_min=pt_beta_min_e4, jitter_sigma=jitter_sigma)
    # The target/diffusion are unbounded: this is a declared high-probability
    # numerical overflow guard, not a truncation of the scientific model.
    box_half_width = box_design["sampling_box_half_width"]
    box = RectBox([-box_half_width] * 24, [box_half_width] * 24, device)
    target_envelope_box = RectBox(
        [-box_design["target_phase_envelope_half_width"]] * 24,
        [box_design["target_phase_envelope_half_width"]] * 24, device)
    jump_safe_core_box = RectBox(
        [-box_design["jump_safe_core_half_width"]] * 24,
        [box_design["jump_safe_core_half_width"]] * 24, device)

    # Basin-map domain must cover the JUMP-REACHABLE order-parameter set, not
    # just the phase minima: a coherent phase-to-phase atom moves qbar by
    # ~2.12, so a jump from a displaced/thermal-tail state parks qbar out to
    # ~3.3 (measured max 3.26 at production dynamics, 2026-07-17 probe).
    # assign() clamps out-of-domain points into boundary cells, which
    # mislabels exactly the LSC-CP-MA transport the study measures; +-4 with
    # unchanged cell size covers double-jump reach with margin.
    basins = GradientFlowBasinMap2D(
        phi4_W_grad, V2, (-4.0, -4.0), (4.0, 4.0),
        n_grid=basin_n_grid, device=device, cache=basin_cache,
        dt_flow=basin_flow_dt, n_flow=basin_flow_steps)

    def qbar(x):
        return x.reshape(-1, pot.Ns, 2).mean(dim=1)          # (N, 2)

    def labels_fn(x):
        return basins.assign(qbar(x))

    # High-accuracy importance reference. Unweighted samples needed by W2/MMD
    # use finite SIR and are explicitly approximate; basin masses and scalar
    # expectations use the lower-variance direct SNIS weights.
    sir_oversample = 16

    def ref_sample(n, gen):
        return laplace.sample_sir(
            n, gen, pot, cfg.beta, oversample=sir_oversample)

    g_p = torch.Generator(device=device)
    g_p.manual_seed(31337)
    with pot.no_count():
        p_x, p_w, reference_diagnostics = laplace.snis_weighted_proposals(
            snis_proposals, g_p, pot, cfg.beta)
        p_metric = qbar(p_x)
        basin_lo = basins.lo.to(dtype=p_metric.dtype, device=device)
        basin_hi = basins.hi.to(dtype=p_metric.dtype, device=device)
        basin_inside = ((p_metric >= basin_lo) & (p_metric <= basin_hi)).all(dim=1)
        reference_diagnostics["weighted_basin_map_outside_mass"] = float(
            p_w[~basin_inside].sum().item())
        p_star = laplace.weighted_category_probabilities(labels_fn(p_x), 4, p_w)
        reference_diagnostics["weighted_outside_target_phase_envelope_mass"] = float(
            p_w[~target_envelope_box.contains(p_x)].sum().item())
        reference_diagnostics["weighted_outside_jump_safe_core_mass"] = float(
            p_w[~jump_safe_core_box.contains(p_x)].sum().item())
        reference_diagnostics["weighted_outside_sampling_box_mass"] = float(
            p_w[~box.contains(p_x)].sum().item())
        p_cv_mean_t = laplace.weighted_expectation(p_metric, p_w)
        p_energy = pot.V(p_x)
        p_energy_mean_t = laplace.weighted_expectation(p_energy, p_w)
        p_energy_var_t = laplace.weighted_expectation(
            (p_energy - p_energy_mean_t).square(), p_w)

    def init_fn(n, gen):
        return means24[0] + 0.05 * torch.randn(n, 24, generator=gen,
                                               device=device, dtype=torch.float64)

    def make_score(q_theta=Q_THETA, q_rho=Q_RHO):
        return ShellScore(pot, law, cfg.lam, cfg.beta, q_theta, q_rho)

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
        * math.exp(cfg.beta * barrier)

    return Experiment(
        name="coupled_phi4", cfg=cfg, pot=pot, law=law, box=box,
        init_fn=init_fn, ref_sample=ref_sample,
        make_score=make_score, labels_fn=labels_fn, p_star=p_star,
        metric_space=qbar,
        pt_beta_min=pt_beta_min_e4,
        # committed exit from --: mean order parameter within 0.35 of another
        # coherent minimum
        exit_committed=lambda x: (
            (qbar(x).unsqueeze(1) - V2[1:].unsqueeze(0)).norm(dim=2).min(dim=1)
            .values < 0.35),
        kramers_tau=kramers,
        cp_drift_cap=drift_cap_e4,
        extras={"minima_2d": V2, "means24": means24, "hessians": H24,
                "laplace": laplace, "basins": basins, "h": h,
                "basin_cache_provenance": basins.cache_provenance(),
                "barrier_minus_minus": barrier, "phases": phases,
                "edge_pairs": edge_pairs, "jitter_sigma": jitter_sigma,
                "sampling_box_design": box_design,
                # Derived from the basin map itself so the outside-mass gate
                # can never measure against a different domain than assign().
                "basin_map_metric_bounds": [basins.lo.tolist(),
                                            basins.hi.tolist()],
                "reference_diagnostics": reference_diagnostics,
                "reference_sample_method": "sampling_importance_resampling",
                "reference_scalar_method": "direct_snis",
                "builder_reference_parameters": {
                    "basin_bounds": [[-4.0, -4.0], [4.0, 4.0]],
                    "basin_n_grid": int(basin_n_grid),
                    "basin_flow_steps": int(basin_flow_steps),
                    "basin_flow_dt": float(basin_flow_dt),
                    "direct_snis_proposals": int(snis_proposals),
                    "direct_snis_seed": 31337,
                    "unweighted_sir_oversample": sir_oversample,
                    "sampling_box_design": box_design,
                },
                # Keep only the low-dimensional weighted cloud needed for a
                # direct-SNIS FES; retaining all 200k x 24 proposals would waste
                # GPU memory after the high-dimensional observables are reduced.
                "reference_metric_points": p_metric.detach(),
                "reference_metric_weights": p_w.detach(),
                # One-dimensional energy values are cheap to retain and let
                # energy-overlap use direct SNIS rather than finite SIR.
                "reference_energy_values": p_energy.detach(),
                "reference_energy_weights": p_w.detach(),
                "reference_cv_means": p_cv_mean_t.detach().cpu().tolist(),
                "reference_energy_mean": float(p_energy_mean_t.item()),
                "reference_energy_var": float(p_energy_var_t.item())},
    )


def _mog40_committed_exit(pot):
    def fn(x):
        d = torch.cdist(x, pot.mu)                 # (N, 40)
        lab = d.argmin(dim=1)
        return (lab != 0) & (d.min(dim=1).values < 2.0)
    return fn


# ===================================================================== E5
def _torus_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise torus distance between (N,2) and (K,2) angle pairs -> (N,K)."""
    d = (a.unsqueeze(-2) - b).abs()
    d = torch.minimum(d, 2.0 * math.pi - d)
    return torch.sqrt((d * d).sum(-1))


def build_e5_alanine(device="cuda", *, reference_cache: str | None = None,
                     n_particles: int = 1000, T: float = 40.0,
                     dt: float = 1.0e-3, seeds=tuple(range(16)),
                     jump_frac_tol: float = 1e-3, jump_n_states: int = 128,
                     jump_q_theta: int = Q_THETA,
                     island_core_rad: float = 0.4) -> Experiment:
    """E5: alanine dipeptide (Ac-Ala-NHMe, vacuum, 22 atoms), a real force field.

    The sampler runs in whitened BAT internal coordinates q_tilde (d = 60), whose
    torsion slots include phi and psi directly, so the jump atoms are pure-torsion
    rotations between Ramachandran basins and the Levy-score chord is Jacobian
    free (S1.6, machine-zero in P3).  The target is the Cartesian Boltzmann
    measure at 300 K: mapping q_tilde -> q -> x pushes mu_q ∝ e^{-beta U_eff}
    forward to mu_x ∝ e^{-beta U}, the same measure the OpenMM metadynamics
    reference samples, so F(phi,psi) is an apples-to-apples comparison (S1.7).

    beta is threaded locally (config.BETA stays 8 for E1/E2/E4); here
    beta = 1/(kB T) at 300 K = 0.40091 mol/kJ (kT = 2.4943 kJ/mol).

    The reference (FES, basins, p_star, reweighted conformer pool) is the cached
    well-tempered metadynamics run; regenerate with
    ``python -m src.e5_alanine.build_reference --seeds 0 1``.
    """
    from .e5_alanine.potential import AlanineDipeptideBAT, E5_TEMPERATURE, e5_beta
    from .e5_alanine.reference import E5Reference
    from .e5_alanine.box import TorusBox
    from .e5_alanine.jump_design import design_jump_law

    E5_BETA = e5_beta(E5_TEMPERATURE)      # <-- E5 temperature (300 K)
    pot = AlanineDipeptideBAT(beta=E5_BETA, device=device)
    cfg = RunConfig(name="alanine_dipeptide", d=pot.d, n_particles=n_particles,
                    T=T, dt=dt, beta=E5_BETA, seeds=tuple(seeds))

    ref = E5Reference(reference_cache, device=device)
    law, jump_record = design_jump_law(
        pot, ref, n_states=jump_n_states, q_theta=jump_q_theta,
        frac_tol=jump_frac_tol)
    drift_cap = float(jump_record["cp_drift_cap"])
    box = TorusBox(pot)
    p_star = ref.p_star

    # ---- init in C7eq (the global FES minimum) ----------------------------
    init_basin = ref.deepest_basin()
    q0 = ref.representative_state(init_basin)

    def init_fn(n, gen):
        return q0.unsqueeze(0) + 0.05 * torch.randn(
            n, pot.d, generator=gen, device=device, dtype=torch.float64)

    def metric_space(qt):
        return pot.to_cv(qt)

    def labels_fn(qt):
        return ref.assign(pot.to_cv(qt))

    def make_score(q_theta=Q_THETA, q_rho=Q_RHO):
        # Exact deterministic quadrature. Correct but costly here
        # (q_theta * A * q_rho BAT reconstructions per particle per step); the
        # deployed estimators are RA / paired-MA, which need only chord energies.
        return ShellScore(pot, law, cfg.lam, cfg.beta, q_theta, q_rho)

    # ---- the slow event: reaching the sparse C7ax/alpha_L island ----------
    # The barrier is the island basin's own escape barrier, NOT the minimum of F
    # along a phi cut: the phi = +-pi line passes through a basin, so the value
    # there is a basin depth rather than a saddle.
    b_phi_zero = ref.phi_cut_min_kJ(0.0)             # dividing line at phi ~ 0
    b_phi_seam = ref.phi_cut_min_kJ(math.pi)         # NOT a dividing line here
    b_phi = max(ref.island_barrier_kJ(), 1e-6)
    kramers = 2.0 * math.pi * math.exp(min(E5_BETA * b_phi, 700.0))
    pos_basins = ref.island_basins()
    pos_minima = ref.minima[pos_basins] if pos_basins else ref.minima[:0]

    def exit_committed(qt):
        cv = pot.to_cv(qt)
        if pos_minima.shape[0] == 0:
            return torch.zeros(cv.shape[0], dtype=torch.bool, device=cv.device)
        return _torus_dist(cv, pos_minima).min(dim=1).values < island_core_rad

    # ---- weighted (direct-SNIS style) reference quantities ----------------
    w = ref.weights
    cv_mean = (w.unsqueeze(1) * ref.cvs).sum(0)
    e_mean = float((w * ref.U_eff).sum().item())
    e_var = float((w * (ref.U_eff - e_mean) ** 2).sum().item())

    return Experiment(
        name="alanine_dipeptide", cfg=cfg, pot=pot, law=law, box=box,
        init_fn=init_fn, ref_sample=lambda n, g: ref.sample(n, g),
        make_score=make_score, labels_fn=labels_fn, p_star=p_star,
        metric_space=metric_space,
        # hot replica must clear the phi barrier: beta_min * b ~ 2
        pt_beta_min=2.0 / b_phi,
        exit_committed=exit_committed,
        kramers_tau=kramers,
        cp_drift_cap=drift_cap,
        extras={
            "reference": ref, "jump_design": jump_record,
            "whitening_provenance": pot.whitening_provenance,
            "temperature_K": E5_TEMPERATURE, "beta": E5_BETA,
            "kT_kJ_per_mol": 1.0 / E5_BETA,
            "phi_cut_min_kJ_at_zero": b_phi_zero,
            "phi_cut_min_kJ_at_seam": b_phi_seam,
            "island_barrier_kJ": b_phi, "island_barrier_kT": E5_BETA * b_phi,
            "basin_escape_kT": ref.basin_escape_kT.cpu().tolist(),
            "init_basin": int(init_basin),
            "island_basins": [int(k) for k in pos_basins],
            "positive_phi_basins": [int(k) for k in pos_basins],
            "island_core_rad": island_core_rad,
            "minima_deg": np.degrees(ref.minima.cpu().numpy()).round(1).tolist(),
            "basin_free_energies_kT": ref.basin_free_energies_kT().cpu().tolist(),
            "reference_ess": ref.ess, "reference_ess_fraction": ref.ess_fraction,
            "reference_seam_mass": ref.seam_mass(),
            "reference_provenance": ref.provenance,
            "h": float(jump_record["h"]),
            # E4-style weighted reference: FES/energy use the importance weights
            # directly; the unweighted cloud W2/MMD need comes from SIR draws.
            "reference_metric_points": ref.cvs.detach(),
            "reference_metric_weights": w.detach(),
            "reference_energy_values": ref.U_eff.detach(),
            "reference_energy_weights": w.detach(),
            "reference_cv_means": cv_mean.detach().cpu().tolist(),
            "reference_energy_mean": e_mean, "reference_energy_var": e_var,
            "reference_sample_method": "wt_metadynamics_reweighted_sir",
            "reference_scalar_method": "direct_snis",
            "builder_reference_parameters": {
                "reference_cache": ref.cache_path,
                "jump_frac_tol": jump_frac_tol,
                "jump_n_states": jump_n_states,
                "jump_q_theta": jump_q_theta,
            },
            # NOTE: no "basin_map_metric_bounds" -- the partition is a torus
            # Voronoi over the whole (-pi, pi]^2 fundamental domain, so unlike
            # E4's grid basin map there is no out-of-domain clamping to guard.
        },
    )


# ============================================================ sampler wiring
def make_sampler_factory(exp: Experiment, dt: float, pt_betas: torch.Tensor,
                         n_particles: int | None = None,
                         score_kwargs: dict | None = None,
                         reference_init: bool = False):
    """Factory ``(method, seed) -> fresh sampler``.

    Initial positions are shared across methods for each seed.  Production
    relaxation runs use ``exp.init_fn``; stationary-trace runs set
    ``reference_init=True`` and start from the experiment's reference sampler.
    CP and exact LSC-CP share the jump stream pathwise.
    """
    dev = exp.p_star.device
    N = n_particles or exp.cfg.n_particles
    eps = exp.cfg.eps
    beta = exp.cfg.beta
    lam = exp.cfg.lam
    score_kwargs = score_kwargs or {}

    def factory(method: str, seed: int):
        g_init = torch.Generator(device=dev)
        g_init.manual_seed(init_seed(seed))
        init = exp.ref_sample if reference_init else exp.init_fn
        x0 = init(N, g_init)
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
        if method == "LSC-CP-MA":                             # paired multi-atom RA
            g_jump = torch.Generator(device=dev)
            g_jump.manual_seed(jump_seed(seed))
            q_theta = score_kwargs.get("q_theta", Q_THETA)
            score = MultiAtomShellScore(exp.pot, exp.law, lam, beta, q_theta)
            return CompoundPoisson(exp.pot, x0, dt, eps, lam, exp.law,
                                   gen, g_jump, exp.box, score=score,
                                   name=method, drift_cap=exp.cp_drift_cap,
                                   jump_mode="paired_multiatom")
        raise ValueError(method)

    return factory


def make_batched_factory(exp: Experiment, dt: float, pt_betas: torch.Tensor,
                         seeds, n_particles: int | None = None,
                         score_kwargs: dict | None = None,
                         reference_init: bool = False):
    """All seeds batched into one (S*N)-particle ensemble per method (the
    wall-clock axis is no longer reported, so sequential per-seed timing is
    unnecessary and the GPU is far better utilised). Per-seed x0 blocks are
    generated exactly as in the sequential path; setting ``reference_init``
    starts every block from an independently seeded reference draw. CP and
    exact LSC-CP still share one jump stream."""
    dev = exp.p_star.device
    N = n_particles or exp.cfg.n_particles
    eps, beta, lam = exp.cfg.eps, exp.cfg.beta, exp.cfg.lam
    score_kwargs = score_kwargs or {}

    def factory(method: str):
        blocks = []
        for seed in seeds:
            g = torch.Generator(device=dev)
            g.manual_seed(init_seed(seed))
            init = exp.ref_sample if reference_init else exp.init_fn
            blocks.append(init(N, g))
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
        if method == "LSC-CP-MA":                             # paired multi-atom RA
            g_jump = torch.Generator(device=dev)
            g_jump.manual_seed(jump_seed(0))
            q_theta = score_kwargs.get("q_theta", Q_THETA)
            score = MultiAtomShellScore(exp.pot, exp.law, lam, beta, q_theta)
            return CompoundPoisson(exp.pot, x0, dt, eps, lam, exp.law,
                                   gen, g_jump, exp.box, score=score,
                                   name=method, drift_cap=exp.cp_drift_cap,
                                   jump_mode="paired_multiatom")
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
    # Use direct weighted SNIS for E4 scalar expectations when available;
    # finite SIR is retained only where an unweighted cloud is mathematically
    # required (W2/MMD/KDE). Energy and FES histograms use direct weights.
    ref_mean_V = float(exp.extras.get(
        "reference_energy_mean", float(ref_V.mean().item())))
    ref_var_V = float(exp.extras.get(
        "reference_energy_var", float(ref_V.var(unbiased=True).item())))
    # Reduced free-energy surface in kBT units. Use the full physically relevant
    # metric space for 1D and the first two coordinates for multidimensional
    # systems (MB active plane and phi4 mean order parameter). A larger frozen
    # reference suppresses histogram noise; probability outside the frozen grid
    # is reported separately rather than silently renormalized away.
    fes_dim = min(2, d_m)
    # Choose the 2D resolution from the empirical sample size, not the much
    # larger reference size: otherwise a 40x40 grid with N=1000 would put as
    # much Jeffreys prior mass as data into the FES.  The rule targets roughly
    # ten empirical observations per cell on average; 1D retains 40 bins.
    fes_bins_per_dim = (40 if fes_dim == 1 else
                        max(8, min(25, int(math.sqrt(n / 10.0)))))
    weighted_fes_points = exp.extras.get("reference_metric_points")
    weighted_fes_weights = exp.extras.get("reference_metric_weights")
    if weighted_fes_points is not None:
        if weighted_fes_weights is None:
            raise ValueError("weighted FES points require matching weights")
        ref_fes_m = weighted_fes_points[:, :fes_dim]
        ref_fes_weights = weighted_fes_weights
        fes_ref_n = int(ref_fes_m.shape[0])
        fes_reference_method = "direct_snis_weighted_histogram"
    else:
        fes_ref_n = max(n, 50_000)
        g_fes = torch.Generator(device=device)
        g_fes.manual_seed(ref_seed + 101)
        ref_fes_x = exp.ref_sample(fes_ref_n, g_fes)
        ref_fes_m = exp.metric_space(ref_fes_x)[:, :fes_dim]
        ref_fes_weights = None
        fes_reference_method = exp.extras.get(
            "reference_sample_method", "direct_reference_sampler")
    fes_edges = []
    for j in range(fes_dim):
        lo_j = float(ref_fes_m[:, j].min().item())
        hi_j = float(ref_fes_m[:, j].max().item())
        pad_j = 0.05 * (hi_j - lo_j + 1e-9)
        fes_edges.append(torch.linspace(
            lo_j - pad_j, hi_j + pad_j, fes_bins_per_dim + 1,
            dtype=torch.float64, device=device))
    fes_edges = tuple(fes_edges)
    # Flattened cells plus one off-grid cell, so the FES comparison conserves
    # mass. Without it a leaking sampler is scored only where it did not leak.
    ref_fes_p, ref_fes_outside = M.binned_probabilities_with_outside(
        ref_fes_m, fes_edges, smooth=0.5,
        sample_weights=ref_fes_weights)
    # The off-grid cell is exempt from the pi_min cut: the reference puts
    # essentially no mass there (the grid is built from the reference's own
    # range), but that is precisely the cell that exposes a leaking sampler.
    fes_always_keep = torch.zeros_like(ref_fes_p, dtype=torch.bool)
    fes_always_keep[-1] = True
    # One expected observation in the empirical ensemble, not five: at 5/n the
    # barrier cells are deleted outright, which is where the corrected and
    # uncorrected dynamics differ most.
    fes_pi_min = 1.0 / n
    # First-coordinate references remain for collaborator-parity CDF/PDF metrics.
    ref_cv = ref_m[:, 0]
    cv_lo, cv_hi = float(ref_cv.min().item()), float(ref_cv.max().item())
    cv_pad = 0.05 * (cv_hi - cv_lo + 1e-9)
    weighted_energy_values = exp.extras.get("reference_energy_values")
    weighted_energy_weights = exp.extras.get("reference_energy_weights")
    if weighted_energy_values is not None:
        if weighted_energy_weights is None:
            raise ValueError("weighted energy values require matching weights")
        energy_reference_values = weighted_energy_values.reshape(-1)
        energy_reference_weights = weighted_energy_weights.reshape(-1).to(
            device=energy_reference_values.device, dtype=torch.float64)
        if (energy_reference_values.numel() != energy_reference_weights.numel()
                or not bool(torch.isfinite(energy_reference_values).all().item())
                or not bool(torch.isfinite(energy_reference_weights).all().item())
                or bool((energy_reference_weights < 0).any().item())
                or float(energy_reference_weights.sum().item()) <= 0.0):
            raise ValueError("invalid direct-SNIS energy reference")
        energy_reference_method = "direct_snis_weighted_histogram"
    else:
        energy_reference_values = ref_V.reshape(-1)
        energy_reference_weights = None
        energy_reference_method = exp.extras.get(
            "reference_sample_method", "direct_reference_sampler")
    e_lo = float(energy_reference_values.min().item())
    e_hi = float(energy_reference_values.max().item())
    e_pad = 0.05 * (e_hi - e_lo + 1e-9)
    e_edges = torch.linspace(e_lo - e_pad, e_hi + e_pad, 41,
                             dtype=torch.float64, device=device)
    _ei = torch.bucketize(energy_reference_values, e_edges[1:-1])
    if energy_reference_weights is None:
        ref_ehist = torch.bincount(
            _ei, minlength=e_edges.shape[0] - 1).to(torch.float64)
    else:
        ref_ehist = torch.zeros(
            e_edges.shape[0] - 1, dtype=torch.float64, device=device)
        ref_ehist.scatter_add_(0, _ei, energy_reference_weights)
    ref_ehist = ref_ehist / ref_ehist.sum()

    # ---- 1D density/CDF target along the CV (collaborator-parity metrics) ---
    # target from the frozen reference sample: empirical CDF + matched-bandwidth
    # KDE (both empirical and target KDE'd identically, so the smoothing bias
    # largely cancels). Grid extends into the tails to catch nonlocal bias mass.
    dens_grid = torch.linspace(cv_lo - 4.0 * cv_pad, cv_hi + 4.0 * cv_pad, 512,
                               dtype=torch.float64, device=device)
    _rcs = torch.sort(ref_cv).values
    target_cdf_g = torch.searchsorted(_rcs, dens_grid, right=True).to(torch.float64) / ref_cv.numel()
    dens_bw = max(float(ref_cv.std().item()) * n ** (-0.2), 1e-3)   # Silverman-ish
    target_pdf_g = M.kde_on_grid(ref_cv, dens_grid, dens_bw)
    chi_mask = (target_cdf_g >= 0.01) & (target_cdf_g <= 0.99)

    is_e1 = exp.name == "double_well"
    if is_e1:
        lo, hi = exp.extras["density_tv_box"]
        nb = exp.extras["density_tv_bins"]
        edges = torch.linspace(lo, hi, nb + 1, dtype=torch.float64, device=device)
        centers = 0.5 * (edges[1:] + edges[:-1])
        logp = -exp.cfg.beta * (centers * centers - 1.0) ** 2
        mass = torch.exp(logp - logp.max())
        target_mass = mass / mass.sum()

    proj10 = None
    if exp.cfg.d == 10:      # latent-metric experiments also report full-10D W2
        proj10 = M.make_projections(10, 200, seed=778, device=device)

    def w2_fn(a, b):
        return M.w2_exact_1d(a, b) if d_m == 1 else M.sliced_w2(a, b, proj)

    uniform = exp.uniform_target
    basin_map_metric_bounds = exp.extras.get("basin_map_metric_bounds")
    if basin_map_metric_bounds is not None:
        basin_map_lo = torch.as_tensor(
            basin_map_metric_bounds[0], dtype=torch.float64, device=device)
        basin_map_hi = torch.as_tensor(
            basin_map_metric_bounds[1], dtype=torch.float64, device=device)

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
            "nonfinite_count": M.nonfinite_count(x),
            "nonfinite_frac": M.nonfinite_frac(x),
        }
        if basin_map_metric_bounds is not None:
            basin_coords = xm[:, :basin_map_lo.numel()]
            basin_inside = ((basin_coords >= basin_map_lo)
                            & (basin_coords <= basin_map_hi)).all(dim=1)
            out["basin_map_outside_mass"] = float(
                (~basin_inside).to(torch.float64).mean().item())
        if is_e1:
            out["TV_density"] = M.density_tv_1d(x, edges, target_mass)
        if proj10 is not None:
            out["W2_10d"] = M.sliced_w2(x, ref_x, proj10)
        # ---- chemistry-native + KSD (potential evals excluded from NFE) ----
        cv = xm[:, 0]
        fes_x = xm[:, :fes_dim]
        fes_p, fes_outside = M.binned_probabilities_with_outside(
            fes_x, fes_edges, smooth=0.5)
        # Uniform weights: reference weighting concentrated ~85% of the score on
        # the ten densest cells, so the metric measured basin-bottom shape only
        # and was blind to barrier and tail placement.
        fes_rmse = M.free_energy_rmse_from_probabilities(
            fes_p, ref_fes_p, pi_min=fes_pi_min, weights="uniform",
            always_keep=fes_always_keep)
        out["FES_RMSE_kBT"] = fes_rmse
        out["e_F"] = fes_rmse  # backwards-compatible alias; now RMSE in kBT
        out["FES_outside_mass"] = fes_outside
        out["basin_KL_target"] = M.basin_kl_target_to_empirical(
            p_hat, exp.p_star, pseudocount=0.5 / x.shape[0])
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
        # 1D density/CDF metrics along the CV (collaborator parity + pdf/cdf L1/L2)
        out.update(M.density_cdf_metrics(cv, dens_grid, target_pdf_g, target_cdf_g,
                                         dens_bw, chi_mask))
        for MM in (40, 80, 120):
            out[f"bin_chi2_M{MM}"] = M.bin_chi2_pit(cv, dens_grid, target_cdf_g, MM)
        out["well_TV"] = M.well_tv(p_hat, exp.p_star)
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
    return metrics_fn, floors, {
        "bandwidth": bw, "ref_x": ref_x, "fes_reference_n": fes_ref_n,
        "fes_dim": fes_dim, "fes_bins_per_dim": fes_bins_per_dim,
        "fes_pi_min": fes_pi_min, "fes_min_expected_count": 1.0,
        "fes_pseudocount_per_bin": 0.5, "fes_weighting": "uniform",
        "fes_offgrid_cell": True,
        "fes_reference_outside_mass": ref_fes_outside,
        "fes_reference_method": fes_reference_method,
        "basin_kl_orientation": "target_to_empirical",
        "basin_kl_pseudocount_per_basin": 0.5,
        "sample_reference_method": exp.extras.get(
            "reference_sample_method", "direct_reference_sampler"),
        "scalar_reference_method": exp.extras.get(
            "reference_scalar_method", "unweighted_reference_sample"),
        "reference_diagnostics": exp.extras.get("reference_diagnostics"),
        "reference_energy_mean": ref_mean_V,
        "reference_energy_var": ref_var_V,
        "energy_reference_method": energy_reference_method,
        "reference_cv_means": exp.extras.get(
            "reference_cv_means", ref_m.mean(dim=0).cpu().tolist()),
    }


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
        logp = -exp.cfg.beta * (centers * centers - 1.0) ** 2
        mass = torch.exp(logp - logp.max())
        target_mass = mass / mass.sum()
        out["TV_density"] = lambda x: M.density_tv_1d(x, edges, target_mass)
    return out
