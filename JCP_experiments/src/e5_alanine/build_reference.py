"""P4: well-tempered metadynamics reference FES(phi, psi) + p_star.

Ground-truth reference for E5.  We run WT-metadynamics on the SAME flexible
Cartesian system the torch model represents (constraints=None), with two
CustomTorsionForce CVs (phi, psi), Langevin at 300 K.  The metadynamics bias
grows during equilibration; then, with the converged bias FROZEN, a production
phase samples the biased ensemble and each frame is reweighted to the unbiased
Boltzmann measure by

    w(s) = exp(+beta V_bias(s)) = exp(-beta ((gamma-1)/gamma) F(s)),

where gamma is the bias factor and F = metad.getFreeEnergy().  The reweighted
Cartesian frames, mapped to whitened internal coordinates q_tilde, are the
reference conformer pool; their (phi, psi) marginal reproduces e^{-beta F}.

This is SETUP/VALIDATION only (OpenMM never runs in the sampler/score hot loop).
Run:  python -m src.e5_alanine.build_reference --seeds 0 1  [--equil-ns .. --prod-ns ..]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "cache", "e5_alanine")

# phi/psi atom quartets (verified in P0)
PHI_ATOMS = (4, 6, 8, 14)
PSI_ATOMS = (6, 8, 14, 16)


def _cv_force(atoms):
    import openmm
    f = openmm.CustomTorsionForce("theta")
    f.addTorsion(*atoms, [])
    return f


def _free_energy_phi_psi(metad, unit) -> np.ndarray:
    """metad.getFreeEnergy() indexed as F[phi, psi] (kJ/mol).

    OpenMM allocates the metadynamics bias grid with ``reversed(variables)``
    (openmm/app/metadynamics.py: ``np.zeros(tuple(v.gridWidth for v in
    reversed(variables)))``), so the returned array is indexed [psi, phi].  Both
    grid widths are equal here, so the transpose is SILENT -- it must be applied
    explicitly or the reference FES comes out mirrored through the origin.
    """
    F = np.asarray(
        metad.getFreeEnergy().value_in_unit(unit.kilojoule_per_mole),
        dtype=np.float64)
    return F.T.copy()


def run_wt_metad(seed: int, *, equil_ns: float = 8.0, prod_ns: float = 10.0,
                 dt_fs: float = 2.0, temperature: float = 300.0,
                 bias_factor: float = 8.0, hill_height_kJ: float = 1.2,
                 bias_width_rad: float = 0.35, grid: int = 100,
                 deposit_every: int = 1000, frame_every_ps: float = 1.0,
                 record_every_ns: float = 0.5, device_index: str = "0",
                 platform_name: str = "CPU", precision: str = "mixed") -> dict:
    """One WT-metadynamics run.

    Performance note: this 22-atom system is dominated by per-step and
    per-deposition overhead, not by force evaluation.  Measured rates --
    OpenCL ~300 steps/s; CPU with 16 threads ~930; CPU SINGLE-THREADED ~2805
    (20 ns/hour), and hill deposition every 100 steps costs 30x more than every
    1000 steps (2 ps, a standard metadynamics interval).  So the default is the
    CPU platform with ``OPENMM_CPU_THREADS=1`` and ``deposit_every=1000``, and
    separate seeds are run as parallel single-threaded processes.

    Returns the FES, the equilibration convergence snapshots, and the production
    frame pool.
    """
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    from openmm.app.metadynamics import Metadynamics, BiasVariable
    from .system import build_alanine_system

    ala = build_alanine_system()
    system = ala.system
    kT = (unit.MOLAR_GAS_CONSTANT_R * temperature * unit.kelvin).value_in_unit(
        unit.kilojoule_per_mole)
    beta = 1.0 / kT
    pi = np.pi

    bv_phi = BiasVariable(_cv_force(PHI_ATOMS), -pi, pi, bias_width_rad, True,
                          gridWidth=grid)
    bv_psi = BiasVariable(_cv_force(PSI_ATOMS), -pi, pi, bias_width_rad, True,
                          gridWidth=grid)
    metad = Metadynamics(system, [bv_phi, bv_psi], temperature * unit.kelvin,
                         bias_factor, hill_height_kJ * unit.kilojoule_per_mole,
                         deposit_every)

    integ = openmm.LangevinMiddleIntegrator(
        temperature * unit.kelvin, 1.0 / unit.picosecond, dt_fs * unit.femtosecond)
    integ.setRandomNumberSeed(1234 + seed)
    platform = openmm.Platform.getPlatformByName(platform_name)
    props = ({"DeviceIndex": device_index, "Precision": precision}
             if platform_name in ("CUDA", "OpenCL") else {})
    sim = app.Simulation(ala.topology, system, integ, platform, props)
    sim.context.setPositions(ala.positions_nm * unit.nanometer)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(temperature * unit.kelvin, 4321 + seed)

    # 1 ns = 1e6 fs, 1 ps = 1e3 fs (getting this wrong silently shortens the run
    # by 1000x and can round `frame_every` to zero -> non-advancing loop)
    steps_per_ns = int(round(1.0e6 / dt_fs))
    steps_per_ps = int(round(1.0e3 / dt_fs))
    record_every = max(1, int(round(record_every_ns * steps_per_ns)))
    frame_every = max(1, int(round(frame_every_ps * steps_per_ps)))

    # -- equilibration: grow the bias, snapshot F for the convergence gate ---
    F_snaps = []
    done = 0
    equil_steps = int(round(equil_ns * steps_per_ns))
    grid_axis0 = np.linspace(-pi, pi, grid, endpoint=False) + (2 * pi / grid) / 2.0
    while done < equil_steps:
        chunk = min(record_every, equil_steps - done)
        metad.step(sim, chunk)
        done += chunk
        F = _free_energy_phi_psi(metad, unit)
        F_snaps.append(F)
        F0 = F - F.min()
        i, j = np.unravel_index(F0.argmin(), F0.shape)
        wsign = np.exp(-beta * F0)
        wsign = wsign / wsign.sum()
        print(f"[seed {seed}] {done // steps_per_ns:>3d}/{equil_ns:.0f} ns  "
              f"min@({np.degrees(grid_axis0[i]):+.0f},{np.degrees(grid_axis0[j]):+.0f}) "
              f"mass(phi<0)={wsign[grid_axis0 < 0].sum():.3f}", flush=True)
    F_final = _free_energy_phi_psi(metad, unit)

    # -- production: FREEZE the bias (plain simulation.step) + collect frames -
    prod_steps = int(round(prod_ns * steps_per_ns))
    frames, cvs = [], []
    done = 0
    while done < prod_steps:
        chunk = min(frame_every, prod_steps - done)
        sim.step(chunk)
        done += chunk
        state = sim.context.getState(getPositions=True)
        frames.append(np.asarray(
            state.getPositions().value_in_unit(unit.nanometer), dtype=np.float64))
        cvs.append(np.asarray(metad.getCollectiveVariables(sim), dtype=np.float64))

    grid_axis = np.linspace(-pi, pi, grid, endpoint=False) + (2 * pi / grid) / 2.0
    return dict(
        seed=seed, F_grid=F_final, F_snaps=np.stack(F_snaps),
        grid_axis=grid_axis, frames=np.stack(frames), cvs=np.stack(cvs),
        beta=beta, kT=kT, bias_factor=bias_factor, temperature=temperature,
        params=dict(equil_ns=equil_ns, prod_ns=prod_ns, dt_fs=dt_fs,
                    hill_height_kJ=hill_height_kJ, bias_width_rad=bias_width_rad,
                    grid=grid, deposit_every=deposit_every,
                    frame_every_ps=frame_every_ps, precision=precision))


# ------------------------------------------------------------- FES analysis
def _bilinear_periodic(F: np.ndarray, axis: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of F on the periodic (phi, psi) grid at pts (M,2)."""
    G = F.shape[0]
    step = 2 * np.pi / G
    fidx = (pts - axis[0]) / step
    i0 = np.floor(fidx).astype(np.int64)
    frac = fidx - i0
    i0 = np.mod(i0, G)
    i1 = np.mod(i0 + 1, G)
    f00 = F[i0[:, 0], i0[:, 1]]
    f10 = F[i1[:, 0], i0[:, 1]]
    f01 = F[i0[:, 0], i1[:, 1]]
    f11 = F[i1[:, 0], i1[:, 1]]
    a, b = frac[:, 0], frac[:, 1]
    return ((1 - a) * (1 - b) * f00 + a * (1 - b) * f10
            + (1 - a) * b * f01 + a * b * f11)


def _torus_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a[:, None, :] - b[None, :, :])
    d = np.minimum(d, 2 * np.pi - d)
    return np.sqrt((d ** 2).sum(-1))


def find_minima(F: np.ndarray, axis: np.ndarray, *, depth_kJ: float,
                merge_rad: float) -> np.ndarray:
    """Local minima of F (periodic 8-neighbour), below depth, merged by torus dist."""
    G = F.shape[0]
    F0 = F - F.min()
    is_min = np.ones_like(F, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            is_min &= F0 <= np.roll(np.roll(F0, di, 0), dj, 1)
    is_min &= F0 < depth_kJ
    ii, jj = np.where(is_min)
    cand = np.stack([axis[ii], axis[jj]], axis=1)
    order = np.argsort(F0[ii, jj])
    cand = cand[order]
    kept = []
    for c in cand:
        if not kept or _torus_dist(c[None], np.stack(kept)).min() > merge_rad:
            kept.append(c)
    return np.stack(kept)


def assign_torus(pts: np.ndarray, minima: np.ndarray) -> np.ndarray:
    return _torus_dist(pts, minima).argmin(axis=1)


def basin_barriers(F: np.ndarray, axis: np.ndarray, minima: np.ndarray):
    """(F at each minimum, lowest-saddle matrix between torus-Voronoi basins).

    The saddle estimate is the lowest F on the shared boundary of two basins on
    the periodic grid, so ``sad[k, j] - Fmin[k]`` is basin k's escape barrier
    toward j.
    """
    G = len(axis)
    PHI, PSI = np.meshgrid(axis, axis, indexing="ij")
    lab = assign_torus(np.stack([PHI.ravel(), PSI.ravel()], axis=1),
                       minima).reshape(G, G)
    K = minima.shape[0]
    Fmin = np.array([F[int(np.argmin(np.abs(axis - minima[k, 0]))),
                       int(np.argmin(np.abs(axis - minima[k, 1])))]
                     for k in range(K)])
    sad = np.full((K, K), np.inf)
    for di, dj in ((1, 0), (0, 1)):
        l2 = np.roll(np.roll(lab, -di, 0), -dj, 1)
        f2 = np.maximum(F, np.roll(np.roll(F, -di, 0), -dj, 1))
        m = lab != l2
        for a, b, fv in zip(lab[m], l2[m], f2[m]):
            if fv < sad[a, b]:
                sad[a, b] = sad[b, a] = fv
    return Fmin, sad


def merge_shallow_minima(F: np.ndarray, axis: np.ndarray, minima: np.ndarray,
                         beta: float, min_barrier_kT: float = 1.0) -> np.ndarray:
    """Drop minima that are not metastable states.

    A raw local minimum of a finite-resolution FES need not be separated from
    its neighbours by any barrier at all: for this system two of the five
    candidates had NEGATIVE escape barriers (the Voronoi boundary lay below
    their own minimum), i.e. they were shoulders of the beta/C5 region rather
    than metastable basins. Counting them as basins both fragments p_star and
    makes an "island occupancy" diagnostic meaningless, because local dynamics
    reaches them freely. We therefore repeatedly delete the shallowest basin
    whose escape barrier is below ``min_barrier_kT`` and let the torus Voronoi
    reassign its cells.
    """
    keep = minima.copy()
    while keep.shape[0] > 2:
        Fmin, sad = basin_barriers(F, axis, keep)
        esc = np.array([np.nanmin(np.where(np.isfinite(sad[k]), sad[k], np.nan))
                        - Fmin[k] for k in range(keep.shape[0])])
        worst = int(np.argmin(esc))
        if esc[worst] * beta >= min_barrier_kT:
            break
        keep = np.delete(keep, worst, axis=0)
    return keep


def p_star_from_fes(F: np.ndarray, axis: np.ndarray, minima: np.ndarray,
                    beta: float) -> np.ndarray:
    """Basin masses = grid quadrature of e^{-beta F} over torus-Voronoi cells."""
    G = F.shape[0]
    PHI, PSI = np.meshgrid(axis, axis, indexing="ij")
    cell = np.stack([PHI.ravel(), PSI.ravel()], axis=1)
    lab = assign_torus(cell, minima)
    w = np.exp(-beta * (F - F.min())).ravel()
    K = minima.shape[0]
    mass = np.zeros(K)
    np.add.at(mass, lab, w)
    return mass / mass.sum()


def reweight_run(run: dict) -> dict:
    """Importance weights w = exp(-beta ((gamma-1)/gamma) F(cv)) + convert frames
    to whitened internal coordinates q_tilde (torch, CPU)."""
    import torch
    from .bat import BATTransform
    from .potential import AlanineDipeptideBAT

    beta, gamma = run["beta"], run["bias_factor"]
    F = run["F_grid"] - run["F_grid"].min()
    cvs = run["cvs"]
    F_at = _bilinear_periodic(F, run["grid_axis"], cvs)
    logw = -beta * ((gamma - 1.0) / gamma) * F_at
    logw -= logw.max()
    w = np.exp(logw)

    torch.set_default_dtype(torch.float64)
    pot = AlanineDipeptideBAT(device="cpu")
    x = torch.tensor(run["frames"].reshape(run["frames"].shape[0], -1))
    q = pot.bat.to_bat(x)
    qt = (q * pot.Dinv)
    with pot.no_count():
        U_eff = pot.V(qt).numpy()
    qt = qt.numpy()
    return dict(qt=qt, cvs=cvs, weights=w, U_eff=U_eff, F=F,
                grid_axis=run["grid_axis"])


def orientation_check(F: np.ndarray, axis: np.ndarray, cvs: np.ndarray,
                      beta: float, gamma: float) -> dict:
    """Verify F is indexed [phi, psi] and not its silent transpose.

    Under the frozen converged bias the production frames are distributed
    ∝ e^{-beta F / gamma}.  Their CVs come from ``getCollectiveVariables`` (true
    [phi, psi] order), so the correlation between the empirical (phi, psi)
    histogram and e^{-beta F/gamma} must be markedly higher for F than for F.T.
    """
    G = F.shape[0]
    edges = np.linspace(-np.pi, np.pi, G + 1)
    H, _, _ = np.histogram2d(cvs[:, 0], cvs[:, 1], bins=[edges, edges])
    H = H / H.sum()

    def _corr(Fx):
        p = np.exp(-beta * (Fx - Fx.min()) / gamma)
        p = p / p.sum()
        return float(np.corrcoef(H.ravel(), p.ravel())[0, 1])

    c, ct = _corr(F), _corr(F.T)
    return {"corr_F": c, "corr_F_transposed": ct, "oriented_correctly": bool(c > ct)}


def _weighted_occupancy(labels: np.ndarray, w: np.ndarray, K: int) -> np.ndarray:
    p = np.zeros(K)
    np.add.at(p, labels, w)
    return p / p.sum()


def run_and_save_seed(seed: int, out_dir: str | None = None, **run_kwargs) -> str:
    """Run one seed and save its raw run + reweighting to a per-seed npz.

    Seeds are independent, so they are launched as parallel single-threaded
    processes and combined afterwards by ``combine_seeds``.
    """
    out_dir = out_dir or CACHE_DIR
    os.makedirs(out_dir, exist_ok=True)
    run = run_wt_metad(seed, **run_kwargs)
    rw = reweight_run(run)
    path = os.path.join(out_dir, f"seed{seed}.npz")
    # the raw Cartesian frames are saved too: they are the expensive product,
    # and any later change to the internal-coordinate convention then only needs
    # a cheap re-conversion rather than a fresh metadynamics run.
    np.savez(path, qt=rw["qt"], cvs=rw["cvs"], weights=rw["weights"],
             U_eff=rw["U_eff"], frames=run["frames"],
             F_grid=run["F_grid"], F_snaps=run["F_snaps"],
             grid_axis=run["grid_axis"], beta=run["beta"], kT=run["kT"],
             bias_factor=run["bias_factor"], seed=seed,
             params=np.array(json.dumps(run["params"])))
    print(f"wrote {path}", flush=True)
    return path


def _load_seed(path: str) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as d:
        run = dict(seed=int(d["seed"]), F_grid=d["F_grid"], F_snaps=d["F_snaps"],
                   grid_axis=d["grid_axis"], beta=float(d["beta"]),
                   kT=float(d["kT"]), bias_factor=float(d["bias_factor"]),
                   params=json.loads(str(d["params"])))
        rw = dict(qt=d["qt"], cvs=d["cvs"], weights=d["weights"],
                  U_eff=d["U_eff"])
    return run, rw


def combine_seeds(seed_paths, depth_kJ: float = 25.0, merge_rad: float = 0.7,
                  out_dir: str | None = None, fig_path: str | None = None) -> dict:
    """Combine per-seed runs into the cached reference (FES, basins, p_star)."""
    out_dir = out_dir or CACHE_DIR
    os.makedirs(out_dir, exist_ok=True)
    loaded = [_load_seed(p) for p in seed_paths]
    runs = [r for r, _ in loaded]
    rew = [w for _, w in loaded]
    seeds = tuple(r["seed"] for r in runs)
    return _analyse(runs, rew, seeds, depth_kJ, merge_rad, out_dir, fig_path)


def build_and_cache(seeds=(0, 1), depth_kJ: float = 25.0, merge_rad: float = 0.7,
                    out_dir: str | None = None, fig_path: str | None = None,
                    **run_kwargs) -> dict:
    """Run WT-metad for each seed sequentially, then combine (see run_and_save_seed
    + combine_seeds for the parallel path)."""
    out_dir = out_dir or CACHE_DIR
    os.makedirs(out_dir, exist_ok=True)

    runs = [run_wt_metad(s, **run_kwargs) for s in seeds]
    rew = [reweight_run(r) for r in runs]
    return _analyse(runs, rew, seeds, depth_kJ, merge_rad, out_dir, fig_path)


def _analyse(runs, rew, seeds, depth_kJ, merge_rad, out_dir, fig_path) -> dict:
    beta = runs[0]["beta"]
    axis = runs[0]["grid_axis"]

    # averaged FES (aligned to its own minimum) -> minima -> torus-Voronoi basins
    F_avg = np.mean([r["F_grid"] - r["F_grid"].min() for r in runs], axis=0)
    F_avg -= F_avg.min()
    minima = find_minima(F_avg, axis, depth_kJ=depth_kJ, merge_rad=merge_rad)
    n_raw = int(minima.shape[0])
    minima = merge_shallow_minima(F_avg, axis, minima, beta)
    K = minima.shape[0]
    Fmin_b, sad_b = basin_barriers(F_avg, axis, minima)
    escape_kT = [float((np.nanmin(np.where(np.isfinite(sad_b[k]), sad_b[k], np.nan))
                        - Fmin_b[k]) * beta) for k in range(K)]

    # per-seed p_star (weighted occupancy) for the reproducibility gate
    per_seed_pstar = []
    for rw in rew:
        lab = assign_torus(rw["cvs"], minima)
        per_seed_pstar.append(_weighted_occupancy(lab, rw["weights"], K))
    per_seed_pstar = np.stack(per_seed_pstar)

    # pooled reference conformer set
    qt = np.concatenate([rw["qt"] for rw in rew], axis=0)
    cvs = np.concatenate([rw["cvs"] for rw in rew], axis=0)
    U_eff = np.concatenate([rw["U_eff"] for rw in rew], axis=0)
    weights = np.concatenate([rw["weights"] for rw in rew], axis=0)
    weights = weights / weights.sum()
    labels = assign_torus(cvs, minima)
    p_star = _weighted_occupancy(labels, weights, K)
    p_star_fes = p_star_from_fes(F_avg, axis, minima, beta)

    # orientation: catches the silent [psi, phi] transpose of the OpenMM grid
    orient = orientation_check(F_avg, axis, cvs, beta, runs[0]["bias_factor"])
    if not orient["oriented_correctly"]:
        raise RuntimeError(f"FES axis orientation check failed: {orient}")

    # periodicity diagnostic (task S2): how much reference mass sits at the
    # +-pi seam of each CV, and where the lowest-density branch cut would be
    seam = {}
    for name, col in (("phi", 0), ("psi", 1)):
        for m in (0.15, 0.30):
            d = np.abs(cvs[:, col] - np.pi)
            d = np.minimum(d, 2 * np.pi - d)
            seam[f"{name}_mass_within_{m}rad_of_seam"] = float(weights[d < m].sum())
        best = None
        for cdeg in range(-180, 180, 5):
            c = np.radians(cdeg)
            d = np.abs(cvs[:, col] - c)
            d = np.minimum(d, 2 * np.pi - d)
            mass = float(weights[d < 0.15].sum())
            if best is None or mass < best[1]:
                best = (cdeg, mass)
        seam[f"{name}_lowest_density_cut_deg"] = best[0]
        seam[f"{name}_lowest_density_cut_mass"] = best[1]

    # convergence: FES change over the last third of the equilibration snapshots
    conv = _convergence_metrics(runs, minima, axis, beta)
    if fig_path:
        _plot_convergence(runs, F_avg, axis, minima, conv, fig_path)

    cache = dict(
        qt=qt, cvs=cvs, U_eff=U_eff, weights=weights, labels=labels,
        F_grid=F_avg, grid_axis=axis, minima=minima, p_star=p_star,
        basin_escape_kT=np.array(escape_kT),
        basin_saddles_kJ=sad_b, basin_min_kJ=Fmin_b,
        p_star_fes=p_star_fes, per_seed_pstar=per_seed_pstar, beta=beta,
        kT=runs[0]["kT"], seeds=np.array(seeds),
        provenance=np.array(json.dumps(dict(
            method="wt_metadynamics_reweighted", seeds=list(seeds),
            n_basins=int(K), n_raw_minima=n_raw,
            basin_escape_kT=[round(v, 3) for v in escape_kT],
            min_barrier_kT=1.0,
            minima_deg=np.degrees(minima).round(1).tolist(),
            depth_kJ=depth_kJ, merge_rad=merge_rad, beta=beta,
            p_star=p_star.round(4).tolist(),
            p_star_fes=p_star_fes.round(4).tolist(),
            per_seed_pstar=per_seed_pstar.round(4).tolist(),
            orientation=orient, seam=seam,
            convergence=conv, run_params=runs[0]["params"]))),
    )
    cache_path = os.path.join(out_dir, "reference.npz")
    np.savez(cache_path, **cache)
    return dict(cache_path=cache_path, minima=minima, p_star=p_star,
                p_star_fes=p_star_fes, per_seed_pstar=per_seed_pstar,
                convergence=conv, K=K, orientation=orient)


def _convergence_metrics(runs, minima, axis, beta) -> dict:
    """Per-basin free energy over the last third of equilibration snapshots and
    the FES drift, for the P4 convergence gate."""
    out = {}
    for r in runs:
        snaps = r["F_snaps"]
        n = snaps.shape[0]
        # A free energy is defined only up to an additive constant, and the
        # well-tempered bias adds a growing UNIFORM offset (measured ~-4.4
        # kJ/mol over the last third). Comparing raw snapshots therefore
        # measures that offset, not convergence: align each snapshot to its own
        # minimum before differencing.
        last3 = snaps[max(0, n - max(1, n // 3)):]
        aligned = np.stack([F - F.min() for F in last3])
        drift = aligned[-1] - aligned[0]
        F_end = aligned[-1]
        wgt = np.exp(-beta * F_end)
        wgt = wgt / wgt.sum()
        drift_grid = float(np.sqrt((drift ** 2).mean()))
        drift_weighted = float(np.sqrt((wgt * drift ** 2).sum()))
        # basin free energies -kT ln sum_cell e^{-beta F} per snapshot
        basin_dF = []
        for F in last3:
            F0 = F - F.min()
            PHI, PSI = np.meshgrid(axis, axis, indexing="ij")
            cell = np.stack([PHI.ravel(), PSI.ravel()], axis=1)
            lab = assign_torus(cell, minima)
            mass = np.zeros(minima.shape[0])
            np.add.at(mass, lab, np.exp(-beta * F0).ravel())
            f = -np.log(mass / mass.sum()) / beta
            basin_dF.append(f - f.min())
        basin_dF = np.stack(basin_dF)
        rng = (basin_dF.max(0) - basin_dF.min(0)) * beta
        # basin masses at the final snapshot, so the gate can separate the
        # basins that carry mass from the convergence-limited tiny ones
        mass_end = np.zeros(minima.shape[0])
        PHI, PSI = np.meshgrid(axis, axis, indexing="ij")
        lab_end = assign_torus(np.stack([PHI.ravel(), PSI.ravel()], axis=1), minima)
        np.add.at(mass_end, lab_end, np.exp(-beta * F_end).ravel())
        mass_end = mass_end / mass_end.sum()
        major = mass_end >= 0.01
        # DRIFT (last - first over the aligned last-third snapshots) is the gated
        # measure of "stable across the last checkpoints". The RANGE (max - min)
        # is also recorded but NOT gated: a range statistic's expectation grows
        # with the number of snapshots it spans, so it is not a consistent
        # estimator -- a longer run supplies more snapshots and need never pass
        # it. Drift is consistent (it tends to zero as the estimate settles) and
        # is exactly the convention already used by fes_drift_last_third above.
        bdrift = np.abs(basin_dF[-1] - basin_dF[0]) * beta
        out[f"seed{int(r['seed'])}"] = dict(
            fes_drift_last_third_kJ=drift_grid,
            fes_drift_last_third_mass_weighted_kJ=drift_weighted,
            basin_dF_drift_kT=float(bdrift.max()),
            basin_dF_drift_major_kT=float(bdrift[major].max()) if major.any() else 0.0,
            basin_dF_drift_per_basin_kT=bdrift.round(3).tolist(),
            basin_dF_range_kT=float(rng.max()),
            basin_dF_range_major_kT=float(rng[major].max()) if major.any() else 0.0,
            basin_mass_end=mass_end.round(5).tolist(),
            basin_dF_range_per_basin_kT=rng.round(3).tolist(),
            basin_dF_final_kT=(basin_dF[-1] * beta).round(3).tolist())
    return out


def _plot_convergence(runs, F_avg, axis, minima, conv, fig_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ext = [-180, 180, -180, 180]
    im = ax[0].imshow((F_avg).T, origin="lower", extent=ext, aspect="auto",
                      cmap="viridis", vmax=40)
    ax[0].scatter(np.degrees(minima[:, 0]), np.degrees(minima[:, 1]),
                  c="red", s=30, marker="x")
    ax[0].set_xlabel(r"$\phi$ (deg)")
    ax[0].set_ylabel(r"$\psi$ (deg)")
    ax[0].set_title("reweighted FES(phi,psi) [kJ/mol]")
    fig.colorbar(im, ax=ax[0])
    for r in runs:
        snaps = r["F_snaps"]
        drift = [float(np.sqrt(np.mean((snaps[k] - snaps[-1]) ** 2)))
                 for k in range(snaps.shape[0])]
        t = np.arange(1, len(drift) + 1) * r["params"]["equil_ns"] / len(drift)
        ax[1].plot(t, drift, marker="o", label=f"seed {int(r['seed'])}")
    ax[1].set_xlabel("equilibration time (ns)")
    ax[1].set_ylabel("RMS |F(t) - F(final)| (kJ/mol)")
    ax[1].set_title("FES convergence")
    ax[1].legend()
    ax[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    fig.savefig(fig_path.replace(".png", ".pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("seed", "combine", "all"), default="all",
                    help="'seed' runs ONE seed (parallel processes); 'combine' "
                         "merges per-seed npz files; 'all' does both serially.")
    ap.add_argument("--seed", type=int, default=0, help="for --mode seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--equil-ns", type=float, default=8.0)
    ap.add_argument("--prod-ns", type=float, default=10.0)
    ap.add_argument("--platform", type=str, default="CPU")
    ap.add_argument("--device-index", type=str, default="0")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--fig", type=str,
                    default=os.path.join(os.path.dirname(os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__)))),
                        "figures", "e5_alanine", "reference_convergence.png"))
    args = ap.parse_args()
    out_dir = args.out_dir or CACHE_DIR

    if args.mode == "seed":
        run_and_save_seed(args.seed, out_dir=out_dir, equil_ns=args.equil_ns,
                          prod_ns=args.prod_ns, platform_name=args.platform,
                          device_index=args.device_index)
        return
    if args.mode == "combine":
        paths = [os.path.join(out_dir, f"seed{s}.npz") for s in args.seeds]
        res = combine_seeds(paths, out_dir=out_dir, fig_path=args.fig)
    else:
        res = build_and_cache(
            seeds=tuple(args.seeds), equil_ns=args.equil_ns,
            prod_ns=args.prod_ns, platform_name=args.platform,
            device_index=args.device_index, out_dir=out_dir, fig_path=args.fig)
    print("cache:", res["cache_path"])
    print("K basins:", res["K"])
    print("p_star:", np.round(res["p_star"], 4).tolist())
    print("p_star (FES integral):", np.round(res["p_star_fes"], 4).tolist())
    print("per-seed p_star:", np.round(res["per_seed_pstar"], 4).tolist())
    print("convergence:", json.dumps(res["convergence"], indent=2))


if __name__ == "__main__":
    main()
