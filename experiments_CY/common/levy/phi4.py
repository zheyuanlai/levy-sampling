"""Two-component phi4 field diagnostics for Phase 16A."""

from __future__ import annotations

import itertools

import numpy as np

from .score import FLOAT_LOG_MAX, _ratio_from_log


SIGN_PHASES = ("--", "-+", "+-", "++")
SIGN_STARTS = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
DEFAULT_COUPLED_PHI4_PARAMS = {
    "ax": 1.1408268818154723,
    "ay": 0.9826939868941122,
    "c": -0.056727600536650935,
    "hx": -0.10525734006273596,
    "hy": -0.2334887129135239,
    "eta": 0.46732375068734666,
}
DEFAULT_COUPLED_DW_PARAMS = DEFAULT_COUPLED_PHI4_PARAMS


def coupled_phi4_local_potential(q, ax=1.0, ay=1.0, c=0.0, hx=0.0, hy=0.0, eta=0.0):
    """Coupled two-component double-well local potential.

    The optional eta term is a weak nonlinear coupling, 0.5 * eta * x^2 y.
    It keeps the quartic double-well origin while breaking repeated edge
        directions in the four-phase geometry.
    """

    q = np.asarray(q, dtype=float)
    x = q[..., 0]
    y = q[..., 1]
    return (
        ax / 4.0 * (x * x - 1.0) ** 2
        + ay / 4.0 * (y * y - 1.0) ** 2
        + c * x * y
        + hx * x
        + hy * y
        + 0.5 * eta * x * x * y
    )


def grad_coupled_phi4_local_potential(q, ax=1.0, ay=1.0, c=0.0, hx=0.0, hy=0.0, eta=0.0):
    q = np.asarray(q, dtype=float)
    x = q[..., 0]
    y = q[..., 1]
    g = np.empty_like(q, dtype=float)
    g[..., 0] = ax * x * (x * x - 1.0) + c * y + hx + eta * x * y
    g[..., 1] = ay * y * (y * y - 1.0) + c * x + hy + 0.5 * eta * x * x
    return g


def hess_coupled_phi4_local_potential(q, ax=1.0, ay=1.0, c=0.0, hx=0.0, hy=0.0, eta=0.0):
    x, y = np.asarray(q, dtype=float)
    return np.array(
        [
            [ax * (3.0 * x * x - 1.0) + eta * y, c + eta * x],
            [c + eta * x, ay * (3.0 * y * y - 1.0)],
        ]
    )


def coupled_dw_local_potential(q, ax=1.0, ay=1.0, c=0.0, hx=0.0, hy=0.0, eta=0.0):
    return coupled_phi4_local_potential(q, ax=ax, ay=ay, c=c, hx=hx, hy=hy, eta=eta)


def grad_coupled_dw_local_potential(q, ax=1.0, ay=1.0, c=0.0, hx=0.0, hy=0.0, eta=0.0):
    return grad_coupled_phi4_local_potential(q, ax=ax, ay=ay, c=c, hx=hx, hy=hy, eta=eta)


def hess_coupled_dw_local_potential(q, ax=1.0, ay=1.0, c=0.0, hx=0.0, hy=0.0, eta=0.0):
    return hess_coupled_phi4_local_potential(q, ax=ax, ay=ay, c=c, hx=hx, hy=hy, eta=eta)


def find_coupled_phi4_minima(params, minimize):
    """Locate the four sign-state local minima from natural sign starts."""

    def W(z):
        return coupled_phi4_local_potential(z, **params)

    def G(z):
        return grad_coupled_phi4_local_potential(z, **params)

    rows = []
    points = []
    for label, start in zip(SIGN_PHASES, SIGN_STARTS):
        res = minimize(W, start.copy(), jac=G, method="BFGS", options={"gtol": 1e-11, "maxiter": 300})
        point = np.asarray(res.x, dtype=float)
        H = hess_coupled_phi4_local_potential(point, **params)
        eigs = np.linalg.eigvalsh(0.5 * (H + H.T))
        sign_ok = np.array_equal(np.sign(point), np.sign(start))
        rows.append(
            {
                "phase": label,
                "start_x": float(start[0]),
                "start_y": float(start[1]),
                "x": float(point[0]),
                "y": float(point[1]),
                "energy": float(W(point)),
                "grad_norm": float(np.linalg.norm(G(point))),
                "hessian_min_eig": float(eigs[0]),
                "hessian_max_eig": float(eigs[-1]),
                "sign_state_ok": bool(sign_ok),
                "optimizer_success": bool(res.success or np.linalg.norm(G(point)) < 1e-7),
            }
        )
        points.append(point)
    return np.asarray(points), rows


def find_coupled_dw_minima(params, minimize):
    return find_coupled_phi4_minima(params, minimize)


def edge_vectors_from_minima(minima, phase_names=SIGN_PHASES):
    minima = np.asarray(minima, dtype=float)
    rows = []
    for i, j in itertools.combinations(range(len(minima)), 2):
        vec = minima[j] - minima[i]
        norm = float(np.linalg.norm(vec))
        rows.append(
            {
                "edge": f"{phase_names[i]}->{phase_names[j]}",
                "i": int(i),
                "j": int(j),
                "from_phase": phase_names[i],
                "to_phase": phase_names[j],
                "dx": float(vec[0]),
                "dy": float(vec[1]),
                "length": norm,
                "angle": float(np.arctan2(vec[1], vec[0])),
            }
        )
    return rows


def minimum_parallel_sine(edge_rows):
    best = 1.0
    pair = ("", "")
    for a, b in itertools.combinations(edge_rows, 2):
        va = np.array([a["dx"], a["dy"]], dtype=float)
        vb = np.array([b["dx"], b["dy"]], dtype=float)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom <= 0.0:
            continue
        sine = float(abs(np.cross(va, vb)) / denom)
        if sine < best:
            best = sine
            pair = (a["edge"], b["edge"])
    return best, pair


def edge_direction_audit(edge_rows):
    min_sine, pair = minimum_parallel_sine(edge_rows)
    lengths = np.array([row["length"] for row in edge_rows], dtype=float)
    return {
        "minimum_sine": float(min_sine),
        "closest_edge_direction_pair_a": pair[0],
        "closest_edge_direction_pair_b": pair[1],
        "edge_length_min": float(np.min(lengths)),
        "edge_length_max": float(np.max(lengths)),
        "edge_length_ratio": float(np.max(lengths) / max(np.min(lengths), 1e-300)),
    }


def mst_edges_from_minima(minima):
    """Return a Euclidean minimum spanning tree on the located minima."""

    minima = np.asarray(minima, dtype=float)
    candidates = []
    for i, j in itertools.combinations(range(len(minima)), 2):
        candidates.append((float(np.linalg.norm(minima[j] - minima[i])), i, j))
    parent = list(range(len(minima)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = []
    for _, i, j in sorted(candidates):
        ri, rj = root(i), root(j)
        if ri == rj:
            continue
        parent[ri] = rj
        edges.append((i, j))
        if len(edges) == len(minima) - 1:
            break
    return edges


def coupled_phi4_graph_edge_sets(minima, phase_names=SIGN_PHASES):
    """Graph families for four coupled-double-well modes."""

    all_edges = list(itertools.combinations(range(4), 2))
    lengths = {edge: float(np.linalg.norm(np.asarray(minima[edge[1]]) - np.asarray(minima[edge[0]]))) for edge in all_edges}
    mst = mst_edges_from_minima(minima)
    cycle = [(0, 1), (1, 2), (2, 3), (3, 0)]
    edge_rows = edge_vectors_from_minima(minima, phase_names=phase_names)
    _, pair = minimum_parallel_sine(edge_rows)
    edge_lookup = {row["edge"]: (int(row["i"]), int(row["j"])) for row in edge_rows}
    duplicate_candidates = [edge_lookup[e] for e in pair if e in edge_lookup]
    if duplicate_candidates:
        duplicate_candidates = sorted(duplicate_candidates, key=lambda edge: lengths[tuple(sorted(edge))])
        removed = tuple(sorted(duplicate_candidates[0]))
    else:
        removed = max(lengths, key=lengths.get)
    five = [edge for edge in all_edges if edge != removed]
    return {
        "MST": mst,
        "cycle": cycle,
        "5": five,
        "complete": all_edges,
    }


def coupled_dw_graph_edge_sets(minima):
    return coupled_phi4_graph_edge_sets(minima)


def build_coupled_phi4_basin_map(
    params,
    minima,
    xlim=(-2.0, 2.0),
    ylim=(-2.0, 2.0),
    grid_n=300,
    step=0.08,
    max_iter=700,
    tol=1e-6,
):
    gx = np.linspace(float(xlim[0]), float(xlim[1]), int(grid_n))
    gy = np.linspace(float(ylim[0]), float(ylim[1]), int(grid_n))
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    z = np.column_stack([X.ravel(), Y.ravel()])
    converged = np.zeros(z.shape[0], dtype=bool)
    final_norm = np.full(z.shape[0], np.inf)
    for _ in range(int(max_iter)):
        g = grad_coupled_phi4_local_potential(z, **params)
        gnorm = np.linalg.norm(g, axis=1)
        final_norm = gnorm
        converged |= gnorm < tol
        if np.all(converged):
            break
        active = ~converged
        z[active] = z[active] - step * g[active] / (1.0 + step * gnorm[active, None])
    labels = np.argmin(np.sum((z[:, None, :] - np.asarray(minima)[None, :, :]) ** 2, axis=2), axis=1)
    labels = labels.reshape(X.shape)
    metadata = {
        "grid_nx": int(len(gx)),
        "grid_ny": int(len(gy)),
        "x_min": float(gx[0]),
        "x_max": float(gx[-1]),
        "y_min": float(gy[0]),
        "y_max": float(gy[-1]),
        "gradient_flow_step": float(step),
        "gradient_flow_max_iter": int(max_iter),
        "gradient_flow_tol": float(tol),
        "basin_label_converged_fraction": float(np.mean(converged)),
        "basin_label_failure_fraction": float(1.0 - np.mean(converged)),
        "final_gradient_norm_mean": float(np.mean(final_norm)),
        "final_gradient_norm_max": float(np.max(final_norm)),
    }
    return gx, gy, labels, metadata


def lookup_coupled_phi4_basin_labels(points, gx, gy, basin_labels):
    points = np.asarray(points, dtype=float)
    gx = np.asarray(gx, dtype=float)
    gy = np.asarray(gy, dtype=float)
    ix = np.searchsorted(gx, points[..., 0], side="left")
    iy = np.searchsorted(gy, points[..., 1], side="left")
    ix = np.clip(ix, 1, len(gx) - 1)
    iy = np.clip(iy, 1, len(gy) - 1)
    left_x = gx[ix - 1]
    right_x = gx[ix]
    left_y = gy[iy - 1]
    right_y = gy[iy]
    ix = np.where(np.abs(points[..., 0] - left_x) <= np.abs(points[..., 0] - right_x), ix - 1, ix)
    iy = np.where(np.abs(points[..., 1] - left_y) <= np.abs(points[..., 1] - right_y), iy - 1, iy)
    return np.asarray(basin_labels)[iy, ix]


def coupled_phi4_local_gibbs_basin_masses(gx, gy, basin_labels, params, eps, n_modes=4):
    X, Y = np.meshgrid(np.asarray(gx, dtype=float), np.asarray(gy, dtype=float), indexing="xy")
    pts = np.stack([X, Y], axis=-1)
    logw = -coupled_phi4_local_potential(pts, **params) / float(eps)
    logw = logw - np.max(logw)
    weights = np.exp(logw)
    dx = float(np.mean(np.diff(gx))) if len(gx) > 1 else 1.0
    dy = float(np.mean(np.diff(gy))) if len(gy) > 1 else 1.0
    raw = np.array([float(np.sum(weights[np.asarray(basin_labels) == k]) * dx * dy) for k in range(int(n_modes))])
    masses = raw / max(float(np.sum(raw)), 1e-300)
    return masses, raw


def homogeneous_site_shift_from_vector(r, n_grid, atol=1e-10):
    """Return the two-component site shift represented by a homogeneous field vector."""

    arr = np.asarray(r, dtype=float).reshape(int(n_grid), 2)
    site_shift = arr[0].copy()
    if not np.allclose(arr, site_shift[None, :], rtol=0.0, atol=atol):
        raise ValueError("GL optimized score requires homogeneous site shifts")
    return site_shift


def coupled_phi4_homogeneous_shift_local_energy_delta(z, r, n_grid, local_params=None):
    """Exact energy difference for homogeneous GL shifts using only local energy.

    For a homogeneous field jump ``r`` with site shift ``d``, the periodic
    gradient energy is invariant under ``q_i -> q_i - d``.  Therefore
    ``E(q-r)-E(q)`` is exactly ``h * sum_i [W(q_i-d)-W(q_i)]``.
    """

    params = DEFAULT_COUPLED_PHI4_PARAMS if local_params is None else dict(local_params)
    n_grid = int(n_grid)
    field = np.asarray(z, dtype=float).reshape(-1, n_grid, 2)
    site_shift = homogeneous_site_shift_from_vector(r, n_grid)
    h = 1.0 / n_grid
    base_local = coupled_phi4_local_potential(field, **params)
    shifted_local = coupled_phi4_local_potential(field - site_shift, **params)
    return h * np.sum(shifted_local - base_local, axis=1)


def coupled_phi4_site_moments(q, n_grid=None):
    """Per-particle polynomial moments for the coupled-phi4 local potential."""

    arr = np.asarray(q, dtype=float)
    if arr.ndim == 2 and arr.shape[-1] != 2:
        if n_grid is None:
            n_grid = arr.shape[-1] // 2
        arr = arr.reshape(-1, int(n_grid), 2)
    elif arr.ndim == 2 and arr.shape[-1] == 2:
        arr = arr.reshape(1, arr.shape[0], 2)
    elif arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError("q must have shape (n_particles, 2*n_grid), (n_grid, 2), or (n_particles, n_grid, 2)")

    x = arr[..., 0]
    y = arr[..., 1]
    return {
        "n_sites": int(arr.shape[1]),
        "sum_x": np.sum(x, axis=1),
        "sum_y": np.sum(y, axis=1),
        "sum_x2": np.sum(x * x, axis=1),
        "sum_y2": np.sum(y * y, axis=1),
        "sum_x3": np.sum(x * x * x, axis=1),
        "sum_y3": np.sum(y * y * y, axis=1),
        "sum_x4": np.sum(x**4, axis=1),
        "sum_y4": np.sum(y**4, axis=1),
        "sum_xy": np.sum(x * y, axis=1),
        "sum_x2y": np.sum(x * x * y, axis=1),
        "sum_xy2": np.sum(x * y * y, axis=1),
        "sum_x2_y2": np.sum(x * x * y * y, axis=1),
    }


def coupled_phi4_shift_delta_from_moments(moments, dx, dy, params=None, h=None):
    """Exact local-energy shift deltas from coupled-phi4 moments.

    ``dx`` and ``dy`` may be scalars or one-dimensional arrays.  The return
    shape is ``(n_particles, n_deltas)``.
    """

    p = DEFAULT_COUPLED_PHI4_PARAMS if params is None else dict(params)
    dx = np.asarray(dx, dtype=float).reshape(-1)
    dy = np.asarray(dy, dtype=float).reshape(-1)
    if dx.shape != dy.shape:
        raise ValueError("dx and dy must have the same shape")

    n = float(moments["n_sites"])
    h = (1.0 / n) if h is None else float(h)
    Sx = np.asarray(moments["sum_x"], dtype=float)[:, None]
    Sy = np.asarray(moments["sum_y"], dtype=float)[:, None]
    Sx2 = np.asarray(moments["sum_x2"], dtype=float)[:, None]
    Sy2 = np.asarray(moments["sum_y2"], dtype=float)[:, None]
    Sx3 = np.asarray(moments["sum_x3"], dtype=float)[:, None]
    Sy3 = np.asarray(moments["sum_y3"], dtype=float)[:, None]
    Sxy = np.asarray(moments["sum_xy"], dtype=float)[:, None]

    dx = dx[None, :]
    dy = dy[None, :]
    dx2 = dx * dx
    dy2 = dy * dy

    delta_x = p["ax"] / 4.0 * (
        -4.0 * dx * Sx3
        + 6.0 * dx2 * Sx2
        - 4.0 * dx2 * dx * Sx
        + n * dx2 * dx2
        + 4.0 * dx * Sx
        - 2.0 * n * dx2
    )
    delta_y = p["ay"] / 4.0 * (
        -4.0 * dy * Sy3
        + 6.0 * dy2 * Sy2
        - 4.0 * dy2 * dy * Sy
        + n * dy2 * dy2
        + 4.0 * dy * Sy
        - 2.0 * n * dy2
    )
    delta_xy = p["c"] * (-dx * Sy - dy * Sx + n * dx * dy)
    delta_field = -p["hx"] * n * dx - p["hy"] * n * dy
    delta_eta = 0.5 * p["eta"] * (
        -dy * Sx2
        - 2.0 * dx * Sxy
        + 2.0 * dx * dy * Sx
        + dx2 * Sy
        - n * dx2 * dy
    )
    return h * (delta_x + delta_y + delta_xy + delta_field + delta_eta)


def coupled_phi4_homogeneous_shift_local_energy_delta_moment(q, deltas, params=None, h=None, n_grid=None):
    """Vectorized exact local-energy deltas for homogeneous site shifts.

    ``deltas`` has shape ``(n_deltas, 2)`` or ``(2,)`` and the result has
    shape ``(n_particles, n_deltas)``.
    """

    deltas = np.asarray(deltas, dtype=float)
    if deltas.shape == (2,):
        deltas = deltas.reshape(1, 2)
    if deltas.ndim != 2 or deltas.shape[1] != 2:
        raise ValueError("deltas must have shape (n_deltas, 2)")
    moments = coupled_phi4_site_moments(q, n_grid=n_grid)
    return coupled_phi4_shift_delta_from_moments(moments, deltas[:, 0], deltas[:, 1], params=params, h=h)


def levy_score_coupled_phi4_edge_shell(
    z,
    jump,
    n_grid,
    local_params,
    eps,
    theta_nodes,
    theta_weights,
    rho_nodes,
    rho_weights,
    log_clip=60.0,
    score_clip=None,
    profile=False,
    return_diagnostics=False,
):
    """Exact moment-optimized Levy-score correction for homogeneous coupled-phi4 GL shell jumps."""

    z = np.asarray(z, dtype=float)
    n_grid = int(n_grid)
    params = DEFAULT_COUPLED_PHI4_PARAMS if local_params is None else dict(local_params)
    theta_nodes = np.asarray(theta_nodes, dtype=float)
    theta_weights = np.asarray(theta_weights, dtype=float)
    rho_nodes = np.asarray(rho_nodes, dtype=float)
    rho_weights = np.asarray(rho_weights, dtype=float)

    timer = None
    t_score0 = t_mom0 = t_build0 = t_delta0 = t_acc0 = None
    profile_times = {
        "score_total_seconds": 0.0,
        "moment_build_seconds": 0.0,
        "delta_build_seconds": 0.0,
        "local_energy_delta_seconds": 0.0,
        "score_accumulation_seconds": 0.0,
    }
    if profile:
        from time import perf_counter

        timer = perf_counter
        t_score0 = timer()
        t_mom0 = timer()
    moments = coupled_phi4_site_moments(z, n_grid=n_grid)
    if profile:
        profile_times["moment_build_seconds"] = timer() - t_mom0
        t_build0 = timer()

    full_shifts = []
    site_deltas = []
    weights = []
    for r0, wm in zip(jump.centers, jump.weights):
        r0 = np.asarray(r0, dtype=float)
        norm = np.linalg.norm(r0)
        u_full = r0 / max(norm, 1e-14)
        site_shift0 = homogeneous_site_shift_from_vector(r0, n_grid)
        site_u = homogeneous_site_shift_from_vector(u_full, n_grid)

        for rho, wrho in zip(rho_nodes, rho_weights):
            r = r0 + rho * u_full
            site_shift = site_shift0 + rho * site_u
            for theta, wtheta in zip(theta_nodes, theta_weights):
                full_shifts.append(r)
                site_deltas.append(theta * site_shift)
                weights.append(float(wm * wrho * wtheta))

    full_shifts = np.asarray(full_shifts, dtype=float)
    site_deltas = np.asarray(site_deltas, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if profile:
        profile_times["delta_build_seconds"] = timer() - t_build0
        t_delta0 = timer()
    energy_delta = coupled_phi4_shift_delta_from_moments(
        moments, site_deltas[:, 0], site_deltas[:, 1], params=params, h=1.0 / n_grid
    )
    if profile:
        profile_times["local_energy_delta_seconds"] = timer() - t_delta0
        t_acc0 = timer()
    log_ratio = -energy_delta / eps
    max_log_ratio = float(np.max(np.abs(log_ratio))) if log_ratio.size else 0.0
    total_count = int(log_ratio.size)
    ratio, clip_count, overflow_count, effective_log_clip = _ratio_from_log(log_ratio, log_clip)
    S = -float(jump.lam) * np.einsum("nk,k,kd->nd", ratio, weights, full_shifts)
    if profile:
        profile_times["score_accumulation_seconds"] = timer() - t_acc0

    raw = S.copy()
    if score_clip is not None:
        S = np.clip(S, -score_clip, score_clip)
    if profile:
        profile_times["score_total_seconds"] = timer() - t_score0

    diagnostics = {
        "score_clip_fraction": float(np.mean(S != raw)) if score_clip is not None else 0.0,
        "logratio_clip_fraction": float(clip_count / max(total_count, 1)),
        "score_changing_logratio_clip_fraction": float(clip_count / max(total_count, 1)),
        "overflow_guard_logratio_clip_fraction": float(overflow_count / max(total_count, 1)),
        "effective_log_clip": float(effective_log_clip if log_ratio.size else FLOAT_LOG_MAX),
        "max_score_norm": float(np.max(np.linalg.norm(raw, axis=1))) if raw.size else 0.0,
        "max_log_ratio": float(max_log_ratio),
        "h_shell": float(getattr(jump, "h_shell", 0.0)),
        "n_theta": int(len(theta_nodes)),
        "n_rho": int(len(rho_nodes)),
        "rho_n": int(len(rho_nodes)),
        "moment_score": 1.0,
    }
    if profile:
        diagnostics.update(profile_times)

    return (S, diagnostics) if return_diagnostics else S


def magnetization(z, n_grid):
    q = np.asarray(z, dtype=float).reshape(-1, n_grid, 2)
    return q.mean(axis=1)


def binder_cumulant_vector(M):
    M = np.asarray(M, dtype=float)
    m2 = np.mean(np.sum(M * M, axis=1))
    m4 = np.mean(np.sum(M * M, axis=1) ** 2)
    return 1.0 - m4 / (2.0 * m2 * m2 + 1e-30)


def susceptibility(M, n_grid, eps):
    M = np.asarray(M, dtype=float)
    r = np.linalg.norm(M, axis=1)
    return n_grid / eps * (np.mean(r * r) - np.mean(r) ** 2)


def gl_energy_parts(z, n_grid, kappa, local_params=None):
    """Return gradient and coupled-double-well local energy parts."""

    params = DEFAULT_COUPLED_PHI4_PARAMS if local_params is None else dict(local_params)
    field = np.asarray(z, dtype=float).reshape(-1, n_grid, 2)
    h = 1.0 / n_grid

    dq = np.roll(field, -1, axis=1) - field
    grad_energy = 0.5 * kappa / h * np.sum(dq * dq, axis=(1, 2))
    local_energy = h * np.sum(coupled_phi4_local_potential(field, **params), axis=1)

    return grad_energy, local_energy


def vector_correlation(z, n_grid):
    q = np.asarray(z, dtype=float).reshape(-1, n_grid, 2)
    C = []
    for r in range(n_grid // 2 + 1):
        qr = np.roll(q, -r, axis=1)
        C.append(np.mean(np.sum(q * qr, axis=2)))
    return np.array(C)


def structure_factor(z, n_grid):
    q = np.asarray(z, dtype=float).reshape(-1, n_grid, 2)
    qhat = np.fft.rfft(q, axis=1)
    S = np.mean(np.abs(qhat[:, :, 0]) ** 2 + np.abs(qhat[:, :, 1]) ** 2, axis=0)
    return S / n_grid


def site_phase_labels(z, n_grid, phases):
    q = np.asarray(z, dtype=float).reshape(-1, n_grid, 2)
    centers = np.asarray(phases, dtype=float)
    diff = q[:, :, None, :] - centers[None, None, :, :]
    return np.argmin(np.sum(diff * diff, axis=3), axis=2)


def domain_wall_density(z, n_grid, phases):
    s = site_phase_labels(z, n_grid, phases=phases)
    walls = s != np.roll(s, -1, axis=1)
    return walls.mean(axis=1)
