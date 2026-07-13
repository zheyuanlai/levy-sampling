"""Raw-CP stationary-law forensic (E1).

Raw CP (compound-Poisson jumps with NO Levy-score correction) does not target
pi; it converges to a *biased* stationary law rho_inf^raw solving the linear
stationary Fokker-Planck-with-jumps (adjoint / forward) equation

    0 = eps rho'' + d/dx[ V'(x) rho ] + lam sum_{a,q} w_{a,q} [ rho(x - r_{a,q}) - rho ],
        eps = 1/beta,

on a 1D grid. The paper's premise is that raw CP is biased; the forensic
question is whether the collaborator's raw CP converges to *this predicted*
biased law (code correct, bias = theory) or to something else (bug hunt).

We solve rho_inf^raw exactly on a fine grid and return its CDF so it can be
overlaid on the empirical raw-CP CDF.

Discretisation
--------------
* Drift + diffusion: conservative finite-volume flux J = V' rho + eps rho' with
  Chang & Cooper (1970) exponential edge weighting, so the lam -> 0 stationary
  solution is EXACTLY the Gibbs density e^{-beta V} to machine precision (this
  is the self-check). No-flux (reflecting) outer edges.
* Jump term: each source node x_j deposits rate lam * w_{a,q} onto the two grid
  nodes bracketing the landing point x_j + r_{a,q} (linear, mass-conserving);
  landings outside the box clamp to the boundary node (matches the sampler's
  box.clip and conserves mass). Weights sum(w_{a,q}) = 1, so the total jump
  out-rate is exactly lam and every column of the assembled forward operator
  sums to zero (a proper CTMC generator transpose).

The stationary density is the (1-D) null vector of the forward operator L; we
obtain it by replacing one redundant conservation row with the normalisation
constraint int rho dx = 1 and solving the sparse linear system.

CPU-only, numpy/scipy, float64. No torch, no GPU.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# numpy 2.x renamed trapz -> trapezoid; support both
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _chang_cooper_delta(w: np.ndarray) -> np.ndarray:
    """Chang-Cooper edge weight delta(w) = 1/w - 1/(e^w - 1), continuous at
    w -> 0 (delta -> 1/2). `w` is the edge Peclet number V'*dx/eps."""
    out = np.empty_like(w)
    small = np.abs(w) < 1e-8
    out[small] = 0.5 - w[small] / 12.0            # series: 1/2 - w/12 + O(w^3)
    ws = w[~small]
    out[~small] = 1.0 / ws - 1.0 / np.expm1(ws)
    return out


def rawcp_stationary_density(dV_dx, beta: float, lam: float,
                             shifts: np.ndarray, weights: np.ndarray,
                             lo: float = -5.2, hi: float = 5.2,
                             n_grid: int = 4001):
    """Stationary density of 1D raw CP on [lo, hi].

    Parameters
    ----------
    dV_dx : callable x -> V'(x), numpy-vectorised.
    beta, lam : inverse temperature and jump intensity (eps = 1/beta).
    shifts, weights : 1D arrays of jump displacements r_{a,q} and probability
        weights w_{a,q} (must sum to 1); use ShellJumpLaw.quadrature_shifts(64).
    Returns (x_grid, rho, cdf) with int rho dx = 1 and cdf in [0, 1].
    """
    eps = 1.0 / beta
    x = np.linspace(lo, hi, n_grid)
    dx = x[1] - x[0]
    G = n_grid
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    shifts = np.asarray(shifts, dtype=float).reshape(-1)

    # ---- drift + diffusion: conservative Chang-Cooper flux ----------------
    # edge midpoints x_{i+1/2}, i = 0 .. G-2
    xm = 0.5 * (x[:-1] + x[1:])
    a_edge = dV_dx(xm)                                    # V'(x_{i+1/2})
    w_pe = a_edge * dx / eps                              # edge Peclet number
    delta = _chang_cooper_delta(w_pe)                    # (G-1,)
    # J_{i+1/2} = a[ delta rho_i + (1-delta) rho_{i+1} ] + eps (rho_{i+1}-rho_i)/dx
    # coefficients of rho_i and rho_{i+1} in the edge flux
    cL = a_edge * delta - eps / dx                       # multiplies rho_i
    cR = a_edge * (1.0 - delta) + eps / dx               # multiplies rho_{i+1}
    # (A* rho)_i = (J_{i+1/2} - J_{i-1/2}) / dx ; no-flux at outer edges
    main = np.zeros(G)
    lower = np.zeros(G - 1)   # sub-diagonal: coeff of rho_{i-1} in eq i
    upper = np.zeros(G - 1)   # super-diagonal: coeff of rho_{i+1} in eq i
    # interior edge contributions
    # eq i gets +J_{i+1/2}/dx (edge i, i=0..G-2 couples nodes i,i+1)
    main[:-1] += cL / dx
    upper += cR / dx
    # eq i gets -J_{i-1/2}/dx (edge i-1 couples nodes i-1,i)
    main[1:] -= cR / dx
    lower -= cL / dx
    L = sp.diags([lower, main, upper], offsets=[-1, 0, 1], format="csr")

    # ---- jump generator (conservative deposit) ----------------------------
    # Accumulate all (target_row, source_col, rate) triples into ONE COO, so
    # column j sums to zero: out-rate -lam on the diagonal, in-deposits summing
    # to +lam distributed over the bracketing landing nodes.
    idx = np.arange(G)
    rows, cols, vals = [], [], []
    for r, w in zip(shifts, weights):
        y = np.clip(x + r, lo, hi)                       # landing points (G,)
        fpos = (y - lo) / dx
        j0 = np.clip(np.floor(fpos).astype(int), 0, G - 2)
        frac = fpos - j0                                 # in [0,1]
        rate = lam * w
        # out-rate from every source node (diagonal)
        rows.append(idx); cols.append(idx); vals.append(np.full(G, -rate))
        # deposit onto bracketing nodes j0 (weight 1-frac) and j0+1 (frac)
        rows.append(j0);     cols.append(idx); vals.append(rate * (1.0 - frac))
        rows.append(j0 + 1); cols.append(idx); vals.append(rate * frac)
    Jgen = sp.coo_matrix((np.concatenate(vals),
                          (np.concatenate(rows), np.concatenate(cols))),
                         shape=(G, G)).tocsr()
    L = (L + Jgen).tocsr()

    # ---- stationary: null vector via normalisation-replaced row -----------
    A = L.tolil()
    # replace the middle row (a redundant conservation eq) with int rho dx = 1
    r0 = G // 2
    A.rows[r0] = list(range(G))
    A.data[r0] = [dx] * G
    b = np.zeros(G)
    b[r0] = 1.0
    rho = spla.spsolve(A.tocsr(), b)
    rho = np.clip(rho, 0.0, None)
    rho = rho / _trapz(rho, x)

    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(x))])
    cdf = cdf / cdf[-1]
    return x, rho, cdf


def gibbs_density(V, beta: float, lo: float = -5.2, hi: float = 5.2,
                  n_grid: int = 4001):
    """Normalised Gibbs density pi ∝ e^{-beta V} on the grid and its CDF."""
    x = np.linspace(lo, hi, n_grid)
    logp = -beta * V(x)
    rho = np.exp(logp - logp.max())
    rho = rho / _trapz(rho, x)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(x))])
    cdf = cdf / cdf[-1]
    return x, rho, cdf


def w1_from_samples(samples: np.ndarray, x_grid: np.ndarray,
                    cdf_grid: np.ndarray) -> float:
    """W1 distance between an empirical sample and a reference CDF given on a
    grid: W1 = int |F_emp(x) - F_ref(x)| dx (1D). Computed on the union grid."""
    s = np.sort(np.asarray(samples).reshape(-1))
    # empirical CDF evaluated on x_grid
    F_emp = np.searchsorted(s, x_grid, side="right") / s.size
    return float(_trapz(np.abs(F_emp - cdf_grid), x_grid))


def w1_between_cdfs(x_grid: np.ndarray, cdf_a: np.ndarray,
                    cdf_b: np.ndarray) -> float:
    """W1 = int |F_a - F_b| dx for two CDFs on a shared grid."""
    return float(_trapz(np.abs(cdf_a - cdf_b), x_grid))


def doublewell_rawcp_forensic(law, beta: float, lam: float,
                              lo: float = -5.2, hi: float = 5.2,
                              n_grid: int = 4001, q_rho: int = 64):
    """Convenience wrapper for E1 (DoubleWell1D V=(x^2-1)^2).

    `law` is the experiment's ShellJumpLaw; its quadrature_shifts(q_rho) supply
    the fine continuous-nu nodes/weights (same as the certificate's jump side).
    Returns dict with predicted and Gibbs (x, rho, cdf) arrays.
    """
    shifts_t, logw_t = law.quadrature_shifts(q_rho)
    shifts = shifts_t.detach().cpu().numpy().reshape(-1)
    weights = np.exp(logw_t.detach().cpu().numpy())

    def dV_dx(x):
        return 4.0 * x * (x * x - 1.0)

    def V(x):
        return (x * x - 1.0) ** 2

    xg, rho_pred, cdf_pred = rawcp_stationary_density(
        dV_dx, beta, lam, shifts, weights, lo, hi, n_grid)
    xg2, rho_gibbs, cdf_gibbs = gibbs_density(V, beta, lo, hi, n_grid)
    return {
        "x": xg, "rho_pred": rho_pred, "cdf_pred": cdf_pred,
        "rho_gibbs": rho_gibbs, "cdf_gibbs": cdf_gibbs,
    }


def selfcheck_lambda_to_zero(beta: float = 8.0, lo: float = -5.2, hi: float = 5.2,
                             n_grid: int = 4001, tol: float = 1e-4) -> float:
    """As lam -> 0 the raw-CP stationary law must be the Gibbs density. Returns
    the max |rho_pred - rho_gibbs| / max(rho_gibbs) (should be ~ machine/grid
    error). Uses a single dummy jump atom (its rate -> 0)."""
    def dV_dx(x):
        return 4.0 * x * (x * x - 1.0)

    def V(x):
        return (x * x - 1.0) ** 2

    shifts = np.array([2.0, -2.0])
    weights = np.array([0.5, 0.5])
    _, rho0, _ = rawcp_stationary_density(dV_dx, beta, 1e-9, shifts, weights,
                                          lo, hi, n_grid)
    xg, rho_g, _ = gibbs_density(V, beta, lo, hi, n_grid)
    err = float(np.max(np.abs(rho0 - rho_g)) / rho_g.max())
    assert err < tol, f"lambda->0 self-check failed: rel err {err:.2e} >= {tol}"
    return err
