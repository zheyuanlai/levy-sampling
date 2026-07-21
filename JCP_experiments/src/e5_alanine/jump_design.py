"""P5: torsion jump-atom construction + forbidden-chord drop rule (S1.9).

Every atom is a PURE-TORSION shift in the whitened internal coordinate q_tilde:
nonzero only in the phi and psi slots, zero in every bond/angle slot.  Two
consequences matter:

  * the Lévy-score chord is Jacobian-free (S1.6, verified machine-zero in P3),
    so the score integrand is the physical energy difference;
  * the whitening leaves torsions untouched (D = 1 there), so an atom is the
    same vector in q and q_tilde and reads directly as a (dphi, dpsi) rotation.

Candidates are the torus-minimal displacements between every ordered pair of FES
basins (a complete graph, so both homotopy directions -- the +- pairs -- are
present by construction).  Each candidate is then SCREENED: an atom whose chord
crosses a forbidden Ramachandran region drives the score integrand exponent
beta[U(q) - U(q - theta r)] past the M_MAX taming cap.  Following E3's dropped
direct A-C atom and E4's dropped diagonals, we reject any atom that saturates
M_MAX for more than ``frac_tol`` of the (source-basin state, theta) samples, and
log every retained/dropped atom with its reason.  Basin pairs left unconnected by
the screen are reached by relay through an intermediate basin, exactly as E3
relays through its middle hub.
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import M_MAX
from ..jumps import ShellJumpLaw, gauss_legendre_01


def wrap_to_pi(t):
    """Wrap angle(s) to (-pi, pi]."""
    if isinstance(t, torch.Tensor):
        return -(torch.remainder(np.pi - t, 2.0 * np.pi) - np.pi)
    return -(np.remainder(np.pi - t, 2.0 * np.pi) - np.pi)


def torsion_atom(pot, dphi, dpsi) -> torch.Tensor:
    """A pure-torsion shift vector in whitened internal coordinates."""
    r = torch.zeros(pot.d, dtype=torch.float64, device=pot.D.device)
    r[pot.phi_slot] = dphi
    r[pot.psi_slot] = dpsi
    return r


def candidate_atoms(pot, minima: torch.Tensor):
    """Torus-minimal (dphi, dpsi) atoms for every ordered basin pair."""
    K = minima.shape[0]
    atoms, meta = [], []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            d = wrap_to_pi(minima[j] - minima[i])
            atoms.append(torsion_atom(pot, float(d[0]), float(d[1])))
            meta.append({"src": i, "dst": j,
                         "dphi_deg": float(np.degrees(float(d[0]))),
                         "dpsi_deg": float(np.degrees(float(d[1])))})
    return torch.stack(atoms), meta


def screen_atom(pot, states: torch.Tensor, r: torch.Tensor, *,
                q_theta: int = 16, m_max: float = M_MAX) -> dict:
    """Chord diagnostics for one atom over source-basin states.

    Returns the fraction of (state, theta) samples whose score-integrand exponent
    beta[U(q) - U(q - theta r)] exceeds the M_MAX taming cap, plus the extreme
    exponents and the jump-landing energy change.
    """
    theta, _ = gauss_legendre_01(q_theta, states.device)
    with pot.no_count():
        U0 = pot._V_raw(states)                                     # (N,)
        chord = states.unsqueeze(1) - theta.view(1, -1, 1) * r      # (N, Qt, d)
        Uc = pot._V_raw(chord)                                      # (N, Qt)
        Ujump = pot._V_raw(states + r)                              # (N,)
    expo = pot.beta * (U0.unsqueeze(1) - Uc)                        # (N, Qt)
    finite = torch.isfinite(expo)
    frac = float((expo > m_max).to(torch.float64).mean().item())
    return {
        "frac_saturating_M_MAX": frac,
        "max_exponent": float(expo[finite].max().item()) if bool(finite.any()) else float("inf"),
        "min_exponent": float(expo[finite].min().item()) if bool(finite.any()) else float("-inf"),
        "median_chord_rise_kT": float(
            (pot.beta * (Uc - U0.unsqueeze(1))).median().item()),
        "median_jump_dU_kT": float((pot.beta * (Ujump - U0)).median().item()),
        "nonfinite_fraction": float((~finite).to(torch.float64).mean().item()),
    }


def _states_in_basin(ref, basin: int, n: int, gen: torch.Generator):
    """Weighted draw of reference states whose basin label is `basin`."""
    mask = ref.labels == basin
    if int(mask.sum()) == 0:
        return None
    w = ref.weights.clone()
    w[~mask] = 0.0
    if float(w.sum()) <= 0.0:
        return None
    idx = torch.multinomial(w / w.sum(), n, replacement=True, generator=gen)
    return ref.qt[idx]


def design_jump_law(pot, ref, *, n_states: int = 128, q_theta: int = 16,
                    frac_tol: float = 1e-3, seed: int = 20260721,
                    h_frac: float = 0.1, m_max: float = M_MAX):
    """Build the screened shell jump law and a full design record.

    Returns (law, record) where ``record`` logs every retained and dropped atom
    with its geography and the reason, plus the relay atoms added to reconnect
    basin pairs whose direct atom was dropped.
    """
    gen = torch.Generator(device=pot.D.device)
    gen.manual_seed(seed)
    atoms, meta = candidate_atoms(pot, ref.minima)

    retained, dropped = [], []
    for a in range(atoms.shape[0]):
        info = dict(meta[a])
        states = _states_in_basin(ref, info["src"], n_states, gen)
        if states is None:
            info["reason"] = "source basin carries no reference mass"
            dropped.append(info)
            continue
        diag = screen_atom(pot, states, atoms[a], q_theta=q_theta, m_max=m_max)
        info.update(diag)
        if diag["nonfinite_fraction"] > 0.0:
            info["reason"] = "nonfinite chord energy"
            dropped.append(info)
        elif diag["frac_saturating_M_MAX"] > frac_tol:
            info["reason"] = (f"chord saturates M_MAX on "
                              f"{diag['frac_saturating_M_MAX']:.3g} of samples "
                              f"(> {frac_tol:g}) -- forbidden-region crossing")
            dropped.append(info)
        else:
            info["reason"] = "retained"
            retained.append((a, info))

    if not retained:
        raise RuntimeError("every candidate atom was dropped by the chord screen")

    keep_idx = [a for a, _ in retained]
    kept_atoms = atoms[keep_idx]
    # connectivity of the retained directed graph (for the relay report)
    K = int(ref.minima.shape[0])
    edges = {(i["src"], i["dst"]) for _, i in retained}
    relays = _relay_report(K, edges)

    weights = torch.full((kept_atoms.shape[0],), 1.0 / kept_atoms.shape[0],
                         dtype=torch.float64, device=kept_atoms.device)
    h = h_frac * float(kept_atoms.norm(dim=1).min().item())
    law = ShellJumpLaw(kept_atoms, weights, h=h)

    record = {
        "n_candidates": int(atoms.shape[0]),
        "n_retained": len(retained), "n_dropped": len(dropped),
        "retained": [i for _, i in retained], "dropped": dropped,
        "h": h, "cp_drift_cap": 2.0 * h,
        "min_atom_norm": float(kept_atoms.norm(dim=1).min().item()),
        "max_atom_norm": float(kept_atoms.norm(dim=1).max().item()),
        "frac_tol": frac_tol, "n_states": n_states, "q_theta": q_theta,
        "relay_pairs": relays,
        "minima_deg": np.degrees(ref.minima.cpu().numpy()).round(1).tolist(),
        "plus_minus_pairs_present": _pm_pairs_present(retained),
    }
    return law, record


def _relay_report(K: int, edges: set) -> list:
    """Ordered basin pairs with no direct retained atom, and a 2-hop relay."""
    out = []
    for i in range(K):
        for j in range(K):
            if i == j or (i, j) in edges:
                continue
            hops = [m for m in range(K)
                    if m not in (i, j) and (i, m) in edges and (m, j) in edges]
            out.append({"src": i, "dst": j, "relay_via": hops,
                        "connected": bool(hops)})
    return out


def _pm_pairs_present(retained) -> bool:
    """Both homotopy directions retained for at least one basin pair."""
    edges = {(i["src"], i["dst"]) for _, i in retained}
    return any((j, i) in edges for (i, j) in edges)


def score_direction_sanity(pot, ref, law, score, gen: torch.Generator,
                           n: int = 64) -> dict:
    """Does the Lévy score push states toward the dominant basin?

    The correction at x points from x toward the region a jump could have
    arrived FROM: S_R(x) = -lambda R (positive magnitude), so it points along
    -R, and the magnitude is largest for the atom whose chord DESCENDS into a
    deep basin.  The right sanity check is therefore whether the phi-component
    of the score agrees in sign with the signed TORUS displacement from the
    state to the deepest basin.

    Sign alone would be misleading here: for this force field the low-barrier
    phi path runs through +-180 (measured 10.9 kJ/mol) rather than through
    phi ~ 0 (32.8 kJ/mol), so from the positive-phi island the route to the
    dominant negative-phi cluster is toward INCREASING phi.
    """
    out = {}
    cvs = ref.cvs
    phi_target = float(ref.minima[ref.deepest_basin(), 0])
    for name, mask in (("neg_phi", cvs[:, 0] < 0), ("pos_phi", cvs[:, 0] >= 0)):
        w = ref.weights.clone()
        w[~mask] = 0.0
        if float(w.sum()) <= 0.0:
            out[name] = None
            continue
        idx = torch.multinomial(w / w.sum(), n, replacement=True, generator=gen)
        q = ref.qt[idx]
        # nu-AVERAGED (mixture) score: sum_a w_a S_{r_a}(x). A single random
        # atom would give direction exactly -R, whose per-particle sign is ~50/50
        # by the +- pairing; the physics is in the magnitude-weighted average,
        # which is what actually enters the drift.
        S = torch.zeros_like(q)
        for a in range(law.atoms.shape[0]):
            Ra = law.atoms[a].unsqueeze(0).expand(n, -1).contiguous()
            Sa, _ = score.score_for_shift(q, Ra)
            S = S + float(law.weights[a]) * Sa
        s_phi = S[:, pot.phi_slot]
        # signed torus displacement from each state to the deepest basin
        disp = wrap_to_pi(phi_target - cvs[idx, 0])
        agree = ((torch.sign(s_phi) == torch.sign(disp))
                 .to(torch.float64).mean())
        out[name] = {
            "mean_score_phi": float(s_phi.mean().item()),
            "mean_disp_to_deepest": float(disp.mean().item()),
            "sign_agreement": float(agree.item()),
        }
    return out
