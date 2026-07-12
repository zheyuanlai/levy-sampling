"""Ingest a user-supplied alanine-dipeptide FES into the cached npz that AlanineFES2D consumes.

This is the bridge for Experiment B: you supply the real ff14SB + GB-implicit (300 K) result as
EITHER a converged FES grid OR a trajectory of backbone dihedrals; this writes
``alanine_fes.npz`` = {phi, psi grid; F(phi,psi); minima; kT}.

Two input modes:
  --grid  path.npz    npz with keys phi (Ng,), psi (Ng,), F (Ng,Ng) [free energy in kJ/mol or
                      kcal/mol]. Passed through (minima detected, kT attached).
  --traj  path.npy    (T, 2) array of (phi, psi) in RADIANS (or --degrees) sampled from MD/metad;
                      histogrammed on a periodic grid -> F = -kT ln p (block-averaged error saved).

Common options: --temperature (K, default 300), --units {kJ,kcal} (default kJ), --grid-n (default
72 for --traj), --minima K (default 4), --out.

Example:
  python -m experiments.jcp_sampling.scripts.ingest_alanine_fes --traj phipsi.npy \
      --temperature 300 --units kJ --out results/jcp_sampling/alanine_fes/alanine_fes.npz
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

KB_KJ = 0.00831446   # kJ/mol/K
KB_KCAL = 0.00198720  # kcal/mol/K


def kT_of(temperature: float, units: str) -> float:
    return (KB_KJ if units == "kJ" else KB_KCAL) * float(temperature)


def _periodic_local_minima(F: np.ndarray, k: int) -> np.ndarray:
    """Return the (i, j) indices of the k deepest strict periodic local minima of F."""
    Ng = F.shape[0]
    cand = []
    for i in range(Ng):
        for j in range(Ng):
            c = F[i, j]
            lo = True
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    if F[(i + di) % Ng, (j + dj) % Ng] <= c:
                        lo = False
                        break
                if not lo:
                    break
            if lo:
                cand.append((c, i, j))
    cand.sort()
    # greedily keep deepest, drop minima within ~30 deg of an already-kept one
    kept = []
    ax = -math.pi + 2 * math.pi * np.arange(Ng) / Ng

    def wrapdist(a, b):
        d = (a - b + math.pi) % (2 * math.pi) - math.pi
        return abs(d)

    for c, i, j in cand:
        pi_, pj_ = ax[i], ax[j]
        if all(wrapdist(pi_, ax[ii]) > 0.5 or wrapdist(pj_, ax[jj]) > 0.5 for _, ii, jj in kept):
            kept.append((c, i, j))
        if len(kept) >= k:
            break
    return np.array([(ax[i], ax[j]) for _, i, j in kept], dtype=np.float32)


def from_grid(path: str):
    d = np.load(path, allow_pickle=True)
    phi = np.asarray(d["phi"], float); psi = np.asarray(d["psi"], float)
    F = np.asarray(d["F"], float)
    return phi, psi, F, None


def from_traj(path: str, degrees: bool, grid_n: int, kT: float, n_blocks: int = 8):
    arr = np.load(path)
    if degrees:
        arr = np.deg2rad(arr)
    arr = (arr + math.pi) % (2 * math.pi) - math.pi
    edges = np.linspace(-math.pi, math.pi, grid_n + 1)
    ax = 0.5 * (edges[:-1] + edges[1:])

    def fes_of(chunk):
        H, _, _ = np.histogram2d(chunk[:, 0], chunk[:, 1], bins=[edges, edges])
        p = H / max(H.sum(), 1.0)
        with np.errstate(divide="ignore"):
            F = -kT * np.log(p)
        F[~np.isfinite(F)] = np.nan
        F = F - np.nanmin(F)
        return F

    F = fes_of(arr)
    # block-averaged error on F over well-sampled bins
    blocks = np.array_split(arr, n_blocks)
    Fb = np.stack([fes_of(b) for b in blocks])
    F_se = np.nanstd(Fb, axis=0) / math.sqrt(n_blocks)
    # fill unsampled bins with a high plateau so the interpolant/labels stay finite
    F = np.where(np.isfinite(F), F, np.nanmax(F) + 5.0 * kT)
    return ax, ax, F, F_se


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--grid")
    src.add_argument("--traj")
    ap.add_argument("--degrees", action="store_true")
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--units", choices=["kJ", "kcal"], default="kJ")
    ap.add_argument("--grid-n", type=int, default=72)
    ap.add_argument("--minima", type=int, default=4)
    ap.add_argument("--out", default="results/jcp_sampling/alanine_fes/alanine_fes.npz")
    args = ap.parse_args()

    kT = kT_of(args.temperature, args.units)
    F_se = None
    if args.grid:
        phi, psi, F, F_se = from_grid(args.grid)
    else:
        phi, psi, F, F_se = from_traj(args.traj, args.degrees, args.grid_n, kT)

    minima = _periodic_local_minima(np.asarray(F, float), args.minima)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(phi=phi, psi=psi, F=F, kT=kT, minima=minima,
                   temperature=args.temperature, units=args.units)
    if F_se is not None:
        payload["F_se"] = F_se
    np.savez(out, **payload)
    print(f"wrote {out}  grid {F.shape}  kT={kT:.4f} {args.units}/mol  "
          f"minima(deg)={np.round(np.rad2deg(minima), 1).tolist()}")


if __name__ == "__main__":
    main()
