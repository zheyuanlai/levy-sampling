"""P4: loader for the well-tempered metadynamics reference (FES, p_star, pool).

``E5Reference`` mirrors the role of ``references.py``'s classes for E1-E4: it
serves reference draws in the sampler's own coordinates (whitened internal
q_tilde) plus the basin partition and target masses.

The cached pool is a set of production frames sampled under the FROZEN converged
metadynamics bias, each carrying the importance weight
``w = exp(-beta ((gamma-1)/gamma) F(phi,psi))`` that converts the biased ensemble
back to the unbiased Boltzmann measure (see ``build_reference.py``).  As in E4,
weighted quantities (FES, basin masses, energy moments) use the weights directly,
while the unweighted cloud required by W2/MMD is produced by SIR resampling.

Regenerate the cache with:
    python -m src.e5_alanine.build_reference --seeds 0 1
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .build_reference import CACHE_DIR


class E5Reference:
    """Reference FES(phi,psi), basin partition, and reweighted conformer pool."""

    def __init__(self, cache_path: str | None = None,
                 device: str | torch.device = "cuda") -> None:
        self.cache_path = cache_path or os.path.join(CACHE_DIR, "reference.npz")
        if not os.path.exists(self.cache_path):
            raise FileNotFoundError(
                f"E5 reference cache {self.cache_path} missing; regenerate with "
                "`python -m src.e5_alanine.build_reference --seeds 0 1`")
        self.device = torch.device(device)
        with np.load(self.cache_path, allow_pickle=False) as data:
            def _t(k, dtype=torch.float64):
                return torch.as_tensor(data[k], dtype=dtype, device=self.device)
            self.qt = _t("qt")                         # (M, 60) whitened internal
            self.cvs = _t("cvs")                       # (M, 2) (phi, psi)
            self.U_eff = _t("U_eff")                   # (M,)
            self.weights = _t("weights")               # (M,) normalized
            self.labels = _t("labels", torch.long)     # (M,)
            self.minima = _t("minima")                 # (K, 2)
            self.p_star = _t("p_star")                 # (K,)
            self.p_star_fes = _t("p_star_fes")
            self.F_grid = _t("F_grid")                 # (G, G) indexed [phi, psi]
            self.grid_axis = _t("grid_axis")           # (G,)
            self.basin_escape_kT = _t("basin_escape_kT")   # (K,)
            self.basin_saddles_kJ = _t("basin_saddles_kJ")  # (K, K)
            self.basin_min_kJ = _t("basin_min_kJ")     # (K,)
            self.beta = float(data["beta"])
            self.kT = float(data["kT"])
            self.provenance = json.loads(str(data["provenance"]))
        self.weights = self.weights / self.weights.sum()
        self.K = int(self.minima.shape[0])
        # put the cached CVs and basin centres into the reporting windows (psi's
        # branch cut is shifted off the populated +-pi seam; see bat.py). Basin
        # assignment uses a torus metric and is unaffected, as is the FES grid
        # lookup, which is taken modulo the grid.
        from .bat import PHI_WINDOW_CENTER, PSI_WINDOW_CENTER, wrap_about
        self._windows = (PHI_WINDOW_CENTER, PSI_WINDOW_CENTER)
        for t in (self.cvs, self.minima):
            t[..., 0] = wrap_about(t[..., 0], PHI_WINDOW_CENTER)
            t[..., 1] = wrap_about(t[..., 1], PSI_WINDOW_CENTER)

    # -- effective sample size of the importance weights ---------------------
    @property
    def ess(self) -> float:
        w = self.weights
        return float((w.sum() ** 2 / (w * w).sum()).item())

    @property
    def ess_fraction(self) -> float:
        return self.ess / float(self.qt.shape[0])

    # -- draws ---------------------------------------------------------------
    def sample(self, n: int, gen: torch.Generator) -> torch.Tensor:
        """SIR draw of n reference conformers in whitened internal coords."""
        idx = torch.multinomial(self.weights, n, replacement=True, generator=gen)
        return self.qt[idx]

    def sample_cv(self, n: int, gen: torch.Generator) -> torch.Tensor:
        idx = torch.multinomial(self.weights, n, replacement=True, generator=gen)
        return self.cvs[idx]

    # -- basin partition (torus Voronoi around the FES minima) ---------------
    def assign(self, cv: torch.Tensor) -> torch.Tensor:
        d = (cv.unsqueeze(-2) - self.minima).abs()
        d = torch.minimum(d, 2.0 * np.pi - d)
        return (d * d).sum(-1).argmin(-1)

    # -- diagnostics ---------------------------------------------------------
    def seam_mass(self, margin: float = 0.15) -> float:
        """Weighted reference mass within `margin` rad of the branch cut of
        EITHER reporting window (task S2 periodicity discipline).

        Euclidean metrics on the reported CVs are valid only if this is
        negligible; psi's window is shifted precisely so that it is.
        """
        total = None
        for j, center in enumerate(self._windows):
            d = (self.cvs[:, j] - (center + np.pi)).abs()
            d = torch.minimum(d, 2.0 * np.pi - d)
            near = d < margin
            total = near if total is None else (total | near)
        return float(self.weights[total].sum().item())

    def basin_free_energies_kT(self) -> torch.Tensor:
        """-ln p_star, relative to the deepest basin (in units of kT)."""
        f = -torch.log(self.p_star)
        return f - f.min()

    # -- FES geometry --------------------------------------------------------
    def F_at(self, cv: torch.Tensor) -> torch.Tensor:
        """Nearest-grid-cell FES value (kJ/mol) at (phi, psi) points."""
        step = 2.0 * np.pi / self.grid_axis.numel()
        idx = torch.remainder(
            torch.round((cv - self.grid_axis[0]) / step).long(),
            self.grid_axis.numel())
        return self.F_grid[idx[..., 0], idx[..., 1]]

    def phi_cut_min_kJ(self, phi_cut: float = 0.0) -> float:
        """Lowest FES value on the phi = phi_cut line, above the global minimum.

        NOTE: this is a barrier ONLY if the line is a genuine dividing surface.
        At phi = +-pi it is not -- a basin sits on that line -- so use
        ``island_barrier_kJ`` for the slow event's barrier.
        """
        i = int(torch.argmin(torch.abs(
            wrap_to_pi_t(self.grid_axis - phi_cut))).item())
        return float((self.F_grid[i, :].min() - self.F_grid.min()).item())

    def island_barrier_kJ(self) -> float:
        """Escape barrier (kJ/mol) of the sparse positive-phi island basin."""
        isl = self.island_basins()
        if not isl:
            return float("nan")
        return float(min(self.basin_escape_kT[k] for k in isl) / self.beta)

    def deepest_basin(self) -> int:
        """Basin index whose FES minimum is the global minimum."""
        return int(torch.argmin(self.F_at(self.minima)).item())

    def island_basins(self) -> list:
        """Metastable basins on the positive-phi side of the phi ~ 0 barrier.

        The alanine Ramachandran map has TWO dividing lines in phi (phi ~ 0 and
        phi ~ +-pi), and the extended beta/C5 region continues across the +-pi
        seam. So "phi > 0" is not the island: it would sweep in the extended
        region, which local dynamics reaches freely. The island is the
        C7ax/alpha_L group, i.e. basins with 0 < phi < 150 deg.
        """
        hi = 150.0 * np.pi / 180.0
        return [k for k in range(self.K)
                if 0.0 < float(self.minima[k, 0]) < hi]

    # kept for backward compatibility with earlier naming
    def positive_phi_basins(self) -> list:
        return self.island_basins()

    def representative_state(self, basin: int) -> torch.Tensor:
        """Pool conformer closest (torus metric) to that basin's FES minimum."""
        d = (self.cvs - self.minima[basin]).abs()
        d = torch.minimum(d, 2.0 * np.pi - d)
        return self.qt[int((d * d).sum(-1).argmin().item())]


def wrap_to_pi_t(t: torch.Tensor) -> torch.Tensor:
    return -(torch.remainder(np.pi - t, 2.0 * np.pi) - np.pi)
