"""E3 reference: the CV grid density of the embedded Muller-Brown surface.

The collective variable is the LATENT pair ``z_{1:2} = (x B^{-T})_{1:2}``, never
the first two sampling coordinates. The surface is sampled in ``x = z B^T`` with
a dense ``B = Q diag(s)``, so every sampling coordinate mixes all ten latent
directions and ``x_{1:2}`` is not a collective variable at all. Everything here
goes through ``potential.collective_variable`` / ``potential.from_latent``.

The primary reference is not a sample bank but the CV grid density itself,
``p_ref(z1, z2) ~ exp(-beta V_MB(z1, z2))`` on the latent box, normalized so
``sum(p) * dA == 1``. The eight auxiliary latent coordinates factorise out of
the Boltzmann weight exactly, so the CV marginal is available in closed form up
to quadrature and no sampling error enters the reference free energy, the
reference FES, or the basin masses.

Frozen alongside it, all at seeds read from the YAML: the CV sample bank used
by the two-sample metrics, the sliced-W2 projection directions, the MMD
bandwidth, and the KDE bandwidth.
"""
from __future__ import annotations

from pathlib import Path
import math

import torch

from ..observables import GradientFlowBasinMap2D
from ..potentials import muller_brown_3well, muller_brown_3well_grad
from ..results import stable_hash
from .base import (REFERENCE_JSON, Reference, as_tensor,
                   check_positive_int, check_seed, frozen_generator,
                   import_metrics, load_npz, passed_check, read_json,
                   save_npz, write_json)

KIND = "cv_grid_density"
EXPERIMENT_ID = "E3"

DEFAULT_BANK_SEED = 424242
#: Coarse bins per axis for the bank-vs-grid consistency check. A 400k bank
#: cannot resolve a 2400^2 grid, so the two are compared after aggregation.
DEFAULT_HISTOGRAM_BINS = 60
#: Rows of the CV grid evaluated per block.
DEFAULT_ROW_CHUNK = 256

#: ``F_ref = -beta^{-1} log p_ref + C`` with ``C`` fixed by ``min F_ref = 0``.
FES_CONVENTION = (
    "F_ref(z) = -beta^{-1} log p_ref(z) + C with C chosen so that "
    "min_z F_ref(z) = 0 over the grid; equivalently F_ref = V_MB - min V_MB, "
    "since p_ref ~ exp(-beta V_MB). Units are energy, not kT.")

CV_GRID_FILE = "cv_grid.npz"
DENSITY_FILE = "density_grid.npz"
FES_FILE = "fes_grid.npz"
BASIN_MAP_FILE = "basin_map.npz"
BASIN_MASSES_FILE = "basin_masses.json"
CV_SAMPLES_FILE = "reference_cv_samples.npz"
DIAGNOSTICS_FILE = "diagnostics.json"

DEFAULT_TOLERANCES = {
    "grid_normalization_abs": 1e-12,
    "fes_consistency_abs": 1e-9,
    #: Multiples of the multinomial noise floor of the bank histogram.
    "bank_histogram_l1_noise_multiple": 3.0,
    "bank_histogram_max_abs_noise_multiple": 6.0,
}


# ============================================================== grid geometry
class _CVGrid:
    """Cell-centred uniform grid on the latent CV box."""

    def __init__(self, lo, hi, shape, device) -> None:
        self.lo = torch.as_tensor(lo, dtype=torch.float64, device=device)
        self.hi = torch.as_tensor(hi, dtype=torch.float64, device=device)
        self.shape = (int(shape[0]), int(shape[1]))
        span = self.hi - self.lo
        self.cell = span / torch.tensor(self.shape, dtype=torch.float64,
                                        device=device)
        offsets = [
            (torch.arange(self.shape[axis], dtype=torch.float64, device=device)
             + 0.5) * self.cell[axis] + self.lo[axis] for axis in (0, 1)]
        self.axis_1, self.axis_2 = offsets
        self.cell_area = float((self.cell[0] * self.cell[1]).item())


def _potential_grid(grid: _CVGrid, row_chunk: int) -> torch.Tensor:
    """``V_MB`` at every cell centre, evaluated in blocks of rows."""
    nx, ny = grid.shape
    values = torch.empty(nx, ny, dtype=torch.float64,
                         device=grid.axis_1.device)
    for start in range(0, nx, int(row_chunk)):
        rows = grid.axis_1[start:start + int(row_chunk)]
        points = torch.stack(
            torch.meshgrid(rows, grid.axis_2, indexing="ij"), dim=-1)
        values[start:start + int(row_chunk)] = muller_brown_3well(points)
    return values


# ================================================================= the class
class MullerBrownReference(Reference):
    """CV grid-density reference for the embedded Muller-Brown surface.

    Frozen at construction: the CV axis grids, the normalized CV density, the
    reference FES, the gradient-flow basin map and its basin masses, the CV
    sample bank, the sliced-W2 projections, and the MMD and KDE bandwidths.
    """

    kind = KIND
    experiment_id = EXPERIMENT_ID

    def __init__(self, *, target, provenance: dict, axis_1: torch.Tensor,
                 axis_2: torch.Tensor, density_grid: torch.Tensor,
                 fes_grid: torch.Tensor, cv_sample_bank: torch.Tensor,
                 basin_map: GradientFlowBasinMap2D,
                 basin_mass_tensor: torch.Tensor, measured: dict,
                 validation: dict) -> None:
        self.target = target
        self.potential = target.potential
        self.beta = float(target.beta)
        self.device = density_grid.device
        self._provenance = dict(provenance)
        self._measured = dict(measured)
        self.validation = dict(validation)
        self.axis_1 = axis_1
        self.axis_2 = axis_2
        self.cv_grid = (axis_1, axis_2)
        self.density_grid = density_grid
        self.fes_grid = fes_grid
        self.cv_sample_bank = cv_sample_bank
        self.basin_map = basin_map
        self.basin_labels = list(measured["basin_labels"])
        self.basin_mass_tensor = basin_mass_tensor
        self.basin_masses = dict(measured["basin_masses"])
        self.latent_lo = torch.as_tensor(provenance["latent_lo"],
                                         dtype=torch.float64, device=self.device)
        self.latent_hi = torch.as_tensor(provenance["latent_hi"],
                                         dtype=torch.float64, device=self.device)
        self.grid_shape = tuple(int(value) for value in provenance["grid_shape"])
        self.cell_area = float(measured["cell_area"])
        self.aux_std = float(measured["aux_std"])
        self.mmd_bandwidth = float(measured["mmd_bandwidth"])
        self.kde_bandwidth = (None if measured["kde_bandwidth"] is None
                              else float(measured["kde_bandwidth"]))
        self.fes_convention = FES_CONVENTION
        (make_projections,) = import_metrics("make_projections")
        self.sw2_projections = make_projections(
            2, provenance["sw2_n_projections"], provenance["sw2_seed"],
            self.device)
        self._cell_cdf: torch.Tensor | None = None
        self._grid_cache: _CVGrid | None = None

    # -- construction ------------------------------------------------------
    @staticmethod
    def provenance_for(config: dict, target) -> dict:
        block = config["reference"]
        bounds = block["latent_bounds"]
        lo = [float(value) for value in bounds[0]]
        hi = [float(value) for value in bounds[1]]
        if len(lo) != 2 or len(hi) != 2 or not all(h > l for l, h in zip(lo, hi)):
            raise ValueError(
                f"reference.latent_bounds must be [[lo1, lo2], [hi1, hi2]] with "
                f"hi > lo, got {bounds!r}")
        shape = [check_positive_int(value, "reference.grid_shape entry")
                 for value in block["grid_shape"]]
        basin = dict(block.get("basin_map") or {})
        metrics_block = config.get("metrics") or {}
        mmd = dict(metrics_block.get("mmd") or {})
        sw2 = dict(metrics_block.get("sw2") or {})
        kde = dict(metrics_block.get("kde_hellinger") or {})
        validation = dict(block.get("validation") or {})
        tolerances = {**DEFAULT_TOLERANCES,
                      **(validation.get("tolerances") or {})}
        return {
            "experiment_id": EXPERIMENT_ID,
            "kind": KIND,
            "potential": target.potential.name,
            "beta": float(target.beta),
            "dimension": int(target.d),
            "collective_variable": "latent_pair_z12",
            "sigma_aux": float(target.potential.sigma_aux),
            "latent_lo": lo,
            "latent_hi": hi,
            "grid_shape": shape,
            "sample_bank_size": check_positive_int(
                block["sample_bank_size"], "reference.sample_bank_size"),
            "bank_seed": check_seed(block.get("bank_seed", DEFAULT_BANK_SEED),
                                    "reference.bank_seed"),
            "row_chunk": check_positive_int(
                block.get("row_chunk", DEFAULT_ROW_CHUNK),
                "reference.row_chunk"),
            "basin_map": {
                "n_grid": check_positive_int(basin.get("n_grid", 600),
                                             "basin_map.n_grid"),
                "flow_steps": check_positive_int(basin.get("flow_steps", 40000),
                                                 "basin_map.flow_steps"),
                "flow_dt": float(basin.get("flow_dt", 1.5e-4)),
                "mass_n_quad": check_positive_int(
                    basin.get("mass_n_quad", 1200), "basin_map.mass_n_quad"),
            },
            "sw2_n_projections": check_positive_int(
                sw2.get("n_projections", 512), "metrics.sw2.n_projections"),
            "sw2_seed": check_seed(sw2.get("projection_seed", 777),
                                   "metrics.sw2.projection_seed"),
            "mmd_bandwidth_points": int(
                mmd.get("bandwidth_reference_points", 4096)),
            "mmd_bandwidth_seed": check_seed(mmd.get("bandwidth_seed", 99),
                                             "metrics.mmd.bandwidth_seed"),
            "kde_enabled": bool(kde.get("enabled", False)),
            "kde_bandwidth_rule": str(kde.get("bandwidth_rule",
                                              "silverman_on_reference_bank")),
            "validation": {
                "histogram_bins": check_positive_int(
                    validation.get("histogram_bins", DEFAULT_HISTOGRAM_BINS),
                    "reference.validation.histogram_bins"),
                "tolerances": {key: float(value)
                               for key, value in sorted(tolerances.items())},
            },
        }

    @classmethod
    def build(cls, config: dict, target, directory: Path, *, device=None,
              verbose: bool = False) -> "MullerBrownReference":
        if directory is None:
            raise ValueError(
                "the E3 reference caches its basin map into the reference "
                "directory, so a directory is required at build time")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        provenance = cls.provenance_for(config, target)
        (median_heuristic,) = import_metrics("median_heuristic")
        beta = float(target.beta)

        with target.no_count():
            grid = _CVGrid(provenance["latent_lo"], provenance["latent_hi"],
                           provenance["grid_shape"], target.device)
            if verbose:
                print(f"[E3] CV grid {provenance['grid_shape']}")
            potential_grid = _potential_grid(grid, provenance["row_chunk"])
            shifted = potential_grid - potential_grid.min()
            unnormalized = torch.exp(-beta * shifted)
            mass = unnormalized.sum() * grid.cell_area
            density_grid = unnormalized / mass
            # F_ref is taken from the log density, not from log(density_grid):
            # V_MB grows to ~5e3 in the corner of the CV box, where exp(-beta V)
            # underflows and a log of the stored density would report a spurious
            # plateau at the float64 underflow level instead of the true barrier.
            log_density = -beta * shifted - torch.log(mass)
            free_energy = -(1.0 / beta) * log_density
            fes_grid = free_energy - free_energy.min()

            if verbose:
                print("[E3] basin map")
            basin_map = _build_basin_map(target, provenance,
                                         directory / BASIN_MAP_FILE)
            basin_mass_tensor = _basin_masses(target, provenance,
                                              directory / BASIN_MAP_FILE, beta)

            generator = frozen_generator(target.device, provenance["bank_seed"])
            cell_cdf = _cell_cdf(density_grid)
            cv_sample_bank = _sample_cv(grid, cell_cdf,
                                        provenance["sample_bank_size"],
                                        generator)
            bandwidth = float(median_heuristic(
                cv_sample_bank, max_points=provenance["mmd_bandwidth_points"],
                seed=provenance["mmd_bandwidth_seed"]))
            kde_bandwidth = (_silverman_bandwidth(cv_sample_bank)
                             if provenance["kde_enabled"] else None)
            if verbose:
                print("[E3] validating")
            validation = _validate(provenance, beta, grid, density_grid,
                                   fes_grid, potential_grid, cv_sample_bank,
                                   mass)

        labels = list(target.extras.get("basin_labels", ["A", "B", "C"]))
        masses = basin_mass_tensor.detach().cpu().tolist()
        measured = {
            "basin_labels": labels,
            "basin_masses": dict(zip(labels, masses)),
            "basin_map": _cache_content(basin_map),
            "cell_area": grid.cell_area,
            "cell_size": [float(value) for value in
                          grid.cell.detach().cpu().tolist()],
            "aux_std": float(target.potential.sigma_aux) / math.sqrt(beta),
            "aux_std_formula": (
                "sigma_aux / sqrt(beta): the Boltzmann marginal of the "
                "auxiliary latent block exp(-beta ||z_aux||^2 / (2 sigma_aux^2)) "
                "is N(0, sigma_aux^2 / beta), not N(0, sigma_aux^2)"),
            "n_auxiliary": int(target.d) - 2,
            "fes_convention": FES_CONVENTION,
            "fes_max": float(fes_grid.max().item()),
            "potential_min": float(potential_grid.min().item()),
            "potential_max": float(potential_grid.max().item()),
            "mmd_bandwidth": bandwidth,
            "mmd_bandwidth_rule": "median_heuristic_on_frozen_cv_bank",
            "kde_bandwidth": kde_bandwidth,
            "kde_bandwidth_rule": (provenance["kde_bandwidth_rule"]
                                   if provenance["kde_enabled"] else None),
            "cv_sample_bank_shape": list(cv_sample_bank.shape),
            "build_device": str(target.device),
        }
        return cls(target=target, provenance=provenance, axis_1=grid.axis_1,
                   axis_2=grid.axis_2, density_grid=density_grid,
                   fes_grid=fes_grid, cv_sample_bank=cv_sample_bank,
                   basin_map=basin_map, basin_mass_tensor=basin_mass_tensor,
                   measured=measured, validation=validation)

    # -- contract ----------------------------------------------------------
    def describe(self) -> dict:
        return {
            "experiment_id": EXPERIMENT_ID,
            "kind": KIND,
            "provenance": dict(self._provenance),
            "provenance_hash": stable_hash(self._provenance),
            **self._measured,
            "validation": dict(self.validation),
        }

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_npz(directory / CV_GRID_FILE, axis_1=self.axis_1,
                 axis_2=self.axis_2, latent_lo=self.latent_lo,
                 latent_hi=self.latent_hi,
                 cell_area=torch.tensor(self.cell_area, dtype=torch.float64))
        save_npz(directory / DENSITY_FILE, density_grid=self.density_grid)
        save_npz(directory / FES_FILE, fes_grid=self.fes_grid)
        save_npz(directory / CV_SAMPLES_FILE,
                 cv_sample_bank=self.cv_sample_bank)
        write_json(directory / BASIN_MASSES_FILE, {
            "labels": self.basin_labels,
            "basin_masses": self.basin_masses,
            "n_quad": self._provenance["basin_map"]["mass_n_quad"],
            "definition": ("gradient-flow basin masses of exp(-beta V_MB) over "
                           "the CV box, conditional on the box"),
        })
        write_json(directory / DIAGNOSTICS_FILE, {
            "basin_map_cache": self.basin_map.cache_provenance(),
            "cell_area": self.cell_area,
            "aux_std": self.aux_std,
            "aux_std_formula": self._measured["aux_std_formula"],
            "fes_convention": FES_CONVENTION,
            "mmd_bandwidth": self.mmd_bandwidth,
            "kde_bandwidth": self.kde_bandwidth,
            "validation": self.validation,
        })
        # basin_map.npz is written by GradientFlowBasinMap2D at build time.
        self.write_describe(directory)

    @classmethod
    def load(cls, directory: Path, target, device) -> "MullerBrownReference":
        directory = Path(directory)
        payload = read_json(directory / REFERENCE_JSON)
        provenance = payload["provenance"]
        grid_arrays = load_npz(directory / CV_GRID_FILE)
        density = load_npz(directory / DENSITY_FILE)["density_grid"]
        fes = load_npz(directory / FES_FILE)["fes_grid"]
        bank = load_npz(directory / CV_SAMPLES_FILE)["cv_sample_bank"]
        measured = {key: value for key, value in payload.items()
                    if key not in ("provenance", "provenance_hash",
                                   "validation", "experiment_id", "kind")}
        basin_map = _build_basin_map(target, provenance,
                                     directory / BASIN_MAP_FILE, device=device)
        labels = list(measured["basin_labels"])
        basin_mass_tensor = torch.tensor(
            [measured["basin_masses"][label] for label in labels],
            dtype=torch.float64, device=torch.device(device))
        return cls(target=target, provenance=provenance,
                   axis_1=as_tensor(grid_arrays["axis_1"], device),
                   axis_2=as_tensor(grid_arrays["axis_2"], device),
                   density_grid=as_tensor(density, device),
                   fes_grid=as_tensor(fes, device),
                   cv_sample_bank=as_tensor(bank, device),
                   basin_map=basin_map,
                   basin_mass_tensor=basin_mass_tensor,
                   measured=measured, validation=payload["validation"])

    # -- sampling ----------------------------------------------------------
    @property
    def cell_cdf(self) -> torch.Tensor:
        if self._cell_cdf is None:
            self._cell_cdf = _cell_cdf(self.density_grid)
        return self._cell_cdf

    def _grid(self) -> _CVGrid:
        if self._grid_cache is None:
            self._grid_cache = _CVGrid(self._provenance["latent_lo"],
                                       self._provenance["latent_hi"],
                                       self._provenance["grid_shape"],
                                       self.device)
        return self._grid_cache

    def sample_cv(self, n: int, generator: torch.Generator) -> torch.Tensor:
        """``(n, 2)`` latent CV draws from the normalized grid density."""
        with self.target.no_count():
            return _sample_cv(self._grid(), self.cell_cdf,
                              check_positive_int(n, "n"), generator)

    def sample(self, n: int, generator: torch.Generator) -> torch.Tensor:
        """``(n, 10)`` draws in SAMPLING coordinates ``x = z B^T``.

        The CV pair comes from the grid density; the eight auxiliary latent
        coordinates are exact Gaussian draws of the factorised Boltzmann
        marginal ``N(0, sigma_aux^2 / beta)``.
        """
        n = check_positive_int(n, "n")
        with self.target.no_count():
            cv = _sample_cv(self._grid(), self.cell_cdf, n, generator)
            auxiliary = self.aux_std * torch.randn(
                n, int(self.target.d) - 2, generator=generator,
                device=self.device, dtype=torch.float64)
            return self.potential.from_latent(torch.cat([cv, auxiliary], dim=1))

    def collective_variable(self, x: torch.Tensor) -> torch.Tensor:
        """The CV of sampling-coordinate points, shape ``(..., 2)``."""
        with self.target.no_count():
            return self.potential.collective_variable(x)


# ================================================================== internals
def _cache_content(basin_map: GradientFlowBasinMap2D) -> dict:
    """The content-determining part of the basin cache provenance.

    The absolute path, the file sha256, and the "created vs loaded" status all
    change between an equivalent fresh build and a rebuild over an existing
    cache -- a ``.npz`` is a zip and carries write timestamps -- so they are
    kept out of ``describe()`` and hence out of ``reference_hash``. They are
    still written verbatim to ``diagnostics.json``.
    """
    provenance = basin_map.cache_provenance()
    return {key: provenance[key] for key in
            ("cache_schema_version", "n_grid", "lo", "hi", "minima", "dt_flow",
             "n_flow")}


def _build_basin_map(target, provenance: dict, cache: Path,
                     device=None) -> GradientFlowBasinMap2D:
    settings = provenance["basin_map"]
    minima = target.extras["latent_minima_stack"]
    return GradientFlowBasinMap2D(
        muller_brown_3well_grad, minima, provenance["latent_lo"],
        provenance["latent_hi"], n_grid=settings["n_grid"],
        device=target.device if device is None else device, cache=str(cache),
        dt_flow=settings["flow_dt"], n_flow=settings["flow_steps"])


def _basin_masses(target, provenance: dict, cache: Path,
                  beta: float) -> torch.Tensor:
    """Basin masses of ``exp(-beta V_MB)``, computed on the host.

    ``GradientFlowBasinMap2D.p_star`` accumulates the quadrature with
    ``scatter_add_``, whose CUDA float atomics sum in a nondeterministic order.
    That moves the masses in the last few bits between otherwise identical
    builds, and those masses are hashed into ``reference_hash``. The quadrature
    is therefore run against a host-side view of the same cached labels, which
    also makes the frozen masses identical on CPU and GPU.
    """
    host_map = _build_basin_map(target, provenance, cache, device="cpu")
    masses = host_map.p_star(lambda z: -beta * muller_brown_3well(z),
                             n_quad=provenance["basin_map"]["mass_n_quad"])
    return masses.to(target.device)


def _cell_cdf(density_grid: torch.Tensor) -> torch.Tensor:
    """Normalized cumulative cell weights, flattened row-major."""
    flat = density_grid.reshape(-1)
    cumulative = torch.cumsum(flat, dim=0)
    cumulative = cumulative / cumulative[-1].clone()
    # The CUDA scan reassociates the additions, so the cumulative weights of a
    # nonnegative density can dip by an ulp; searchsorted needs them sorted.
    cumulative = torch.cummax(cumulative, dim=0).values.clamp(0.0, 1.0)
    cumulative[-1] = 1.0
    return cumulative


def _sample_cv(grid: _CVGrid, cell_cdf: torch.Tensor, n: int,
               generator: torch.Generator) -> torch.Tensor:
    """Categorical over cells, then uniform inside the chosen cell."""
    device = cell_cdf.device
    u = torch.rand(n, generator=generator, device=device, dtype=torch.float64)
    flat = torch.clamp(torch.searchsorted(cell_cdf, u), max=cell_cdf.numel() - 1)
    ny = grid.shape[1]
    i, j = torch.div(flat, ny, rounding_mode="floor"), flat % ny
    jitter = (torch.rand(n, 2, generator=generator, device=device,
                         dtype=torch.float64) - 0.5)
    centres = torch.stack([grid.axis_1[i], grid.axis_2[j]], dim=1)
    return centres + jitter * grid.cell


def _silverman_bandwidth(points: torch.Tensor) -> float:
    """Silverman's rule in ``d`` dimensions on an isotropic scale.

    ``h = sigma * (4 / ((d + 2) n))^{1 / (d + 4)}`` with
    ``sigma = sqrt(mean_j Var[z_j])``, the isotropic scale matching the
    isotropic Gaussian kernel used by the KDE.
    """
    n, d = points.shape
    sigma = float(points.var(dim=0, unbiased=True).mean().sqrt().item())
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("degenerate CV bank: zero variance")
    return sigma * (4.0 / ((d + 2) * n)) ** (1.0 / (d + 4))


def _largest_divisor_at_most(n: int, cap: int) -> int:
    for candidate in range(min(int(n), int(cap)), 0, -1):
        if int(n) % candidate == 0:
            return candidate
    return 1


def _histogram_bin_index(points: torch.Tensor, lo: torch.Tensor,
                         hi: torch.Tensor, bins) -> torch.Tensor:
    counts = torch.as_tensor(bins, dtype=torch.float64, device=points.device)
    fraction = (points - lo) / (hi - lo)
    ij = torch.clamp((fraction * counts).long(), torch.zeros_like(counts).long(),
                     counts.long() - 1)
    return ij[:, 0] * int(bins[1]) + ij[:, 1]


def _validate(provenance: dict, beta: float, grid: _CVGrid,
              density_grid: torch.Tensor, fes_grid: torch.Tensor,
              potential_grid: torch.Tensor, cv_sample_bank: torch.Tensor,
              mass: torch.Tensor) -> dict:
    tolerances = provenance["validation"]["tolerances"]
    bins = provenance["validation"]["histogram_bins"]
    checks: list[dict] = []

    normalization_error = abs(
        float((density_grid.sum() * grid.cell_area).item()) - 1.0)
    checks.append(passed_check(
        "grid_normalization", normalization_error,
        tolerances["grid_normalization_abs"],
        statistic="|sum(p_ref) * dA - 1|",
        extra={"unnormalized_mass": float(mass.item())}))

    # F_ref must be V_MB up to an additive constant, checked against the stored
    # density where that density is representable. Cells whose Boltzmann weight
    # underflows carry no information about the normalization.
    offset = fes_grid - (potential_grid - potential_grid.min())
    fes_error = float((offset - offset.mean()).abs().max().item())
    # Only cells whose density is a normal float can round-trip through a log:
    # in the subnormal band the stored value keeps a handful of bits, so its
    # logarithm is uncertain by up to ~1e-2 in energy units for reasons that
    # have nothing to do with the normalization being checked.
    representable = density_grid >= torch.finfo(torch.float64).tiny
    recovered = -(1.0 / beta) * torch.log(density_grid[representable])
    density_offset = fes_grid[representable] - recovered
    density_error = float(
        (density_offset - density_offset.mean()).abs().max().item())
    checks.append(passed_check(
        "fes_consistency", max(fes_error, density_error),
        tolerances["fes_consistency_abs"],
        statistic=("max spread of F_ref - (V_MB - min V_MB) and of "
                   "F_ref + beta^{-1} log p_ref over representable cells"),
        extra={"potential_offset_spread": fes_error,
               "density_offset_spread": density_error,
               "mean_offset": float(offset.mean().item()),
               "normal_float_cells": int(representable.sum().item()),
               "nonzero_cells": int((density_grid > 0).sum().item()),
               "total_cells": int(representable.numel()),
               "convention": FES_CONVENTION}))

    # Bank-vs-grid consistency, judged against the multinomial noise floor of
    # a finite bank rather than against an absolute number. The coarse bins are
    # forced to divide the fine grid so that every fine cell lies wholly inside
    # one coarse bin; otherwise a cell straddling a coarse boundary donates all
    # its grid mass to one side while its samples split across both, and the
    # comparison measures that misalignment instead of the bank.
    nx, ny = grid.shape
    bins_x = _largest_divisor_at_most(nx, bins)
    bins_y = _largest_divisor_at_most(ny, bins)
    n_bank = int(cv_sample_bank.shape[0])
    grid_mass = (density_grid * grid.cell_area).reshape(
        bins_x, nx // bins_x, bins_y, ny // bins_y).sum(dim=(1, 3)).reshape(-1)
    bank_index = _histogram_bin_index(cv_sample_bank, grid.lo, grid.hi,
                                      (bins_x, bins_y))
    bank_mass = torch.bincount(bank_index, minlength=bins_x * bins_y).to(
        torch.float64) / float(n_bank)
    difference = (bank_mass - grid_mass).abs()
    max_abs = float(difference.max().item())
    l1 = float(difference.sum().item())
    per_bin_noise = torch.sqrt(grid_mass * (1.0 - grid_mass) / float(n_bank))
    l1_floor = float((math.sqrt(2.0 / math.pi) * per_bin_noise.sum()).item())
    max_floor = float(per_bin_noise.max().item())
    checks += [
        passed_check(
            "bank_histogram_l1", l1,
            tolerances["bank_histogram_l1_noise_multiple"] * l1_floor,
            statistic="sum_b |p_bank(b) - p_grid(b)|",
            extra={"bins_requested": bins, "bins": [bins_x, bins_y],
                   "n_bank": n_bank,
                   "multinomial_l1_noise_floor": l1_floor,
                   "ratio_to_noise_floor": l1 / l1_floor if l1_floor else
                   float("inf")}),
        passed_check(
            "bank_histogram_max_abs", max_abs,
            tolerances["bank_histogram_max_abs_noise_multiple"] * max_floor,
            statistic="max_b |p_bank(b) - p_grid(b)|",
            extra={"multinomial_max_noise_floor": max_floor,
                   "ratio_to_noise_floor": max_abs / max_floor if max_floor
                   else float("inf")}),
    ]
    return {"checks": checks,
            "validated": all(check["passed"] for check in checks)}


def build_reference(config: dict, target, directory: Path, *, device=None,
                    verbose: bool = False) -> MullerBrownReference:
    """Entry point named by ``reference.builder`` in ``configs/E3.yaml``."""
    return MullerBrownReference.build(config, target, directory, device=device,
                                      verbose=verbose)
