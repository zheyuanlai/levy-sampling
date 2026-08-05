"""The official metrics, computed once at run time.

Everything a figure is allowed to quote is defined here and written to
``metrics_timeseries.csv`` by the run pipeline. Plot notebooks read those
numbers; they never redefine or recompute them. Scatter, CDF, histogram, and KDE
panels are display renderings of saved snapshots and never override a value in
the metrics table.

The comparison inputs are frozen once per experiment: one reference subsample of
the same size as a seed block, one set of projection directions, and one kernel
bandwidth chosen on the reference bank. Every method, seed, and checkpoint then
uses exactly the same settings, so a difference between two curves is a
difference between samplers rather than between metric configurations.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from . import metrics as M
from . import observables as O


def _frozen_subsample(bank: torch.Tensor, n: int) -> torch.Tensor:
    """Deterministic evenly spaced subsample of a frozen reference bank."""
    total = int(bank.shape[0])
    if total <= n:
        return bank
    index = torch.linspace(0, total - 1, int(n), device=bank.device).round().long()
    return bank[index]


def _bandwidth(reference, points: torch.Tensor) -> float:
    frozen = getattr(reference, "mmd_bandwidth", None)
    return float(frozen) if frozen else M.median_heuristic(points)


def _projections(reference, dimension: int, config: dict, device):
    frozen = getattr(reference, "sw2_projections", None)
    if frozen is not None:
        return frozen.to(device=device)
    sw2 = (config.get("metrics", {}) or {}).get("sw2", {}) or {}
    return M.make_projections(dimension,
                              int(sw2.get("n_projections", 512)),
                              int(sw2.get("projection_seed", 777)), device)


class MeasurementSuite:
    """Base class: the interface the run pipeline and stationarity rely on."""

    experiment_id = "base"

    def __init__(self, experiment, reference) -> None:
        self.experiment = experiment
        self.reference = reference
        self.target = experiment.target
        self.config = experiment.config
        self.device = experiment.device
        self._description: dict = {}

    def metrics(self, x: torch.Tensor) -> dict[str, float]:
        """Official seed-level metrics for one block of samples."""
        raise NotImplementedError

    def snapshot_arrays(self, x: torch.Tensor) -> dict[str, np.ndarray]:
        """Reduced coordinates saved alongside a sample snapshot."""
        return {}

    def labels(self, x: torch.Tensor) -> torch.Tensor:
        """Partition labels used by the stationarity diagnostic."""
        raise NotImplementedError

    def stationarity_observable(self, x: torch.Tensor) -> torch.Tensor:
        """A per-particle scalar whose autocorrelation is worth measuring."""
        raise NotImplementedError

    def describe(self) -> dict:
        return {"experiment_id": self.experiment_id, **self._description}


# ======================================================================== E1
class DoubleWellMeasurements(MeasurementSuite):
    """Exact one-dimensional comparisons against the inverse-CDF reference."""

    experiment_id = "E1"

    def __init__(self, experiment, reference) -> None:
        super().__init__(experiment, reference)
        n = experiment.particles
        self.reference_sample = _frozen_subsample(reference.sample_bank, n)
        self.bandwidth = _bandwidth(reference, self.reference_sample)
        self.grid = reference.grid
        self.target_cdf = reference.cdf
        self.reference_basin_masses = reference.basin_mass_tensor.to(self.device)
        # The reference stores moments as {"m1": ..., "m2": ...}; the order set
        # is whatever it validated, so read it off rather than assuming 1..4.
        self.moment_orders = tuple(sorted(
            int(key[1:]) for key in reference.moments if key.startswith("m")))
        self.reference_moments = torch.tensor(
            [reference.moments[f"m{order}"] for order in self.moment_orders],
            dtype=torch.float64, device=self.device)
        self._description = {
            "reference_kind": reference.kind,
            "reference_subsample_size": int(self.reference_sample.shape[0]),
            "mmd_kernel": "rbf",
            "mmd_bandwidth": self.bandwidth,
            "mmd_bandwidth_rule": "median heuristic, frozen on the reference bank",
            "primary_metrics": ["W2_exact_1d", "MMD2_biased", "KS"],
        }

    def metrics(self, x: torch.Tensor) -> dict[str, float]:
        column = x[:, 0]
        empirical = M.occupancy((column > 0).long(), 2)
        moments = torch.stack([(column ** order).mean()
                               for order in self.moment_orders])
        return {
            "W2_exact_1d": M.w2_exact_1d(x, self.reference_sample),
            "MMD2_biased": M.mmd2_biased(x, self.reference_sample,
                                         self.bandwidth),
            "MMD2_unbiased": M.mmd2_unbiased(x, self.reference_sample,
                                             self.bandwidth),
            "KS": M.ks_distance_cdf(column, self.grid, self.target_cdf),
            "CDF_L2": M.cdf_l2(column, self.grid, self.target_cdf),
            "W1_cdf": M.w1_from_cdf(column, self.grid, self.target_cdf),
            "basin_mass_left": float(empirical[0].item()),
            "basin_mass_error": M.total_variation(empirical,
                                                  self.reference_basin_masses),
            **{f"moment_{order}_error":
               float((moments[index] - self.reference_moments[index]).abs().item())
               for index, order in enumerate(self.moment_orders)},
        }

    def snapshot_arrays(self, x: torch.Tensor) -> dict[str, np.ndarray]:
        return {"position": x[:, 0].detach().cpu().numpy()}

    def labels(self, x: torch.Tensor) -> torch.Tensor:
        return (x[:, 0] > 0).long()

    def stationarity_observable(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]


# ======================================================================== E2
class MoG40Measurements(MeasurementSuite):
    """Mode coverage under the frozen hard-assignment descriptor.

    The descriptor is ``a(x) = argmax_k log N(x; mu_k, I)``. Its true masses
    ``p*_k`` are not exactly ``1/40``, so both the mode-weight divergence and the
    reference coverage line use the frozen estimates from the reference bank.
    """

    experiment_id = "E2"

    def __init__(self, experiment, reference) -> None:
        super().__init__(experiment, reference)
        n = experiment.particles
        self.reference_sample = _frozen_subsample(reference.sample_bank, n)
        self.bandwidth = _bandwidth(reference, self.reference_sample)
        self.projections = _projections(reference, 2, self.config, self.device)
        self.p_star = reference.descriptor_masses.to(self.device)
        self.n_components = int(self.p_star.shape[0])
        self.emc_star = float(reference.emc_star)
        self.occupancy_threshold = 0.5 / self.n_components
        self._description = {
            "reference_kind": reference.kind,
            "descriptor": "argmax_k log N(x; mu_k, I)",
            "descriptor_masses_are_uniform": False,
            "emc_definition": "-sum_k p_k log p_k / log K",
            "emc_star": self.emc_star,
            "emc_star_standard_error": float(
                getattr(reference, "emc_star_standard_error", float("nan"))),
            "mmd_bandwidth": self.bandwidth,
            "n_projections": int(self.projections.shape[0]),
            "primary_metrics": ["EMC", "mode_weight_JS", "per_mode_occupancy"],
        }

    def metrics(self, x: torch.Tensor) -> dict[str, float]:
        labels = self.reference.assign(x)
        p_hat = M.occupancy(labels, self.n_components)
        ratio = M.occupancy_ratio(p_hat, self.p_star)
        out = {
            "EMC": M.entropic_mode_coverage(p_hat),
            "EMC_star": self.emc_star,
            "effective_mode_fraction": M.effective_mode_fraction(p_hat),
            "mode_weight_JS": M.jensen_shannon_divergence(p_hat, self.p_star),
            "mode_weight_TV": M.total_variation(p_hat, self.p_star),
            "max_occupancy_error": M.max_absolute_error(p_hat, self.p_star),
            "mode_count_above_threshold": float(
                (p_hat > self.occupancy_threshold).sum().item()),
            "SW2": M.sliced_w2(x, self.reference_sample, self.projections),
            "MMD2_biased": M.mmd2_biased(x, self.reference_sample,
                                         self.bandwidth),
            "MMD2_unbiased": M.mmd2_unbiased(x, self.reference_sample,
                                             self.bandwidth),
        }
        counts = (p_hat * x.shape[0]).round().to(torch.int64)
        for k in range(self.n_components):
            out[f"mode_count_{k:03d}"] = int(counts[k].item())
            out[f"mode_occupancy_ratio_{k:03d}"] = float(ratio[k].item())
        return out

    def snapshot_arrays(self, x: torch.Tensor) -> dict[str, np.ndarray]:
        return {"mode_label": self.reference.assign(x).detach().cpu().numpy()}

    def labels(self, x: torch.Tensor) -> torch.Tensor:
        return self.reference.assign(x)

    def stationarity_observable(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]


# ======================================================================== E3
class MullerBrownMeasurements(MeasurementSuite):
    """Two-sample agreement on the two-dimensional LATENT collective variable.

    The CV is ``z_{1:2} = (x B^{-T})_{1:2}``. Method samples are compared with a
    frozen bank drawn from the reference grid density, so no method needs to
    supply a density of its own.
    """

    experiment_id = "E3"

    def __init__(self, experiment, reference) -> None:
        super().__init__(experiment, reference)
        n = experiment.particles
        self.reference_cv = _frozen_subsample(reference.cv_sample_bank, n)
        self.bandwidth = _bandwidth(reference, self.reference_cv)
        self.projections = _projections(reference, 2, self.config, self.device)
        self.kde_bandwidth = getattr(reference, "kde_bandwidth", None)
        self.density_grid = getattr(reference, "density_grid", None)
        self.cv_grid = getattr(reference, "cv_grid", None)
        self.basin_map = getattr(reference, "basin_map", None)
        self.reference_basin_masses = getattr(reference, "basin_mass_tensor",
                                              None)
        self._z1_grid, self._z1_cdf = self._latent_marginal_cdf()
        self._description = {
            "reference_kind": reference.kind,
            "collective_variable": "latent pair z_{1:2} = (x B^{-T})_{1:2}",
            "mmd_bandwidth": self.bandwidth,
            "mmd_bandwidth_rule": "median heuristic, frozen on the CV bank",
            "n_projections": int(self.projections.shape[0]),
            "kde_bandwidth": self.kde_bandwidth,
            "primary_metrics": ["CV_SW2", "CV_MMD2_biased"],
        }

    def _latent_marginal_cdf(self):
        values = torch.sort(self.reference_cv[:, 0]).values
        grid = torch.linspace(float(values[0]), float(values[-1]), 512,
                              dtype=torch.float64, device=self.device)
        cdf = torch.searchsorted(values.contiguous(), grid,
                                 right=True).to(torch.float64) / values.numel()
        return grid, cdf

    def _cv(self, x: torch.Tensor) -> torch.Tensor:
        return O.latent_cv(self.target, x)

    def metrics(self, x: torch.Tensor) -> dict[str, float]:
        cv = self._cv(x)
        out = {
            "CV_SW2": M.sliced_w2(cv, self.reference_cv, self.projections),
            "CV_MMD2_biased": M.mmd2_biased(cv, self.reference_cv,
                                            self.bandwidth),
            "CV_MMD2_unbiased": M.mmd2_unbiased(cv, self.reference_cv,
                                                self.bandwidth),
            # Reported as what it is: the KS distance of the latent z1 marginal.
            # It is not a ten-dimensional statement and is not main-text.
            "latent_z1_KS": M.ks_distance_cdf(cv[:, 0], self._z1_grid,
                                              self._z1_cdf),
        }
        if self.kde_bandwidth and self.density_grid is not None:
            axis_x, axis_y = self.cv_grid
            estimate = M.kde_on_grid_2d(cv, axis_x, axis_y,
                                        float(self.kde_bandwidth))
            cell_area = float((axis_x[1] - axis_x[0]) * (axis_y[1] - axis_y[0]))
            out["CV_KDE_Hellinger"] = M.squared_hellinger_grid(
                estimate, self.density_grid, cell_area)
        if self.basin_map is not None:
            labels = self.basin_map.assign(cv)
            inside = labels != O.OUTSIDE_LABEL
            out["basin_outside_mass"] = float(
                (~inside).to(torch.float64).mean().item())
            if self.reference_basin_masses is not None and bool(inside.any()):
                n_basins = int(self.reference_basin_masses.shape[0])
                p_hat = M.occupancy(labels[inside], n_basins)
                out["basin_JS"] = M.jensen_shannon_divergence(
                    p_hat, self.reference_basin_masses.to(self.device))
                out["basin_TV"] = M.total_variation(
                    p_hat, self.reference_basin_masses.to(self.device))
        return out

    def snapshot_arrays(self, x: torch.Tensor) -> dict[str, np.ndarray]:
        return {"latent_cv": self._cv(x).detach().cpu().numpy()}

    def labels(self, x: torch.Tensor) -> torch.Tensor:
        if self.basin_map is None:
            return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return self.basin_map.assign(self._cv(x))

    def stationarity_observable(self, x: torch.Tensor) -> torch.Tensor:
        return self._cv(x)[:, 0]


# ======================================================================== E4
class CoupledQuarticChainMeasurements(MeasurementSuite):
    """Static equilibrium observables of the coupled quartic chain.

    Every metric is computed from the 24-D configurations or from a quantity
    determined by them. No first-passage time, transition count, round trip,
    relay path, or kinetic transition matrix appears anywhere.
    """

    experiment_id = "E4"

    def __init__(self, experiment, reference) -> None:
        super().__init__(experiment, reference)
        n = experiment.particles
        self.reference_order_parameter = _frozen_subsample(
            reference.order_parameter_bank, n)
        self.bandwidth = _bandwidth(reference, self.reference_order_parameter)
        self.projections = _projections(reference, 2, self.config, self.device)
        self.basin_map = reference.basin_map
        self.site_minima = self.target.extras["refined_site_minima"]
        self.n_sites = int(self.target.extras["n_sites"])
        self.beta = float(self.target.beta)
        self.targets = reference.observable_targets
        self.n_phases = len(self.target.extras["phases"])
        self._description = {
            "reference_kind": reference.kind,
            "order_parameter": "m = (1/N_s) sum_i q_i",
            "susceptibility": "chi = beta N_s Cov(m)",
            "binder_convention": "O(2): U_4 = 1 - E||m||^4 / (2 (E||m||^2)^2)",
            "mmd_bandwidth": self.bandwidth,
            "n_projections": int(self.projections.shape[0]),
            "primary_metrics": [
                "phase_weight_JS", "order_parameter_SW2",
                "order_parameter_MMD2_biased", "energy_per_site_MAE",
                "susceptibility_relative_frobenius"],
            "excludes_path_event_metrics": True,
        }

    def _reference_tensor(self, key: str) -> torch.Tensor:
        return torch.as_tensor(self.targets[key], dtype=torch.float64,
                               device=self.device)

    def metrics(self, x: torch.Tensor) -> dict[str, float]:
        m = O.order_parameter(self.target, x)
        labels = O.phase_labels(m, self.basin_map)
        p_hat, outside_mass = O.phase_probabilities(labels, self.n_phases)
        energy_per_site = O.energy_per_site(self.target, x)
        total_energy = energy_per_site * self.n_sites
        coherence = O.coherence(self.target, x)
        correlation = O.two_point_correlation(self.target, x).mean(dim=0)
        kink = O.kink_density(self.target, x, self.site_minima)
        chi = O.susceptibility(m, self.beta, self.n_sites)

        p_star = self._reference_tensor("phase_probabilities")
        chi_star = self._reference_tensor("susceptibility")
        correlation_star = self._reference_tensor("two_point_correlation")
        m_star = self._reference_tensor("order_parameter_mean")

        out = {
            # main text
            "phase_weight_JS": M.jensen_shannon_divergence(p_hat, p_star),
            "order_parameter_SW2": M.sliced_w2(
                m, self.reference_order_parameter, self.projections),
            "order_parameter_MMD2_biased": M.mmd2_biased(
                m, self.reference_order_parameter, self.bandwidth),
            "energy_per_site_MAE": abs(
                float(energy_per_site.mean().item())
                - float(self.targets["energy_per_site_mean"])),
            "susceptibility_relative_frobenius": O.relative_frobenius_error(
                chi, chi_star),
            # supplement
            "order_parameter_MMD2_unbiased": M.mmd2_unbiased(
                m, self.reference_order_parameter, self.bandwidth),
            "phase_weight_TV": M.total_variation(p_hat, p_star),
            "max_phase_mass_error": M.max_absolute_error(p_hat, p_star),
            "phase_outside_mass": outside_mass,
            "energy_per_site_variance_error": abs(
                float(energy_per_site.var(unbiased=True).item())
                - float(self.targets["energy_per_site_variance"])),
            "coherence_mean_error": abs(
                float(coherence.mean().item())
                - float(self.targets["coherence_mean"])),
            "correlation_relative_L2": O.correlation_relative_l2(
                correlation, correlation_star),
            "kink_density_error": abs(
                float(kink.mean().item())
                - float(self.targets["kink_density_mean"])),
            "heat_capacity_error": abs(
                O.heat_capacity_per_site(total_energy, self.beta, self.n_sites)
                - float(self.targets["heat_capacity"])),
            "binder_cumulant_error": abs(
                O.binder_cumulant(m) - float(self.targets["binder_cumulant"])),
            "order_parameter_mean_error": float(
                (m.mean(dim=0) - m_star).norm().item()),
            "marginal_KS_mx": M.ks_distance_samples(
                m[:, 0], self.reference_order_parameter[:, 0]),
            "marginal_KS_my": M.ks_distance_samples(
                m[:, 1], self.reference_order_parameter[:, 1]),
            "marginal_KS_m_norm": M.ks_distance_samples(
                O.order_parameter_norm(m),
                O.order_parameter_norm(self.reference_order_parameter)),
        }
        for index in range(self.n_phases):
            out[f"phase_occupancy_{index}"] = float(p_hat[index].item())
        for row in range(2):
            for column in range(2):
                out[f"susceptibility_{row}{column}"] = float(chi[row, column].item())
        for lag in range(correlation.shape[0]):
            out[f"correlation_r{lag}"] = float(correlation[lag].item())
        return out

    def snapshot_arrays(self, x: torch.Tensor) -> dict[str, np.ndarray]:
        m = O.order_parameter(self.target, x)
        return {
            "order_parameter": m.detach().cpu().numpy(),
            "phase_label": O.phase_labels(m, self.basin_map).detach().cpu().numpy(),
            "energy_per_site": O.energy_per_site(
                self.target, x).detach().cpu().numpy(),
            "coherence": O.coherence(self.target, x).detach().cpu().numpy(),
            "kink_density": O.kink_density(
                self.target, x, self.site_minima).detach().cpu().numpy(),
        }

    def labels(self, x: torch.Tensor) -> torch.Tensor:
        return O.phase_labels(O.order_parameter(self.target, x), self.basin_map)

    def stationarity_observable(self, x: torch.Tensor) -> torch.Tensor:
        return O.order_parameter(self.target, x)[:, 0]


_SUITES = {
    "E1": DoubleWellMeasurements,
    "E2": MoG40Measurements,
    "E3": MullerBrownMeasurements,
    "E4": CoupledQuarticChainMeasurements,
}


def build_measurement_suite(experiment, reference) -> MeasurementSuite:
    """The official metric suite for one experiment."""
    suite_class = _SUITES.get(experiment.experiment_id)
    if suite_class is None:
        raise KeyError(
            f"no measurement suite for {experiment.experiment_id!r}")
    with experiment.target.no_count():
        return suite_class(experiment, reference)
