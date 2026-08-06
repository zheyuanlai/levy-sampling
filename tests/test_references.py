"""Properties the four frozen references must have.

Every reference is built here at a reduced size, through the same
``load_experiment`` -> ``ensure_reference`` path a run uses, so the objects under
test are the real ones. Only the grid, bank, chain, and bootstrap sizes differ
from production; no code path does.

Tolerances. Nothing in this file compares against a round number picked by eye.
Each tolerance is either float64 rounding at the magnitude of the quantity being
compared, or a Monte Carlo band computed in the test from the sample size and
the null distribution of the statistic, with the distribution named in a
comment. Where a reference ships its own frozen tolerance the test recomputes
the statistic independently and says which frozen number it is being held to.
"""
from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.measurements import build_measurement_suite
from src import metrics as M
from src import observables as O
from src.config import load_experiment, load_yaml
from src.observables import OUTSIDE_LABEL
from src.references import e4 as e4_module
from src.references.base import frozen_generator
from src.references.e1 import DoubleWellReference
from src.references.e4 import ReferenceValidationError

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
E4_ACCEPTANCE = (REPOSITORY_ROOT / "configs" / "experiments"
                 / "E4_reference_acceptance.yaml")

EPS = float(torch.finfo(torch.float64).eps)


# ============================================================ reduced configs
#: E1 keeps the production grid -- the quadrature costs milliseconds -- and only
#: shrinks the sample bank and the self-W2 replicate count.
E1_SMALL = {
    "protocol": {"particles": 200, "seeds": 2, "final_time": 1.0},
    "reference": {"n_grid": 200001, "sample_bank_size": 50000,
                  "validation": {"grid_sizes": [50001, 100001, 200001],
                                 "self_w2_replicates": 2}},
}

E2_SMALL = {
    "protocol": {"particles": 200, "seeds": 2, "final_time": 1.0},
    "reference": {"sample_bank_size": 200000, "metric_bank_size": 20000,
                  "emc_bootstrap_blocks": 100,
                  "emc_bootstrap_replicates": 100},
}

E3_SMALL = {
    "protocol": {"particles": 200, "seeds": 2, "final_time": 1.0},
    "reference": {"grid_shape": [400, 400], "sample_bank_size": 40000,
                  "basin_map": {"n_grid": 120, "flow_steps": 3000}},
}

E4_SMALL = {
    "protocol": {"particles": 100, "seeds": 1, "final_time": 0.5},
    "reference": {
        "pt_mala": {"n_runs": 4, "chains_per_run": 2, "n_replicas": 3,
                    "burn_in_steps": 200, "total_steps": 1200,
                    "thinning": 10},
        "snis": {"n_runs": 4, "proposals_per_run": 3000},
        "basin_map": {"n_grid": 80, "n_flow": 1500},
        "order_parameter_grid": {"n_bins": 40},
        "bootstrap": {"replicates": 60, "seed": 90210},
    },
}


def _experiment(experiment_id: str, root: Path, overrides: dict):
    return load_experiment(experiment_id, device="cpu", results_root=root,
                           overrides=overrides)


@pytest.fixture(scope="module")
def e1_reference(tmp_path_factory):
    experiment = _experiment("E1", tmp_path_factory.mktemp("e1"), E1_SMALL)
    return experiment, experiment.ensure_reference()


@pytest.fixture(scope="module")
def e2_reference(tmp_path_factory):
    experiment = _experiment("E2", tmp_path_factory.mktemp("e2"), E2_SMALL)
    return experiment, experiment.ensure_reference()


@pytest.fixture(scope="module")
def e3_reference(tmp_path_factory):
    experiment = _experiment("E3", tmp_path_factory.mktemp("e3"), E3_SMALL)
    return experiment, experiment.ensure_reference()


# ==================================================================== E1
def _cdf_max_difference(coarse, fine, points: torch.Tensor) -> float:
    return float((coarse.exact_cdf_on(points)
                  - fine.exact_cdf_on(points)).abs().max().item())


def test_the_reference_cdf_converges_as_the_quadrature_grid_is_refined(
        e1_reference, tmp_path):
    """Trapezoidal quadrature is second order, so halving the spacing must
    shrink the distance to the finest CDF -- and the stored record must say so.
    """
    experiment, reference = e1_reference
    sizes = [25001, 50001, 100001, 200001]
    built = {}
    for size in sizes:
        overrides = copy.deepcopy(E1_SMALL)
        overrides["reference"]["n_grid"] = size
        # An empty grid_sizes list skips the refinement block of the inner
        # build: this test does the refinement itself.
        overrides["reference"]["validation"] = {"grid_sizes": [],
                                                "self_w2_replicates": 0}
        built[size] = DoubleWellReference.build(
            {**experiment.config, **overrides,
             "reference": {**experiment.config["reference"],
                           **overrides["reference"]}},
            experiment.target, tmp_path / f"grid{size}")
    lo, hi = built[sizes[-1]].bounds
    points = torch.linspace(lo, hi, 20001, dtype=torch.float64)
    differences = [_cdf_max_difference(built[size], built[sizes[-1]], points)
                   for size in sizes[:-1]]
    assert differences == sorted(differences, reverse=True), differences
    assert all(later < earlier for earlier, later
               in zip(differences, differences[1:]))
    # Second order: each halving of the spacing should cut the error by ~4.
    for earlier, later in zip(differences, differences[1:]):
        assert later < 0.5 * earlier, (earlier, later)

    record = reference.validation["grid_refinement"]
    reported = {row["n_grid"]: row["cdf_max_abs_difference"]
                for row in record["per_grid"]}
    assert set(reported) == set(E1_SMALL["reference"]["validation"]["grid_sizes"])
    assert reported[record["finest"]] == 0.0
    coarser = [reported[size] for size in sorted(reported)
               if size != record["finest"]]
    assert coarser == sorted(coarser, reverse=True)
    check = next(item for item in reference.validation["checks"]
                 if item["check"] == "grid_refinement_cdf")
    assert check["passed"], check


def test_the_reference_cdf_is_monotone_and_spans_zero_to_one(e1_reference):
    _, reference = e1_reference
    assert bool((torch.diff(reference.cdf) >= 0.0).all())
    assert float(reference.cdf[0].item()) == 0.0
    assert float(reference.cdf[-1].item()) == 1.0
    assert reference.cdf.shape == reference.grid.shape


def test_inverse_cdf_draws_follow_the_reference_cdf(e1_reference):
    """One-sample KS against the exact CDF the draws were made from.

    ``sample`` inverts the piecewise-linear CDF and ``exact_cdf_on`` evaluates
    that same piecewise-linear CDF, so this is an exact null: ``sqrt(n) D_n``
    follows the Kolmogorov distribution, whose 0.999 quantile is 1.949.
    """
    _, reference = e1_reference
    n = 200_000
    draws = reference.sample(n, frozen_generator(reference.device, 12345))
    assert draws.shape == (n, 1)
    ordered = torch.sort(draws[:, 0]).values
    cdf = reference.exact_cdf_on(ordered)
    index = torch.arange(1, n + 1, dtype=torch.float64,
                         device=reference.device)
    ks = float(torch.maximum(index / n - cdf, cdf - (index - 1) / n).max())
    assert ks * math.sqrt(n) < 1.949, ks * math.sqrt(n)

def test_e1_saves_reference_floors_for_every_primary_metric(e1_reference):
    _, reference = e1_reference
    floors = reference.validation["sampling_floors"]
    assert floors["particles"] == E1_SMALL["protocol"]["particles"]
    assert floors["replicates"] == 2
    assert floors["mmd_bandwidth"] > 0.0
    for metric in ("W2_exact_1d", "MMD2_biased", "KS"):
        record = floors[metric]
        assert len(record["values"]) >= floors["replicates"]
        assert record["mean"] >= 0.0
        assert record["sd"] >= 0.0




def test_basin_masses_sum_to_one_and_the_double_well_is_symmetric(
        e1_reference):
    """``V = (x^2 - 1)^2`` is even and the grid is symmetric about 0, so the two
    basin masses must agree to quadrature roundoff, not merely to a tolerance.
    """
    _, reference = e1_reference
    masses = reference.basin_mass_tensor
    assert abs(float(masses.sum().item()) - 1.0) <= 4.0 * EPS
    left, right = (float(value) for value in masses)
    # The reference's own frozen refinement tolerance on a basin mass is 1e-8;
    # the symmetry of an even integrand on a symmetric grid is far tighter.
    assert abs(left - right) <= 64.0 * EPS, (left, right)


def test_the_e1_reference_round_trips_through_disk(e1_reference, tmp_path):
    experiment, reference = e1_reference
    directory = tmp_path / "e1-roundtrip"
    reference.save(directory)
    loaded = DoubleWellReference.load(directory, experiment.target,
                                      reference.device)
    assert loaded.describe() == reference.describe()
    for name in ("grid", "pdf", "cdf", "sample_bank"):
        assert torch.equal(getattr(loaded, name), getattr(reference, name))
    assert loaded.describe() == loaded.describe()


# ==================================================================== E2
def test_the_descriptor_is_the_argmax_of_the_component_log_density(
        e2_reference):
    """``assign`` must agree label for label with an independent argmax.

    The independent side evaluates the documented density
    ``log N(x; mu_k, I_2) = -||x - mu_k||^2 / 2 - log(2 pi)`` from distances
    computed by a different algorithm (``cdist``), so the two agree only if the
    rule really is the argmax and not something that merely resembles it.
    """
    experiment, reference = e2_reference
    generator = frozen_generator(reference.device, 20260805)
    box = 60.0 * (torch.rand(2048, 2, generator=generator, dtype=torch.float64,
                             device=reference.device) - 0.5)
    points = torch.cat([box, reference.sample(2048, generator)])
    means = experiment.target.extras["component_means"]
    log_density = -0.5 * torch.cdist(points, means) ** 2 - math.log(2.0 * math.pi)
    ordered = torch.sort(log_density, dim=1, descending=True).values
    # No point may sit on a Voronoi boundary, or rounding could flip its label.
    assert float((ordered[:, 0] - ordered[:, 1]).min().item()) > 1e-9
    assert torch.equal(reference.assign(points), log_density.argmax(dim=1))


def test_descriptor_masses_form_a_strictly_positive_distribution(e2_reference):
    _, reference = e2_reference
    p_star = reference.descriptor_masses
    assert int(p_star.shape[0]) == reference.n_components
    # p* is a vector of counts divided by one bank size, so its sum carries at
    # most K roundings of a float64 division.
    assert abs(float(p_star.sum().item()) - 1.0) <= reference.n_components * EPS
    assert bool((p_star > 0.0).all())


def test_the_reference_coverage_line_is_the_entropy_of_the_measured_masses(
        e2_reference):
    """``emc_star`` must be ``entropic_mode_coverage(p*)`` and not the value 1
    the figure would draw if the descriptor masses were assumed uniform.
    """
    _, reference = e2_reference
    p_star = reference.descriptor_masses
    assert M.entropic_mode_coverage(p_star) == reference.emc_star
    assert reference.emc_star != 1.0
    assert 0.0 < reference.emc_star < 1.0
    # The estimator really is a function of the argument, not a constant: a
    # two-category occupancy has EMC log 2 / log K exactly.
    K = reference.n_components
    half = torch.zeros(K, dtype=torch.float64, device=reference.device)
    half[0] = half[1] = 0.5
    assert M.entropic_mode_coverage(half) == pytest.approx(
        math.log(2.0) / math.log(K), rel=8.0 * EPS)
    # And a uniform occupancy is the only one that reaches 1, which p* does not.
    uniform = torch.full((K,), 1.0 / K, dtype=torch.float64,
                         device=reference.device)
    assert M.entropic_mode_coverage(uniform) == pytest.approx(1.0, abs=8.0 * EPS)
    assert M.entropic_mode_coverage(p_star) < M.entropic_mode_coverage(uniform)


def test_the_coverage_of_a_fresh_exact_draw_matches_the_reference_line(
        e2_reference):
    """A finite-sample plug-in EMC sits below ``emc_star`` by its own bias.

    The band is the reference's own like-for-like floor: the plug-in bias
    ``(K - 1) / (2 n log K)`` plus four standard deviations of the coverage of
    exact draws of the same size, both computed here rather than assumed.
    """
    _, reference = e2_reference
    n = 5000
    floor = reference.emc_at_sample_size(n, replicates=30, seed=4242)
    draw = reference.sample(n, frozen_generator(reference.device, 777))
    coverage = M.entropic_mode_coverage(
        M.occupancy(reference.assign(draw), reference.n_components))
    band = floor["plugin_bias_at_n"] + 4.0 * floor["emc_std"]
    assert abs(coverage - reference.emc_star) <= band, (coverage, band)


def test_the_mode_weights_of_a_fresh_exact_draw_agree_with_p_star(
        e2_reference):
    """``8 n JS(p_hat, p*)`` is asymptotically chi-square on ``K - 1`` degrees
    of freedom under the null that the draw came from ``p*``; the band is the
    mean plus five standard deviations of that distribution.
    """
    _, reference = e2_reference
    n = 20000
    K = reference.n_components
    draw = reference.sample(n, frozen_generator(reference.device, 20261))
    p_hat = M.occupancy(reference.assign(draw), K)
    divergence = M.jensen_shannon_divergence(p_hat, reference.descriptor_masses)
    statistic = 8.0 * n * divergence
    assert statistic <= (K - 1) + 5.0 * math.sqrt(2.0 * (K - 1)), statistic


# ==================================================================== E3
def test_the_cv_grid_density_is_normalized(e3_reference):
    """``sum(p) * dA`` must be 1 to the rounding of one blocked reduction."""
    _, reference = e3_reference
    total = float((reference.density_grid.sum()
                   * reference.cell_area).item())
    n_cells = int(reference.density_grid.numel())
    # torch reduces pairwise, so the relative error of the sum is bounded by
    # about log2(N) eps; the frozen acceptance tolerance is 1e-12, far above it.
    assert abs(total - 1.0) <= 8.0 * math.log2(n_cells) * EPS
    check = next(item for item in reference.validation["checks"]
                 if item["check"] == "grid_normalization")
    assert check["passed"] and check["tolerance"] == 1e-12


def test_the_reference_fes_is_the_muller_brown_surface_up_to_a_constant(
        e3_reference):
    """``F_ref = V_MB + C`` both against the potential and against the stored
    density, the latter only where that density is a normal float.
    """
    from src.potentials import muller_brown_3well

    _, reference = e3_reference
    points = torch.stack(torch.meshgrid(reference.axis_1, reference.axis_2,
                                        indexing="ij"), dim=-1)
    potential = muller_brown_3well(points)
    offset = reference.fes_grid - (potential - potential.min())
    spread = float((offset - offset.mean()).abs().max().item())
    # F_ref is built as -(1/beta) * (-beta * shifted) - (1/beta) log(mass), so
    # the only error is float64 rounding at the magnitude of V_MB itself.
    assert spread <= 8.0 * EPS * float(potential.max().item()), spread

    tiny = torch.finfo(torch.float64).tiny
    representable = reference.density_grid >= tiny
    assert bool(representable.any())
    recovered = -(1.0 / reference.beta) * torch.log(
        reference.density_grid[representable])
    log_offset = reference.fes_grid[representable] - recovered
    log_spread = float((log_offset - log_offset.mean()).abs().max().item())
    scale = float(reference.fes_grid[representable].abs().max().item())
    assert log_spread <= 8.0 * EPS * scale, log_spread


def test_the_cv_sample_bank_reproduces_the_grid_density(e3_reference):
    """Binned bank mass against binned grid mass, judged against the L1 noise
    floor of a multinomial of the same size: ``E|Delta_b| = sqrt(2/pi) sigma_b``.
    """
    _, reference = e3_reference
    bank = reference.cv_sample_bank
    n_bank = int(bank.shape[0])
    nx, ny = reference.grid_shape
    bins_x, bins_y = 40, 40
    assert nx % bins_x == 0 and ny % bins_y == 0, (
        "coarse bins must divide the fine grid so no fine cell straddles a "
        "coarse boundary")
    grid_mass = (reference.density_grid * reference.cell_area).reshape(
        bins_x, nx // bins_x, bins_y, ny // bins_y).sum(dim=(1, 3)).reshape(-1)
    fraction = ((bank - reference.latent_lo)
                / (reference.latent_hi - reference.latent_lo))
    counts = torch.tensor([bins_x, bins_y], dtype=torch.float64,
                          device=bank.device)
    ij = torch.clamp((fraction * counts).long(),
                     torch.zeros_like(counts).long(), counts.long() - 1)
    flat = ij[:, 0] * bins_y + ij[:, 1]
    bank_mass = torch.bincount(flat, minlength=bins_x * bins_y).to(
        torch.float64) / float(n_bank)

    l1 = float((bank_mass - grid_mass).abs().sum().item())
    sigma = torch.sqrt(grid_mass * (1.0 - grid_mass) / float(n_bank))
    l1_floor = float((math.sqrt(2.0 / math.pi) * sigma.sum()).item())
    # The frozen acceptance allows three times the noise floor.
    assert l1 <= 3.0 * l1_floor, (l1, l1_floor)


def test_sampling_coordinate_draws_carry_the_grid_cv_distribution(
        e3_reference):
    """``sample`` returns ``x = z B^T`` in R^10; its latent CV must be
    distributed like ``sample_cv``. Two-sample KS at the 0.999 level is
    ``1.949 sqrt(2/n)``.
    """
    experiment, reference = e3_reference
    n = 20000
    x = reference.sample(n, frozen_generator(reference.device, 31337))
    assert x.shape == (n, int(experiment.target.potential.d))
    assert x.shape[1] > 2
    cv = O.latent_cv(experiment.target, x)
    z = reference.sample_cv(n, frozen_generator(reference.device, 99991))
    critical = 1.949 * math.sqrt(2.0 / n)
    for axis in (0, 1):
        distance = M.ks_distance_samples(cv[:, axis], z[:, axis])
        assert distance < critical, (axis, distance, critical)


def test_the_e3_basin_masses_sum_to_one(e3_reference):
    _, reference = e3_reference
    masses = reference.basin_mass_tensor
    assert int(masses.shape[0]) == len(reference.basin_labels)
    assert abs(float(masses.sum().item()) - 1.0) <= 8.0 * EPS
    assert bool((masses > 0.0).all())


def test_the_collective_variable_is_the_latent_pair_not_the_first_two_axes(
        e3_reference):
    """``x_{1:2}`` is not a collective variable: ``B`` is dense, so the latent
    pair mixes all ten sampling coordinates. A refactor that swapped the two
    would make this difference exactly zero.
    """
    experiment, _ = e3_reference
    dimension = int(experiment.target.potential.d)
    generator = frozen_generator(experiment.device, 5150)
    x = torch.randn(256, dimension, generator=generator, dtype=torch.float64,
                    device=experiment.device)
    cv = O.latent_cv(experiment.target, x)
    assert cv.shape == (256, 2)
    relative = float(((cv - x[:, :2]).norm() / cv.norm()).item())
    assert relative > 0.5, relative


# ==================================================================== E4
@pytest.fixture(scope="module")
def e4_experiment(tmp_path_factory):
    """The E4 target and jump law only; no reference is built here."""
    return _experiment("E4", tmp_path_factory.mktemp("e4-target"), E4_SMALL)


@pytest.fixture(scope="module")
def e4_configurations(e4_experiment):
    """Random 24-D configurations spanning several phases."""
    target = e4_experiment.target
    generator = frozen_generator(target.device, 8675309)
    coherent = target.extras["coherent_states"]
    index = torch.randint(0, coherent.shape[0], (64,), generator=generator,
                          device=target.device)
    jitter = 0.35 * torch.randn(64, int(target.potential.d),
                                generator=generator, dtype=torch.float64,
                                device=target.device)
    return coherent[index] + jitter


def _sites(experiment, x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], int(experiment.target.potential.n_sites), 2)


def test_the_e4_per_sample_observables_match_their_documented_formulas(
        e4_experiment, e4_configurations):
    """Each per-sample observable re-derived from the formula in its docstring
    and in ``configs/experiments/E4.yaml``, independently of the potential class.
    """
    from src.potentials import site_potential

    target = e4_experiment.target
    potential = target.potential
    x = e4_configurations
    q = _sites(e4_experiment, x)
    n_sites = int(potential.n_sites)
    delta = 1.0 / n_sites
    forward = torch.roll(q, shifts=-1, dims=1)
    tolerance = 64.0 * EPS

    # m = (1/N_s) sum_i q_i
    expected_m = q.mean(dim=1)
    m = O.order_parameter(target, x)
    assert torch.allclose(m, expected_m, rtol=tolerance, atol=tolerance)
    assert torch.allclose(O.order_parameter_norm(m), expected_m.norm(dim=1),
                          rtol=tolerance, atol=tolerance)

    # V = kappa/(2 delta) sum_i ||q_{i+1} - q_i||^2 + delta sum_i W(q_i)
    difference = forward - q
    expected_v = ((potential.kappa / (2.0 * delta))
                  * (difference * difference).sum(dim=(1, 2))
                  + delta * site_potential(q, potential.coefficients).sum(dim=1))
    energy_per_site = O.energy_per_site(target, x)
    assert torch.allclose(energy_per_site, expected_v / n_sites,
                          rtol=tolerance, atol=tolerance)

    # G = (1/N_s) sum_i ||q_{i+1} - q_i||^2
    expected_g = (difference * difference).sum(dim=2).mean(dim=1)
    assert torch.allclose(O.coherence(target, x), expected_g, rtol=tolerance,
                          atol=tolerance)

    # C(r) = (1/N_s) sum_i q_i . q_{i+r}, r = 0..floor(N_s/2)
    expected_c = torch.stack(
        [(q * torch.roll(q, shifts=-r, dims=1)).sum(dim=2).mean(dim=1)
         for r in range(n_sites // 2 + 1)], dim=1)
    correlation = O.two_point_correlation(target, x)
    assert correlation.shape == (x.shape[0], n_sites // 2 + 1)
    assert torch.allclose(correlation, expected_c, rtol=tolerance,
                          atol=tolerance)

    # K = (1/N_s) sum_i 1{l_i != l_{i+1}}, l_i = argmin_s ||q_i - mu_s||
    minima = target.extras["refined_site_minima"]
    labels = (q.unsqueeze(2) - minima).norm(dim=-1).argmin(dim=-1)
    expected_k = (labels != torch.roll(labels, shifts=-1, dims=1)).to(
        torch.float64).mean(dim=1)
    assert torch.allclose(O.kink_density(target, x, minima), expected_k,
                          rtol=tolerance, atol=tolerance)


def test_the_susceptibility_is_beta_times_the_sites_times_the_covariance():
    """chi = beta N_s Cov(m) with the N-1 denominator, on synthetic data."""
    generator = torch.Generator().manual_seed(4711)
    m = torch.randn(500, 2, generator=generator, dtype=torch.float64)
    m[:, 1] = 0.7 * m[:, 0] + 0.3 * m[:, 1]         # correlate the components
    beta, n_sites = 8.0, 12
    expected = beta * n_sites * torch.as_tensor(
        np.cov(m.numpy().T, ddof=1), dtype=torch.float64)
    chi = O.susceptibility(m, beta, n_sites)
    assert chi.shape == (2, 2)
    assert torch.allclose(chi, expected, rtol=1e-12, atol=1e-12)
    assert torch.allclose(chi, chi.T, rtol=0.0, atol=0.0)
    # Uniform weights must reproduce the unweighted estimator exactly.
    weights = torch.full((m.shape[0],), 1.0 / m.shape[0], dtype=torch.float64)
    assert torch.allclose(O.susceptibility_weighted(m, weights, beta, n_sites),
                          chi, rtol=1e-12, atol=1e-12)


def test_the_binder_cumulant_uses_the_o2_convention():
    """``U_4 = 1 - E||m||^4 / (2 (E||m||^2)^2)``: 1/2 for a frozen direction,
    0 for the isotropic Gaussian limit, and never the scalar 3-denominator.
    """
    generator = torch.Generator().manual_seed(90210)
    m = torch.randn(4000, 2, generator=generator, dtype=torch.float64)
    squared = (m * m).sum(dim=1)
    expected = 1.0 - float((squared * squared).mean().item()) / (
        2.0 * float(squared.mean().item()) ** 2)
    assert O.binder_cumulant(m) == pytest.approx(expected, rel=8.0 * EPS)

    # A frozen magnitude gives E||m||^4 = (E||m||^2)^2 exactly, hence U_4 = 1/2.
    angle = torch.linspace(0.0, 2.0 * math.pi, 97, dtype=torch.float64)[:-1]
    ring = 1.3 * torch.stack([angle.cos(), angle.sin()], dim=1)
    assert O.binder_cumulant(ring) == pytest.approx(0.5, abs=1e-12)

    # The scalar per-axis convention divides by 3, not 2, and must stay separate.
    component = m[:, 0]
    scalar = component * component
    expected_component = 1.0 - float((scalar * scalar).mean().item()) / (
        3.0 * float(scalar.mean().item()) ** 2)
    assert O.binder_cumulant_component(component) == pytest.approx(
        expected_component, rel=8.0 * EPS)
    assert O.binder_cumulant(m) != O.binder_cumulant_component(component)


def test_the_heat_capacity_is_taken_from_total_not_per_site_energies():
    """``c_V = beta^2 Var[V] / N_s`` with the ``N-1`` denominator; feeding it
    ``V / N_s`` would understate it by exactly ``N_s^2``.
    """
    generator = torch.Generator().manual_seed(31415)
    energies = 40.0 + 3.0 * torch.randn(1000, generator=generator,
                                        dtype=torch.float64)
    beta, n_sites = 8.0, 12
    variance = float(energies.var(unbiased=True).item())
    expected = beta * beta / n_sites * variance
    assert O.heat_capacity_per_site(energies, beta, n_sites) == pytest.approx(
        expected, rel=8.0 * EPS)
    understated = O.heat_capacity_per_site(energies / n_sites, beta, n_sites)
    assert understated == pytest.approx(expected / n_sites ** 2, rel=1e-12)


def test_the_weighted_e4_estimators_reduce_to_the_unweighted_ones(
        e4_experiment, e4_configurations):
    """The SNIS arm uses the weighted aggregators. Each one documents that it
    reduces to its unweighted counterpart at uniform weights, and the reference
    would be inconsistent between its two arms if any of them did not.
    """
    target = e4_experiment.target
    x = e4_configurations
    n = int(x.shape[0])
    weights = torch.full((n,), 1.0 / n, dtype=torch.float64, device=x.device)
    beta, n_sites = float(target.beta), int(target.potential.n_sites)
    minima = target.extras["refined_site_minima"]

    m = O.order_parameter(target, x)
    energies = O.energy_per_site(target, x)
    coherences = O.coherence(target, x)
    correlations = O.two_point_correlation(target, x)
    kinks = O.kink_density(target, x, minima)
    tolerance = 1e-11

    pairs = [
        (O.order_parameter_mean(m), O.order_parameter_mean_weighted(m, weights)),
        (O.energy_per_site_mean(energies),
         O.energy_per_site_mean_weighted(energies, weights)),
        (O.energy_per_site_variance(energies),
         O.energy_per_site_variance_weighted(energies, weights)),
        (O.coherence_mean(coherences),
         O.coherence_mean_weighted(coherences, weights)),
        (O.two_point_correlation_mean(correlations),
         O.two_point_correlation_mean_weighted(correlations, weights)),
        (O.kink_density_mean(kinks), O.kink_density_mean_weighted(kinks, weights)),
        (O.susceptibility(m, beta, n_sites),
         O.susceptibility_weighted(m, weights, beta, n_sites)),
    ]
    for unweighted, weighted in pairs:
        assert torch.allclose(unweighted, weighted, rtol=tolerance,
                              atol=tolerance)
    total = energies * n_sites
    assert O.heat_capacity_per_site(total, beta, n_sites) == pytest.approx(
        O.heat_capacity_per_site_weighted(total, weights, beta, n_sites),
        rel=tolerance)
    assert O.binder_cumulant(m) == pytest.approx(
        O.binder_cumulant_weighted(m, weights), rel=tolerance)


def test_the_derived_correlation_quantities_match_their_formulas(
        e4_experiment, e4_configurations):
    """``C_conn(r) = C(r) - ||E[m]||^2`` and the two relative errors the
    cross-check gates quote.
    """
    target = e4_experiment.target
    x = e4_configurations
    m = O.order_parameter(target, x)
    mean_m = O.order_parameter_mean(m)
    correlation = O.two_point_correlation_mean(O.two_point_correlation(target, x))
    connected = O.connected_two_point_correlation(correlation, mean_m)
    assert torch.allclose(connected, correlation - (mean_m * mean_m).sum(),
                          rtol=8.0 * EPS, atol=8.0 * EPS)

    reference_matrix = torch.tensor([[2.0, 0.5], [0.5, 3.0]],
                                    dtype=torch.float64)
    estimate = reference_matrix + torch.tensor([[0.1, 0.0], [0.0, -0.2]],
                                               dtype=torch.float64)
    expected = (math.sqrt(0.1 ** 2 + 0.2 ** 2)
                / math.sqrt(4.0 + 0.25 + 0.25 + 9.0))
    assert O.relative_frobenius_error(estimate, reference_matrix) == \
        pytest.approx(expected, rel=8.0 * EPS)

    star = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64)
    hat = star + torch.tensor([0.1, -0.1, 0.0], dtype=torch.float64)
    assert O.correlation_relative_l2(hat, star) == pytest.approx(
        math.sqrt(0.02 / (1.0 + 0.25 + 0.0625)), rel=8.0 * EPS)


def test_phase_probabilities_normalize_over_the_inside_samples_only():
    labels = torch.tensor([0, 0, 1, 2, 3, OUTSIDE_LABEL, OUTSIDE_LABEL, 1])
    probabilities, outside = O.phase_probabilities(labels, 4)
    assert outside == pytest.approx(2.0 / 8.0, rel=8.0 * EPS)
    assert float(probabilities.sum().item()) == pytest.approx(1.0, abs=8.0 * EPS)
    assert probabilities.tolist() == pytest.approx(
        [2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], rel=8.0 * EPS)


def test_the_weighted_effective_count_uses_the_frozen_formula():
    """``n_eff = (sum_mask w)^2 / sum_mask w^2`` on a hand-computed example."""
    weights = torch.tensor([0.4, 0.1, 0.5], dtype=torch.float64)
    mask = torch.tensor([True, True, False])
    assert M.weighted_effective_count(weights, mask) == pytest.approx(
        0.5 ** 2 / (0.4 ** 2 + 0.1 ** 2), rel=8.0 * EPS)
    equal = torch.full((7,), 1.0 / 7.0, dtype=torch.float64)
    assert M.weighted_effective_count(equal, torch.ones(7, dtype=torch.bool)) \
        == pytest.approx(7.0, rel=8.0 * EPS)
    assert M.weighted_effective_count(weights,
                                      torch.zeros(3, dtype=torch.bool)) == 0.0
    # Scale invariance: the formula is a ratio of homogeneous degree zero.
    assert M.weighted_effective_count(100.0 * weights, mask) == pytest.approx(
        M.weighted_effective_count(weights, mask), rel=8.0 * EPS)


def _e4_source_tree() -> ast.Module:
    return ast.parse(Path(e4_module.__file__).read_text(encoding="utf-8"))


def _called_names(node: ast.AST) -> set[str]:
    names = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Call):
            function = item.func
            if isinstance(function, ast.Attribute):
                names.add(function.attr)
            elif isinstance(function, ast.Name):
                names.add(function.id)
    return names


def test_the_e4_reference_calls_the_shared_effective_count_rather_than_its_own():
    tree = _e4_source_tree()
    defined = {node.name for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef)}
    assert not [name for name in defined if "effective_count" in name], (
        "e4.py defines its own effective-count helper; the frozen formula lives "
        "in metrics.weighted_effective_count")
    assert "weighted_effective_count" in _called_names(tree)


def test_the_block_length_is_at_least_twice_the_slowest_autocorrelation():
    """``recommended_block_length`` is ``ceil(multiplier * max_f tau_int(f))``."""
    generator = np.random.default_rng(2024)
    series = {}
    for name, rho in (("fast", 0.1), ("slow", 0.9)):
        values = np.zeros(4000)
        for step in range(1, values.size):
            values[step] = rho * values[step - 1] + generator.normal()
        series[name] = values
    taus = [M.autocorrelation_time(values) for values in series.values()]
    slowest = max(taus)
    assert M.autocorrelation_time(series["slow"]) == slowest
    length = M.recommended_block_length(series, 2.0)
    assert length >= 2.0 * slowest
    assert length == math.ceil(2.0 * slowest)


def test_e4_snis_run_agreement_uses_frozen_family_wise_cutoffs():
    """The four maxima preserve the old 2-SE significance per gate family."""
    gates = load_yaml(E4_ACCEPTANCE)["snis_gates"]
    target = gates["run_agreement_family_wise_two_sided_significance"]
    assert target == pytest.approx(math.erfc(2.0 / math.sqrt(2.0)))
    thresholds = gates["max_run_difference_in_combined_se"]
    comparisons = {
        "phase_probability": 24,
        "susceptibility": 24,
        "energy_per_site": 6,
        "coherence_mean": 6,
    }
    assert set(thresholds) == set(comparisons)
    for name, count in comparisons.items():
        fields = e4_module._multiple_comparison_fields(
            thresholds[name], count, target)
        assert thresholds[name] == pytest.approx(
            fields["bonferroni_equivalent_threshold_in_se"])
        assert (count * fields["per_comparison_two_sided_significance"]
                <= target + 1e-14)
        assert fields["family_wise_significance_under_independence"] <= target


def test_e4_half_run_consistency_uses_a_frozen_family_wise_cutoff():
    """The 16 half-run checks are one family, not 16 independent 2-SE tests.

    One comparison per continuous observable per independent run. At a raw 2 SE
    each, about 0.73 false failures are expected per build, so a correct
    reference is rejected roughly half the time. The cutoff preserves the same
    family-wise significance the SNIS run-agreement families already use.
    """
    gates = load_yaml(E4_ACCEPTANCE)["pt_mala_gates"]
    target = gates["half_run_family_wise_two_sided_significance"]
    assert target == pytest.approx(math.erfc(2.0 / math.sqrt(2.0)))

    comparisons = gates["half_run_family_comparisons"]
    threshold = gates["max_half_run_difference_in_combined_se"]
    fields = e4_module._multiple_comparison_fields(
        threshold, comparisons, target)
    assert threshold == pytest.approx(
        fields["bonferroni_equivalent_threshold_in_se"])
    assert (comparisons * fields["per_comparison_two_sided_significance"]
            <= target + 1e-14)
    assert fields["family_wise_significance_under_independence"] <= target

    # The correction must be a correction, not an amnesty: a run whose halves
    # genuinely disagree still has to fail.
    assert threshold < 2.0 * math.sqrt(comparisons)
    raw_expected_false_failures = comparisons * math.erfc(2.0 / math.sqrt(2.0))
    assert raw_expected_false_failures > 0.5, (
        "if a raw 2 SE threshold were not over-rejecting, the correction "
        "would not be justified")


def test_too_few_blocks_fails_the_gate_instead_of_shrinking_the_block_length():
    """The block length is set by the autocorrelation, so a run that yields too
    few blocks must be reported as too short, not re-blocked.
    """
    acceptance = load_yaml(E4_ACCEPTANCE)
    minimum = int(acceptance["uncertainty"]["min_effective_blocks"])
    block_length = 37
    n_blocks_per_chain = minimum - 1
    n_checkpoints = block_length * n_blocks_per_chain
    required = minimum * block_length
    record = e4_module._block_length_gate(
        acceptance, block_length=block_length,
        n_blocks_per_chain=n_blocks_per_chain,
        n_blocks_total=n_blocks_per_chain * 8, max_tau=18.4,
        n_checkpoints=n_checkpoints, required_checkpoints=required)
    assert record["passed"] is False
    assert record["direction"] == "min"
    assert record["observed_value"] == float(n_blocks_per_chain)
    assert record["threshold"] == float(minimum)
    # The reported block length is unchanged: nothing was re-blocked to pass.
    assert record["block_length"] == block_length
    message = record["diagnostic_message"]
    assert "must NOT be shrunk" in message
    assert "EXTENDED" in message and str(required) in message

    passing = e4_module._block_length_gate(
        acceptance, block_length=block_length, n_blocks_per_chain=minimum,
        n_blocks_total=minimum * 8, max_tau=18.4,
        n_checkpoints=block_length * minimum, required_checkpoints=required)
    assert passing["passed"] is True
    assert "must NOT be shrunk" not in passing["diagnostic_message"]


def _synthetic_summaries(layout, *, n_units: int, unit_size: int,
                         scale_x: float, scale_y: float, seed: int):
    """Summary rows for one arm, from synthetic per-sample observables.

    The two components of ``m`` are given very different scales on purpose: the
    ``xx`` entry of the susceptibility then dominates the matrix and carries
    most of the elementwise uncertainty, while the two arms are made to differ
    only in ``yy``.
    """
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(n_units):
        # A per-unit offset makes the units exchangeable draws of a random
        # effect, so there is genuine between-unit variation to resample.
        offset = torch.randn(2, generator=generator, dtype=torch.float64)
        offset = offset * torch.tensor([0.3 * scale_x, 0.3 * scale_y],
                                       dtype=torch.float64)
        m = offset + torch.randn(unit_size, 2, generator=generator,
                                 dtype=torch.float64) * torch.tensor(
            [scale_x, scale_y], dtype=torch.float64)
        energies = 1.0 + 0.1 * m[:, 1]
        coherences = 0.5 + 0.05 * m[:, 1]
        correlations = torch.stack(
            [(m * m).sum(dim=1) * (0.9 ** lag)
             for lag in range(layout.n_lags)], dim=1)
        kinks = torch.zeros(unit_size, dtype=torch.float64)
        labels = (m[:, 1] > 0).to(torch.long)
        features = e4_module._feature_matrix(
            layout, labels=labels, energies=energies, m=m,
            coherences=coherences, correlations=correlations, kinks=kinks)
        rows.append(e4_module._summary_row(features, None))
    return np.asarray(rows, dtype=float)


def test_the_cross_check_standard_error_is_a_whole_statistic_bootstrap():
    """The relative-Frobenius SE must come from resampling PT blocks and SNIS
    runs and recomputing the ratio, never from combining elementwise SEs.
    """
    tree = _e4_source_tree()
    cross_check = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.FunctionDef)
                       and node.name == "_cross_check_bootstrap")
    assert "hierarchical_bootstrap" in _called_names(cross_check)
    assert "_relative_frobenius" in _called_names(cross_check)

    layout = e4_module._Layout(2, 3)
    beta, n_sites, replicates, seed = 8.0, 12, 400, 90210
    # The two arms agree on m_x and differ only in the spread of m_y, so the
    # statistic is sensitive to one matrix entry while the elementwise
    # uncertainty is dominated by another the two arms agree on.
    pt = _synthetic_summaries(layout, n_units=40, unit_size=25, scale_x=10.0,
                              scale_y=1.0, seed=1234)
    snis = _synthetic_summaries(layout, n_units=8, unit_size=125, scale_x=10.0,
                                scale_y=1.25, seed=5678)
    bootstrap = e4_module._cross_check_bootstrap(
        pt, snis, layout, beta=beta, n_sites=n_sites, replicates=replicates,
        seed=seed)
    whole = bootstrap["susceptibility_relative_frobenius_se"]
    assert math.isfinite(whole) and whole > 0.0

    # The strawman: propagate the elementwise standard errors of the two
    # susceptibility matrices through the Frobenius norm as if the entries were
    # independent and the map were linear.
    pt_point = e4_module._derive(pt.sum(axis=0), layout, beta=beta,
                                 n_sites=n_sites)
    pt_se = e4_module._bootstrap_statistics(
        pt, layout, beta=beta, n_sites=n_sites, replicates=replicates,
        seed=seed, template=pt_point)
    snis_point = e4_module._derive(snis.sum(axis=0), layout, beta=beta,
                                   n_sites=n_sites)
    snis_se = e4_module._bootstrap_statistics(
        snis, layout, beta=beta, n_sites=n_sites, replicates=replicates,
        seed=seed, template=snis_point)
    denominator = float(np.linalg.norm(pt_point["susceptibility"]))
    elementwise = float(np.sqrt(
        (np.asarray(pt_se["susceptibility"]) ** 2
         + np.asarray(snis_se["susceptibility"]) ** 2).sum())) / denominator

    # A bootstrap standard deviation from R replicates carries a relative Monte
    # Carlo error of 1/sqrt(2(R-1)), under 4% here. The two numbers disagree by
    # far more than that: the numerator is a NORM of the difference, so the
    # whole-statistic bootstrap sees a folded distribution and the correlation
    # between numerator and denominator, neither of which survives adding
    # elementwise variances in quadrature.
    monte_carlo = 1.0 / math.sqrt(2.0 * (replicates - 1))
    assert monte_carlo < 0.04
    discrepancy = abs(whole - elementwise) / whole
    assert discrepancy > 6.0 * monte_carlo, (whole, elementwise, discrepancy)


# ------------------------------------------------------ acceptance gating
def _acceptance_copy(directory: Path, *, tighten: bool) -> Path:
    """A COPY of the committed acceptance file, never the committed file.

    Every threshold except the one under test is opened up so that a
    reduced-budget build passes; that isolates the effect of the single
    tightened threshold.
    """
    payload = copy.deepcopy(load_yaml(E4_ACCEPTANCE))
    infinity = float("inf")
    # A batch-means MCSE is NaN when fewer than two whole blocks fit in a
    # chain, and NaN passes nothing, so the block rule is relaxed as well.
    payload["uncertainty"]["block_length_multiplier"] = 0.01
    payload["uncertainty"]["min_effective_blocks"] = 1
    payload["uncertainty"]["bootstrap"]["replicates"] = int(
        E4_SMALL["reference"]["bootstrap"]["replicates"])
    payload["pt_mala_gates"].update({
        "min_entry_events_per_phase": 0, "max_split_rhat": infinity,
        "min_bulk_ess": 0, "min_tail_ess": 0, "min_phase_indicator_ess": 0,
        "max_block_mcse_fraction_of_sd": infinity,
        "max_half_run_difference_in_combined_se": infinity,
        "every_cold_chain_visits_all_phases": False})
    payload["snis_gates"].update({
        "min_total_ess": 0, "min_ess_fraction": 0.0,
        "max_normalized_weight": 1.0,
        "min_weighted_effective_count_per_phase": 0,
        "max_run_difference_in_combined_se": {
            name: infinity for name in payload["snis_gates"][
                "max_run_difference_in_combined_se"]},
        "require_coverage": {"all_four_phases": False,
                             "nonzero_kink_configurations": False,
                             "coherence_upper_decile": False}})
    for block in payload["cross_check_gates"].values():
        for key in list(block):
            if key.endswith("_floor") or key == "combined_se_multiplier":
                block[key] = infinity
    if tighten:
        payload["cross_check_gates"]["susceptibility"].update(
            {"relative_frobenius_floor": 0.0, "combined_se_multiplier": 0.0})
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("acceptance_tightened.yaml" if tighten
                        else "acceptance_relaxed.yaml")
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _e4_overrides(acceptance: Path) -> dict:
    overrides = copy.deepcopy(E4_SMALL)
    overrides["reference"]["acceptance"] = str(acceptance)
    return overrides


#: Required by the acceptance file: every gate record carries these fields.
_GATE_FIELDS = ("metric", "threshold", "observed_value", "standard_error",
                "block_length", "bootstrap_definition", "bootstrap_seed",
                "bootstrap_replicates", "passed", "diagnostic_message")

TIGHTENED_GATE = "cross_check_susceptibility_relative_frobenius"


@pytest.fixture(scope="module")
def e4_passing_reference(tmp_path_factory):
    """The reduced-budget build under the relaxed acceptance copy: the baseline
    the tightened build is compared against."""
    root = tmp_path_factory.mktemp("e4-pass")
    acceptance = _acceptance_copy(root / "config", tighten=False)
    experiment = _experiment("E4", root, _e4_overrides(acceptance))
    return experiment, experiment.ensure_reference()


def test_a_reduced_budget_reference_validates_under_the_relaxed_copy(
        e4_passing_reference):
    experiment, reference = e4_passing_reference
    assert reference.reference_validated is True
    assert reference.failed_gates == []
    assert reference.validation_records
    for record in reference.validation_records:
        for field in _GATE_FIELDS:
            assert field in record, (record["metric"], field)
    record = json.loads((experiment.paths.reference_dir
                         / "reference_validation.json").read_text("utf-8"))
    assert record["reference_validated"] is True
    assert record["n_failed"] == 0

def test_e4_runtime_metrics_include_required_distributions_and_raw_vectors(
        e4_passing_reference):
    experiment, reference = e4_passing_reference
    suite = build_measurement_suite(experiment, reference)
    x = reference.sample_bank[:experiment.particles]
    metrics = suite.metrics(x)
    required = {
        "marginal_W1_mx", "marginal_W1_my", "marginal_W1_m_norm",
        "energy_per_site_KS", "energy_per_site_W1",
        "energy_per_site_MMD2_biased", "energy_per_site_MMD2_unbiased",
        "coherence_KS", "coherence_W1", "kink_density_KS",
        "kink_density_W1", "kink_zero_probability",
        "kink_high_tail_probability", "connected_correlation_relative_L2",
        "heat_capacity_per_site", "heat_capacity_per_site_reference",
        "binder_cumulant", "binder_cumulant_reference",
    }
    assert required <= set(metrics)
    phase_total = metrics["phase_outside_count"] + sum(
        metrics[f"phase_count_{index}"] for index in range(suite.n_phases))
    assert phase_total == experiment.particles
    connected = [key for key in metrics
                 if key.startswith("connected_correlation_r")
                 and key != "connected_correlation_relative_L2"]
    assert connected
    assert suite.describe()["metric_definition_hash"]




def test_a_failing_gate_never_produces_a_validated_reference(tmp_path):
    """One tightened threshold must make the build raise, mark the stored record
    unvalidated, and record the failing gate in full.
    """
    acceptance = _acceptance_copy(tmp_path / "config", tighten=True)
    experiment = _experiment("E4", tmp_path, _e4_overrides(acceptance))
    with pytest.raises(ReferenceValidationError) as raised:
        experiment.ensure_reference()
    failed = [gate["metric"] for gate in raised.value.failed_gates]
    assert failed == [TIGHTENED_GATE], failed

    path = experiment.paths.reference_dir / "reference_validation.json"
    assert path.is_file(), "the failure must be written down, not just raised"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["reference_validated"] is False
    assert record["n_failed"] == 1
    assert record["acceptance_file"] == str(acceptance)
    gate = next(item for item in record["gates"]
                if item["metric"] == TIGHTENED_GATE)
    for field in _GATE_FIELDS:
        assert field in gate, field
    assert gate["passed"] is False
    assert gate["threshold"] == 0.0
    assert gate["observed_value"] > 0.0
    assert gate["standard_error"] > 0.0
    # A whole-statistic gate must name its bootstrap; a blocked gate names its
    # block length. This one is the bootstrap kind.
    assert "hierarchical bootstrap" in gate["bootstrap_definition"]
    assert gate["bootstrap_seed"] == E4_SMALL["reference"]["bootstrap"]["seed"]
    assert gate["bootstrap_replicates"] == (
        E4_SMALL["reference"]["bootstrap"]["replicates"])
    assert gate["diagnostic_message"]
    assert all(item["passed"] for item in record["gates"]
               if item["metric"] != TIGHTENED_GATE)

    # The failed build remains on disk as evidence. A second process must not
    # mistake its matching provenance for a promotable cached reference.
    reloaded = _experiment("E4", tmp_path, _e4_overrides(acceptance))
    with pytest.raises(ReferenceValidationError) as cached:
        reloaded.ensure_reference()
    assert [gate["metric"] for gate in cached.value.failed_gates] == [
        TIGHTENED_GATE]
