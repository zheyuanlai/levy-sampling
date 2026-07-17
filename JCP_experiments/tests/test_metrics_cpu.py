"""CPU regression tests for thermodynamic and convergence metrics."""
import math

import numpy as np
import torch

from src import metrics as M


def test_basin_kl_target_to_empirical_orientation_and_smoothing():
    p_star = torch.tensor([0.5, 0.5], dtype=torch.float64)
    assert M.basin_kl_target_to_empirical(p_star, p_star) == 0.0
    assert M.basin_kl_target_to_empirical(
        7.0 * p_star, p_star
    ) == 0.0

    missing = torch.tensor([1.0, 0.0], dtype=torch.float64)
    assert math.isinf(M.basin_kl_target_to_empirical(missing, p_star))

    got = M.basin_kl_target_to_empirical(
        missing, p_star, pseudocount=0.1
    )
    expected = 0.5 * math.log(36.0 / 11.0)
    assert abs(got - expected) < 1e-12


def test_binned_probabilities_supports_nd_and_rejects_overflow():
    x = torch.tensor([
        [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75],
        [-2.0, 0.25],
    ], dtype=torch.float64)
    edges = (
        torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
        torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
    )
    p = M.binned_probabilities(x, edges)
    assert p.shape == (2, 2)
    assert torch.allclose(p, torch.full((2, 2), 0.25, dtype=torch.float64))


def test_weighted_binned_probabilities_are_scale_invariant():
    x = torch.tensor([0.1, 0.2, 0.8, 0.9], dtype=torch.float64)
    edges = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    weights = torch.tensor([1.0, 1.0, 3.0, 5.0], dtype=torch.float64)
    expected = torch.tensor([0.2, 0.8], dtype=torch.float64)
    got = M.binned_probabilities(x, edges, sample_weights=weights)
    scaled = M.binned_probabilities(x, edges, sample_weights=17.0 * weights)
    assert torch.allclose(got, expected)
    assert torch.allclose(scaled, expected)


def test_reduced_free_energy_uses_density_for_nonuniform_bins():
    # Uniform density has bin masses proportional to bin widths and flat F.
    p = torch.tensor([0.25, 0.75], dtype=torch.float64)
    volume = torch.tensor([1.0, 3.0], dtype=torch.float64)
    A = M.reduced_free_energy(p, volume)
    assert torch.allclose(A, A[0].expand_as(A), atol=1e-14)


def test_free_energy_rmse_is_aligned_reduced_kbt_error():
    p_ref = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
    p_hat = torch.tensor([0.25, 0.25, 0.5], dtype=torch.float64)
    expected = math.log(2.0) * math.sqrt(2.0 / 3.0)
    got = M.free_energy_rmse_from_probabilities(p_hat, p_ref)
    assert abs(got - expected) < 1e-12
    # Probability normalisation/additive offsets cannot change the result.
    assert M.free_energy_rmse_from_probabilities(7.0 * p_ref, p_ref) < 1e-14


def test_free_energy_profile_physical_units_and_error_beta_independence():
    edges = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    cv = torch.tensor([0.25, 0.75, 1.25], dtype=torch.float64)
    F2, p = M.free_energy_profile(cv, edges, beta=2.0, smooth=0.0)
    assert abs(float(F2[1]) - math.log(2.0) / 2.0) < 1e-12
    ref_F, ref_p = M.free_energy_profile(cv, edges, beta=2.0, smooth=0.5)
    e2 = M.free_energy_profile_error(cv, edges, 2.0, ref_F, ref_p, 0.0)
    ref_F8, ref_p8 = M.free_energy_profile(cv, edges, beta=8.0, smooth=0.5)
    e8 = M.free_energy_profile_error(cv, edges, 8.0, ref_F8, ref_p8, 0.0)
    assert abs(e2 - e8) < 1e-14


def test_iat_and_ess_handle_constant_short_and_ar1_series():
    assert math.isinf(M.iat_1d(np.ones(100)))
    assert M.ess_from_series(np.ones((3, 100))) == 0.0
    assert math.isnan(M.iat_1d(np.array([1.0])))
    assert math.isnan(M.ess_from_series(np.array([[1.0]])))

    rng = np.random.default_rng(123)
    x = np.empty(30_000)
    eps = rng.normal(size=x.size)
    x[0] = eps[0]
    for i in range(1, x.size):
        x[i] = 0.8 * x[i - 1] + eps[i]
    # AR(1) theoretical IAT is (1+phi)/(1-phi)=9.
    assert 6.0 < M.iat_1d(x) < 13.0


def test_average_tie_ranks_and_discrete_split_rhat():
    ranks = M._average_ranks(np.array([1.0, 1.0, 2.0, 3.0, 3.0]))
    assert np.allclose(ranks, [1.5, 1.5, 3.0, 4.5, 4.5])

    base = np.tile([0.0, 0.0, 1.0, 1.0], 250)
    chains = np.vstack([np.roll(base, k) for k in range(4)])
    bulk, folded = M.split_rhat_components(chains)
    assert np.isfinite(bulk) and np.isfinite(folded)
    assert 0.99 <= M.split_rhat(chains) < 1.01


def test_constant_split_rhat_is_undefined_not_false_convergence():
    assert math.isnan(M.split_rhat(np.zeros((4, 100))))


def test_folded_rhat_detects_scale_nonconvergence():
    rng = np.random.default_rng(9)
    chains = np.vstack([
        rng.normal(scale=1.0, size=3000),
        rng.normal(scale=1.0, size=3000),
        rng.normal(scale=4.0, size=3000),
        rng.normal(scale=4.0, size=3000),
    ])
    bulk, folded = M.split_rhat_components(chains)
    assert bulk < 1.02
    assert folded > 1.08
    assert M.split_rhat(chains) == max(bulk, folded)


def test_passage_estimators_filter_home_and_handle_censoring():
    labels = np.array([
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 1, 0, 1],
    ])
    times, events = M.first_passage_observations(
        labels, home=0, far=1, dt=1.0, steps_per_frame=1
    )
    assert np.array_equal(times, [2.0, 4.0, 4.0])
    assert np.array_equal(events, [True, True, False])
    assert abs(M.kaplan_meier_rmst(times, events, tau=4.0) - 10.0 / 3.0) < 1e-12
    assert abs(M.committed_mfpt(labels, 0, 1, 1.0, 1) - 10.0 / 3.0) < 1e-12
    assert abs(M.exponential_waiting_time_mle(labels, 0, 1, 1.0, 1) - 5.0) < 1e-12


def test_committed_rmst_uses_declared_horizon_when_all_chains_hit_early():
    # Both chains hit at t=1, while the common recorded horizon is t=4.
    # The KM survival is zero after t=1, so RMST remains 1; more importantly,
    # the wrapper must pass the declared horizon rather than infer it from the
    # latest event.
    labels = np.asarray([
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
    ])
    times, events = M.first_passage_observations(labels, 0, 1, 1.0, 1)
    assert np.array_equal(times, [1.0, 1.0])
    assert events.all()
    assert M.committed_mfpt(labels, 0, 1, 1.0, 1) == 1.0


def test_passage_no_events_and_no_eligible_chains():
    no_hits = np.zeros((5, 2), dtype=int)
    assert M.committed_mfpt(no_hits, 0, 1, 0.5, 2) == 4.0
    assert math.isinf(M.exponential_waiting_time_mle(no_hits, 0, 1, 0.5, 2))

    no_home = np.ones((5, 2), dtype=int)
    assert math.isnan(M.committed_mfpt(no_home, 0, 1, 1.0, 1))


def test_energy_hist_overlap_penalizes_outside_grid_mass():
    edges = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    ref_hist = torch.tensor([0.5, 0.5], dtype=torch.float64)
    inside = torch.tensor([0.25, 0.75, 1.25, 1.75], dtype=torch.float64)
    assert M.energy_hist_overlap(inside, edges, ref_hist) == 1.0

    # Half the observations lie outside the frozen support.  They contribute
    # zero overlap rather than being silently clamped into boundary bins.
    tailed = torch.tensor([0.25, 1.25, -10.0, 10.0], dtype=torch.float64)
    assert M.energy_hist_overlap(tailed, edges, ref_hist) == 0.5


def test_round_trips_require_observed_home_before_far():
    labels = np.asarray([
        [1, 0, 2],
        [0, 1, 1],
        [0, 0, 0],
    ])
    # chain 0 starts far then returns home: no trip; chain 1 observes the full
    # home->far->home sequence: one trip; chain 2 sees far before any home.
    assert M.round_trips(labels, home=0, far=1) == 1.0 / 3.0
