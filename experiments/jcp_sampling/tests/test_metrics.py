
import numpy as np
import torch

from experiments.jcp_sampling.core.metrics import (
    basin_population_metrics,
    basin_tv_series,
    cdf_sup_error,
    density_l1_error,
    first_all_basin_coverage_time,
    integrated_autocorrelation_time,
    ess_from_iat,
    mixing_metrics,
    threshold_time,
)


def test_cdf_and_density_l1_zero_for_identical():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(20000)
    assert cdf_sup_error(a, a) == 0.0
    assert density_l1_error(a, a) < 1e-9


def test_cdf_sup_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(20000)
    assert cdf_sup_error(a, a + 2.0) > 0.3


def test_coverage_threshold_and_tv_series():
    T, N, nb = 30, 4, 3
    H = np.zeros((T, N), dtype=int)
    for n in range(N):
        H[:, n] = np.clip(np.arange(T) // 5, 0, nb - 1)  # each chain sweeps 0->1->2
    times = np.arange(T) * 1.0
    cov, frac = first_all_basin_coverage_time(H, times, nb)
    assert frac == 1.0
    assert abs(cov - 10.0) < 1e-9   # all three basins first seen when label hits 2 (t=10)
    tv = basin_tv_series(H, [1 / 3, 1 / 3, 1 / 3])
    assert tv.shape == (T,)
    assert abs(tv[0] - 2 / 3) < 1e-9  # all mass in basin 0 at t=0 -> TV = 2/3
    assert np.isnan(threshold_time(tv, times, tau=0.0)) or threshold_time(tv, times, tau=0.5) <= times[-1]


def test_basin_metrics_perfect():
    labels = torch.tensor([0, 1, 0, 1])
    tgt = torch.tensor([0.5, 0.5])
    m = basin_population_metrics(labels, tgt)
    assert m["basin_population_error"] < 1e-8
    assert m["basin_kl"] < 1e-6


def test_iat_and_ess_are_finite():
    x = np.sin(np.linspace(0, 4, 50))
    iat = integrated_autocorrelation_time(x)
    assert np.isfinite(iat)
    assert iat >= 1
    assert ess_from_iat(100, iat) > 0


def test_mixing_metrics_frozen_vs_flipping():
    # A frozen ensemble (all chains stay in basin 0) has no slow-mode sampling:
    # zero transitions and ESS reported as 0 with the frozen flag set, regardless of CV.
    frozen = np.zeros((200, 16), dtype=int)
    mf = mixing_metrics(frozen, np.zeros(200), iat_stride=1)
    assert mf["n_transitions_total"] == 0
    assert mf["ess"] == 0.0
    assert mf["mixing_frozen"] == 1

    # An ensemble whose chains actually flip between basins registers transitions,
    # a positive ESS, and is not flagged frozen.
    rng = np.random.default_rng(0)
    flip = (rng.random((400, 16)) < 0.5).astype(int)
    cv = flip.mean(axis=1)  # fluctuating ensemble CV
    mp = mixing_metrics(flip, cv, iat_stride=1)
    assert mp["n_transitions_total"] > 0
    assert mp["ess"] > 0.0
    assert mp["mixing_frozen"] == 0

    # High-dimensional case: joint label saturates (escaped -> 1) but the continuous CV
    # keeps fluctuating; ESS must be positive (this was the prior false-frozen bug).
    T, N, B = 300, 64, 20
    blocks = (rng.random((T, N, B)) < 0.8).astype(int)
    labels_hd = (blocks * (2 ** np.arange(B))).sum(-1)  # huge-cardinality joint label
    cv_hd = blocks.sum(-1).mean(axis=1)                 # deep-count ensemble mean
    mh = mixing_metrics(labels_hd, cv_hd, iat_stride=1)
    assert mh["mixing_frozen"] == 0
    assert mh["ess"] > 0.0
