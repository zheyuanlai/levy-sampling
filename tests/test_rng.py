"""Batch-independent reproducibility of the named random streams."""
from __future__ import annotations

import pytest
import torch

from src.rng import (EnsembleStreams, STREAM_NAMES, canonical_pair_group,
                     seed_for)

DEVICE = torch.device("cpu")


def streams(seeds, *, family="FLA", pair_group=None, experiment="E1"):
    return EnsembleStreams(experiment, family, pair_group or {"alpha": 1.7},
                           tuple(seeds), DEVICE)


# --------------------------------------------------------- the keyed mapping
def test_seed_derivation_is_stable_and_order_independent():
    first = seed_for("E1", "FLA", {"alpha": 1.7}, 3, "diffusion_gen")
    second = seed_for("E1", "FLA", {"alpha": 1.7}, 3, "diffusion_gen")
    assert first == second
    # Key order inside the pairing group must not matter.
    assert first == seed_for("E1", "FLA", {"alpha": 1.7, }, 3, "diffusion_gen")
    reordered = canonical_pair_group({"b": 2, "a": 1})
    assert reordered == canonical_pair_group({"a": 1, "b": 2})


def test_distinct_coordinates_give_distinct_seeds():
    base = seed_for("E1", "FLA", {"alpha": 1.7}, 3, "diffusion_gen")
    assert base != seed_for("E2", "FLA", {"alpha": 1.7}, 3, "diffusion_gen")
    assert base != seed_for("E1", "ULA", {"alpha": 1.7}, 3, "diffusion_gen")
    assert base != seed_for("E1", "FLA", {"alpha": 1.8}, 3, "diffusion_gen")
    assert base != seed_for("E1", "FLA", {"alpha": 1.7}, 4, "diffusion_gen")
    assert base != seed_for("E1", "FLA", {"alpha": 1.7}, 3, "jump_bank_gen")


def test_unknown_stream_names_are_refused():
    with pytest.raises(ValueError, match="unknown stream"):
        seed_for("E1", "FLA", {}, 0, "not_a_declared_stream")


# ------------------------------------------------- batch-independent draws
def test_single_seed_matches_the_same_seed_inside_a_batch():
    """Running seed 3 alone must be bitwise identical to seed 3 in a campaign."""
    n_per_seed = 64
    alone = streams([3]).randn("diffusion_gen", (n_per_seed, 2))
    batch = streams([0, 1, 2, 3, 4, 5, 6, 7]).randn("diffusion_gen",
                                                    (n_per_seed, 2))
    block = batch[3 * n_per_seed:4 * n_per_seed]
    assert torch.equal(alone, block)


def test_adding_or_removing_seeds_leaves_existing_streams_untouched():
    n_per_seed = 32
    small = streams([0, 1]).randn("diffusion_gen", (n_per_seed, 3))
    large = streams([0, 1, 2, 3]).randn("diffusion_gen", (n_per_seed, 3))
    assert torch.equal(small, large[:2 * n_per_seed])
    reordered = streams([1, 0]).randn("diffusion_gen", (n_per_seed, 3))
    assert torch.equal(reordered[:n_per_seed], small[n_per_seed:])
    assert torch.equal(reordered[n_per_seed:], small[:n_per_seed])


def test_repeated_draws_advance_each_seed_independently():
    bundle = streams([0, 1])
    first = bundle.randn("diffusion_gen", (16, 1))
    second = bundle.randn("diffusion_gen", (16, 1))
    assert not torch.equal(first, second)
    solo = streams([1])
    solo_first = solo.randn("diffusion_gen", (16, 1))
    solo_second = solo.randn("diffusion_gen", (16, 1))
    assert torch.equal(first[16:], solo_first)
    assert torch.equal(second[16:], solo_second)


def test_named_streams_are_mutually_independent():
    # The sample correlation of two independent streams is O(1/sqrt(n)), so the
    # threshold has to scale with n rather than being a fixed number.
    n = 200_000
    threshold = 4.0 / n ** 0.5
    bundle = streams([0])
    draws = {name: bundle.randn(name, (n, 1))
             for name in ("diffusion_gen", "init_gen", "jump_bank_gen",
                          "mh_uniform_gen")}
    names = list(draws)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            assert not torch.equal(draws[first], draws[second])
            correlation = torch.corrcoef(
                torch.stack([draws[first][:, 0], draws[second][:, 0]]))[0, 1]
            assert abs(float(correlation)) < threshold, (first, second,
                                                         float(correlation))


def test_multi_dimensional_draws_concatenate_on_the_requested_axis():
    """Parallel tempering draws (K, N, d) per seed and joins on the particle axis."""
    bundle = streams([0, 1])
    joined = bundle.randn("diffusion_gen", (3, 16, 2), cat_dim=1)
    assert tuple(joined.shape) == (3, 32, 2)
    solo = streams([1]).randn("diffusion_gen", (3, 16, 2), cat_dim=1)
    assert torch.equal(joined[:, 16:, :], solo)


# ----------------------------------------------- canonical/tamed pairing
def test_canonical_and_tamed_variants_share_their_named_streams():
    """The pairing group excludes ``tame``, so both variants draw the same numbers."""
    canonical = streams([0, 1], pair_group={"method_family": "FLA", "alpha": 1.7})
    tamed = streams([0, 1], pair_group={"method_family": "FLA", "alpha": 1.7})
    for name in ("init_gen", "stable_noise_gen"):
        assert torch.equal(canonical.randn(name, (64, 2)),
                           tamed.randn(name, (64, 2)))


def test_mala_and_pt_pairs_share_the_gaussian_and_uniform_streams():
    canonical = streams([0], family="MALA",
                        pair_group={"method_family": "MALA"})
    tamed = streams([0], family="MALA", pair_group={"method_family": "MALA"})
    assert torch.equal(canonical.randn("diffusion_gen", (128, 1)),
                       tamed.randn("diffusion_gen", (128, 1)))
    assert torch.equal(canonical.rand("mh_uniform_gen", (128,)),
                       tamed.rand("mh_uniform_gen", (128,)))


def test_distinct_parameter_groups_do_not_share_streams():
    a = streams([0], pair_group={"method_family": "FLA", "alpha": 1.7})
    b = streams([0], pair_group={"method_family": "FLA", "alpha": 1.8})
    assert not torch.equal(a.randn("stable_noise_gen", (64, 1)),
                           b.randn("stable_noise_gen", (64, 1)))


# ---------------------------------------------------------------- manifest
def test_provenance_records_what_a_reader_needs():
    provenance = streams([0, 1, 2]).provenance()
    assert provenance["per_seed_generator"] is True
    assert provenance["seed_execution_order"] == [0, 1, 2]
    assert set(provenance["stream_seed_mapping"]["0"]) == set(STREAM_NAMES)
    # The pairing must be described as common random numbers, because canonical
    # and tamed variants may calibrate to different timesteps and would then not
    # be two discretisations of one continuous-time path.
    assert provenance["pairing_semantics"] == (
        "common_random_numbers_not_pathwise_coupling")
    assert "rng_implementation" in provenance
    assert "seed_derivation" in provenance


def test_seed_block_index_labels_every_particle():
    index = streams([2, 5]).seed_block_index(4)
    assert index.tolist() == [2, 2, 2, 2, 5, 5, 5, 5]
