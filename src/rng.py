"""Named random streams with per-seed, batch-independent generators.

Two identities are kept apart on purpose.

*Variant identity* is the full parameter set actually run, including ``tame``.
*Random-stream pairing identity* is the ``rng_pair_group``: the subset of
parameters that decides which variants share random numbers. A canonical
(``tame: false``) and a tamed (``tame: true``) variant of the same method at the
same hyperparameters carry the same ``rng_pair_group`` and therefore draw the
same initial states and the same named streams.

Every ``(experiment, method family, pair group, seed index, stream name)`` tuple
owns its own ``torch.Generator``, seeded through a stable keyed hash. Batched
draws are produced one seed block at a time, each block with a per-seed shape
that does not depend on how many seeds are in the batch, and then concatenated
in a fixed seed order. Running seed 3 alone therefore consumes exactly the same
numbers as seed 3 inside an eight-seed campaign, and adding or removing seeds or
whole methods leaves every existing stream untouched.

This is common-random-number pairing, not pathwise coupling: when canonical and
tamed variants calibrate to different timesteps, sharing a named stream does not
make them two discretisations of one continuous-time Brownian or Levy path.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math

import torch

RNG_IMPLEMENTATION = "torch.Generator/Philox-or-MT19937"
SEED_DERIVATION = "blake2b-64/v2"

#: Stream names every sampler is allowed to draw from. Kept in sync with
#: ``rng_streams`` in ``configs/registry.yaml``.
STREAM_NAMES: tuple[str, ...] = (
    "init_gen",
    "diffusion_gen",
    "stable_noise_gen",
    "jump_bank_gen",
    "poisson_gen",
    "mh_uniform_gen",
)

_SEED_MODULUS = 1 << 63


def canonical_pair_group(pair_group: Mapping[str, object] | None) -> str:
    """Serialize a pairing group to a stable string.

    Key order and float formatting are normalized so that ``{"alpha": 1.7}``
    written by a YAML loader and by a notebook literal hash identically.
    """
    if not pair_group:
        return "{}"
    normalized: dict[str, object] = {}
    for key in sorted(pair_group):
        value = pair_group[key]
        if isinstance(value, bool):
            normalized[str(key)] = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"pair-group value {key!r} must be finite")
            # repr of a float round-trips exactly and is stable across releases.
            normalized[str(key)] = repr(float(value))
        elif isinstance(value, int):
            normalized[str(key)] = int(value)
        else:
            normalized[str(key)] = str(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def seed_for(experiment_id: str, method_family: str,
             paired_parameter_group: Mapping[str, object] | None,
             seed_index: int, stream_name: str) -> int:
    """Stable keyed seed for one (variant-pairing, seed, stream) triple.

    The mapping is a hash, not a counter, so it does not depend on execution
    order, on which methods ran before, or on how many seeds are in the batch.
    """
    if stream_name not in STREAM_NAMES:
        raise ValueError(
            f"unknown stream {stream_name!r}; declare it in registry.yaml "
            f"and in rng.STREAM_NAMES (known: {list(STREAM_NAMES)})")
    if isinstance(seed_index, bool) or int(seed_index) != seed_index:
        raise ValueError("seed_index must be an integer")
    key = "\x1f".join((
        SEED_DERIVATION,
        str(experiment_id),
        str(method_family),
        canonical_pair_group(paired_parameter_group),
        str(int(seed_index)),
        str(stream_name),
    ))
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SEED_MODULUS


class SeedStreams:
    """The named generators owned by a single seed index."""

    def __init__(self, experiment_id: str, method_family: str,
                 pair_group: Mapping[str, object] | None, seed_index: int,
                 device: torch.device) -> None:
        self.experiment_id = str(experiment_id)
        self.method_family = str(method_family)
        self.pair_group = dict(pair_group or {})
        self.seed_index = int(seed_index)
        self.device = torch.device(device)
        self._generators: dict[str, torch.Generator] = {}
        self._seeds: dict[str, int] = {}

    def seed(self, stream_name: str) -> int:
        if stream_name not in self._seeds:
            self._seeds[stream_name] = seed_for(
                self.experiment_id, self.method_family, self.pair_group,
                self.seed_index, stream_name)
        return self._seeds[stream_name]

    def generator(self, stream_name: str) -> torch.Generator:
        generator = self._generators.get(stream_name)
        if generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed(stream_name))
            self._generators[stream_name] = generator
        return generator

    def seed_mapping(self) -> dict[str, int]:
        return {name: self.seed(name) for name in STREAM_NAMES}


class EnsembleStreams:
    """Named streams over an ordered list of seeds, one generator per seed.

    All draw helpers take a *per-seed* shape and return the seed blocks
    concatenated along ``cat_dim`` in the declared seed order.
    """

    #: Recorded in the manifest so a reader can tell this apart from a future
    #: counter-based implementation.
    per_seed_generator = True

    def __init__(self, experiment_id: str, method_family: str,
                 pair_group: Mapping[str, object] | None,
                 seeds: Sequence[int], device: torch.device,
                 dtype: torch.dtype = torch.float64) -> None:
        seeds = tuple(int(s) for s in seeds)
        if not seeds:
            raise ValueError("at least one seed is required")
        if len(set(seeds)) != len(seeds):
            raise ValueError("seed indices must be unique")
        self.seeds = seeds
        self.device = torch.device(device)
        self.dtype = dtype
        self.experiment_id = str(experiment_id)
        self.method_family = str(method_family)
        self.pair_group = dict(pair_group or {})
        self._per_seed = {
            seed: SeedStreams(experiment_id, method_family, pair_group, seed,
                              self.device)
            for seed in seeds
        }

    # -- introspection ----------------------------------------------------
    def streams_for(self, seed: int) -> SeedStreams:
        return self._per_seed[int(seed)]

    def generator(self, seed: int, stream_name: str) -> torch.Generator:
        return self._per_seed[int(seed)].generator(stream_name)

    def seed_mapping(self) -> dict[str, dict[str, int]]:
        """``{seed: {stream: derived seed}}`` for the run manifest."""
        return {str(seed): self._per_seed[seed].seed_mapping()
                for seed in self.seeds}

    def provenance(self) -> dict:
        return {
            "rng_implementation": RNG_IMPLEMENTATION,
            "seed_derivation": SEED_DERIVATION,
            "per_seed_generator": True,
            "seed_execution_order": list(self.seeds),
            "rng_pair_group": {
                "experiment_id": self.experiment_id,
                "method_family": self.method_family,
                **self.pair_group,
            },
            "stream_seed_mapping": self.seed_mapping(),
            "pairing_semantics": "common_random_numbers_not_pathwise_coupling",
        }

    # -- draws ------------------------------------------------------------
    def _blocks(self, stream_name: str, fn, cat_dim: int) -> torch.Tensor:
        parts = [fn(self._per_seed[seed].generator(stream_name))
                 for seed in self.seeds]
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=cat_dim)

    def randn(self, stream_name: str, shape_per_seed: Sequence[int],
              *, cat_dim: int = 0, dtype: torch.dtype | None = None
              ) -> torch.Tensor:
        shape = tuple(int(s) for s in shape_per_seed)
        dtype = self.dtype if dtype is None else dtype
        return self._blocks(
            stream_name,
            lambda g: torch.randn(shape, generator=g, device=self.device,
                                  dtype=dtype),
            cat_dim)

    def rand(self, stream_name: str, shape_per_seed: Sequence[int],
             *, cat_dim: int = 0, dtype: torch.dtype | None = None
             ) -> torch.Tensor:
        shape = tuple(int(s) for s in shape_per_seed)
        dtype = self.dtype if dtype is None else dtype
        return self._blocks(
            stream_name,
            lambda g: torch.rand(shape, generator=g, device=self.device,
                                 dtype=dtype),
            cat_dim)

    def poisson(self, stream_name: str, rates_per_seed: torch.Tensor,
                *, cat_dim: int = 0) -> torch.Tensor:
        """Poisson counts for a rate tensor shaped like ONE seed block."""
        return self._blocks(
            stream_name,
            lambda g: torch.poisson(rates_per_seed, generator=g),
            cat_dim)

    def categorical(self, stream_name: str, weights: torch.Tensor,
                    n_per_seed: int, *, cat_dim: int = 0) -> torch.Tensor:
        """Draw ``n_per_seed`` component indices per seed from ``weights``."""
        n_per_seed = int(n_per_seed)
        probabilities = weights.reshape(1, -1).expand(n_per_seed, -1)
        return self._blocks(
            stream_name,
            lambda g: torch.multinomial(
                probabilities, 1, replacement=True, generator=g).squeeze(1),
            cat_dim)

    def symmetric_alpha_stable(self, stream_name: str,
                               shape_per_seed: Sequence[int], alpha: float,
                               *, cat_dim: int = 0,
                               dtype: torch.dtype | None = None
                               ) -> torch.Tensor:
        """Chambers-Mallows-Stuck draws from S-alpha-S(1), per coordinate.

        No tail truncation: a truncated stable law is not stable, and the whole
        point of the FLA comparator is its heavy tails.
        """
        shape = tuple(int(s) for s in shape_per_seed)
        dtype = self.dtype if dtype is None else dtype
        alpha = float(alpha)
        if not 0.0 < alpha <= 2.0:
            raise ValueError("alpha must lie in (0, 2]")

        def draw(generator: torch.Generator) -> torch.Tensor:
            phi = (torch.rand(shape, generator=generator, device=self.device,
                              dtype=dtype) - 0.5) * math.pi
            w = -torch.log(torch.rand(shape, generator=generator,
                                      device=self.device, dtype=dtype))
            return (torch.sin(alpha * phi) / torch.cos(phi) ** (1.0 / alpha)
                    * (torch.cos((1.0 - alpha) * phi) / w)
                    ** ((1.0 - alpha) / alpha))

        return self._blocks(stream_name, draw, cat_dim)

    def seed_block_index(self, n_per_seed: int) -> torch.Tensor:
        """``(S * n_per_seed,)`` tensor labelling each particle with its seed."""
        n_per_seed = int(n_per_seed)
        return torch.repeat_interleave(
            torch.as_tensor(self.seeds, dtype=torch.int64, device=self.device),
            n_per_seed)


def build_streams(experiment_id: str, method_family: str,
                  pair_group: Mapping[str, object] | None,
                  seeds: Iterable[int], device, dtype=torch.float64
                  ) -> EnsembleStreams:
    return EnsembleStreams(experiment_id, method_family, pair_group,
                           tuple(seeds), torch.device(device), dtype)
