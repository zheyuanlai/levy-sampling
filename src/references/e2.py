"""E2 reference: the exact mixture, plus frozen hard-assignment mode masses.

Sampling E2 exactly is free -- ``pi`` is an equal-weight mixture of
``N(mu_k, I_2)`` at every beta, so :meth:`MoG40Reference.sample` draws i.i.d.
from it. The delicate part is the mode descriptor.

The descriptor is a HARD assignment by component log-density,
``a(x) = argmax_k log N(x; mu_k, I_2)``. Under that rule the descriptor masses
``p*_k = P[a(X) = k]`` need not equal the mixture weights ``1/40``: the 40
centres are drawn uniformly in a box, so their Voronoi cells have unequal areas
and a crowded centre can lose net mass to its neighbours. Assuming ``1/40``
would charge a sampler for a bias the target itself has, so ``p*`` is estimated
once from a large exact bank, frozen, and shipped with a multinomial standard
error and a block-bootstrap standard error on the derived EMC.

At the shipped bank size the estimate turns out to be statistically
indistinguishable from uniform (an isolated pair of centres exchanges mass
symmetrically, so only three-way crowding breaks the balance). The build
therefore also records a chi-square against uniform and the plug-in entropy
bias ``(K - 1) / (2 M log K)``, because at this precision the departure of the
plug-in ``emc_star`` from 1 is that bias rather than a property of the target.
Both numbers are diagnostics; the frozen ``p*`` and ``emc_star`` remain the
measured plug-in values.

Everything else a two-sample metric needs -- the smaller sample bank and the
MMD bandwidth -- is frozen here too, at seeds read from the YAML.
"""
from __future__ import annotations

from pathlib import Path
import math

import torch

from ..results import stable_hash
from .base import (REFERENCE_JSON, Reference, as_tensor,
                   check_positive_int, check_seed, frozen_generator,
                   import_metrics, load_npz, read_json, save_npz, write_json)

KIND = "exact_gaussian_mixture"
EXPERIMENT_ID = "E2"

#: The descriptor this reference is defined against. Recorded verbatim in
#: ``describe()`` so a downstream occupancy can never be scored against masses
#: that were computed under a different assignment rule.
DESCRIPTOR_DEFINITION = (
    "a(x) = argmax_k log N(x; mu_k, I_2) over the 40 mixture components. "
    "Equal weights and isotropic unit covariances make this the nearest-centre "
    "Voronoi rule, but it is implemented and defined as the argmax of the "
    "component log-density.")

DEFAULT_BANK_SEED = 424242
DEFAULT_METRIC_BANK_SEED = 848484
DEFAULT_BOOTSTRAP_SEED = 515151
DEFAULT_METRIC_BANK_SIZE = 200_000
DEFAULT_DESCRIPTOR_BANK_SIZE = 4_000_000
DEFAULT_BOOTSTRAP_BLOCKS = 200
DEFAULT_BOOTSTRAP_REPLICATES = 200
#: Largest number of points assigned in one shot. 250k x 40 float64 distances
#: is 80 MB, which fits alongside anything else on the device.
DEFAULT_CHUNK_SIZE = 250_000

MASSES_FILE = "descriptor_masses.npz"
BANK_FILE = "reference_samples.npz"
DIAGNOSTICS_FILE = "diagnostics.json"


def _normalized_entropy(p: torch.Tensor) -> torch.Tensor:
    """``-sum p log p / log K`` along the last axis, with ``0 log 0 = 0``."""
    safe = torch.where(p > 0, p, torch.ones_like(p))
    entropy = -(p * torch.log(safe)).sum(dim=-1)
    return entropy / math.log(p.shape[-1])


# ================================================================= the class
class MoG40Reference(Reference):
    """Exact-mixture reference with frozen hard-assignment descriptor masses.

    Frozen at construction: the descriptor masses ``p*`` and their multinomial
    standard errors, the reference EMC line ``emc_star`` and its block-bootstrap
    standard error, the metric sample bank, and the MMD bandwidth. ``sample``
    itself is exact and needs nothing frozen.
    """

    kind = KIND
    experiment_id = EXPERIMENT_ID

    def __init__(self, *, target, provenance: dict, descriptor_masses,
                 descriptor_masses_standard_error, block_counts,
                 sample_bank: torch.Tensor, measured: dict) -> None:
        self.target = target
        self.potential = target.potential
        self.beta = float(target.beta)
        self.device = sample_bank.device
        self.n_components = int(provenance["n_components"])
        self._provenance = dict(provenance)
        self._measured = dict(measured)
        self.descriptor_masses = descriptor_masses
        self.descriptor_masses_standard_error = descriptor_masses_standard_error
        self.block_counts = block_counts
        self.sample_bank = sample_bank
        self.descriptor_bank_size = int(provenance["sample_bank_size"])
        self.metric_bank_size = int(provenance["metric_bank_size"])
        self.emc_star = float(measured["emc_star"])
        self.emc_star_standard_error = float(measured["emc_star_standard_error"])
        self.mmd_bandwidth = float(measured["mmd_bandwidth"])
        self.descriptor_definition = DESCRIPTOR_DEFINITION

    #: Alias used by the occupancy metrics.
    @property
    def p_star(self) -> torch.Tensor:
        return self.descriptor_masses

    # -- construction ------------------------------------------------------
    @staticmethod
    def provenance_for(config: dict, target) -> dict:
        block = config["reference"]
        descriptor = dict(block.get("descriptor") or {})
        kind = descriptor.get("kind", "argmax_component_log_density")
        if kind != "argmax_component_log_density":
            raise ValueError(
                f"E2 reference descriptor must be "
                f"'argmax_component_log_density', got {kind!r}")
        n_components = int(descriptor.get(
            "n_components", config["target"].get("n_components", 40)))
        if n_components != int(target.potential.n_components):
            raise ValueError(
                f"descriptor.n_components ({n_components}) disagrees with the "
                f"target's {target.potential.n_components} components")
        bank_size = check_positive_int(
            block.get("sample_bank_size", DEFAULT_DESCRIPTOR_BANK_SIZE),
            "reference.sample_bank_size")
        blocks = check_positive_int(
            block.get("emc_bootstrap_blocks", DEFAULT_BOOTSTRAP_BLOCKS),
            "reference.emc_bootstrap_blocks")
        if bank_size % blocks:
            raise ValueError(
                f"reference.sample_bank_size ({bank_size}) must be divisible by "
                f"emc_bootstrap_blocks ({blocks}) so the bootstrap blocks are "
                "equally sized")
        mmd = dict((config.get("metrics") or {}).get("mmd") or {})
        return {
            "experiment_id": EXPERIMENT_ID,
            "kind": KIND,
            "potential": target.potential.name,
            "beta": float(target.beta),
            "dimension": int(target.d),
            "n_components": n_components,
            "center_seed": int(config["target"].get("center_seed", 0)),
            "center_range": [float(value) for value in
                             config["target"].get("center_range", (-40.0, 40.0))],
            "descriptor_kind": kind,
            "sample_bank_size": bank_size,
            "bank_seed": check_seed(block.get("bank_seed", DEFAULT_BANK_SEED),
                                    "reference.bank_seed"),
            "metric_bank_size": check_positive_int(
                block.get("metric_bank_size", DEFAULT_METRIC_BANK_SIZE),
                "reference.metric_bank_size"),
            "metric_bank_seed": check_seed(
                block.get("metric_bank_seed", DEFAULT_METRIC_BANK_SEED),
                "reference.metric_bank_seed"),
            "chunk_size": check_positive_int(
                block.get("chunk_size", DEFAULT_CHUNK_SIZE),
                "reference.chunk_size"),
            "emc_bootstrap_blocks": blocks,
            "emc_bootstrap_replicates": check_positive_int(
                block.get("emc_bootstrap_replicates",
                          DEFAULT_BOOTSTRAP_REPLICATES),
                "reference.emc_bootstrap_replicates"),
            "emc_bootstrap_seed": check_seed(
                block.get("emc_bootstrap_seed", DEFAULT_BOOTSTRAP_SEED),
                "reference.emc_bootstrap_seed"),
            "mmd_bandwidth_points": int(
                mmd.get("bandwidth_reference_points", 4096)),
            "mmd_bandwidth_seed": check_seed(mmd.get("bandwidth_seed", 99),
                                             "metrics.mmd.bandwidth_seed"),
        }

    @classmethod
    def build(cls, config: dict, target, directory: Path | None = None, *,
              device=None, verbose: bool = False) -> "MoG40Reference":
        provenance = cls.provenance_for(config, target)
        (median_heuristic,) = import_metrics("median_heuristic")
        potential = target.potential
        n_components = provenance["n_components"]
        blocks = provenance["emc_bootstrap_blocks"]
        bank_size = provenance["sample_bank_size"]
        block_size = bank_size // blocks

        with target.no_count():
            generator = frozen_generator(target.device, provenance["bank_seed"])
            counts = torch.zeros(blocks, n_components, dtype=torch.float64,
                                 device=target.device)
            for index in range(blocks):
                remaining = block_size
                while remaining > 0:
                    take = min(remaining, provenance["chunk_size"])
                    points = potential.sample_exact(take, generator)
                    labels = potential.component_log_density(points).argmax(-1)
                    counts[index] += torch.bincount(
                        labels, minlength=n_components).to(torch.float64)
                    remaining -= take
                if verbose and (index + 1) % 20 == 0:
                    print(f"[E2] descriptor bank {(index + 1) * block_size}"
                          f"/{bank_size}")

            total = counts.sum()
            if int(total.item()) != bank_size:
                raise RuntimeError(
                    f"descriptor bank lost samples: {int(total.item())} labelled "
                    f"of {bank_size} drawn")
            p_star = counts.sum(dim=0) / float(bank_size)
            standard_error = torch.sqrt(
                p_star * (1.0 - p_star) / float(bank_size))
            emc_star = float(_normalized_entropy(p_star).item())
            emc_se, bootstrap = _bootstrap_emc_standard_error(
                counts, provenance["emc_bootstrap_replicates"],
                provenance["emc_bootstrap_seed"])

            metric_generator = frozen_generator(target.device,
                                                provenance["metric_bank_seed"])
            sample_bank = _draw_bank(potential, provenance["metric_bank_size"],
                                     metric_generator, provenance["chunk_size"])
            bandwidth = float(median_heuristic(
                sample_bank, max_points=provenance["mmd_bandwidth_points"],
                seed=provenance["mmd_bandwidth_seed"]))

        smallest = float(p_star.min().item())
        largest = float(p_star.max().item())
        expected = float(bank_size) / n_components
        chi_square = float((((counts.sum(dim=0) - expected) ** 2)
                            / expected).sum().item())
        support = int((p_star > 0).sum().item())
        plugin_bias = ((support - 1)
                       / (2.0 * bank_size * math.log(n_components)))
        measured = {
            "descriptor_definition": DESCRIPTOR_DEFINITION,
            "p_star": p_star.detach().cpu().tolist(),
            "p_star_standard_error": standard_error.detach().cpu().tolist(),
            "p_star_min": smallest,
            "p_star_max": largest,
            "p_star_uniform": 1.0 / n_components,
            "p_star_max_over_min": largest / smallest if smallest > 0 else
            float("inf"),
            "p_star_max_abs_deviation_from_uniform": float(
                (p_star - 1.0 / n_components).abs().max().item()),
            "p_star_total_variation_from_uniform": float(
                (0.5 * (p_star - 1.0 / n_components).abs().sum()).item()),
            "p_star_chi_square_vs_uniform": chi_square,
            "p_star_chi_square_dof": n_components - 1,
            "emc_star": emc_star,
            "emc_star_standard_error": emc_se,
            "emc_star_standard_error_method": bootstrap,
            "emc_star_plugin_bias": plugin_bias,
            "emc_star_miller_madow": emc_star + plugin_bias,
            "emc_star_note": (
                "the reference EMC line under the hard component-log-density "
                "assignment, as the plug-in estimator on the frozen bank. Its "
                "deficit below 1 should be read against emc_star_plugin_bias "
                "(K-1)/(2 M log K), the downward bias of any plug-in EMC at "
                "bank size M, and against p_star_chi_square_vs_uniform on "
                "K-1 degrees of freedom: at this bank size the descriptor "
                "masses are not resolvably different from 1/K, so a sampler's "
                "own EMC bias at N particles (much larger, same formula with "
                "M -> N) dominates any comparison against this line"),
            "mmd_bandwidth": bandwidth,
            "mmd_bandwidth_rule": "median_heuristic_on_frozen_metric_bank",
            "sample_bank_shape": list(sample_bank.shape),
            "build_device": str(target.device),
        }
        return cls(target=target, provenance=provenance,
                   descriptor_masses=p_star,
                   descriptor_masses_standard_error=standard_error,
                   block_counts=counts, sample_bank=sample_bank,
                   measured=measured)

    # -- contract ----------------------------------------------------------
    def describe(self) -> dict:
        return {
            "experiment_id": EXPERIMENT_ID,
            "kind": KIND,
            "provenance": dict(self._provenance),
            "provenance_hash": stable_hash(self._provenance),
            **self._measured,
        }

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_npz(directory / MASSES_FILE,
                 p_star=self.descriptor_masses,
                 p_star_standard_error=self.descriptor_masses_standard_error,
                 block_counts=self.block_counts)
        save_npz(directory / BANK_FILE, sample_bank=self.sample_bank)
        write_json(directory / DIAGNOSTICS_FILE, {
            key: self._measured[key] for key in (
                "descriptor_definition", "p_star_min", "p_star_max",
                "p_star_uniform", "p_star_max_over_min",
                "p_star_max_abs_deviation_from_uniform",
                "p_star_total_variation_from_uniform",
                "p_star_chi_square_vs_uniform", "p_star_chi_square_dof",
                "emc_star", "emc_star_standard_error",
                "emc_star_standard_error_method", "emc_star_plugin_bias",
                "emc_star_miller_madow", "emc_star_note", "mmd_bandwidth")})
        self.write_describe(directory)

    @classmethod
    def load(cls, directory: Path, target, device) -> "MoG40Reference":
        directory = Path(directory)
        payload = read_json(directory / REFERENCE_JSON)
        masses = load_npz(directory / MASSES_FILE)
        bank = load_npz(directory / BANK_FILE)
        measured = {key: value for key, value in payload.items()
                    if key not in ("provenance", "provenance_hash",
                                   "experiment_id", "kind")}
        return cls(
            target=target,
            provenance=payload["provenance"],
            descriptor_masses=as_tensor(masses["p_star"], device),
            descriptor_masses_standard_error=as_tensor(
                masses["p_star_standard_error"], device),
            block_counts=as_tensor(masses["block_counts"], device),
            sample_bank=as_tensor(bank["sample_bank"], device),
            measured=measured)

    def sample(self, n: int, generator: torch.Generator) -> torch.Tensor:
        """``(n, 2)`` exact i.i.d. draws from the equal-weight mixture."""
        with self.target.no_count():
            return _draw_bank(self.potential, check_positive_int(n, "n"),
                              generator, self._provenance["chunk_size"])

    # -- the descriptor ----------------------------------------------------
    def assign(self, x: torch.Tensor) -> torch.Tensor:
        """Hard mode label ``argmax_k log N(x; mu_k, I_2)``, shape ``x.shape[:-1]``.

        Implemented as the argmax of the component log-density, which is the
        definition the frozen masses were estimated under.
        """
        if x.ndim < 1 or x.shape[-1] != 2:
            raise ValueError(
                f"points must have shape (..., 2), got {tuple(x.shape)}")
        flat = x.reshape(-1, 2).to(dtype=torch.float64, device=self.device)
        chunk = self._provenance["chunk_size"]
        with self.target.no_count():
            parts = [self.potential.component_log_density(
                flat[start:start + chunk]).argmax(-1)
                for start in range(0, flat.shape[0], chunk)]
        return torch.cat(parts).reshape(x.shape[:-1])


def _draw_bank(potential, n: int, generator: torch.Generator,
               chunk: int) -> torch.Tensor:
    parts = []
    remaining = int(n)
    while remaining > 0:
        take = min(remaining, int(chunk))
        parts.append(potential.sample_exact(take, generator))
        remaining -= take
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def _bootstrap_emc_standard_error(counts: torch.Tensor, replicates: int,
                                  seed: int) -> tuple[float, dict]:
    """Block bootstrap of EMC over the equal-size chunks of the frozen bank.

    Resampling whole blocks rather than individual points keeps the estimator
    honest about the fact that the bank was drawn as a sequence of chunks, and
    needs no delta-method linearisation of the entropy near a small ``p_k``.
    """
    blocks = counts.shape[0]
    generator = frozen_generator(counts.device, seed)
    index = torch.randint(0, blocks, (int(replicates), blocks),
                          generator=generator, device=counts.device)
    resampled = counts[index].sum(dim=1)
    p = resampled / resampled.sum(dim=1, keepdim=True)
    values = _normalized_entropy(p)
    return float(values.std(unbiased=True).item()), {
        "method": "block_bootstrap",
        "n_blocks": int(blocks),
        "block_size": int(counts.sum(dim=1)[0].item()),
        "n_replicates": int(replicates),
        "seed": int(seed),
        "bootstrap_mean_emc": float(values.mean().item()),
    }


def build_reference(config: dict, target, directory: Path | None = None, *,
                    device=None, verbose: bool = False) -> MoG40Reference:
    """Entry point named by ``reference.builder`` in ``configs/E2.yaml``."""
    return MoG40Reference.build(config, target, directory, device=device,
                                verbose=verbose)
