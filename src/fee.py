"""Force-equivalent evaluation cost (FEE).

    N_FEE = N_F + rho * N_V_eq^extra,      rho = C_V / C_F

``C_V`` and ``C_F`` are both *amortized wall time per configuration*:

    C_V = potential batch wall time / configurations in the batch
    C_F = force batch wall time     / configurations in the batch

so ``rho`` is dimensionless. Microbenchmarks may use batched kernels, but the
batch time is divided by the number of configurations before anything is
multiplied by an oracle count -- multiplying a per-batch time by a
per-configuration count would be a unit error of the batch size.

FEE is an oracle-cost proxy, deliberately not a full cost model. It excludes
communication, host-device transfers, Python and framework overhead, random
number generation, accept/reject logic, and parallel efficiency.

Structured kernels
------------------
The coupled quartic chain computes chord energies through an exact moment
identity. Its measured cost is converted into an equivalent number of potential
evaluations rather than being reported as generic ``V()`` calls:

    C_structured-extra = structured batch wall time / particles in the batch
    N_V_eq^extra       = C_structured-extra / C_V     (per particle per step)

The kernel is affine in the chord count (moments once per particle, then a
polynomial per chord), so the calibration fits a fixed per-particle part and a
per-chord part and reports both.

Comparability
-------------
Runs may only share a FEE axis when their ``fee_calibration_hash`` matches. A
plotter must refuse to merge mismatched calibrations unless an explicit
compatibility record says the workload and cost unit are comparable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch

from .device import DTYPE, device_provenance, software_version_key, synchronize

COST_UNIT = "amortized_time_per_configuration"
DEFAULT_WARMUP = 50
DEFAULT_REPETITIONS = 200


@dataclass(frozen=True)
class FEECalibration:
    """A frozen per-configuration cost calibration for one device and target."""

    device: str
    device_name: str
    dtype: str
    software_version: str
    target_implementation: str
    particle_batch_size: int
    shifted_potential_batch_shape: tuple[int, ...]
    warmup: int
    repetitions: int
    synchronization: bool
    statistic: str
    cost_unit: str
    C_V: float
    C_F: float
    rho: float
    device_index: int | None = None
    device_uuid: str | None = None
    #: Structured chord kernel, affine in the chord count. Both are seconds.
    structured_fixed_seconds_per_particle: float | None = None
    structured_seconds_per_particle_chord: float | None = None
    structured_chord_counts_measured: tuple[int, ...] = ()
    provenance: dict = field(default_factory=dict, compare=False)

    # -- identity ----------------------------------------------------------
    def identity_payload(self) -> dict:
        """Everything the hash covers. Provenance details are excluded so that
        an unrelated package bump does not invalidate a calibration, while the
        numerical stack that actually sets the cost is included."""
        return {
            "device": self.device,
            "device_name": self.device_name,
            "device_index": self.device_index,
            "device_uuid": self.device_uuid,
            "dtype": self.dtype,
            "software_version": self.software_version,
            "target_implementation": self.target_implementation,
            "particle_batch_size": int(self.particle_batch_size),
            "shifted_potential_batch_shape": list(
                self.shifted_potential_batch_shape),
            "warmup": int(self.warmup),
            "repetitions": int(self.repetitions),
            "synchronization": bool(self.synchronization),
            "statistic": self.statistic,
            "cost_unit": self.cost_unit,
            "C_V": _round_significant(self.C_V),
            "C_F": _round_significant(self.C_F),
            "rho": _round_significant(self.rho),
            "structured_fixed_seconds_per_particle": _round_significant(
                self.structured_fixed_seconds_per_particle),
            "structured_seconds_per_particle_chord": _round_significant(
                self.structured_seconds_per_particle_chord),
            "structured_chord_counts_measured": list(
                self.structured_chord_counts_measured),
        }

    @property
    def hash(self) -> str:
        payload = json.dumps(self.identity_payload(), sort_keys=True,
                             separators=(",", ":"))
        return hashlib.blake2b(payload.encode("utf-8"),
                               digest_size=16).hexdigest()

    def to_dict(self) -> dict:
        record = asdict(self)
        record["shifted_potential_batch_shape"] = list(
            self.shifted_potential_batch_shape)
        record["structured_chord_counts_measured"] = list(
            self.structured_chord_counts_measured)
        record["fee_calibration_hash"] = self.hash
        return record

    # -- cost accounting ---------------------------------------------------
    def extra_potential_equivalent(self, counters: dict) -> float:
        """``N_V_eq^extra`` from raw oracle counters.

        Generic extra potential calls count one for one. Structured chord units
        are converted through the measured kernel cost.
        """
        equivalent = float(counters.get("n_extra_potential", 0.0))
        chord_units = float(counters.get("n_structured_extra_chord_units", 0.0))
        particle_calls = float(
            counters.get("n_structured_extra_particle_calls", 0.0))
        if chord_units or particle_calls:
            if (self.structured_seconds_per_particle_chord is None
                    or self.structured_fixed_seconds_per_particle is None):
                raise ValueError(
                    "structured chord units were recorded but this calibration "
                    "has no structured-kernel measurement")
            seconds = (self.structured_fixed_seconds_per_particle * particle_calls
                       + self.structured_seconds_per_particle_chord * chord_units)
            equivalent += seconds / self.C_V
        return equivalent

    def fee(self, counters: dict) -> float:
        """``N_FEE = N_F + rho * N_V_eq^extra``."""
        return (float(counters.get("n_force", 0.0))
                + self.rho * self.extra_potential_equivalent(counters))

    def cost_row(self, counters: dict) -> dict:
        """Raw and derived cost fields for ``cost_timeseries.csv``."""
        equivalent = self.extra_potential_equivalent(counters)
        return {
            "n_potential_only": int(counters.get("n_potential_only", 0)),
            "n_potential_baseline": int(counters.get("n_potential_baseline", 0)),
            "n_force_only": int(counters.get("n_force_only", 0)),
            "n_value_and_force": int(counters.get("n_value_and_force", 0)),
            "n_structured_extra_chord_units": int(
                counters.get("n_structured_extra_chord_units", 0)),
            "n_structured_extra_particle_calls": int(
                counters.get("n_structured_extra_particle_calls", 0)),
            "n_force": int(counters.get("n_force", 0)),
            "n_extra_potential": int(counters.get("n_extra_potential", 0)),
            "n_extra_potential_equivalent": equivalent,
            "rho": self.rho,
            "n_fee": self.fee(counters),
            "fee_calibration_hash": self.hash,
            "fee_cost_unit": self.cost_unit,
        }


def _round_significant(value, digits: int = 6):
    """Round to a fixed significant-digit budget so that re-measuring the same
    machine reproduces the same hash despite last-bit timing jitter."""
    if value is None:
        return None
    value = float(value)
    if value == 0.0 or not math.isfinite(value):
        return value
    magnitude = math.floor(math.log10(abs(value)))
    return round(value, -(magnitude - digits + 1))


def _time_median(callable_, *, warmup: int, repetitions: int,
                 device: torch.device) -> float:
    for _ in range(warmup):
        callable_()
    synchronize(device)
    samples = []
    for _ in range(repetitions):
        synchronize(device)
        start = time.perf_counter()
        callable_()
        synchronize(device)
        samples.append(time.perf_counter() - start)
    return float(statistics.median(samples))


def calibrate(target, *, particle_batch_size: int = 4096,
              chord_counts: tuple[int, ...] = (16, 64, 128),
              warmup: int = DEFAULT_WARMUP,
              repetitions: int = DEFAULT_REPETITIONS,
              generator_seed: int = 20260805) -> FEECalibration:
    """Measure ``C_V``, ``C_F``, and any structured chord kernel on this host.

    Both costs are amortized per configuration. Counting is switched off, so a
    calibration never pollutes a run's oracle counters.
    """
    device = target.device
    dtype = target.dtype
    dimension = target.d
    generator = torch.Generator(device=device)
    generator.manual_seed(int(generator_seed))
    x = torch.randn(int(particle_batch_size), dimension, generator=generator,
                    device=device, dtype=dtype)

    potential = target.potential
    with target.no_count():
        seconds_value = _time_median(lambda: potential.V(x), warmup=warmup,
                                     repetitions=repetitions, device=device)
        seconds_force = _time_median(lambda: potential.grad_V(x), warmup=warmup,
                                     repetitions=repetitions, device=device)
    cost_value = seconds_value / particle_batch_size
    cost_force = seconds_force / particle_batch_size

    structured_fixed = None
    structured_per_chord = None
    measured_counts: tuple[int, ...] = ()
    if getattr(potential, "structured_value_delta", False):
        structured_fixed, structured_per_chord, measured_counts = (
            _calibrate_structured(target, x, chord_counts, warmup=warmup,
                                  repetitions=max(20, repetitions // 4),
                                  generator=generator))

    provenance = device_provenance(device, dtype)
    return FEECalibration(
        device=device.type,
        device_name=provenance.get("gpu_name") or provenance.get("cpu_model"),
        device_index=provenance.get("device_index"),
        device_uuid=provenance.get("gpu_uuid"),
        dtype=str(dtype).replace("torch.", ""),
        software_version=software_version_key(device, dtype),
        target_implementation=f"{type(potential).__module__}."
                              f"{type(potential).__qualname__}",
        particle_batch_size=int(particle_batch_size),
        shifted_potential_batch_shape=tuple(int(c) for c in chord_counts),
        warmup=int(warmup),
        repetitions=int(repetitions),
        synchronization=True,
        statistic="median",
        cost_unit=COST_UNIT,
        C_V=cost_value,
        C_F=cost_force,
        rho=cost_value / cost_force,
        structured_fixed_seconds_per_particle=structured_fixed,
        structured_seconds_per_particle_chord=structured_per_chord,
        structured_chord_counts_measured=measured_counts,
        provenance=provenance,
    )


def _calibrate_structured(target, x, chord_counts, *, warmup, repetitions,
                          generator):
    """Least-squares fit of ``t/particle = a + b * n_chords``."""
    potential = target.potential
    n_particles = x.shape[0]
    counts, per_particle_seconds = [], []
    with target.no_count():
        for n_chords in chord_counts:
            n_chords = int(n_chords)
            if n_chords < 1:
                continue
            site_shift = 0.1 * torch.randn(n_chords, 2, generator=generator,
                                           device=x.device, dtype=x.dtype)
            shifts = (site_shift.unsqueeze(1)
                      .expand(n_chords, potential.n_sites, 2)
                      .reshape(n_chords, potential.d).contiguous())
            seconds = _time_median(
                lambda s=shifts: potential.value_delta(x, s),
                warmup=warmup, repetitions=repetitions, device=x.device)
            counts.append(float(n_chords))
            per_particle_seconds.append(seconds / n_particles)
    if len(counts) < 2:
        raise ValueError(
            "the structured kernel needs at least two chord counts to separate "
            "its fixed and per-chord costs")
    mean_count = sum(counts) / len(counts)
    mean_seconds = sum(per_particle_seconds) / len(per_particle_seconds)
    covariance = sum((c - mean_count) * (s - mean_seconds)
                     for c, s in zip(counts, per_particle_seconds))
    variance = sum((c - mean_count) ** 2 for c in counts)
    slope = covariance / variance
    intercept = mean_seconds - slope * mean_count
    # A negative fitted slope or intercept is timing noise, not a negative cost.
    return (max(intercept, 0.0), max(slope, 0.0),
            tuple(int(c) for c in counts))


# ---------------------------------------------------------------- caching
def load_or_calibrate(target, cache_dir: str | Path, *, refresh: bool = False,
                      **kwargs) -> FEECalibration:
    """Reuse a matching cached calibration, or measure and store a new one.

    The cache key covers the device, dtype, numerical stack, target
    implementation, and the declared workload shape, so a different GPU, dtype,
    torch build, or target never silently shares a ``rho``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    potential = target.potential
    device_record = device_provenance(target.device, target.dtype)
    key_payload = {
        "device": target.device.type,
        "device_index": device_record.get("device_index"),
        "device_uuid": device_record.get("gpu_uuid"),
        "device_name": device_provenance(target.device, target.dtype).get(
            "gpu_name") or device_provenance(target.device,
                                             target.dtype).get("cpu_model"),
        "software_version": software_version_key(target.device, target.dtype),
        "target_implementation": f"{type(potential).__module__}."
                                 f"{type(potential).__qualname__}",
        "particle_batch_size": int(kwargs.get("particle_batch_size", 4096)),
        "chord_counts": list(kwargs.get("chord_counts", (16, 64, 128))),
        "warmup": int(kwargs.get("warmup", DEFAULT_WARMUP)),
        "repetitions": int(kwargs.get("repetitions", DEFAULT_REPETITIONS)),
    }
    key = hashlib.blake2b(
        json.dumps(key_payload, sort_keys=True,
                   separators=(",", ":")).encode("utf-8"),
        digest_size=12).hexdigest()
    path = cache_dir / f"fee_{key}.json"
    if path.is_file() and not refresh:
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored.pop("fee_calibration_hash", None)
        stored["shifted_potential_batch_shape"] = tuple(
            stored["shifted_potential_batch_shape"])
        stored["structured_chord_counts_measured"] = tuple(
            stored["structured_chord_counts_measured"])
        return FEECalibration(**stored)
    calibration = calibrate(target, **kwargs)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(calibration.to_dict(), indent=2, sort_keys=True),
                   encoding="utf-8")
    tmp.replace(path)
    return calibration


def assert_comparable(calibrations, *, compatibility_record: dict | None = None
                      ) -> str:
    """Refuse to place runs with different calibrations on one FEE axis.

    Returns the shared hash when every calibration agrees. When they differ, an
    explicit compatibility record listing all the hashes involved -- and stating
    that the workload and cost unit are comparable -- is the only way through.
    """
    hashes = sorted({calibration.hash if isinstance(calibration, FEECalibration)
                     else str(calibration) for calibration in calibrations})
    if len(hashes) == 1:
        return hashes[0]
    if compatibility_record is None:
        raise ValueError(
            "runs with different fee_calibration_hash values cannot share a FEE "
            f"axis: {hashes}. Supply a compatibility record proving the "
            "calibration workload and cost unit are comparable.")
    declared = sorted(str(h) for h in compatibility_record.get("hashes", ()))
    if declared != hashes:
        raise ValueError(
            f"the compatibility record covers {declared}, but the runs use "
            f"{hashes}")
    if not compatibility_record.get("cost_unit_comparable"):
        raise ValueError(
            "the compatibility record does not assert a comparable cost unit")
    if not compatibility_record.get("workload_comparable"):
        raise ValueError(
            "the compatibility record does not assert a comparable workload")
    return compatibility_record.get("merged_axis_label", "+".join(hashes))
