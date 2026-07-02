
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch


@dataclass
class FiniteJumpBank:
    """Finite activity jump measure nu(dr)=intensity * sum_e weights_e delta_{r_e}."""

    name: str
    vectors: torch.Tensor
    weights: torch.Tensor
    intensity: float
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vectors = torch.as_tensor(self.vectors, dtype=torch.float32)
        self.weights = torch.as_tensor(self.weights, dtype=torch.float32, device=self.vectors.device)
        self.intensity = float(self.intensity)
        self.validate()

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[-1])

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def rates(self) -> torch.Tensor:
        return self.intensity * self.weights

    def validate(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError(f"jump vectors must be 2D, got shape {tuple(self.vectors.shape)}")
        if self.vectors.shape[0] == 0:
            raise ValueError("jump bank must contain at least one vector")
        if self.weights.ndim != 1 or self.weights.shape[0] != self.vectors.shape[0]:
            raise ValueError("weights must be 1D with one entry per jump vector")
        if not torch.isfinite(self.vectors).all():
            raise ValueError("jump vectors contain nonfinite values")
        if not torch.isfinite(self.weights).all():
            raise ValueError("jump weights contain nonfinite values")
        if (self.weights < 0).any():
            raise ValueError("jump weights must be nonnegative")
        s = float(self.weights.sum().item())
        if s <= 0:
            raise ValueError("jump weights must have positive sum")
        self.weights = self.weights / s
        if self.intensity < 0 or not torch.isfinite(torch.tensor(self.intensity)):
            raise ValueError("jump intensity must be a finite nonnegative scalar")

    def to(self, device=None, dtype=None) -> "FiniteJumpBank":
        return FiniteJumpBank(
            self.name,
            self.vectors.to(device=device, dtype=dtype or self.vectors.dtype),
            self.weights.to(device=device, dtype=dtype or self.weights.dtype),
            self.intensity,
            dict(self.metadata),
        )

    def sample_increment(self, leading_shape: Sequence[int], generator: torch.Generator, dt: float,
                         device=None, dtype=None) -> tuple[torch.Tensor, dict]:
        """Sample compound-Poisson increments via independent per-edge Bernoulli events.

        This is exact up to ignoring multiple firings of the same edge in a single small step;
        configs keep dt*rate small. Diagnostics expose event counts.
        """
        device = device or self.vectors.device
        dtype = dtype or self.vectors.dtype
        vec = self.vectors.to(device=device, dtype=dtype)
        rates = self.rates.to(device=device, dtype=dtype)
        probs = (rates * float(dt)).clamp(0.0, 1.0)
        u = torch.rand((*leading_shape, self.size), generator=generator, device=device, dtype=dtype)
        fire = (u < probs).to(dtype)
        inc = torch.einsum("...e,ed->...d", fire, vec)
        diag = {
            "jump_events": int(fire.sum().item()),
            "jump_event_rate_per_particle": float(fire.sum().item() / max(1, int(torch.tensor(leading_shape).prod().item()) if leading_shape else 1)),
            "max_edge_probability": float(probs.max().item()) if probs.numel() else 0.0,
        }
        return inc, diag


def _maybe_wrap(bank: FiniteJumpBank, cfg: dict) -> FiniteJumpBank:
    """Wrap jump vectors to the minimum image on [-pi, pi) for torus targets (alanine)."""
    if not cfg.get("wrap"):
        return bank
    import math as _m
    bank.vectors = (bank.vectors + _m.pi) % (2 * _m.pi) - _m.pi
    return bank


def _as_minima(minima) -> torch.Tensor:
    m = torch.as_tensor(minima, dtype=torch.float32)
    if m.ndim == 1:
        m = m[:, None]
    return m


def double_well_shell(minima=(-1.0, 1.0), scale: float = 1.0, intensity: float = 1.0) -> FiniteJumpBank:
    m = _as_minima(minima)
    if m.shape[0] != 2:
        raise ValueError("double_well_shell expects exactly two minima")
    r = (m[1] - m[0]) * float(scale)
    vectors = torch.stack([r, -r], dim=0)
    return FiniteJumpBank(
        "double_well_shell",
        vectors,
        torch.ones(2) / 2,
        intensity,
        {"minima": m.tolist(), "scale": float(scale)},
    )


def minima_complete_graph(minima, intensity: float = 1.0, symmetric: bool = True,
                          scale: float = 1.0) -> FiniteJumpBank:
    m = _as_minima(minima)
    vecs = []
    edges = []
    for i in range(m.shape[0]):
        for j in range(m.shape[0]):
            if i == j:
                continue
            if not symmetric and j < i:
                continue
            vecs.append((m[j] - m[i]) * float(scale))
            edges.append((int(i), int(j)))
    vectors = torch.stack(vecs, dim=0)
    return FiniteJumpBank("minima_complete_graph", vectors, torch.ones(len(vecs)), intensity,
                          {"edges": edges, "minima": m.tolist(), "symmetric": symmetric, "scale": float(scale)})


def minima_edge_graph(minima, edges: Iterable[tuple[int, int]], intensity: float = 1.0,
                      symmetric: bool = True, scale: float = 1.0) -> FiniteJumpBank:
    m = _as_minima(minima)
    vecs = []
    out_edges = []
    for i, j in edges:
        vecs.append((m[j] - m[i]) * float(scale)); out_edges.append((int(i), int(j)))
        if symmetric:
            vecs.append((m[i] - m[j]) * float(scale)); out_edges.append((int(j), int(i)))
    vectors = torch.stack(vecs, dim=0)
    return FiniteJumpBank("minima_edge_graph", vectors, torch.ones(len(vecs)), intensity,
                          {"edges": out_edges, "minima": m.tolist(), "symmetric": symmetric, "scale": float(scale)})


def random_matched_length_control(reference_bank: FiniteJumpBank, seed: int = 0,
                                  preserve_lengths: bool = True) -> FiniteJumpBank:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    lengths = reference_bank.vectors.detach().cpu().norm(dim=1)
    dirs = torch.randn(reference_bank.size, reference_bank.dim, generator=gen)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
    if preserve_lengths:
        vectors = dirs * lengths[:, None]
    else:
        vectors = dirs * float(lengths.mean().item())
    return FiniteJumpBank(
        "random_matched_length_control",
        vectors,
        reference_bank.weights.detach().cpu().clone(),
        reference_bank.intensity,
        {"matched_to": reference_bank.name, "seed": int(seed), "preserve_lengths": preserve_lengths},
    )


def manywell_block_flip(n_blocks: int, displacement: float, intensity_per_block: float) -> FiniteJumpBank:
    dim = 2 * int(n_blocks)
    vecs = []
    for b in range(int(n_blocks)):
        v = torch.zeros(dim)
        v[2 * b] = float(displacement)
        vecs.extend([v.clone(), -v.clone()])
    intensity = float(n_blocks) * float(intensity_per_block)
    return FiniteJumpBank("manywell_block_flip", torch.stack(vecs), torch.ones(len(vecs)), intensity,
                          {"n_blocks": int(n_blocks), "displacement": float(displacement),
                           "intensity_per_block": float(intensity_per_block)})


def build_jump_bank(kind: str, potential, cfg: dict) -> FiniteJumpBank:
    """Build a jump bank and optionally override its ``name`` from ``cfg['name']``.

    A ``name`` override is required whenever a config uses two banks of the same ``kind`` (e.g. the
    triple-well adjacent vs overlong edge graphs, or several ``double_well_shell`` scales): the bank
    name is the per-run identifier for CSV rows and sample filenames, so identical names collide.
    """
    bank = _build_jump_bank_impl(kind, potential, cfg)
    override = cfg.get("name")
    if override:
        bank.name = str(override)
    return bank


def _build_jump_bank_impl(kind: str, potential, cfg: dict) -> FiniteJumpBank:
    kind = kind or cfg.get("kind")
    if kind == "none":
        dim = int(getattr(potential, "dim", 1))
        return FiniteJumpBank("none", torch.zeros(1, dim), torch.ones(1), 0.0, {"kind": "none"})
    if kind == "double_well_shell":
        return double_well_shell(potential.minima(), cfg.get("scale", 1.0), cfg.get("intensity", 1.0))
    if kind == "minima_complete_graph":
        bank = minima_complete_graph(potential.minima(), cfg.get("intensity", 1.0), cfg.get("symmetric", True), cfg.get("scale", 1.0))
        return _maybe_wrap(bank, cfg)
    if kind == "minima_edge_graph":
        bank = minima_edge_graph(potential.minima(), cfg.get("edges", []), cfg.get("intensity", 1.0), cfg.get("symmetric", True), cfg.get("scale", 1.0))
        return _maybe_wrap(bank, cfg)
    if kind == "random_matched_length_control":
        base_cfg = dict(cfg.get("reference", {}))
        base_kind = base_cfg.pop("kind")
        base = build_jump_bank(base_kind, potential, base_cfg)
        return random_matched_length_control(base, cfg.get("seed", 0), cfg.get("preserve_lengths", True))
    if kind == "manywell_block_flip":
        return manywell_block_flip(potential.n_blocks, cfg.get("displacement", getattr(potential, "well_sep", 2.0)), cfg.get("intensity_per_block", 0.1))
    raise ValueError(f"unknown jump bank kind: {kind}")
