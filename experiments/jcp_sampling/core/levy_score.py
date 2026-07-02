
from __future__ import annotations

import numpy as np
import torch

from .jump_banks import FiniteJumpBank


def gauss_legendre_01(n: int, device=None, dtype=torch.float32):
    nodes, weights = np.polynomial.legendre.leggauss(int(n))
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    return torch.tensor(nodes, device=device, dtype=dtype), torch.tensor(weights, device=device, dtype=dtype)


def _score_one_chunk(potential_fn, x: torch.Tensor, bank: FiniteJumpBank, beta: float,
                     theta_nodes: torch.Tensor, theta_weights: torch.Tensor,
                     jump_chunk: int, exponent_clip: float, score_clip: float | None):
    vectors = bank.vectors.to(device=x.device, dtype=x.dtype)
    weights = bank.weights.to(device=x.device, dtype=x.dtype)
    Vx = potential_fn(x)
    S = torch.zeros_like(x)
    exp_min = float("inf")
    exp_max = float("-inf")
    clipped = 0
    total = 0
    for j0 in range(0, bank.size, int(jump_chunk)):
        r = vectors[j0:j0 + int(jump_chunk)]
        w = weights[j0:j0 + int(jump_chunk)]
        # xq: (P, Q, M, D)
        xq = x[:, None, None, :] - theta_nodes[None, :, None, None].to(x.dtype) * r[None, None, :, :]
        Vq = potential_fn(xq.reshape(-1, x.shape[-1])).reshape(x.shape[0], theta_nodes.numel(), r.shape[0])
        expo_raw = -float(beta) * (Vq - Vx[:, None, None])
        exp_min = min(exp_min, float(expo_raw.min().item()))
        exp_max = max(exp_max, float(expo_raw.max().item()))
        clipped += int(((expo_raw < -exponent_clip) | (expo_raw > exponent_clip)).sum().item())
        total += int(expo_raw.numel())
        expo = expo_raw.clamp(-float(exponent_clip), float(exponent_clip))
        ratio = torch.exp(expo)
        coeff = (theta_weights.to(device=x.device, dtype=x.dtype)[None, :, None] * ratio).sum(dim=1)
        coeff = coeff * (float(bank.intensity) * w[None, :])
        S = S - torch.einsum("pm,md->pd", coeff, r)
    if score_clip is not None:
        S = S.clamp(-float(score_clip), float(score_clip))
    diag = {
        "levy_score_norm_mean": float(S.norm(dim=-1).mean().item()),
        "levy_score_norm_max": float(S.norm(dim=-1).max().item()),
        "levy_exponent_min": exp_min if total else 0.0,
        "levy_exponent_max": exp_max if total else 0.0,
        "levy_exponent_clipped_frac": float(clipped / max(total, 1)),
        "levy_nonfinite_count": int((~torch.isfinite(S)).sum().item()),
        "levy_quadrature_evals": int(x.shape[0] * theta_nodes.numel() * bank.size),
    }
    return S, diag


def stationary_levy_score(potential_fn, x: torch.Tensor, bank: FiniteJumpBank, beta: float,
                          n_theta: int = 8, theta_nodes=None, theta_weights=None,
                          particle_chunk: int | None = None, jump_chunk: int = 64,
                          exponent_clip: float = 60.0, score_clip: float | None = 100.0,
                          return_diagnostics: bool = False):
    original_shape = x.shape
    flat = x.reshape(-1, original_shape[-1])
    if bank.intensity == 0.0:
        S = torch.zeros_like(flat).reshape(original_shape)
        diag = {"levy_score_norm_mean": 0.0, "levy_score_norm_max": 0.0,
                "levy_exponent_min": 0.0, "levy_exponent_max": 0.0,
                "levy_exponent_clipped_frac": 0.0, "levy_nonfinite_count": 0,
                "levy_quadrature_evals": 0}
        return (S, diag) if return_diagnostics else S
    if theta_nodes is None or theta_weights is None:
        theta_nodes, theta_weights = gauss_legendre_01(n_theta, flat.device, flat.dtype)
    particle_chunk = particle_chunk or flat.shape[0]
    outs = []
    diags = []
    for p0 in range(0, flat.shape[0], int(particle_chunk)):
        S, d = _score_one_chunk(potential_fn, flat[p0:p0 + int(particle_chunk)], bank, beta,
                                theta_nodes, theta_weights, jump_chunk, exponent_clip, score_clip)
        outs.append(S); diags.append(d)
    S = torch.cat(outs, dim=0).reshape(original_shape)
    n = max(1, len(diags))
    diag = {
        "levy_score_norm_mean": float(np.mean([d["levy_score_norm_mean"] for d in diags])),
        "levy_score_norm_max": float(max(d["levy_score_norm_max"] for d in diags)),
        "levy_exponent_min": float(min(d["levy_exponent_min"] for d in diags)),
        "levy_exponent_max": float(max(d["levy_exponent_max"] for d in diags)),
        "levy_exponent_clipped_frac": float(np.mean([d["levy_exponent_clipped_frac"] for d in diags])),
        "levy_nonfinite_count": int(sum(d["levy_nonfinite_count"] for d in diags)),
        "levy_quadrature_evals": int(sum(d["levy_quadrature_evals"] for d in diags)),
    }
    return (S, diag) if return_diagnostics else S


def count_levy_quadrature_evals(n_particles: int, n_theta: int, n_jumps: int) -> int:
    return int(n_particles) * int(n_theta) * int(n_jumps)
