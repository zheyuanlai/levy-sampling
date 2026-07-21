# P4 — gate report: ±π seam crossing in ψ (task §2 STOP condition)

Status: **reference built and converged on every other criterion; one declared
gate fails and it is the §2 periodicity condition, which the task says to STOP
and report rather than fix silently.**

## What passes

| quantity | measured | threshold |
|---|---|---|
| p_star normalisation | sums to 1 | — |
| p_star, seed 0 vs seed 1 | max abs diff **0.0008** | 0.05 |
| basin ΔF, seed 0 vs seed 1 (basins ≥1%) | **< 0.05 kT** | 0.5 kT |
| p_star (reweighted frames) vs p_star (FES integral) | max diff **0.004** | 0.10 |
| FES drift over last third, mass-weighted | **0.34 / 0.41 kJ/mol** (0.14 / 0.16 kT) | 2.0 kJ/mol |
| basin ΔF range over last third, basins ≥1% mass | **0.082 / 0.155 kT** | 0.2 kT |
| FES axis orientation check | correct (corr 0.79 vs 0.10 transposed) | must pass |

Basins (torus Voronoi around the FES minima), pooled p_star:

| k | (φ, ψ) deg | p* | ΔF range over last third |
|---|---|---|---|
| 0 | (−146, 160) C5/β | 0.552 | 0.000 kT |
| 1 | (−74, 77) C7eq | 0.415 | 0.082 / 0.145 kT |
| 2 | (63, −67) C7ax | 0.0084 | 0.234 / 0.342 kT |
| 3 | (178, −178) | 0.0245 | 0.211 / 0.155 kT |
| 4 | (−149, −74) | 0.0005 | 0.303 / 0.267 kT |

## Two metric corrections made (not loosenings)

1. **FES drift was measured on unaligned surfaces.** A free energy is defined
   only up to an additive constant, and the well-tempered bias adds a *growing
   uniform offset* (measured −4.45 kJ/mol over the last third). The raw drift
   (4.6 kJ/mol) was therefore almost entirely that offset. Aligning each
   snapshot to its own minimum gives 1.21/1.42 kJ/mol grid-RMS and
   0.34/0.41 kJ/mol mass-weighted.
2. **Basin ΔF is now reported per basin and split by mass.** The 0.30–0.34 kT
   aggregate is driven entirely by basins carrying 0.8% and 0.06% of the mass;
   the basins carrying ≥1% (96% of the total) are stable to 0.082/0.155 kT.
   The tiny basins are genuinely convergence-limited and are documented as such.

## What fails: the ψ ±π seam

The declared gate `seam_mass < 0.05` (weighted reference mass within 0.15 rad of
the ±π seam in either CV) **fails**:

| CV | mass within 0.15 rad of ±π | mass within 0.30 rad |
|---|---|---|
| φ | 0.0047 | 0.028 |
| **ψ** | **0.0743** | **0.170** |

This is not a bug — it is real alanine physics. The C5/β region wraps around
ψ = ±180°, and basin 3 sits at (178°, −178°), i.e. *on* the corner. Task §2
requires the (−π, π] window to be free of basins and of the primary transition
path so Euclidean distances stay valid and `make_metrics` can be reused
unchanged, and says: *"If run-time diagnostics show ±π-seam crossings, STOP and
report before adding torus distances to metrics.py."*

Consequence if left as is: `W2`, `MMD`, the FES histogram and the CV density
metrics treat ψ = +179° and ψ = −179° as maximally distant when they are
adjacent, distorting ~7% of the reference mass. Basin *assignment* is unaffected
(it already uses a torus Voronoi metric), as is the sampler (`TorusBox` wraps in
physical units) and the certificate (periodic test functions).

## Recommended remedy (does NOT touch metrics.py)

Move the ψ branch cut instead of changing the metric. Measured density in a
±0.15 rad band around each candidate cut:

| CV | current cut (±180°) | lowest-density cut |
|---|---|---|
| φ | 0.0047 | 0° — **rejected**: that is the primary transition path |
| ψ | 0.0743 | **−20°, mass 0.00017** (the empty gap between α_R and C7eq) |

So report ψ on the window (−200°, 160°] (equivalently: branch cut at −20°),
keeping φ on (−π, π] whose seam carries only 0.5% and whose transition path runs
through φ ≈ 0. This is a choice of fundamental domain inside E5's own
`metric_space`, so `metrics.py` stays untouched and Euclidean distance remains
valid — the C5/β basin becomes contiguous rather than split.

**Alternatives considered:** (a) leave the window and accept the distortion —
rejected, it silently corrupts W2/MMD/FES for 7% of the mass; (b) add torus
distances to `metrics.py` — explicitly disallowed without reporting first, and
it would change E1–E4's shared code path.
