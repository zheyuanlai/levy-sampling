# S1. Does the choice of jump measure matter? A supplementary study for E4

## The question

The manuscript's concluding section argues that the stationarity identity holds
for any finite \(\nu\) meeting its moment assumptions, so *the choice of atoms
affects efficiency, not correctness*. That claim is never tested. Every one of
E1–E4 uses a \(\nu\) built with knowledge of where the target's mass is: E4's
eight atoms are literally the edges of the \(\phi^4\) phase square, each one a
coherent displacement from one phase to an adjacent one. A referee can
reasonably ask whether LSC-CP's advantage in Fig. 8 comes from the score
correction or from having been handed the answer in the jump law.

This study gives LSC-CP a \(\nu\) that knows nothing about the four coherent
phases — specifically, the same symmetric \(\alpha\)-stable increment family
that FLA already uses as the *uncorrected* heavy-tailed control. That turns the
FLA column of Fig. 8 into a controlled comparison: one increment family, with
and without the correction.

## The jump measure

For dimension \(m\) and scale \(\sigma\),

$$\kappa_{\sigma,c} = \mathrm{Law}(\sigma\eta),\qquad
\eta\in\mathbb R^{m}\ \text{with i.i.d.}\ \eta_i\sim S\alpha S(1)\ \text{conditioned on}\ |\eta_i|\le c,$$

with \(\alpha = 1.7\) and \(\lambda = 1\), matching every other experiment. The
coordinates are drawn by `src/samplers.py:sample_sas`, the Chambers–Mallows–Stuck
routine FLA itself uses, so the increment family is literally FLA's.

**The truncation is forced by the theory, not by numerics.** A genuine
\(\alpha\)-stable Lévy measure has infinite activity and is therefore not a
compound-Poisson \(\nu\) at all; the construction of Sec. II needs a finite
measure. Restricting the stable law to a finite shell is what makes it
admissible. Truncating *coordinatewise* rather than radially keeps the law
factorised, which is what allows a deterministic product quadrature in two
dimensions.

Two designs, differing in exactly the way that matters:

| | \(\nu\)-2 (composed) | \(\nu\)-24 (FLA-matched) |
|---|---|---|
| dimension of \(\kappa\) | 2, tiled coherently to all 12 sites | 24, applied directly to the state |
| relation to FLA | same family, composed as a collective move | same family, same application as FLA's noise |
| chain coherence | preserved; the gradient energy is invariant | destroyed; every jump pays the full \(\tfrac{\varkappa}{2\delta}\sum\lVert q_{i+1}-q_i\rVert^2\) |
| homogeneous? | yes, so `CoupledPhi4.V_delta`'s moment-exact fast path is valid | no |
| deterministic quadrature | yes, \(q_u^2\) nodes | no — a product rule would need \(q_u^{24}\) |

The last row is the collaborator's point, and it is a result rather than an
obstacle to engineer around: **in 24 dimensions the Lévy score has no usable
quadrature, so the realised-displacement estimator is not a convenience but the
only way to evaluate the correction at all.**

## What is held fixed

\(\lambda = 1\) and the mean jump length \(\mathbb E\lVert R\rVert\), both matched
to the manuscript's phase-edge atoms (\(\lVert r_a\rVert = 6.9281\)). For
\(\nu\)-2 that means a per-site length of \(6.9281/\sqrt{12} = 2.0\), exactly the
phase spacing. Equal transport budget per unit time, so a difference in outcome
is attributable to the *shape* of \(\nu\), not to how far it moves particles.

\(\sigma\) follows in closed form from a single fixed-seed Monte-Carlo estimate of
\(\mathbb E\lVert\eta\rVert\), since \(\mathbb E\lVert R\rVert = \sigma\,\mathbb
E\lVert\eta\rVert\) is linear in \(\sigma\).

Two knobs are swept around a reference point at \(q = 0.99\), \(L = 1\):

- the retained tail mass \(q = P(|\eta_i|\le c) \in \{0.90, 0.95, 0.99\}\),
  giving \(c = 2.64, 3.47, 7.29\);
- the jump-length multiplier \(L \in \{0.5, 1, 2\}\).

The bank sweep runs at the reference point and the sensitivity points lie on the
two axes through it, so the axes stay separable.

\(q = 0.999\) was considered and dropped: \(c = 26.7\) admits single-coordinate
displacements twelve times the phase spacing, which forces a \(\pm 28\) box and a
31M-cell basin map in order to measure a foregone conclusion.

## Quantities that follow from \(\nu\), and how they were re-derived

Three things in `build_e4` depend on the jump law, and all three are recomputed
rather than inherited.

**Drift cap.** The manuscript uses \(2h\) with \(h = 0.1\min_a\lVert r_a\rVert\),
i.e. \(0.2\min_a\lVert r_a\rVert\), after finding that a cap of
\(\max_a\lVert r_a\rVert\) let the taming-saturated score overshoot the deepest
phase. Expressed in the only scale a law without atoms has, that rule is
\(\text{cap} = 0.2\,\mathbb E\lVert R\rVert\). It is a restatement, not a new
convention: at the matched mean length it reproduces the manuscript's
1.385621 to four decimal places for both designs.

**Sampling box.** `_phi4_sampling_box_design` is unchanged except that it now
accepts the law's own componentwise reach in place of the shell-derived one.

**Basin-map domain.** Pinned to the sampling box. States are clipped into the
box componentwise and \(\bar q\) is a mean of site coordinates, so
\(|\bar q| \le\) the box half-width and the outside-mass diagnostic is
structurally zero — the failure mode that once mislabelled exactly the transport
E4 exists to measure. One shared domain (\(\pm 18\), the widest box in the grid,
at the frozen 0.01 cell size) is used for every configuration, so all designs are
scored against one \(p^\star\). It reproduces the frozen E4 \(p^\star\) to
\(2\times10^{-15}\).

## Arms

| arm | what it is | chord energies per particle per step |
|---|---|---|
| Raw-CP | uncorrected ablation with the identical \(\nu\) | 0 |
| LSC-CP | deterministic product quadrature, \(q_u = 16\); \(\nu\)-2 only | 8192 |
| LSC-CP-RA (\(A\)) | paired i.i.d. bank of size \(A \in \{1,2,4,8,16,32\}\) | \(32A\) |
| LSC-CP-RA (128) | converged reference; \(\nu\)-24 only | 4096 |

For an i.i.d. bank with uniform weights \(1/A\), the atomwise rates
\(\lambda(1/A)\mathrm{d}t\) sum to \(\lambda\,\mathrm{d}t\) and each increment is
distributed as \(\kappa\), so **the jump process is identical in law for every
\(A\)** — only the score's variance changes. \(A = 1\) reduces exactly to the
single-atom estimator. That makes the sweep a clean variance/cost curve rather
than a family of different dynamics.

Each design gets one ground truth, and only where the sweep needs it: \(\nu\)-2
has the deterministic quadrature, \(\nu\)-24 has no quadrature at all and so uses
a converged large bank.

The manuscript's phase-edge columns are read from the frozen
`results/coupled_phi4/` CSVs rather than rerun, so the baseline numbers match the
paper exactly.

## Protocol

Same as E4 — \(d = 24\), 16 seeds \(\times\) 1000 particles, \(T = 100\),
\(\Delta t = 0.002\), \(\beta = 8\), \(q_\theta = 32\) — with two deliberate
differences.

*Wall-clock is not reported.* The study runs on a shared GPU, so timings are not
comparable to the manuscript's dedicated-device protocol. Physical time, NFE and
chord-energy counts are hardware-independent and are reported.

*Chord energies are reported alongside NFE.* `n_Vdelta` charges one unit per
chord point whether it went through the \(O(N_s)\) moment trick or a full lattice
sweep, and the realised-displacement estimators always take the full sweep, so
NFE alone understates their cost relative to the deterministic arm.

## Verification

`scripts/check_e4_jump_design.py` (CPU, ~10 min) — all pass:

- the frozen E4 build path is unchanged: shell half-width 0.6928105591973582,
  box \(\pm 5\), drift cap 1.3856211183947165, eight phase-edge atoms;
- `IIDBankScore` at \(A=1\) reproduces `RandomAtomicShellScore` on the same
  displacement to 0 ulp;
- the \(\nu\)-2 product quadrature self-converges (\(q_u = 24\) differs from
  \(q_u = 32\) by \(5.4\times10^{-3}\)) and agrees with a \(2^{16}\)-atom
  Monte-Carlo score to \(1.3\times10^{-2}\), inside that reference's own
  \(2.9\times10^{-2}\) seed-to-seed spread;
- the applied jump rate is \(A\)-independent, as the estimator argument claims;
- \(\nu\)-24 is refused by both the `V_delta` homogeneity guard and the product
  quadrature; \(\nu\)-2 and the phase-edge law pass both;
- the swapped-law box follows the law's own reach, and the basin domain follows
  `basin_bounds` with no second hard-coded copy left behind;
- the drift-cap rule reproduces \(2h\) at the matched mean length;
- **the value of the \(M_{\max}\) cap does not change the trajectory.** This one
  matters because \(\nu\)-24 drives the score into that cap far more often than
  any manuscript run does, and the study reports that fraction as a measure of
  how large the correction has to be. Raising the cap from 600 to 700 moves the
  chain by \(7\times10^{-9}\) over 150 steps — round-off, against a drift cap of
  1.39. The mechanism is direct: once the score saturates the tamed drift, one
  step moves exactly one cap length whatever the score's magnitude, verified to
  0 ulp on post-jump excursion states.

`scripts/validate_release.py --require-figures` passes and its JSON report is
byte-identical to the pre-study baseline; `git status` shows no modification
under `results/coupled_phi4/` or `figures/`.

## Results

Full campaign: 12 configurations, 36 arms, 16 seeds × 1000 particles, \(T=100\).
Values are terminal \(\mathrm{SW}_2\), mean ± standard deviation across seeds;
\(z\) uses the standard error \(\mathrm{sd}/\sqrt{16}\).

### The two designs answer the question differently

| | Raw-CP | corrected | \(z\) (corrected vs raw) |
|---|---|---|---|
| phase-edge (manuscript, frozen) | 0.4104 ± 0.0211 | **0.1241 ± 0.0310** (exact) | — |
| \(\nu\)-2 composed | 0.4430 ± 0.0178 | **0.1119 ± 0.0293** (exact) | 38.6 |
| \(\nu\)-24 FLA-matched | 0.2480 ± 0.0135 | **0.3017 ± 0.0059** (bank, \(A=64\)) | 14.6, *the wrong way* |

**\(\nu\)-2 vindicates the manuscript's claim.** A jump measure that encodes
nothing about the four coherent phases — heavy-tailed \(\alpha\)-stable
displacements with no knowledge of where the minima are — reaches
\(\mathrm{SW}_2 = 0.1119\), statistically indistinguishable from the
hand-designed phase-edge law's 0.1241 (\(z = 1.2\)) and from its paired
estimator's 0.1002 (\(z = 1.1\)). Raw-CP on the identical \(\nu\) gives 0.4430
(\(z = 38.6\)), so the accuracy comes from the score, not from the jump law.
This is the direct answer to "did LSC-CP just get handed the answer in \(\nu\)?"
— it did not.

**\(\nu\)-24 breaks the finite-step scheme.** With the same increment family
applied incoherently across all 24 coordinates, the corrected dynamics is
*worse* than the uncorrected ablation on the identical \(\nu\) (0.3017 against
0.2480, \(z = 14.6\)) and worse than FLA itself (0.1937, \(z = 13.1\)). More
bank does not rescue it: \(A=1\) gives 0.2651 and \(A=64\) gives 0.3017, moving
away from the answer rather than toward it.

### Why more bank helps one design and hurts the other

| \(A\) | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| \(\nu\)-2 | 0.2522 | 0.2342 | 0.2023 | 0.1917 | 0.1563 | 0.1413 | — |
| \(\nu\)-24 | 0.2651 | 0.2746 | 0.3644 | 0.3572 | 0.3333 | 0.3172 | 0.3017 |

For \(\nu\)-2 the sweep falls monotonically toward the exact quadrature's
0.1119 (\(A=32\) versus \(A=1\): \(z = 9.6\)), reaching 0.1413 at one eighth the
exact arm's chord-energy cost. The bank size behaves exactly as an estimator
knob should.

For \(\nu\)-24 it does not, and the score-clip fraction says why: it rises from
\(2.2\times10^{-3}\) at \(A=1\) to \(1.4\text{–}1.7\times10^{-2}\) for
\(A \ge 4\), against \(2.1\times10^{-4}\) for the manuscript's E4. A larger bank
finds a more extreme atom, which drives the score's log-magnitude higher, which
saturates the tamed drift more often. Since the cap's *value* is inert (verified
above), what this measures is the correction's magnitude, not a numerical
artifact — and on post-jump states 93.8% of particles are already saturated, so
the corrected drift has degenerated into a fixed-length conveyor running back
along the reverse chord.

The continuous-time identity is not in question. What fails is the
discretization: when an incoherent 24-dimensional jump makes the chord density
ratio span hundreds of orders of magnitude, a fixed-step tamed Euler–Poisson
integrator cannot realize the compensating flux. This is the caveat the
manuscript's concluding section already states — *"the finite-step sampler also
contains quadrature, score-cap, taming, and time-discretization errors"* — now
with a measured example of when it bites and a diagnostic (the clip fraction)
that predicts it.

### The result is not an artifact of scale, truncation, or the box

At \(A=8\), across \(q \in \{0.90, 0.95, 0.99\}\) and \(L \in \{0.5, 1, 2\}\),
\(\nu\)-2 stays in 0.167–0.275 and \(\nu\)-24 in 0.226–0.576. The ordering never
crosses. Longer jumps hurt both and hurt \(\nu\)-24 far more: at \(L=2\) its
clip fraction reaches 0.366.

Doubling the box's jump allowance moves every terminal metric by 0.0–3.0%
(\(\nu\)-2 by 0.0% on all three metrics, \(\nu\)-24 by at most 3.0% on basin TV).
The findings are about \(\nu\), not about the numerical wall.

### For the manuscript

Two sentences' worth, if the supplementary material is one paragraph:

> The claim that \(\nu\) affects efficiency rather than correctness was tested on
> E4 by replacing the eight phase-edge atoms with the same symmetric
> \(\alpha\)-stable increment family used by the FLA control, at matched jump
> intensity and mean jump length. Composed as a per-site displacement applied
> coherently across the chain, the \(\alpha\)-stable law recovers the target as
> well as the hand-designed one (\(\mathrm{SW}_2 = 0.1119\) against 0.1241,
> \(z = 1.2\)) while its uncorrected ablation does not (0.4430), confirming that
> the correction and not the jump geometry supplies the accuracy. Applied
> incoherently in all 24 coordinates, as FLA applies its noise, the same family
> instead drives the Lévy score into taming saturation on 94% of post-jump
> states, and the finite-step corrected dynamics becomes less accurate than its
> own uncorrected ablation — a concrete instance of the discretization caveat of
> Sec. V rather than of the continuous-time identity, and one that the
> score-cap activation fraction diagnoses directly.
