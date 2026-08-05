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

*Pending the production campaign.* The smoke and pilot stages establish that
every arm constructs and runs, and already show one signal clearly enough to
record here: **\(\nu\)-24 drives the Lévy score into its numerical cap two orders
of magnitude more often than the manuscript's structured law does** — a
score-clip fraction of \(1.6\text{–}1.9\times10^{-2}\) against the \(10^{-2}\)
threshold the repository gates on and the \(2.1\times10^{-4}\) the manuscript
reports for E4 — while \(\nu\)-2 stays between \(10^{-4}\) and
\(5\times10^{-3}\). The clip fraction also grows with bank size, from
\(2.3\times10^{-3}\) at \(A=1\) to \(1.6\times10^{-2}\) at \(A=128\), which is
what a larger bank should do if the score's magnitude is set by the most
favourable draw in it.

Because the cap is inert, this is a statement about the correction, not about
the arithmetic: under an incoherent 24-dimensional jump law, the corrected drift
spends a substantial fraction of its time fully saturated, moving one cap length
per step along the reverse chord. It is a conveyor, not a gentle correction.

A box-sensitivity control (the reference point of each design repeated with the
box sized for two jumps instead of one) is included so that this cannot be
dismissed as an artifact of the numerical wall.
