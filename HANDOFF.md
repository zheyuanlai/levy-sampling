# Refactor handoff — open questions and findings

Branch `refactor`. The restructuring specified in the review is implemented:
run/plot split, configuration as real run input, atomic per-variant results,
correct tamed MALA/PT proposal densities, iid `LSC-CP-RA(A)` with a shared bank,
FEE from measured oracle counters, per-seed RNG streams, the four references,
and the figure specifications. 136 tests and 246 release-validation checks pass.

No production run has been executed. Everything below was measured on reduced
but genuine runs.

This document covers only the things that need **your decision** or that
**differ from the specification**, plus the timing you will need to plan the
production campaign.

---

## 1. Decisions needed

### 1.1 An E4 acceptance gate will likely fail in production

`snis_run_agreement_susceptibility` and `snis_run_agreement_phase_probability`
are evaluated as the **maximum over 24 correlated standardized comparisons**
(4 SNIS runs × 4 phases / quantities) against a fixed 2-SE bound.

- Per-comparison two-sided significance: 0.0455
- Family-wise false-failure probability under independence: **0.67**
- Bonferroni-equivalent bound: 3.11 SE

At reduced budget, `snis_run_agreement_phase_probability` fails on the
**unmodified** proposal as well, and the statistic's distribution does not
improve with budget, so it will probably fail in production too. This is the
bound, not the sampler.

The acceptance file is frozen, so the code implements the literal reading and
does not adjust the threshold. Each of these gate records now carries
`n_comparisons`, `per_comparison_two_sided_significance`,
`family_wise_significance_under_independence`, and
`bonferroni_equivalent_threshold_in_se` so the inflation is visible rather than
inferred.

**Decision:** is the intended reading per-quantity (each comparison against
2 SE) or family-wise (the maximum against a multiplicity-corrected bound)? If
the latter, `E4_reference_acceptance.yaml` needs the corrected bound.

### 1.2 E2's coverage reference line was changed

The specification says to draw `EMC*` as the reference line on the coverage
panel, and not to default to 1. Measured on the exact mixture:

| quantity | value |
|---|---|
| asymptotic `EMC*` (4×10⁶ bank) | 0.9999865 |
| χ² of `p*` against uniform | 39.70 on 39 dof |
| plug-in entropy bias at that bank size | 1.32×10⁻⁶ |
| `1 − EMC*` | 1.35×10⁻⁶ |

The descriptor masses are **not resolvably different from 1/40** at this bank
size, and the entire deficit of `EMC*` below 1 is plug-in estimator bias rather
than a property of the target.

That matters because normalized entropy is biased low by roughly
`(K−1)/(2N log K)`, and at the production particle count the bias is far larger
than the reference deficit:

| N | coverage of an **exact** sample | predicted bias |
|---|---|---|
| 2500 | 0.99795 ± 0.00050 | 0.00211 |
| 6000 | 0.99913 ± 0.00020 | 0.00088 |

So a sampler drawing perfectly from the target would sit four standard
deviations below the asymptotic `EMC*` line purely from estimator bias.

**What the code does now:** the figure draws a band measured from exact samples
of the same size (`MoG40Reference.emc_at_sample_size`), with the asymptotic
`EMC*` kept as a faint context line. This is the E2 analogue of the
reference-versus-reference sampling floor E1 reports for `W₂`.

**Decision:** confirm this substitution, or say if you want the asymptotic line
as the primary reference despite the bias.

### 1.3 Untamed LSC has no admissible timestep

On **E1, E2, and E3**, canonical (untamed) `LSC-CP`, `LSC-CP-RA`, and
`LSC-CP-RA (A=4)` are unstable at every timestep on the dyadic refinement grid.

Mechanism: the Lévy score saturates its magnitude cap (`M_MAX = 600`) for
2–3% of particles, the untamed drift throws them out of the numerical box, and
the boundary-rejection fraction sits near 11% and **does not fall as `dt`
shrinks**.

These variants are recorded with status `uncalibratable` and the diagnosis
"unstable at every timestep tried (boundary_reject_fraction) and it does not
improve as the timestep shrinks". The run continues; the other variants are
unaffected. Nothing is hidden and nothing is silently substituted.

**Consequence for the figures:** a canonical-only panel silently loses the LSC
family. Figure E1.1 was therefore switched to the tamed row. This costs the
comparison nothing, because at cap 1.0 taming is inactive for the local methods
on E1 — canonical and tamed ULA agree to four significant figures on every
metric.

**Decision:** confirm that the canonical-vs-tamed ablation is expected to report
"canonical LSC does not run" as its result, rather than this indicating a
configuration problem on our side.

---

## 2. A finding that changes the E4 reference

### The E4 chain cannot host a domain wall

`snis_coverage_nonzero_kink` was failing with a count of exactly zero, and the
prescribed escalation is "improve the SNIS proposal". Implementing the
kink/antikink construction is what showed why the region is not what its name
suggests.

The chain's own wall profile is

```
width = sqrt(kappa/2) / delta = sqrt(2.5/2) * 12 = 13.4 sites
```

against `N_s = 12`. A wall is **wider than the whole chain**, so there is
nothing for it to sit on. Verified directly: all twelve tanh kink/antikink
candidates (4 adjacent phase pairs × wall separations {3, 6, 9}) relax under
gradient descent to kink density **exactly zero**. Unrelaxed, their means sit at
`beta·ΔV = 320–890`, i.e. numerically zero importance weight, so covering them
would make the counter nonzero while leaving the arm just as blind.

The PT reference agrees. Its nonzero-kink sample has `V = 2.14`, all
`q_x ≈ −0.85`, all `q_y ≈ 0`, and coherence **below** the bank mean. The sites
straddle zero, so the nearest-minimum labels split. It is a **collective-flip
transition state, not a domain wall**.

**What the code does now:** the proposal covers that region with the four
homogeneous phase-boundary saddles `1_{N_s} ⊗ v_saddle`, at
`beta(V − V_min) = 8.2–8.4`, matching the region's measured occupancy. Each has
exactly one negative Hessian eigenvalue (the collective flip), handled by
spectral reflection with the reflected eigenvalues recorded, never silently
clamped. The kink/antikink branch stays live in the code: a surviving pair would
be admitted with its exact Hessian.

Result, at matched reduced budget:

| | before | after |
|---|---|---|
| `snis_coverage_nonzero_kink` | 0 | 4321 |
| weighted effective count in region | — | 3545 |
| weighted mean kink density | — | 6.6×10⁻⁵ (PT arm: 4.6×10⁻⁵) |
| IS-ESS fraction | 0.534 | 0.699 |
| max normalized weight | 9.1×10⁻⁴ | 3.4×10⁻⁴ |

No trade-off: the weight diagnostics improved, because the saddle components are
broader than the four tight coherent ones and fill inter-basin tails the
coherent block under-covered. Consistent over six seed bases. All ten
cross-check gates still pass.

**Implication for the write-up:** if the paper describes this gate as covering
"domain-wall configurations", that description is wrong for `N_s = 12`. The gate
is still meaningful — it detects the transition-state region — but its physical
interpretation differs from its name.

---

## 3. Verified numerics

Two independent checks that the cost accounting is real rather than assumed.

**Oracle counters land exactly on theory.** Cost is recorded when a sampler
calls the target, never inferred from `steps × particles`:

| experiment | method | extra potential per particle-step | theory |
|---|---|---|---|
| E3 | LSC-CP | 512 | `J·Q_θ = 32 × 16` |
| E3 | LSC-CP-RA (A=4) | 64 | `A·Q_θ = 4 × 16` |
| E4 | LSC-CP | 1024 | `J·Q_θ = 64 × 16` |
| E4 | LSC-CP-RA (A=4) | 64 | `A·Q_θ = 4 × 16` |

On E4 these land in the **structured** chord counter, never the generic
potential counter, because the chain uses an exact moment kernel; FEE converts
them through the measured kernel cost.

**E1 metrics are physically coherent** (1500 particles × 4 seeds, T = 20):

| variant | W₂ (mean ± sd over seeds) | KS |
|---|---|---|
| FLA α=1.7, tamed | 0.084 ± 0.047 | 0.039 |
| LSC-CP, tamed | 0.107 ± 0.010 | 0.034 |
| LSC-CP-RA, tamed | 0.147 ± 0.039 | 0.036 |
| Raw-CP, canonical | 0.202 ± 0.036 | 0.050 |
| ULA / ULD | 1.287 / 1.296 | 0.493 / 0.498 |

ULA and ULD never cross the barrier in `T = 20` (initialised in the left well),
so `KS ≈ 0.5`: half the mass is missing. Raw-CP crosses but stays biased. FLA
reaching a lower `W₂` than LSC-CP while not preserving the target is the same
behaviour the previous manuscript reported on E1.

---

## 4. Production timing (measured, one H200)

Per-step wall time for the heaviest method (`LSC-CP`) at production ensemble
size, and the implied cost of **one variant**:

| experiment | steps | particles | GPU ms/step | GPU | CPU ms/step | CPU |
|---|---|---|---|---|---|---|
| E1 | 20 000 | 48 000 | 2.4 | 0.01 h | 160 | 0.9 h |
| E3 | 40 000 | 16 000 | 11.5 | 0.13 h | 802 | 8.9 h |
| E4 | 50 000 | 8 000 | 6.0 | 0.08 h | 278 | 3.9 h |
| E2 | 10 000 | 20 000 | 351 | 0.97 h | 21 346 | 59.3 h |

**CPU is not viable** — roughly 60× slower. E2 dominates because its annulus
score quadrature is `q_θ × q_ρ × m_φ = 16 × 4 × 64 = 4096` chords per particle
per step.

Full matrix on one GPU: roughly 7–9 hours, dominated by E2, plus the E4
reference (~25 min of PT-MALA at production length, plus SNIS and two
2000-replicate bootstraps).

To run:

```bash
conda env create -f environment.yml && conda activate jcp-levy-release
python scripts/run_experiment.py E1          # then E3, E4, E2
python scripts/build_catalog.py --all results/
```

`--device auto` picks CUDA when available and CPU otherwise; `--device cuda:1`
pins a specific device. There is no GPU allow-list and no environment variable
that can forbid a run.

---

## 5. Smaller deviations from the specification, and why

- **Checkpoint schedule is defined in simulation time, not steps.** Canonical
  and tamed variants calibrate to different timesteps, so a step-based schedule
  put paired variants on different time grids and the paired ablation compared
  curves sampled at different places. Over a factor-of-four timestep change both
  variants now land on the same 221 checkpoint times to within half a step.

- **MALA/PT timestep selection searches upward as well as downward.**
  Acceptance falls as `dt` grows, so a refinement loop that only halves can
  never satisfy an acceptance *band* — it can only push acceptance higher. The
  acceptance search runs first, then the agreement refinement starts from an
  efficient timestep. During refinement only a *collapsed* acceptance counts as
  an instability.

- **Timestep agreement is compared within Monte Carlo error.** Two pilots at
  different timesteps are independent samples, so their difference is
  discretisation bias plus noise. The tolerance is widened by the combined
  bootstrap standard error, and the statistics are robust (median, IQR) rather
  than standard deviations — the uncorrected methods put real mass in the
  quartic tails, where a sample standard deviation of the energy wanders tens of
  percent between pilots and cannot detect a trend.

- **`jitter_sigma` is deleted from the schema, not defaulted to zero**, and any
  unrecognised key in the E4 jump-law block is now an error, so a stale config
  cannot reintroduce it silently.

- **Wall-clock is not recorded as a scientific metric**, so the untimed warm-up
  steps were removed as dead weight.

---

## 6. What is not done

- **No production run.** All runs so far were deliberately reduced.
- **The E4 reference has not been built at production budget.** At ~10% budget
  15 of 73 gates fail; 13 of those are the PT R-hat / half-run / block-length
  gates, which are budget artifacts (observed block length 90 saved checkpoints
  against a maximum `tau_int` of 44.6, so production gives ~100 blocks per chain
  and passes). The remaining two are §1.1.
- **Old results are removed from the branch** (306 files, 225 MB, twelve of them
  naming the deleted component-stratified estimator). Recoverable from history
  with `git checkout 619f141 -- results figures`.
