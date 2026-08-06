# Refactor handoff after independent audit

Date: 2026-08-06
Branch: `refactor`
Baseline commit: `96b5c1d` (`refactor` and `origin/refactor` currently match)
Audit fixes: present in the local working tree; not committed or pushed yet

## Executive conclusion

The collaborator's requested source architecture and scientific method definitions
are implemented. An independent audit found several places where the first
refactor was structurally present but did not yet enforce or persist the full
scientific contract. Those code-level gaps have now been fixed.

Current verification:

- `145 passed` in the full test suite.
- `246/246` source-package validation checks pass.
- The frozen-release validator correctly fails at `254/307` because production
  E1--E4 outcomes, final figures, copied manifests/configs, and executed
  notebooks do not exist yet.
- No production trajectory or production E4 reference was launched during this
  audit.

The correct status is therefore:

> **Source implementation ready for collaborator review; production evidence and
> frozen release not ready.**

This distinction is deliberate. A reduced test run is not being presented as a
scientific result, and empty or unrelated local directories can no longer make
release validation pass.

---

## 1. Independent-audit issues and dispositions

| Issue | Disposition after this audit | Main evidence |
|---|---|---|
| Failed calibration existed only in console output | Fixed. Every `uncalibratable` variant is saved atomically as a first-class negative outcome with resolved config, calibration table, diagnosis, manifest hashes, and `COMPLETE`. Other variants continue. | `src/pipeline.py`, `src/results.py`, `src/catalog.py`, `scripts/run_experiment.py` |
| A failed E4 reference could be reloaded from cache | Fixed. A cached or newly built E4 reference must pass `assert_valid_for_use`; a persisted failed reference remains evidence but cannot enter production. | `src/references/base.py`, `src/references/__init__.py`, `src/references/e4.py` |
| MALA calibration named ESS but did not measure it | Fixed. The second half of each pilot records a per-seed temporal trace, computes ESS, and requires an ESS fraction of at least `0.05`. | `src/calibration.py`, `configs/methods/MALA.yaml` |
| Metropolis calibration did not enforce the full target band | Fixed. The chosen coarse timestep must actually lie inside the frozen acceptance band; high acceptance is no longer silently accepted as a completed calibration. | `src/calibration.py` |
| PT used only aggregate local acceptance and swap acceptance | Fixed. PT saves every replica's MALA acceptance, tracks labelled-walker hot-to-cold round trips, and enforces swap-band plus round-trip gates at the final calibrated local timestep. | `src/samplers.py`, `src/calibration.py`, `configs/methods/PT.yaml` |
| PT could fail a mixing gate before local-`dt` calibration | Fixed. The first ladder pass gates placement only. After local-`dt` calibration, the ladder is always retuned and the mixing gate is enforced. | `src/calibration.py` |
| PT method YAML fields could be ignored | Fixed. The ladder target band, burn fraction, and minimum round trips now come from `configs/methods/PT.yaml`; experiment values are only fallbacks where appropriate. | `src/calibration.py` |
| LSC score certificate was declared but not materialized | Fixed. Full LSC saves a numerical quadrature self-convergence certificate; RA(A) saves the structural iid/full-law/every-step/shared-bank certificate. | `src/calibration.py` |
| Full-LSC quadrature comparison was diagnostic only | Fixed. Base versus doubled quadrature now has hard median and 95th-percentile log-magnitude/direction gates. Failure is saved as `uncalibratable`, with the doubled setting recorded as the next unverified candidate. | `src/calibration.py`, `configs/methods/LSC_CP.yaml` |
| E1 drew a floor only for W2 | Fixed. The reference now freezes production-size sampling floors for `W2_exact_1d`, `MMD2_biased`, and `KS`, using the same MMD bandwidth convention as runtime metrics. | `src/references/e1.py`, `src/plotting.py` |
| E4 runtime metrics omitted required distribution/raw data | Fixed. Added marginal W1, energy KS/W1/biased+unbiased MMD, coherence KS/W1, kink KS/W1/zero/high-tail probabilities, connected correlation vectors/error, phase counts, raw energy/coherence/kink values, and raw/reference heat-capacity and Binder values. | `src/measurements.py`, `src/metrics.py`, `configs/experiments/E4.yaml` |
| E4 SNIS run-agreement maxima used uncorrected 2-SE component thresholds | Fixed and collaborator-approved. The four gate families now use frozen Bonferroni cutoffs preserving the original 2-SE two-sided significance at the family level: approximately 3.11 SE for the two 24-comparison families and 2.67 SE for the two 6-comparison families. | `configs/experiments/E4_reference_acceptance.yaml`, `src/references/e4.py`, `tests/test_references.py` |
| Metric code could change without invalidating old runs | Fixed. Every metric row, resolved config, and complete manifest carries a hash of the executable metric/observable definitions. Manifest scanning rejects stale definitions. | `src/measurements.py`, `src/pipeline.py`, `src/results.py` |
| Snapshots did not carry complete cost identity | Fixed. Snapshots now save raw/equivalent force and extra-potential counts, per-particle values, `rho`, FEE hash, and FEE unit. Resolved config saves the complete FEE calibration and checkpoint costs. | `src/pipeline.py` |
| FEE cache identity could alias different GPUs | Fixed. Actual CUDA index and GPU UUID enter the calibration cache key and calibration hash. Plain `cuda` resolves the actual current device index. | `src/device.py`, `src/fee.py` |
| Requested plot methods could disappear silently | Fixed. Plot loading raises if any method requested after tame/variant filtering is absent. E2.1 and E3.1 use the tamed view so the LSC family is not silently lost. | `src/plotting.py`, `configs/plots/manuscript.yaml` |
| Source validation created `results/` | Fixed. It now probes root writability through a temporary directory and leaves a clean source checkout clean. | `scripts/validate_release.py` |
| Release validation checked only “some directory/some PNG” | Fixed. It now checks the full default outcome matrix, exact default configs rather than reduced runs, run integrity, reference evidence, copied manifest/config hashes, all configured figure formats and tame views, and all eight executed notebooks. | `scripts/validate_release.py` |

### Negative-outcome semantics

A valid outcome now has one of two statuses:

- `complete`: a production trajectory with the full artifact set and current
  metric-definition hash;
- `uncalibratable`: durable negative calibration evidence, with no production
  trajectory falsely implied.

Catalog scanning admits both kinds as evidence. Plot selection defaults to
`status="complete"`, so a negative outcome cannot be mistaken for a curve.

The command-line runner continues through all variants but returns nonzero if
any variant failed or was uncalibratable. This makes batch evidence durable
without making an incomplete campaign appear successful.

---

## 2. Scientific implementation status

### 2.1 Canonical and tamed variants

All registered methods that support taming expand to both variants, including
MALA and PT.

Tamed MALA uses the actual Gaussian proposal

`q_c(y | x) = Normal(y; x + dt b_c(x), (2 dt / beta) I)`,

and recomputes the reverse tamed drift at `y`. Tamed PT applies the same rule
at each replica's own inverse temperature. Swap acceptance is unchanged by the
tame flag.

The tests cover proposal log densities, detailed balance, target moments,
canonical/tamed stream pairing, replica moments, cold-replica correctness, and
swap-formula invariance.

### 2.2 LSC-CP-RA(A)

`LSC-CP-RA` remains one iid estimator family. For every particle and step:

1. `A` atoms are drawn iid from the full normalized jump law;
2. the same empirical Lévy measure drives the score and Poisson increment;
3. the bank is refreshed at the next step.

`A = 1, 4, 8` are parameter variants, not separate methods. The deleted
component-stratified construction is not revived or relabelled.

### 2.3 Calibration thresholds introduced by this audit

These values are now explicit and hashed into calibration identity:

| Gate | Current value |
|---|---:|
| MALA minimum pilot ESS fraction | 0.05 |
| PT minimum post-burn-in round trips | 1 |
| Full-LSC median quadrature tolerance | 0.05 |
| Full-LSC 95th-percentile tolerance | 4 × 0.05 = 0.20 |
| PT ladder pilot length | experiment `calibration.pt.pilot_steps` (20,000 in the defaults) |
| PT ladder burn fraction | 0.5 |

These executable defaults were approved on 2026-08-06. In particular, one
round trip is a minimum nonzero mixing certificate, not a strong claim of PT
efficiency.

The full-LSC quadrature gate compares the configured rule with a doubled rule.
A failing rule is **not** silently replaced by the doubled one; the latter is
recorded as an unverified next candidate. Extending the quadrature grid is
therefore an explicit scientific action.

---

## 3. Scientific decisions: resolved and remaining

### 3.1 E4 multiplicity in four SNIS agreement gates — resolved

The E4 implementation has four maximum-over-run agreement gates. With four
independent SNIS runs there are six run pairs:

- phase probability: 6 pairs × 4 components = 24 comparisons;
- susceptibility: 6 pairs × 4 matrix entries = 24 comparisons;
- energy per site: 6 pairs × 1 scalar = 6 comparisons;
- coherence mean: 6 pairs × 1 scalar = 6 comparisons.

**Approved decision (2026-08-06):** preserve the original two-sided 2-SE
significance, approximately `0.0455003`, at each gate-family level using frozen
Bonferroni cutoffs selected before the production reference is observed:

- `3.1060815350389697` SE (approximately 3.11) for each 24-comparison family;
- `2.6700773593737384` SE (approximately 2.67) for each 6-comparison family.

Each gate record saves its comparison count, family-wise target, configured
cutoff, implied per-comparison significance, and independence-model family-wise
false-failure probability. The PT-versus-SNIS scientific tolerances are
unchanged because they include pre-specified physical absolute/relative floors
and are not merely pairwise run-consistency tests.

### 3.2 E2 EMC reference presentation

The asymptotic exact-bank value is approximately
`EMC* = 0.9999865`, but finite-sample plug-in entropy bias at the production
sample size is much larger than `1 - EMC*`.

Current figure behavior:

- primary comparison: exact samples at the same `N`, shown as a frozen band;
- secondary context: asymptotic `EMC*`, shown faintly.

This is a like-for-like finite-sample comparison and avoids making a perfect
sampler look deficient due only to estimator bias.

**Decision:** confirm this presentation, or make the asymptotic line primary.

### 3.3 Canonical LSC instability and strict plot completeness

Reduced genuine runs previously found canonical LSC variants uncalibratable on
E1--E3: boundary rejection remained high as `dt` shrank because the untamed
score saturated its magnitude cap.

The new code now preserves such outcomes durably. It also refuses to render a
plot specification that requests a missing completed method.

Consequently, if production confirms canonical LSC is uncalibratable:

- tamed snapshot figures remain well defined;
- paired/base curves can show available completed variants;
- a canonical-only curve specification requesting LSC will fail visibly rather
  than silently omit it.

**Decision:** should the canonical-only figure contain an explicit
“uncalibratable” annotation, or should canonical LSC be removed from that
figure's requested method set while its negative result is reported in a table?
The current strict failure forces this decision before release.

### 3.4 Physical interpretation of the E4 “kink” region — resolved

For `N_s = 12`, the estimated wall width is about 13.4 sites, wider than the
chain. Gradient-relaxed kink/antikink candidates collapse to zero kink density.
The PT sample classified as nonzero kink is instead a collective-flip
transition-state configuration.

The SNIS proposal now covers this region with homogeneous phase-boundary
saddles and improved its weight diagnostics. The observable name remains for
schema continuity, but manuscript text should not call this a stable domain
wall for this system size.

**Approved decision (2026-08-06):** retain the metric/schema name
`kink_density` for compatibility, but describe it in the manuscript as a
transition-state/high-gradient fraction. Do not call it a stable domain wall.

### 3.5 E4 production reference gate

No production E4 reference exists. Reduced-budget evidence previously showed
that most failures were chain-length/block-count artifacts. The SNIS
multiplicity rule is now frozen as described above.

The code now guarantees that any failed reference remains unusable. The
production reference must pass every frozen primary gate before any E4
production method uses it.

---

## 4. Verification performed after the fixes

Commands:

```bash
python -m compileall -q src tests scripts
git diff --check
conda run -n jcp-refactor python -m pytest tests/ -q
conda run -n jcp-refactor python scripts/validate_release.py
conda run -n jcp-refactor python scripts/validate_release.py --release
```

Results:

| Check | Result |
|---|---|
| Python compilation | pass |
| Git whitespace check | pass |
| Full tests | **145 passed** |
| Source validation | **246/246 passed** |
| Frozen-release validation | **254/307 passed; expected failure because production/release artifacts are absent** |

The focused regressions added in this audit cover:

- persisted `uncalibratable` outcomes and plot exclusion;
- stale metric-definition rejection;
- complete snapshot/resolved FEE identity;
- FEE hash separation by device index and GPU UUID;
- E1 floors for all three primary metrics;
- E4 required distribution, vector, count, and raw-observable columns;
- cached failed E4 reference rejection;
- requested-method plot completeness;
- per-replica PT acceptance and observed completed round trips.

---

## 5. Why the frozen-release validator currently fails

The validator now expects the exact default campaign, not reduced runs with the
same method labels.

Each experiment has 24 default variant outcomes, so the complete matrix is
`4 experiments × 24 = 96 outcomes`.

An outcome may be `complete` or `uncalibratable`, but its
`resolved_config.yaml` must match the committed default experiment YAML
exactly. A reduced/debug run cannot satisfy the production matrix.

Current blockers:

- E1, E2, and E3 new-format production run directories are absent.
- E4's new-format directory contains no admitted production outcomes or
  production reference evidence.
- `figures/`, `resolved_configs/`, `manifests/`, and
  `executed_notebooks/` are absent.
- None of the 27 configured figure names (including tame-view renditions) exists
  in all four formats: PNG, PDF, SVG, and TIFF.
- The eight executed notebook copies and successful
  `execution_report.json` are absent.

This is correct behavior. The old validator could pass shallow existence checks;
the new validator cannot.

---

## 6. Production and release checklist

Do not start the full campaign until the remaining scientific decisions in
section 3 and the clean-revision requirement below are resolved.

1. Commit the audited implementation and approved E4 multiplicity thresholds,
   so production artifacts record an immutable Git revision. Freeze the
   remaining EMC presentation and canonical-LSC reporting decisions before
   release; they do not alter trajectories.
2. Run the required preflight:
   ```bash
   conda run -n jcp-refactor python -m pytest tests/ -q
   conda run -n jcp-refactor python scripts/validate_release.py
   ```
3. Build the E4 production reference and require a clean validation:
   ```bash
   conda run -n jcp-refactor python scripts/build_reference.py E4
   ```
4. Run the four default experiments, preserving every complete or
   uncalibratable outcome:
   ```bash
   conda run -n jcp-refactor python scripts/run_experiment.py E1
   conda run -n jcp-refactor python scripts/run_experiment.py E2
   conda run -n jcp-refactor python scripts/run_experiment.py E3
   conda run -n jcp-refactor python scripts/run_experiment.py E4
   ```
5. Rebuild catalogs by scanning manifests:
   ```bash
   conda run -n jcp-refactor python scripts/build_catalog.py --all results/
   ```
6. Execute the source notebooks into `executed_notebooks/` as the frozen
   execution record and export all configured figure formats.
7. Populate the top-level `resolved_configs/` and `manifests/` collections
   with exact copies of the selected 96 outcome files. The release validator
   checks file hashes, not just filenames.
8. Require:
   ```bash
   conda run -n jcp-refactor python scripts/validate_release.py --release
   ```
   to pass before building or sharing the frozen archive.
9. Build the archive:
   ```bash
   conda run -n jcp-refactor python scripts/build_release.py --frozen dist/release.zip
   ```

A full run should not delete or overwrite a prior run directory. Superseded
evidence is retained and may be marked invalid, never erased.

---

## 7. Prior reduced-run findings retained for discussion

These numbers were measured before this independent audit on reduced but genuine
runs. They were not rerun here and are not production claims.

### Measured oracle-counter sanity

| Experiment | Method | Extra potential per particle-step | Theory |
|---|---|---:|---:|
| E3 | LSC-CP | 512 | 32 × 16 |
| E3 | LSC-CP-RA (A=4) | 64 | 4 × 16 |
| E4 | LSC-CP | 1024 | 64 × 16 |
| E4 | LSC-CP-RA (A=4) | 64 | 4 × 16 |

For E4 these are structured-kernel counters converted by measured calibration,
not fabricated generic `V()` calls.

### Reduced E1 physical sanity at `T = 20`

| Variant | W2 mean ± SD over seeds | KS |
|---|---:|---:|
| FLA (alpha = 1.7), tamed | 0.084 ± 0.047 | 0.039 |
| LSC-CP, tamed | 0.107 ± 0.010 | 0.034 |
| LSC-CP-RA, tamed | 0.147 ± 0.039 | 0.036 |
| Raw-CP, canonical | 0.202 ± 0.036 | 0.050 |
| ULA / ULD | 1.287 / 1.296 | 0.493 / 0.498 |

These results remain useful as mechanism sanity checks only.

---

## 8. Working-tree and repository safety

- The audit did not edit or delete the protected old ICLR output directories.
- Pre-existing untracked user files/directories were left untouched.
- Only temporary `.orig`/`.rej` files created by the fallback patch utility
  during this session were removed.
- No commit, push, or pull request was created.
- To put this handoff and the fixes on GitHub, review the diff, commit it on
  `refactor`, and push explicitly.

---

# Production campaign, 2026-08-06

This section records the first full production campaign on `refactor`, the
defects it exposed, and the state of the evidence now in the repository.

## Scope actually delivered

**E1, E2 and E3 are complete. E4 is not**, and that is a deliberate, recorded
outcome rather than an omission. Each delivered experiment carries its full
24-outcome default variant matrix.

| Experiment | complete | uncalibratable | admitted |
|---|---:|---:|---:|
| E1 double well | 18 | 6 | 24 |
| E2 40-mode mixture | 16 | 8 | 24 |
| E3 Müller--Brown | 14 | 10 | 24 |
| E4 quartic chain | -- | -- | 0 |

## Reduced scale

Particles, seeds and `final_time` were cut roughly twentyfold from the original
production values, as an explicit temporary edit to the single default
configuration of each experiment. There is still exactly one configuration per
experiment and no second profile. `plot_snapshots.time_values` were rescaled to
match, since they are absolute times and would otherwise request checkpoints
that were never saved. Seeds stay at 4 so seed-level uncertainty remains
estimable.

## Defects found and fixed

1. **The score quadrature certificate measured the wrong quantity.**
   `log_parts` returns `S = -exp(M) v`, so `M` is only the per-particle maximum
   exponent. Doubling the node count halves every quadrature weight, which drops
   `M` by exactly `log 2` and doubles `|v|`, leaving the score unchanged. The
   certificate compared `M` alone and therefore reported a fixed ~0.693
   discrepancy for a perfectly converged rule, failing **every** LSC-CP variant
   on E1--E3. It now compares `log|S| = M + log|v|`. The frozen 0.05 / 0.20
   tolerances are unchanged; only the quantity they apply to is corrected.
   Measured after the fix: 1e-6 at production settings, while `q_theta` of
   2, 4, 8 and 16 are still rejected decisively.

2. **`score.q_theta = 16` was genuinely under-resolved.** Independently
   confirmed twice: `shell_score_dense_theta` (a 200,001-node composite-Simpson
   rule that exists to validate the Gauss--Legendre theta rule) agrees with
   `q_theta` 32 and 64 to seven digits and differs from 16 by 33%; and the
   corrected certificate rejects 16-vs-32 at 0.156 against a 0.05 tolerance.
   Raised to 32 in all four experiments. `q_rho` was already converged.

3. **The PT ladder raised on its non-enforcing pass.** `tune_pt_ladder` accepts
   `enforce_mixing` and folds it into `gate_pass`, but then raised on
   `not gate["pass"]` unconditionally, so the first ladder pass -- invoked with
   `enforce_mixing=False` precisely because it is meant to gate placement before
   the local timestep is known -- could still abort the run. It now raises only
   on the enforcing pass. (Identified by the collaborator.)

4. **The PT swap band was too narrow.** The attainable geometric-ladder swap
   acceptance sits just above 0.40 on these targets, so `[0.2, 0.4]` rejected
   every candidate ladder. Widened to `[0.2, 0.45]`. (Identified by the
   collaborator.)

5. **Release validation contradicted the retire-don't-delete policy.** The
   frozen-release check failed an experiment if the scanner rejected any run
   directory, but superseding a run means marking it `INVALID` and keeping it,
   so every rebuilt campaign necessarily leaves rejected directories behind. The
   check now separates retired runs from unexplained rejections -- a missing
   manifest, a hash mismatch, an unknown schema -- and fails only on the latter.

6. **`resolved_configs/` and `manifests/` had no producer.** The release
   validator hashes both collections but nothing created them.
   `scripts/collect_release_artifacts.py` now does, applying the validator's own
   selection rule so the two cannot drift, and pruning stale copies.

## Uncalibratable outcomes are results, not failures

24 of 72 outcomes are `uncalibratable`, and the pattern is coherent: essentially
every **canonical** (untamed) variant fails, while its tamed counterpart
calibrates. The drift is not truncated, so at the step size the run stage
settles on, canonical variants are genuinely unstable. This is direct evidence
for the necessity of taming.

- canonical CP/LSC (`LSC-CP`, `LSC-CP-RA` at A=1,4,8) on all three experiments:
  boundary rejection, 12 outcomes;
- E3 canonical FLA at every alpha: boundary rejection, 3 outcomes. Measured
  rejection was 6.6% at `dt = 6.25e-4` against a 2% gate, decaying slower than
  linearly toward a nonzero floor, so reaching the gate would need roughly 2.5
  million steps. This is a property of heavy-tailed jumps against a finite box,
  not a grid-length problem;
- PT on E1 (both), E2 (both) and E3 (canonical): 5 outcomes;
- MALA on E2 and E3 (both variants): 4 outcomes. The acceptance search budget
  was fixed at 8 iterations, which could not reach the band from `initial_dt`;
  it is now `calibration.dt.acceptance_search_iterations`, defaulting to 12 and
  set to 14 for MALA. Fixing it removed the arbitrary limit and exposed the real
  constraints, which are different on the two experiments and are both
  substantive:

  * **E2.** The search now reaches `dt = 5.12`, where acceptance is 0.6230 and
    genuinely inside `[0.4, 0.75]`. It fails instead on `temporal_ess_fraction`,
    which is `nan` there, because the pilot runs
    `final_time * time_fraction / dt = 25 * 0.25 / 5.12`, i.e. a single step.
    That is not merely a pilot-length problem: at `dt = 5.12` a production run
    to `final_time = 25` is five steps against a 220-checkpoint schedule, so
    even a passing calibration would yield an unusable trajectory. MALA on this
    40-mode mixture wants a timestep comparable to the whole simulation horizon,
    because the mixture spans a wide domain. Raising `final_time` back to 100
    only reaches twenty steps, so this is not purely an artefact of the reduced
    scale.
  * **E3.** The acceptance band and the ESS gate have **disjoint** timestep
    windows, so no `dt` satisfies both:

    | dt | 0.04 | 0.08 | 0.16 |
    |---|---|---|---|
    | acceptance | 0.710 in band | 0.418 in band | 0.121 outside |
    | ESS fraction | 0.0197 below | 0.0362 below | 0.0840 above |

    The band closes before the ESS gate opens. This is a statement about MALA on
    the 10D Müller--Brown at these pilot settings, not about the search.

  Both are left as recorded negative evidence. MALA appears in no manuscript
  figure, so nothing is silently dropped.

Figures render these as explicit `"<method>: uncalibratable"` legend entries in
the method's own colour, so a negative result is visible rather than silently
absent.

## E4: reference not validated

Five build attempts, none of which passed the frozen acceptance gates. The
failing reference and its complete 73-gate `reference_validation.json` are kept
as evidence with `reference_validated: false`, so no E4 run can cite it, and no
E4 production trajectory exists.

| Attempt | Configuration | Gates failed | Worst |
|---|---|---:|---:|
| 1 | 2M steps, `beta_min` 1.0, 12 replicas | 4 | 4.06 SE |
| 2 | 6M steps, `beta_min` 1.0, 12 replicas | 5 | 5.71 SE |
| 3 | 2M steps, `beta_min` 0.25, 12 replicas | 2 | 4.62 SE |
| 4 | 3M steps, `beta_min` 0.1, 12 replicas, swap 5 | 3 | 3.08 SE |
| 5 | 3M steps, `beta_min` 0.1, **24 replicas**, swap 5 | 3 | 2.82 SE |

Two findings matter for whoever picks this up.

**Extending the run is not always the right escalation.** These gates are
denominated in standard errors, so a longer run shrinks the SE faster than it
removes residual drift: attempt 2 doubled the length and failed *more* gates
than attempt 1. Improving temperature mixing was what actually helped.

**The `pt_half_run_consistency` family has an uncorrected multiplicity
problem.** It applies 16 simultaneous comparisons at a raw 2.0 SE threshold, so
0.73 false failures are expected by chance, and attempt 5's three failures all
land on a single run at 2.82, 2.51 and 2.13 SE with sub-percent physical
differences (energy per site 0.125176 vs 0.125790). Section 3.1 of this document
records that the analogous **SNIS** agreement families already received frozen
Bonferroni cutoffs (3.11 SE and 2.67 SE); the PT half-run family never did. The
matching cutoff for 16 comparisons is **2.984 SE**, under which attempt 5 would
validate.

This correction was **not** applied. Section 3.1 stresses that its cutoffs were
frozen *before* the production reference was observed, and choosing one here
after seeing which gates failed would not carry the same guarantee. The decision
was to leave the gates untouched and report E4 as unresolved. Applying the
correction is a legitimate option, but it belongs to whoever owns the frozen
acceptance criteria, and it should be frozen before the next build is run.

## Release status

`python scripts/validate_release.py` passes 246/246.
`python scripts/validate_release.py --release` passes 304/319. The 15 remaining
failures are all expected consequences of the above:

- 10 are E4 (no runs, no completed trajectory, no E4 figures);
- 5 are executed notebooks: the four `*_run.ipynb` and E4's plot notebook.
  Production trajectories were produced with `scripts/run_experiment.py`, which
  does exactly what a run notebook does; the three E1--E3 plot notebooks execute
  cleanly and regenerate every figure from saved results.

A complete frozen release therefore still requires a validated E4 reference.
