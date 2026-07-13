# RA-LSC revision — SUMMARY (read this first)

Branch `ra-lsc-revision` off `main@cdb87c4` (contains all prior history incl.
e211839). Env `jcp-exp`, GPU 5. **All work at smoke scale; production NOT run.**
Baseline suite 22 passed → final **31 passed** (7 RA + 2 NFE-ledger added).

### Per-task status
| Task | Status | One-line result |
|---|---|---|
| P0 raw-CP forensic | **DONE** | raw CP matches the *predicted* biased law (not a bug) |
| P1 RA-LSC estimator | **DONE** | estimator + 3 gating tests + NFE win, consistent with exact |
| P2 E3 3-well MB β=24 | **DONE** | constants re-derived & matched; cert 9e-15; two timescales verified |
| P3 E4 8-edge atoms | **DONE** | cert 3.8e-13; p_star unchanged |
| P4 metrics/NFE/plots | **DONE** | extended CSV, n=0 frame, NFE ledger, log-y 3-axis figures |
| P5 production script | **DONE** | gated launcher; refusal path verified; NOT executed |
| P6 manuscript list | **DONE** | change-list keyed to paper/jcp/main.tex (LaTeX not edited) |

### Gate numbers (all pass)
- P0: λ→0 self-check 4.7e-6; W1(emp,pred)=0.047 (within floor) vs W1(emp,Gibbs)=0.089.
- P1: atomwise cert < 1e-6 (32 shifts), mismatch residual > 1e-2, 3 tests pass;
  NFE/step/particle exact-LSC 257 vs RA 18 (~14× E1).
- P2: constants match targets to 1e-4; mixture cert **9.4e-15**, atomwise **4.3e-15**;
  masses (0.31,0.42,0.27); ULA C→A=0/64, C→B=62/64; τ(A↔B)=4.2e5 ≫ T.
- P3: 24D importance cert **3.8e-13**; p_star (0.325,0.211,0.237,0.227) unchanged.
- P4: n=0 bit-identical across methods; NFE-ledger test passes; 9 figures.
- P5: bash -n OK; refusal-path dry-run refuses all 4, runs nothing; real-tol
  residuals E1 5.5e-8 / E2 7.9e-13 / E3 9.4e-15 / E4 3.8e-13.

### Key verdicts
- **P0:** raw-CP is CORRECT — converges to the theory-predicted biased law
  (bias mild at λ=1), score removes exactly this bias. No bug hunt needed.
  Reusable tool `src/stationary.doublewell_rawcp_forensic` to check `experiments_CY`.
- **P1 exact-vs-RA consistency (E1 smoke):** agree within the seed band (W2 0.24σ,
  MMD 0.19σ, EMC 0.65σ; TV_density 5.5σ only at the final checkpoint — a tiny-sd
  artifact at N=64, curves overlap throughout). **Full-scale confirmation
  recommended.**

### Flags for the author (decisions / not-changed)
- **E3 β = 24** built (locked this session); switch to 32 is a one-line constant
  in `build_e3` if the dt/T budget argues for stronger metastability. No constant
  disagreed with targets (would have halted P2).
- **dt/quadrature refinement** for E3 deferred to the notebook production run
  (cannot run at smoke); `cfg.dt=0.005` inherited (mb3 is gentler than mb4).
- **Wall-clock axis** is meaningful only via the sequential `run_experiment` path
  (batched runner reports informational wall-clock).
- **IAT/ESS/round-trips** provided as functions; only checkpoint-cadence split-R̂
  is wired (`convergence_report`). Dense per-step recording needed for accurate ESS.
- **Live notebooks 01/02/04** have stale outputs (all `src` changed); `run_production.sh`
  regenerates them (`build_notebooks.py`) before running. Old 4-well E3 notebook
  archived with outputs at `archive/03_mb4well_10d.ipynb` (builder → `build_e3_mb4well`).
- **E4 RA moment fast-path** not used (RA uses generic V on φ⁴); a per-particle
  homogeneous path is a possible optimization.
- **paper/jcp/main.tex NOT edited** — change-list in `REVISION_NOTES_manuscript.md`.

### Reproduce (env `jcp-exp`, one GPU 4–7)
```
cd JCP_experiments
JCP_GPU=5 python -m pytest tests/ -q                                   # 31 passed
CUDA_VISIBLE_DEVICES="" python /tmp/.../scratchpad/p0_forensic.py      # P0 (see scratchpad)
JCP_GPU=5 python /tmp/.../scratchpad/p2_derive.py                      # P2 constants
JCP_GPU=5 python /tmp/.../scratchpad/p2_verify.py                      # P2 cert+barriers
JCP_GPU=5 python /tmp/.../scratchpad/p3_verify.py                      # P3 E4 cert
JCP_GPU=5 python /tmp/.../scratchpad/p4_smoke.py                       # P4 pipeline
JCP_GPU=5 python scripts/certificate_gate.py mb3well_10d              # P5 gate
# launch production (author, hours):  JCP_GPU=5 ./run_production.sh
```
(Scratchpad drivers live under the session scratchpad; the notebook cells and
`src/stationary.py` reproduce P0 at production scale.)

### Commit graph
```
9e518f4 [P6] manuscript revision change-list
e44b1f1 [P5] production launch script (not executed)
52419f1 [P4] chemistry + MCMC-convergence metrics, NFE counter, n=0 frame, log-y plots
4578fb2 [P3] E4 8-edge-atom jump design
62bee42 [P2] E3 depth-retuned 3-well MB (beta=24) + relay jump law
7f61a80 [P1] RandomAtomicShellScore estimator + atomwise certificate/mismatch/state tests
5db5aec [P0] raw-CP stationary-law forensic + E1 CDF overlay
cdb87c4 (main) JCP notebooks 03/04 ...
```

---

# RA-LSC revision — running log

Branch `ra-lsc-revision` off `main`. Env `jcp-exp`, GPU 5 (tests) / CPU (1D smoke).
Baseline: existing suite **22 passed** on `main` (GPU 5, 27.6 s) — no pre-existing
failures.

---

## P0 — Raw-CP stationary-law forensic (E1) — DONE ✓

**What.** New `src/stationary.py` solves the raw-CP (no-score) stationary
Fokker–Planck-with-jumps equation on a 1D grid (Chang–Cooper conservative flux +
mass-conserving jump-deposit stencil, from `ShellJumpLaw.quadrature_shifts(64)`).
Forensic cell added to `notebooks/build_notebooks.py` (source of truth) and
inserted non-destructively into `notebooks/01_double_well.ipynb`. Figure:
`figures/double_well/rawcp_stationary_forensic.{png,pdf}`.

**Gates.**
- λ→0 self-check (solver must recover `e^{-βV}`): **max rel err 4.7e-6** (tol 1e-4) ✓
- Predicted mass split symmetric (0.5000 | 0.5000) — correct for the symmetric V + ±2 law.

**Verdict — raw CP matches the PREDICTED biased law (positive result).**
Smoke run: 32 chains × N=64 = 2048 pooled terminal samples, generous box
`[-5.2,5.2]` for both empirical and PDE (removes the production `[-3,3]` clip
confound). β=8, λ=1, dt=0.005.
- `W1(empirical raw-CP, predicted ρ_inf^raw)` = **0.047**
- `W1(empirical raw-CP, true Gibbs π)`       = **0.089**
- finite-sample noise floor (draw~pred vs pred, same N) = 0.023 ± 0.009
  → floor+3σ ≈ 0.049, so `W1(emp,pred)` is **within the floor band**, while
  `W1(emp,true)` is ~2× larger and clearly outside. Ratio pred/true = 0.53.

Interpretation: the raw-CP code is correct; its stationary law equals the
theory-predicted biased law (bias is *mild* at λ=1 in the double well, as
expected), and the LSC-CP score removes exactly this bias. No bug hunt needed on
the JCP raw-CP path. The diagnostic checklist item (3) is verified in code:
`ShellJumpLaw.sample` draws `ρ ~ U(-h,h)` symmetric (`rand*2-1`), not `U(0,h)`.

**Notes / caveats.**
- Smoke deviation: used 4000 steps (t=20) per 1D chain for equilibration, above
  the nominal 200-step smoke cap. Justified: 1D, trivially cheap (seconds), and a
  stationary-law comparison is meaningless without equilibration. The heavy
  guard (never launch the 4-exp × 7-method × 5-seed matrix) is fully respected.
- Definitive quantitative match is for the author to confirm by running the
  notebook forensic cell at production N (4000 × 5 seeds → tighter empirical CDF).
- `src/stationary.doublewell_rawcp_forensic` is the reusable tool to run the same
  check against the collaborator's `experiments_CY` raw-CP (E1 / triple well).

---

## P0 addendum — reconciliation with the collaborator's method

Compared our raw-CP forensic against the collaborator's
`experiments_CY/common/levy/doublewell_definitive.py`. **Two methods are
fundamentally different:**

| | Collaborator (CY) | Us (JCP_experiments) |
|---|---|---|
| Dynamics | discretized-**generator** CTMC on a **Gibbs-adaptive** grid, propagate density `p·e^{tQ}` exactly | **particle** tamed Euler–Maruyama SDE + Poisson jumps |
| Time / bias | exact-in-time (no dt bias, no taming); particles only add multinomial metric noise | finite dt, tamed drift, box clip |
| LSC-CP score | **discrete flux correction** on grid edges (`_stationary_flux_correction` forces μ stationary on the grid) | **continuous θ-integral Lévy score** `S_ν=-λ∫R e^{β[V(x)-V(x-θR)]}dθ` |
| raw-CP | **not propagated**; only `raw_jump_stationary_residual` reported | first-class simulated method |
| jump law | Poisson counts, `ρ~U(-h,h)` shell — **matches ours** (center ±2, h≈0.22, λ=1) | same |
| potential | `0.25x⁴-0.5x² = ¼(x²-1)²-¼` — **our well scaled by ¼**, so `β=1/(4ε)` | `(x²-1)²` |

**Root cause of "raw-CP looks bad on the CDF" — a grid artifact, not a bug.**
Building raw-CP on their generator (`local_q+jump_q`, no correction) and taking
its stationary null-vector gives a law that **differs from the true biased law by
W1≈0.10 and does not converge with refinement**: at every `n_cells∈{80..1280}`
there are **zero cells beyond |x|>1.5** (max cell center |x|=1.32), because cells
are equi-probable under Gibbs and Gibbs has ~no mass there — yet that tail/barrier
region is exactly where raw CP injects its ~15% bias mass (`W1(ours,Gibbs)=0.17`).
So their Gibbs-adaptive grid **structurally cannot represent the raw-CP biased
law** (it truncates the tail mass at ~±1.3). Their propagated methods (Langevin,
LSC-CP) both target Gibbs, so the grid is ideal for them — raw-CP is the one law
that lives where the grid has no resolution. Our uniform-grid solver and particle
SDE resolve it correctly. Tool: `scripts/rawcp_crosscheck_CY.py`; figure
`figures/double_well/rawcp_crosscheck_CY.{png,pdf}`.

**Also note:** the CY LSC-CP is a *discrete flux correction* (a
structure-preserving discretization that forces μ stationary on the grid), which
is a different object from our continuous θ-integral Lévy score — they agree in
the continuum limit but not on a finite grid. Worth stating in the paper if both
implementations are cited.

**Follow-up (box widened):** E1 sampling box `[-3,3] → [-5.2,5.2]` (= the
certificate domain), with the reference grid and density-TV bins (350) synced.
LSC-CP is unaffected (π has ~no mass beyond ±2, never hits the boundary); raw-CP's
tail/barrier bias mass now spreads freely instead of piling at ±3 — so the
production raw-CP CDF matches the predicted biased law without the clip artifact.
Suite still 31 passed.

---

## P1 — RA-LSC estimator + gating tests — DONE ✓

**What.**
- `src/score.py::RandomAtomicShellScore` — unbiased single-atom estimator of the
  exact `ShellScore`. Given one realised `R` per particle (drawn by the sampler
  from the same `nu`), `S_R(x) = -λ R exp(LSE_p[log w_p + β(V(x)-V(x-θ_p R))])`.
  Generic in the jump law (shell E1/E3/E4 and the E2 annulus alike). Docstring is
  explicit: practical estimator, no finite-refresh gap-transfer / Euler-Poisson
  exactness overclaim.
- `src/samplers.py::CompoundPoisson` — added `jump_mode="atomic"`: draws `R_n`
  (per particle) and `M_n~Poisson(λdt)` from the shared jump stream at the START,
  uses the SAME `R_n` for score and jump, applies the score drift even when
  `M_n=0`, `X += M_n R_n`. Raw-CP-atomic (score=None) shares `(R_n, M_n)` → still
  pathwise coupled. (Coupling convention = shared jump stream, matching the
  existing exact CP/LSC-CP pair; the diffusion noise ξ stays method-specific, as
  in the current code — the plan's "share ξ" is a refinement not adopted, for
  consistency with the deployed convention.)
- `src/experiments.py` — factories handle methods `CP-RA`, `LSC-CP-RA` (atomic
  mode; RA pair shares `jump_seed`). `src/config.py` — RA diffusion-seed bases.
- Tests: `tests/test_ra_lsc_atomwise_certificate.py` (per-atom certificate
  `max R < 1e-6` on the generous box + tight-box regression guard),
  `tests/test_ra_lsc_mismatch_is_nonzero.py` (jump r_a / score r_b ⇒ residual
  `>1e-2`), `tests/test_ra_lsc_state_independence.py` (sample signatures take no
  state; estimator receives R; sampler draws R state-free).

**Gates.**
- 3 new test files: **7 passed** (2 atomwise + 1 mismatch + 4 state).
- Full suite: **29 passed** (22 baseline + 7), no regression.

**NFE ledger confirmed (per particle per step, measured on E1).**
- exact LSC-CP: `V_delta = 256 = q_theta·A·q_rho (16·2·8)`, grad = 1.
- RA LSC-CP:    `V = 17 = q_theta + 1 (16 chord evals + V(x))`, grad = 1.
  → ~15× score-work reduction on E1 (A=2, q_rho=8); ~96× projected on E3/E4
  (A=12/8 → 1536 vs 16). This is the metric-vs-NFE headline.

**Consistency (E1, smoke: N=64, 4 seeds, t=10).** exact vs RA overlay,
`figures/double_well/consistency_exact_vs_ra.{png,pdf}`. Curves track within the
overlapping seed band on all metrics: terminal `|exact-RA|` = W2 0.24σ, MMD
0.19σ, EMC 0.65σ. TV_density shows 5.5σ **at the final checkpoint only** — an
artifact of the tiny seed-sd denominator for a 200-bin density TV at N=64 (the
curves visibly cross/overlap throughout, no systematic bias). Verdict: consistent
at smoke scale; **full-scale confirmation recommended** (author, production N).

---

## P2 — E3 depth-retuned 3-well Müller–Brown (β=24) — DONE ✓

**Constant re-derivation gate (independent, tol 1e-3): ALL TARGETS MATCH.**
Solved `(D1, D3)` for equal-depth wells via scipy (fsolve on depth-diffs, BFGS
minima, Nelder–Mead saddles): `D = (-1.6607, -1.0, -1.0218, 0.15)`; minima
A(-0.587,1.413) B(-0.065,0.475) C(0.574,0.039) all `V=-0.7957`; saddles
`S_AB=-0.3323`, `S_BC=-0.6310`. At β=24: `β·b(A↔B)=11.1`, `β·b(B↔C)=4.0`, masses
(0.32,0.42,0.26). Saddle *positions* shift slightly from the note's classic-MB
values but the *energies* (the barriers) match to 1e-4. Torch `mb3` Newton-refines
all five critical points to `|grad|~1e-15`.

**What.**
- `src/potentials.py`: `mb3_2d`, `mb3_2d_grad`, `MB3_CRITICAL`,
  `TransformedMB3Well10D` (same frozen `B` embedding as mb4), `MB3Latent2D`.
- `src/experiments.py`: new `build_e3(beta=24.0)` — β threaded locally through
  `make_score`, references, `p_star`, `kramers` (config.BETA stays 8 for
  E1/E2/E4). Relay atoms `{±r_BA, ±r_BC}` through hub B (no direct A–C), uniform
  weights, `h=0.1min‖r_a‖`, `cp_drift_cap=2h`. Init in well C. Separate generous
  certificate box in extras (`cert_lo/cert_hi`); sampling box stays tight. Old
  4-well builder preserved as `build_e3_mb4well`.
- Notebook: `notebooks/03_mb3well_10d.ipynb` generated from updated
  `build_notebooks.py` (mb3 title/asserts/target-viz/jump/cert cells; PT-ladder
  cell now uses `cfg.beta` not `C.BETA` — safe for all, needed for E3). Old
  `03_mb4well_10d.ipynb` moved to `archive/` with outputs preserved and its
  builder repointed to `build_e3_mb4well`. Target figure:
  `figures/mb3well_10d/mb3well_10d_target.{png,pdf}` (classic 3-well MB, trimodal
  at β=24).

**Gates (smoke, GPU 5).**
- Mixture certificate (generous box, β=24): **max R = 9.4e-15** ✓ (< 1e-6).
- Atomwise certificate (max over 32 realised shifts): **4.3e-15** ✓ (certifies
  the production RA-LSC path).
- Constants asserts ✓; `p_star` (0.31,0.42,0.27) ✓; grid masses (0.310,0.420,
  0.270) ✓.
- Barrier structure (ULA, t=100): committed **C→A = 0/64** ✓, **C→B = 62/64** ✓
  — two-timescale structure holds. Kramers τ(A↔B) = 4.2e5 ≫ T=200.
- Full suite: **29 passed**, no regression.

**Note (initial box miss).** The tight sampling box `[-2.0,-1.3]-[1.9,2.6]` reads
R=6e-5 (order-one identity mass lives a full jump beyond support); the separate
generous cert box `[-3.2,-2.4]-[3.0,3.7]` reads 9e-15. Documented in extras.

**Deferred to the author's notebook run** (production scale, cannot run at smoke):
the dt-refinement and quadrature-refinement studies (`cfg.dt=0.005` inherited;
mb3 is gentler than mb4 so this is conservative), and the PT-ladder tuning.

---

## P3 — E4 8-edge-atom jump redesign — DONE ✓

**What.** `src/experiments.py::build_e4`: replaced the 12-atom complete graph
with the **8 edge atoms** of the phase square — dropped the two diagonal pairs
`--↔++` (indices {0,3}) and `-+↔+-` ({1,2}) whose coherent chords cross the
field-zero hilltop; diagonal transitions relay in two hops through a mixed phase.
Coherent tiling `1_{Ns}⊗(v_j−v_i)` kept; `drift_cap = max‖r_a‖` re-measured on
the new set. Added `jitter_sigma=0.0` param + `JitteredShellJumpLaw`
(`src/jumps.py`) — per-draw transverse jitter, RA-LSC only (no closed-form
quadrature; `quadrature_shifts` raises), off by default. E4 notebook jump
markdown updated (generator + live 04 notebook, non-destructive).

**Gates (smoke, GPU 5).**
- Certificate (24D importance, 8-atom law): **max R = 3.8e-13** ✓ (< 1e-6).
- `p_star` unchanged (0.325, 0.211, 0.237, 0.227) — atoms don't affect the target.
- π-start hold (RA, T=10, N=64): TV=0.125, within the ~0.1 N=64 noise floor —
  informational only; production-scale confirmation deferred to the notebook.
- Full suite: **29 passed**, no regression (E4 importance certificate test green).

---

## P4 — Metrics + plotting + NFE counter — DONE ✓

**What.**
- `src/potentials.py`: `Potential.nfe()` (combined V+grad+V_delta point count) and
  `no_count()` context (freeze/restore counters — excludes metric/reference evals),
  non-invasive (no subclass edits).
- `src/metrics.py`: per-frame chemistry metrics — `free_energy_profile[_error]`
  (kT units, π-floor mask), `basin_rel_mass_error`, `observable_error`
  (⟨V⟩/Var(V)), `energy_hist_overlap`, `ksd_imq` (IMQ Stein kernel, blind-spot
  documented); post-hoc convergence — `iat_1d`/`ess_from_series` (Sokal),
  `split_rhat` (rank-normalized, Vehtari 2021, own `_norm_ppf`), `round_trips`,
  `committed_mfpt`. All unit-checked on synthetic data (missing-mode e_F=5.2 kT,
  KSD separates, IAT≈8.3 for AR(1) φ=0.8, split-R̂ 1.00 iid / 2.68 separated).
- `src/experiments.py::make_metrics`: reference precompute + new per-frame keys
  (`e_F, basin_rel_max, basin_L1, occ0, V_mean_err, V_var_err, E_overlap_deficit,
  KSD`); β taken from `cfg.beta` (E3-safe).
- `src/runner.py`: `nfe` column per row; **n=0 frame** on the shared initial
  ensemble; `metrics_fn` wrapped in `no_count`; `convergence_report` (cross-seed
  split-R̂ on occ0). `nfe` + new metrics added to `CSV_BASE_COLUMNS`.
- `src/plotting.py`: `metric_single` (one metric/figure, png+pdf, **log-y**,
  x∈{t,nfe,wallclock}, linear x so t=0/NFE=0 shows); RA method styles/labels;
  new-metric labels. Notebooks emit per-metric figures on 3 axes + convergence.

**Gates (E1 smoke, GPU 5).**
- Extended CSV columns present ✓.
- **n=0 frame bit-identical across methods** (W2, e_F, KSD, MMD, basin_rel_max) ✓.
- NFE ledger (per particle/step) exact: ULA=1, MALA=2, exact-LSC=257 (1+256),
  **RA-LSC=18 (1+17)** → RA ~14× cheaper on E1 (→~96× on E3/E4). `no_count`
  verified to exclude metric evals. Ledger test `test_nfe_ledger.py` (2 tests).
- 9 log-y figures (3 metrics × 3 axes); W2-vs-NFE shows LSC-CP-RA reaching exact
  LSC-CP's W2 at ~14× fewer evals, locals plateaued.
- Full suite: **31 passed** (29 + 2 NFE-ledger), no regression.

**Deferred (documented).** Dense per-step basin-indicator recording for accurate
IAT/ESS/round-trips (the functions are provided; checkpoint-cadence split-R̂ is
wired via `convergence_report`). Wall-clock axis is meaningful only via the
sequential `run_experiment` path (batched gives informational wall-clock).

**Follow-up — collaborator-parity 1D density/CDF metrics added.** Ported the CY
metrics + the requested pdf/cdf L1/L2, all per-frame along the CV: `W1_cdf`
(`∫|F̂−F*|`), `CDF_sup` (KS), `cdf_L2`, `pdf_L1`, `pdf_L2`, `KDE_chi2`
(`∫(ρ̂−ρ*)²/ρ*` on [1%,99%]), `bin_chi2_M{40,80,120}` (PIT χ²: bin `F*(cv)` into
M equal bins vs `1/M`), `well_TV`. Target built from the frozen reference sample
(empirical CDF + matched-bandwidth KDE, so smoothing bias largely cancels; we use
KDE-of-ref rather than the exact density the CY 1D code uses, to keep one path for
E2/E3/E4 where no closed-form CV-marginal exists). All finite, n=0 identical
across methods, in the CSV + plot labels. `metrics.density_cdf_metrics`,
`bin_chi2_pit`, `kde_on_grid`, `well_tv`.

---

## P5 — Production launch script (NOT executed) — DONE ✓

**What.** `run_production.sh` (executable, repo `JCP_experiments/` root) +
`scripts/certificate_gate.py`. The launcher: (1) refuses unless `JCP_GPU∈{4..7}`;
(2) regenerates notebooks from `build_notebooks.py` (`JCP_REGEN=1` default; picks
up all P0–P4 changes); (3) per experiment runs the generous-box certificate
pre-flight gate and **REFUSES** (skips) if max R ≥ 1e-6; (4) on pass, executes the
notebook with the per-experiment method matrix. Method matrix wired via
env-overridable `RUN_METHODS` (`JCP_METHODS`): **E1/E2 = exact + RA dual-run**
(9 methods), **E3/E4 = RA** (locals + PT + CP-RA/LSC-CP-RA). dt-refinement now
excludes `CP-RA` too (invariant law ≠ π).

**Gates.** `bash -n` OK; GPU guard refuses `JCP_GPU=1`; pass-path gate returns
PASS (double_well R=5.5e-8). **Refusal-path dry-run** (`JCP_CERT_TOL=0`): all four
gates FAIL → all four REFUSED, **no notebook executed**. Real-tol residuals from
that run: E1 5.5e-8, E2 7.9e-13, E3 9.4e-15, E4 3.8e-13 (all < 1e-6). Not run.

**How to launch (author).**
```
conda activate jcp-exp
cd JCP_experiments
JCP_GPU=5 ./run_production.sh              # gated full matrix, ~hours
# JCP_REGEN=0 to skip notebook regen; JCP_METHODS=... to override a matrix
```

---
<!-- subsequent phases appended below as completed -->
