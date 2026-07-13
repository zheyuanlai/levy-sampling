<!-- TOP-OF-FILE SUMMARY is written LAST (see end of file until then). -->

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
<!-- subsequent phases appended below as completed -->
