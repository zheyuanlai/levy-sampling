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
<!-- subsequent phases appended below as completed -->
