# Manuscript revision change-list (port to `paper/jcp/main.tex`)

The code revision (P0–P5, branch `ra-lsc-revision`) implies the following
manuscript changes. This is a change-list, not a patch: the LaTeX was left
untouched (out of the code-revision scope) so the author can apply these
deliberately. Section names refer to `paper/jcp/main.tex`.

---

## 0. Naming (throughout)

Keep **LSC-CP** as the method name (the proven exact Lévy-score correction).
Introduce the random-atomic variant as an **estimator of the Lévy score**, e.g.
"randomized / random-atomic estimator (RA-LSC)", *not* a separate method. This
pre-empts "did you propose one method or two?" — the theory is one method; RA is
a practical estimator of the same score.

---

## 1. §"Implementable algorithm" — add a "Practical estimator" subsection

The exact score costs `q_theta·A·q_rho` potential evaluations per particle per
step (measured: 256 on E1, 1024–1536 on E3/E4). Add an Algorithm box + three
short Remarks (proofs to supplement, ≤ half a page each).

**Setup.** With the shell law `ν(dr)=λ Σ_a w̄_a Shell_a(dr)`, take the natural
selection distribution `ρ = ν/λ` (what `ShellJumpLaw.sample` already draws), so
the Radon–Nikodym weight is constant, `w(R)=λ`. Per refresh step draw one
`R_n ~ ρ` and use, for the whole step, the atomic generator and matched score
$$A_{\varepsilon,R}f = [-\nabla V + S_R]\cdot\nabla f + \varepsilon\Delta f
  + \lambda[f(\cdot+R)-f],\qquad
  S_R(x) = -\lambda R\!\int_0^1\! e^{\beta[V(x)-V(x-\theta R)]}\,d\theta .$$

**Algorithm box (Euler–Poisson):**
$$\tilde S_n = -\lambda R_n\exp\!\big(\mathrm{LSE}_p[\log w_p + \beta(V(X_n)-V(X_n-\theta_p R_n))]\big),$$
$$Y_n = X_n + h[-\nabla V(X_n)+\tilde S_n]_{\rm tamed} + \sqrt{2\varepsilon h}\,\xi_n,\quad
  M_n\sim\mathrm{Poisson}(\lambda h),\quad X_{n+1}=Y_n+M_n R_n .$$
Three conditions (state them): (i) `R_n` drawn before the step; (ii) the SAME
`R_n` drives score and jump; (iii) `\tilde S_n` acts even when `M_n=0`.

**Remark 1 (unbiased).** `\hat\nu_R=w(R)\delta_R`, `E_ρ[\hat\nu_R]=ν`, hence
`E_ρ[A_{ε,R}]=A_{ε,ν}` (Fubini).

**Remark 2 (atomwise μ-invariance).** For EVERY fixed `R` and `φ∈C_c^∞`, the
chord fundamental-theorem-of-calculus identity gives
$$\int S_R\cdot\nabla\phi\,d\mu
 = -\lambda\!\int p(y)\!\int_0^1\! R\cdot\nabla\phi(y+\theta R)\,d\theta\,dy
 = -\int J_R\phi\,d\mu,$$
so `∫A_{ε,R}φ dμ = 0`. (This is the key differentiator from SGLD-type
randomization: what is randomized is the *paired* jump + score-correction, each
fixed-R subsystem individually exact — not merely correct in expectation.)

**Remark 3 (ideal kernel, any h).** By Remark 2, the ideal random-scan kernel
`P_h = E_R[P_h^R]` satisfies `μP_h = μ` at ANY refresh length `h` (an exact
equality, not a small-h limit).

**Remark 4 (multi-atom estimator; variance control — needed for the benchmarks).**
The single-`R_n` estimator is unbiased but its per-step variance grows with the
atom count `A` and with `β`: the realized shift lands in one atom's shell, so the
other atoms' score contributions are unseen that step, and — because the score
enters the *nonlinear* tamed drift — this variance re-appears as a discretization
bias. Empirically it is negligible at small `A` / moderate `β` (E1 `A=2`, E2
annulus, `β=8`: the single-atom estimator matches the exact score, W₂ 2.5 vs 3.0
on E2) but breaks at `β=24` (E3: over-populates the far well, `occ(A)=0.44` vs
`π=0.31`) and `A=8` (E4). The fix is the **multi-atom estimator**
`\tilde S_n = Σ_a w̄_a \tilde S_{a,n}` — one shell draw `ρ_a` per atom, scored
against the full atom set — costing `A·q_theta` evaluations (still ≪ exact's
`A·q_theta·q_rho`, e.g. 128 vs 1024 on E4). It is the estimator deployed on E3/E4
and inherits Remarks 1–3: each atomwise sub-generator is individually μ-invariant,
so any fixed convex combination is too. Positioning for the paper: single-atom RA
is the minimal estimator (valid and cheapest where `A`, `β` are small); multi-atom
RA is the robust default at high `β` / many atoms; exact LSC-CP is the proven
reference. All three estimate the same Lévy score.

**Do NOT claim** (one sentence, for rigor): finite-h spectral-gap transfer, the
high-refresh-rate averaging limit, or exact target-preservation of the
Euler–Poisson discretization — the discretization bias is O(h) weak error,
controlled by the dt-refinement rule, same order as every other scheme here.

---

## 2. §"Numerical validation" intro — benchmark-design principle

Add a short paragraph decoupling the two exponentials:
$$\underbrace{\text{mass ratio}}_{\text{multimodality}}\sim e^{-\beta\Delta V_{\rm depth}},
  \qquad
  \underbrace{\text{escape time}}_{\text{metastability}}\sim e^{+\beta b_{\rm barrier}} .$$
A good nonlocal-sampling benchmark needs BOTH `β ΔV_depth = O(1)` (so the missing
mass is measurable) AND `β b_barrier ≫ 1` (so local methods fail) — i.e. the
barrier must exceed the depth gap. State the corollary explicitly: multimodality
determines whether the task is non-trivial, metastability whether local methods
fail; a target can have either without the other.

---

## 3. §"Transformed Müller–Brown landscape" — replace with depth-retuned 3-well (β=24)

The current transformed-MB example should be the **depth-retuned 3-well** target.
Add the impossibility argument + the retuning:

- **Impossibility (standard MB).** Depth gap `V_B-V_A=0.659` exceeds every
  barrier but A's own exit (`0.401`): comparable masses need `β≲4.6` ⇒ largest
  `βb≲1.8` (no metastability); metastability needs `β≳25` ⇒ mass ratio `e^{-16}`
  (effectively unimodal). Standard MB is multimodal OR metastable, never both.
- **Fix.** Keep the standard MB functional form; retune depths to
  `D=(-1.6607,-1,-1.0218,0.15)` so the three wells are equal-depth
  (`V=-0.7957`); saddles `S_AB=-0.3323`, `S_BC=-0.6310`. Now β is a free dial.
- **At β=24:** `βb(A↔B)=11.1` (slow), `βb(B↔C)=4.0` (moderate), masses
  `(0.32,0.42,0.26)` — a genuine two-timescale trimodal target. Embed in the
  same 10D map; init in well C; ULA reaches B but never A in T=200 (Kramers
  τ(A↔B)≈4.2×10⁵), while LSC-CP populates A via relay jumps.
- **Relay jump law:** 4 atoms `{±r_BA, ±r_BC}` through the middle hub B (no
  direct A–C atom). State that the atom set is fixed and every atom fires from
  every state (never gated on the current basin — else ρ becomes state-dependent
  and Remark 2's argument fails).
- Keep the 4-well plateau-island variant as an appendix stress test.

Figure: `figures/mb3well_10d/mb3well_10d_target.{pdf,png}` (the recognizable
3-well MB, trimodal at β=24).

---

## 4. §"Double well" — raw-CP predicted-bias forensic (positive result)

Raw CP is biased by design; the new result is that it converges to the
*theory-predicted* biased law, not to something else. Add: solve the linear
stationary integro-ODE
$$0=\tfrac1\beta\rho''+\partial_x[V'\rho]+\lambda\!\sum_{a,q} w_{a,q}[\rho(x-r_{a,q})-\rho]$$
(1D grid, Chang–Cooper conservative flux; `λ→0` recovers `e^{-βV}` to 5e-6),
overlay CDFs. Result (E1, β=8, λ=1): `W1(empirical,predicted)=0.047` within the
finite-sample floor, `W1(empirical,Gibbs)=0.089` — raw CP tracks the predicted
biased law; the LSC-CP score removes exactly this bias.
Figure: `figures/double_well/rawcp_stationary_forensic.{pdf,png}`.

---

## 5. §"Implementable algorithm" / results — NFE & wall-clock narrative

The RA estimator drops the per-step score cost from `q_theta·A·q_rho` to
`q_theta` potential evaluations. Measured per particle per step: exact LSC-CP =
257 (1 grad + 256), RA-LSC = 18 (1 grad + 17) on E1 (**~14×**); ~**96×** on
E3/E4 (`A=12/8`). Report every distributional metric on three axes: physical
time `n·Δt`, **NFE** (one V or ∇V evaluation = 1, counted inside the potential,
metric/reference evaluations excluded), and wall-clock. On the NFE axis RA-LSC
reaches exact-LSC's error at a fraction of the cost — the headline that the
`n·Δt` and wall-clock axes alone did not show. Validate RA against exact on
E1/E2 (agreement within the seed band; consistency figure in the appendix).

---

## 6. §"Coupled φ⁴" — 8-edge-atom jump design

The complete graph over the 4 phases includes the diagonals `--↔++`, `-+↔+-`,
whose coherent chords pass through the field-zero hilltop. Use the **8 edge
atoms** of the phase square; relay diagonal transitions through a mixed phase in
two hops. (Optional per-site jitter is free under the RA estimator — no
closed-form quadrature over the jump law is needed.)

**Taming cap (report as a discretization detail).** The coherent 24-D shifts make
the Lévy score astronomically large, so under `tame(b,h,c)=b/(1+h‖b‖/c)` the score
step saturates at length `~c`. Set `c = 2h_shell` (one shell width, matching E3);
a larger cap (`c = max‖r_a‖`) lets the saturated score overshoot the deepest
phase (`occ(--)=0.50` vs `π=0.325`) and lose to raw-CP. raw-CP is cap-insensitive
on every reported metric, so the tight cap is a fair, method-agnostic choice —
exact LSC-CP and the multi-atom estimator agree at it (the overshoot is
deterministic taming saturation, not estimator variance). At `c=2h`, LSC-CP-MA
beats raw-CP on all metrics (W₂ 0.11 vs 0.29, per-basin mass error 0.08 vs 0.25).

---

## 7. Metrics reported (results tables/figures)

Beyond sliced-W₂ / TV / MMD / EMC / EJS, report the chemistry-native and MCMC
diagnostics now in the pipeline: free-energy profile error `e_F` (units `k_BT`,
π-floor masked), per-basin relative mass error, ⟨V⟩ / Var(V) (heat-capacity
analog) and energy-histogram overlap, KSD (IMQ; note its mode-imbalance blind
spot — secondary), and cross-seed rank-normalized split-R̂ on the slow basin
occupancy. All start at the shared `n=0` point (identical across methods).
