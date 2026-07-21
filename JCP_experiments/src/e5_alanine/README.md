# E5 — alanine dipeptide (Ac-Ala-NHMe, vacuum, 22 atoms)

A real-force-field benchmark for LSC-CP. The sampler runs in **whitened BAT
internal coordinates**, where the backbone torsions φ and ψ are literally two
coordinates, so the nonlocal jumps are pure-torsion rotations between
Ramachandran basins and the Lévy-score chord is *Jacobian-free*.

## Environment (pinned)

Conda env **`jcp-e5`** (cloned from `jcp-exp`, so the E1–E4 suite reproduces):

| package | version |
|---|---|
| python | 3.12.13 |
| torch | 2.13.0+cu130 |
| numpy | 2.4.6 |
| scipy | 1.18.0 |
| openmm | 8.5.2 |
| openmmtools | 0.26.0 |
| mdtraj | 1.11.1 |

```bash
conda create --clone jcp-exp -n jcp-e5 -y
conda activate jcp-e5
conda install -c conda-forge openmm openmmtools mdtraj -y
```

OpenMM is used for **setup and validation only** (P0 parameter extraction, P1
cross-validation, P4 reference). It is never called in the sampler/score hot
loop; production energy is the torch code in `cartesian.py` / `potential.py`.
`system.py` forces `JAX_PLATFORMS=cpu` before importing openmmtools, because
its pymbar dependency pulls in JAX, which would otherwise grab the
torch-visible GPU.

## Units

OpenMM convention throughout: energy **kJ/mol**, length **nm**, charge **e**,
angles **rad**. `ONE_4PI_EPS0 = 138.93545764438` kJ·nm/(mol·e²), probed directly
from OpenMM rather than hard-coded.

Temperature **T = 300 K**, derived (not hard-coded) from the molar gas constant:
`β = 1/(k_B T) = 0.4009079 mol/kJ`, `k_B T = 2.4943388 kJ/mol`, `ε = 1/β`.
`config.BETA = 8` is untouched — E5 threads its own β through `RunConfig`,
exactly as E3 threads β = 24.

## System

`openmmtools.testsystems.AlanineDipeptideVacuum(constraints=None)` — **fully
flexible**, so every bond is harmonic and the potential is the smooth 66-D
function the torch force field and the BAT model represent.

| quantity | value |
|---|---|
| atoms | 22 (ACE 0–5, ALA 6–15, NME 16–21) |
| Cartesian dimension D | 66 |
| internal dimension (D−6) | 60 = 21 bonds + 20 angles + 19 torsions |
| harmonic bonds / angles | 21 / 36 |
| periodic torsions | 52 |
| nonbonded exceptions | 98 (excluded 1-2/1-3 and scaled 1-4) |
| constraints | 0 |

### φ / ψ definitions

φ = τ(ACE:C, ALA:N, ALA:CA, ALA:C) = atoms **(4, 6, 8, 14)**
ψ = τ(ALA:N, ALA:CA, ALA:C, NME:N) = atoms **(6, 8, 14, 16)**

Both identified by residue/atom name and verified against
`mdtraj.compute_phi/compute_psi` (index sets identical, values agreeing to
< 1e-6 rad). The signed dihedral uses the atan2 form of §1.8; OpenMM's
`CustomTorsionForce` θ was checked to use the *same* sign convention.

## Coordinates (`bat.py`)

Z-matrix rooted at **(ACE:C=4, ALA:N=6, ALA:CA=8)**, which fixes the 6 external
DOF. The first two non-root atoms are forced to be ALA:C=14 and NME:N=16, so

* `q[5]` = τ₁₄ = **φ**, `q[8]` = τ₁₆ = **ψ**.

### Correlated (leader/offset) torsions — important

Atoms sharing a bond/angle parent pair (C, B) rotate about the **same axis**. If
each kept an absolute torsion, moving one while holding its siblings fixed would
*distort the local geometry* rather than rotate the fragment: with absolute
torsions the measured curvature of the φ coordinate was **822 kJ/mol/rad²**
(and 341 for ψ), because changing τ₁₄ moved ALA:C and its subtree while leaving
HA and CB behind, breaking the tetrahedral CA centre.

So the first atom placed about each axis is the **leader** and carries the proper
torsion; its siblings carry the (near-constant, stiff) **offset** from the leader.
Moving φ or ψ then rotates the whole fragment — the physical Ramachandran
motion. There are 8 leader and 11 offset torsions. The reparametrisation is
unit-triangular in the torsion block, so **log|J| is unchanged**.

Without this, a pure-φ jump atom would be exactly the §1.4 pathology (a chord
driven hundreds of k_BT above the basins).

### Jacobian

    log|det ∂x_free/∂q| = ln r₂₃ + Σ_{non-root} [ 2 ln b + ln sin a ]

The `ln r₂₃` term is the frame boundary term from root atom 3 (a 2-D polar
element: one power of the bond, no sine). Verified against the numerical
log|det| of the 60×60 internal Jacobian block to ~1e-14. **No term depends on a
torsion**, which is the premise of the Jacobian-free chord (§1.6).

## Whitening D (`potential.py`)

`V(q̃) = U_eff(D q̃)`, `U_eff(q) = U(x(q)) − (1/β) J(q)`, with a **fixed diagonal**
D. Provenance: the diagonal of the U_eff Hessian at the **basin minimum**.

The supplied conformer is *not* a stationary point (|∇U_eff| = 105, almost all of
it in φ), which inflates the raw curvature ~7×, so we first descend to the
nearest minimum — it lands in **C5/β at (φ, ψ) = (−147.5°, 159.9°)** with
|∇U_eff| ≈ 3e-3 — and take the Hessian there.

* **D = 1 for φ and ψ only** — they must stay affine, unit-scale coordinates, so
  the jump atoms read directly as rotations.
* every other slot: `D_ii = sqrt(1/(β H_ii))`, capped at 1.

| slot class | D range | whitened curvature |
|---|---|---|
| bonds | 0.0023 – 0.0030 | ε = 2.4943 |
| angles | 0.040 – 0.071 | ε = 2.4943 |
| other torsions | 0.043 – 1.0 | ≤ ε |
| φ, ψ | 1.0 (unwhitened) | 79.9 / 33.0 |

Whitened Hessian spectrum at the minimum: 0.56 – 87 (condition number ≈ 156),
versus a bond curvature of ~4.8e5 in raw internal coordinates. Cached in
`cache/e5_alanine/whitening.npz` with a layout signature, so a change of
coordinate convention invalidates it.

**Time step.** Max whitened curvature 79.9 ⇒ `dt·H = 0.08` at `dt = 1e-3`.
Verified against equipartition (⟨U_eff⟩ ≈ U_min + (d/2)k_BT = 167.4,
std = 13.7): at `dt = 1e-3`, ⟨U_eff⟩ = 164.4, std = 13.5. Production `dt = 1e-3`.

## Box (`box.py`) and the ±π seam

`TorusBox` **wraps** the torsion slots to (−π, π] and **clamps** bonds/angles.

* Torsions have no boundary, so `contains` always reports them inside and
  wrapping is never miscounted as a boundary clip. Wrapping is exact: U_eff is
  2π-periodic in every torsion (verified energy-invariant to 8.5e-14).
* Wrapping is done in **physical** units. In whitened units a torsion's period is
  2π/D, and only φ/ψ have D = 1 — wrapping the other torsion slots as if they had
  period 2π would destroy the state.
* Bonds/angles carry a *physical* boundary, not merely an overflow guard: the
  Jacobian has `ln b` and `ln sin a`, so the box keeps b ∈ [0.3, 3]×b_ref and
  a ∈ [0.10, π−0.10] rad. These sit many thermal σ from equilibrium and never
  bind in normal sampling.

**Periodicity discipline (§2) — measured, and it forced a decision.** φ is
reported on the standard (−π, π]: only **0.5%** of the reference mass lies within
0.15 rad of its seam. ψ is **not**: the C5/β region genuinely wraps around
ψ = ±180°, putting **7.4%** of the mass at that seam (17% within 0.3 rad), which
would make Euclidean W2/MMD/FES treat ψ = +179° and ψ = −179° as maximally
distant. Per §2 this was reported rather than patched silently (see
`GATE_FAILURE_P4.md`), and resolved by moving ψ's **branch cut to −20°** — the
empty gap between α_R and C7eq; the measured density minimum sits at −15°, and −20° achieves the same effect — i.e. reporting ψ on
**(−20°, 340°]**. Seam mass drops **0.0780 → 0.0049**, gated below 0.05.

This is a choice of *fundamental domain* inside E5's own `to_cv`, so `metrics.py`
is untouched, Euclidean distance stays valid, and the C5/β basin is contiguous
rather than split. Window centres live in `bat.py`
(`PHI_WINDOW_CENTER`, `PSI_WINDOW_CENTER`; note a window centred at *c* has its
cut at *c*+π). Basin assignment independently uses a **torus** Voronoi metric,
and `TorusBox` wraps in physical units, so neither depends on the window.

**Reading the plots.** Because ψ is reported on (−20°, 340°], an E5 Ramachandran
plot is *not* the conventional (−180°, 180°] one: C5/β still sits near ψ = +160°,
but α_R / C7ax appear near ψ ≈ +290° instead of −70°. Ensemble metrics are
unaffected (W2/MMD compare static ensembles; basin labels use the torus metric),
but a ψ *time series* shows an apparent jump at the cut whenever a trajectory
passes continuously between C7eq and α_R — wrap for display, never for metrics.

**Deviation from §1.9.** The task assumed the lower-barrier φ path runs through
φ ≈ 0. For this force field it does not: the φ = 0 dividing line sits at
**32.8 kJ/mol**, while the C7ax island's own escape barrier is only
**11.7 kJ/mol (4.67 kT)**, reached around φ ≈ ±180. The jump atoms use
torus-minimal displacements and so take that route automatically;
`island_barrier_kJ()` (a basin escape barrier), not a φ-cut minimum, sets
`kramers_tau` and `pt_beta_min`. A φ-cut minimum would be wrong here anyway: the
φ = ±180 line passes through a basin, so the value on it is a basin depth, not a
saddle.

## Reference (`build_reference.py`, `reference.py`)

Well-tempered metadynamics on the same flexible Cartesian system, two
`CustomTorsionForce` CVs (φ, ψ), Langevin 300 K. Bias grows during
equilibration; then, with the bias **frozen**, production frames are reweighted
to the unbiased Boltzmann measure by

    w(s) = exp(+β V_bias(s)) = exp(−β ((γ−1)/γ) F(s)).

OpenMM returns `F = −(γ/(γ−1))·V_bias` exactly, so this weight *is* the applied
bias: the reweighting is **unbiased for any static bias**, and metadynamics
convergence controls only the variance.

Two silent traps, both guarded:

1. **Grid axis order.** OpenMM allocates the bias grid with `reversed(variables)`,
   so `getFreeEnergy()` is indexed `[ψ, φ]`. With equal grid widths the transpose
   is invisible; left uncorrected the FES comes out mirrored through the origin
   (global minimum at (+157°, −142°) instead of C5/β). `_free_energy_phi_psi`
   transposes, and `orientation_check` correlates the frame histogram against
   e^{−βF/γ} to catch a regression.
2. **Units.** 1 ns = 1e6 fs; an earlier `steps_per_ns = 1000/dt_fs` silently ran
   1000× short and rounded the frame stride to zero.

Performance: this 22-atom system is overhead-bound, not force-bound. Measured —
OpenCL ≈ 300 steps/s, CPU 16 threads ≈ 930, **CPU single-threaded ≈ 2805**
(20 ns/hour); depositing every 100 steps costs ~30× more than every 1000 steps
(2 ps, a standard interval). So the default is the CPU platform with
`OPENMM_CPU_THREADS=1`, `deposit_every=1000`, and seeds run as **parallel
single-threaded processes**. (OpenMM's CUDA platform fails on this box with
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.)

```bash
OPENMM_CPU_THREADS=1 python -m src.e5_alanine.build_reference --mode seed --seed 0 &
OPENMM_CPU_THREADS=1 python -m src.e5_alanine.build_reference --mode seed --seed 1 &
wait
python -m src.e5_alanine.build_reference --mode combine --seeds 0 1
```

Raw Cartesian frames are cached alongside the converted coordinates, so a later
change of internal-coordinate convention needs only a cheap re-conversion.

Basins are a **torus Voronoi** partition around the FES minima (no grid domain,
hence none of E4's out-of-domain clamping hazard), and `p_star` is the weighted
basin occupancy of the reweighted pool, cross-checked against the FES integral.

**Only metastable minima count as basins.** Raw minimum-finding returned five
candidates, but two had *negative* escape barriers — the Voronoi boundary lay
below their own minimum — so they were shoulders of the β/C5 region, not
metastable states. `merge_shallow_minima` drops any basin whose escape barrier is
under 1 kT. This matters twice over: it de-fragments `p_star`, and it makes
"island occupancy" meaningful, since local dynamics reaches a shoulder freely
(counting one as an island made ULA appear to cross the barrier).

| basin | (φ, ψ) | p* | escape barrier |
|---|---|---|---|
| C5/β | (−145.8°, 160.2°) | 0.5766 | 2.54 kT (global minimum) |
| C7eq | (−73.8°, 77.4°) | 0.4152 | 2.35 kT |
| **C7ax** | (63.0°, −66.6°) | **0.0082** | **4.67 kT (11.7 kJ/mol)** ← the slow event |

The 1 kT criterion is **not tuned**: sweeping it leaves the partition unchanged
over a 40× range, and the structure only changes where a *real* barrier is met.

| merge threshold | resulting partition |
|---|---|
| 0.05 – 2.0 kT | **K = 3** (C5/β, C7eq, C7ax) — unchanged |
| ≥ 2.4 kT | K = 2 (C7eq absorbed into C5/β, whose true separation is 2.53 kT) |

Both spurious shoulders are already gone at the smallest threshold tried, i.e.
they would be dropped by *any* positive criterion, and the deployed 1 kT sits in
the interior of the plateau rather than on an edge. Reproduce by sweeping
`merge_shallow_minima(F, axis, minima, beta, min_barrier_kT=x)`.

Measured convergence (2 seeds × 8 ns equilibration + 10 ns production):
p_star reproduces across seeds to **0.0008** (basin ΔF to 0.001 kT); p_star vs
the FES integral 0.0007; aligned FES drift over the last third 0.34/0.41 kJ/mol
mass-weighted; basin ΔF range 0.073/0.148 kT for basins ≥1% mass, 0.225/0.348 kT
for the 0.8% island (documented as convergence-limited); importance-weight
ESS 3210 (16% of a 20000-frame pool).

## Jump law (`jump_design.py`)

Candidates are the torus-minimal (Δφ, Δψ) displacements between every ordered
pair of FES basins, so both homotopy directions (the ± pairs) are present by
construction. Each is **screened**: an atom whose chord crosses a forbidden
Ramachandran region drives the score exponent β[U(q) − U(q−θr)] past M_MAX, and
is dropped (with its geography and reason logged), following E3's dropped direct
A–C atom and E4's dropped diagonals. Pairs left unconnected are reached by relay
through an intermediate basin.

`h = 0.1·min_a‖r_a‖`, `cp_drift_cap = 2h` (the E3/E4 rule), uniform weights.
With three basins there are 6 ordered pairs; all 6 are retained
(`h = 0.1915`, `cap = 0.3830`). After the leader/offset torsion fix the chords
are benign — max score exponent ≈ 4 kT against a cap of 600 — so nothing needed
dropping. The drop rule and its log are kept because they make that a
*measurement* rather than an assumption.

Atoms are nonzero **only** in the φ/ψ slots, which is what makes the chord
Jacobian-free; the shell jitter stays in the same plane.

## Certificate (`certificate.py`)

60-D rules out a quadrature grid, so — as for E4 (24-D) — R(φ) is the **shifted
form**, which reduces to the pointwise θ-quadrature defect with no O(1)
cancellation left to Monte Carlo. μ expectations are taken against the
reweighted metadynamics pool rather than a Laplace proposal. The certificate is
**atomwise** (the stricter statement: the mixture residual is bounded by the max
over atoms).

Test functions are **periodic** on the torus, `f = sin(mφ + nψ + c)` — a tanh
ridge would be inadmissible — and **jump-aligned**, since random ridges in high
dimension have a·r̂ ~ 1/√d and are blind to the jump direction.

The **direct** form is reported but not gated: its integrand p·S is O(1) exactly
where p is exponentially small, so it is not μ-estimable in 60-D. That is
precisely why E4 and E5 certify with the shifted form; the tight-domain reading
is reported alongside the generous (full-torus) one.

## Gate summary

| phase | gate | measured |
|---|---|---|
| P0 | φ/ψ vs mdtraj; force-term counts | indices identical, < 1e-6 rad |
| P1 | E vs OpenMM rel 1e-6, F rel 1e-5, grad vs FD 1e-6 | ~1e-15 / ~1e-15 / 2e-9 |
| P2 | round trip 1e-10; log|J| vs logdet 1e-6; torsion-independence 1e-10 | 2e-15 / 1e-14 / **0.0** |
| P3 | grad vs FD 1e-6; Jacobian-free chord < 1e-10 | 2.4e-9 / **4.6e-13** |
| P4 | p_star across 2 seeds < 0.05; basin ΔF (≥1% mass) < 0.2 kT; seam mass < 0.05 | **0.0008** / **0.073, 0.148 kT** / **0.0049** |
| P5 | R(φ) generous < 1e-6; no M_MAX saturation | **2.5e-15**; max exponent ≈ 4 vs cap 600 |
| P6 | nonfinite_count == 0; LSC-CP crosses, locals trapped, raw CP biased | **0**; see table below |
| P7 | launcher dry-run; per-metric pdf+png | clean; 12 metrics × 2 formats |

P6 smoke (N = 2000, 600 steps; target island occupancy 0.0082):

| method | island occupancy | basin L1 |
|---|---|---|
| ULA / MALA / BAOAB / PT | **0.0000** (trapped) | 0.42 / 0.42 / 0.78 / 0.33 |
| FLA | 0.0290 | 0.568 |
| CP / CP-RA (raw) | 0.209 / 0.223 (**~25× over-populated**) | 0.401 / 0.430 |
| LSC-CP-RA / LSC-CP-MA | 0.055 / 0.002 | 0.301 / **0.261** |

Locals cannot cross; raw CP crosses but grossly over-populates the island; the
score correction restores it with a measurably smaller basin error. (Short
horizon, so these are transient occupancies, not equilibrated masses.)

## Modelling choice

The **flexible** BAT model of §1.7 is used; the §3 rigid-geometry (torsion-only)
fallback was **not** required — after the leader/offset torsion fix and the
basin-Hessian whitening, `dt = 1e-3` is comfortable.

Note this is a *sampling* construction: identity-metric Langevin on U_eff, not
the curved-space kinetics (no position-dependent mass metric or Fixman term).
Since the object of study is the invariant distribution, this is the correct and
simplest choice, and it is applied identically to every method.
