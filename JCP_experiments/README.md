# JCP experiments — LSC-CP compound-Poisson sampling benchmark

This standalone experiment tree supports academic discussion of
**Lévy-score-corrected compound-Poisson (LSC-CP)** equilibrium sampling. The
current examples are the quartic double well, MoG40, the depth-balanced
Müller–Brown-type system in 10 dimensions, the coupled two-component
\(\phi^4\) chain, and **alanine dipeptide under a real force field** (E5). The
exact continuous-time generator preserves the Boltzmann target. The
random-atomic and multi-atom implementations pair each realized atom bank with
the Poisson jumps generated from that same bank.

Raw CP uses the same nonlocal geometry without the score correction. It is a
mechanism/invariant-law diagnostic, not a fair target-preserving efficiency
baseline.

## Quick start

```bash
# jcp-exp runs E1-E4; E5 additionally needs openmm/openmmtools/mdtraj, which are
# installed in jcp-e5 (a clone of jcp-exp). Use jcp-e5 to run the whole suite.
conda activate jcp-e5
cd /home/zheyuanlai/levy-sampling/JCP_experiments

# Required preflight. The GPU tests use one selected device; CPU regression
# tests include metrics, run gates, reference handling, and stationary traces.
JCP_GPU=4 python -m pytest tests tests_cpu -q

cd notebooks
python build_notebooks.py

# Bounded launcher: one GPU per child, at most two children concurrently.
cd ..

# Safe gate first: regenerate, test, and run all selected smokes, then stop.
./run_production.sh --gpus 0,1 --max-concurrent 2 --smoke-only

# Launch the long full-notebook campaign only after reviewing the smoke
# artifacts and status. Each notebook writes resolved model/cache provenance,
# then its measured certificate JSON before asserting the certificate gate.
./run_production.sh --gpus 0,1 --max-concurrent 2
```

The generated notebooks are `01_double_well.ipynb`, `02_mog40.ipynb`,
`03_mb3well_10d.ipynb`, `04_coupled_phi4.ipynb`, and
`05_alanine_dipeptide.ipynb`. Do not run production cells merely to regenerate
or inspect notebooks.

E5 additionally requires its metadynamics reference cache to exist before any
run; see the E5 section below for the regeneration command.

## Immutable run layout

The launcher and notebooks refuse to overwrite an existing run. The default
layout is:

```text
results/jcp_sampling/<run-id>/
  launch_plan.json
  status.json
  smoke/
    <experiment>/
      stdout.log
      stderr.log
      smoke_plan.json
      status.json
      artifacts/
        original_config.yaml
        resolved_config.json
        smoke_metrics.csv
        smoke_manifest.json
  <experiment>/
    stdout.log
    stderr.log
    run_plan.json
    status.json
    notebook_status.json
    executed_notebook.ipynb
    artifacts/
      original_config.yaml
      resolved_preflight_config.json
      certificate_result.json
      resolved_config.json
      manifest.json
      metrics_timeseries.csv
      summary.csv
      figures/
      stationarity/
        <method>_summary.csv
        <method>_traces.npz
        all_methods_summary.csv
```

Set `JCP_RUN_ID` to deliberately share one run ID across notebooks. Set
`JCP_RESULTS_ROOT` only when using a different immutable output root. Cached
basin maps remain under `JCP_experiments/cache/`; they are reusable numerical
cache data, not run results. The source YAML is written once before model
construction and is never appended. `resolved_preflight_config.json` freezes
the actual model, jump bank, numerical box, builder parameters, cache paths and
SHA256 values before certification. `certificate_result.json` is written before
the pass assertion; `resolved_config.json` is reserved for post-refinement
choices such as timestep, quadrature, PT ladder, and trace protocol. A failed
notebook status records which of these stages was preserved.

## Method matrix and the two Lévy-score estimators

Every experiment runs **eight** methods: four baselines (`ULA`, `MALA`,
`BAOAB`, `PT`), two non-target-preserving diagnostics (`FLA`, `CP`), and
**two LSC-CP arms**. The arms are declared in `launch_production.py`:

```text
DUAL_RA = ULA,MALA,FLA,BAOAB,PT,CP,LSC-CP,LSC-CP-RA   # E1 double_well, E2 mog40
DUAL_MA = ULA,MALA,FLA,BAOAB,PT,CP,LSC-CP,LSC-CP-MA   # E3, E4, E5
```

Both arms estimate the same exact Lévy score

```text
S(x) = -lam * int nu(dr) r int_0^1 exp(beta[V(x) - V(x - theta r)]) dtheta
```

and differ only in how the `nu` and `theta` expectations are taken:

| arm | estimator | randomness | cost per particle per step |
|---|---|---|---|
| `LSC-CP` | deterministic Gauss–Legendre quadrature (`ShellScore`) | none | `q_theta * A * q_rho` chord energies |
| `LSC-CP-RA` | one realised displacement (`RandomAtomicShellScore`) | one draw, shared with the jump | `q_theta` |
| `LSC-CP-MA` | one realised displacement per atom family, with atomwise Poisson counts (`MultiAtomShellScore`) | a stratified bank, shared with the jump | `A * q_theta` |

MA is the A-atom generalisation of RA — the atom index is stratified rather
than multinomially sampled, and the single Poisson count becomes `A` atomwise
counts — so figures label it `LSC-CP-RA (A)` (`(4)` on E3, `(8)` on E4). RA
needs only `law.sample`, so it is the only realised-displacement arm available
for E2's continuous annulus law; MA additionally requires a finite shell bank.

**Numerical integration only — no closed forms.** Every deployed arm obtains the
score from chord energy differences `V(x - theta r) - V(x)` alone; the density's
normalising constant cancels identically, so nothing about the target beyond a
callable `V` is required. E2 previously shipped an analytic `MoG40Score` that
integrated `theta` and `rho` in closed form using the mixture means. That does
not generalise to a black-box target, so it has been demoted to a comparator
(retained for `tests/` and `scripts/certificate_gate.py`) and E2 now integrates
numerically like every other experiment. `tests_cpu` asserts that no deployed
LSC arm has a zero-NFE score path.

**E2 jump band.** `AnnulusJumpLaw(10.0, 14.0)`: `rho ~ Unif[10, 14]`,
`phi ~ Unif[0, 2pi)`. The band is narrow so the radial rule needs few nodes,
while `b = 14` keeps the mode graph connected — measured one connected
component at both the 1e-3 and 3e-3 mode-hit thresholds (diameter 12 / 14),
matching the former `[4, 15]` band's 11 / 13. Connectivity is all that is
required: as on E3, modes need only be reachable through intermediates, not
pairwise. The annulus rule is limited by its **angular** order — `q_rho`
saturates by 3–4 while `m_phi` 32 → 48 improves the median relative score error
67x — so `build_e2` sets its own `(q_theta, q_rho, m_phi)` defaults instead of
inheriting the global `M_PHI`.

**Blocking.** `ShellScore` materialises an `(N, q_theta*J, d)` chord tensor,
which at production `N` exceeds device memory once `q_theta*J` is in the
thousands. It therefore blocks over particles; `_log_parts` is row-wise, so the
blocked result is bit-identical to the unblocked one.

## Free-energy (FES) metric

`FES_RMSE_kBT` is an additive-constant-aligned RMSE over histogram cells, in
`kBT`. Three properties matter for reading it:

* **uniform cell weights.** Reference weighting concentrated ~85% of the score
  on the ten densest cells, so the metric measured basin-bottom shape only and
  was blind to barrier and tail placement.
* **an off-grid cell.** `binned_probabilities_with_outside` appends one cell
  holding the mass that left the histogram domain, so the comparison conserves
  mass. Without it, a sampler that leaks has its worst-placed mass silently
  dropped and renormalised away — which inverted the ranking on E1, where raw
  CP leaked 2.86% against LSC-CP's 0.47% and scored *better*. The off-grid cell
  is exempt from the `pi_min` cut, since the reference legitimately puts almost
  no mass there.
* **`pi_min = 1/n`**, one expected observation, not five. At `5/n` the barrier
  cells were deleted outright — exactly where corrected and uncorrected
  dynamics differ most.

`FES_outside_mass` remains reported separately as the raw, unsmoothed leaked
fraction.

## Coupled-phi4 numerical domain

The thermodynamic target is unbounded; the sampling box is only a numerical
overflow guard. For E4 its half-width is derived, not tuned to make a smoke
pass. A simultaneous Laplace-mixture coordinate envelope with union-bound tail
budget `1e-8` is padded by one maximum componentwise shell-jump reach; the
hottest PT envelope is also covered, and the maximum is rounded outward. The
current values are `B_beta=1.91041`, `B_beta_min=3.56877`, and
`R_infinity=2.20000`, giving `ceil(max(B_beta+R_infinity, B_beta_min)) = 5`.
This is a high-probability one-jump envelope, not compact support: Gaussian
noise, optional jitter, and multiple jumps remain unbounded.

Generic state clipping and CP-specific jump-boundary clipping are reported
separately. The latter is normalized by the exact number of applied jumps and
is gated at zero by default. E4 also reports `basin_map_outside_mass` before
its two-dimensional basin lookup clamps to the map edge; only target-preserving
methods gate on excessive outside mass, so raw-CP/FLA negative results are not
filtered out. Before using E4 for a manuscript claim, run a short box
sensitivity comparison at half-width 5 versus 6 or 7 and require phase/FES/
energy results to agree within uncertainty.

## E5 — alanine dipeptide (real force field)

E5 is the first example whose energy is a real molecular force field rather than
an analytic surface: the fully flexible `AlanineDipeptideVacuum`
(`constraints=None`, 22 atoms, 66 Cartesian DOF, 21 bonds / 36 angles / 52
periodic torsions / 98 nonbonded exceptions) at **T = 300 K**. β is threaded
locally (`config.BETA = 8` is untouched for E1/E2/E4), derived from the gas
constant: β = 0.40091 mol/kJ, k_BT = 2.4943 kJ/mol.

**Coordinates.** The sampler runs in whitened BAT internal coordinates
(d = 60 = 21 bonds + 20 angles + 19 torsions), so φ and ψ are literally two
coordinates and the jumps are pure-torsion rotations between Ramachandran
basins. Two design points carry the experiment:

* *Correlated torsions.* Atoms sharing a rotation axis use a leader/offset
  convention, so moving φ rotates the whole fragment instead of distorting the
  Cα centre (which would make φ a stiff, unphysical direction — curvature 822
  versus 80 kJ/mol/rad²).
* *Jacobian-free chord.* The BAT Jacobian
  `ln r23 + Σ[2 ln b + ln sin a]` contains no torsion, so for a pure-torsion
  jump the Lévy-score integrand reduces to the physical energy difference.
  Verified machine-zero (4.6e-13).

**Jump design.** Torus-minimal (Δφ, Δψ) atoms between every ordered pair of FES
basins (so ± pairs by construction), each screened against a forbidden-chord
rule and dropped with its geography logged — the analogue of E3's dropped direct
A–C atom and E4's dropped diagonals. `h = 0.1 min‖r_a‖`, drift cap `2h`.

**Basins.** Three metastable states, after dropping raw FES minima whose escape
barrier is under 1 kT (two of five candidates had *negative* escape barriers —
shoulders of the β/C5 region, not basins): C5/β 0.577, C7eq 0.415, and the sparse
**C7ax island 0.0082** behind an 11.7 kJ/mol (4.67 kT) escape barrier, which is the slow
event. Note the low-barrier φ route runs through ±180, not through φ ≈ 0
(32.8 kJ/mol there).

**Periodicity.** φ is reported on (−π, π], but ψ genuinely straddles the ±π seam
(7.4% of the reference mass — the C5/β region wraps), so ψ is reported with its
branch cut moved to −20°, i.e. on (−20°, 340°]; seam mass 0.078 → 0.005. This is
a fundamental-domain choice inside E5, so `metrics.py` is untouched.

**Methods.** E5 is registered with the E3/E4 `DUAL_MA` set,
`ULA,MALA,FLA,BAOAB,PT,CP,LSC-CP,LSC-CP-MA`, for consistency with the other
experiments — but **the exact arm is unverified here and may be impractical**.
Deterministic quadrature costs `q_theta * A * q_rho` BAT reconstructions per
particle per step against a real force field, which is why the original E5
design deployed only the realised-displacement estimator. Benchmark the
`LSC-CP` arm before trusting an E5 production launch, and drop it back to a
single MA arm if the cost is prohibitive.

E5 has never been run to production: the registration and smoke exist, but
there are no E5 results, and its reference cache must be rebuilt first.
`launch_production.py` deselects `test_e5_*` from its preflight when
`alanine_dipeptide` is not among the selected experiments, so an out-of-band
reference rebuild cannot block an unrelated E1–E4 campaign.

**Reference.** Well-tempered metadynamics on (φ, ψ), reweighted under the frozen
converged bias by `w = exp(β V_bias)` — exact for any static bias, so
metadynamics convergence controls variance rather than bias. Basins are a torus
Voronoi partition around the FES minima, so unlike E4 there is no grid basin map
and no out-of-domain clamping hazard. The reference is a **convergence-limited
experimental input** and is documented as such.

**Certificate.** 60-D rules out a quadrature grid, so R(φ) is the shifted form
(as for E4 in 24-D), atomwise, with μ expectations taken against the reweighted
metadynamics pool. Test functions are periodic on the torus and jump-aligned.
The direct form is reported but not gated: its integrand p·S is O(1) exactly
where p is exponentially small.

Regenerate the reference (needed once before running E5):

```bash
OPENMM_CPU_THREADS=1 python -m src.e5_alanine.build_reference --mode seed --seed 0 &
OPENMM_CPU_THREADS=1 python -m src.e5_alanine.build_reference --mode seed --seed 1 &
wait
python -m src.e5_alanine.build_reference --mode combine --seeds 0 1
```

Full details, pinned versions, units, the ±π-seam discipline and the gate table
are in [`src/e5_alanine/README.md`](src/e5_alanine/README.md).

## Stationary efficiency protocol

Sparse ensemble-relaxation checkpoints are not used to estimate IAT, ESS, or
R-hat. Each notebook runs separate small-chain trajectories with uniformly
spaced post-step observations and records basin indicators, energy, collective
variables, and positions. Every observable row reports target, signed and
absolute bias, IAT, ESS, R-hat, MCSE, ESS/sec, ESS/gradient evaluation,
ESS/potential evaluation, and ESS/Lévy-score quadrature evaluation.

Only an explicitly certified exact draw for the box-restricted MALA kernel
could justify a zero-burn MALA trace. None of the current numerical-grid,
inverse-CDF, unbounded-mixture, or finite-SIR references makes that exact
claim, so every current method receives a charged settling run. ULA, BAOAB,
and the split LSC-CP integrators also have finite-step invariant laws that need
not equal the target. PT requires settling because its cold draw is replicated
over a non-equilibrium temperature ladder. The coupled \(\phi^4\) unweighted
reference cloud is finite SIR, hence approximate and not i.i.d.; this
provenance is stored. Direct weighted SNIS supplies its basin, energy, and
collective-variable targets.

By default, the settling period is one production horizon \(T\), followed by
approximately one \(T\)-long recorded trajectory. Thus approximate kernels run
about twice the stationary-trace simulation steps of exact-start MALA. The
small trace batch defaults to four seeds and eight chains per seed; adjust
`JCP_TRACE_SEEDS`, `JCP_TRACE_CHAINS`, `JCP_TRACE_DRAWS`,
`JCP_TRACE_SETTLING_BURN_FRACTION`, or `JCP_PT_TRACE_BURN_FRACTION` explicitly
when performing timing studies.

## Source layout

```text
src/
  experiments.py    experiment builders and sampler factories
  potentials.py     potentials and evaluation counters
  jumps.py           jump laws and compound-Poisson updates
  score.py           exact, random-atomic, and paired multi-atom scores
  samplers.py        ULA, MALA, FLA, BAOAB, PT, raw CP, and LSC-CP
  metrics.py         distributional, thermodynamic, and convergence metrics
  stationarity.py    uniform trace collection, IAT/ESS/R-hat, CSV/NPZ output
  certificate.py     weak generator-level stationarity certificates
  e5_alanine/        E5: alanine dipeptide (real force field)
    system.py          deterministic OpenMM build + phi/psi identification
    extract_params.py  P0 force-field extraction -> params.npz (committed)
    cartesian.py       batched float64 torch force field (validated vs OpenMM)
    bat.py             differentiable BAT transform + analytic log|Jacobian|
    potential.py       U_eff in whitened internal coordinates
    build_reference.py well-tempered metadynamics reference (setup only)
    reference.py       FES / basins / p_star / reweighted conformer pool
    box.py             TorusBox (wraps phi,psi; clamps stiff DOF)
    jump_design.py     torsion jump atoms + forbidden-chord drop rule
    certificate.py     atomwise shifted-form R(phi) on the torus
  references.py      direct references, finite SIR, and weighted SNIS
  runner.py          timing, refinement gates, manifests, and passage diagnostics
notebooks/            generated experiment notebooks
standalone_experiment_note/  separate LaTeX academic experiment discussion
tests/                GPU/scientific implementation tests
tests_cpu/            CPU-only regression tests
```

## Interpretation constraints

LSC-CP is evaluated for equilibrium sampling and observables, not physical
reaction kinetics. Generator-level invariance, finite-step target bias, and
stationary sampling efficiency are separate claims and are reported
separately. Negative, unstable, censored, or failed runs must be retained in
their immutable run directories.
