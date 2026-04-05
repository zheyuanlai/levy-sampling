# LSB-MC Repository Audit and Implementation Notes

## Repository Purpose

This repository implements **LSB-MC (Levy-Score-Based Monte Carlo)** for Boltzmann sampling, as described in the accompanying manuscript. LSB-MC augments overdamped Langevin diffusion with compound Poisson jumps and a Levy-score correction to accelerate exploration of metastable energy landscapes while preserving the target Boltzmann distribution.

---

## Manuscript-Stated Theory (paper/ directory)

### Global Convention (Section 2, Preliminaries)

**Target Distribution** (Eq. 36):
```
p_∞(x) = Z^{-1} exp(-2V(x)/σ²)
```

**Diffusion Baseline** (Eq. 7):
```
dX_t = -∇V(X_t) dt + σ dB_t
```

**LSB-MC Jump-Augmented SDE** (Eq. 32, Section 3):
```
dZ_t = (-∇V(Z_{t-}) + S_L^s(Z_{t-})) dt + σ dB_t + dL_t
```
where:
- `S_L^s(x)` is the stationary Levy-score correction (Eq. 27):
  ```
  S_L^s(x) = -∫₀¹ ∫ r exp{-2(V(x-θr)-V(x))/σ²} ν(dr) dθ
  ```
- `dL_t` is a compound Poisson jump process with Levy measure `ν`

### Example-Specific Conventions (Section 5, Numerical Experiments)

The manuscript **intentionally uses different parameterizations** for different examples:

1. **Ring Potential** (lines 42-46):
   - SDE: `dX = -∂_x V dt + √ε dB`
   - **Matches global convention** with `σ = √ε`

2. **Four-Well Potential** (lines 82-86):
   - SDE: `dX = -(1/2)∂_x V dt + ε dB`
   - **Rescaled**: drift has factor 1/2, noise is `ε` (not `√ε`)

3. **Muller-Brown Potential** (lines 134-138):
   - SDE: `dX = -(1/2)∂_x V dt + ε dB`
   - **Rescaled**: same as four-well

4. **Lennard-Jones** (lines 192-196):
   - SDE: `dX^(i) = -(1/2)∂_{x_i} V dt + ε dB`
   - **Rescaled**: same as four-well/Mueller

5. **10D Separable** (lines 228-230):
   - SDE: `dX^(i) = -∂_{x_i} V dt + ε dB`
   - **Matches global convention** with `σ = ε`

**Interpretation**: The factor-1/2 rescaling is **equivalent** to defining an effective potential `V_eff = (1/2)V` with noise `σ = ε`. The target distribution becomes `exp(-2·(1/2)V/ε²) = exp(-V/ε²)`, which is still a valid Boltzmann distribution.

---

## Current Code State (Audit Findings)

### File-by-File Drift/Target Analysis

| File | Drift Formula | Target logpi | Manuscript Match? | Notes |
|------|---------------|--------------|-------------------|-------|
| `ring.py:110` | `bx = -dVx` | `logpi = -2V/eps` | ✓ YES | Correct global convention |
| `fourwells.py:88` | `bx = -0.5*dVx` | `logpi = -V/eps²` | ✓ YES | Matches manuscript line 82-86 |
| `mueller.py:110` | `bx = -0.5*dVx` | `logpi = -V/eps²` | ✓ YES | Matches manuscript line 134-138 |
| `high_dim.py:240` | `drift = -gradV` | Target implicit | ✓ YES | Matches manuscript line 228-230 |
| `lennard_jones_potential.py:317` | `bx = -0.5*dVx` | `logpi = -V/eps²` | ✓ YES | Matches manuscript line 192-196 |

**CONCLUSION**: All current code is **CORRECT** and matches the manuscript's example-specific conventions. The apparent "factor 0.5 discrepancy" from the initial audit is actually an intentional rescaling used in the manuscript for certain examples.

### Sampler Implementations

1. **Diffusion** (`step_diff` functions)
   - Present in: `ring.py`, `fourwells.py`, `mueller.py`, `high_dim.py`, `lennard_jones_potential.py`
   - **Status**: ✓ Correct, matches manuscript conventions

2. **LSB-MC / Levy** (`step_levy` functions)
   - Jump-diffusion with compound Poisson jumps + score correction
   - Present in all experiments
   - **Status**: ✓ Correct implementation

3. **MALA** (`step_mala` functions)
   - Present in: `ring.py`, `fourwells.py`, `mueller.py`, `lennard_jones_potential.py`
   - Uses `grad_logpi = -∇V/eps²` (four-well, Mueller, LJ) or `-2∇V/eps` (ring)
   - **Status**: ✓ Correct, consistent with respective target densities

4. **MALA-Levy** (`step_malevy` functions)
   - MALA + symmetric Levy jump MH move
   - **Status**: ✓ Correct

---

## Historical Reference (Deleted FLMC Code)

### FLA.py (Deleted in Commit 843946a)

**Key Observations**:

1. **Potential Naming Convention**:
   - FLA.py used `U` notation (e.g., `U_doublewell`, `gradU_doublewell`)
   - Current code uses `V` notation (matches manuscript)
   - FLA defined: `U_fourwell = 0.5 * ((x²-a²)² + (y²-a²)²)` (line ~319)
   - Current fourwells.py: `V_fourwell = (x²-a²)² + (y²-a²)²`
   - **This is the source of the factor-2 difference!**

2. **FLA Target Density**:
   - FLA used parameter `beta` (inverse temperature)
   - `logpi = -beta * U` (line ~385)
   - For four-well: `U = 0.5*V_current`, so `beta*U = (beta/2)*V_current`

3. **Drift in FLA**:
   - `b = -gradU` (line ~388)
   - Since `U = 0.5*V_current`, this gives `b = -0.5*gradV_current`
   - **Matches current code!**

4. **FLMC Sampler**:
   - Used alpha-stable Levy noise via `sample_symmetric_alpha_stable`
   - Chambers-Mallows-Stuck algorithm for symmetric alpha-stable distributions
   - **Not compound Poisson** (different from LSB-MC)

**Verdict**: FLA.py is **mostly consistent** but used a different potential scaling convention (`U = V/2` for some examples). The deleted code is a useful reference for FLMC implementation but should be adapted to match current `V` notation.

---

## Five Main Manuscript Examples

1. **Ginzburg-Landau Double-Well** (1D)
   - Manuscript: Section 5.1
   - Potential: `V(x) = (1/4)x⁴ - (1/2)x²`
   - SDE: `dX = (X - X³) dt + ε dB` (note: drift = -∇V)
   - Script: `doublewell.py`
   - Outputs: `doublewell_output/doublewell_metrics.png`, `doublewell_bias.png`, `doublewell_final_density.png`
   - **Status**: ✓ Complete with FLMC baseline and average bias metric

2. **Ring Potential** (2D)
   - Manuscript: Section 5.2
   - Script: `ring.py`
   - Outputs: `ring_final_errors_convergence.png`, `ring_final_spatial_error.png`
   - **Status**: ✓ Complete and correct

3. **Four-Well Potential** (2D)
   - Manuscript: Section 5.3
   - Script: `fourwells.py`
   - Outputs: `fourwell_errors.png`, `fourwell_spatial_error.png`
   - **Status**: ✓ Complete and correct

4. **Muller-Brown Potential** (2D)
   - Manuscript: Section 5.4
   - Script: `mueller.py`
   - Outputs: `mueller_analysis_convergence.png`, `mueller_analysis_spatial_err.png`
   - **Status**: ✓ Complete and correct

5. **Lennard-Jones Cluster** (higher-dimensional: N particles in R^d)
   - Manuscript: Section 5.5
   - Script: `lennard_jones_potential.py`
   - Outputs: `lennard_jones_n{N}_d{d}_highdim_metrics.png`
   - **Status**: ✓ Complete and correct

---

## Additional / Auxiliary Experiments

1. **10D Separable Double-Well Benchmark**
   - Manuscript: Section 5.6
   - Script: `high_dim.py`
   - Outputs: `high_dim_metrics.png`
   - **Status**: ✓ Auxiliary benchmark (not one of the five main examples)

---

## Current Metrics

### Low-Dimensional Metrics (1D/2D)
- **W2 / Wasserstein-2 Distance**: via POT library (`ot.sinkhorn2`) or exact 1D quantile matching
- **L1 / MAE**: `∫ |p_empirical - p_true| dx`
- **L2 / RMSE**: `√∫ (p_empirical - p_true)^2 dx`
- **Average Bias**: `|E[observable(X)] - E[observable(X_∞)]|` (for 1D double-well with observable = mean(x))
- Spatial error heatmaps

### High-Dimensional Metrics
- **Sliced W2**: Projection-based Wasserstein distance
- **Orthant L1/L2 Errors**: For separable potentials (`high_dim.py`)
- **Pair-Distance Histogram L1/L2**: For LJ clusters (`lennard_jones_potential.py`)

---

## Future Implementation Plan

### Objective
Add **FLMC (Fractional Langevin Monte Carlo) as a baseline for low-dimensional experiments only**.

### Scope
1. **Double-Well** (1D):
   - Restore from deleted FLA.py
   - Add FLMC sampler (alpha-stable noise)
   - Add **average bias** metric for estimating mean

2. **Ring / Four-Well / Muller-Brown** (2D):
   - Add FLMC sampler
   - Keep W2/L1/L2 as primary metrics (same as current)

3. **Lennard-Jones / High-Dim**:
   - **DO NOT MODIFY** - leave as diffusion vs LSB-MC only

### Implementation Constraints
- FLMC uses **alpha-stable Levy noise** (not compound Poisson)
- Reference: deleted `FLA.py` (commit 843946a)
- FLMC should be a third method alongside diffusion and LSB-MC
- Current plotting infrastructure assumes 2-4 methods; verify flexibility
- **Must match current `V` notation** (not FLA's `U` notation)

---

## Do Not Break List

### Project-Level Invariants

1. **Sampler / Target Consistency**
   - Preserve consistency between target density and samplers
   - Each file is internally consistent (verified above)
   - Do NOT change drift without changing target density formula

2. **Jump-Process Semantics**
   - LSB-MC: compound Poisson with score correction
   - FLMC (future): alpha-stable without score correction
   - Keep these distinct

3. **Metrics and Output Contracts**
   - Preserve metric definitions (W2, L1, L2)
   - Keep reference-sample generation aligned with target distribution

4. **Estimation / Comparison Workflow**
   - Long-time sampling → empirical density
   - Grid-based density estimation with optional smoothing
   - Cross-method comparisons on same grid

5. **Numerical Stability**
   - Tamed Euler-Maruyama: `dt*drift/(1 + dt*norm)`
   - Clipping: `EXPO_CLIP`, `LOGR_CLIP`, `S_CLIP`
   - Jump caps in ring/LJ to prevent explosion

---

## Resolved Issues from Initial Audit

### 1. ✅ "Drift Factor Inconsistency" - RESOLVED
**Initial Finding**: Ring uses `-∇V`, others use `-0.5∇V`

**Resolution**: This is **intentional and correct**:
- Ring: follows global convention directly
- Four-well, Mueller, LJ: follow manuscript's example-specific rescaling (factor 1/2 in drift)
- Both are valid Boltzmann distributions with different effective potentials
- Each file is internally consistent (drift ↔ target density ↔ score correction)

### 2. ✅ "Missing Factor 2 in logpi" - RESOLVED
**Resolution**:
- Ring: `logpi = -2V/eps` (global convention)
- Others: `logpi = -V/eps²` (equivalent to rescaled potential)
- Both match their respective manuscript specifications

### 3. ✅ "MALA Target Consistency" - RESOLVED
**Resolution**: MALA implementations correctly use `grad_logpi` consistent with their respective target densities

---

## Implementation Status

### ✅ Stage 1 Complete: FLMC Core + Double-Well (1D)

**Completed Components**:
1. ✅ **FLMC Utilities** (`flmc_utils.py`)
   - `c_alpha(alpha)`: Normalization constant for alpha-stable noise
   - `sample_symmetric_alpha_stable(rng, size, alpha)`: Chambers-Mallows-Stuck sampler
   - `step_flmc_1d(...)`: FLMC stepping for 1D potentials
   - `step_flmc_2d(...)`: FLMC stepping for 2D potentials

2. ✅ **Double-Well Experiment** (`doublewell.py`)
   - Potential: `V(x) = x⁴/4 - x²/2` (matches manuscript Section 5.1)
   - Three methods:
     - Diffusion: `dX = -∇V dt + σ dB`
     - FLMC: `dX = -c_alpha·∇V dt + dt^(1/alpha) dZ` (Z = alpha-stable)
     - LSB-MC: `dZ = (-∇V + S_L^s) dt + σ dB + dL` (dL = compound Poisson)
   - Metrics: W2, L1, L2, Average Bias (observable = mean(x))
   - Outputs: 3 figures (metrics, bias, final density)

**Validation Results** (updated after Stage 1.5):
- Full experiment runs successfully (T=40, N=8000, α=1.5, λ=1.0)
- Final W2 distances: Diffusion = 1.04, FLMC = 0.34, LSB-MC = 0.16
- Final bias: Diffusion = 0.72, FLMC = 0.005, LSB-MC = 0.012
- LSB-MC achieves best W2 convergence (6.5× faster than diffusion)
- Both FLMC and LSB-MC show significant improvement over pure diffusion

### 🔄 Stage 2 Pending: FLMC Integration for 2D Low-Dimensional Experiments

**Remaining Work**:
1. ⏳ **Ring Potential** (`ring.py`)
   - Add FLMC sampler using `step_flmc_2d`
   - Integrate into existing experiment loop
   - Update plotting to include FLMC curves

2. ⏳ **Four-Well Potential** (`fourwells.py`)
   - Add FLMC sampler
   - Keep existing W2/L1/L2 metrics

3. ⏳ **Muller-Brown Potential** (`mueller.py`)
   - Add FLMC sampler
   - Keep existing W2/L1/L2 metrics

4. ⏳ **Plotting Scripts**
   - Update `plot_density_compare.py` for FLMC
   - Update `plot_absolute_error.py` for FLMC

**Not Planned**:
- ❌ Lennard-Jones: No FLMC (high-dimensional, keep as-is)
- ❌ 10D separable: No FLMC (auxiliary benchmark)

---

## File Structure Summary

```
.
├── flmc_utils.py                # FLMC utilities (alpha-stable sampling)
├── doublewell.py                # Double-well (1D) with diffusion + FLMC
├── doublewell_output/           # Double-well experiment outputs
├── paper/                       # Manuscript TeX source (authoritative)
│   ├── main.tex
│   └── sections/
│       ├── 02_preliminaries.tex  # Global convention (Eq. 7, 36)
│       ├── 03_methodology.tex    # LSB-MC theory (Eq. 27, 32)
│       └── 05_numerical_experiments.tex  # Example-specific SDEs
├── ring.py                      # Ring potential (2D)
├── fourwells.py                 # Four-well potential (2D)
├── mueller.py                   # Muller-Brown potential (2D)
├── high_dim.py                  # Auxiliary 10D separable double-well
├── lennard_jones_potential.py   # LJ cluster (high-D)
├── plot_density_compare.py      # Density visualization
├── plot_absolute_error.py       # Spatial error heatmaps
├── stitch_density_compare.py    # Grid stitching
├── stitch_abs_error.py          # Error stitching
├── density_compare/             # Density plots
├── abs_error/                   # Error heatmaps
├── *.png                        # Output figures
└── CLAUDE.md                    # This file
```

**Deleted (commit 843946a):**
- `FLA.py` - FLMC implementation with rescaled potential notation
- `fla_figures/` - FLMC output plots

---

## Dependencies

- `numpy` (core)
- `matplotlib` (plotting)
- `POT` / `ot` (optimal transport for W2 distance) - **REQUIRED** for most experiments
- `scipy` (optional)

---

## Theory-to-Code Quick Reference

| Manuscript Symbol | Code Variable | Location |
|-------------------|---------------|----------|
| `V(x)` | `V_ring`, `V_fourwell`, `V_mueller`, `V_lj_xy`, `V_high_dim` | Potential functions |
| `∇V(x)` | `gradV_*` | Gradient functions |
| `p_∞(x)` | `pi` | Precomputed on grid |
| `-∇V` or `-0.5∇V` | `bx, by` or `drift` | Diffusion drift (example-specific) |
| `S_L^s` | `Sx, Sy` or `score_int` | Levy score correction |
| `σ` or `ε` | `eps`, `sigma` | Noise scale |
| `λ` | `lam` | Jump intensity |
| `σ_L` | `sigma_L` | Jump magnitude |
| `ν(dr)` | `mults`, `pm` | Jump-size distribution (compound Poisson) |

---

## Manuscript Equation Cross-Reference

- **Eq. 7**: Diffusion SDE `dX = -∇V dt + σ dB`
- **Eq. 36**: Target density `p_∞ ∝ exp(-2V/σ²)`
- **Eq. 27**: Stationary Levy score `S_L^s(x) = -∫₀¹∫ r exp{-2(V(x-θr)-V(x))/σ²} ν(dr) dθ`
- **Eq. 32**: LSB-MC SDE `dZ = (-∇V + S_L^s) dt + σ dB + dL`

---

---

## Recent Updates

**2026-04-05 (Stage 1.5: LSB-MC Added to Double-Well)**
- Implemented 1D LSB-MC score precomputation in `doublewell.py`
- Added `step_lsbmc_1d()` with compound Poisson jumps + score correction
- Updated all metrics and plotting to compare 3 methods
- Validation: LSB-MC shows best performance (W2=0.16 vs 0.34 FLMC vs 1.04 diffusion)
- Double-well now provides complete 3-way comparison for manuscript

**2026-04-05 (Stage 1: FLMC Core + Double-Well)**
- Created `flmc_utils.py` with alpha-stable sampling utilities
- Implemented `doublewell.py` with diffusion + FLMC baselines
- Added average bias metric for observable = mean(x)
- Validated: Full experiment runs successfully with expected convergence

**2026-04-05 (Convention Reconciliation Audit)**
- Resolved all apparent drift/target inconsistencies
- Verified manuscript conventions match code implementations
- Confirmed all files are theoretically correct

---

Last Updated: 2026-04-05
