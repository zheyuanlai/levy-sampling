# AGENTS.md

## Project context
- This repo implements LSB-MC for Boltzmann sampling.
- Manuscript sources are under `paper/`.
- `CLAUDE.md` contains the latest audit and implementation notes; read it before making changes.

## Current state
- Stage 1 and Stage 1.5 are complete.
- Stage 2 is complete for the low-dimensional 2D workflows.
- `flmc_utils.py` exists.
- `doublewell.py` exists and compares Diffusion, FLMC, and LSB-MC.
- `ring.py`, `fourwells.py`, and `mueller.py` now include Diffusion, LSB-MC, FLMC, MALA, and MALA-Levy with W2 / L1 / L2 as the primary metrics.
- `plot_density_compare.py` and `plot_absolute_error.py` include FLMC for the low-dimensional 2D examples.
- `high_dim.py` and `lennard_jones_potential.py` remain unchanged.
- In this sandbox, plain `import ot` currently aborts with `OMP: Error #179: Function Can't open SHM2 failed`, so authoritative POT-backed W2 runs must be done in a normal local environment where `ot` imports cleanly.

## Task constraints
- Preserve the already-verified file-specific conventions.
- Do not reopen convention reconciliation unless a new contradiction is found.
- Keep FLMC distinct from LSB-MC:
  - FLMC = alpha-stable Levy noise, no score correction
  - LSB-MC = compound Poisson + Levy-score correction
- For 2D experiments, keep W2 / L1 / L2 as the primary metrics.
- Do not add metastability-specific or compound-Poisson-specific metrics.
- Do not modify:
  - `high_dim.py`
  - `lennard_jones_potential.py`

## Validation
- Run smoke tests after edits.
- Report exact commands used.
- Preserve existing methods:
  - Diffusion
  - LSB-MC
  - MALA
  - MALA-Levy
  - plus FLMC
