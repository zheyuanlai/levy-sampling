"""E5 -- alanine dipeptide (Ac-Ala-NHMe, vacuum, 22 atoms) LSC-CP benchmark.

Real-molecule fifth experiment for the LSC-CP JCP benchmark. Modules:

* ``system``    -- deterministic OpenMM system build (setup/validation only) +
                   phi/psi backbone-torsion identification and a numpy dihedral.
* ``cartesian`` -- ``AlanineDipeptideCartesian(Potential)``: batched float64
                   torch force field validated against OpenMM (P1).
* ``bat``       -- differentiable batched BAT transform + analytic log|Jacobian|.
* ``potential`` -- ``AlanineDipeptideBAT(Potential)`` = U_eff with whitening.
* ``reference`` -- well-tempered metadynamics reference FES(phi, psi) loader.
* ``box``       -- ``TorusBox`` (wraps phi, psi to (-pi, pi], wide in stiff DOF).
* ``jump_design``-- torsion jump-atom construction + forbidden-chord drop rule.

OpenMM is used ONLY for setup (params.npz extraction, P0), cross-validation
(P1) and the metadynamics reference (P4). Production energy/score/sampler paths
are pure torch on GPU. See ``README.md`` for pinned versions and units.
"""
