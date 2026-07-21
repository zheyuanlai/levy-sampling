"""E5 P0 gate: the committed params.npz faithfully mirrors the OpenMM system.

Gate (task S P0):
  * phi/psi index sets match mdtraj compute_phi/compute_psi;
  * dihedral values agree with mdtraj to 1e-6 rad;
  * every force term in the OpenMM System is represented in params (assert the
    counts of bonds/angles/torsions/exceptions/particles);
  * units recorded; ONE_4PI_EPS0 matches OpenMM's Coulomb constant.

CPU/OpenMM only (no torch, no GPU).
"""
from __future__ import annotations

import numpy as np

from src.e5_alanine.system import (build_alanine_system, dihedral,
                                    load_params, mdtraj_phi_psi,
                                    probe_one_4pi_eps0)


def _wrap_diff(a, b):
    return float(abs(((a - b + np.pi) % (2 * np.pi)) - np.pi))


def test_params_match_openmm_system():
    params = load_params()
    ala = build_alanine_system()
    system = ala.system
    forces = {f.__class__.__name__: f for f in system.getForces()}

    # -- every force term is represented (counts) ----------------------------
    assert int(params["n_atoms"]) == system.getNumParticles() == 22
    assert int(params["n_bonds"]) == forces["HarmonicBondForce"].getNumBonds()
    assert int(params["n_angles"]) == forces["HarmonicAngleForce"].getNumAngles()
    assert int(params["n_torsions"]) == forces["PeriodicTorsionForce"].getNumTorsions()
    assert int(params["n_exceptions"]) == forces["NonbondedForce"].getNumExceptions()
    assert int(params["n_nb_particles"]) == forces["NonbondedForce"].getNumParticles()
    # concrete expected cardinalities for the flexible 22-atom system
    assert (int(params["n_bonds"]), int(params["n_angles"]),
            int(params["n_torsions"]), int(params["n_exceptions"])) == (21, 36, 52, 98)
    # array shapes agree with the recorded counts
    assert params["bond_idx"].shape == (21, 2)
    assert params["angle_idx"].shape == (36, 3)
    assert params["torsion_idx"].shape == (52, 4)
    assert params["exc_idx"].shape == (98, 2)
    assert params["nb_charge"].shape == params["nb_sigma"].shape == (22,)

    # -- flexible (no constraints), NoCutoff vacuum --------------------------
    assert system.getNumConstraints() == 0
    import openmm
    assert forces["NonbondedForce"].getNonbondedMethod() == openmm.NonbondedForce.NoCutoff

    # -- phi/psi by-name == mdtraj, values agree to 1e-6 rad -----------------
    phi_md, psi_md, phi_v_md, psi_v_md = mdtraj_phi_psi(
        ala.topology, ala.positions_nm)
    assert tuple(params["phi_quartet"].tolist()) == phi_md == ala.phi_quartet
    assert tuple(params["psi_quartet"].tolist()) == psi_md == ala.psi_quartet
    phi_v = float(dihedral(ala.positions_nm, ala.phi_quartet))
    psi_v = float(dihedral(ala.positions_nm, ala.psi_quartet))
    assert _wrap_diff(phi_v, phi_v_md) < 1e-6, _wrap_diff(phi_v, phi_v_md)
    assert _wrap_diff(psi_v, psi_v_md) < 1e-6, _wrap_diff(psi_v, psi_v_md)

    # -- ONE_4PI_EPS0 matches OpenMM's Coulomb constant ----------------------
    assert abs(float(params["one_4pi_eps0"]) - probe_one_4pi_eps0()) < 1e-9
    assert abs(float(params["one_4pi_eps0"]) - 138.93545764438) < 1e-6

    # -- units metadata present ---------------------------------------------
    import json
    units = json.loads(str(params["units_json"]))
    for key in ("energy", "length", "charge", "angle"):
        assert key in units
    assert units["energy"] == "kJ/mol" and units["length"] == "nm"


def test_params_reproduce_openmm_bond_angle_torsion_values():
    """Reloading the committed params equals a fresh OpenMM extraction exactly."""
    import openmm.unit as unit

    params = load_params()
    ala = build_alanine_system()
    forces = {f.__class__.__name__: f for f in ala.system.getForces()}

    hb = forces["HarmonicBondForce"]
    for b in range(hb.getNumBonds()):
        i, j, r0, k = hb.getBondParameters(b)
        assert tuple(params["bond_idx"][b]) == (i, j)
        assert abs(params["bond_r0"][b] - r0.value_in_unit(unit.nanometer)) < 1e-12
        assert abs(params["bond_k"][b]
                   - k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer ** 2)) < 1e-6

    ha = forces["HarmonicAngleForce"]
    for a in range(ha.getNumAngles()):
        i, j, k_, t0, kk = ha.getAngleParameters(a)
        assert tuple(params["angle_idx"][a]) == (i, j, k_)
        assert abs(params["angle_theta0"][a] - t0.value_in_unit(unit.radian)) < 1e-6

    pt = forces["PeriodicTorsionForce"]
    for t in range(pt.getNumTorsions()):
        i, j, k_, l, n, phase, kk = pt.getTorsionParameters(t)
        assert tuple(params["torsion_idx"][t]) == (i, j, k_, l)
        assert int(params["torsion_periodicity"][t]) == int(n)
        assert abs(params["torsion_phase"][t] - phase.value_in_unit(unit.radian)) < 1e-12

    # exception pairs are unique unordered pairs, covering all 1-2/1-3/1-4
    exc_pairs = {frozenset(map(int, p)) for p in params["exc_idx"]}
    assert len(exc_pairs) == 98
