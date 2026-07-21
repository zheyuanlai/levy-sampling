"""P0: extract the alanine-dipeptide force-field parameters into params.npz.

Run once; the resulting ``params.npz`` is committed and is the single source of
truth for the torch force field (``cartesian.py``).  Production never calls
OpenMM.  Everything is in OpenMM units: energy kJ/mol, length nm, charge e, angle
rad.

Usage:  python -m src.e5_alanine.extract_params
"""
from __future__ import annotations

import json

import numpy as np

from .system import (PARAMS_PATH, build_alanine_system, dihedral,
                     mdtraj_phi_psi, probe_one_4pi_eps0)

UNITS = {
    "energy": "kJ/mol", "length": "nm", "charge": "e (elementary)",
    "angle": "rad", "bond_k": "kJ/(mol nm^2)", "angle_k": "kJ/(mol rad^2)",
    "torsion_k": "kJ/mol", "eps": "kJ/mol", "sigma": "nm",
    "chargeProd": "e^2", "one_4pi_eps0": "kJ nm / (mol e^2)",
}


def extract() -> dict:
    """Build the flexible OpenMM system and return every FF term as arrays."""
    import openmm
    import openmm.unit as unit

    ala = build_alanine_system()
    system = ala.system
    forces = {f.__class__.__name__: f for f in system.getForces()}

    for required in ("HarmonicBondForce", "HarmonicAngleForce",
                     "PeriodicTorsionForce", "NonbondedForce"):
        if required not in forces:
            raise RuntimeError(f"missing {required} in the OpenMM system")

    # -- harmonic bonds: 0.5 k (r - r0)^2 ------------------------------------
    hb = forces["HarmonicBondForce"]
    bond_idx, bond_r0, bond_k = [], [], []
    for b in range(hb.getNumBonds()):
        i, j, r0, k = hb.getBondParameters(b)
        bond_idx.append((i, j))
        bond_r0.append(r0.value_in_unit(unit.nanometer))
        bond_k.append(k.value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer ** 2))

    # -- harmonic angles: 0.5 k (theta - theta0)^2 ---------------------------
    ha = forces["HarmonicAngleForce"]
    angle_idx, angle_t0, angle_k = [], [], []
    for a in range(ha.getNumAngles()):
        i, j, k_, t0, kk = ha.getAngleParameters(a)
        angle_idx.append((i, j, k_))
        angle_t0.append(t0.value_in_unit(unit.radian))
        angle_k.append(kk.value_in_unit(
            unit.kilojoule_per_mole / unit.radian ** 2))

    # -- periodic torsions: k (1 + cos(n phi - phase)) -----------------------
    pt = forces["PeriodicTorsionForce"]
    tor_idx, tor_n, tor_phase, tor_k = [], [], [], []
    for t in range(pt.getNumTorsions()):
        i, j, k_, l, n, phase, kk = pt.getTorsionParameters(t)
        tor_idx.append((i, j, k_, l))
        tor_n.append(int(n))
        tor_phase.append(phase.value_in_unit(unit.radian))
        tor_k.append(kk.value_in_unit(unit.kilojoule_per_mole))

    # -- nonbonded per-atom (charge, sigma, eps) -----------------------------
    nb = forces["NonbondedForce"]
    if nb.getNonbondedMethod() != openmm.NonbondedForce.NoCutoff:
        raise RuntimeError("vacuum system must use NoCutoff nonbonded method")
    q, sig, eps = [], [], []
    for p in range(nb.getNumParticles()):
        c, s, e = nb.getParticleParameters(p)
        q.append(c.value_in_unit(unit.elementary_charge))
        sig.append(s.value_in_unit(unit.nanometer))
        eps.append(e.value_in_unit(unit.kilojoule_per_mole))

    # -- nonbonded exceptions (excluded 1-2/1-3 + scaled 1-4) ----------------
    exc_idx, exc_qq, exc_sig, exc_eps = [], [], [], []
    for x in range(nb.getNumExceptions()):
        i, j, qq, s, e = nb.getExceptionParameters(x)
        exc_idx.append((i, j))
        exc_qq.append(qq.value_in_unit(unit.elementary_charge ** 2))
        exc_sig.append(s.value_in_unit(unit.nanometer))
        exc_eps.append(e.value_in_unit(unit.kilojoule_per_mole))

    one_4pi_eps0 = probe_one_4pi_eps0()

    # -- phi/psi cross-check (by-name vs mdtraj) -----------------------------
    phi_md, psi_md, phi_v_md, psi_v_md = mdtraj_phi_psi(
        ala.topology, ala.positions_nm)
    if phi_md != ala.phi_quartet or psi_md != ala.psi_quartet:
        raise RuntimeError(
            f"phi/psi quartet mismatch: by-name {ala.phi_quartet}/{ala.psi_quartet} "
            f"vs mdtraj {phi_md}/{psi_md}")
    phi_v = float(dihedral(ala.positions_nm, ala.phi_quartet))
    psi_v = float(dihedral(ala.positions_nm, ala.psi_quartet))

    def _wrap_diff(a, b):
        return float(abs(((a - b + np.pi) % (2 * np.pi)) - np.pi))

    if _wrap_diff(phi_v, phi_v_md) > 1e-6 or _wrap_diff(psi_v, psi_v_md) > 1e-6:
        raise RuntimeError("dihedral value disagrees with mdtraj beyond 1e-6 rad")

    return dict(
        n_atoms=np.int64(ala.system.getNumParticles()),
        one_4pi_eps0=np.float64(one_4pi_eps0),
        atom_residues=ala.atom_residues, atom_names=ala.atom_names,
        elements=ala.elements, masses_amu=ala.masses_amu,
        ref_positions_nm=ala.positions_nm,
        phi_quartet=np.array(ala.phi_quartet, dtype=np.int64),
        psi_quartet=np.array(ala.psi_quartet, dtype=np.int64),
        bond_idx=np.array(bond_idx, dtype=np.int64),
        bond_r0=np.array(bond_r0), bond_k=np.array(bond_k),
        angle_idx=np.array(angle_idx, dtype=np.int64),
        angle_theta0=np.array(angle_t0), angle_k=np.array(angle_k),
        torsion_idx=np.array(tor_idx, dtype=np.int64),
        torsion_periodicity=np.array(tor_n, dtype=np.int64),
        torsion_phase=np.array(tor_phase), torsion_k=np.array(tor_k),
        nb_charge=np.array(q), nb_sigma=np.array(sig), nb_eps=np.array(eps),
        exc_idx=np.array(exc_idx, dtype=np.int64),
        exc_chargeProd=np.array(exc_qq), exc_sigma=np.array(exc_sig),
        exc_eps=np.array(exc_eps),
        n_bonds=np.int64(hb.getNumBonds()),
        n_angles=np.int64(ha.getNumAngles()),
        n_torsions=np.int64(pt.getNumTorsions()),
        n_exceptions=np.int64(nb.getNumExceptions()),
        n_nb_particles=np.int64(nb.getNumParticles()),
        units_json=np.array(json.dumps(UNITS)),
        openmm_version=np.array(openmm.__version__),
    )


def main() -> None:
    params = extract()
    np.savez(PARAMS_PATH, **params)
    print(f"wrote {PARAMS_PATH}")
    print(f"  atoms={int(params['n_atoms'])} bonds={int(params['n_bonds'])} "
          f"angles={int(params['n_angles'])} torsions={int(params['n_torsions'])} "
          f"exceptions={int(params['n_exceptions'])}")
    print(f"  phi={params['phi_quartet'].tolist()} "
          f"psi={params['psi_quartet'].tolist()} "
          f"ONE_4PI_EPS0={float(params['one_4pi_eps0']):.11f}")


if __name__ == "__main__":
    main()
