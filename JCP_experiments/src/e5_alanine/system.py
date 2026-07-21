"""Deterministic OpenMM build of alanine dipeptide + backbone-torsion tools.

SETUP / VALIDATION ONLY.  Nothing here is called in the sampler/score hot loop;
production energy is the torch code in ``cartesian.py`` / ``potential.py``.

The reference system is ``openmmtools.testsystems.AlanineDipeptideVacuum`` built
**fully flexible** (``constraints=None``): 22 atoms, D = 66 Cartesian DOF, all
bonds harmonic, no holonomic constraints -- exactly the smooth potential the
torch force field and the BAT internal-coordinate model represent.

Units follow OpenMM: energy kJ/mol, length nm, charge e (elementary), angles rad.
"""
from __future__ import annotations

import os

# pymbar (an openmmtools dependency) imports JAX, which otherwise grabs the
# torch-visible GPU.  Force JAX onto the CPU BEFORE openmmtools is imported so a
# process that mixes this module with torch-on-GPU is safe.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from dataclasses import dataclass

import numpy as np

# --- atom identity (verified in P0 against the built topology) --------------
N_ATOMS = 22
# phi = tau(ACE:C, ALA:N, ALA:CA, ALA:C); psi = tau(ALA:N, ALA:CA, ALA:C, NME:N)
PHI_ATOM_NAMES = (("ACE", "C"), ("ALA", "N"), ("ALA", "CA"), ("ALA", "C"))
PSI_ATOM_NAMES = (("ALA", "N"), ("ALA", "CA"), ("ALA", "C"), ("NME", "N"))

PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.npz")


@dataclass
class AlanineSystem:
    system: object          # openmm.System (flexible, constraints=None)
    topology: object        # openmm.app.Topology
    positions_nm: np.ndarray  # (22, 3) float64, reference conformer, nm
    atom_residues: np.ndarray  # (22,) <U
    atom_names: np.ndarray     # (22,) <U
    elements: np.ndarray       # (22,) <U
    masses_amu: np.ndarray     # (22,) float64
    phi_quartet: tuple         # 4 atom indices
    psi_quartet: tuple         # 4 atom indices


def build_alanine_system() -> AlanineSystem:
    """Build the deterministic flexible alanine-dipeptide vacuum system."""
    import openmm.unit as unit
    from openmmtools import testsystems

    ts = testsystems.AlanineDipeptideVacuum(constraints=None)
    system, topology = ts.system, ts.topology
    if system.getNumConstraints() != 0:
        raise RuntimeError("expected a constraint-free (flexible) system")
    if system.getNumParticles() != N_ATOMS:
        raise RuntimeError(f"expected {N_ATOMS} atoms, got {system.getNumParticles()}")

    positions_nm = np.array(
        ts.positions.value_in_unit(unit.nanometer), dtype=np.float64)

    residues, names, elems, masses = [], [], [], []
    for a in topology.atoms():
        residues.append(a.residue.name)
        names.append(a.name)
        elems.append(a.element.symbol if a.element is not None else "?")
    for i in range(system.getNumParticles()):
        masses.append(system.getParticleMass(i).value_in_unit(unit.amu))
    name2idx = {(r, n): i for i, (r, n) in enumerate(zip(residues, names))}

    phi = tuple(name2idx[key] for key in PHI_ATOM_NAMES)
    psi = tuple(name2idx[key] for key in PSI_ATOM_NAMES)

    return AlanineSystem(
        system=system, topology=topology, positions_nm=positions_nm,
        atom_residues=np.array(residues), atom_names=np.array(names),
        elements=np.array(elems), masses_amu=np.array(masses, dtype=np.float64),
        phi_quartet=phi, psi_quartet=psi)


def dihedral(pos: np.ndarray, quartet) -> float:
    """Signed dihedral tau(p1,p2,p3,p4) in (-pi, pi] via the atan2 form (S1.8).

    pos: (..., N, 3); quartet: 4 atom indices.  Returns the angle(s) in radians.
    """
    p1, p2, p3, p4 = (pos[..., quartet[k], :] for k in range(4))
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2n = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    x = np.sum(n1 * n2, axis=-1)
    y = np.sum(np.cross(n1, n2) * b2n, axis=-1)
    return np.arctan2(y, x)


def mdtraj_phi_psi(topology, positions_nm: np.ndarray):
    """(phi_quartet, psi_quartet, phi_val, psi_val) from mdtraj compute_phi/psi.

    Cross-check of the by-name quartet identification and the dihedral values.
    """
    import mdtraj as md

    xyz = np.asarray(positions_nm, dtype=np.float64).reshape(1, -1, 3)
    traj = md.Trajectory(xyz, md.Topology.from_openmm(topology))
    phi_idx, phi_val = md.compute_phi(traj)
    psi_idx, psi_val = md.compute_psi(traj)
    return (tuple(int(i) for i in phi_idx[0]),
            tuple(int(i) for i in psi_idx[0]),
            float(phi_val[0, 0]), float(psi_val[0, 0]))


def openmm_reference_context(system):
    """A double-precision Reference-platform Context for energy/force checks."""
    import openmm
    import openmm.unit as unit

    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, integrator, platform)
    return context


def openmm_energy_force(context, positions_nm: np.ndarray):
    """(E kJ/mol, F kJ/mol/nm) for one (N,3) conformer via a Reference context."""
    import openmm.unit as unit

    context.setPositions(positions_nm * unit.nanometer)
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = np.array(
        state.getForces().value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer), dtype=np.float64)
    return float(energy), forces


def probe_one_4pi_eps0() -> float:
    """OpenMM's exact Coulomb constant (kJ*nm)/(mol*e^2) via a two-charge probe."""
    import openmm
    import openmm.unit as unit

    probe = openmm.System()
    probe.addParticle(1.0)
    probe.addParticle(1.0)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    nb.addParticle(1.0 * unit.elementary_charge, 1.0 * unit.nanometer,
                   0.0 * unit.kilojoule_per_mole)
    nb.addParticle(-1.0 * unit.elementary_charge, 1.0 * unit.nanometer,
                   0.0 * unit.kilojoule_per_mole)
    probe.addForce(nb)
    ctx = openmm_reference_context(probe)
    # E = k * q1 q2 / r = k * (-1) / 2  ->  k = -2 E
    energy, _ = openmm_energy_force(ctx, np.array([[0.0, 0.0, 0.0],
                                                   [2.0, 0.0, 0.0]]))
    return -2.0 * energy


def load_params() -> dict:
    """Load the committed force-field parameters (P0) as a dict of arrays."""
    if not os.path.exists(PARAMS_PATH):
        raise FileNotFoundError(
            f"{PARAMS_PATH} missing; run "
            "`python -m src.e5_alanine.extract_params` to regenerate it (P0).")
    with np.load(PARAMS_PATH, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}
