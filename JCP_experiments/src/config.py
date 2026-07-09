"""Global constants and run configuration for the LSC-CP JCP benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------- physics
BETA: float = 8.0                 # inverse temperature
EPS: float = 1.0 / BETA           # epsilon = 1/beta = 0.125
LAMBDA: float = 1.0               # jump intensity (all experiments)

# ---------------------------------------------------------------- numerics
M_MAX: float = 600.0              # cap on the log-magnitude of the Levy score
Q_THETA: int = 16                 # Gauss-Legendre nodes on [0,1] (theta)
Q_RHO: int = 8                    # Gauss-Legendre nodes on [-h,h] (shell rho)
M_PHI: int = 32                   # trapezoid directions for the MoG40 score
K_MAX_JUMPS: int = 8              # Poisson truncation: P(N>8 | lam*dt<=0.01) < 1e-20

# ---------------------------------------------------------------- protocol
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
N_WARMUP_STEPS: int = 20          # untimed warm-up steps (allocator/JIT)
N_CHECKPOINTS: int = 50           # metric cadence, fixed in t, shared by methods

METHODS: tuple[str, ...] = ("ULA", "MALA", "FLA", "BAOAB", "PT", "CP", "LSC-CP")

# pretty labels used in figures / prose (BAOAB is kinetic Langevin, NOT HMC)
METHOD_LABELS: dict[str, str] = {
    "ULA": "ULA",
    "MALA": "MALA",
    "FLA": "FLA",
    "BAOAB": "kinetic Langevin (BAOAB)",
    "PT": "parallel tempering",
    "CP": "raw CP",
    "LSC-CP": "LSC-CP",
}


@dataclass
class RunConfig:
    """Per-experiment production run configuration."""
    name: str
    d: int
    n_particles: int
    T: float
    dt: float
    lam: float = LAMBDA
    beta: float = BETA
    n_checkpoints: int = N_CHECKPOINTS
    seeds: tuple[int, ...] = SEEDS
    q_theta: int = Q_THETA
    q_rho: int = Q_RHO
    extra: dict = field(default_factory=dict)

    @property
    def eps(self) -> float:
        return 1.0 / self.beta

    @property
    def n_steps(self) -> int:
        return int(round(self.T / self.dt))

    @property
    def steps_per_checkpoint(self) -> int:
        return max(1, self.n_steps // self.n_checkpoints)


# deterministic RNG scheme: one generator per (method, seed); the jump stream
# is a dedicated generator seeded identically for raw CP and LSC-CP so the two
# are pathwise coupled (same jump times and increments).
METHOD_SEED_BASE: dict[str, int] = {m: 10_000 * (i + 1) for i, m in enumerate(METHODS)}
JUMP_STREAM_BASE: int = 900_000
INIT_STREAM_BASE: int = 800_000
REF_STREAM_BASE: int = 700_000


def diffusion_seed(method: str, seed: int) -> int:
    return METHOD_SEED_BASE[method] + seed


def jump_seed(seed: int) -> int:
    # shared by CP and LSC-CP on purpose (pathwise coupling)
    return JUMP_STREAM_BASE + seed


def init_seed(seed: int) -> int:
    return INIT_STREAM_BASE + seed
