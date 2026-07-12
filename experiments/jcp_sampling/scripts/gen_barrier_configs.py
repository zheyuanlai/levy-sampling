"""Generate the barrier-height-sweep configs (Experiment A: barrier-free-theorem test).

Emits two families under configs/double_well_barrier/:
  rate/  -- start-at-equilibrium runs measuring the inter-well transition rate k(H) (primary);
            swept over H at eps in {0.5, 0.25}. k = transition_rate_per_time (mixing_metrics).
  mfpt/  -- metastable (left-well) start measuring MFPT/threshold time (secondary); eps=0.5 only.
            coverage_time_all_basins == mean first left->right passage for the 2-basin system.

Same fixed +-2 shell jump bank (double_well_shell scale 1.0, intensity 1.0) at every H, so any
H-independence of the LSC-CP rate is a property of the method, not tuning. Also writes a suite
file configs/suites/barrier.yaml listing all configs. Run:

  python -m experiments.jcp_sampling.scripts.gen_barrier_configs
"""
from __future__ import annotations

from pathlib import Path

H_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
EPS_GRID = [0.5, 0.25]

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs" / "double_well_barrier"
SUITE = ROOT / "configs" / "suites" / "barrier.yaml"

BANK = "- {kind: double_well_shell, scale: 1.0, intensity: 1.0}"
METHOD_CFGS = (
    "method_cfgs:\n"
    "  RawCP: {dt: 0.003, n_theta: 8, jump_chunk: 8, tame_cap: 5.0}\n"
    "  LSBMC: {dt: 0.003, n_theta: 8, jump_chunk: 8, tame_cap: 5.0}\n"
)


def _tag(v: float) -> str:
    return str(v).replace(".", "").rstrip("0") or "0"


def rate_config(H: float, eps: float) -> str:
    return (
        f"# Barrier-free-theorem sweep (Exp A) RATE point: V=H(x^2-1)^2, H={H}, eps={eps}.\n"
        f"# Start at Gibbs equilibrium; inter-well transition rate k=transition_rate_per_time is\n"
        f"# lambda_inter. Fixed +-2 shell bank (H-independent). Predicted: local k ~ exp(-H/eps),\n"
        f"# LSC-CP k flat; raw CP k flat but biased equilibrium (CDF-sup elevated).\n"
        f"experiment_name: barrier_rate_H{_tag(H)}_eps{_tag(eps)}\n"
        f"target: double_well_barrier\n"
        f"target_cfg: {{H: {H}, eps: {eps}}}\n"
        f"jump_banks:\n{BANK}\n"
        f"run:\n"
        f"  n_particles: 2048\n"
        f"  n_steps: 20000\n"
        f"  dt: 0.003\n"
        f"  record_every: 200\n"
        f"  iat_points: 2000\n"
        f"  seeds: [0, 1, 2]\n"
        f"  n_ref: 20000\n"
        f"  init: equilibrium\n"
        f"reference: {{n_ref: 20000, seed: 321}}\n"
        f"methods: [ULA, RawCP, LSBMC]\n"
        f"{METHOD_CFGS}"
    )


def mfpt_config(H: float, eps: float) -> str:
    return (
        f"# Barrier-free-theorem sweep (Exp A) MFPT point: V=H(x^2-1)^2, H={H}, eps={eps}.\n"
        f"# Metastable left-well start; coverage_time_all_basins = mean first left->right passage\n"
        f"# (MFPT), threshold_time_tv = mixing time. Local MFPT ~ exp(H/eps); LSC-CP flat. High-H\n"
        f"# local points are censored (coverage_fraction<1) -- read alongside the rate runs.\n"
        f"experiment_name: barrier_mfpt_H{_tag(H)}_eps{_tag(eps)}\n"
        f"target: double_well_barrier\n"
        f"target_cfg: {{H: {H}, eps: {eps}}}\n"
        f"jump_banks:\n{BANK}\n"
        f"run:\n"
        f"  n_particles: 2048\n"
        f"  n_steps: 50000\n"
        f"  dt: 0.003\n"
        f"  record_every: 500\n"
        f"  iat_points: 2500\n"
        f"  seeds: [0, 1, 2]\n"
        f"  n_ref: 20000\n"
        f"reference: {{n_ref: 20000, seed: 321}}\n"
        f"methods: [ULA, RawCP, LSBMC]\n"
        f"{METHOD_CFGS}"
    )


def main() -> None:
    (CFG_DIR / "rate").mkdir(parents=True, exist_ok=True)
    (CFG_DIR / "mfpt").mkdir(parents=True, exist_ok=True)
    paths = []
    for eps in EPS_GRID:
        for H in H_GRID:
            p = CFG_DIR / "rate" / f"H{_tag(H)}_eps{_tag(eps)}.yaml"
            p.write_text(rate_config(H, eps))
            paths.append(p)
    # MFPT only at eps=0.5 and H<=2 (where local relaxes within budget; higher H is censored and
    # covered by the rate runs).
    for H in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        p = CFG_DIR / "mfpt" / f"H{_tag(H)}_eps05.yaml"
        p.write_text(mfpt_config(H, 0.5))
        paths.append(p)
    rel = [str(p.relative_to(ROOT.parents[1])) for p in paths]
    SUITE.write_text(
        "# Experiment A: barrier-height sweep (barrier-free-theorem test).\n"
        "configs:\n" + "".join(f"- {r}\n" for r in rel)
    )
    print(f"wrote {len(paths)} configs; suite -> {SUITE}")


if __name__ == "__main__":
    main()
