
from __future__ import annotations

from experiments.jcp_sampling.core.experiment import run_config

SMOKE_CONFIGS = [
    "experiments/jcp_sampling/configs/smoke/double_well_smoke.yaml",
    "experiments/jcp_sampling/configs/smoke/four_well_smoke.yaml",
    "experiments/jcp_sampling/configs/smoke/muller_brown_smoke.yaml",
    "experiments/jcp_sampling/configs/smoke/manywell_smoke.yaml",
]


def main():
    for cfg in SMOKE_CONFIGS:
        run_config(cfg, output_root="results/jcp_sampling", tag="smoke")


if __name__ == "__main__":
    main()
