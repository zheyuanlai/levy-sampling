
from pathlib import Path

import yaml

from experiments.jcp_sampling.core.jump_banks import build_jump_bank
from experiments.jcp_sampling.core.potentials import build_potential


def test_smoke_configs_build_targets_and_banks():
    for path in Path("experiments/jcp_sampling/configs/smoke").glob("*.yaml"):
        cfg = yaml.safe_load(path.read_text())
        pot = build_potential({"kind": cfg["target"], "target_cfg": cfg.get("target_cfg", {})})
        assert pot.dim > 0
        for b in cfg.get("jump_banks", []):
            bank = build_jump_bank(b.get("kind"), pot, b)
            assert bank.dim == pot.dim
            assert bank.weights.sum().item() > 0
