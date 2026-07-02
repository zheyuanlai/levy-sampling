
from __future__ import annotations

import argparse

from experiments.jcp_sampling.core.experiment import run_config


def main():
    ap = argparse.ArgumentParser(description="Run one JCP sampling experiment config.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-root", default="results/jcp_sampling")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    out = run_config(args.config, output_root=args.output_root, tag=args.tag, device=args.device)
    print(out)


if __name__ == "__main__":
    main()
