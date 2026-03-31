from __future__ import annotations

import argparse
from pathlib import Path

from semantic_comm import PSComOptimizer, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAGIN-enabled PSCom alternating optimization."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    opt = PSComOptimizer(cfg)
    final, history = opt.run()
    print("\nFinal objective:", f"{final.objective:.4e}")
    print("Iterations:", len(history))
    print("a_s:", opt.a_s)
    print("a_u:", opt.a_u)
    print("eta:", opt.eta)
    print("fk:", opt.fk)
    print("bk:", opt.bk)
    print("pk:", opt.pk)
    print("HU:", opt.hu, "theta:", opt.theta, "LU:", opt.lu)


if __name__ == "__main__":
    main()
