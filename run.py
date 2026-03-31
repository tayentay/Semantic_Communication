from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from semantic_comm import PSComOptimizer, load_config


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run SAGIN-enabled PSCom alternating optimization."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=base_dir / "config" / "default.yaml",
        help="Path to YAML configuration file (defaults relative to this script).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save objective/component curves to --plot-path.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=base_dir / "outputs" / "pscom_history.png",
        help="Where to save the plot when --plot is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    opt = PSComOptimizer(cfg)
    final, history = opt.run()
    print("\nFinal objective (energy units):", f"{final.objective:.4e}")
    print("Iterations:", len(history))
    print("a_s:", opt.a_s)
    print("a_u:", opt.a_u)
    print("eta:", opt.eta)
    print("fk:", opt.fk)
    print("bk:", opt.bk)
    print("pk:", opt.pk)
    print("HU:", opt.hu, "theta:", opt.theta, "LU:", opt.lu)

    if args.plot and history:
        args.plot_path.parent.mkdir(parents=True, exist_ok=True)
        iters = list(range(1, len(history) + 1))
        obj = [s.objective for s in history]
        e_s = [s.e_s for s in history]
        e_su = [s.e_su for s in history]
        e_u = [s.e_u for s in history]
        e_ug = [s.e_ug for s in history]

        plt.figure(figsize=(7, 4))
        plt.plot(iters, obj, label="objective", linewidth=2)
        plt.plot(iters, e_s, label="e_s")
        plt.plot(iters, e_su, label="e_su")
        plt.plot(iters, e_u, label="e_u")
        plt.plot(iters, e_ug, label="e_ug")
        plt.xlabel("Iteration")
        plt.ylabel("Energy")
        plt.title("PSCom optimization history")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot_path)
        plt.close()
        print("Saved plot to:", args.plot_path)


if __name__ == "__main__":
    main()
