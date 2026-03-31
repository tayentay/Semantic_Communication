from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from semantic_comm import PSComOptimizer, load_config


SCHEMES = ["pscom", "no_semantic", "sat_only", "uav_only", "random"]


def _freeze_updates(
    opt: PSComOptimizer, *, task: bool = False, semantic: bool = False, geometry: bool = False
) -> None:
    """Optionally freeze subsets of optimizer updates for baselines."""

    if task:
        opt._update_task_allocation = lambda: None  # type: ignore[attr-defined]
    if semantic:
        opt._update_semantic_ratio = lambda: None  # type: ignore[attr-defined]
    if geometry:
        opt._update_altitude_beamwidth = lambda: None  # type: ignore[attr-defined]
        opt._update_location = lambda: None  # type: ignore[attr-defined]


def _run_scheme(name: str, cfg, rng: np.random.Generator) -> Dict:
    opt = PSComOptimizer(cfg)
    K = opt.K

    if name == "pscom":
        final, history = opt.run()
    elif name == "no_semantic":
        opt.a_s = np.zeros(K)
        opt.a_u = np.zeros(K)
        opt.eta = np.ones(K)
        _freeze_updates(opt, task=True, semantic=True)
        final, history = opt.run()
    elif name == "sat_only":
        opt.a_s = np.ones(K)
        opt.a_u = np.zeros(K)
        _freeze_updates(opt, task=True, semantic=True)
        final, history = opt.run()
    elif name == "uav_only":
        opt.a_s = np.zeros(K)
        opt.a_u = np.ones(K)
        _freeze_updates(opt, task=True, semantic=True)
        final, history = opt.run()
    elif name == "random":
        modes = rng.choice(["raw", "sat", "uav"], size=K)
        a_s = np.zeros(K)
        a_u = np.zeros(K)
        eta = np.zeros(K)
        for idx, mode in enumerate(modes):
            if mode == "sat":
                a_s[idx] = 1.0
            elif mode == "uav":
                a_u[idx] = 1.0
            lo = opt.gts[idx].min_compression
            eta[idx] = rng.uniform(lo, 1.0)
        opt.a_s = a_s
        opt.a_u = a_u
        opt.eta = eta

        uav = opt.scenario.uav
        opt.theta = float(rng.uniform(uav.beamwidth_range[0], uav.beamwidth_range[1]))
        opt.hu = float(rng.uniform(uav.height_range[0], uav.height_range[1]))
        xs = [gt.x for gt in opt.gts]
        ys = [gt.y for gt in opt.gts]
        pad = opt.cfg.location_grid_step
        opt.lu = np.array(
            [
                rng.uniform(min(xs) - pad, max(xs) + pad),
                rng.uniform(min(ys) - pad, max(ys) + pad),
            ]
        )

        _freeze_updates(opt, task=True, semantic=True, geometry=True)
        final, history = opt.run()
    else:
        raise ValueError(f"Unknown scheme: {name}")

    return {
        "scheme": name,
        "objective": final.objective,
        "e_s": final.e_s,
        "e_su": final.e_su,
        "e_u": final.e_u,
        "e_ug": final.e_ug,
        "iterations": len(history),
    }


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Compare PSCom baselines from the paper (full, no semantic, sat-only, UAV-only, random)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=base_dir / "config" / "default.yaml",
        help="Path to YAML configuration file (defaults relative to this script).",
    )
    parser.add_argument(
        "--schemes",
        type=str,
        default=",".join(SCHEMES),
        help="Comma-separated list of schemes to run. Options: " + ",".join(SCHEMES),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for random baseline reproducibility.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write results as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    requested = [s.strip() for s in args.schemes.split(",") if s.strip()]
    unknown = [s for s in requested if s not in SCHEMES]
    if unknown:
        raise ValueError(f"Unknown schemes: {unknown}. Allowed: {SCHEMES}")

    results: List[Dict] = []
    for scheme in requested:
        res = _run_scheme(scheme, cfg, rng)
        results.append(res)
        print(
            f"[{scheme}] obj={res['objective']:.4e} "
            f"(e_s={res['e_s']:.3e}, e_su={res['e_su']:.3e}, e_u={res['e_u']:.3e}, e_ug={res['e_ug']:.3e}) "
            f"iters={res['iterations']}"
        )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["scheme", "objective", "e_s", "e_su", "e_u", "e_ug", "iterations"]
            )
            writer.writeheader()
            writer.writerows(results)
        print("Saved results to", args.csv)


if __name__ == "__main__":
    main()
