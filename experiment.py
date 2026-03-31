"""experiment.py – Comprehensive comparison experiments for PSCom over SAGIN.

Reproduces the numerical-results style of:
  "Energy-Efficient Probabilistic Semantic Communication Over
   Space-Air-Ground Integrated Networks"

Experiments
-----------
  convergence – Objective vs. iteration (proposed algorithm convergence curve)
  latency     – Total energy vs. latency budget T_max (all schemes)
  uav_power   – Total energy vs. UAV power budget P_u (all schemes)
  sat_power   – Total energy vs. satellite transmit power P_s (all schemes)
  num_gts     – Total energy vs. number of ground terminals K (all schemes)
  bar         – Bar chart comparing all schemes at the default configuration
  summary     – 2×2 panel combining latency / uav_power / sat_power / num_gts sweeps

Usage
-----
  # Run all experiments with default config
  python experiment.py

  # Run specific experiments only
  python experiment.py --experiments convergence,latency,bar

  # Override config / output directory and export CSV tables
  python experiment.py --config config/default.yaml --output-dir outputs/paper_figs --csv
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from semantic_comm.config import load_config
from semantic_comm.models import GroundTerminal, SimulationConfig
from semantic_comm.optimizer import PSComOptimizer


# ─── Scheme metadata ─────────────────────────────────────────────────────────

SCHEMES = ["pscom", "no_semantic", "sat_only", "uav_only", "random"]

SCHEME_LABELS: Dict[str, str] = {
    "pscom": "Proposed PSCom",
    "no_semantic": "No Semantic",
    "sat_only": "Satellite Only",
    "uav_only": "UAV Only",
    "random": "Random",
}

SCHEME_STYLES: Dict[str, dict] = {
    "pscom":       dict(color="tab:blue",   marker="o", linestyle="-"),
    "no_semantic": dict(color="tab:orange", marker="s", linestyle="--"),
    "sat_only":    dict(color="tab:green",  marker="^", linestyle="-."),
    "uav_only":    dict(color="tab:red",    marker="D", linestyle=":"),
    "random":      dict(color="tab:purple", marker="x", linestyle=(0, (3, 1, 1, 1))),
}

EXPERIMENTS = ["convergence", "latency", "uav_power", "sat_power", "num_gts", "bar", "summary"]

# Setting convergence_tol below zero guarantees the condition
#   abs(prev_obj - obj) <= convergence_tol
# is never satisfied, so the optimizer runs for the full max_outer_iterations.
_CONVERGENCE_DISABLED = -1.0


# ─── Config helpers ───────────────────────────────────────────────────────────

def _silent_copy(cfg: SimulationConfig) -> SimulationConfig:
    """Deep-copy *cfg* with logging silenced (verbosity=0)."""
    cfg2 = copy.deepcopy(cfg)
    cfg2.verbosity = 0
    return cfg2


def _make_gts_for_k(k: int, template_cfg: SimulationConfig) -> List[GroundTerminal]:
    """
    Build *k* GTs arranged on a circle of radius 150 m, inheriting the
    overhead / data-size parameters of the first GT in *template_cfg*.
    """
    base = template_cfg.scenario.gt_positions[0]
    radius = 150.0
    gts: List[GroundTerminal] = []
    for i in range(k):
        angle = 2 * math.pi * i / k
        gts.append(
            GroundTerminal(
                x=radius * math.cos(angle),
                y=radius * math.sin(angle),
                data_size=base.data_size,
                min_compression=base.min_compression,
                overhead=copy.deepcopy(base.overhead),
            )
        )
    return gts


# ─── Scheme runner ────────────────────────────────────────────────────────────

def _freeze_updates(
    opt: PSComOptimizer,
    *,
    task: bool = False,
    semantic: bool = False,
    geometry: bool = False,
) -> None:
    if task:
        opt._update_task_allocation = lambda: None  # type: ignore[method-assign]
    if semantic:
        opt._update_semantic_ratio = lambda: None  # type: ignore[method-assign]
    if geometry:
        opt._update_altitude_beamwidth = lambda: None  # type: ignore[method-assign]
        opt._update_location = lambda: None  # type: ignore[method-assign]


def _run_scheme(name: str, cfg: SimulationConfig, rng: np.random.Generator) -> float:
    """Run scheme *name* once and return the final total energy objective."""
    opt = PSComOptimizer(cfg)
    K = opt.K

    if name == "pscom":
        final, _ = opt.run()

    elif name == "no_semantic":
        # η=1, no semantic offloading; task/semantic updates frozen
        opt.a_s[:] = 0.0
        opt.a_u[:] = 0.0
        opt.eta[:] = 1.0
        _freeze_updates(opt, task=True, semantic=True)
        final, _ = opt.run()

    elif name == "sat_only":
        # All GTs offloaded to satellite
        opt.a_s[:] = 1.0
        opt.a_u[:] = 0.0
        _freeze_updates(opt, task=True, semantic=True)
        final, _ = opt.run()

    elif name == "uav_only":
        # All GTs offloaded to UAV
        opt.a_s[:] = 0.0
        opt.a_u[:] = 1.0
        _freeze_updates(opt, task=True, semantic=True)
        final, _ = opt.run()

    elif name == "random":
        # Random task allocation, random UAV geometry; no further optimisation
        modes = rng.choice(["none", "sat", "uav"], size=K)
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
        opt.theta = float(rng.uniform(*uav.beamwidth_range))
        opt.hu = float(rng.uniform(*uav.height_range))
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
        final, _ = opt.run()

    else:
        raise ValueError(f"Unknown scheme: {name!r}")

    return final.objective


def _run_scheme_avg(
    name: str,
    cfg: SimulationConfig,
    rng: np.random.Generator,
    n_random_trials: int = 5,
) -> float:
    """
    Run *name* and return the objective.  For the random baseline, average
    *n_random_trials* independent draws to reduce variance.
    """
    if name != "random":
        return _run_scheme(name, cfg, rng)
    return float(np.mean([_run_scheme(name, cfg, rng) for _ in range(n_random_trials)]))


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def _plot_sweep(
    *,
    x_vals: List,
    results: Dict[str, List[float]],
    schemes: List[str],
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    ax: Optional[plt.Axes] = None,
) -> None:
    """Line-plot a parametric sweep.  If *ax* is provided, draw into it."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))
    for s in schemes:
        ax.plot(
            x_vals,
            results[s],
            label=SCHEME_LABELS.get(s, s),
            **SCHEME_STYLES.get(s, {}),
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if standalone:
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  → saved {path}")


def _save_csv(data: Dict, path: Path, x_label: str, schemes: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[x_label] + schemes)
        writer.writeheader()
        for i, xv in enumerate(data["x"]):
            writer.writerow({x_label: xv, **{s: data[s][i] for s in schemes}})
    print(f"  → CSV saved {path}")


# ─── Individual experiments ───────────────────────────────────────────────────

def exp_convergence(cfg: SimulationConfig, output_dir: Path) -> None:
    """
    Plot the proposed algorithm's convergence: objective and individual energy
    components across outer iterations.
    """
    cfg2 = _silent_copy(cfg)
    cfg2.max_outer_iterations = 20
    cfg2.convergence_tol = _CONVERGENCE_DISABLED

    opt = PSComOptimizer(cfg2)
    _, history = opt.run()

    iters = list(range(1, len(history) + 1))
    total_energy        = [s.objective for s in history]
    sat_comp_energy     = [s.e_s  for s in history]
    sat_trans_energy    = [s.e_su for s in history]
    uav_comp_energy     = [s.e_u  for s in history]
    uav_trans_energy    = [s.e_ug for s in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(iters, total_energy, marker="o", linewidth=2, color="tab:blue", label="Total Energy")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Total Energy (J)")
    ax1.set_title("Convergence of Proposed Algorithm")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(iters, sat_comp_energy,  marker="s", label=r"$E_s$ (Sat. Comp.)")
    ax2.plot(iters, sat_trans_energy, marker="^", label=r"$E_{su}$ (Sat. Trans.)")
    ax2.plot(iters, uav_comp_energy,  marker="D", label=r"$E_u$ (UAV Comp.)")
    ax2.plot(iters, uav_trans_energy, marker="x", label=r"$E_{ug}$ (UAV Trans.)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Energy Component (J)")
    ax2.set_title("Energy Components vs. Iteration")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "convergence.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → saved {path}")


def exp_latency_sweep(
    cfg: SimulationConfig,
    output_dir: Path,
    rng: np.random.Generator,
    schemes: List[str] = SCHEMES,
    latency_values: Optional[List[float]] = None,
) -> Dict:
    if latency_values is None:
        latency_values = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00]

    results: Dict[str, List[float]] = {s: [] for s in schemes}
    for T in latency_values:
        cfg2 = _silent_copy(cfg)
        cfg2.scenario.latency_budget = T
        for s in schemes:
            results[s].append(_run_scheme_avg(s, cfg2, rng))
        print(
            f"  T={T:.2f}  "
            + "  ".join(f"{s}={results[s][-1]:.3e}" for s in schemes)
        )

    _plot_sweep(
        x_vals=latency_values,
        results=results,
        schemes=schemes,
        xlabel=r"Latency Budget $T_{\max}$ (s)",
        ylabel="Total Energy (J)",
        title="Total Energy vs. Latency Budget",
        path=output_dir / "sweep_latency.png",
    )
    return {"x": latency_values, **results}


def exp_uav_power_sweep(
    cfg: SimulationConfig,
    output_dir: Path,
    rng: np.random.Generator,
    schemes: List[str] = SCHEMES,
    power_values: Optional[List[float]] = None,
) -> Dict:
    if power_values is None:
        power_values = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0]

    results: Dict[str, List[float]] = {s: [] for s in schemes}
    for P in power_values:
        cfg2 = _silent_copy(cfg)
        cfg2.scenario.uav.power_budget = P
        for s in schemes:
            results[s].append(_run_scheme_avg(s, cfg2, rng))
        print(
            f"  P_u={P:.1f}W  "
            + "  ".join(f"{s}={results[s][-1]:.3e}" for s in schemes)
        )

    _plot_sweep(
        x_vals=power_values,
        results=results,
        schemes=schemes,
        xlabel=r"UAV Power Budget $P_u$ (W)",
        ylabel="Total Energy (J)",
        title="Total Energy vs. UAV Power Budget",
        path=output_dir / "sweep_uav_power.png",
    )
    return {"x": power_values, **results}


def exp_sat_power_sweep(
    cfg: SimulationConfig,
    output_dir: Path,
    rng: np.random.Generator,
    schemes: List[str] = SCHEMES,
    power_values: Optional[List[float]] = None,
) -> Dict:
    if power_values is None:
        power_values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

    results: Dict[str, List[float]] = {s: [] for s in schemes}
    for P in power_values:
        cfg2 = _silent_copy(cfg)
        cfg2.scenario.satellite.power = P
        for s in schemes:
            results[s].append(_run_scheme_avg(s, cfg2, rng))
        print(
            f"  P_s={P:.1f}W  "
            + "  ".join(f"{s}={results[s][-1]:.3e}" for s in schemes)
        )

    _plot_sweep(
        x_vals=power_values,
        results=results,
        schemes=schemes,
        xlabel=r"Satellite Transmit Power $P_s$ (W)",
        ylabel="Total Energy (J)",
        title="Total Energy vs. Satellite Transmit Power",
        path=output_dir / "sweep_sat_power.png",
    )
    return {"x": power_values, **results}


def exp_num_gts_sweep(
    cfg: SimulationConfig,
    output_dir: Path,
    rng: np.random.Generator,
    schemes: List[str] = SCHEMES,
    k_values: Optional[List[int]] = None,
) -> Dict:
    if k_values is None:
        k_values = [2, 3, 4, 5, 6]

    results: Dict[str, List[float]] = {s: [] for s in schemes}
    for K in k_values:
        cfg2 = _silent_copy(cfg)
        cfg2.scenario.gt_positions = _make_gts_for_k(K, cfg)
        for s in schemes:
            results[s].append(_run_scheme_avg(s, cfg2, rng))
        print(
            f"  K={K}  "
            + "  ".join(f"{s}={results[s][-1]:.3e}" for s in schemes)
        )

    _plot_sweep(
        x_vals=k_values,
        results=results,
        schemes=schemes,
        xlabel="Number of Ground Terminals $K$",
        ylabel="Total Energy (J)",
        title="Total Energy vs. Number of Ground Terminals",
        path=output_dir / "sweep_num_gts.png",
    )
    return {"x": k_values, **results}


def exp_bar_comparison(
    cfg: SimulationConfig,
    output_dir: Path,
    rng: np.random.Generator,
    schemes: List[str] = SCHEMES,
) -> Dict[str, float]:
    """Bar chart of per-scheme total energy at the default configuration."""
    cfg2 = _silent_copy(cfg)
    energies: Dict[str, float] = {}
    for s in schemes:
        energies[s] = _run_scheme_avg(s, cfg2, rng)
        print(f"  {SCHEME_LABELS.get(s, s):20s}  {energies[s]:.4e} J")

    labels = [SCHEME_LABELS.get(s, s) for s in schemes]
    values = [energies[s] for s in schemes]
    colors = [SCHEME_STYLES[s]["color"] for s in schemes]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", width=0.55)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{val:.2e}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Total Energy (J)")
    ax.set_title("Scheme Comparison at Default Configuration")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = output_dir / "bar_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → saved {path}")
    return energies


def exp_summary(
    cfg: SimulationConfig,
    output_dir: Path,
    rng: np.random.Generator,
    schemes: List[str] = SCHEMES,
    sweep_data: Optional[Dict[str, Dict]] = None,
) -> None:
    """
    2×2 summary panel combining the four parametric sweeps.

    If *sweep_data* contains pre-computed results for all four sweeps, they are
    reused directly; otherwise, each sweep is re-run.
    """
    if sweep_data is None:
        sweep_data = {}

    # Run missing sweeps silently
    if "latency" not in sweep_data:
        sweep_data["latency"] = exp_latency_sweep(cfg, output_dir, rng, schemes)
    if "uav_power" not in sweep_data:
        sweep_data["uav_power"] = exp_uav_power_sweep(cfg, output_dir, rng, schemes)
    if "sat_power" not in sweep_data:
        sweep_data["sat_power"] = exp_sat_power_sweep(cfg, output_dir, rng, schemes)
    if "num_gts" not in sweep_data:
        sweep_data["num_gts"] = exp_num_gts_sweep(cfg, output_dir, rng, schemes)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panels = [
        (axes[0, 0], sweep_data["latency"],   r"$T_{\max}$ (s)",         "Latency Budget"),
        (axes[0, 1], sweep_data["uav_power"],  r"$P_u$ (W)",              "UAV Power Budget"),
        (axes[1, 0], sweep_data["sat_power"],  r"$P_s$ (W)",              "Satellite Power"),
        (axes[1, 1], sweep_data["num_gts"],    "Number of GTs $K$",       "No. of Ground Terminals"),
    ]
    for ax, data, xlabel, title in panels:
        _plot_sweep(
            x_vals=data["x"],
            results=data,
            schemes=schemes,
            xlabel=xlabel,
            ylabel="Total Energy (J)",
            title=title,
            path=output_dir / "_unused.png",  # not used in embedded mode
            ax=ax,
        )

    fig.suptitle("PSCom Energy Efficiency – Parametric Sweeps", fontsize=13, y=1.01)
    fig.tight_layout()
    path = output_dir / "summary_sweeps.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Paper-aligned comparison experiments for PSCom over SAGIN."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=base_dir / "config" / "default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "outputs" / "experiments",
        help="Directory where figures (and optional CSV) are saved.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=",".join(EXPERIMENTS),
        help=(
            "Comma-separated list of experiments to run.  "
            f"Options: {', '.join(EXPERIMENTS)}"
        ),
    )
    parser.add_argument(
        "--schemes",
        type=str,
        default=",".join(SCHEMES),
        help=(
            "Comma-separated list of schemes to include.  "
            f"Options: {', '.join(SCHEMES)}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (used for the random baseline).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also save sweep results as CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    requested = [e.strip() for e in args.experiments.split(",") if e.strip()]
    unknown_e = [e for e in requested if e not in EXPERIMENTS]
    if unknown_e:
        raise ValueError(f"Unknown experiments: {unknown_e}.  Allowed: {EXPERIMENTS}")

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    unknown_s = [s for s in schemes if s not in SCHEMES]
    if unknown_s:
        raise ValueError(f"Unknown schemes: {unknown_s}.  Allowed: {SCHEMES}")

    sweep_cache: Dict[str, Dict] = {}

    for exp in requested:
        print(f"\n{'='*60}\nExperiment: {exp}\n{'='*60}")

        if exp == "convergence":
            exp_convergence(cfg, args.output_dir)

        elif exp == "latency":
            data = exp_latency_sweep(cfg, args.output_dir, rng, schemes=schemes)
            sweep_cache["latency"] = data
            if args.csv:
                _save_csv(data, args.output_dir / "sweep_latency.csv", "latency_budget", schemes)

        elif exp == "uav_power":
            data = exp_uav_power_sweep(cfg, args.output_dir, rng, schemes=schemes)
            sweep_cache["uav_power"] = data
            if args.csv:
                _save_csv(data, args.output_dir / "sweep_uav_power.csv", "uav_power_W", schemes)

        elif exp == "sat_power":
            data = exp_sat_power_sweep(cfg, args.output_dir, rng, schemes=schemes)
            sweep_cache["sat_power"] = data
            if args.csv:
                _save_csv(data, args.output_dir / "sweep_sat_power.csv", "sat_power_W", schemes)

        elif exp == "num_gts":
            data = exp_num_gts_sweep(cfg, args.output_dir, rng, schemes=schemes)
            sweep_cache["num_gts"] = data
            if args.csv:
                _save_csv(data, args.output_dir / "sweep_num_gts.csv", "num_gts", schemes)

        elif exp == "bar":
            exp_bar_comparison(cfg, args.output_dir, rng, schemes=schemes)

        elif exp == "summary":
            exp_summary(cfg, args.output_dir, rng, schemes=schemes, sweep_data=sweep_cache)

    print(f"\nAll done.  Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
