from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import yaml

from .models import (
    GroundTerminal,
    PiecewiseOverhead,
    ScenarioParams,
    SatelliteParams,
    SimulationConfig,
    SystemConstants,
    UavParams,
)


def load_config(path: str | Path) -> SimulationConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    consts = SystemConstants(**raw["constants"])
    sat = SatelliteParams(**raw["satellite"])
    uav = UavParams(**raw["uav"])
    gts: List[GroundTerminal] = []
    for gt in raw["ground_terminals"]:
        overhead = PiecewiseOverhead(
            slopes=gt["overhead"]["slopes"],
            intercepts=gt["overhead"]["intercepts"],
            boundaries=gt["overhead"]["boundaries"],
        )
        gts.append(
            GroundTerminal(
                x=gt["x"],
                y=gt["y"],
                data_size=gt["data_size"],
                min_compression=gt["min_compression"],
                overhead=overhead,
            )
        )

    scenario_kwargs: Dict[str, Any] = {
        "distance_su": raw["distance_su"],
        "sat_bandwidth": raw["sat_bandwidth"],
        "gt_positions": gts,
        "satellite": sat,
        "uav": uav,
        "constants": consts,
        "latency_budget": raw["latency_budget"],
    }
    scenario_kwargs.update(raw.get("scenario_overrides", {}))
    scenario = ScenarioParams(**scenario_kwargs)

    sim_kwargs: Dict[str, Any] = raw.get("simulation", {})
    return SimulationConfig(scenario=scenario, **sim_kwargs)
