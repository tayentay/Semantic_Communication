"""
Energy-efficient probabilistic semantic communication (PSCom) over SAGIN.

This package provides data structures and an alternating optimization solver
that follow the formulation in "Energy-Efficient Probabilistic Semantic
Communication Over Space-Air-Ground Integrated Networks."
"""

from .config import load_config
from .models import (
    GroundTerminal,
    PiecewiseOverhead,
    SystemConstants,
    SatelliteParams,
    UavParams,
    ScenarioParams,
    SimulationConfig,
)
from .optimizer import PSComOptimizer
from .envs import SemanticComEnv

__all__ = [
    "load_config",
    "GroundTerminal",
    "PiecewiseOverhead",
    "SystemConstants",
    "SatelliteParams",
    "UavParams",
    "ScenarioParams",
    "SimulationConfig",
    "PSComOptimizer",
    "SemanticComEnv",
]
