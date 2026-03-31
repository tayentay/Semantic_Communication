from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple
import numpy as np


@dataclass
class PiecewiseOverhead:
    """Piecewise-linear computation overhead Ok(ηk) parameters for a GT."""

    slopes: Sequence[float]
    intercepts: Sequence[float]
    boundaries: Sequence[float]

    def __post_init__(self) -> None:
        if not (len(self.slopes) == len(self.intercepts) == len(self.boundaries)):
            raise ValueError("slopes, intercepts, and boundaries must align in length")
        # Ensure numpy arrays for vector operations
        self.slopes = np.array(self.slopes, dtype=float)
        self.intercepts = np.array(self.intercepts, dtype=float)
        self.boundaries = np.array(self.boundaries, dtype=float)

    @property
    def segments(self) -> int:
        return len(self.slopes)

    def segment_index(self, eta: float) -> int:
        """Return segment index d for the provided compression ratio η (1-based)."""
        for idx, boundary in enumerate(self.boundaries):
            if eta >= boundary:
                return idx
        return self.segments - 1

    def value(self, eta: float) -> float:
        d = self.segment_index(eta)
        return float(self.slopes[d] * eta + self.intercepts[d])

    def midpoint(self, d: int) -> float:
        """Midpoint compression ratio of segment d (0-based)."""
        upper = 1.0 if d == 0 else self.boundaries[d - 1]
        lower = self.boundaries[d]
        return 0.5 * (upper + lower)


@dataclass
class GroundTerminal:
    x: float
    y: float
    data_size: float
    min_compression: float
    overhead: PiecewiseOverhead

    def __post_init__(self) -> None:
        self.x = float(self.x)
        self.y = float(self.y)
        self.data_size = float(self.data_size)
        self.min_compression = float(self.min_compression)


@dataclass
class SystemConstants:
    tau: float
    kappa: float
    speed_of_light: float
    noise_psd: float
    wavelength_su: float
    beam_gain_satellite: float
    g0: float
    uav_mainlobe_gain: float = 2.2846

    def __post_init__(self) -> None:
        self.tau = float(self.tau)
        self.kappa = float(self.kappa)
        self.speed_of_light = float(self.speed_of_light)
        self.noise_psd = float(self.noise_psd)
        self.wavelength_su = float(self.wavelength_su)
        self.beam_gain_satellite = float(self.beam_gain_satellite)
        self.g0 = float(self.g0)
        self.uav_mainlobe_gain = float(self.uav_mainlobe_gain)


@dataclass
class SatelliteParams:
    power: float
    computation_capacity: float

    def __post_init__(self) -> None:
        self.power = float(self.power)
        self.computation_capacity = float(self.computation_capacity)


@dataclass
class UavParams:
    power_budget: float
    bandwidth: float
    computation_capacity: float
    height_range: Tuple[float, float]
    beamwidth_range: Tuple[float, float]

    def __post_init__(self) -> None:
        self.power_budget = float(self.power_budget)
        self.bandwidth = float(self.bandwidth)
        self.computation_capacity = float(self.computation_capacity)
        self.height_range = (float(self.height_range[0]), float(self.height_range[1]))
        self.beamwidth_range = (float(self.beamwidth_range[0]), float(self.beamwidth_range[1]))


@dataclass
class ScenarioParams:
    distance_su: float
    sat_bandwidth: float
    gt_positions: List[GroundTerminal]
    satellite: SatelliteParams
    uav: UavParams
    constants: SystemConstants
    latency_budget: float
    initial_uav_xy: Tuple[float, float] = (0.0, 0.0)
    initial_uav_height: float | None = None
    initial_beamwidth: float | None = None

    def __post_init__(self) -> None:
        self.distance_su = float(self.distance_su)
        self.sat_bandwidth = float(self.sat_bandwidth)
        if self.initial_uav_height is None:
            self.initial_uav_height = 0.5 * (
                self.uav.height_range[0] + self.uav.height_range[1]
            )
        if self.initial_beamwidth is None:
            self.initial_beamwidth = 0.5 * (
                self.uav.beamwidth_range[0] + self.uav.beamwidth_range[1]
            )

    @property
    def num_gts(self) -> int:
        return len(self.gt_positions)


@dataclass
class SimulationConfig:
    scenario: ScenarioParams
    max_outer_iterations: int = 5
    max_task_allocation_iters: int = 20
    max_segment_search_iters: int = 10
    beamwidth_step: float = 0.01
    location_grid_step: float = 50.0
    subgradient_step: float = 0.1
    convergence_tol: float = 1e-4
    verbosity: int = 1
