from __future__ import annotations

import math
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np

from .optimizer import PSComOptimizer
from .config import load_config


class SemanticComEnv(gym.Env):
    """
    A lightweight Gymnasium environment that wraps the PSCom alternating optimizer.

    The action controls per-GT semantic compression ratios together with UAV geometry
    (beamwidth, altitude, and planar location). Remaining variables are updated by the
    optimizer's closed-form / heuristic steps. The reward is the negative total energy
    with latency and coverage penalties, enabling PPO to search for lower-energy
    operating points. The PPO loop is adapted from vwxyzjn/ppo-implementation-details.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg_path: str | None = None,
        config=None,
        episode_length: int = 32,
        reward_scale: float = 10.0,
        latency_penalty: float = 25.0,
        coverage_penalty: float = 5.0,
    ) -> None:
        super().__init__()
        if config is None:
            if cfg_path is None:
                raise ValueError("Either config or cfg_path must be provided.")
            config = load_config(cfg_path)
        self.cfg = config
        self.scenario = config.scenario
        self.K = self.scenario.num_gts
        self.episode_length = episode_length
        self.reward_scale = reward_scale
        self.latency_penalty = latency_penalty
        self.coverage_penalty = coverage_penalty

        self._bbox = self._compute_bbox()
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

        self._step_count = 0
        self.opt: PSComOptimizer | None = None
        self._last_obj = math.inf

    # -------------
    # Gym API
    # -------------
    def reset(
        self, *, seed: int | None = None, options: Dict | None = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._step_count = 0
        self._last_obj = math.inf
        self.opt = PSComOptimizer(self.cfg)
        obs = self._get_observation()
        info: Dict[str, float] = {}
        return obs, info

    def step(self, action: np.ndarray):
        assert self.opt is not None, "reset() must be called before step()."
        self._step_count += 1
        self._apply_action(np.array(action, dtype=float))

        # Run optimizer sub-steps except geometry / semantic ratio (handled by action)
        self.opt._update_task_allocation()
        self.opt._update_computation_capacity()
        self.opt._update_power_bandwidth()

        obj = self.opt._objective()
        latency_violation = self._latency_violation()
        coverage_violation = self._coverage_violation()

        reward = -self.reward_scale * obj
        reward -= self.latency_penalty * latency_violation
        reward -= self.coverage_penalty * coverage_violation

        converged = abs(self._last_obj - obj) <= self.cfg.convergence_tol
        terminated = bool(
            self._step_count >= self.episode_length or converged
        )
        truncated = False

        self._last_obj = obj
        obs = self._get_observation()
        info = {
            "objective": obj,
            "latency_violation": latency_violation,
            "coverage_violation": coverage_violation,
        }
        return obs, reward, terminated, truncated, info

    # -------------
    # Helpers
    # -------------
    def _compute_bbox(self) -> Tuple[float, float, float, float]:
        xs = [gt.x for gt in self.scenario.gt_positions]
        ys = [gt.y for gt in self.scenario.gt_positions]
        pad = max(self.cfg.location_grid_step, 50.0)
        return (min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad)

    def _build_action_space(self) -> gym.Space:
        # K entries for eta, followed by beamwidth, altitude, x, y (all normalized to [-1, 1])
        dim = self.K + 4
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(dim,), dtype=np.float32)

    def _build_observation_space(self) -> gym.Space:
        # Global feats: hu, theta, total_power_ratio, total_bw_ratio, latency_slack, max_latency
        # Per-GT feats: a_s, a_u, eta, fk_ratio, pk_ratio, bk_ratio, dist_norm, rate_norm, latency_ratio
        per_gt_feats = 9
        total_feats = 6 + self.K * per_gt_feats
        high = np.full((total_feats,), np.inf, dtype=np.float32)
        low = -high
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _apply_action(self, action: np.ndarray) -> None:
        assert action.shape[0] == self.K + 4
        # Semantic ratios
        for k in range(self.K):
            eta_raw = float(action[k])
            eta = (eta_raw + 1) / 2  # map [-1,1] -> [0,1]
            eta = self.scenario.gt_positions[k].min_compression + eta * (
                1.0 - self.scenario.gt_positions[k].min_compression
            )
            self.opt.eta[k] = max(self.scenario.gt_positions[k].min_compression, min(1.0, eta))

        # Beamwidth, altitude
        theta_raw, h_raw, x_raw, y_raw = action[self.K :]
        theta_min, theta_max = self.scenario.uav.beamwidth_range
        h_min, h_max = self.scenario.uav.height_range
        self.opt.theta = float(theta_min + (theta_raw + 1) * 0.5 * (theta_max - theta_min))
        self.opt.hu = float(h_min + (h_raw + 1) * 0.5 * (h_max - h_min))

        # Planar location
        x_min, x_max, y_min, y_max = self._bbox
        self.opt.lu = np.array(
            [
                float(x_min + (x_raw + 1) * 0.5 * (x_max - x_min)),
                float(y_min + (y_raw + 1) * 0.5 * (y_max - y_min)),
            ]
        )

    def _latency_violation(self) -> float:
        t_s = self.opt._t_s()
        t_su = self.opt._t_su()
        per_gt = []
        for k in range(self.K):
            per_gt.append(t_s + t_su + self.opt._t_u(k) + self.opt._t_ug(k))
        max_latency = max(per_gt)
        return max(0.0, max_latency - self.scenario.latency_budget)

    def _coverage_violation(self) -> float:
        # Count how many GTs fall outside the current coverage cone
        coverage_radius = self.opt.hu * math.tan(self.opt.theta)
        violations = 0
        for k in range(self.K):
            dist = np.linalg.norm(
                self.opt.lu - np.array([self.scenario.gt_positions[k].x, self.scenario.gt_positions[k].y])
            )
            if dist > coverage_radius:
                violations += 1
        return float(violations)

    def _get_observation(self) -> np.ndarray:
        assert self.opt is not None
        # Latency stats
        t_s = self.opt._t_s()
        t_su = self.opt._t_su()
        per_gt_latency = np.array(
            [t_s + t_su + self.opt._t_u(k) + self.opt._t_ug(k) for k in range(self.K)]
        )
        max_latency = float(np.max(per_gt_latency))
        slack = float(self.scenario.latency_budget - max_latency)

        pk_sum = float(np.sum(self.opt.pk))
        bk_sum = float(np.sum(self.opt.bk))
        uav = self.scenario.uav
        global_feats = [
            float(self.opt.hu / uav.height_range[1]),
            float(
                (self.opt.theta - uav.beamwidth_range[0])
                / max(uav.beamwidth_range[1] - uav.beamwidth_range[0], 1e-9)
            ),
            float(pk_sum / max(uav.power_budget, 1e-9)),
            float(bk_sum / max(uav.bandwidth, 1e-9)),
            slack,
            max_latency,
        ]

        per_gt_feats = []
        for k, gt in enumerate(self.scenario.gt_positions):
            dist = self.opt._distance_uav_gt(k)
            rate = self.opt._r_k(k)
            per_gt_feats.extend(
                [
                    float(self.opt.a_s[k]),
                    float(self.opt.a_u[k]),
                    float(self.opt.eta[k]),
                    float(self.opt.fk[k] / max(uav.computation_capacity, 1e-9)),
                    float(self.opt.pk[k] / max(uav.power_budget, 1e-9)),
                    float(self.opt.bk[k] / max(uav.bandwidth, 1e-9)),
                    float(dist / max(uav.height_range[1], 1e-9)),
                    float(rate / max(uav.bandwidth, 1e-9)),
                    float(per_gt_latency[k] / max(self.scenario.latency_budget, 1e-9)),
                ]
            )

        obs = np.array(global_feats + per_gt_feats, dtype=np.float32)
        return obs


def make_env(cfg_path: str, episode_length: int, reward_scale: float, latency_penalty: float, coverage_penalty: float):
    def thunk():
        return SemanticComEnv(
            cfg_path=cfg_path,
            episode_length=episode_length,
            reward_scale=reward_scale,
            latency_penalty=latency_penalty,
            coverage_penalty=coverage_penalty,
        )

    return thunk
