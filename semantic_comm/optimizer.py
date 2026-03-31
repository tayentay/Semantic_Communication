from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .models import GroundTerminal, SimulationConfig


@dataclass
class IterationStats:
    objective: float
    e_s: float
    e_su: float
    e_u: float
    e_ug: float


class PSComOptimizer:
    def __init__(self, config: SimulationConfig) -> None:
        self.cfg = config
        self.scenario = config.scenario
        self.K = self.scenario.num_gts
        self.gts: List[GroundTerminal] = self.scenario.gt_positions

        # Decision variables (initialized with reasonable defaults)
        self.a_s = np.ones(self.K)  # semantic compression at satellite
        self.a_u = np.zeros(self.K)  # semantic compression at UAV
        self.eta = np.array(
            [max(gt.min_compression, 0.8) for gt in self.gts], dtype=float
        )
        self.fk = np.full(self.K, self.scenario.uav.computation_capacity / self.K)
        self.bk = np.full(self.K, self.scenario.uav.bandwidth / self.K)
        self.pk = np.full(self.K, self.scenario.uav.power_budget / self.K)
        self.hu = float(self.scenario.initial_uav_height)
        self.theta = float(self.scenario.initial_beamwidth)
        self.lu = np.array(self.scenario.initial_uav_xy, dtype=float)
        self.lam = np.zeros(self.K)

    # --------------------
    # Utility computations
    # --------------------
    def _overhead(self, k: int, eta: float | None = None) -> float:
        return self.gts[k].overhead.value(self.eta[k] if eta is None else eta)

    def _distance_uav_gt(self, k: int) -> float:
        gt = self.gts[k]
        horiz = np.linalg.norm(self.lu - np.array([gt.x, gt.y]))
        return math.sqrt(horiz**2 + self.hu**2)

    def _r_su(self) -> float:
        const = self.scenario.constants
        sat = self.scenario.satellite
        gain = const.beam_gain_satellite * const.wavelength_su / (
            4 * math.pi * self.scenario.distance_su
        )
        h_su = math.sqrt(gain)
        return self.scenario.sat_bandwidth * math.log2(
            1 + h_su**2 * sat.power / (self.scenario.sat_bandwidth * const.noise_psd)
        )

    def _r_k(self, k: int) -> float:
        const = self.scenario.constants
        horiz = np.linalg.norm(self.lu - np.array([self.gts[k].x, self.gts[k].y]))
        # Coverage check
        coverage_radius = self.hu * math.tan(self.theta)
        if horiz > coverage_radius:
            return 1e-9  # effectively infeasible
        gk = const.g0 / (self._distance_uav_gt(k) ** 2)
        numerator = const.uav_mainlobe_gain * gk * self.pk[k]
        denom = (self.theta**2) * self.bk[k] * const.noise_psd
        snr = max(numerator / denom, 1e-12)
        return self.bk[k] * math.log2(1 + snr)

    def _t_s(self) -> float:
        const = self.scenario.constants
        return const.tau * sum(self.a_s[k] * self._overhead(k) for k in range(self.K)) / (
            self.scenario.satellite.computation_capacity
        )

    def _t_u(self, k: int) -> float:
        const = self.scenario.constants
        if self.a_u[k] <= 0:
            return 0.0
        return const.tau * self.a_u[k] * self._overhead(k) / max(self.fk[k], 1e-9)

    def _t_su(self) -> float:
        r_su = self._r_su()
        const = self.scenario.constants
        t_trans = sum(
            (self.a_s[k] * self.eta[k] * self.gts[k].data_size
             + (1 - self.a_s[k]) * self.gts[k].data_size)
            / max(r_su, 1e-9)
            for k in range(self.K)
        )
        t_prop = self.scenario.distance_su / const.speed_of_light
        return t_trans + t_prop

    def _t_ug(self, k: int) -> float:
        num = (self.a_s[k] + self.a_u[k]) * self.eta[k] * self.gts[k].data_size + (
            1 - self.a_s[k] - self.a_u[k]
        ) * self.gts[k].data_size
        return num / max(self._r_k(k), 1e-9)

    def _energies(self) -> Tuple[float, float, float, float]:
        const = self.scenario.constants
        sat = self.scenario.satellite
        t_s = self._t_s()
        e_s = const.kappa * t_s * sat.computation_capacity**3

        t_trans = sum(
            (self.a_s[k] * self.eta[k] * self.gts[k].data_size
             + (1 - self.a_s[k]) * self.gts[k].data_size)
            / max(self._r_su(), 1e-9)
            for k in range(self.K)
        )
        e_su = t_trans * sat.power

        e_u = const.kappa * sum(
            self._t_u(k) * (self.fk[k] ** 3) for k in range(self.K)
        )

        e_ug = sum(self._t_ug(k) * self.pk[k] for k in range(self.K))
        return e_s, e_su, e_u, e_ug

    def _objective(self) -> float:
        return sum(self._energies())

    # --------------------
    # Subproblem updates
    # --------------------
    def _update_task_allocation(self) -> None:
        const = self.scenario.constants
        sat = self.scenario.satellite
        for it in range(self.cfg.max_task_allocation_iters):
            r_su = self._r_su()
            rk = np.array([self._r_k(k) for k in range(self.K)])
            a_s_prev = self.a_s.copy()
            a_u_prev = self.a_u.copy()
            for k in range(self.K):
                ok = self._overhead(k)
                term_comm = self.gts[k].data_size * (1 - self.eta[k])
                a_s_coeff = (
                    const.tau * const.kappa * ok * sat.computation_capacity**2
                    - term_comm * (sat.power / max(r_su, 1e-9) + self.pk[k] / max(rk[k], 1e-9))
                    + self.lam[k]
                    * (
                        const.tau * ok / max(sat.computation_capacity, 1e-9)
                        - term_comm * (1 / max(r_su, 1e-9) + 1 / max(rk[k], 1e-9))
                    )
                )
                a_u_coeff = (
                    const.tau * const.kappa * ok * self.fk[k] ** 2
                    - term_comm * (self.pk[k] / max(rk[k], 1e-9))
                    + self.lam[k]
                    * (
                        const.tau * ok / max(self.fk[k], 1e-9)
                        - term_comm / max(rk[k], 1e-9)
                    )
                )
                # Decision per (32)
                if a_s_coeff >= 0 and a_u_coeff >= 0:
                    self.a_s[k], self.a_u[k] = 0.0, 0.0
                elif (a_s_coeff >= 0 and a_u_coeff < 0) or (a_u_coeff < a_s_coeff < 0):
                    self.a_s[k], self.a_u[k] = 0.0, 1.0
                elif (a_s_coeff < 0 and a_u_coeff >= 0) or (a_s_coeff < a_u_coeff < 0):
                    self.a_s[k], self.a_u[k] = 1.0, 0.0
                else:
                    self.a_s[k], self.a_u[k] = 1.0, 0.0

            # Subgradient update for latency constraint
            t_s = self._t_s()
            t_su = self._t_su()
            for k in range(self.K):
                t_u = self._t_u(k)
                t_ug = self._t_ug(k)
                surplus = t_s + t_su + t_u + t_ug - self.scenario.latency_budget
                step = self.cfg.subgradient_step / math.sqrt(it + 1)
                self.lam[k] = max(self.lam[k] + step * surplus, 0.0)

            if np.allclose(a_s_prev, self.a_s) and np.allclose(a_u_prev, self.a_u):
                break

    def _update_semantic_ratio(self) -> None:
        # Segment selection using midpoint heuristic
        for k in range(self.K):
            best_eta = self.eta[k]
            best_cost = float("inf")
            for d in range(self.gts[k].overhead.segments):
                candidate_eta = self.gts[k].overhead.midpoint(d)
                if candidate_eta < self.gts[k].min_compression:
                    continue
                self.eta[k] = candidate_eta
                cost = self._objective()
                if cost < best_cost:
                    best_cost = cost
                    best_eta = candidate_eta
            self.eta[k] = max(best_eta, self.gts[k].min_compression)

    def _update_computation_capacity(self) -> None:
        const = self.scenario.constants
        t_s = self._t_s()
        t_su = self._t_su()
        for k in range(self.K):
            if self.a_u[k] <= 0:
                self.fk[k] = 0.0
                continue
            remaining = max(
                self.scenario.latency_budget - t_s - t_su - self._t_ug(k), 1e-6
            )
            self.fk[k] = const.tau * self.a_u[k] * self._overhead(k) / remaining
        # Normalize to computation budget
        total_f = self.fk.sum()
        cap = self.scenario.uav.computation_capacity
        if total_f > cap and total_f > 0:
            self.fk *= cap / total_f

    def _update_power_bandwidth(self) -> None:
        uav = self.scenario.uav
        # Equal bandwidth allocation as a starting point
        self.bk = np.full(self.K, uav.bandwidth / self.K)
        t_s = self._t_s()
        t_su = self._t_su()
        for k in range(self.K):
            uk = self.gts[k].data_size * (
                (self.a_s[k] + self.a_u[k]) * self.eta[k]
                + (1 - self.a_s[k] - self.a_u[k])
            )
            denom = max(self.scenario.latency_budget - t_s - t_su - self._t_u(k), 1e-6)
            uk /= denom
            vk = (
                self.scenario.constants.uav_mainlobe_gain
                * self.scenario.constants.g0
                / (self.theta**2 * self.scenario.constants.noise_psd)
            ) / max(self._distance_uav_gt(k) ** 2, 1e-9)
            self.pk[k] = self.bk[k] * (2 ** (uk / max(self.bk[k], 1e-9)) - 1) / max(
                vk, 1e-9
            )

        total_power = self.pk.sum()
        if total_power > uav.power_budget and total_power > 0:
            self.pk *= uav.power_budget / total_power

    def _update_altitude_beamwidth(self) -> None:
        uav = self.scenario.uav
        best_energy = float("inf")
        best_theta = self.theta
        best_h = self.hu
        theta_vals = np.arange(
            uav.beamwidth_range[0],
            uav.beamwidth_range[1] + 1e-9,
            self.cfg.beamwidth_step,
        )
        Lmax = max(
            np.linalg.norm(self.lu - np.array([gt.x, gt.y])) for gt in self.gts
        )
        for theta in theta_vals:
            h_candidate = max(uav.height_range[0], Lmax * math.tan(theta))
            h_candidate = min(h_candidate, uav.height_range[1])
            prev_theta, prev_h = self.theta, self.hu
            self.theta, self.hu = theta, h_candidate
            energy = self._objective()
            if energy < best_energy:
                best_energy = energy
                best_theta = theta
                best_h = h_candidate
            self.theta, self.hu = prev_theta, prev_h
        self.theta, self.hu = best_theta, best_h

    def _update_location(self) -> None:
        # Build feasible grid from GT bounding box
        xs = [gt.x for gt in self.gts]
        ys = [gt.y for gt in self.gts]
        pad = self.cfg.location_grid_step
        x_min, x_max = min(xs) - pad, max(xs) + pad
        y_min, y_max = min(ys) - pad, max(ys) + pad
        grid_x = np.arange(x_min, x_max + 1e-9, self.cfg.location_grid_step)
        grid_y = np.arange(y_min, y_max + 1e-9, self.cfg.location_grid_step)
        best_energy = float("inf")
        best_lu = self.lu
        for x in grid_x:
            for y in grid_y:
                self.lu = np.array([x, y])
                # Coverage check
                coverage_radius = self.hu * math.tan(self.theta)
                if any(
                    np.linalg.norm(self.lu - np.array([gt.x, gt.y])) > coverage_radius
                    for gt in self.gts
                ):
                    continue
                energy = self._objective()
                if energy < best_energy:
                    best_energy = energy
                    best_lu = self.lu.copy()
        self.lu = best_lu

    # --------------------
    # Public API
    # --------------------
    def iterate(self) -> IterationStats:
        self._update_task_allocation()
        self._update_semantic_ratio()
        self._update_computation_capacity()
        self._update_power_bandwidth()
        self._update_altitude_beamwidth()
        self._update_location()
        e_s, e_su, e_u, e_ug = self._energies()
        return IterationStats(
            objective=e_s + e_su + e_u + e_ug, e_s=e_s, e_su=e_su, e_u=e_u, e_ug=e_ug
        )

    def run(self) -> Tuple[IterationStats, List[IterationStats]]:
        history: List[IterationStats] = []
        prev_obj = float("inf")
        for i in range(self.cfg.max_outer_iterations):
            stats = self.iterate()
            history.append(stats)
            if self.cfg.verbosity:
                print(
                    f"[Iter {i+1}] obj={stats.objective:.4e} "
                    f"(e_s={stats.e_s:.3e}, e_su={stats.e_su:.3e}, "
                    f"e_u={stats.e_u:.3e}, e_ug={stats.e_ug:.3e})"
                )
            if abs(prev_obj - stats.objective) <= self.cfg.convergence_tol:
                break
            prev_obj = stats.objective
        return history[-1], history
