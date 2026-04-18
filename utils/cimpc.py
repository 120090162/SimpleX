from dataclasses import dataclass
from typing import Any

import crocoddyl
import numpy as np
import pinocchio
import simplex
from scipy.spatial.transform import Rotation as R

from utils.costs import (
    add_symmetric_control_cost,
    make_reference_state,
    make_regulating_costs,
    set_air_time_cost,
    set_foot_slip_clearance_costs,
)
from utils.models import IAM_shoot
from utils.sim_utils import find_quadruped_foot_frame_ids


@dataclass
class TrotCIMPCConfig:
    """Configuration for trot-motion CI-MPC with FDDP."""

    horizon: int = 20
    ocp_dt: float = 2.5e-2
    mpc_hz: float = 40.0
    maxiter: int = 4
    is_feasible: bool = False
    init_reg: float = 0.1

    kp: float | np.ndarray = 40.0
    kd: float | np.ndarray = 1.0

    # Air-time cost activation (5.1.4)
    enable_air_time_cost: bool = True
    air_eps: float = 3e-2
    air_trigger_steps: int = 12  # 12 * 0.025 = 0.3s


def pack_state(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Pack measured generalized coordinates/velocities into Crocoddyl state."""
    return np.concatenate([q, v])


class TrotCIMPC:
    """CI-MPC controller for real-time trot-motion discovery.

    Intended use with a fast simulation loop and a slower
    MPC loop:
    1. MuJoCo provides measured state `(q, v)`.
    2. This controller solves FDDP at low frequency.
    3. Between solves, the latest feed-forward torque is blended with PD.
    """

    def __init__(
        self,
        pin_model: pinocchio.Model,
        simulator: simplex.SimulatorX,
        dsimulator: simplex.SimulatorDerivatives,
        solver_type: simplex.ConstraintSolverType = simplex.ConstraintSolverType.CLARABEL,
        config: TrotCIMPCConfig | None = None,
        q_nominal: np.ndarray | None = None,
    ) -> None:
        self.pin_model = pin_model
        self.state = crocoddyl.StateMultibody(pin_model.copy())
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self.simulator = simulator
        self.dsimulator = dsimulator
        self.solver_type = solver_type
        self.config = config if config is not None else TrotCIMPCConfig()

        self.q_nominal = (
            q_nominal.copy()
            if q_nominal is not None
            else (
                pin_model.referenceConfigurations["home"].copy()
                if "home" in pin_model.referenceConfigurations
                else pinocchio.neutral(pin_model)
            )
        )

        self.foot_frame_ids = find_quadruped_foot_frame_ids(pin_model)
        self._pin_data = pin_model.createData()

        self._swing_count = [0] * len(self.foot_frame_ids)
        self._air_time_active = [False] * len(self.foot_frame_ids)

        self.v_des = 0.0
        self.w_des = 0.0

        self._solver: crocoddyl.SolverFDDP | None = None
        self._last_u_ff = np.zeros(self.actuation.nu)
        self._last_x_des = None

    def set_velocity_command(self, v_des: float, w_des: float) -> None:
        """Set forward and yaw velocity command for trot reference generation."""
        self.v_des = float(v_des)
        self.w_des = float(w_des)

    def reset(self) -> None:
        """Reset warm-start and adaptive cost states."""
        self._solver = None
        self._last_u_ff[:] = 0.0
        self._last_x_des = None
        self._swing_count = [0] * len(self.foot_frame_ids)
        self._air_time_active = [False] * len(self.foot_frame_ids)

    def solve_once(self, x_measured: np.ndarray) -> dict[str, Any]:
        """Solve one FDDP MPC problem using the provided measured state.

        Args:
            x_measured: Current measured state `[q, v]`.

        Returns:
            Solver summary dict.
        """
        self._solve_once(x_measured)
        return {
            "solved": True,
            "u_ff": self._last_u_ff.copy(),
            "air_time_active": self._air_time_active.copy(),
        }

    def compute_command(self, x_measured: np.ndarray) -> np.ndarray:
        """Compute control command using latest feed-forward and PD layer."""
        return self._compute_pd_command(x_measured, self._last_u_ff)

    def update(self, x_measured: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Convenience API: solve once and immediately return command."""
        info = self.solve_once(x_measured)
        u_cmd = self.compute_command(x_measured)
        return u_cmd, info

    def _solve_once(self, x_measured: np.ndarray) -> None:
        problem = self._build_problem(x_measured)
        solver = crocoddyl.SolverFDDP(problem)
        xs_init, us_init = self._build_warm_start(problem, x_measured)
        solver.solve(
            xs_init,
            us_init,
            self.config.maxiter,
            self.config.is_feasible,
            self.config.init_reg,
        )

        self._solver = solver
        self._last_u_ff = np.asarray(solver.us[0]).copy()
        print(
            f"solver us: {[np.asarray(u) for u in solver.us]}"
        )  # Debug print for feed-forward torques
        self._last_x_des = (
            np.asarray(solver.xs[1]).copy()
            if len(solver.xs) > 1
            else np.asarray(solver.xs[0]).copy()
        )

    def _build_problem(self, x_measured: np.ndarray) -> crocoddyl.ShootingProblem:
        x_ref = self._build_reference_state(x_measured)
        running_cost, terminal_cost = make_regulating_costs(
            self.state, self.actuation, x_ref
        )
        add_symmetric_control_cost(
            running_cost, self.state, self.actuation, gait="trot"
        )

        q = x_measured[: self.state.nq]
        v = x_measured[-self.state.nv :]
        heights = self._compute_foot_heights(q, v)
        if self.foot_frame_ids:
            set_foot_slip_clearance_costs(
                running_cost,
                self.state,
                self.actuation,
                self.foot_frame_ids,
                heights,
            )

            if self.config.enable_air_time_cost:
                self._update_air_time_activation(heights)
                for frame_id, active in zip(self.foot_frame_ids, self._air_time_active):
                    set_air_time_cost(
                        running_cost,
                        self.state,
                        self.actuation,
                        frame_id,
                        active=active,
                    )

        action_models = IAM_shoot(
            self.config.horizon,
            self.state,
            self.actuation,
            [running_cost, terminal_cost],
            self.simulator,
            self.dsimulator,
            self.config.ocp_dt,
            self.solver_type,
        )
        return crocoddyl.ShootingProblem(
            x_measured, action_models[:-1], action_models[-1]
        )

    def _build_reference_state(self, x_measured: np.ndarray) -> np.ndarray:
        q_measured = x_measured[: self.state.nq]

        rpy = R.from_quat(q_measured[3:7]).as_euler("xyz")
        distance = (self.v_des + 0.1) * self.config.ocp_dt * self.config.horizon
        yaw_ref = rpy[2] + self.w_des * self.config.ocp_dt * self.config.horizon

        base_position_ref = q_measured[:3].copy()
        base_position_ref[0] += distance * np.cos(rpy[2])
        base_position_ref[1] += distance * np.sin(rpy[2])

        base_rpy_ref = np.array([rpy[0], rpy[1], yaw_ref])
        return make_reference_state(
            self.state,
            self.q_nominal,
            base_position=base_position_ref,
            base_rpy=base_rpy_ref,
        )

    def _build_warm_start(
        self, problem: crocoddyl.ShootingProblem, x_measured: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if self._solver is None:
            xs = [x_measured.copy() for _ in range(problem.T + 1)]
            us = problem.quasiStatic([x_measured.copy() for _ in range(problem.T)])
            return xs, us

        xs = [np.asarray(x).copy() for x in self._solver.xs[1:]]
        xs.append(np.asarray(self._solver.xs[-1]).copy())
        us = [np.asarray(u).copy() for u in self._solver.us[1:]]

        us.append(np.zeros(self.actuation.nu))

        xs[0] = x_measured.copy()
        return xs, us

    def _compute_pd_command(
        self, x_measured: np.ndarray, u_ff: np.ndarray
    ) -> np.ndarray:
        if self._last_x_des is None:
            return u_ff.copy()

        q_meas = x_measured[: self.state.nq]
        v_meas = x_measured[-self.state.nv :]
        q_des = self._last_x_des[: self.state.nq]
        v_des = self._last_x_des[-self.state.nv :]

        q_meas_j = q_meas[-self.actuation.nu :]
        v_meas_j = v_meas[-self.actuation.nu :]
        q_des_j = q_des[-self.actuation.nu :]
        v_des_j = v_des[-self.actuation.nu :]

        kp = self._expand_gain(self.config.kp)
        kd = self._expand_gain(self.config.kd)

        return u_ff + kp * (q_des_j - q_meas_j) + kd * (v_des_j - v_meas_j)

    def _expand_gain(self, gain: float | np.ndarray) -> np.ndarray:
        if np.isscalar(gain):
            return np.full(self.actuation.nu, float(gain))
        gain_vec = np.asarray(gain, dtype=float)
        if gain_vec.shape != (self.actuation.nu,):
            raise ValueError(
                f"Gain shape mismatch: expected {(self.actuation.nu,)}, got {gain_vec.shape}"
            )
        return gain_vec

    def _compute_foot_heights(self, q: np.ndarray, v: np.ndarray) -> list[float]:
        pinocchio.forwardKinematics(self.pin_model, self._pin_data, q, v)
        pinocchio.updateFramePlacements(self.pin_model, self._pin_data)
        return [self._pin_data.oMf[i].translation[2] for i in self.foot_frame_ids]

    def _update_air_time_activation(self, heights: list[float]) -> None:
        for k, h in enumerate(heights):
            if h > self.config.air_eps:
                self._swing_count[k] += 1
            elif not self._air_time_active[k]:
                self._swing_count[k] = 0

            if self._swing_count[k] >= self.config.air_trigger_steps:
                self._air_time_active[k] = True
