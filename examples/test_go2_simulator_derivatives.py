import os
import sys

import numpy as np
import pinocchio as pin
import simplex

from absl import app
from absl import flags

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.params import _ASSETS_DIR, _CONFIGS_DIR, BLACK
from utils.viz_utils import add_floor
from utils.sim_utils import (
    SimulationArgs,
    read_sim_config,
    add_system_collision_pairs,
    remove_BVH_models,
    add_material_and_compliance,
    setup_joint_constraints,
    setup_simplex_simulator,
)


_EPS = flags.DEFINE_float("eps", 1e-6, "Finite-difference epsilon.")
_ABS_TOL = flags.DEFINE_float("abs_tol", 5e-3, "Absolute tolerance.")
_REL_TOL = flags.DEFINE_float("rel_tol", 5e-2, "Relative tolerance.")
_SEED = flags.DEFINE_integer("seed", 2568, "Random seed.")
_CONTACT_SOLVER = flags.DEFINE_enum(
    "contact_solver",
    "clarabel",
    ["admm", "pgs", "clarabel"],
    "Constraint solver used in simulator.step.",
)
_DT = flags.DEFINE_float("dt", 1e-3, "Simulation step size.")


def _solver_type_from_name(name: str) -> simplex.ConstraintSolverType:
    if name == "admm":
        return simplex.ConstraintSolverType.ADMM
    if name == "pgs":
        return simplex.ConstraintSolverType.PGS
    if name == "clarabel":
        return simplex.ConstraintSolverType.CLARABEL
    raise ValueError(f"Unsupported solver: {name}")


def _setup_go2_simulator():
    pin_model_path = _ASSETS_DIR / "unitree_go2/mjcf/go2.xml"
    # pin_model_path = _ASSETS_DIR / "unitree_go2/go2.xml"
    if not pin_model_path.exists():
        raise FileNotFoundError(f"Pinocchio model not found: {pin_model_path}")

    model = pin.buildModelFromMJCF(pin_model_path.as_posix())
    collision_model = pin.buildGeomFromMJCF(
        model, pin_model_path.as_posix(), pin.GeometryType.COLLISION
    )
    visual_model = pin.buildGeomFromMJCF(
        model, pin_model_path.as_posix(), pin.GeometryType.VISUAL
    )

    q0 = model.referenceConfigurations["home"].copy()
    add_floor(collision_model, visual_model, BLACK)
    add_system_collision_pairs(model, collision_model, q0, security_margin=1e-2)
    remove_BVH_models(collision_model)

    args = read_sim_config(
        SimulationArgs(),
        (_CONFIGS_DIR / "go2_sim_config.yaml").as_posix(),
        is_show=False,
    )
    args.contact_solver = _CONTACT_SOLVER.value
    args.dt = _DT.value
    add_material_and_compliance(collision_model, args.material, args.compliance)
    setup_joint_constraints(model, args.joint_limit, args.joint_friction, args.damping)

    simulator = simplex.SimulatorX(model, collision_model)
    setup_simplex_simulator(simulator, args)
    dsim = simplex.SimulatorDerivatives(simulator)
    return simulator, dsim, model, q0


def _step_anew(
    simulator: simplex.SimulatorX,
    q: np.ndarray,
    v: np.ndarray,
    tau: np.ndarray,
    dt: float,
    solver_type: simplex.ConstraintSolverType,
) -> np.ndarray:
    simulator.reset()
    simulator.step(q, v, tau, dt, solver_type)
    return simulator.state.anew.copy()


def _finite_differences_anew(
    simulator: simplex.SimulatorX,
    model: pin.Model,
    q: np.ndarray,
    v: np.ndarray,
    tau: np.ndarray,
    dt: float,
    eps: float,
    solver_type: simplex.ConstraintSolverType,
):
    nv = model.nv
    da_dq = np.zeros((nv, nv))
    da_dv = np.zeros((nv, nv))
    da_dtau = np.zeros((nv, nv))

    for i in range(nv):
        dq = np.zeros(nv)
        dq[i] = eps
        q_plus = pin.integrate(model, q, dq)
        q_minus = pin.integrate(model, q, -dq)
        a_plus = _step_anew(simulator, q_plus, v, tau, dt, solver_type)
        a_minus = _step_anew(simulator, q_minus, v, tau, dt, solver_type)
        da_dq[:, i] = (a_plus - a_minus) / (2.0 * eps)

    for i in range(nv):
        dv = np.zeros(nv)
        dv[i] = eps
        a_plus = _step_anew(simulator, q, v + dv, tau, dt, solver_type)
        a_minus = _step_anew(simulator, q, v - dv, tau, dt, solver_type)
        da_dv[:, i] = (a_plus - a_minus) / (2.0 * eps)

    for i in range(nv):
        dtau = np.zeros(nv)
        dtau[i] = eps
        a_plus = _step_anew(simulator, q, v, tau + dtau, dt, solver_type)
        a_minus = _step_anew(simulator, q, v, tau - dtau, dt, solver_type)
        da_dtau[:, i] = (a_plus - a_minus) / (2.0 * eps)

    return da_dq, da_dv, da_dtau


def _matrix_error_stats(name: str, analytic: np.ndarray, finite_diff: np.ndarray):
    print(f"{name}: analytic=\n{analytic}\nfinite_diff=\n{finite_diff}")
    diff = analytic - finite_diff
    abs_err = np.max(np.abs(diff))
    rel_err = np.linalg.norm(diff) / max(1e-12, np.linalg.norm(finite_diff))
    print(
        f"{name}: max_abs_err={abs_err:.3e}, rel_fro_err={rel_err:.3e}, "
        f"||analytic||={np.linalg.norm(analytic):.3e}, ||fd||={np.linalg.norm(finite_diff):.3e}"
    )
    return abs_err, rel_err


def main(argv):
    del argv
    np.random.seed(_SEED.value)
    pin.seed(_SEED.value)

    simulator, dsim, model, q0 = _setup_go2_simulator()
    solver_type = _solver_type_from_name(_CONTACT_SOLVER.value)

    nv = model.nv
    q = q0.copy()
    v = np.zeros(nv)
    tau = np.zeros(nv)
    # if nv > 6:
    #     tau[6:] = 0.5 * np.random.randn(nv - 6)

    simulator.reset()
    simulator.step(q, v, tau, _DT.value, solver_type)
    dsim.stepDerivatives(simulator, q, v, tau, _DT.value)

    fd_dq, fd_dv, fd_dtau = _finite_differences_anew(
        simulator, model, q, v, tau, _DT.value, _EPS.value, solver_type
    )

    results = [
        _matrix_error_stats("danew_dq", dsim.danew_dq.copy(), fd_dq),
        # _matrix_error_stats("danew_dv", dsim.danew_dv.copy(), fd_dv),
        # _matrix_error_stats("danew_dtau", dsim.danew_dtau.copy(), fd_dtau),
    ]

    # for i, (abs_err, rel_err) in enumerate(results):
    #     if abs_err > _ABS_TOL.value and rel_err > _REL_TOL.value:
    #         names = ["danew_dq", "danew_dv", "danew_dtau"]
    #         raise AssertionError(
    #             f"{names[i]} check failed: max_abs_err={abs_err:.3e}, rel_fro_err={rel_err:.3e}, "
    #             f"abs_tol={_ABS_TOL.value:.3e}, rel_tol={_REL_TOL.value:.3e}"
    #         )

    # print("PASS: SimulatorDerivatives danew_* matches finite differences on go2.xml.")


if __name__ == "__main__":
    app.run(main)
