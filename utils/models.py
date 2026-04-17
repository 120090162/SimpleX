import numpy as np
import pinocchio
import crocoddyl
import simplex

from line_profiler import profile


# 存robot/cost/actuation模型和中间变量用的
# model对应的data
class DAD_contact(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        self.pinocchio = pinocchio.Model.createData(model.state.pinocchio)
        self.multibody = crocoddyl.DataCollectorMultibody(self.pinocchio)
        self.actuation = model.actuation.createData()
        self.costs = model.costs.createData(self.multibody)
        self.costs.shareMemory(self)


# model
class DAM_contact(crocoddyl.DifferentialActionModelAbstract):
    def __init__(
        self,
        state,
        actuationModel,
        costModel,
        simulator,
        dsimulator,
        dt,
        solver_type=simplex.ConstraintSolverType.CLARABEL,
    ):
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, state, actuationModel.nu, costModel.nr
        )
        self.actuation = actuationModel
        self.costs = costModel
        self.simulator = simulator
        self.dsimulator = dsimulator
        self.dt = dt
        self.solver_type = solver_type

    @profile
    def calc(self, data, x, u=None):
        q, v = x[: self.state.nq], x[-self.state.nv :]
        if u is None:  # 最后那一步N默认u=None
            # [TODO] 这行不一定有用，后续可以为了运行速度删掉
            pinocchio.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)

            self.costs.calc(data.costs, x)
            data.cost = data.costs.cost
        else:
            self.actuation.calc(data.actuation, x, u)  # float base
            tau = data.actuation.tau

            # get next state using Simulator
            self.simulator.step(q, v, tau, self.dt, self.solver_type)
            data.xout[:] = self.simulator.state.anew

            # Computing the cost value and residuals
            pinocchio.forwardKinematics(self.state.pinocchio, data.pinocchio, q, v)
            pinocchio.updateFramePlacements(self.state.pinocchio, data.pinocchio)

            self.costs.calc(data.costs, x, u)
            data.cost = data.costs.cost

    @profile
    def calcDiff(self, data, x, u=None):
        if u is None:
            self.costs.calcDiff(data.costs, x)
        else:
            nq, nv = self.state.nq, self.state.nv
            q, v = x[:nq], x[-nv:]

            # Computing the actuation derivatives
            self.actuation.calcDiff(data.actuation, x, u)
            tau = data.actuation.tau

            # 使用SimulatorDerivatives计算物理导数
            # tau = B @ u
            self.dsimulator.stepDerivatives(self.simulator, q, v, tau, self.dt)

            # ddq_dq = da_dq + da_dtau @ dtau_dq
            # ddq_dv = da_dv + da_dtau @ dtau_dv
            data.Fx[:, :] = (
                np.hstack([self.dsimulator.danew_dq, self.dsimulator.danew_dv])
                + self.dsimulator.danew_dtau @ data.actuation.dtau_dx
            )
            # ddq_du = da_dtau @ dtau_du
            data.Fu[:, :] = self.dsimulator.danew_dtau @ data.actuation.dtau_du
            # Computing the cost derivatives
            self.costs.calcDiff(data.costs, x, u)

    def createData(self):
        data = DAD_contact(self)
        return data


# 构造one shoot problem
def IAM_shoot(
    N,
    state,
    actuation,
    costs,
    simulator,
    dsimulator,
    DT,
    solver_type=simplex.ConstraintSolverType.CLARABEL,
):
    assert N > 1
    # running cost for the first N-1 steps, terminal cost for the last step
    dmodelr = DAM_contact(
        state, actuation, costs[0], simulator, dsimulator, DT, solver_type
    )
    dmodelt = DAM_contact(
        state, actuation, costs[1], simulator, dsimulator, DT, solver_type
    )
    actionmodels = [crocoddyl.IntegratedActionModelEuler(dmodelr, DT)] * N + [
        crocoddyl.IntegratedActionModelEuler(dmodelt, 0.0)
    ]
    return actionmodels


if __name__ == "__main__":
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from utils.params import _ASSETS_DIR, BLACK
    from utils.sim_utils import (
        SimulationArgs,
        add_system_collision_pairs,
        add_material_and_compliance,
        remove_BVH_models,
        setup_joint_constraints,
        setup_simplex_simulator,
    )
    from utils.viz_utils import add_floor

    print("[models.py] build local Pinocchio + SimpleX simulator demo")
    model_path = _ASSETS_DIR / "unitree_go2/go2.xml"
    if not model_path.exists():
        raise FileNotFoundError(f"Local model not found: {model_path}")

    # 1) 导入本地模型
    pin_model = pinocchio.buildModelFromMJCF(model_path.as_posix())
    collision_model = pinocchio.buildGeomFromMJCF(
        pin_model, model_path.as_posix(), pinocchio.GeometryType.COLLISION
    )
    visual_model = pinocchio.buildGeomFromMJCF(
        pin_model, model_path.as_posix(), pinocchio.GeometryType.VISUAL
    )
    q0 = (
        pin_model.referenceConfigurations["home"]
        if "home" in pin_model.referenceConfigurations
        else pinocchio.neutral(pin_model)
    )
    v0 = np.zeros(pin_model.nv)
    print(f"Loaded model: nq={pin_model.nq}, nv={pin_model.nv}")

    # 2) 初始化 SimpleX 相关内容
    add_floor(collision_model, visual_model, BLACK)
    add_system_collision_pairs(pin_model, collision_model, q0, security_margin=1e-2)
    remove_BVH_models(collision_model)
    add_material_and_compliance(collision_model, material="metal", compliance=0.0)
    setup_joint_constraints(
        pin_model, joint_limit=False, joint_friction=0.0, damping=0.0
    )

    sim = simplex.SimulatorX(pin_model, collision_model)
    sim_args = SimulationArgs()
    sim_args.process_args()
    setup_simplex_simulator(sim, sim_args)
    dsim = simplex.SimulatorDerivatives(sim)
    sim.reset()
    solver_type = simplex.ConstraintSolverType.CLARABEL

    # 3) 构造 state/actuation/cost（DAM_contact & DAD_contact 示例）
    state = crocoddyl.StateMultibody(pin_model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    running_cost = crocoddyl.CostModelSum(state, actuation.nu)
    terminal_cost = crocoddyl.CostModelSum(state, actuation.nu)

    xref = np.concatenate([q0, np.zeros(pin_model.nv)])
    x_reg_running = crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelState(state, xref, actuation.nu)
    )
    x_reg_terminal = crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelState(state, xref, actuation.nu)
    )
    u_reg_running = crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    running_cost.addCost("xReg", x_reg_running, 1e-2)
    running_cost.addCost("uReg", u_reg_running, 1e-4)
    terminal_cost.addCost("xReg", x_reg_terminal, 1.0)

    dt = 1e-3
    dam = DAM_contact(state, actuation, running_cost, sim, dsim, dt, solver_type)
    dad = dam.createData()
    x0 = xref.copy()
    u0 = np.zeros(actuation.nu)
    dam.calc(dad, x0, u0)
    dam.calcDiff(dad, x0, u0)

    print(
        f"DAM_contact done: cost={dad.cost:.6f}, Fx={dad.Fx.shape}, Fu={dad.Fu.shape}, xout={dad.xout.shape}"
    )
    print(f"DAD_contact type: {type(dad).__name__}")

    # 4) IAM_shoot 示例
    horizon = 5
    actionmodels = IAM_shoot(
        horizon,
        state,
        actuation,
        [running_cost, terminal_cost],
        sim,
        dsim,
        dt,
        solver_type,
    )
    print(
        f"IAM_shoot done: running={horizon}, total_models={len(actionmodels)}, terminal_dt={actionmodels[-1].dt}"
    )
