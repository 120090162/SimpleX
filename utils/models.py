import numpy as np
import pinocchio
import crocoddyl
import simplex

from line_profiler import profile

from .transforms import *
from .costs import set_lf_cost


# 存robot/cost/actuation模型和中间变量用的
class DAD_contact(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        self.pinocchio = pinocchio.Model.createData(model.state.pinocchio)
        self.multibody = crocoddyl.DataCollectorMultibody(self.pinocchio)
        self.actuation = model.actuation.createData()
        self.costs = model.costs.createData(self.multibody)
        self.costs.shareMemory(self)
        self.Minv = None


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

        # BoxFDDP需要这些属性
        self.u_lb = -np.array(state.pinocchio.effortLimit[-12:])
        self.u_ub = np.array(state.pinocchio.effortLimit[-12:])

    @profile
    def calc(self, data, x, u=None):
        if u is None:  # 最后那一步N默认u=None
            q, v = x[: self.state.nq], x[-self.state.nv :]
            if v[2] < -q[2] / self.dt:
                v[2] = -q[2] / self.dt

            pinocchio.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)
            self.costs.calc(data.costs, x)
            data.cost = data.costs.cost
        else:
            q, v = x[: self.state.nq], x[-self.state.nv :]
            self.actuation.calc(data.actuation, x, u)  # float base
            tau = data.actuation.tau

            pinocchio.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)
            pinocchio.updateFramePlacements(self.state.pinocchio, data.pinocchio)

            self.simulator.step(q, v, tau.copy(), self.dt, self.solver_type)
            data.xout[:] = (self.simulator.state.vnew - v) / self.dt

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
            self.dsimulator.stepDerivatives(self.simulator, q, v, tau.copy(), self.dt)

            da_dq = self.dsimulator.dvnew_dq / self.dt
            da_dv = (self.dsimulator.dvnew_dv - np.eye(nv)) / self.dt
            da_dtau = self.dsimulator.dvnew_dtau / self.dt

            data.Fx[:, :] = np.hstack([da_dq, da_dv]) + da_dtau @ data.actuation.dtau_dx
            data.Fu[:, :] = da_dtau @ data.actuation.dtau_du

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
