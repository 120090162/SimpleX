from tqdm import tqdm
import crocoddyl
import pinocchio
import numpy as np

from .models import DAD_contact, DAM_contact, IAM_contact, IAM_shoot
from .costs import set_la_cost, target_cost
from .transforms import *


# 测试用
def cimpc(total_time, x0, actionmodels, DT=2.5e-2):
    steps = np.floor(total_time / DT).astype(int)
    maxiter = 1
    is_feasible = False
    init_reg = 0.1  # 重要

    traj = []
    inp_traj = []

    problem = crocoddyl.ShootingProblem(x0, actionmodels[:-1], actionmodels[-1])
    solver = crocoddyl.SolverBoxFDDP(problem)
    xs = [x0] * (solver.problem.T + 1)  # list of array即可
    us = solver.problem.quasiStatic([x0] * solver.problem.T)
    # us = [np.zeros(12)]*solver.problem.T
    print("Initialization success !")
    solver.solve(xs, us, maxiter, is_feasible, init_reg)

    print(f"Initial cost: {solver.cost}")

    traj.append(solver.xs[0])
    inp_traj.append(solver.us[0])

    for i in tqdm(range(steps)):

        xi = solver.xs.tolist()[1:]
        xi.append(xi[-1])
        ui = solver.us.tolist()[1:]
        ui.append(np.zeros(12))  # 巨重要
        # ui.append(ui[-1])

        problem = crocoddyl.ShootingProblem(xi[0], actionmodels[:-1], actionmodels[-1])
        solver = crocoddyl.SolverBoxFDDP(problem)

        solver.solve(xi, ui, maxiter, is_feasible, init_reg)

        print(f"Step {i+1}, cost: {solver.cost}")

        traj.append(solver.xs[0])
        inp_traj.append(solver.us[0])

    return (traj, inp_traj)


# 测试用, 加入了air time cost
def cimpc_adaptive(total_time, x0, action_component, air_eps=3e-2):
    N, state, actuation, costs, contact_model, DT, rho = action_component
    costs_r, costs_t = costs

    steps = np.floor(total_time / DT).astype(int)
    maxiter = 1
    is_feasible = False
    init_reg = 0.1

    traj = []
    inp_traj = []

    actionmodels = IAM_shoot(
        N, state, actuation, [costs_r, costs_t], contact_model, DT, rho
    )

    problem = crocoddyl.ShootingProblem(x0, actionmodels[:-1], actionmodels[-1])
    solver = crocoddyl.SolverBoxFDDP(problem)
    xs = [x0] * (solver.problem.T + 1)  # list of array即可
    us = solver.problem.quasiStatic([x0] * solver.problem.T)
    # us = [np.zeros(12)]*solver.problem.T
    print("Initialization success !")
    solver.solve(xs, us, maxiter, is_feasible, init_reg)

    print(f"Initial cost: {solver.cost}")

    traj.append(solver.xs[0])
    inp_traj.append(solver.us[0])

    swing_count = [0] * len(contact_model.contact_ids)
    la_active = [False] * len(contact_model.contact_ids)
    data = pinocchio.Model.createData(state.pinocchio)

    for i in tqdm(range(steps)):

        q, v = traj[-1][: state.nq], traj[-1][-state.nv :]
        pinocchio.forwardKinematics(state.pinocchio, data, q)
        pinocchio.updateFramePlacements(state.pinocchio, data)
        for k, contact_id in enumerate(contact_model.contact_ids):
            oMf = data.oMf[contact_id]
            height = oMf.translation[2]
            if height > air_eps:
                swing_count[k] += 1
            elif not la_active[k]:
                swing_count[k] = 0
            if swing_count[k] >= 12:
                la_active[k] = True
                set_la_cost(costs_r, state, actuation, contact_id, True)

        actionmodels = IAM_shoot(
            N, state, actuation, [costs_r, costs_t], contact_model, DT, rho
        )

        xi = solver.xs.tolist()[1:]
        xi.append(xi[-1])
        ui = solver.us.tolist()[1:]
        ui.append(np.zeros(12))

        problem = crocoddyl.ShootingProblem(xi[0], actionmodels[:-1], actionmodels[-1])
        solver = crocoddyl.SolverBoxFDDP(problem)

        solver.solve(xi, ui, maxiter, is_feasible, init_reg)

        print(f"Step {i+1}, cost: {solver.cost}")

        traj.append(solver.xs[0])
        inp_traj.append(solver.us[0])

        # print(swing_count)

    return (traj, inp_traj)


# 下面是demo, 没有包括air time cost
def walking(total_time, action_component_wo_cost, v, u_static=[], DT=2.5e-2):
    N, state, actuation, contact_model, DT, rho = action_component_wo_cost
    steps = np.floor(total_time / DT).astype(int)
    maxiter = 1
    is_feasible = False
    init_reg = 0.1  # 重要
    pose = "bounding"
    ts = np.linspace(DT, total_time + DT, steps)

    traj = []
    inp_traj = []

    x0 = state.pinocchio.defaultState
    xtarget = x0.copy()
    # 直立
    # quat = rpy_to_quaternion(np.array([0,-3.14/2,0]))
    # xtarget[2] += 0.32
    # xtarget[0] -= 0.2
    # xtarget[3] = quat.x
    # xtarget[4] = quat.y
    # xtarget[5] = quat.z
    # xtarget[6] = quat.w
    # xtarget[8] = 3.14/2
    # xtarget[11] = 3.14/2
    # xtarget[14] = 3.14/2
    # xtarget[15] = 0
    # xtarget[17] = 3.14/2
    # xtarget[18] = 0

    costs_r, costs_t = target_cost(state.pinocchio, state, actuation, xtarget, pose)
    actionmodels = IAM_shoot(
        N, state, actuation, [costs_r, costs_t], contact_model, DT, rho
    )

    problem = crocoddyl.ShootingProblem(x0, actionmodels[:-1], actionmodels[-1])
    solver = crocoddyl.SolverBoxFDDP(problem)
    xs = [x0] * (solver.problem.T + 1)  # list of array即可

    if u_static == []:
        us = solver.problem.quasiStatic([x0] * solver.problem.T)
    else:
        us = [u_static[0] * 0.95] * solver.problem.T
    print("Initialization success !")
    solver.solve(xs, us, maxiter, is_feasible, init_reg)

    print(f"Initial cost: {solver.cost}")

    traj.append(solver.xs[0])
    inp_traj.append(solver.us[0])

    for i in tqdm(range(steps)):

        xi = solver.xs.tolist()[1:]
        xi.append(xi[-1])
        ui = solver.us.tolist()[1:]
        ui.append(np.zeros(12))  # 巨重要
        # ui.append(ui[-1])

        # if ts[i]>=1:
        # xtarget[0] = xi[0][0] + 0.55 # (v+0.1)*DT*20
        # xtarget[0] += (v+0.1)*DT*20
        xtarget[0] = xi[0][0] + (v + 0.1) * DT * 20

        costs_r, costs_t = target_cost(state.pinocchio, state, actuation, xtarget, pose)
        actionmodels = IAM_shoot(
            N, state, actuation, [costs_r, costs_t], contact_model, DT, rho
        )

        problem = crocoddyl.ShootingProblem(xi[0], actionmodels[:-1], actionmodels[-1])
        solver = crocoddyl.SolverBoxFDDP(problem)

        solver.solve(xi, ui, maxiter, is_feasible, init_reg)

        traj.append(solver.xs[0])
        inp_traj.append(solver.us[0])

    print(f"Step {i+1}, cost: {solver.cost}")

    return (traj, inp_traj)


def flip(total_time, action_component_wo_cost, u_static=[], DT=2.5e-2):
    N, state, actuation, contact_model, DT, rho = action_component_wo_cost
    steps = np.floor(total_time / DT).astype(int)
    maxiter = 1
    is_feasible = False
    init_reg = 0.1  # 重要
    pose = "bounding"

    ts = np.linspace(DT, total_time + DT, steps)
    h = lambda t: 0.5 * (1 - (2 * t / total_time - 1) ** 2)

    traj = []
    inp_traj = []

    x0 = state.pinocchio.defaultState
    xtarget = x0.copy()

    costs_r, costs_t = target_cost(state.pinocchio, state, actuation, xtarget, pose)
    actionmodels = IAM_shoot(
        N, state, actuation, [costs_r, costs_t], contact_model, DT, rho
    )

    problem = crocoddyl.ShootingProblem(x0, actionmodels[:-1], actionmodels[-1])
    solver = crocoddyl.SolverBoxFDDP(problem)
    xs = [x0] * (solver.problem.T + 1)  # list of array即可
    if u_static == []:
        us = solver.problem.quasiStatic([x0] * solver.problem.T)
    else:
        us = [u_static[0] * 0.95] * solver.problem.T
    print("Initialization success !")
    solver.solve(xs, us, maxiter, is_feasible, init_reg)

    print(f"Initial cost: {solver.cost}")

    traj.append(solver.xs[0])
    inp_traj.append(solver.us[0])

    for i in tqdm(range(steps)):

        xi = solver.xs.tolist()[1:]
        xi.append(xi[-1])
        ui = solver.us.tolist()[1:]
        ui.append(np.zeros(12))  # 巨重要
        # ui.append(ui[-1])

        # xtarget[2] = x0[2] + h(ts[i])
        # xtarget[0] = -ts[i]/2
        if ts[i] < 1:
            xtarget[2] = x0[2] + 0.5
            quat = rpy_to_quaternion(np.array([0, -np.pi, 0]))
        else:
            xtarget[2] = x0[2]
        quat = rpy_to_quaternion(np.array([0, -0 * np.pi, 0]))
        xtarget[3] = quat.x
        xtarget[4] = quat.y
        xtarget[5] = quat.z
        xtarget[6] = quat.w

        costs_r, costs_t = target_cost(state.pinocchio, state, actuation, xtarget, pose)
        actionmodels = IAM_shoot(
            N, state, actuation, [costs_r, costs_t], contact_model, DT, rho
        )

        problem = crocoddyl.ShootingProblem(xi[0], actionmodels[:-1], actionmodels[-1])
        solver = crocoddyl.SolverBoxFDDP(problem)

        solver.solve(xi, ui, maxiter, is_feasible, init_reg)

        traj.append(solver.xs[0])
        inp_traj.append(solver.us[0])

    print(f"Step {i+1}, cost: {solver.cost}")

    return (traj, inp_traj)


def spining(total_time, action_component_wo_cost, v, u_static=[], DT=2.5e-2):
    N, state, actuation, contact_model, DT, rho = action_component_wo_cost
    steps = np.floor(total_time / DT).astype(int)
    maxiter = 1
    is_feasible = False
    init_reg = 0.1  # 重要
    pose = ""

    traj = []
    inp_traj = []

    x0 = state.pinocchio.defaultState
    xtarget = x0.copy()

    costs_r, costs_t = target_cost(state.pinocchio, state, actuation, xtarget, pose)
    actionmodels = IAM_shoot(
        N, state, actuation, [costs_r, costs_t], contact_model, DT, rho
    )

    problem = crocoddyl.ShootingProblem(x0, actionmodels[:-1], actionmodels[-1])
    solver = crocoddyl.SolverBoxFDDP(problem)
    xs = [x0] * (solver.problem.T + 1)  # list of array即可
    if u_static == []:
        us = solver.problem.quasiStatic([x0] * solver.problem.T)
    else:
        us = [u_static[0] * 0.95] * solver.problem.T
    print("Initialization success !")
    solver.solve(xs, us, maxiter, is_feasible, init_reg)

    print(f"Initial cost: {solver.cost}")

    traj.append(solver.xs[0])
    inp_traj.append(solver.us[0])

    # xtarget[2] += 0.05
    x0rpy = np.array(quat_to_rpy(x0[3:7]))

    for i in tqdm(range(steps)):

        xi = solver.xs.tolist()[1:]
        xi.append(xi[-1])
        ui = solver.us.tolist()[1:]
        ui.append(np.zeros(12))  # 巨重要
        # ui.append(ui[-1])

        x0rpy[2] += v * DT * 20
        quat = rpy_to_quaternion(x0rpy)
        xtarget[3] = quat.x
        xtarget[4] = quat.y
        xtarget[5] = quat.z
        xtarget[6] = quat.w

        costs_r, costs_t = target_cost(state.pinocchio, state, actuation, xtarget, pose)
        actionmodels = IAM_shoot(
            N, state, actuation, [costs_r, costs_t], contact_model, DT, rho
        )

        problem = crocoddyl.ShootingProblem(xi[0], actionmodels[:-1], actionmodels[-1])
        solver = crocoddyl.SolverBoxFDDP(problem)

        solver.solve(xi, ui, maxiter, is_feasible, init_reg)

        traj.append(solver.xs[0])
        inp_traj.append(solver.us[0])

    print(f"Step {i+1}, cost: {solver.cost}")

    return (traj, inp_traj)
