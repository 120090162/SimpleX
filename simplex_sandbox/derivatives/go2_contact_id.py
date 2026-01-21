import pinocchio as pin
import numpy as np
from pin_utils import addSystemCollisionPairs
from simulation_utils import (
    addFloor,
    setPhysicsProperties,
)
from simulation_args import SimulationArgs
import simplex
from compute_derivatives import computeStepDerivatives, finiteDifferencesStep
import example_robot_data as erd


class ScriptArgs(SimulationArgs):
    go1: bool = False
    hand_stand: bool = False
    display_target_traj: bool = False
    noptim: int = 100
    step_size: float = 1e-3
    linesearch: bool = False
    debug: bool = False
    cpp: bool = False
    linesearch: bool = False
    save: bool = False
    maxit_linesearch: int = 1000
    min_step_size: float = 1e-14
    finite_differences: bool = False
    eps_fd: float = 1e-6


args = ScriptArgs().parse_args()
allowed_solvers = ["ADMM", "PGS"]
if args.contact_solver not in allowed_solvers:
    print(
        f"Error: unsupported simulator. Avalaible simulators: {allowed_solvers}. Exiting"
    )
    exit(1)
np.random.seed(args.seed)
pin.seed(args.seed)

# ============================================================================
# SCENE CREATION
# ============================================================================
# Create model
if args.go1:
    robot = erd.load("go1")
    model = robot.model
    geom_model = robot.collision_model
    visual_model = robot.visual_model
else:
    rmodel, rgeom_model, _ = pin.buildModelsFromMJCF("./robots/go2/mjcf/go2.xml")
    ff_model = pin.Model()
    ff_id = ff_model.addJoint(
        0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "robot_freeflyer"
    )
    ff_model.addJointFrame(ff_id)
    ff_geom_model = pin.GeometryModel()
    frame_id = ff_model.getFrameId("robot_freeflyer")
    model, geom_model = pin.appendModel(
        ff_model, rmodel, ff_geom_model, rgeom_model, frame_id, pin.SE3.Identity()
    )

    # Add plane in geom_model
    visual_model = geom_model.copy()
addFloor(geom_model, visual_model)
setPhysicsProperties(geom_model, args.material, args.compliance)

# Initial state
# model.lowerPositionLimit = -np.ones((model.nq, 1))
# model.upperPositionLimit = np.ones((model.nq, 1))
# q0 = pin.randomConfiguration(model)
if args.go1:
    q0 = model.referenceConfigurations["standing"]
    if args.hand_stand:
        q0 = np.array(
            [
                0.26,
                0.0,
                0.43,
                0.0,
                0.70710678,
                0.0,
                0.70710678,
                0.0,
                -0.1,
                -1.6,
                0.0,
                -0.1,
                -1.6,
                0.0,
                0.8,
                -1.853,
                0.0,
                0.8,
                -1.853,
            ]
        )
else:
    q0 = pin.neutral(model)
    q0[2] = 0.4
v0 = np.zeros(model.nv)
fext = [pin.Force(np.random.random(6)) for _ in range(model.njoints)]
print("q0 = ", q0)
print("v0 = ", v0)
addSystemCollisionPairs(model, geom_model, q0)


actuation = np.zeros((model.nv, model.nv - 6))
actuation[6:, :] = np.eye(model.nv - 6)

data = model.createData()
geom_data = geom_model.createData()
simulator = simple.Simulator(model, data, geom_model, geom_data)
simulator.admm_constraint_solver_settings.absolute_precision = args.tol
simulator.admm_constraint_solver_settings.relative_precision = args.tol_rel
simulator.admm_constraint_solver_settings.max_iter = args.maxit
dsim = simple.SimulatorDerivatives(simulator)


def computeCost(tau):
    simulator.reset()
    simulator.step(q0, v0, actuation @ tau, args.dt)
    acc = (simulator.vnew - v0) / args.dt
    cost = 0.5 * (np.linalg.norm(acc) ** 2)
    return cost


tau_optim_init = np.zeros(actuation.shape[1])

if args.save:
    costs = []
    grads = []

tau_optim = tau_optim_init.copy()
for n in range(args.noptim):
    simulator.reset()
    simulator.step(q0, v0, actuation @ tau_optim, args.dt)
    if args.finite_differences:
        dvnew_dq, dvnew_dv, dvnew_dtau = finiteDifferencesStep(
            simulator, q0, v0, actuation @ tau_optim, args.dt
        )
        dvnew_dtau = dsim.dvnew_dtau.copy() @ actuation
    if args.cpp:
        dsim.stepDerivatives(simulator, q0, v0, actuation @ tau_optim, args.dt)
        dvnew_dtau = dsim.dvnew_dtau.copy() @ actuation
    else:
        dqnew_dq, dqnew_dv, dqnewdtau, dvnew_dq, dvnew_dv, dvnew_dtau = (
            computeStepDerivatives(
                simulator, q0, v0, actuation @ tau_optim, fext, args.dt
            )
        )
        dvnew_dtau = dvnew_dtau @ actuation
    if args.debug:
        print(f"{dvnew_dtau=}")
        print(f"norm dvnew_dtau {np.linalg.norm(dvnew_dtau)}")
    q = simulator.qnew.copy()
    v = simulator.vnew.copy()
    if args.debug:
        input()

    # Compute cost
    acc = (simulator.vnew - v0) / args.dt
    cost = 0.5 * np.dot(acc, acc)

    # Compute cost gradient
    grad_cost = dvnew_dtau.T @ acc

    if args.save:
        costs.append(cost)
        grads.append(np.linalg.norm(grad_cost))

    if args.linesearch:
        # Gauss newton step
        H_GN = np.dot(dvnew_dtau.T, dvnew_dtau)
        H_GN_inv = np.linalg.inv(H_GN + np.eye(model.nv - 6) * 1e-6)
        dtau = -H_GN_inv @ grad_cost
        expected_improvement = -0.5 * grad_cost @ dtau
        step_size = 1.0
        linesearch_it = 0
        while (
            step_size >= args.min_step_size and linesearch_it <= args.maxit_linesearch
        ):
            tau_optim_next = tau_optim + step_size * dtau
            cost_next = computeCost(tau_optim_next)
            if cost_next < cost - step_size * expected_improvement:
                break
            step_size /= 2
            linesearch_it += 1
        tau_optim = tau_optim_next
    else:
        # Gradient step
        tau_optim -= args.step_size * grad_cost

    print(f"\n---- ITERATION {n} ----")
    print(f"Current cost = {cost}")
    print(f"Current norm grad cost = {np.linalg.norm(grad_cost)}")

if args.save:
    costs = np.array(costs)
    grads = np.array(grads)
    if args.linesearch:
        method = "GN"
    else:
        method = "GD"
    if args.finite_differences:
        np.save(f"./results/fd_{method}_costs_invdyn.npy", costs)
        np.save(f"./results/fd_{method}_grads_invdyn.npy", grads)
    else:
        np.save(f"./results/{method}_costs_invdyn.npy", costs)
        np.save(f"./results/{method}_grads_invdyn.npy", grads)
