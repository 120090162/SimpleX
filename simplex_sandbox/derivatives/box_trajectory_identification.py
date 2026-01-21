import simple
import pinocchio as pin
import numpy as np
import hppfcl
import time
from pin_utils import addSystemCollisionPairs
from viz_utils import RED, GREEN, BEIGE
from simulation_utils import (
    SimulationArgs,
    setPhysicsProperties,
    addFloor,
)

from compute_derivatives import computeStepDerivatives, finiteDifferencesStep


class ScriptArgs(SimulationArgs):
    display_target_traj: bool = False
    display_optim: bool = False
    noptim: int = 100
    step_size: float = 1e-3
    linesearch: bool = False
    debug: bool = False
    cpp: bool = False
    torque: bool = False
    stop: float = 1e-5
    save: bool = False
    maxit_linesearch: int = 1000
    min_step_size: float = 1e-14
    finite_differences: bool = False
    eps_fd: float = 1e-6
    ellipsoid: bool = False


args = ScriptArgs().parse_args()
allowed_solvers = ["ADMM", "PGS"]
if args.contact_solver not in allowed_solvers:
    print(
        f"Error: unsupported simulator. Avalaible simulators: {allowed_solvers}. Exiting"
    )
    exit(1)
np.random.seed(args.seed)
pin.seed(args.seed)
np.set_printoptions(suppress=True)

# ============================================================================
# SCENE CREATION
# ============================================================================
# Create model
model = pin.Model()
geom_model = pin.GeometryModel()
visual_model = pin.GeometryModel()

# Add cube freeflyer
mass = 1.0
freeflyer = pin.JointModelFreeFlyer()
M = pin.SE3.Identity()
joint_object = model.addJoint(0, freeflyer, M, "joint_object")
if args.ellipsoid:
    radii = np.array([0.1, 0.2, 0.3])
    model.appendBodyToJoint(
        joint_object,
        pin.Inertia.FromEllipsoid(mass, radii[0], radii[1], radii[2]),
        M,
    )
else:
    cube_size = 0.1
    ball_radius = 0.005
    model.appendBodyToJoint(
        joint_object,
        pin.Inertia.FromBox(mass, cube_size / 2, cube_size / 2, cube_size / 2),
        M,
    )

# Add cube geom_model (8 balls)
if args.ellipsoid:
    ellipsoid_shape = hppfcl.Ellipsoid(radii)
    geom_ellipsoid = pin.GeometryObject(
        "ellipsoid", joint_object, joint_object, pin.SE3.Identity(), ellipsoid_shape
    )
    geom_ellipsoid.meshColor = RED
    visual_model.addGeometryObject(geom_ellipsoid)
    geom_model.addGeometryObject(geom_ellipsoid)
else:
    box_shape = hppfcl.Box(cube_size, cube_size, cube_size)
    geom_box = pin.GeometryObject(
        "box", joint_object, joint_object, pin.SE3.Identity(), box_shape
    )
    geom_box.meshColor = RED
    visual_model.addGeometryObject(geom_box)
    geom_model.addGeometryObject(geom_box)
    extent = cube_size / 2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                name = f"cube_corner_{i}_{j}_{k}"
                corner = np.array(
                    [
                        (-1) ** (i + 1) * extent,
                        (-1) ** (j + 1) * extent,
                        (-1) ** (k + 1) * extent,
                    ]
                )
                ball_shape = hppfcl.Sphere(ball_radius)
                M = pin.SE3.Identity()
                M.translation = corner
                ball_geom = pin.GeometryObject(name, joint_object, M, ball_shape)
                ball_geom.meshColor = GREEN
                geom_model.addGeometryObject(ball_geom)
                visual_model.addGeometryObject(ball_geom)

addFloor(geom_model, visual_model)
setPhysicsProperties(geom_model, args.material, args.compliance)


q0 = pin.neutral(model)
if args.ellipsoid:
    q0[2] = radii[2]
else:
    q0[2] = cube_size / 2 + ball_radius
addSystemCollisionPairs(model, geom_model, q0)

v0_target = 2 * (0.5 - np.random.rand(model.nv)) * 3
v0_target[2] = 0
v0_optim_init = v0_target + 2 * (0.5 - np.random.rand(model.nv)) * 3
v0_optim_init[2] = 0

data = model.createData()
geom_data = geom_model.createData()
simulator = simple.Simulator(model, data, geom_model, geom_data)
simulator.admm_constraint_solver_settings.absolute_precision = args.tol
simulator.admm_constraint_solver_settings.relative_precision = args.tol_rel
simulator.admm_constraint_solver_settings.max_iter = args.maxit
simulator.admm_constraint_solver_settings.mu = args.mu_prox

T = args.horizon

# First let's optimize to find v0 such that final position of trajectory is qtarget.
# Get qtarget from trajectory
simulator.reset()
q = q0.copy()
v = v0_target.copy()
tau = np.zeros(model.nv)
for i in range(T):
    simulator.step(q, v, tau, args.dt)
    q = simulator.qnew.copy()
    v = simulator.vnew.copy()
qtarget = simulator.qnew.copy()
Mtarget = simulator.geom_data.oMg[0]
if args.ellipsoid:
    target_shape = hppfcl.Ellipsoid(radii)
else:
    target_shape = hppfcl.Box(cube_size, cube_size, cube_size)
geom_target = pin.GeometryObject("target", 0, 0, Mtarget, target_shape)
geom_target.meshColor = GREEN
geom_target.meshColor[3] = 0.5
visual_model.addGeometryObject(geom_target)

if args.display:
    from pinocchio.visualize import MeshcatVisualizer
    import meshcat

    # visualize the trajectory
    viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    viewer.delete()
    viewer["/Background"].set_property("top_color", BEIGE[:3].tolist())
    viewer["/Background"].set_property("bottom_color", BEIGE[:3].tolist())
    viewer["/Lights/SpotLight/<object>"].set_property("position", [-10, -10, -10])
    viewer["/Lights/PointLightPositiveX/<object>"].set_property(
        "position", [10, 10, 10]
    )
    vizer: MeshcatVisualizer = MeshcatVisualizer(model, geom_model, visual_model)
    vizer.initViewer(viewer=viewer, open=False, loadModel=True)
    vizer.display(q0)

# Visualize trajectory
fext = [pin.Force(np.random.random(6)) for _ in range(model.njoints)]
if args.display_target_traj:
    i = 0
    while True:
        q = q0.copy()
        if i % 2 == 0:
            v = v0_target.copy()
        else:
            v = v0_optim_init.copy()
        i += 1
        simulator.reset()
        for i in range(T):
            simulator.step(q, v, tau, args.dt)
            q = simulator.qnew.copy()
            v = simulator.vnew.copy()
            if args.display:
                time.sleep(args.dt)
                vizer.display(q)

dsim = simple.SimulatorDerivatives(simulator)
dsim.contact_solver_derivatives.implicit_gradient_solver_type = (
    simple.ImplicitGradientSystemSolver.COD
)


if args.torque:

    def computeCost(tau0):
        q = q0.copy()
        v = v0.copy()
        simulator.step(q, v, tau0, args.dt)
        q = simulator.qnew.copy()
        v = simulator.vnew.copy()
        tau = np.zeros(model.nv)
        simulator.step(q, v, tau, args.dt, T - 1)
        dq = pin.difference(model, qtarget, simulator.qnew.copy())
        cost = 0.5 * (np.linalg.norm(dq) ** 2)
        return cost
else:

    def computeCost(v0):
        q = q0.copy()
        v = v0.copy()
        tau = np.zeros(model.nv)
        simulator.step(q, v, tau, args.dt, T)
        dq = pin.difference(model, qtarget, simulator.qnew.copy())
        cost = 0.5 * (np.linalg.norm(dq) ** 2)
        return cost


# Optimize
if not args.torque:
    v0_optim = v0_optim_init.copy()
else:
    v0 = v0_optim_init.copy()
    tau0_optim = np.zeros(model.nv)

# simulator.step(q0, v0_optim_init, tau0_optim, args.dt)
# dsim.stepDerivatives(simulator, q0, v0_optim_init, tau0_optim, args.dt)
# dqnew_dq, dqnew_dv, dqnewdtau, dvnew_dq, dvnew_dv, dvnew_dtau = computeStepDerivatives(
#     simulator, q0, v0_optim_init, tau0_optim, fext, args.dt
# )

if args.save:
    costs = []
    grads = []

try:
    for n in range(args.noptim):
        # Compute trajectory gradients
        dqdv0 = np.zeros((model.nv, model.nv))
        dvdv0 = np.zeros((model.nv, model.nv))
        q = q0.copy()
        if args.torque:
            tau0 = tau0_optim.copy()
            v = v0.copy()
        else:
            v = v0_optim.copy()
            tau = np.zeros(model.nv)
        simulator.reset()
        for i in range(T):
            if args.debug:
                print(f"\n--------- TIMESTEP {i} ---------")
            if args.torque and i == 0:
                tau = tau0.copy()
            else:
                tau = np.zeros(model.nv)
            simulator.step(q, v, tau, args.dt)
            if args.finite_differences:
                dvnew_dq, dvnew_dv, dvnew_dtau = finiteDifferencesStep(
                    simulator, q, v, tau, args.dt
                )
                dqnew_dq, dqnew_dvnew = pin.dIntegrate(
                    simulator.model, q, args.dt * simulator.vnew
                )
                dqnew_dq += dqnew_dvnew @ (args.dt * dvnew_dq)
                dqnew_dv = dqnew_dvnew @ (args.dt * dvnew_dv)
                dqnewdtau = dqnew_dvnew @ (args.dt * dvnew_dtau)
            else:
                if args.cpp:
                    dsim.stepDerivatives(simulator, q, v, tau, args.dt)
                    dvnew_dv = dsim.dvnew_dv.copy()
                    dvnew_dq = dsim.dvnew_dq.copy()
                    dvnew_dtau = dsim.dvnew_dtau.copy()
                    dqnew_dq, dqnew_dvnew = pin.dIntegrate(
                        simulator.model, q, args.dt * simulator.vnew
                    )
                    dqnew_dq += dqnew_dvnew @ (args.dt * dvnew_dq)
                    dqnew_dv = dqnew_dvnew @ (args.dt * dvnew_dv)
                    dqnewdtau = dqnew_dvnew @ (args.dt * dvnew_dtau)
                else:
                    dqnew_dq, dqnew_dv, dqnewdtau, dvnew_dq, dvnew_dv, dvnew_dtau = (
                        computeStepDerivatives(simulator, q, v, tau, fext, args.dt)
                    )
            if args.debug:
                print(f"{dvnew_dv=}")
                print(f"{dvnew_dq=}")
                print(f"norm dvnew_dv {np.linalg.norm(dvnew_dv)}")
                print(f"norm dvnew_dq {np.linalg.norm(dvnew_dq)}")
            if args.debug:
                # print(f"{dqnew_dv=}")
                # print(f"{dqnew_dq=}")
                print(f"norm dqnew_dv {np.linalg.norm(dqnew_dv)}")
                print(f"norm dqnew_dq {np.linalg.norm(dqnew_dq)}")
            if i == 0:
                if args.torque:
                    dqdtau0 = dqnewdtau
                    dvdtau0 = dvnew_dtau
                else:
                    dqdv0 = dqnew_dv
                    dvdv0 = dvnew_dv
            else:
                if args.torque:
                    dqdtau0_next = np.dot(dqnew_dq, dqdtau0) + np.dot(dqnew_dv, dvdtau0)
                    dvdtau0_next = np.dot(dvnew_dq, dqdtau0) + np.dot(dvnew_dv, dvdtau0)
                    dqdtau0 = dqdtau0_next.copy()
                    dvdtau0 = dvdtau0_next.copy()
                else:
                    dqdv0_next = np.dot(dqnew_dq, dqdv0) + np.dot(dqnew_dv, dvdv0)
                    dvdv0_next = np.dot(dvnew_dq, dqdv0) + np.dot(dvnew_dv, dvdv0)
                    dqdv0 = dqdv0_next.copy()
                    dvdv0 = dvdv0_next.copy()
            if args.debug:
                # print(f"{dqdv0=}")
                # print(f"{dvdv0=}")
                print(f"norm dqdv0 {np.linalg.norm(dqdv0)}")
                print(f"norm dvdv0 {np.linalg.norm(dvdv0)}")
            q = simulator.qnew.copy()
            v = simulator.vnew.copy()
            if args.debug:
                input()

        # Compute cost

        # print(f"{dqdtau0=}")
        dq = pin.difference(model, qtarget, simulator.qnew.copy())
        if args.debug:
            print(f"{dqdv0=}")
            print(f"{dq=}")
        cost = 0.5 * np.dot(dq, dq)

        # Compute cost gradient
        ddifference = pin.dDifference(model, qtarget, simulator.qnew.copy())[1]
        if args.torque:
            grad_cost = (ddifference @ dqdtau0).transpose() @ dq
        else:
            grad_cost = (ddifference @ dqdv0).transpose() @ dq

        if args.save:
            costs.append(cost)
            grads.append(np.linalg.norm(grad_cost))

        if np.linalg.norm(grad_cost) <= args.stop:
            break

        if args.torque:
            if args.linesearch:
                # Gauss newton step
                H_GN = np.dot(dqdtau0.transpose(), dqdtau0)
                H_GN_inv = np.linalg.inv(H_GN + np.eye(model.nv) * 1e-6)
                dtau0 = -H_GN_inv @ grad_cost
                expected_improvement = -0.5 * grad_cost @ dtau0
                step_size = 1.0
                linesearch_it = 0
                while (
                    step_size >= args.min_step_size
                    and linesearch_it <= args.maxit_linesearch
                ):
                    tau0_optim_next = tau0_optim + step_size * dtau0
                    cost_next = computeCost(tau0_optim_next)
                    if cost_next < cost - step_size * expected_improvement:
                        break
                    step_size /= 2
                    linesearch_it += 1
                tau0_optim = tau0_optim_next
            else:
                tau0_optim -= args.step_size * grad_cost
        else:
            if args.linesearch:
                # Gauss newton step
                H_GN = np.dot(dqdv0.transpose(), dqdv0)
                H_GN_inv = np.linalg.inv(H_GN + np.eye(model.nv) * 1e-6)
                dv0 = -H_GN_inv @ grad_cost
                expected_improvement = -0.5 * grad_cost @ dv0
                step_size = 1.0
                linesearch_it = 0
                while (
                    step_size >= args.min_step_size
                    and linesearch_it <= args.maxit_linesearch
                ):
                    v0_optim_next = v0_optim + step_size * dv0
                    cost_next = computeCost(v0_optim_next)
                    if cost_next < cost - step_size * expected_improvement:
                        break
                    step_size /= 2
                    linesearch_it += 1
                v0_optim = v0_optim_next
            else:
                # Gradient step
                v0_optim -= args.step_size * grad_cost

        print(f"\n---- ITERATION {n} ----")
        print(f"Current cost = {cost}")
        print(f"Current norm grad cost = {np.linalg.norm(grad_cost)}")
        if args.display and args.display_optim:
            if n % 50 == 0:
                q = q0.copy()
                if args.torque:
                    tau0 = tau0_optim.copy()
                    v = v0.copy()
                else:
                    v = v0_optim.copy()
                    tau = np.zeros(model.nv)
                simulator.reset()
                for j in range(T):
                    if args.torque and j == 0:
                        tau = tau0.copy()
                    else:
                        tau = np.zeros(model.nv)
                    simulator.step(q, v, tau, args.dt)
                    q = simulator.qnew.copy()
                    v = simulator.vnew.copy()
                    vizer.display(q)
except KeyboardInterrupt:
    pass


if args.save:
    costs = np.array(costs)
    grads = np.array(grads)
    if args.torque:
        optimized_qty = "tau0"
    else:
        optimized_qty = "v0"
    if args.linesearch:
        method = "GN"
    else:
        method = "GD"
    if args.finite_differences:
        np.save(
            f"./sandbox/results/{optimized_qty}_fd_{method}_costs_trajid.npy", costs
        )
        np.save(
            f"./sandbox/results/{optimized_qty}_fd_{method}_grads_trajid.npy", grads
        )
    else:
        np.save(f"./sandbox/results/{optimized_qty}_{method}_costs_trajid.npy", costs)
        np.save(f"./sandbox/results/{optimized_qty}_{method}_grads_trajid.npy", grads)

# display final traj
if args.display:
    while True:
        q = q0.copy()
        if args.torque:
            tau0 = tau0_optim.copy()
            v = v0.copy()
        else:
            v = v0_optim.copy()
            tau = np.zeros(model.nv)
        simulator.reset()
        for i in range(T):
            if args.torque and i == 0:
                tau = tau0.copy()
            else:
                tau = np.zeros(model.nv)
            simulator.step(q, v, tau, args.dt)
            q = simulator.qnew.copy()
            v = simulator.vnew.copy()
            vizer.display(q)
        time.sleep(1.0)
