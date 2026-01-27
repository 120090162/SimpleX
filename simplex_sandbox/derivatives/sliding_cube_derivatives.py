import pinocchio as pin
import simplex
import numpy as np
import hppfcl
from pin_utils import addSystemCollisionPairs
from simplex_sandbox.utils.viz_utils import RED, GREEN, BLUE, BLACK, PINK, GREY, BEIGE
from simulation_utils import (
    SimulationArgs,
    setPhysicsProperties,
    addFloor,
    simulateSytem,
    DefaultPolicy,
)
from compute_derivatives import computeNCPDerivatives, computeStepDerivatives


class DerivativesArgs(SimulationArgs):
    flying: bool = False
    sliding: bool = False
    sticking: bool = False
    sliding_and_sticking: bool = False


args = DerivativesArgs().parse_args()
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
cube_size = 1.0
ball_radius = 0.05
mass = 1.0
freeflyer = pin.JointModelFreeFlyer()
M = pin.SE3.Identity()
joint_cube = model.addJoint(0, freeflyer, M, "joint_cube")
model.appendBodyToJoint(
    joint_cube,
    pin.Inertia.FromBox(mass, cube_size / 2, cube_size / 2, cube_size / 2),
    M,
)

# Add cube geom_model (8 balls)
M = pin.SE3.Identity()
box_shape = hppfcl.Box(cube_size, cube_size, cube_size)
geom_box = pin.GeometryObject("box", joint_cube, joint_cube, M, box_shape)
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
            ball_geom = pin.GeometryObject(name, joint_cube, M, ball_shape)
            ball_geom.meshColor = GREEN
            geom_model.addGeometryObject(ball_geom)
            visual_model.addGeometryObject(ball_geom)

addFloor(geom_model, visual_model)
setPhysicsProperties(geom_model, args.material, args.compliance)

# time step
dt = 1e-3

# # ============================================================================
# # ============================================================================
# ============================================================================
# Case of flying cube
# Initial state
if args.flying:
    q0 = pin.neutral(model)
    q0[2] = cube_size / 2 + ball_radius + 1.0
    v0 = np.zeros(model.nv)
    addSystemCollisionPairs(model, geom_model, q0)

    data = model.createData()
    geom_data = geom_model.createData()
    simulator = simple.Simulator(model, data, geom_model, geom_data)
    simulator.admm_constraint_solver_settings.absolute_precision = 1e-10
    simulator.admm_constraint_solver_settings.relative_precision = 1e-12
    simulator.admm_constraint_solver_settings.max_iter = 1000
    simulator.admm_constraint_solver_settings.mu = 1e-6
    tau = np.zeros(model.nv)
    T = args.horizon
    q, v = q0.copy(), v0.copy()
    simulator.step(q, v, tau, dt)

    # ============================================================================
    # testing step derivatives
    ndtheta = model.nv
    dqdtheta = np.zeros((model.nv, ndtheta))
    dvdtheta = np.zeros((model.nv, ndtheta))
    dtaudtheta = np.eye(model.nv)
    fext = [pin.Force(np.random.random(6)) for i in range(model.njoints)]
    dqnew_dq, dqnew_dv, dqnewdtau, dvnew_dq, dvnew_dv, dvnew_dtau = (
        computeStepDerivatives(simulator, q, v, tau, fext, dt)
    )
    sim_deriv = simple.SimulatorDerivatives(simulator)
    sim_deriv.measure_timings = True
    for i in range(1000):
        sim_deriv.stepDerivatives(simulator, q, v, tau, dt)
    assert np.linalg.norm(dvnew_dv - sim_deriv.dvnew_dv) < 1e-4
    assert np.linalg.norm(dvnew_dtau - sim_deriv.dvnew_dtau) < 1e-4
    delta = 1e-6
    dqnew_dq_fd = np.zeros((model.nv, model.nv))
    dvnew_dq_fd = np.zeros((model.nv, model.nv))
    for i in range(model.nv):
        dq = np.zeros(model.nv)
        dq[i] = delta
        q_plus = pin.integrate(model, q, dq)
        q_minus = pin.integrate(model, q, -dq)
        simulator.reset()
        simulator.step(q_minus, v, tau, dt)
        qnew_minus = simulator.qnew.copy()
        vnew_minus = simulator.vnew.copy()
        simulator.reset()
        simulator.step(q_plus, v, tau, dt)
        qnew_plus = simulator.qnew.copy()
        vnew_plus = simulator.vnew.copy()
        dqnew_dq_fd[:, i] = pin.difference(model, qnew_minus, qnew_plus) / (2 * delta)
        dvnew_dq_fd[:, i] = (vnew_plus - vnew_minus) / (2 * delta)

    assert np.linalg.norm(dqnew_dq - dqnew_dq_fd) < 1e-2 * np.linalg.norm(dqnew_dq)
    assert np.linalg.norm(dvnew_dq - dvnew_dq_fd) < 1e-2 * np.linalg.norm(dvnew_dq)

    dqnew_dv_fd = np.zeros((model.nv, model.nv))
    dvnew_dv_fd = np.zeros((model.nv, model.nv))
    for i in range(model.nv):
        dv = np.zeros(model.nv)
        dv[i] = delta
        v_plus = v + dv
        v_minus = v - dv
        simulator.reset()
        simulator.step(q, v_minus, tau, dt)
        qnew_minus = simulator.qnew.copy()
        vnew_minus = simulator.vnew.copy()
        simulator.reset()
        simulator.step(q, v_plus, tau, dt)
        qnew_plus = simulator.qnew.copy()
        vnew_plus = simulator.vnew.copy()
        dqnew_dv_fd[:, i] = pin.difference(model, qnew_minus, qnew_plus) / (2 * delta)
        dvnew_dv_fd[:, i] = (vnew_plus - vnew_minus) / (2 * delta)

    assert np.linalg.norm(dqnew_dv - dqnew_dv_fd) < 1e-2 * np.linalg.norm(dqnew_dv)
    assert np.linalg.norm(dvnew_dv - dvnew_dv_fd) < 1e-2 * np.linalg.norm(dvnew_dv)

    dqnewdtau_fd = np.zeros((model.nv, model.nv))
    dvnew_dtau_fd = np.zeros((model.nv, model.nv))
    for i in range(model.nv):
        dtau = np.zeros(model.nv)
        dtau[i] = delta
        tau_plus = tau + dtau
        tau_minus = tau - dtau
        simulator.reset()
        simulator.step(q, v, tau_minus, dt)
        qnew_minus = simulator.qnew.copy()
        vnew_minus = simulator.vnew.copy()
        simulator.reset()
        simulator.step(q, v, tau_plus, dt)
        qnew_plus = simulator.qnew.copy()
        vnew_plus = simulator.vnew.copy()
        dqnewdtau_fd[:, i] = pin.difference(model, qnew_minus, qnew_plus) / (2 * delta)
        dvnew_dtau_fd[:, i] = (vnew_plus - vnew_minus) / (2 * delta)

    assert np.linalg.norm(dqnewdtau - dqnewdtau_fd) < 1e-2 * np.linalg.norm(dqnewdtau)
    assert np.linalg.norm(dvnew_dtau - dvnew_dtau_fd) < 1e-2 * np.linalg.norm(
        dvnew_dtau
    )

# # ============================================================================
# # ============================================================================
# ============================================================================
# Case of sliding cube
# Initial state
if args.sliding:
    q0 = pin.neutral(model)
    q0[2] = cube_size / 2 + ball_radius
    v0 = np.zeros(model.nv)
    v0[1] = 3.0
    addSystemCollisionPairs(model, geom_model, q0)
    tau = np.zeros(model.nv)
    T = args.horizon

    data = model.createData()
    geom_data = geom_model.createData()
    simulator = simple.Simulator(model, data, geom_model, geom_data)
    simulator.admm_constraint_solver_settings.absolute_precision = 1e-10
    simulator.admm_constraint_solver_settings.relative_precision = 1e-12
    simulator.admm_constraint_solver_settings.max_iter = 1000
    simulator.admm_constraint_solver_settings.mu = 1e-6
    simulator.step(q0, v0, tau, dt)
    lam = simulator.constraint_problem.point_contact_constraint_forces()
    contact_chol = simulator.constraint_problem.constraint_cholesky_decomposition.copy()
    mu_reg = 1e-12
    contact_chol.updateDamping(mu_reg)
    nc = contact_chol.numContacts()
    Del = (
        contact_chol.getDelassusCholeskyExpression().matrix() - np.eye(3 * nc) * mu_reg
    )
    J = simulator.constraint_problem.constraint_cholesky_decomposition.matrix()[
        : 3 * nc, -model.nv :
    ]
    sig = Del @ lam + simulator.constraint_problem.g()
    P = np.zeros((3 * nc, 3 * nc))
    Q = np.zeros((3 * nc, 3 * nc))
    R = np.zeros((3 * nc, 2 * nc))
    RTQR = np.zeros((2 * nc, 2 * nc))
    G_tilde_bis = np.zeros((2 * nc, 2 * nc))
    # building the matrix to invert
    for i in range(nc):
        lam_n = lam[3 * i + 2]
        sig_T = sig[3 * i : 3 * i + 2]
        sig_T_norm = sig_T / np.linalg.norm(sig_T)
        mu = simulator.constraint_problem.cones[i].mu
        H = np.eye(2) - np.outer(sig_T_norm, sig_T_norm)
        P[3 * i : 3 * i + 2, 3 * i : 3 * i + 2] = H * lam_n * mu / np.linalg.norm(sig_T)
        P[3 * i + 2, 3 * i + 2] = 1.0
        Q[3 * i : 3 * i + 2, 3 * i : 3 * i + 2] = np.eye(2)
        Q[3 * i : 3 * i + 2, 3 * i + 2] = mu * sig_T_norm
        R[3 * i : 3 * i + 3, 2 * i] = lam[3 * i : 3 * i + 3] / np.linalg.norm(
            lam[3 * i : 3 * i + 3]
        )
        ez = np.array([0, 0, 1])
        R[3 * i : 3 * i + 3, 2 * i + 1] = np.cross(
            ez, sig[3 * i : 3 * i + 3] / np.linalg.norm(sig[3 * i : 3 * i + 3])
        )
        alpha = ((mu) * (lam_n)) / (np.linalg.norm(lam[3 * i : 3 * i + 3]))
        RTQR[2 * i + 1, 2 * i + 1] = 1.0

    G_tilde = P @ Del + Q
    G_tilde_pinv_numpy = np.linalg.pinv(G_tilde)
    G_tilde_bis = R.T @ G_tilde @ R
    G_tilde_bis_pinv_numpy = np.linalg.pinv(G_tilde_bis)

    RTQR_numpy = R.T @ Q @ R
    error_rtqr = np.linalg.norm(RTQR - RTQR_numpy)

    # random rhs
    errs = []
    rel_errs = []
    errs_lstsq = []
    rel_errs_lstsq = []
    n_samples = 1
    for i in range(n_samples):
        ndtheta = 1
        dGlamgdtheta = np.random.rand(3 * nc, ndtheta)
        rhs = -P @ (dGlamgdtheta)
        x_numpy = G_tilde_pinv_numpy @ rhs
        rhs_bis = -R.T @ P @ (dGlamgdtheta)
        x_numpy_bis = R @ (G_tilde_bis_pinv_numpy @ rhs_bis)
        x_lstsq = np.linalg.lstsq(G_tilde, rhs, rcond=None)[0]
        err = np.linalg.norm(x_numpy_bis - x_numpy)
        rel_err = err / np.linalg.norm(x_numpy_bis)
        err_lstsq = np.linalg.norm(x_numpy_bis - x_lstsq)
        rel_err_lstsq = err_lstsq / np.linalg.norm(x_numpy_bis)
        errs.append(err)
        rel_errs.append(rel_err)
        errs_lstsq.append(err_lstsq)
        rel_errs_lstsq.append(rel_err_lstsq)
        x_sim = computeNCPDerivatives(simulator, dGlamgdtheta)
        err = np.linalg.norm(x_sim - x_numpy)
        rel_err = err / np.linalg.norm(x_numpy)
    sol_err = np.max(errs)
    sol_rel_err = np.max(rel_errs)
    sol_lstsq_err = np.max(err_lstsq)
    sol_lstsq_rel_err = np.max(rel_err_lstsq)
    assert np.linalg.norm(x_sim - x_numpy) < 1e-8

    # ============================================================================
    # testing NCP grads against finite differences
    data_ncp = model.createData()
    geom_data_ncp = geom_model.createData()
    simulator_ncp = simple.Simulator(model, data_ncp, geom_model, geom_data_ncp)
    simulator_ncp.admm_constraint_solver_settings.absolute_precision = 1e-10
    simulator_ncp.admm_constraint_solver_settings.relative_precision = 1e-12
    simulator_ncp.admm_constraint_solver_settings.max_iter = 1000
    simulator_ncp.admm_constraint_solver_settings.mu = 1e-6
    simulator_ncp.step(q0, v0, tau, dt)
    ncp_derivatives = simple.ContactSolverDerivatives(simulator_ncp.constraint_problem)
    ncp_derivatives.measure_timings = True
    g = simulator.constraint_problem.g()
    R = simulator.constraint_problem.compliances()
    cones = simulator.constraint_problem.cones
    ndtheta = model.nv
    dGlamgdtheta = J
    dlam_dtheta = computeNCPDerivatives(simulator, dGlamgdtheta)
    for i in range(1):
        ncp_derivatives.jvp(dGlamgdtheta)
    dlam_dtheta_cpp = ncp_derivatives.dlam_dtheta()
    gradL = dlam_dtheta.T @ lam
    dlam_dtheta_fd = np.zeros((3 * nc, ndtheta))
    # finite differences wrt g
    for i in range(ndtheta):
        dv = np.zeros(model.nv)
        delta = 1e-6
        dv[i] = delta
        dg = J @ dv
        contact_solver_plus = pin.ADMMConstraintSolver(
            3 * nc,
            simulator.admm_constraint_solver_settings.mu,
            simulator.admm_constraint_solver_settings.tau,
            simulator.admm_constraint_solver_settings.rho_power,
            simulator.admm_constraint_solver_settings.rho_power_factor,
            simulator.admm_constraint_solver_settings.ratio_primal_dual,
            simulator.admm_constraint_solver_settings.max_it_largest_eigenvalue_solver,
        )
        has_converged = contact_solver_plus.solve(
            pin.DelassusOperatorDense(Del),
            g + dg,
            cones,
            R,
            primal_solution=np.zeros(3 * nc),
        )
        lam_fd_plus = contact_solver_plus.getPrimalSolution().copy()

        contact_solver_minus = pin.ADMMConstraintSolver(
            3 * nc,
            simulator.admm_constraint_solver_settings.mu,
            simulator.admm_constraint_solver_settings.tau,
            simulator.admm_constraint_solver_settings.rho_power,
            simulator.admm_constraint_solver_settings.rho_power_factor,
            simulator.admm_constraint_solver_settings.ratio_primal_dual,
            simulator.admm_constraint_solver_settings.max_it_largest_eigenvalue_solver,
        )
        has_converged = contact_solver_minus.solve(
            pin.DelassusOperatorDense(Del),
            g - dg,
            cones,
            R,
            primal_solution=np.zeros(3 * nc),
        )
        lam_fd_minus = contact_solver_minus.getPrimalSolution().copy()
        dlam_dtheta_fd[:, i] = (lam_fd_plus - lam_fd_minus) / (delta * 2)

    assert np.linalg.norm(dlam_dtheta_fd - dlam_dtheta) < 1e-2 * np.linalg.norm(
        dlam_dtheta
    )
    assert np.linalg.norm(dlam_dtheta_fd - dlam_dtheta_cpp) < 1e-2 * np.linalg.norm(
        dlam_dtheta_cpp
    )

    # ============================================================================
    # testing step derivatives
    ndtheta = model.nv
    q = q0.copy()
    v = v0.copy()
    dqdtheta = np.zeros((model.nv, ndtheta))
    dvdtheta = np.zeros((model.nv, ndtheta))
    dtaudtheta = np.eye(model.nv)
    fext = [pin.Force(np.random.random(6)) for i in range(model.njoints)]
    dqnew_dq, dqnew_dv, dqnewdtau, dvnew_dq, dvnew_dv, dvnew_dtau = (
        computeStepDerivatives(simulator, q, v, tau, fext, dt)
    )
    sim_deriv = simple.SimulatorDerivatives(simulator)
    sim_deriv.measure_timings = True
    for i in range(1000):
        sim_deriv.stepDerivatives(simulator, q, v, tau, dt)
    assert np.linalg.norm(dvnew_dv - sim_deriv.dvnew_dv) < 1e-4
    assert np.linalg.norm(dvnew_dtau - sim_deriv.dvnew_dtau) < 1e-4
    delta = 1e-6
    dqnew_dq_fd = np.zeros((model.nv, model.nv))
    dvnew_dq_fd = np.zeros((model.nv, model.nv))
    for i in range(model.nv):
        dq = np.zeros(model.nv)
        dq[i] = delta
        q_plus = pin.integrate(model, q, dq)
        q_minus = pin.integrate(model, q, -dq)
        simulator.reset()
        simulator.step(q_minus, v, tau, dt)
        qnew_minus = simulator.qnew.copy()
        vnew_minus = simulator.vnew.copy()
        simulator.reset()
        simulator.step(q_plus, v, tau, dt)
        qnew_plus = simulator.qnew.copy()
        vnew_plus = simulator.vnew.copy()
        dqnew_dq_fd[:, i] = pin.difference(model, qnew_minus, qnew_plus) / (2 * delta)
        dvnew_dq_fd[:, i] = (vnew_plus - vnew_minus) / (2 * delta)

    dqnew_dv_fd = np.zeros((model.nv, model.nv))
    dvnew_dv_fd = np.zeros((model.nv, model.nv))
    for i in range(model.nv):
        dv = np.zeros(model.nv)
        dv[i] = delta
        v_plus = v + dv
        v_minus = v - dv
        simulator.reset()
        simulator.step(q, v_minus, tau, dt)
        qnew_minus = simulator.qnew.copy()
        vnew_minus = simulator.vnew.copy()
        simulator.reset()
        simulator.step(q, v_plus, tau, dt)
        qnew_plus = simulator.qnew.copy()
        vnew_plus = simulator.vnew.copy()
        dqnew_dv_fd[:, i] = pin.difference(model, qnew_minus, qnew_plus) / (2 * delta)
        dvnew_dv_fd[:, i] = (vnew_plus - vnew_minus) / (2 * delta)

    assert np.linalg.norm(dqnew_dv - dqnew_dv_fd) < 1e-4
    assert np.linalg.norm(dvnew_dv - dvnew_dv_fd) < 1e-4

    dqnewdtau_fd = np.zeros((model.nv, model.nv))
    dvnew_dtau_fd = np.zeros((model.nv, model.nv))
    for i in range(model.nv):
        dtau = np.zeros(model.nv)
        dtau[i] = delta
        tau_plus = tau + dtau
        tau_minus = tau - dtau
        simulator.reset()
        simulator.step(q, v, tau_minus, dt)
        qnew_minus = simulator.qnew.copy()
        vnew_minus = simulator.vnew.copy()
        simulator.reset()
        simulator.step(q, v, tau_plus, dt)
        qnew_plus = simulator.qnew.copy()
        vnew_plus = simulator.vnew.copy()
        dqnewdtau_fd[:, i] = pin.difference(model, qnew_minus, qnew_plus) / (2 * delta)
        dvnew_dtau_fd[:, i] = (vnew_plus - vnew_minus) / (2 * delta)

    assert np.linalg.norm(dvnew_dtau - dvnew_dtau_fd) < 1e-4
    assert np.linalg.norm(dqnewdtau - dqnewdtau_fd) < 1e-4

# # ============================================================================
# # ============================================================================
# ============================================================================
# Case of non moving cube
if args.sticking:
    q0 = pin.neutral(model)
    q0[2] = cube_size / 2 + ball_radius
    v0 = np.zeros(model.nv)
    addSystemCollisionPairs(model, geom_model, q0)
    data = model.createData()
    geom_data = geom_model.createData()
    simulator = simple.Simulator(model, data, geom_model, geom_data)
    simulator.admm_constraint_solver_settings.absolute_precision = 1e-10
    simulator.admm_constraint_solver_settings.relative_precision = 1e-12
    simulator.admm_constraint_solver_settings.max_iter = 1000
    simulator.admm_constraint_solver_settings.mu = 1e-6
    v = np.zeros(model.nv)
    q = q0.copy()
    tau = np.zeros(model.nv)
    simulator.reset()
    simulator.step(q, v, tau, dt)
    lam = simulator.constraint_problem.point_contact_constraint_forces()
    contact_chol = simulator.constraint_problem.constraint_cholesky_decomposition
    mu = 1e-12
    contact_chol.updateDamping(mu)
    nc = contact_chol.numContacts()
    Del = contact_chol.getDelassusCholeskyExpression().matrix() - np.eye(3 * nc) * mu
    J = simulator.constraint_problem.constraint_cholesky_decomposition.matrix()[
        : 3 * nc, -model.nv :
    ]
    sig = Del @ lam + simulator.constraint_problem.g()

    # random rhs
    ndtheta = 1
    dGlamgdtheta = np.random.rand(3 * nc, ndtheta)
    rhs = -(dGlamgdtheta)
    x_numpy = np.linalg.pinv(Del) @ rhs
    x_sim = computeNCPDerivatives(simulator, dGlamgdtheta)
    assert np.linalg.norm(x_sim - x_numpy) < 1e-8

    # ============================================================================
    # testing NCP grads against finite differences
    g = simulator.constraint_problem.g()
    R = simulator.constraint_problem.compliances()
    cones = simulator.constraint_problem.cones
    ndtheta = model.nv
    dGlamgdtheta = J
    dlam_dtheta = computeNCPDerivatives(simulator, dGlamgdtheta)
    ncp_derivatives = simple.ContactSolverDerivatives(simulator.constraint_problem)
    ncp_derivatives.measure_timings = True
    for i in range(1000):
        ncp_derivatives.jvp(dGlamgdtheta)
    dlam_dtheta_cpp = ncp_derivatives.dlam_dtheta()
    gradL = dlam_dtheta.T @ lam
    dlam_dtheta_fd = np.zeros((3 * nc, ndtheta))
    # finite differences wrt g
    for i in range(ndtheta):
        dv = np.zeros(model.nv)
        delta = 1e-6
        dv[i] = delta
        dg = J @ dv
        contact_solver_plus = pin.ADMMConstraintSolver(
            3 * nc,
            simulator.admm_constraint_solver_settings.mu,
            simulator.admm_constraint_solver_settings.tau,
            simulator.admm_constraint_solver_settings.rho_power,
            simulator.admm_constraint_solver_settings.rho_power_factor,
            simulator.admm_constraint_solver_settings.ratio_primal_dual,
            simulator.admm_constraint_solver_settings.max_it_largest_eigenvalue_solver,
        )
        has_converged = contact_solver_plus.solve(
            pin.DelassusOperatorDense(Del),
            g + dg,
            cones,
            R,
            primal_solution=np.zeros(3 * nc),
        )
        lam_fd_plus = contact_solver_plus.getPrimalSolution().copy()

        contact_solver_minus = pin.ADMMConstraintSolver(
            3 * nc,
            simulator.admm_constraint_solver_settings.mu,
            simulator.admm_constraint_solver_settings.tau,
            simulator.admm_constraint_solver_settings.rho_power,
            simulator.admm_constraint_solver_settings.rho_power_factor,
            simulator.admm_constraint_solver_settings.ratio_primal_dual,
            simulator.admm_constraint_solver_settings.max_it_largest_eigenvalue_solver,
        )
        has_converged = contact_solver_minus.solve(
            pin.DelassusOperatorDense(Del),
            g - dg,
            cones,
            R,
            primal_solution=np.zeros(3 * nc),
        )
        lam_fd_minus = contact_solver_minus.getPrimalSolution().copy()
        dlam_dtheta_fd[:, i] = (lam_fd_plus - lam_fd_minus) / (delta * 2)

    assert np.linalg.norm(dlam_dtheta_fd - dlam_dtheta) < 1e-2 * np.linalg.norm(
        dlam_dtheta
    )
    assert np.linalg.norm(dlam_dtheta_fd - dlam_dtheta_cpp) < 1e-2 * np.linalg.norm(
        dlam_dtheta_cpp
    )

    # ============================================================================
    # testing step derivatives
    ndtheta = model.nv
    dqdtheta = np.zeros((model.nv, ndtheta))
    dvdtheta = np.zeros((model.nv, ndtheta))
    dtaudtheta = np.eye(model.nv)
    fext = [pin.Force(np.random.random(6)) for i in range(model.njoints)]
    dqnew_dq, dqnew_dv, dqnewdtau, dvnew_dq, dvnew_dv, dvnew_dtau = (
        computeStepDerivatives(simulator, q, v, tau, fext, dt)
    )
    sim_deriv = simple.SimulatorDerivatives(simulator)
    sim_deriv.measure_timings = True
    for i in range(1000):
        sim_deriv.stepDerivatives(simulator, q, v, tau, dt)
    assert np.linalg.norm(dvnew_dv - sim_deriv.dvnew_dv) < 1e-4
    assert np.linalg.norm(dvnew_dtau - sim_deriv.dvnew_dtau) < 1e-4
    delta = 1e-6
    dqnew_dq_fd = np.zeros((model.nv, model.nv))
    dvnew_dq_fd = np.zeros((model.nv, model.nv))
    for i in range(model.nv):
        dq = np.zeros(model.nv)
        dq[i] = delta
        q_plus = pin.integrate(model, q, dq)
        q_minus = pin.integrate(model, q, -dq)
        simulator.reset()
        simulator.step(q_minus, v, tau, dt)
        qnew_minus = simulator.qnew.copy()
        vnew_minus = simulator.vnew.copy()
        simulator.reset()
        simulator.step(q_plus, v, tau, dt)
        qnew_plus = simulator.qnew.copy()
        vnew_plus = simulator.vnew.copy()
        dqnew_dq_fd[:, i] = pin.difference(model, qnew_minus, qnew_plus) / (2 * delta)
        dvnew_dq_fd[:, i] = (vnew_plus - vnew_minus) / (2 * delta)

    dqnew_dv_fd = np.zeros((model.nv, model.nv))
    dvnew_dv_fd = np.zeros((model.nv, model.nv))
    for i in range(model.nv):
        dv = np.zeros(model.nv)
        dv[i] = delta
        v_plus = v + dv
        v_minus = v - dv
        simulator.reset()
        simulator.step(q, v_minus, tau, dt)
        qnew_minus = simulator.qnew.copy()
        vnew_minus = simulator.vnew.copy()
        simulator.reset()
        simulator.step(q, v_plus, tau, dt)
        qnew_plus = simulator.qnew.copy()
        vnew_plus = simulator.vnew.copy()
        dqnew_dv_fd[:, i] = pin.difference(model, qnew_minus, qnew_plus) / (2 * delta)
        dvnew_dv_fd[:, i] = (vnew_plus - vnew_minus) / (2 * delta)

    assert np.linalg.norm(dqnew_dv - dqnew_dv_fd) < 1e-4
    assert np.linalg.norm(dvnew_dv - dvnew_dv_fd) < 1e-4

    dqnewdtau_fd = np.zeros((model.nv, model.nv))
    dvnew_dtau_fd = np.zeros((model.nv, model.nv))
    for i in range(model.nv):
        dtau = np.zeros(model.nv)
        dtau[i] = delta
        tau_plus = tau + dtau
        tau_minus = tau - dtau
        simulator.reset()
        simulator.step(q, v, tau_minus, dt)
        qnew_minus = simulator.qnew.copy()
        vnew_minus = simulator.vnew.copy()
        simulator.reset()
        simulator.step(q, v, tau_plus, dt)
        qnew_plus = simulator.qnew.copy()
        vnew_plus = simulator.vnew.copy()
        dqnewdtau_fd[:, i] = pin.difference(model, qnew_minus, qnew_plus) / (2 * delta)
        dvnew_dtau_fd[:, i] = (vnew_plus - vnew_minus) / (2 * delta)

    assert np.linalg.norm(dqnewdtau - dqnewdtau_fd) < 1e-4
    assert np.linalg.norm(dvnew_dtau - dvnew_dtau_fd) < 1e-4

# # ============================================================================
# # ============================================================================
# # ============================================================================
# # Case mixing sliding and sticking contacts
if args.sliding_and_sticking:
    # Add cube freeflyer
    cube_size = 1.0
    ball_radius = 0.05
    mass = 1.0
    freeflyer = pin.JointModelFreeFlyer()
    M = pin.SE3.Identity()
    joint_cube = model.addJoint(0, freeflyer, M, "joint_cube2")
    model.appendBodyToJoint(
        joint_cube,
        pin.Inertia.FromBox(mass, cube_size / 2, cube_size / 2, cube_size / 2),
        M,
    )

    # Add cube geom_model (8 balls)
    M = pin.SE3.Identity()
    box_shape = hppfcl.Box(cube_size, cube_size, cube_size)
    geom_box = pin.GeometryObject("box", joint_cube, joint_cube, M, box_shape)
    geom_box.meshColor = RED
    visual_model.addGeometryObject(geom_box)
    geom_model.addGeometryObject(geom_box)
    extent = cube_size / 2
    for i in range(2):
        for j in range(2):
            for k in range(2):
                name = f"cube_corner_{i}_{j}_{k}_2"
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
                ball_geom = pin.GeometryObject(name, joint_cube, M, ball_shape)
                ball_geom.meshColor = GREEN
                geom_model.addGeometryObject(ball_geom)
                visual_model.addGeometryObject(ball_geom)

    setPhysicsProperties(geom_model, args.material, args.compliance)

    # ============================================================================
    # Case of sliding and sticking cubes
    # Initial state
    q0 = pin.neutral(model)
    q0[2] = cube_size / 2 + ball_radius
    q0[7] = 2 * (cube_size / 2 + ball_radius)
    q0[9] = cube_size / 2 + ball_radius
    v0 = np.zeros(model.nv)
    v0[1] = 3.0

    print("q0 = ", q0)
    print("v0 = ", v0)
    addSystemCollisionPairs(model, geom_model, q0)

    data = model.createData()
    geom_data = geom_model.createData()
    simulator = simple.Simulator(model, data, geom_model, geom_data)
    simulator.admm_constraint_solver_settings.absolute_precision = 1e-10
    simulator.admm_constraint_solver_settings.relative_precision = 1e-12
    simulator.admm_constraint_solver_settings.max_iter = 1000
    simulator.admm_constraint_solver_settings.mu = 1e-6
    tau = np.zeros(model.nv)
    T = args.horizon
    q, v = q0.copy(), v0.copy()

    simulator.step(q, v, tau, dt)
    qnew = simulator.qnew.copy()
    vnew = simulator.vnew.copy()
    nc = simulator.constraint_problem.getNumberOfContacts()
    J = simulator.constraint_problem.constraint_cholesky_decomposition.matrix()[
        : 3 * nc, -model.nv :
    ]
    lam = simulator.constraint_problem.point_contact_constraint_forces()
    contact_chol = simulator.constraint_problem.constraint_cholesky_decomposition
    mu = 1e-12
    contact_chol.updateDamping(mu)
    Del = contact_chol.getDelassusCholeskyExpression().matrix() - np.eye(3 * nc) * mu

    # ============================================================================
    # testing NCP grads against finite diff
    ncp_derivatives = simple.ContactSolverDerivatives(simulator.constraint_problem)
    ncp_derivatives.measure_timings = True
    g = simulator.constraint_problem.g()
    R = simulator.constraint_problem.compliances()
    cones = simulator.constraint_problem.cones
    ndtheta = model.nv
    dGlamgdtheta = J
    dlam_dtheta = computeNCPDerivatives(simulator, dGlamgdtheta)
    for i in range(1000):
        ncp_derivatives.jvp(dGlamgdtheta)
    dlam_dtheta_cpp = ncp_derivatives.dlam_dtheta()
    gradL = dlam_dtheta.T @ lam
    dlam_dtheta_fd = np.zeros((3 * nc, ndtheta))
    # finite differences wrt g
    for i in range(ndtheta):
        dv = np.zeros(model.nv)
        delta = 1e-6
        dv[i] = delta
        dg = J @ dv
        contact_solver_plus = pin.ADMMConstraintSolver(
            3 * nc,
            simulator.admm_constraint_solver_settings.mu,
            simulator.admm_constraint_solver_settings.tau,
            simulator.admm_constraint_solver_settings.rho_power,
            simulator.admm_constraint_solver_settings.rho_power_factor,
            simulator.admm_constraint_solver_settings.ratio_primal_dual,
            simulator.admm_constraint_solver_settings.max_it_largest_eigenvalue_solver,
        )
        has_converged = contact_solver_plus.solve(
            pin.DelassusOperatorDense(Del),
            g + dg,
            cones,
            R,
            primal_solution=np.zeros(3 * nc),
        )
        lam_fd_plus = contact_solver_plus.getPrimalSolution().copy()

        contact_solver_minus = pin.ADMMConstraintSolver(
            3 * nc,
            simulator.admm_constraint_solver_settings.mu,
            simulator.admm_constraint_solver_settings.tau,
            simulator.admm_constraint_solver_settings.rho_power,
            simulator.admm_constraint_solver_settings.rho_power_factor,
            simulator.admm_constraint_solver_settings.ratio_primal_dual,
            simulator.admm_constraint_solver_settings.max_it_largest_eigenvalue_solver,
        )
        has_converged = contact_solver_minus.solve(
            pin.DelassusOperatorDense(Del),
            g - dg,
            cones,
            R,
            primal_solution=np.zeros(3 * nc),
        )
        lam_fd_minus = contact_solver_minus.getPrimalSolution().copy()
        dlam_dtheta_fd[:, i] = (lam_fd_plus - lam_fd_minus) / (delta * 2)

    assert np.linalg.norm(dlam_dtheta_fd - dlam_dtheta) < 1e-2 * np.linalg.norm(
        dlam_dtheta
    )
    assert np.linalg.norm(dlam_dtheta_fd - dlam_dtheta_cpp) < 1e-2 * np.linalg.norm(
        dlam_dtheta_cpp
    )

    # ============================================================================
    # testing step derivatives
    ndtheta = model.nv
    dqdtheta = np.zeros((model.nv, ndtheta))
    dvdtheta = np.zeros((model.nv, ndtheta))
    dtaudtheta = np.eye(model.nv)
    fext = [pin.Force(np.random.random(6)) for i in range(model.njoints)]
    dqnew_dq, dqnew_dv, dqnewdtau, dvnew_dq, dvnew_dv, dvnew_dtau = (
        computeStepDerivatives(simulator, q, v, tau, fext, dt)
    )
    sim_deriv = simple.SimulatorDerivatives(simulator)
    sim_deriv.measure_timings = True
    for i in range(1000):
        sim_deriv.stepDerivatives(simulator, q, v, tau, dt)
    assert np.linalg.norm(dvnew_dv - sim_deriv.dvnew_dv) < 1e-4
    assert np.linalg.norm(dvnew_dtau - sim_deriv.dvnew_dtau) < 1e-4
    delta = 1e-6
    dqnew_dq_fd = np.zeros((model.nv, model.nv))
    dvnew_dq_fd = np.zeros((model.nv, model.nv))
    for i in range(model.nv):
        dq = np.zeros(model.nv)
        dq[i] = delta
        q_plus = pin.integrate(model, q, dq)
        q_minus = pin.integrate(model, q, -dq)
        simulator.reset()
        simulator.step(q_minus, v, tau, dt)
        qnew_minus = simulator.qnew.copy()
        vnew_minus = simulator.vnew.copy()
        simulator.reset()
        simulator.step(q_plus, v, tau, dt)
        qnew_plus = simulator.qnew.copy()
        vnew_plus = simulator.vnew.copy()
        dqnew_dq_fd[:, i] = pin.difference(model, qnew_minus, qnew_plus) / (2 * delta)
        dvnew_dq_fd[:, i] = (vnew_plus - vnew_minus) / (2 * delta)

    dqnew_dv_fd = np.zeros((model.nv, model.nv))
    dvnew_dv_fd = np.zeros((model.nv, model.nv))
    for i in range(model.nv):
        dv = np.zeros(model.nv)
        dv[i] = delta
        v_plus = v + dv
        v_minus = v - dv
        simulator.reset()
        simulator.step(q, v_minus, tau, dt)
        qnew_minus = simulator.qnew.copy()
        vnew_minus = simulator.vnew.copy()
        simulator.reset()
        simulator.step(q, v_plus, tau, dt)
        qnew_plus = simulator.qnew.copy()
        vnew_plus = simulator.vnew.copy()
        dqnew_dv_fd[:, i] = pin.difference(model, qnew_minus, qnew_plus) / (2 * delta)
        dvnew_dv_fd[:, i] = (vnew_plus - vnew_minus) / (2 * delta)

    dqnewdtau_fd = np.zeros((model.nv, model.nv))
    dvnew_dtau_fd = np.zeros((model.nv, model.nv))
    for i in range(model.nv):
        dtau = np.zeros(model.nv)
        dtau[i] = delta
        tau_plus = tau + dtau
        tau_minus = tau - dtau
        simulator.reset()
        simulator.step(q, v, tau_minus, dt)
        qnew_minus = simulator.qnew.copy()
        vnew_minus = simulator.vnew.copy()
        simulator.reset()
        simulator.step(q, v, tau_plus, dt)
        qnew_plus = simulator.qnew.copy()
        vnew_plus = simulator.vnew.copy()
        dqnewdtau_fd[:, i] = pin.difference(model, qnew_minus, qnew_plus) / (2 * delta)
        dvnew_dtau_fd[:, i] = (vnew_plus - vnew_minus) / (2 * delta)
