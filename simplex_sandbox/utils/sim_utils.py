import pinocchio as pin
import coal
import numpy as np
import tap
import simplex
import time
import warnings

from typing import Dict
from pinocchio.visualize import MeshcatVisualizer
from simplex_sandbox.utils.viz_common import BLACK, BEIGE

try:
    import mujoco

    mujoco_imported = True
except ImportError:
    mujoco_imported = False

SUPPORTED_BACKENDS = ["simplex", "mujoco"]
SUPPORTED_SOLVERS = ["admm", "pgs", "clarabel"]


class SimulationArgs(tap.Tap):
    horizon: int = 100000
    dt: float = 1e-3
    contact_solver: str = "clarabel"
    # contact_solver: str = "admm"
    maxit: int = 1000  # solver maxit
    tol: float = 1e-6  # absolute constraint solver tol
    tol_rel: float = 1e-6  # relative constraint solver tol
    warmstart_constraint_velocities: int = (
        1  # warm start the dual variables (constraint velocities) of the solver?
    )
    warmstart_mu_prox: int = 0
    mu_prox: float = 1e-4  # prox value for admm
    admm_proximal_rule: str = "manual"
    material: str = "metal"  # contact friction
    damping: float = 0
    joint_friction: float = 0
    joint_limit: bool = False
    compliance: float = 0
    max_contacts_per_pair: int = 4
    patch_tolerance: float = 1e-3
    collision_pairs_init_margin: float = 1e-2
    baumgarte_kp: float = 0
    baumgarte_kd: float = 0
    backend: str = "simplex"
    seed: int = 2568
    random_init_vel: bool = False
    display: bool = False
    display_contacts: bool = False
    display_traj: bool = False
    display_step: int = -1
    delassus_type: str = "rigid_body"
    solve_ccp: bool = False
    debug: bool = False
    breakpoint: bool = False
    debug_step: int = -1
    admm_update_rule: str = "spectral"
    ratio_primal_dual: float = 10
    tau: float = 1.0
    tau_prox: float = 1.0
    warmstart_rho: int = 1
    rho: float = 10.0
    rho_power: float = 0.05
    rho_power_factor: float = 0.05
    lanczos_size: int = 10
    max_delassus_decomposition_updates: int = 100000
    dual_momentum: float = 0  # in [0, 1], 0 is no momentum
    rho_momentum: float = 0  # in [0, 1], 0 is no momentum
    rho_update_ratio: float = 0  # must be positive
    rho_min_update_frequency: int = 1  # must be >= 1
    anderson_acceleration_capacity: int = (
        0  # history >= 2 will trigger the anderson acceleration
    )
    linear_update_rule_factor: float = 2
    record_metrics: bool = False
    plot_metrics: int = 1
    flamegraph: bool = False  # flamegraph mode, will run the simulation multiple times
    fps: int = 60
    pd_controller: bool = False  # use PD controller
    pd_controller_qtar: np.ndarray = np.zeros(12)
    pd_controller_vtar: np.ndarray = np.zeros(12)
    pd_controller_kp: float = 1.0
    pd_controller_kd: float = 0.0

    def process_args(self) -> None:
        if self.flamegraph:
            self.display = False
            self.display_traj = False
            self.display_step = -1
            self.debug = False
            self.debug_step = -1
            self.record_metrics = False
        #
        if self.admm_update_rule == "spectral":
            self.admm_update_rule = pin.ADMMUpdateRule.SPECTRAL
        elif self.admm_update_rule == "osqp":
            self.admm_update_rule = pin.ADMMUpdateRule.OSQP
        elif self.admm_update_rule == "linear":
            self.admm_update_rule = pin.ADMMUpdateRule.LINEAR
        elif self.admm_update_rule == "constant":
            self.admm_update_rule = pin.ADMMUpdateRule.CONSTANT
        else:
            raise NotImplementedError
        #
        # if self.admm_proximal_rule == "manual":
        #     self.admm_proximal_rule = pin.ADMMProximalRule.MANUAL
        # elif self.admm_proximal_rule == "automatic":
        #     self.admm_proximal_rule = pin.ADMMProximalRule.AUTOMATIC
        # else:
        #     raise NotImplementedError
        #
        if self.delassus_type == "dense":
            self.delassus_type = simplex.DelassusType.DENSE
        elif self.delassus_type == "cholesky":
            self.delassus_type = simplex.DelassusType.CHOLESKY
        elif self.delassus_type == "rigid_body":
            self.delassus_type = simplex.DelassusType.RIGID_BODY
        else:
            raise NotImplementedError
        #
        if self.contact_solver not in SUPPORTED_SOLVERS:
            print("Solver not supported, please select from: ", SUPPORTED_SOLVERS)
            raise NotImplementedError
        #
        if self.backend not in SUPPORTED_BACKENDS:
            print("Backend not supported, please select from: ", SUPPORTED_BACKENDS)
            raise NotImplementedError


def removeBVHModelsIfAny(geom_model: pin.GeometryModel):
    for gobj in geom_model.geometryObjects:
        gobj: pin.GeometryObject
        bvh_types = [coal.BV_OBBRSS, coal.BV_OBB, coal.BV_AABB]
        ntype = gobj.geometry.getNodeType()
        if ntype in bvh_types:
            gobj.geometry.buildConvexHull(True, "Qt")
            gobj.geometry = gobj.geometry.convex


def setupJointsConstraints(model: pin.Model, args: SimulationArgs):
    # --> joint damping and friction
    model.damping[:] = np.zeros(model.nv)
    model.upperDryFrictionLimit[:] = np.zeros(model.nv)
    model.lowerDryFrictionLimit[:] = np.zeros(model.nv)
    joint_idx = 0
    for j, joint in enumerate(model.joints):
        if model.names[j] != "universe":
            if joint.nv == 1:
                model.damping[joint_idx] = args.damping
            if joint.shortname() != "JointModelFreeFlyer":
                model.lowerDryFrictionLimit[
                    joint_idx : joint_idx + joint.nv
                ] = -args.joint_friction
                model.upperDryFrictionLimit[joint_idx : joint_idx + joint.nv] = (
                    args.joint_friction
                )
            joint_idx += joint.nv

    # --> joint limits
    if not args.joint_limit:
        for i in range(model.nq):
            model.lowerPositionLimit[i] = np.finfo("d").min
            model.upperPositionLimit[i] = np.finfo("d").max
        # for i in range(model.nq):
        #     model.positionLimitMargin[i] = np.finfo("d").max


def addSystemCollisionPairs(
    model: pin.Model,
    geom_model: pin.GeometryModel,
    qref: np.ndarray,
    security_margin=1e-3,
):
    """
    Add the right collision pairs of a model, given qref.
    qref is here as a `T-pose`. The function uses this pose to determine which objects are in collision
    in this ref pose. If objects are in collision, they are not added as collision pairs, as they are considered
    to always be in collision.
    """
    import itertools

    data = model.createData()
    geom_data = geom_model.createData()
    pin.updateGeometryPlacements(model, data, geom_model, geom_data, qref)
    geom_model.removeAllCollisionPairs()
    num_col_pairs = 0
    for i, j in itertools.combinations(range(geom_model.ngeoms), 2):
        gobj_i: pin.GeometryObject = geom_model.geometryObjects[i]
        gobj_j: pin.GeometryObject = geom_model.geometryObjects[j]
        if gobj_i.name == "floor" or gobj_j.name == "floor":
            num_col_pairs += 1
            col_pair = pin.CollisionPair(i, j)
            geom_model.addCollisionPair(col_pair)
        else:
            if gobj_i.parentJoint != gobj_j.parentJoint:
                # Compute collision between the geometries. Only add the collision pair if there is no collision.
                M1 = geom_data.oMg[i]
                M2 = geom_data.oMg[j]
                colreq = coal.CollisionRequest()
                colreq.security_margin = security_margin
                colres = coal.CollisionResult()
                coal.collide(gobj_i.geometry, M1, gobj_j.geometry, M2, colreq, colres)
                if not colres.isCollision():
                    num_col_pairs += 1
                    col_pair = pin.CollisionPair(i, j)
                    geom_model.addCollisionPair(col_pair)


def addFloor(
    collision_model: pin.GeometryModel,
    visual_model: pin.GeometryModel | None,
    color: tuple[float, ...] = (0.5, 0.5, 0.5, 0.4),
    use_cube_for_visual=True,
    visual_cube_height=0.01,
) -> int | tuple[int, int]:
    # lookup floor in existing collision model
    for gobj in collision_model.geometryObjects:
        gobj: pin.GeometryObject
        if gobj.name == "floor":
            gobj.meshColor[:] = color
            if visual_model is not None:
                visual_model.geometryObjects[
                    visual_model.getGeometryId("floor")
                ].meshColor[:] = color
            return collision_model.getGeometryId("floor")

    # if not, add color
    floor_collision_shape = coal.Halfspace(0, 0, 1, 0)
    M = pin.SE3.Identity()
    floor_offset = np.array([0.0, 0.0, -0.01])
    M.translation += floor_offset
    floor_collision_object = pin.GeometryObject("floor", 0, 0, M, floor_collision_shape)
    floor_collision_object.meshColor[:] = color
    coll_gid = collision_model.addGeometryObject(floor_collision_object)

    if visual_model is None:
        return coll_gid

    if use_cube_for_visual:
        floor_visual_shape = coal.Box(20, 20, visual_cube_height)
        Mvis = pin.SE3.Identity()
        Mvis.translation[:] = (0.0, 0.0, -visual_cube_height / 2)
        Mvis.translation += floor_offset
        floor_visual_object = pin.GeometryObject(
            "floor", 0, 0, Mvis, floor_visual_shape
        )
        floor_visual_object.meshColor[:] = color
    else:
        floor_visual_object = floor_collision_object.copy()
        floor_visual_object.meshColor[:] = color
    visu_gid = visual_model.addGeometryObject(floor_visual_object)
    return coll_gid, visu_gid


def addMaterialAndCompliance(
    geom_model: pin.GeometryModel, material: str, compliance: float
):
    for gobj in geom_model.geometryObjects:
        if material == "ice":
            gobj.physicsMaterial.materialType = pin.PhysicsMaterialType.ICE
        elif material == "plastic":
            gobj.physicsMaterial.materialType = pin.PhysicsMaterialType.PLASTIC
        elif material == "wood":
            gobj.physicsMaterial.materialType = pin.PhysicsMaterialType.WOOD
        elif material == "metal":
            gobj.physicsMaterial.materialType = pin.PhysicsMaterialType.METAL
        elif material == "concrete":
            gobj.physicsMaterial.materialType = pin.PhysicsMaterialType.CONCRETE
        else:
            raise Exception("Material unknown.")

        # Compliance
        gobj.physicsMaterial.compliance = compliance


def setupSimulatorFromArgs(sim: simplex.SimulatorX, args: SimulationArgs):
    sim.config.measure_timings = args.record_metrics
    sim.config.warmstart_constraint_velocities = args.warmstart_constraint_velocities
    sim.workspace.constraint_problem.setMaxNumberOfContactsPerCollisionPair(
        args.max_contacts_per_pair
    )
    sim.workspace.constraint_problem.is_ncp = not args.solve_ccp
    # Delassus type
    sim.workspace.constraint_problem.delassus_type = args.delassus_type
    # PGS
    sim.config.constraint_solvers_configs.pgs_config.stat_record = (
        args.debug or args.debug_step >= 0
    )
    sim.config.constraint_solvers_configs.pgs_config.max_iterations = args.maxit
    sim.config.constraint_solvers_configs.pgs_config.absolute_precision = (
        args.tol
    )
    sim.config.constraint_solvers_configs.pgs_config.relative_precision = (
        args.tol_rel
    )
    sim.config.constraint_solvers_configs.pgs_config.absolute_complementarity_tol = args.tol
    sim.config.constraint_solvers_configs.pgs_config.relative_complementarity_tol = args.tol_rel
    # CLARABEL
    clarabel_config = sim.config.constraint_solvers_configs.clarabel_config
    clarabel_config.max_iter = args.maxit
    clarabel_config.absolute_precision = args.tol
    clarabel_config.relative_precision = args.tol_rel
    clarabel_config.tol_feas = args.tol
    clarabel_config.tol_ktratio = args.tol
    clarabel_config.verbose = False
    # ADMM
    sim.config.constraint_solvers_configs.admm_config.stat_record = (
        args.debug or args.debug_step >= 0
    )
    sim.config.constraint_solvers_configs.admm_config.max_iterations = args.maxit
    sim.config.constraint_solvers_configs.admm_config.absolute_precision = (
        args.tol
    )
    sim.config.constraint_solvers_configs.admm_config.relative_precision = (
        args.tol_rel
    )
    sim.config.constraint_solvers_configs.admm_config.ratio_primal_dual = (
        args.ratio_primal_dual
    )
    sim.config.constraint_solvers_configs.admm_config.mu_prox = args.mu_prox
    sim.config.constraint_solvers_configs.admm_config.warmstart_mu_prox = (
        args.warmstart_mu_prox
    )
    sim.config.constraint_solvers_configs.admm_config.rho = args.rho
    sim.config.constraint_solvers_configs.admm_config.warmstart_rho = args.warmstart_rho
    sim.config.constraint_solvers_configs.admm_config.tau = args.tau
    sim.config.constraint_solvers_configs.admm_config.tau_prox = args.tau_prox
    sim.config.constraint_solvers_configs.admm_config.rho_power_factor = (
        args.rho_power_factor
    )
    sim.config.constraint_solvers_configs.admm_config.rho_power = (
        args.rho_power
    )
    sim.config.constraint_solvers_configs.admm_config.rho_momentum = (
        args.rho_momentum
    )
    sim.config.constraint_solvers_configs.admm_config.rho_update_ratio = (
        args.rho_update_ratio
    )
    sim.config.constraint_solvers_configs.admm_config.rho_min_update_frequency = (
        args.rho_min_update_frequency
    )
    sim.config.constraint_solvers_configs.admm_config.anderson_acceleration_capacity = (
        args.anderson_acceleration_capacity
    )
    sim.config.constraint_solvers_configs.admm_config.lanczos_size = (
        args.lanczos_size
    )
    sim.config.constraint_solvers_configs.admm_config.max_delassus_decomposition_updates = args.max_delassus_decomposition_updates
    sim.config.constraint_solvers_configs.admm_config.dual_momentum = (
        args.dual_momentum
    )
    sim.config.constraint_solvers_configs.admm_config.linear_update_rule_factor = (
        args.linear_update_rule_factor
    )
    # sim.config.constraint_solvers_configs.admm_config.admm_update_rule = (
    #     args.admm_update_rule
    # )
    # sim.config.constraint_solvers_configs.admm_config.admm_proximal_rule = (
    #     args.admm_proximal_rule
    # )

    for cm in sim.workspace.constraint_problem.frictional_point_constraint_models:
        cm.baumgarte_corrector_parameters.Kp = args.baumgarte_kp
        cm.baumgarte_corrector_parameters.Kd = args.baumgarte_kd

    for patch_req in sim.geom_data.contactPatchRequests:
        patch_req.setPatchTolerance(args.patch_tolerance)


def plotContactSolver(
    sim: simplex.SimulatorX,
    args: SimulationArgs,
    t: int,
    q: np.ndarray,
    v: np.ndarray,
):
    import matplotlib.pyplot as plt

    if args.debug or (t >= args.debug_step and args.debug_step >= 0):
        stats: pin.SolverStats = sim.workspace.constraint_solvers.admm_solver.stats
        if args.contact_solver == "admm":
            solver = sim.workspace.constraint_solvers.admm_solver
            solver_result = sim.workspace.constraint_solvers.admm_result
        if args.contact_solver == "pgs":
            solver = sim.workspace.constraint_solvers.pgs_solver
            solver_result = sim.workspace.constraint_solvers.pgs_result
        if args.contact_solver == "clarabel":
            solver = sim.workspace.constraint_solvers.clarabel_solver
        stats = solver.stats
        primal_feas = solver_result.primal_feasibility
        dual_feas = solver_result.dual_feasibility
        it = solver_result.iterations
        if stats.size() > 0:
            plt.cla()
            plt.clf()
            constraint_problem_type = "CCP" if args.solve_ccp else "NCP"
            title = f"{args.contact_solver}, {constraint_problem_type}\nStep {t}, it = {it}, primal feas = {primal_feas:.2e}, dual feas = {dual_feas:.2e}"
            if args.contact_solver == "admm":
                title += f", delassus update count: {stats.delassus_decomposition_update_count}"
            plt.suptitle(title)

            # Create the main plot with log scale
            ax1 = plt.gca()
            ax1.clear()
            lines1 = ax1.plot(stats.primal_feasibility, label="primal feas")
            lines1 += ax1.plot(stats.dual_feasibility, label="dual feas")
            # lines1 += ax1.plot(stats.linear_system_residual, label="lin system residual")
            # lines1 += ax1.plot(stats.linear_system_consistency, label="lin system consistency")
            if not args.solve_ccp:
                lines1 += ax1.plot(stats.dual_feasibility_ncp, label="dual feas NCP")

            lines2 = []
            lines3 = []

            if args.contact_solver == "admm":
                # Create secondary y-axis for rho and mu_prox with log scale
                ax2 = ax1.twinx()
                lines2 = ax2.plot(
                    stats.rho, label="rho", color="tab:red", linestyle="-."
                )
                lines2 += ax2.plot(
                    stats.mu_prox, label="mu_prox", color="tab:purple", linestyle="-."
                )
                ax2.set_yscale("log")
                ax2.set_ylabel("Rho / Mu_prox (log scale)", color="tab:red")
                ax2.tick_params(axis="y", labelcolor="tab:red")

                # Create third y-axis for anderson_size with linear scale
                ax3 = ax1.twinx()
                ax3.spines["right"].set_position(
                    ("outward", 60)
                )  # Offset the third axis
                lines3 = ax3.plot(
                    stats.anderson_size,
                    label="anderson_size",
                    color="tab:orange",
                    linestyle="--",
                )
                ax3.set_ylabel("Anderson Size", color="tab:orange")
                ax3.tick_params(axis="y", labelcolor="tab:orange")

                # Force integer ticks on anderson_size axis
                from matplotlib.ticker import MaxNLocator

                ax3.yaxis.set_major_locator(MaxNLocator(integer=True))

            if args.contact_solver == "pgs":
                lines1 += ax1.plot(stats.complementarity, label="complementarity")

            ax1.set_yscale("log")
            ax1.set_ylabel("Feasibility (log scale)")

            # Combine all lines and labels for a single legend
            all_lines = lines1 + lines2 + lines3
            all_labels = [l.get_label() for l in all_lines]
            ax1.legend(all_lines, all_labels, loc="upper right", ncol=2)

            plt.show(block=False)
            plt.pause(0.1)
        print(f"{t=}")
        print(f"{q=}")
        print(f"{v=}")
        print(f"{sim.state.qnew=}")
        print(f"{sim.state.vnew=}")
        print(f"{sim.model.nq=}")
        print(f"{sim.model.nv=}")
        print(f"{sim.workspace.constraint_solvers.admm_result.iterations=}")
        print(f"{sim.workspace.constraint_solvers.pgs_result.iterations=}")
        print(f"{sim.workspace.constraint_problem.constraint_forces=}")
        print(f"{sim.workspace.constraint_problem.constraint_problem_size=}")
        print(f"{sim.workspace.constraint_problem.joint_friction_constraint_size=}")
        print(f"{sim.workspace.constraint_problem.joint_limit_constraint_max_size=}")
        print(f"{sim.workspace.constraint_problem.joint_limit_constraint_size=}")
        print(f"{sim.workspace.constraint_problem.point_anchor_constraint_size=}")
        print(f"{sim.workspace.constraint_problem.frame_anchor_constraint_size=}")
        print(f"{sim.workspace.constraint_problem.point_contact_constraint_size=}")
        print(f"{sim.workspace.constraint_problem.getNumberOfContacts()=}")
        print("Constraint solver timings: ", sim.timings.timings_constraint_solver.user)
        print(f"{t=}")


def cmdline_debug():
    """Enhanced debug with tab completion and keyboard shortcuts"""
    import inspect
    import readline
    import rlcompleter
    import sys

    frame = inspect.currentframe().f_back
    local_vars = frame.f_locals
    global_vars = frame.f_globals

    # Set up completion
    completer = rlcompleter.Completer(namespace={**global_vars, **local_vars})
    readline.set_completer(completer.complete)
    readline.parse_and_bind("tab: complete")

    print("Enhanced debug mode with tab completion.")
    print("Commands: 'exit'/'c', 'vars', 'help', or any Python code")
    print("Shortcuts: TAB (completion), Ctrl+C (new prompt), Ctrl+D (exit program)")

    while True:
        try:
            cmd = input("debug> ")

            if cmd.lower() in ["exit", "quit", "continue", "c"]:
                break
            elif cmd.lower() == "vars":
                print("Local variables:")
                for name, value in local_vars.items():
                    if not name.startswith("_"):
                        print(f"  {name}: {type(value)}")
            elif cmd.lower() == "help":
                print("Available commands:")
                print("  vars, help, exit/c, or any Python code")
                print("  Use TAB for autocompletion")
                print("  Ctrl+C: interrupt current input, get new prompt")
                print("  Ctrl+D: exit entire program")
            else:
                try:
                    result = eval(cmd, global_vars, local_vars)
                    if result is not None:
                        print(result)
                except SyntaxError:
                    exec(cmd, global_vars, local_vars)

        except KeyboardInterrupt:
            # Ctrl+C - just continue to new prompt
            print("\nKeyboardInterrupt - continuing to new prompt")
            continue

        except EOFError:
            # Ctrl+D - exit the entire program
            print("\nExiting program...")
            sys.exit(0)

        except Exception as e:
            print(f"Error: {e}")


def displaySimulationTrajectory(
    vizer: MeshcatVisualizer, qs: np.ndarray, args: SimulationArgs
):
    import time
    from simplex_sandbox.utils.viz_utils import subSample

    fps = float(args.fps)
    dt_vis = 1.0 / fps
    qs, ts = subSample(qs, args.dt * args.horizon, fps)
    while True:
        last_t = 0
        for i, q in enumerate(qs):
            if args.display_step > 0:
                if ts[i] - last_t > args.display_step:
                    vizer.display(q)
                    last_t = ts[i]
                    input(f"[Stopped simulation at t={ts[i]}]")
            else:
                step_start = time.time()
                vizer.display(q)
                time_until_next_step = dt_vis - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


class SimulationMetrics:
    def __init__(self, horizon: int):
        self.horizon = horizon
        self.problem_size = np.zeros(self.horizon)
        self.joint_friction_size = np.zeros(self.horizon)
        self.point_anchor_size = np.zeros(self.horizon)
        self.frame_anchor_size = np.zeros(self.horizon)
        self.joint_limit_size = np.zeros(self.horizon)
        self.contact_size = np.zeros(self.horizon)
        self.solver_numits = np.zeros(self.horizon)
        self.delassus_updates = np.zeros(self.horizon)
        self.step_timings = np.zeros(self.horizon)
        self.col_timings = np.zeros(self.horizon)
        self.broadphase_col_timings = np.zeros(self.horizon)
        self.narrowphase_col_timings = np.zeros(self.horizon)
        self.solver_timings = np.zeros(self.horizon)


def plotSimulationMetrics(metrics: SimulationMetrics, args: SimulationArgs):
    import matplotlib.pyplot as plt

    # Create figure with tight layout for better spacing
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 8))
    plt.tight_layout(pad=5.0)  # Add padding between subplots

    # Define colors for better distinction
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
    linestyles = ["--", "--", "-.", ":", "-.", "--"]
    markers = ["o", "s", "^", "v", ",", ">"]
    linewidth = 1

    ax[0, 0].plot(
        metrics.problem_size,
        label="Problem size",
        color=colors[0],
        linewidth=linewidth,
        markevery=args.horizon // 10,
        linestyle=linestyles[0],
        marker=markers[0],
    )
    if np.sum(metrics.joint_friction_size) > 0:
        ax[0, 0].plot(
            metrics.joint_friction_size,
            label="Joint friction size",
            color=colors[1],
            linewidth=linewidth,
            markevery=args.horizon // 10,
            linestyle=linestyles[1],
            marker=markers[1],
        )
    if np.sum(metrics.joint_limit_size) > 0:
        ax[0, 0].plot(
            metrics.joint_limit_size,
            label="Joint limit size",
            color=colors[2],
            linewidth=linewidth,
            markevery=args.horizon // 10,
            linestyle=linestyles[2],
            marker=markers[2],
        )
    if np.sum(metrics.point_anchor_size) > 0:
        ax[0, 0].plot(
            metrics.point_anchor_size,
            label="Point anchor size",
            color=colors[3],
            linewidth=linewidth,
            markevery=args.horizon // 10,
            linestyle=linestyles[3],
            marker=markers[3],
        )
    if np.sum(metrics.frame_anchor_size) > 0:
        ax[0, 0].plot(
            metrics.frame_anchor_size,
            label="Frame anchor size",
            color=colors[4],
            linewidth=linewidth,
            markevery=args.horizon // 10,
            linestyle=linestyles[4],
            marker=markers[4],
        )
    if np.sum(metrics.contact_size) > 0:
        ax[0, 0].plot(
            metrics.contact_size,
            label="Contacts size",
            color=colors[5],
            linewidth=linewidth,
            markevery=args.horizon // 10,
            linestyle=linestyles[5],
            marker=markers[5],
        )
    #
    ax[0, 0].set_ylabel("Problem size")
    # ax[0, 0].set_title("Problem size along trajectory")
    ax[0, 0].legend(loc="best", fontsize=9)  # Add legend to this subplot
    ax[0, 0].grid(True, alpha=0.3)  # Add grid for better readability

    # Top-right: Solver iterations
    ax[0, 1].plot(
        metrics.solver_numits,
        color=colors[0],
        linewidth=linewidth,
        linestyle=linestyles[0],
    )
    ax[0, 1].set_ylabel("Solver iters")
    # ax[0, 1].set_title(f"{args.contact_solver} solver iters along trajectory")
    ax[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Delassus updates
    ax[1, 0].plot(
        metrics.delassus_updates,
        color=colors[1],
        linewidth=linewidth,
        linestyle=linestyles[0],
    )
    ax[1, 0].set_xlabel("Timestep")
    ax[1, 0].set_ylabel("Delassus updates")
    ax[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Timings with individual legend
    ax[1, 1].plot(
        metrics.step_timings,
        label="Step timings",
        color=colors[0],
        linewidth=linewidth,
        linestyle=linestyles[0],
    )
    ax[1, 1].plot(
        metrics.solver_timings,
        label="Solver timings",
        color=colors[1],
        linewidth=linewidth,
        linestyle=linestyles[0],
    )
    ax[1, 1].plot(
        metrics.col_timings,
        label="Collision detection timings",
        color=colors[2],
        linewidth=linewidth,
        linestyle=linestyles[0],
    )
    ax[1, 1].plot(
        metrics.narrowphase_col_timings,
        label="Narrowphase collision detection timings",
        color=colors[3],
        linewidth=linewidth,
        linestyle=linestyles[0],
    )
    ax[1, 1].plot(
        metrics.broadphase_col_timings,
        label="Broadphase collision detection timings",
        color=colors[4],
        linewidth=linewidth,
        linestyle=linestyles[0],
    )
    ax[1, 1].set_xlabel("Timestep")
    ax[1, 1].set_ylabel("Timings (micro seconds)")
    ax[1, 1].set_yscale("log")
    # ax[1, 1].set_title("Simulation timings along trajectory")
    ax[1, 1].legend(loc="best", fontsize=9)  # Add legend to this subplot
    ax[1, 1].grid(True, alpha=0.3)

    # Main title
    constraint_problem = "CCP" if args.solve_ccp else "NCP"
    run_time = np.sum(metrics.step_timings) * 1e-6
    step_freq = (metrics.step_timings.size / run_time) * 1e-3
    fig.suptitle(
        f"{args.contact_solver}, {constraint_problem}\ntol = {args.tol}, tol_rel = {args.tol_rel}, maxit = {args.maxit}, dt = {args.dt}, horizon = {args.horizon}, step freq = {step_freq:.2f} Khz",
    )

    # Remove the global legend call since we're using individual legends
    plt.pause(0.1)
    plt.show(block=False)
    input("[Press ENTER to continue]")


def printSimulationPerfStats(step_timings: np.ndarray):
    run_time = np.sum(step_timings) * 1e-6
    print("============================================")
    print("SIMULATION")
    print(f"Time elapsed: {run_time} (s)")
    print(
        "Mean timings time step: {mean:.2f} +/- {std:.2f} microseconds".format(
            mean=np.mean(step_timings), std=np.std(step_timings)
        )
    )
    print(
        "Steps frequency: {freq:.2f} kHz".format(
            freq=((step_timings.size) / run_time) * 1e-3
        )
    )
    print("============================================")


def pd_controller(
    model: pin.Model,
    q: np.ndarray,
    v: np.ndarray,
    qtar: np.ndarray,
    vtar: np.ndarray,
    kp: float,
    kd: float,
    tau_zero: np.ndarray,
):
    # assumes that you can control the last n joints
    idxs = vtar.shape[0]
    dq = pin.difference(model, q, qtar)
    tau_zero[-idxs:] = kp * dq[-idxs:] + kd * (vtar - v[-idxs:])
    return tau_zero


def simulateSystem(
    sim: simplex.SimulatorX,
    q0: np.ndarray,
    v0: np.ndarray,
    args: SimulationArgs,
    vizer: MeshcatVisualizer = None,
    prev_vizer: MeshcatVisualizer = None,
):
    from simplex_sandbox.utils.viz_utils import register_object

    q = q0.copy()
    v = v0.copy()
    qs = np.zeros((args.horizon, sim.model.nq))
    metrics = SimulationMetrics(args.horizon)
    zero_torque = np.zeros(sim.model.nv)

    if args.contact_solver == "admm":
        solver = sim.workspace.constraint_solvers.admm_solver
        solver_type = simplex.ConstraintSolverType.ADMM
    elif args.contact_solver == "pgs":
        solver = sim.workspace.constraint_solvers.pgs_solver
        solver_type = simplex.ConstraintSolverType.PGS
    elif args.contact_solver == "clarabel":
        solver = sim.workspace.constraint_solvers.clarabel_solver
        solver_type = simplex.ConstraintSolverType.CLARABEL
    else:
        raise NotImplementedError

    sim.reset()
    prev_contact_placements = None
    for t in range(args.horizon):
        try:
            if args.pd_controller:
                zero_torque = pd_controller(
                    sim.model,
                    q,
                    v,
                    args.pd_controller_qtar,
                    args.pd_controller_vtar,
                    args.pd_controller_kp,
                    args.pd_controller_kd,
                    zero_torque,
                )
            sim.step(q, v, zero_torque, args.dt, solver_type)
        except Exception as e:
            if (
                args.display
                and (
                    args.debug
                    or (t >= args.debug_step and args.debug_step >= 0)
                    or args.display_traj
                )
            ) and (vizer is not None):
                if prev_vizer is not None:
                    prev_vizer.display(q)
                vizer.display(sim.state.qnew)
            input(
                f"[Simulation step failed at t = {t}!]\n{e}\n[PRESS ENTER TO CONTINUE]"
            )
            break

        # --> Record metrics
        if args.record_metrics:
            metrics.problem_size[t] = (
                sim.workspace.constraint_problem.constraints_problem_size
            )
            metrics.joint_friction_size[t] = (
                sim.workspace.constraint_problem.joint_friction_constraint_size
            )
            metrics.point_anchor_size[t] = (
                sim.workspace.constraint_problem.bilateral_constraints_size
            )
            metrics.frame_anchor_size[t] = (
                sim.workspace.constraint_problem.weld_constraints_size
            )
            metrics.joint_limit_size[t] = (
                sim.workspace.constraint_problem.joint_limit_constraint_size
            )
            metrics.contact_size[t] = (
                sim.workspace.constraint_problem.frictional_point_constraints_size
            )
            #
            metrics.step_timings[t] = sim.timings.timings_step.user
            metrics.col_timings[t] = sim.timings.timings_collision_detection.user
            metrics.broadphase_col_timings[t] = (
                sim.timings.timings_broadphase_collision_detection.user
            )
            metrics.narrowphase_col_timings[t] = (
                sim.timings.timings_narrowphase_collision_detection.user
            )
            metrics.solver_timings[t] = sim.timings.timings_constraint_solver.user
            #
            metrics.solver_numits[t] = solver.getIterationCount()
            if args.contact_solver == "admm":
                metrics.delassus_updates[t] = (
                    solver.getDelassusDecompositionUpdateCount()
                )

        if args.debug or (t >= args.debug_step and args.debug_step >= 0):
            plotContactSolver(sim, args, t, q, v)

        if (
            args.display
            and (
                args.debug
                or (t >= args.debug_step and args.debug_step >= 0)
                or args.display_traj
            )
        ) and (vizer is not None):
            if prev_vizer is not None:
                prev_vizer.display(q)
                prev_vizer.viewer["prev_contact_info"].delete()
            vizer.display(sim.state.qnew)
            vizer.viewer["contact_info"].delete()

            if args.display_contacts:
                if prev_vizer is not None and prev_contact_placements is not None:
                    for j, oMc in enumerate(prev_contact_placements):
                        sphere = coal.Sphere(0.005)
                        cp_name = f"prev_contact_info/contact_point_{j}"
                        print(f"{j}th contact: ", oMc.translation)
                        register_object(prev_vizer, sphere, cp_name, oMc, BLACK)

                contact_placements = (
                    sim.workspace.constraint_problem.point_contact_constraint_placements
                )
                for j, oMc in enumerate(contact_placements):
                    sphere = coal.Sphere(0.005)
                    cp_name = f"contact_info/contact_point_{j}"
                    print(f"{j}th contact: ", oMc.translation)
                    register_object(vizer, sphere, cp_name, oMc, BLACK)

                prev_contact_placements = contact_placements.copy()

            if args.debug or (t >= args.debug_step and args.debug_step >= 0):
                if args.breakpoint:
                    cmdline_debug()
                else:
                    input("[Press enter to continue]")

        q = sim.state.qnew.copy()
        v = sim.state.vnew.copy()
        qs[t, :] = q[:]

    if args.record_metrics:
        printSimulationPerfStats(metrics.step_timings)

    return metrics, qs


def runSimpleXSimulation(
    sim: simplex.SimulatorX,
    args: SimulationArgs,
    q0: np.ndarray,
    v0: np.ndarray,
    vizer: MeshcatVisualizer = None,
    prev_vizer: MeshcatVisualizer = None,
):
    setupSimulatorFromArgs(sim, args)

    if args.flamegraph:
        for _ in range(25):
            simulateSystem(sim, q0, v0, args)
    else:
        input("[Press ENTER to simulate]")
        sim_metrics, qs = simulateSystem(sim, q0, v0, args, vizer, prev_vizer)

        if args.record_metrics and args.plot_metrics:
            plotSimulationMetrics(sim_metrics, args)
            print("Plotting metrics...")

        if args.display:
            displaySimulationTrajectory(vizer, qs, args)


def runSimpleXSimulationFromModel(
    model: pin.Model,
    geom_model: pin.GeometryModel,
    visual_model: pin.GeometryModel,
    q0: np.ndarray,
    v0: np.ndarray,
    args: SimulationArgs,
    add_floor: bool = False,
    point_and_frame_anchor_constraints_dict: Dict[str, pin.ConstraintModel] = None,
    add_system_collision_pairs: bool = True,
):
    from .viz_utils import createVisualizer

    # Add floor and collision pairs (floor may already be present in XML file)
    if add_floor:
        addFloor(geom_model, visual_model)
    if add_system_collision_pairs:
        addSystemCollisionPairs(model, geom_model, q0, args.collision_pairs_init_margin)

    # Finalize setup of model/geom_model
    # --> remove BVH and replace by convex
    removeBVHModelsIfAny(geom_model)
    # --> set material properties for contacts
    addMaterialAndCompliance(geom_model, args.material, args.compliance)
    # --> set joint limits, joint friction, joint damping
    setupJointsConstraints(model, args)

    # Visualize the trajectory
    if args.display_contacts:
        visual_model = geom_model
    vizer = None
    prev_vizer = None
    if args.display:
        print(
            "[Created a visualizer to display the trajectory. Please run a `meshcat-server` in the terminal if a meshcat server is not already running.]"
        )
        vizer, _ = createVisualizer(
            model, geom_model, visual_model, zmq_url="tcp://127.0.0.1:6000"
        )
        vizer.display(q0)
        if args.debug or args.debug_step > 0 or args.breakpoint:
            print(
                "[Created a debug visualizer to display the previous steps in the trajectory. Please run another `meshcat-server` in the terminal if a meshcat server is not already running.]"
            )
            prev_vizer, _ = createVisualizer(
                model, geom_model, visual_model, zmq_url="tcp://127.0.0.1:6001"
            )
            prev_vizer.display(q0)

    # Simulation
    if point_and_frame_anchor_constraints_dict is not None:
        sim = simplex.SimulatorX(model, geom_model)
        print(point_and_frame_anchor_constraints_dict)
        sim.addPointAnchorConstraints(
            point_and_frame_anchor_constraints_dict["bilateral_point_constraint_models"]
        )
        sim.addFrameAnchorConstraints(
            point_and_frame_anchor_constraints_dict["weld_constraint_models"]
        )
    else:
        sim = simplex.SimulatorX(model, geom_model)

    runSimpleXSimulation(sim, args, q0, v0, vizer, prev_vizer)


def mergeModels(
    model0: pin.Model,
    geom_model0: pin.GeometryModel,
    visual_model0: pin.GeometryModel,
    q0: np.ndarray,
    v0: np.ndarray,
    model1: pin.Model,
    geom_model1: pin.GeometryModel,
    visual_model1: pin.GeometryModel,
    q1: np.ndarray,
    v1: np.ndarray,
    parent_joint: int = 0,
    placement=pin.SE3.Identity(),
):
    """
    Merge two Pinocchio (q, model, geom_model) sets into one composite model.

    Parameters:
    - q0: np.array, configuration vector for model0
    - model0: pinocchio.Model, first model
    - geom0: pinocchio.GeometryModel, geometry of first model
    - q1: np.array, configuration vector for model1
    - model1: pinocchio.Model, second model
    - geom1: pinocchio.GeometryModel, geometry of second model
    - parent_joint: int, joint ID in model0 where model1 will be attached
    - placement: pinocchio.SE3, pose of model1 frame relative to parent_joint
    - prefix: str, name prefix for all joints and frames of model1

    Returns:
    - q: np.array, concatenated configuration vector
    - model: pinocchio.Model, merged model
    - geom_model: pinocchio.GeometryModel, merged geometry model
    """

    # Append the second model under the given parent joint
    model, geom_model = pin.appendModel(
        model0, model1, geom_model0, geom_model1, parent_joint, placement
    )
    _, visual_model = pin.appendModel(
        model0, model1, visual_model0, visual_model1, parent_joint, placement
    )

    # Concatenate configurations and init vel
    q = np.concatenate([q0, q1])
    v = np.concatenate([v0, v1])

    return model, geom_model, visual_model, q, v


def createPyramid(mass: float, levels: int, cube_size: float = 1.0):
    """
    Create a Pinocchio model and geometry_model representing a pyramid of cubes.

    Parameters:
    - mass: float, mass of each cube
    - levels: int, number of levels in the pyramid
    - cube_size: float, edge length of each cube (default 1.0)

    Returns:
    - model: pinocchio.Model containing joints and inertias for each cube
    - geom_model: pinocchio.GeometryModel containing visual geometry for each cube
    """
    # Initialize empty model and geometry model
    model = pin.Model()
    geom_model = pin.GeometryModel()

    # Iterate over pyramid levels
    for i in range(levels):
        # Number of cubes in this level
        row_count = levels - i
        # Height of this level (z-axis)
        z = (i + 0.5) * (cube_size + 1e-2)

        # Center the row around x=0
        for j in range(row_count):
            x = (j - (row_count - 1) / 2.0) * (cube_size + 1e-2)
            placement = pin.SE3(np.eye(3), np.array([x, 0.0, z]))
            name = f"cube_{i}_{j}"

            # Add a free-flyer joint for this cube
            joint_id = model.addJoint(0, pin.JointModelFreeFlyer(), placement, name)

            inertia = pin.Inertia.FromBox(
                mass, cube_size / 2.0, cube_size / 2.0, cube_size / 2.0
            )
            # Attach inertia to the joint
            model.appendBodyToJoint(joint_id, inertia, pin.SE3.Identity())

            # Create box geometry for visualization
            box = coal.Box(np.array([cube_size, cube_size, cube_size]))
            geom = pin.GeometryObject(name + "_geom", joint_id, box, pin.SE3.Identity())
            geom_model.addGeometryObject(geom)

    return model, geom_model


def create3DPyramid(mass, levels, cube_size=1.0, spacing=1e-2):
    """
    Create a Pinocchio model and geometry_model representing a 3D pyramid of cubes.

    Each level forms an n x n grid of cubes (where n = levels - level_index),
    stacked vertically with a small spacing.

    Parameters:
    - mass: float, mass of each cube
    - levels: int, number of levels in the pyramid
    - cube_size: float, edge length of each cube (default 1.0)
    - spacing: float, gap between cubes (default 1e-2)

    Returns:
    - model: pinocchio.Model containing free-flyer joints and inertias for each cube
    - geom_model: pinocchio.GeometryModel containing visual geometry for each cube
    """
    # Initialize model and geometry model
    model = pin.Model()
    geom_model = pin.GeometryModel()

    # Precompute half sizes for inertia
    half = cube_size / 2.0

    # Loop over each pyramid level
    for level in range(levels):
        n = levels - level
        z = (level + 0.5) * (cube_size + spacing)

        # Create an n x n grid in the XY plane
        for i in range(n):
            for j in range(n):
                # Center grid around (0,0)
                x = (i - (n - 1) / 2.0) * (cube_size + spacing)
                y = (j - (n - 1) / 2.0) * (cube_size + spacing)
                placement = pin.SE3(np.eye(3), np.array([x, y, z]))
                name = f"cube_{level}_{i}_{j}"

                # Add free-flyer joint for this cube
                joint_id = model.addJoint(0, pin.JointModelFreeFlyer(), placement, name)

                # Attach inertia (uniform cube)
                inertia = pin.Inertia.FromBox(mass, half, half, half)
                model.appendBodyToJoint(joint_id, inertia, pin.SE3.Identity())

                # Create visual geometry (box)
                box = coal.Box(np.array([cube_size, cube_size, cube_size]))
                geom = pin.GeometryObject(
                    name + "_geom", joint_id, box, pin.SE3.Identity()
                )
                color = BEIGE
                geom.meshColor = color
                geom_model.addGeometryObject(geom)

    return model, geom_model


def createFreeflyerModel(freeflyer_name: str):
    """
    Create a Pinocchio model with a single free-flyer joint.

    Returns:
    - q0: np.array, neutral configuration vector for the free-flyer
    - model: pinocchio.Model with one free-flyer joint
    - geom_model: pinocchio.GeometryModel (empty)
    """
    # Initialize an empty model and geometry model
    model = pin.Model()
    geom_model = pin.GeometryModel()

    # Add a single free-flyer joint named 'base'
    ff_id = model.addJoint(
        0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), freeflyer_name
    )
    model.addJointFrame(ff_id)

    # The neutral configuration for a free-flyer is size 7 (quaternion + translation)
    q0 = pin.neutral(model)

    return model, geom_model, geom_model.copy(), q0


def addRobotName(model, geom_model, visual_model, robot_name):
    for i in range(1, len(model.joints)):
        model.names[i] += "_" + robot_name
    for i in range(1, len(model.frames)):
        model.frames[i].name += "_" + robot_name
    for i in range(len(geom_model.geometryObjects)):
        geom_model.geometryObjects[i].name += "_" + robot_name
    for i in range(len(visual_model.geometryObjects)):
        visual_model.geometryObjects[i].name += "_" + robot_name
    model = model.copy()
    geom_model = geom_model.copy()
    visual_model = visual_model.copy()


def runMujocoXML(model_path: str, args: SimulationArgs):
    if not mujoco_imported:
        warnings.warn(
            "Can't run this function, as module 'mujoco' was not found. Please install mujoco."
        )
        return
    from robot_descriptions.loaders.mujoco import load_robot_description as mlr
    import mujoco.viewer

    if model_path.endswith(".xml"):
        m = mujoco.MjModel.from_xml_path(model_path)
    else:
        m = mlr(model_path)
    m.opt.cone = 1  # Elliptic
    m.opt.solver = 2  # Newton
    m.opt.timestep = args.dt
    m.opt.iterations = args.maxit
    m.opt.tolerance = args.tol
    m.opt.ls_iterations = 50
    m.opt.ls_tolerance = 1e-2
    d = mujoco.MjData(m)
    d.qpos = m.qpos0
    q0 = d.qpos.copy()
    v0 = d.qvel.copy()
    a0 = d.qacc.copy()

    print(f"{q0=}")
    print(f"{v0=}")

    step_timings = np.zeros(args.horizon)
    for t in range(args.horizon):
        start_time = time.perf_counter_ns()
        mujoco.mj_step(m, d)
        end_time = time.perf_counter_ns()
        step_timings[t] = (end_time - start_time) * 1e-3
    printSimulationPerfStats(step_timings)

    d.qpos = q0
    d.qvel = v0
    d.qacc = a0
    show_ui = False
    display_contacts = False
    with mujoco.viewer.launch_passive(
        m, d, show_left_ui=show_ui, show_right_ui=show_ui
    ) as viewer:
        input("[Press enter to display trajectory]")
        while True:
            d.qpos = q0
            d.qvel = v0
            d.qacc = a0
            for t in range(args.horizon):
                step_start = time.time()
                # mj_step can be replaced with code that also evaluates
                # a policy and applies a control signal before stepping the physics.
                mujoco.mj_step(m, d)

                # Example modification of a viewer option: toggle contact points every two seconds.
                if display_contacts:
                    with viewer.lock():
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = (
                            1  # int(d.time % 2)
                        )
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = (
                            1  # int(d.time % 2)
                        )
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = (
                            1  # int(d.time % 2)
                        )

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if args.debug:
                    if args.display_step <= 0 or (
                        args.display_step > 0 and t % args.display_step == 0
                    ):
                        print(d.qpos)
                        input(f"==== TIMESTEP {t} ====")
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
