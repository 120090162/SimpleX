import itertools
import pinocchio as pin
import coal
import numpy as np
import tap
import simplex
import os
import yaml
from utils.logger import LOGGER

SUPPORTED_SOLVERS = ["admm", "pgs", "clarabel"]


class SimulationArgs(tap.Tap):
    """
    Configuration class for SimpleX simulation parameters.
    Inherits from `tap.Tap` to allow parsing arguments from the command line
    or updating via YAML. Holds hyperparameters for solvers, contact physics,
    constraints, and debugging options.
    """

    horizon: int = 100000
    dt: float = 1e-3
    sim_fps: int = 1000
    render_fps: int = 60
    contact_solver: str = "clarabel"
    maxit: int = 1000  # solver maxit
    tol: float = 1e-6  # absolute constraint solver tol
    tol_rel: float = 1e-6  # relative constraint solver tol
    warmstart_constraint_velocities: int = (
        1  # warm start the dual variables (constraint velocities) of the solver?
    )
    warmstart_mu_prox: int = 0
    mu_prox: float = 1e-4  # prox value for admm
    # admm_proximal_rule: str = "manual"
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
    delassus_type: str = "rigid_body"
    solve_ccp: bool = False
    debug: bool = False
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

    def process_args(self) -> None:
        # setup the ADMM update rule
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
        # setup the Delassus operator type
        if self.delassus_type == "dense":
            self.delassus_type = simplex.DelassusType.DENSE
        elif self.delassus_type == "cholesky":
            self.delassus_type = simplex.DelassusType.CHOLESKY
        elif self.delassus_type == "rigid_body":
            self.delassus_type = simplex.DelassusType.RIGID_BODY
        else:
            raise NotImplementedError
        # check the contact solver
        if self.contact_solver not in SUPPORTED_SOLVERS:
            print("Solver not supported, please select from: ", SUPPORTED_SOLVERS)
            raise NotImplementedError


def add_system_collision_pairs(
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


def remove_BVH_models(geom_model: pin.GeometryModel):
    """
    Replace Bounding Volume Hierarchies (BVH) in the geometry model with their convex hulls.
    This is a preprocessing step to ensure contact solvers (like ADMM/PGS) work stably,
    as complex non-convex BVH meshes (OBB, AABB) can cause performance or convergence issues.
    """
    for gobj in geom_model.geometryObjects:
        gobj: pin.GeometryObject
        bvh_types = [coal.BV_OBBRSS, coal.BV_OBB, coal.BV_AABB]
        ntype = gobj.geometry.getNodeType()
        if ntype in bvh_types:
            gobj.geometry.buildConvexHull(True, "Qt")
            gobj.geometry = gobj.geometry.convex


def add_material_and_compliance(
    geom_model: pin.GeometryModel, material: str, compliance: float
):
    """
    Assign physical material properties and compliance to all geometry objects.
    Determines the friction coefficients based on the material type (e.g., 'metal', 'wood', 'ice')
    and sets the compliance (softness/elasticity) to model contact behaviors.
    """
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


def setup_joint_constraints(
    model: pin.Model, joint_limit: bool, joint_friction: float, damping: float
):
    """
    Configure internal joint mechanics such as damping, dry friction, and position limits.
    Applies specific Coulomb friction limits and damping values to all non-FreeFlyer joints.
    Also drops positional constraints if `joint_limit` is set to False.
    """
    # --> joint damping and friction
    model.damping[:] = np.zeros(model.nv)
    model.upperDryFrictionLimit[:] = np.zeros(model.nv)
    model.lowerDryFrictionLimit[:] = np.zeros(model.nv)
    joint_idx = 0
    for j, joint in enumerate(model.joints):
        if model.names[j] != "universe":
            if joint.nv == 1:
                model.damping[joint_idx] = damping
            if joint.shortname() != "JointModelFreeFlyer":
                model.lowerDryFrictionLimit[joint_idx : joint_idx + joint.nv] = (
                    -joint_friction
                )
                model.upperDryFrictionLimit[joint_idx : joint_idx + joint.nv] = (
                    joint_friction
                )
            joint_idx += joint.nv

    # --> joint limits
    if not joint_limit:
        for i in range(model.nq):
            model.lowerPositionLimit[i] = np.finfo("d").min
            model.upperPositionLimit[i] = np.finfo("d").max
        # for i in range(model.nq):
        #     model.positionLimitMargin[i] = np.finfo("d").max


def print_sim_config(args: SimulationArgs):
    """
    Print all parameters of the SimulationArgs configuration object.
    Helper function to inspect the current state of simulation arguments.
    """
    print(LOGGER.INFO + "=== Simulation Configuration ===")
    for k in args.__class__.__annotations__.keys():
        if hasattr(args, k):
            print(f"\t{k}: {getattr(args, k)}")
    print(LOGGER.INFO + "==================================")


def read_sim_config(
    args: SimulationArgs, yaml_path: str, is_show: bool = False
) -> SimulationArgs:
    """
    Update a SimulationArgs configuration object using values loaded from a YAML file.
    If the configuration dictionary has corresponding fields, they are dynamically updated
    using `setattr`. Calls `process_args()` sequentially to finalize enum mappings.
    If `is_show` is True, prints the configuration after loading.
    """
    print(LOGGER.INFO + f"Loading sim config from: {yaml_path}")
    with open(yaml_path, "r") as f:
        config_dict = yaml.safe_load(f)

    if config_dict:
        for k, v in config_dict.items():
            if hasattr(args, k):
                setattr(args, k, v)
            else:
                print(LOGGER.WARNING + f"Unknown config key ignored: {k}")

    args.process_args()

    if is_show:
        print_sim_config(args)

    return args


def setup_simplex_simulator(sim: simplex.SimulatorX, args: SimulationArgs):
    """
    Inject the high-level Python configuration into the underlying C++ SimpleX simulator engine.
    Maps parameters from `SimulationArgs` into the `sim.config` struct, including solver tuning,
    absolute/relative tolerances, Baumgarte stabilization metrics, and contact patch parameters.
    """
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
    sim.config.constraint_solvers_configs.pgs_config.absolute_precision = args.tol
    sim.config.constraint_solvers_configs.pgs_config.relative_precision = args.tol_rel
    sim.config.constraint_solvers_configs.pgs_config.absolute_complementarity_tol = (
        args.tol
    )
    sim.config.constraint_solvers_configs.pgs_config.relative_complementarity_tol = (
        args.tol_rel
    )
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
    sim.config.constraint_solvers_configs.admm_config.absolute_precision = args.tol
    sim.config.constraint_solvers_configs.admm_config.relative_precision = args.tol_rel
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
    sim.config.constraint_solvers_configs.admm_config.rho_power = args.rho_power
    sim.config.constraint_solvers_configs.admm_config.rho_momentum = args.rho_momentum
    sim.config.constraint_solvers_configs.admm_config.rho_update_ratio = (
        args.rho_update_ratio
    )
    sim.config.constraint_solvers_configs.admm_config.rho_min_update_frequency = (
        args.rho_min_update_frequency
    )
    sim.config.constraint_solvers_configs.admm_config.anderson_acceleration_capacity = (
        args.anderson_acceleration_capacity
    )
    sim.config.constraint_solvers_configs.admm_config.lanczos_size = args.lanczos_size
    sim.config.constraint_solvers_configs.admm_config.max_delassus_decomposition_updates = (
        args.max_delassus_decomposition_updates
    )
    sim.config.constraint_solvers_configs.admm_config.dual_momentum = args.dual_momentum
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
