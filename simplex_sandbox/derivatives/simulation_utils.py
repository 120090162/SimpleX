import time
from typing import Dict
import numpy as np
import tap
from viz_utils import RED, GREEN, BLUE, BLACK, PINK, GREY, BEIGE, PURPLE
from viz_utils import (
    register_line,
    register_arrowed_line,
    register_object,
    transform_object,
    sub_sample,
)
import hppfcl
import pinocchio as pin
import simple
import matplotlib.pyplot as plt
from simulation_args import SimulationArgs


class Policy:
    def __init__(self):
        pass

    def act(
        self, simulator: simple.Simulator, q: np.ndarray, v: np.ndarray, dt
    ) -> np.ndarray:
        pass


class DefaultPolicy(Policy):
    def __init__(self, model: pin.Model):
        self.actuation = None

    def act(
        self, simulator: simple.Simulator, q: np.ndarray, v: np.ndarray, dt
    ) -> np.ndarray:
        return np.zeros(simulator.model.nv)


class FreeFloatingRobotDampingPolicy(Policy):
    def __init__(self, model: pin.Model, damping_factor: float):
        self.actuation = np.zeros((model.nv, model.nv - 6))
        self.actuation[6:, :] = np.eye(model.nv - 6)
        self.damping_factor = damping_factor

    def act(
        self, simulator: simple.Simulator, q: np.ndarray, v: np.ndarray, dt
    ) -> np.ndarray:
        # Note: simulator and model should coincide
        tau_act = -self.damping_factor * v[6:]
        return self.actuation @ tau_act


class RobotArmDampingPolicy(Policy):
    def __init__(self, model: pin.Model, damping_factor: float):
        self.actuation = np.eye(model.nv)
        self.damping_factor = damping_factor

    def act(
        self, simulator: simple.Simulator, q: np.ndarray, v: np.ndarray, dt
    ) -> np.ndarray:
        # Note: simulator and model should coincide
        tau_act = -self.damping_factor * v
        return self.actuation @ tau_act


def setPhysicsProperties(
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

        # Compliance
        gobj.physicsMaterial.compliance = compliance


def removeBVHModelsIfAny(geom_model: pin.GeometryModel):
    for gobj in geom_model.geometryObjects:
        gobj: pin.GeometryObject
        bvh_types = [hppfcl.BV_OBBRSS, hppfcl.BV_OBB, hppfcl.BV_AABB]
        ntype = gobj.geometry.getNodeType()
        if ntype in bvh_types:
            gobj.geometry.buildConvexHull(True, "Qt")
            gobj.geometry = gobj.geometry.convex


def addFloor(geom_model: pin.GeometryModel, visual_model: pin.GeometryModel):
    color = GREY
    color[3] = 0.5
    # Collision object
    # floor_collision_shape = hppfcl.Box(10, 10, 2)
    # M = pin.SE3(np.eye(3), np.zeros(3))
    # M.translation = np.array([0.0, 0.0, -(1.99 / 2.0)])
    floor_collision_shape = hppfcl.Halfspace(0, 0, 1, 0)
    # floor_collision_shape = hppfcl.Plane(0, 0, 1, 0)
    # floor_collision_shape.setSweptSphereRadius(0.5)
    M = pin.SE3.Identity()
    floor_collision_object = pin.GeometryObject("floor", 0, 0, M, floor_collision_shape)
    floor_collision_object.meshColor = color
    geom_model.addGeometryObject(floor_collision_object)

    # Visual object
    floor_visual_shape = hppfcl.Box(10, 10, 0.01)
    floor_visual_object = pin.GeometryObject(
        "floor", 0, 0, pin.SE3.Identity(), floor_visual_shape
    )
    floor_visual_object.meshColor = color
    visual_model.addGeometryObject(floor_visual_object)


def simulateSytem(
    model: pin.Model,
    geom_model: pin.GeometryModel,
    visual_model: pin.GeometryModel,
    q0: np.ndarray,
    v0: np.ndarray,
    policy: Policy,
    args: Dict,
):
    print(f"Number of bodies in model = {model.nbodies}")
    print("Num geom obj = ", len(geom_model.geometryObjects))
    if args.debug:
        for i, inertia in enumerate(model.inertias):
            print("------------------------")
            print(f"Inertia {i} =\n {inertia}")

        for gobj in geom_model.geometryObjects:
            print(
                f"name = {gobj.name}, {gobj.geometry.getNodeType()}, parent joint = {gobj.parentJoint}"
            )

    if args.display:
        from pinocchio.visualize import MeshcatVisualizer
        import meshcat

        if args.debug:
            if args.display_collision_model:
                rendered_model = geom_model
            else:
                rendered_model = visual_model
            for gobj in rendered_model.geometryObjects:
                color = gobj.meshColor
                if gobj.name == "floor":
                    color[3] = 0.2
                else:
                    color[3] = args.debug_transparency
                gobj.meshColor = color

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
        if args.display_collision_model:
            vizer.displayCollisions(True)
            vizer.displayVisuals(False)
        else:
            vizer.displayCollisions(False)
            vizer.displayVisuals(True)

    data = model.createData()
    geom_data = geom_model.createData()

    com = pin.centerOfMass(model, data, q0)
    print("Center of mass = ", com)
    if args.display and args.debug and args.display_com:
        sphere_com = hppfcl.Sphere(0.05)
        Mcom = pin.SE3.Identity()
        Mcom.translation = com
        register_object(vizer, sphere_com, "com", Mcom, PURPLE)

    print(f"System total mass = {pin.computeTotalMass(model, data)}")
    print(f"Armature: {model.armature}")

    for col_req in geom_data.collisionRequests:
        col_req: hppfcl.CollisionRequest
        col_req.security_margin = 0.0
        col_req.break_distance = 0.0
        col_req.gjk_tolerance = 1e-6
        col_req.epa_tolerance = 1e-6
        col_req.gjk_initial_guess = hppfcl.GJKInitialGuess.CachedGuess
        col_req.gjk_variant = hppfcl.GJKVariant.DefaultGJK

    for patch_req in geom_data.contactPatchRequests:
        patch_req.setPatchTolerance(args.patch_tolerance)

    # Simulation parameters
    if args.contact_solver == "ADMM" or args.contact_solver == "PGS":
        simulator = simple.Simulator(model, data, geom_model, geom_data)
        # PGS
        simulator.pgs_constraint_solver_settings.absolute_precision = args.tol
        simulator.pgs_constraint_solver_settings.relative_precision = args.tol_rel
        simulator.pgs_constraint_solver_settings.max_iter = args.maxit
        # ADMM
        simulator.admm_constraint_solver_settings.absolute_precision = args.tol
        simulator.admm_constraint_solver_settings.relative_precision = args.tol_rel
        simulator.admm_constraint_solver_settings.max_iter = args.maxit
        simulator.admm_constraint_solver_settings.mu = args.mu_prox
        #
        simulator.warm_start_constraints_forces = args.warm_start
        simulator.measure_timings = True
        # Contact patch settings
        simulator.constraints_problem.setMaxNumberOfContactsPerCollisionPair(
            args.max_patch_size
        )
        # Baumgarte settings
        simulator.constraints_problem.Kp = args.Kp
        simulator.constraints_problem.Kd = args.Kd
        if args.admm_update_rule == "spectral":
            simulator.admm_constraint_solver_settings.admm_update_rule = (
                pin.ADMMUpdateRule.SPECTRAL
            )
        elif args.admm_update_rule == "linear":
            simulator.admm_constraint_solver_settings.admm_update_rule = (
                pin.ADMMUpdateRule.LINEAR
            )
        else:
            print(f"ERROR - no match for admm update rule {args.admm_update_rule}")
            exit(1)
    dt = args.dt
    T = args.horizon

    print(
        f"[Main simulation will be repeated {args.num_repetitions} times to gather timings]"
    )

    if args.display:
        input("[Press enter to simulate.]")
    step_timings = 0
    contact_solver_timings = 0
    for _ in range(args.num_repetitions):
        q = q0.copy()
        v = v0.copy()
        tau = np.zeros(model.nv)
        fext = [pin.Force(np.zeros(6)) for _ in range(model.njoints)]
        simulator.resetBeforeLooping()
        for t in range(T):
            tau = policy.act(simulator, q, v, dt)
            if args.contact_solver == "ADMM":
                simulator.step(q, v, tau, fext, dt)
            else:
                simulator.stepPGS(q, v, tau, fext, dt)
            if args.debug and t % args.debug_step == 0:
                if args.display:
                    vizer.display(simulator.qnew)
                if args.display_com:
                    com = pin.centerOfMass(model, data, simulator.qnew)
                    Mcom = pin.SE3.Identity()
                    Mcom.translation = com
                    transform_object(vizer, sphere_com, "com", Mcom)

                print(f"\n========== TIMESTEP {t} ===========")
                num_contacts = simulator.constraints_problem.getNumberOfContacts()
                print(f"===> Num contact points = {num_contacts}")
                print(
                    f"===> Timings of contact solver = { simulator.getContactSolverCPUTimes().user } us"
                )
                print(
                    f"===> Timings of step function = { simulator.getStepCPUTimes().user } us"
                )
                if args.display_state:
                    print(f"===> Joint config [q] =\n {q}")
                    print(f"===> Joint vel [v] =\n {v}")
                    print(f"    ----> v.norm() = {np.linalg.norm(v)}")
                    print(f"===> Joint torque [tau] =\n {tau}")
                    print(f"===> Free joint vel [vfree] =\n {simulator.vfree}")
                    print(f"===> Updated joint config [qnew] =\n {simulator.qnew}")
                    print(f"===> Updated joint vel [vnew] =\n {simulator.vnew}")
                    print(f"    ----> vnew.norm() = {np.linalg.norm(simulator.vnew)}")
                    frictional_point_constraints_forces = simulator.constraints_problem.frictional_point_constraints_forces()
                    print(
                        f"===> Contact forces [frictional_point_constraints_forces()] =\n {frictional_point_constraints_forces}"
                    )
                    print(
                        f"    ----> frictional_point_constraints_forces().norm() = {np.linalg.norm(frictional_point_constraints_forces)}"
                    )
                    print("===> Total forces (external + contact):")
                    for i in range(model.njoints):
                        print(
                            f"    ---> {model.names[i]}, ftotal =\n{simulator.ftotal[i]}"
                        )
                if args.display_contacts:
                    vizer.viewer["contact_info"].delete()
                    print("===> Contact information:")
                    for i in range(num_contacts):
                        # Display contact point
                        cp = simulator.constraints_problem.contact_points[i]
                        sphere = hppfcl.Sphere(0.01)
                        cp_name = f"contact_info/contact_point_{i}"
                        M = pin.SE3.Identity()
                        M.translation = cp
                        register_object(vizer, sphere, cp_name, M, BLACK)

                        # Display contact force
                        print(
                            f"    --> cp {i} = ",
                            cp,
                        )
                        normal = simulator.constraints_problem.contact_normals[i]
                        print(
                            f"    --> normal {i} = ",
                            normal,
                        )
                        frictional_point_constraints_forces = simulator.constraints_problem.frictional_point_constraints_forces()
                        fcontact = frictional_point_constraints_forces[
                            3 * i : 3 * i + 3
                        ]
                        print(
                            f"    --> contact force {i} = {fcontact}",
                        )
                        constraint_model: pin.RigidConstraintModel = (
                            simulator.constraints_problem.getConstraintModel(i)
                        )
                        joint1_id = constraint_model.joint1_id
                        i1Mc = constraint_model.joint1_placement
                        wMc = data.oMi[joint1_id].act(i1Mc)
                        spatial_force_loc = pin.Force(fcontact, np.zeros(3))
                        spatial_force: pin.Force = wMc.act(spatial_force_loc)
                        # To give a nice visual of the contact force, I'll assume the following:
                        # 1Kg of force (~10 Newtons) is 0.1 meters -> so we need to divide the contact
                        # force by 100.
                        visual_factor = 1e-2
                        # Note: the - sign is because the normal goes from body 1 to body 2, but we
                        # want to view the force exerted by body 2 on body 1.
                        # new_cp = cp - normal * fcontact[2] * visual_factor
                        new_cp = cp - spatial_force.linear * visual_factor
                        force_arrow_name = f"contact_info/contact_force_{i}"
                        register_arrowed_line(
                            vizer, cp, new_cp, force_arrow_name, 0.005, RED
                        )
                if args.display_step:
                    # Re-print solver info, nice for debugging
                    print(f"===> Num contact points = {num_contacts}")
                    print(
                        f"===> Timings of contact solver = { simulator.getContactSolverCPUTimes().user } us"
                    )
                    print(
                        f"===> Timings of step function = { simulator.getStepCPUTimes().user } us"
                    )
                    input(f"[Timestep {t} - press enter to continue.]")

            # Update simulator state
            step_timings += simulator.getStepCPUTimes().user
            contact_solver_timings += simulator.getContactSolverCPUTimes().user
            q = simulator.qnew.copy()
            v = simulator.vnew.copy()
    step_timings *= 1e-6  # convert micro seconds to seconds
    print("============================================")
    print("SIMULATION")
    print("Time elapsed: ", step_timings)
    print("Mean timings time step: ", step_timings / (T * args.num_repetitions))
    print("Steps per second: ", (T * args.num_repetitions) / (step_timings))
    print(
        "Mean timings contact solver: ",
        contact_solver_timings / (T * args.num_repetitions),
    )
    print("============================================")

    if args.display:
        # remove contact info if any
        vizer.viewer["contact_info"].delete()

        # recompute and store trajectory
        print("[Recomputing trajectory for displaying it...]")
        q, v = q0.copy(), v0.copy()
        fext = [pin.Force(np.zeros(6)) for _ in range(model.njoints)]
        tau = np.zeros(model.nv)
        qs, vs = [q], [v]
        numit = []
        step_timings = []
        contact_solver_timings = []
        vnorm = []
        contact_forces_norm = []
        num_contacts = []
        mechanical_energy = []
        potential_energy = []
        kinetic_energy = []
        simulator.resetBeforeLooping()
        for t in range(T):
            tau = policy.act(simulator, q, v, dt)
            if args.contact_solver == "ADMM":
                simulator.step(q, v, tau, fext, dt)
            else:
                simulator.stepPGS(q, v, tau, fext, dt)
            q = simulator.qnew.copy()
            v = simulator.vnew.copy()
            #
            # Save trajectory for display
            qs.append(q.copy())
            vs.append(v.copy())
            #
            # Save metrics
            vnorm.append(np.linalg.norm(v))
            nc = simulator.constraints_problem.getNumberOfContacts()
            num_contacts.append(nc)
            contact_forces = simulator.constraints_problem.contact_forces()
            contact_forces_norm.append(np.linalg.norm(contact_forces))
            numit.append(simulator.admm_constraint_solver.getIterationCount())
            step_timings.append(simulator.getStepCPUTimes().user)
            contact_solver_timings.append(simulator.getContactSolverCPUTimes().user)
            mechanical_energy.append(pin.computeMechanicalEnergy(model, data, q, v))
            potential_energy.append(pin.computePotentialEnergy(model, data, q))
            kinetic_energy.append(pin.computeKineticEnergy(model, data, q, v))

        if args.plot_metrics:
            plt.figure()
            plt.plot(vs, label=[f"Joint {i}" for i in range(model.nv)])
            plt.xlabel("Timestep")
            plt.ylabel("Joint velocities")
            plt.legend()
            plt.ion()
            plt.show()

            plt.figure()
            plt.plot(mechanical_energy, label="Mechanical energy")
            plt.plot(potential_energy, label="Potential energy")
            plt.plot(kinetic_energy, label="Kinetic energy")
            plt.xlabel("Timestep")
            plt.ylabel("Energy")
            plt.legend()
            plt.ion()
            plt.show()

            fontsize = 12
            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
            if args.plot_hist:
                ax[0, 0].hist(
                    numit,
                    bins=args.maxit // 2,
                    align="mid",
                    color=PURPLE,
                    density=True,
                    histtype="bar",
                    cumulative=False,
                    linewidth=3,
                    rwidth=0.8,
                )
                ax[0, 0].set_xlabel("# of iterations", fontsize=fontsize)
                ax[0, 0].set_ylabel("Ratio of problems", fontsize=fontsize)
                ax[0, 0].set_title(
                    f"{args.contact_solver} solver iterations",
                    fontsize=fontsize,
                )
            else:
                # Number of iterations along trajectory
                ax[0, 0].plot(numit, "+", color=PURPLE, linewidth=1)
                ax[0, 0].set_xlabel("Timestep", fontsize=fontsize)
                ax[0, 0].set_ylabel("# of iterations", fontsize=fontsize)
                ax[0, 0].set_title(
                    f"{args.contact_solver} solver iterations along trajectory",
                    fontsize=fontsize,
                )

            if args.plot_hist:
                # Distribution of number of contact points
                num_contacts = np.array(num_contacts, dtype=np.int32)
                ax[0, 1].hist(
                    num_contacts,
                    align="mid",
                    color=PURPLE,
                    density=True,
                    histtype="bar",
                    cumulative=False,
                    linewidth=3,
                    rwidth=0.8,
                )
                ax[0, 1].set_xlabel("# contact points", fontsize=fontsize)
                ax[0, 1].set_ylabel("Ratio of problems", fontsize=fontsize)
                ax[0, 1].set_title(
                    "Distribution # of contact points",
                    fontsize=fontsize,
                )
            else:
                # Number of contacts along trajectory
                ax[0, 1].plot(num_contacts, "+", color=PURPLE, linewidth=1)
                ax[0, 1].set_xlabel("Timestep", fontsize=fontsize)
                ax[0, 1].set_ylabel("# of contact points", fontsize=fontsize)
                ax[0, 1].set_title(
                    f"Number of contact points along trajectory",
                    fontsize=fontsize,
                )

            # Contact solver timings
            ax[0, 2].plot(contact_solver_timings, "+", linewidth=3, color=PURPLE)
            ax[0, 2].set_xlabel("Timestep", fontsize=fontsize)
            ax[0, 2].set_ylabel("Contact solver timings", fontsize=fontsize)
            ax[0, 2].set_title(
                "Contact solver timings along trajectory",
                fontsize=fontsize,
            )

            # Joint velocity
            ax[1, 0].plot(vnorm, linewidth=3, color=PURPLE)
            ax[1, 0].set_xlabel("Timestep", fontsize=fontsize)
            ax[1, 0].set_ylabel("Joint vel norm", fontsize=fontsize)
            ax[1, 0].set_title(
                "Joint velocity norm along trajectory",
                fontsize=fontsize,
            )

            # Contact forces
            ax[1, 1].plot(contact_forces_norm, "+", linewidth=1, color=PURPLE)
            ax[1, 1].set_xlabel("Timestep", fontsize=fontsize)
            ax[1, 1].set_ylabel("Contact forces norm", fontsize=fontsize)
            ax[1, 1].set_yscale("log")
            ax[1, 1].set_title(
                "Contact forces norm along trajectory",
                fontsize=fontsize,
            )

            # Step timings
            ax[1, 2].plot(step_timings, "+", linewidth=3, color=PURPLE)
            ax[1, 2].set_xlabel("Timestep", fontsize=fontsize)
            ax[1, 2].set_ylabel("`Step` timings", fontsize=fontsize)
            ax[1, 2].set_title(
                "`Step` timings along trajectory",
                fontsize=fontsize,
            )

            plt.suptitle(
                f"{args.plot_title}\ntol = {args.tol}, maxit = {args.maxit}, dt = {args.dt}, horizon = {args.horizon}"
            )
            plt.ion()
            plt.show()

        max_fps = args.max_fps
        fps = min([max_fps, 1.0 / dt])
        dt_vis = 1.0 / float(fps)
        qs = sub_sample(qs, dt * T, fps)
        vizer.display(qs[0])
        input("[Press enter to display simulated trajectory]")
        while True:
            for t in range(len(qs)):
                step_start = time.time()
                vizer.display(qs[t])
                time_until_next_step = dt_vis - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
