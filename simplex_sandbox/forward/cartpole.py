import pinocchio as pin
import numpy as np
import hppfcl
from simplex_sandbox.utils.sim_utils import SimulationArgs, runSimpleXSimulationFromModel

GREY = np.array([192, 201, 229, 255]) / 255


class ScriptArgs(SimulationArgs):
    floor: bool = False  # add a floor
    display: bool = True  # display the simulation
    # display_step: int = 1  # display every n steps
    display_traj: bool = True  # display the trajectory
    pd_controller: bool = False  # use a PD controller
    random_init_vel: bool = True  # random initial velocity


args = ScriptArgs().parse_args()
np.random.seed(args.seed)
pin.seed(args.seed)


def createCartpole(N, add_floor):
    model = pin.Model()
    geom_model = pin.GeometryModel()

    if add_floor:
        # add floor
        floor_collision_shape = hppfcl.Halfspace(0, 0, 1, 0)
        M = pin.SE3.Identity()
        floor_collision_object = pin.GeometryObject(
            "floor", 0, 0, M, floor_collision_shape
        )
        color = GREY
        color[3] = 0.5
        floor_collision_object.meshColor = color
        geom_floor = geom_model.addGeometryObject(floor_collision_object)

    parent_id = 0

    cart_radius = 0.1
    cart_length = 5 * cart_radius
    cart_mass = 1.0
    joint_name = "joint_cart"

    geometry_placement = pin.SE3.Identity()
    geometry_placement.rotation = pin.Quaternion(
        np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
    ).toRotationMatrix()

    joint_id = model.addJoint(
        parent_id, pin.JointModelPY(), pin.SE3.Identity(), joint_name
    )

    body_inertia = pin.Inertia.FromCylinder(cart_mass, cart_radius, cart_length)
    body_placement = geometry_placement
    model.appendBodyToJoint(
        joint_id, body_inertia, body_placement
    )  # We need to rotate the inertia as it is expressed in the LOCAL frame of the geometry

    shape_cart = hppfcl.Cylinder(cart_radius, cart_length)

    geom_cart = pin.GeometryObject(
        "shape_cart", joint_id, geometry_placement, shape_cart
    )
    geom_cart.meshColor = np.array([1.0, 0.1, 0.1, 1.0])
    geom_model.addGeometryObject(geom_cart)

    parent_id = joint_id
    joint_placement = pin.SE3.Identity()
    body_mass = 1.0
    body_radius = 0.1
    for k in range(N):
        joint_name = "joint_" + str(k + 1)
        joint_id = model.addJoint(
            parent_id, pin.JointModelRX(), joint_placement, joint_name
        )

        body_inertia = pin.Inertia.FromSphere(body_mass, body_radius)
        body_placement = joint_placement.copy()
        body_placement.translation[2] = 1.0
        model.appendBodyToJoint(joint_id, body_inertia, body_placement)

        geom1_name = "ball_" + str(k + 1)
        shape1 = hppfcl.Sphere(body_radius)
        geom1_obj = pin.GeometryObject(geom1_name, joint_id, body_placement, shape1)
        geom1_obj.meshColor = np.ones((4))
        geom_ball = geom_model.addGeometryObject(geom1_obj)
        if add_floor:
            geom_model.addCollisionPair(pin.CollisionPair(geom_floor, geom_ball))

        geom2_name = "bar_" + str(k + 1)
        shape2 = hppfcl.Cylinder(body_radius / 4.0, body_placement.translation[2])
        shape2_placement = body_placement.copy()
        shape2_placement.translation[2] /= 2.0

        geom2_obj = pin.GeometryObject(geom2_name, joint_id, shape2_placement, shape2)
        geom2_obj.meshColor = np.array([0.0, 0.0, 0.0, 1.0])
        geom_model.addGeometryObject(geom2_obj)

        # update parent id to add next pendulum
        parent_id = joint_id
        joint_placement = body_placement.copy()

    end_frame = pin.Frame(
        "end_effector_frame",
        model.getJointId("joint_" + str(N)),
        0,
        body_placement,
        pin.FrameType(3),
    )
    model.addFrame(end_frame)
    geom_model.collision_pairs = []
    model.qinit = np.zeros(model.nq)
    model.qinit[1] = 0.0 * np.pi
    model.qref = pin.neutral(model)
    return model, geom_model


# ============================================================================
# SCENE CREATION
# ============================================================================
# Create model
model, geom_model = createCartpole(1, args.floor)
visual_model = geom_model.copy()

# Initial state
q0 = pin.neutral(model)
if args.random_init_vel:
    v0 = np.random.randn(model.nv)
else:
    v0 = np.zeros(model.nv)

if args.pd_controller:
    args.pd_controller_qtar = q0
    args.pd_controller_vtar = np.zeros(model.nv)

runSimpleXSimulationFromModel(
    model, geom_model, visual_model, q0, v0, args, add_floor=False
)
