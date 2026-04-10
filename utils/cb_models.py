import pinocchio as pin
import numpy as np
from hppfcl import Plane, Sphere, Ellipsoid, Halfspace, Box, HeightFieldAABB
from pycontact.utils.pin_utils import complete_orthonormal_basis
from pycontact import ContactProblem, DelassusDense, DelassusPinocchio
from pycontact.simulators import CCPNewtonPrimalSimulator
from pathlib import Path
import os


def create_balls(length=[0.2], mass=[1.0], mu=0.9, el=0.5, comp=0.0):
    assert len(length) == len(mass) or len(length) == 1 or len(mass) == 1
    N = max(len(length), len(mass))
    if len(length) == 1:
        length = length * N
    if len(mass) == 1:
        mass = mass * N
    rmodel = pin.Model()
    rgeomModel = pin.GeometryModel()

    rgeomModel.frictions = []
    rgeomModel.compliances = []
    rgeomModel.elasticities = []
    # create plane for floor
    n = np.array([0.0, 0.0, 1])
    plane_shape = Halfspace(n, 0)
    T = pin.SE3(np.eye(3), np.zeros(3))
    plane = pin.GeometryObject("plane", 0, 0, T, plane_shape)
    plane.meshColor = np.array([0.5, 0.5, 0.5, 1.0])
    plane_id = rgeomModel.addGeometryObject(plane)
    ball_ids = []
    for n_ball in range(N):
        a = length[n_ball]
        m = mass[n_ball]
        freeflyer = pin.JointModelFreeFlyer()
        joint = rmodel.addJoint(0, freeflyer, pin.SE3.Identity(), "ball_" + str(n_ball))
        rmodel.appendBodyToJoint(
            joint, pin.Inertia.FromSphere(m, a / 2), pin.SE3.Identity()
        )
        ball_shape = Sphere(a / 2)
        geom_ball = pin.GeometryObject(
            "ball_" + str(n_ball), joint, joint, pin.SE3.Identity(), ball_shape
        )
        geom_ball.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball_id = rgeomModel.addGeometryObject(geom_ball)
        for id in ball_ids:
            col_pair = pin.CollisionPair(id, ball_id)
            rgeomModel.addCollisionPair(col_pair)
            rgeomModel.frictions += [mu]
            rgeomModel.compliances += [comp]
            rgeomModel.elasticities += [el]
        ball_ids += [ball_id]
        col_pair = pin.CollisionPair(plane_id, ball_id)
        rgeomModel.addCollisionPair(col_pair)
        rgeomModel.frictions += [mu]
        rgeomModel.compliances += [comp]
        rgeomModel.elasticities += [el]

    rmodel.qref = pin.neutral(rmodel)
    rmodel.qinit = rmodel.qref.copy()
    for n_ball in range(N):
        rmodel.qinit[7 * n_ball] += a
        rmodel.qinit[7 * n_ball + 2] += 0.1

    data = rmodel.createData()
    rgeom_data = rgeomModel.createData()
    for req in rgeom_data.collisionRequests:
        req.security_margin = 1e-3
    actuation = np.zeros((rmodel.nv, 1))
    actuation[2, 0] = 1.0
    visual_model = rgeomModel.copy()
    visual_data = visual_model.createData()
    return rmodel, rgeomModel, visual_model, data, rgeom_data, visual_data, actuation


def random_configurations_balls(model, data, geom_model, geom_data):
    valid_conf = False
    N_balls = int(model.nq / 7)
    lower_limit = np.array([-0.5, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0] * N_balls)
    upper_limit = np.array([0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0] * N_balls)
    for _ in range(100):
        q = pin.randomConfiguration(model, lower_limit, upper_limit * 1.0)
        valid_conf = True
        pin.updateGeometryPlacements(model, data, geom_model, geom_data, q)
        pin.computeCollisions(geom_model, geom_data, True)
        for res in geom_data.collisionResults:
            if res.isCollision():
                valid_conf = False
        if valid_conf:
            return q
    return model.q_init


def create_cubes(length=[0.2], mass=[1.0], mu=0.9, el=0.1, comp=0.0):
    assert len(length) == len(mass) or len(length) == 1 or len(mass) == 1
    N = max(len(length), len(mass))
    if len(length) == 1:
        length = length * N
    if len(mass) == 1:
        mass = mass * N
    rmodel = pin.Model()
    rgeomModel = pin.GeometryModel()
    rgeomModel.frictions = []
    rgeomModel.compliances = []
    rgeomModel.elasticities = []

    n = np.array([0.0, 0.0, 1])
    # plane_shape = Plane(n, 0)
    plane_shape = Halfspace(n, 0)
    T = pin.SE3(np.eye(3), np.zeros(3))
    plane = pin.GeometryObject("plane", 0, 0, T, plane_shape)
    plane.meshColor = np.array([0.5, 0.5, 0.5, 1.0])
    plane_id = rgeomModel.addGeometryObject(plane)

    ball_ids = []
    for n_cube in range(N):
        a = length[n_cube]
        m = mass[n_cube]
        freeflyer = pin.JointModelFreeFlyer()
        jointCube = rmodel.addJoint(
            0, freeflyer, pin.SE3.Identity(), "joint1_" + str(n_cube)
        )
        M = pin.SE3(np.eye(3), np.matrix([0.0, 0.0, 0.0]).T)
        rmodel.appendBodyToJoint(jointCube, pin.Inertia.FromBox(m, a, a, a), M)
        # rmodel.qref = pin.neutral(rmodel)
        # rmodel.qinit = rmodel.qref.copy()
        # rmodel.qinit[2] += 0.1
        # data = rmodel.createData()
        r = np.array([a / 4, a / 4, a / 4])

        # add balls to cube

        R = pin.utils.eye(3)
        t = np.matrix([a / 2, -a / 2, -a / 2]).T
        # ball_shape1 = Ellipsoid(r)
        ball_shape1 = Sphere(a / 50)
        geom_ball1 = pin.GeometryObject(
            "ball1_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape1
        )
        geom_ball1.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball1_id = rgeomModel.addGeometryObject(geom_ball1)
        col_pair1 = pin.CollisionPair(plane_id, ball1_id)
        rgeomModel.addCollisionPair(col_pair1)
        rgeomModel.frictions += [mu]
        rgeomModel.compliances += [comp]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([a / 2, a / 2, -a / 2]).T
        # ball_shape2 = Ellipsoid(r)
        ball_shape2 = Sphere(a / 50)
        geom_ball2 = pin.GeometryObject(
            "ball2_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape2
        )
        geom_ball2.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball2_id = rgeomModel.addGeometryObject(geom_ball2)
        col_pair2 = pin.CollisionPair(plane_id, ball2_id)
        rgeomModel.addCollisionPair(col_pair2)
        rgeomModel.frictions += [mu]
        rgeomModel.compliances += [comp]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([a / 2, a / 2, a / 2]).T
        # ball_shape3 = Ellipsoid(r)
        ball_shape3 = Sphere(a / 50)
        geom_ball3 = pin.GeometryObject(
            "ball3_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape3
        )
        geom_ball3.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball3_id = rgeomModel.addGeometryObject(geom_ball3)
        col_pair3 = pin.CollisionPair(plane_id, ball3_id)
        rgeomModel.addCollisionPair(col_pair3)
        rgeomModel.frictions += [mu]
        rgeomModel.compliances += [comp]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([a / 2, -a / 2, a / 2]).T
        # ball_shape4 = Ellipsoid(r)
        ball_shape4 = Sphere(a / 50)
        geom_ball4 = pin.GeometryObject(
            "ball4_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape4
        )
        geom_ball4.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball4_id = rgeomModel.addGeometryObject(geom_ball4)
        col_pair4 = pin.CollisionPair(plane_id, ball4_id)
        rgeomModel.addCollisionPair(col_pair4)
        rgeomModel.frictions += [mu]
        rgeomModel.compliances += [comp]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([-a / 2, -a / 2, -a / 2]).T
        # ball_shape5 = Ellipsoid(r)
        ball_shape5 = Sphere(a / 50)
        geom_ball5 = pin.GeometryObject(
            "ball5_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape5
        )
        geom_ball5.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball5_id = rgeomModel.addGeometryObject(geom_ball5)
        col_pair5 = pin.CollisionPair(plane_id, ball5_id)
        rgeomModel.addCollisionPair(col_pair5)
        rgeomModel.frictions += [mu]
        rgeomModel.compliances += [comp]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([-a / 2, a / 2, -a / 2]).T
        # ball_shape6 = Ellipsoid(r)
        ball_shape6 = Sphere(a / 50)
        geom_ball6 = pin.GeometryObject(
            "ball6_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape6
        )
        geom_ball6.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball6_id = rgeomModel.addGeometryObject(geom_ball6)
        col_pair6 = pin.CollisionPair(plane_id, ball6_id)
        rgeomModel.addCollisionPair(col_pair6)
        rgeomModel.frictions += [mu]
        rgeomModel.compliances += [comp]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([-a / 2, a / 2, a / 2]).T
        # ball_shape7 = Ellipsoid(r)
        ball_shape7 = Sphere(a / 50)
        geom_ball7 = pin.GeometryObject(
            "ball7_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape7
        )
        geom_ball7.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball7_id = rgeomModel.addGeometryObject(geom_ball7)
        col_pair7 = pin.CollisionPair(plane_id, ball7_id)
        rgeomModel.addCollisionPair(col_pair7)
        rgeomModel.frictions += [mu]
        rgeomModel.compliances += [comp]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([-a / 2, -a / 2, a / 2]).T
        # ball_shape8 = Ellipsoid(r)
        ball_shape8 = Sphere(a / 50)
        geom_ball8 = pin.GeometryObject(
            "ball8_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape8
        )
        geom_ball8.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball8_id = rgeomModel.addGeometryObject(geom_ball8)
        col_pair8 = pin.CollisionPair(plane_id, ball8_id)
        rgeomModel.addCollisionPair(col_pair8)
        rgeomModel.frictions += [mu]
        rgeomModel.compliances += [comp]
        rgeomModel.elasticities += [el]
        for id in ball_ids:
            col_pair = pin.CollisionPair(id, ball1_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball2_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball3_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball4_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball5_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball6_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball7_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball8_id)
            rgeomModel.addCollisionPair(col_pair)
            rgeomModel.frictions += [mu] * 8
            rgeomModel.compliances += [comp] * 8
            rgeomModel.elasticities += [el] * 8
        ball_ids += [
            ball1_id,
            ball2_id,
            ball3_id,
            ball4_id,
            ball5_id,
            ball6_id,
            ball7_id,
            ball8_id,
        ]

    rmodel.qref = pin.neutral(rmodel)
    rmodel.qinit = rmodel.qref.copy()
    rmodel.qinit[2] += a / 2 + a / 50
    for n_cube in range(1, N):
        a = length[n_cube]
        rmodel.qinit[7 * n_cube + 1] = rmodel.qinit[7 * (n_cube - 1) + 1] + a + 0.03
        rmodel.qinit[7 * n_cube + 2] += a / 2
    data = rmodel.createData()
    rgeom_data = rgeomModel.createData()
    for req in rgeom_data.collisionRequests:
        req.security_margin = 1e-3
    actuation = np.eye(rmodel.nv)
    visual_model = rgeomModel.copy()
    for n_cube in range(N):
        R = pin.utils.eye(3)
        t = np.matrix([0.0, 0.0, 0.0]).T
        box_shape = Box(a, a, a)
        geom_box = pin.GeometryObject(
            "box_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), box_shape
        )
        geom_box.meshColor = np.array([0.0, 0.0, 1.0, 0.6])
        box_id = visual_model.addGeometryObject(geom_box)  # only for visualisation
    visual_data = visual_model.createData()
    return (
        rmodel,
        rgeomModel,
        visual_model,
        data,
        rgeom_data,
        visual_data,
        actuation,
    )


def random_configurations(model, data, geom_model, geom_data):
    valid_conf = False
    lower_limit = np.zeros(model.nq)
    upper_limit = np.ones(model.nq)
    for _ in range(100):
        q = pin.randomConfiguration(model, lower_limit, upper_limit * 2.0)
        valid_conf = True
        pin.updateGeometryPlacements(model, data, geom_model, geom_data, q)
        pin.computeCollisions(geom_model, geom_data, True)
        for res in geom_data.collisionResults:
            if res.isCollision():
                valid_conf = False
        if valid_conf:
            return q
    return model.q_init


def addPlaneToGeomModel(
    geom_model,
    n=np.array([0.0, 0.0, 1.0]),
    center=np.zeros(3),
    mu=0.9,
    el=0.0,
    comp=0.0,
    visual=False,
):  # adding a plane to the current model
    ex, ey = complete_orthonormal_basis(n)
    R = np.array([ex, ey, n]).T
    plane_shape = Plane(n, 0)
    T = pin.SE3(R, center)
    n_plane = geom_model.ngeoms
    plane = pin.GeometryObject("plane" + str(n_plane), 0, 0, T, plane_shape)
    plane.meshColor = np.array([0.5, 0.5, 0.5, 1.0])
    plane_id = geom_model.addGeometryObject(plane)
    if not visual:
        # adding collision pairs
        for id in range(len(geom_model.geometryObjects) - 1):
            # TODO avoid plane/plane collision detection
            if not isinstance(geom_model.geometryObjects[id].geometry, Plane):
                col_pair = pin.CollisionPair(id, plane_id)
                geom_model.addCollisionPair(col_pair)
                geom_model.frictions += [mu]
                geom_model.compliances += [comp]
                geom_model.elasticities += [el]
    return geom_model


def addBallToGeomModel(geom_model, joint, n_ball, a, mu, el, comp, visual=False):
    ball_shape = Sphere(a / 2)
    geom_ball = pin.GeometryObject(
        "ball_" + str(n_ball), joint, joint, pin.SE3.Identity(), ball_shape
    )
    geom_ball.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball_id = geom_model.addGeometryObject(geom_ball)
    if not visual:
        for id in range(len(geom_model.geometryObjects) - 1):
            col_pair = pin.CollisionPair(id, ball_id)
            geom_model.addCollisionPair(col_pair)
            geom_model.frictions += [mu]
            geom_model.compliances += [comp]
            geom_model.elasticities += [el]
    return geom_model


def addCubeToGeomModel(
    geom_model,
    jointCube,
    n_cube,
    a,
    mu,
    el,
    comp,
    color=np.array([0.0, 0.0, 1.0, 0.6]),
    visual=False,
    geom_with_box=False,
):
    R = pin.utils.eye(3)
    t = np.matrix([a / 2, -a / 2, -a / 2]).T
    ball_shape1 = Sphere(a / 50)
    geom_ball1 = pin.GeometryObject(
        "ball1_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape1
    )
    geom_ball1.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball1_id = geom_model.addGeometryObject(geom_ball1)

    R = pin.utils.eye(3)
    t = np.matrix([a / 2, a / 2, -a / 2]).T
    ball_shape2 = Sphere(a / 50)
    geom_ball2 = pin.GeometryObject(
        "ball2_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape2
    )
    geom_ball2.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball2_id = geom_model.addGeometryObject(geom_ball2)

    R = pin.utils.eye(3)
    t = np.matrix([a / 2, a / 2, a / 2]).T
    ball_shape3 = Sphere(a / 50)
    geom_ball3 = pin.GeometryObject(
        "ball3_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape3
    )
    geom_ball3.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball3_id = geom_model.addGeometryObject(geom_ball3)

    R = pin.utils.eye(3)
    t = np.matrix([a / 2, -a / 2, a / 2]).T
    ball_shape4 = Sphere(a / 50)
    geom_ball4 = pin.GeometryObject(
        "ball4_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape4
    )
    geom_ball4.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball4_id = geom_model.addGeometryObject(geom_ball4)

    R = pin.utils.eye(3)
    t = np.matrix([-a / 2, -a / 2, -a / 2]).T
    ball_shape5 = Sphere(a / 50)
    geom_ball5 = pin.GeometryObject(
        "ball5_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape5
    )
    geom_ball5.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball5_id = geom_model.addGeometryObject(geom_ball5)

    R = pin.utils.eye(3)
    t = np.matrix([-a / 2, a / 2, -a / 2]).T
    ball_shape6 = Sphere(a / 50)
    geom_ball6 = pin.GeometryObject(
        "ball6_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape6
    )
    geom_ball6.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball6_id = geom_model.addGeometryObject(geom_ball6)

    R = pin.utils.eye(3)
    t = np.matrix([-a / 2, a / 2, a / 2]).T
    ball_shape7 = Sphere(a / 50)
    geom_ball7 = pin.GeometryObject(
        "ball7_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape7
    )
    geom_ball7.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball7_id = geom_model.addGeometryObject(geom_ball7)

    R = pin.utils.eye(3)
    t = np.matrix([-a / 2, -a / 2, a / 2]).T
    ball_shape8 = Sphere(a / 50)
    geom_ball8 = pin.GeometryObject(
        "ball8_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape8
    )
    geom_ball8.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball8_id = geom_model.addGeometryObject(geom_ball8)
    if visual or geom_with_box:
        R = pin.utils.eye(3)
        t = np.matrix([0.0, 0.0, 0.0]).T
        box_shape = Box(a, a, a)
        geom_box = pin.GeometryObject(
            "box_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), box_shape
        )
        geom_box.meshColor = color
        box_id = geom_model.addGeometryObject(geom_box)  # only for visualisation
    if not visual:
        n_self_collide = 9 if geom_with_box else 8
        for id in range(len(geom_model.geometryObjects) - n_self_collide):
            col_pair = pin.CollisionPair(id, ball1_id)
            geom_model.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball2_id)
            geom_model.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball3_id)
            geom_model.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball4_id)
            geom_model.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball5_id)
            geom_model.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball6_id)
            geom_model.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball7_id)
            geom_model.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball8_id)
            geom_model.addCollisionPair(col_pair)
            if geom_with_box:
                col_pair = pin.CollisionPair(id, box_id)
                geom_model.addCollisionPair(col_pair)
            geom_model.frictions += [mu] * 8
            geom_model.compliances += [comp] * 8
            geom_model.elasticities += [el] * 8
    return geom_model


def addCubeToGeomModelFull(
    geom_model, jointCube, n_cube, a, mu, el, comp, color=np.array([0.0, 0.0, 1.0, 0.6])
):
    # Add the cube itself
    M = pin.SE3.Identity()
    box_shape = Box(a, a, a)
    geom_box = pin.GeometryObject(
        "box_" + str(n_cube), jointCube, jointCube, M, box_shape
    )
    geom_box.meshColor = color
    box_id = geom_model.addGeometryObject(geom_box)

    # Add balls for each corner of the cube
    R = pin.utils.eye(3)
    t = np.matrix([a / 2, -a / 2, -a / 2]).T
    ball_shape1 = Sphere(a / 50)
    geom_ball1 = pin.GeometryObject(
        "ball1_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape1
    )
    geom_ball1.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball1_id = geom_model.addGeometryObject(geom_ball1)

    R = pin.utils.eye(3)
    t = np.matrix([a / 2, a / 2, -a / 2]).T
    ball_shape2 = Sphere(a / 50)
    geom_ball2 = pin.GeometryObject(
        "ball2_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape2
    )
    geom_ball2.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball2_id = geom_model.addGeometryObject(geom_ball2)

    R = pin.utils.eye(3)
    t = np.matrix([a / 2, a / 2, a / 2]).T
    ball_shape3 = Sphere(a / 50)
    geom_ball3 = pin.GeometryObject(
        "ball3_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape3
    )
    geom_ball3.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball3_id = geom_model.addGeometryObject(geom_ball3)

    R = pin.utils.eye(3)
    t = np.matrix([a / 2, -a / 2, a / 2]).T
    ball_shape4 = Sphere(a / 50)
    geom_ball4 = pin.GeometryObject(
        "ball4_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape4
    )
    geom_ball4.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball4_id = geom_model.addGeometryObject(geom_ball4)

    R = pin.utils.eye(3)
    t = np.matrix([-a / 2, -a / 2, -a / 2]).T
    ball_shape5 = Sphere(a / 50)
    geom_ball5 = pin.GeometryObject(
        "ball5_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape5
    )
    geom_ball5.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball5_id = geom_model.addGeometryObject(geom_ball5)

    R = pin.utils.eye(3)
    t = np.matrix([-a / 2, a / 2, -a / 2]).T
    ball_shape6 = Sphere(a / 50)
    geom_ball6 = pin.GeometryObject(
        "ball6_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape6
    )
    geom_ball6.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball6_id = geom_model.addGeometryObject(geom_ball6)

    R = pin.utils.eye(3)
    t = np.matrix([-a / 2, a / 2, a / 2]).T
    ball_shape7 = Sphere(a / 50)
    geom_ball7 = pin.GeometryObject(
        "ball7_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape7
    )
    geom_ball7.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball7_id = geom_model.addGeometryObject(geom_ball7)

    R = pin.utils.eye(3)
    t = np.matrix([-a / 2, -a / 2, a / 2]).T
    ball_shape8 = Sphere(a / 50)
    geom_ball8 = pin.GeometryObject(
        "ball8_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape8
    )
    geom_ball8.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball8_id = geom_model.addGeometryObject(geom_ball8)

    # Add collision pairs:
    for id in range(len(geom_model.geometryObjects) - (8 + 1)):
        col_pair = pin.CollisionPair(id, box_id)
        geom_model.addCollisionPair(col_pair)
        #
        col_pair = pin.CollisionPair(id, ball1_id)
        geom_model.addCollisionPair(col_pair)
        #
        col_pair = pin.CollisionPair(id, ball2_id)
        geom_model.addCollisionPair(col_pair)
        #
        col_pair = pin.CollisionPair(id, ball3_id)
        geom_model.addCollisionPair(col_pair)
        #
        col_pair = pin.CollisionPair(id, ball4_id)
        geom_model.addCollisionPair(col_pair)
        #
        col_pair = pin.CollisionPair(id, ball5_id)
        geom_model.addCollisionPair(col_pair)
        #
        col_pair = pin.CollisionPair(id, ball6_id)
        geom_model.addCollisionPair(col_pair)
        #
        col_pair = pin.CollisionPair(id, ball7_id)
        geom_model.addCollisionPair(col_pair)
        #
        col_pair = pin.CollisionPair(id, ball8_id)
        geom_model.addCollisionPair(col_pair)
        #
        geom_model.frictions += [mu] * (8 + 1)
        geom_model.compliances += [comp] * (8 + 1)
        geom_model.elasticities += [el] * (8 + 1)
    return geom_model


def addCube(
    model,
    geom_model,
    visual_model,
    actuation,
    a,
    m,
    mu,
    el,
    comp,
    color=np.array([0.0, 0.0, 1.0, 0.6]),
    actuated=False,
    geom_with_box=False,
):  # adding a cube to the current model
    freeflyer = pin.JointModelFreeFlyer()
    n_cube = model.nbodies
    jointCube = model.addJoint(
        0, freeflyer, pin.SE3.Identity(), "joint1_" + str(n_cube)
    )
    M = pin.SE3(np.eye(3), np.matrix([0.0, 0.0, 0.0]).T)
    model.appendBodyToJoint(jointCube, pin.Inertia.FromBox(m, 0.2, 0.2, 0.2), M)

    # add balls to cube
    geom_model = addCubeToGeomModel(
        geom_model, jointCube, n_cube, a, mu, el, comp, geom_with_box=geom_with_box
    )
    visual_model = addCubeToGeomModel(
        visual_model, jointCube, n_cube, a, mu, el, comp, color, True
    )
    data = model.createData()
    geom_data = geom_model.createData()
    for req in geom_data.collisionRequests:
        req.security_margin = 1e-3
    visual_data = visual_model.createData()
    actuation_pred = actuation.copy()
    if actuated:
        actuation = np.zeros((model.nv, actuation_pred.shape[1] + 6))
        actuation[-6:, -6:] = np.eye(6)
    else:
        actuation = np.zeros((model.nv, actuation_pred.shape[1]))
    actuation[: actuation_pred.shape[0], : actuation_pred.shape[1]] = actuation_pred
    model.qinit = np.concatenate(
        [model.qinit, np.array([0.0, 2 * a, 2 * a, 0.0, 0.0, 0.0, 1.0])]
    )
    model.qref = np.concatenate(
        [model.qref, np.array([0.0, 2 * a, 2 * a, 0.0, 0.0, 0.0, 1.0])]
    )
    return model, geom_model, visual_model, data, geom_data, visual_data, actuation


def addBall(
    model, geom_model, visual_model, actuation, a, m, mu, el, comp, actuated=False
):  # adding a ball to the current model
    freeflyer = pin.JointModelFreeFlyer()
    n_ball = model.nbodies
    joint = model.addJoint(0, freeflyer, pin.SE3.Identity(), "ball_" + str(n_ball))
    model.appendBodyToJoint(joint, pin.Inertia.FromSphere(m, a / 2), pin.SE3.Identity())
    geom_model = addBallToGeomModel(geom_model, joint, n_ball, a, mu, el, comp)
    visual_model = addBallToGeomModel(
        visual_model, joint, n_ball, a, mu, el, comp, True
    )
    data = model.createData()
    geom_data = geom_model.createData()
    for req in geom_data.collisionRequests:
        req.security_margin = 1e-3
    visual_data = visual_model.createData()
    actuation_pred = actuation.copy()
    if actuated:
        actuation = np.zeros((model.nv, actuation_pred.shape[1] + 6))
        actuation[-6:, -6:] = np.eye(6)
    else:
        actuation = np.zeros((model.nv, actuation_pred.shape[1]))
    actuation[: actuation_pred.shape[0], : actuation_pred.shape[1]] = actuation_pred
    model.qinit = np.concatenate(
        [model.qinit, np.array([0.0, 2 * a, 2 * a, 0.0, 0.0, 0.0, 1.0])]
    )
    model.qref = np.concatenate(
        [model.qref, np.array([0.0, 2 * a, 2 * a, 0.0, 0.0, 0.0, 1.0])]
    )
    return model, geom_model, visual_model, data, geom_data, visual_data, actuation


def addRandomObject(model, geom_model, visual_model, actuation):
    obj_type = np.random.randint(2)
    if obj_type == 0:
        a = 0.2 + 0.1 * np.random.rand()
        m = 1.0 + 0.1 * np.random.rand()
        mu = 0.9 + 0.1 * np.random.rand()
        el = 0.2 + 0.1 * np.random.rand()
        comp = 0.0
        (
            model,
            geom_model,
            visual_model,
            data,
            geom_data,
            visual_data,
            actuation,
        ) = addBall(model, geom_model, visual_model, actuation, a, m, mu, el, comp)
    elif obj_type == 1:
        a = 0.2 + 0.1 * np.random.rand()
        m = 1.0 + 0.1 * np.random.rand()
        mu = 0.9 + 0.1 * np.random.rand()
        el = 0.2 + 0.1 * np.random.rand()
        comp = 0.0
        (
            model,
            geom_model,
            visual_model,
            data,
            geom_data,
            visual_data,
            actuation,
        ) = addCube(model, geom_model, visual_model, actuation, a, m, mu, el, comp)
    elif obj_type == 2 and False:
        mu = 0.9 + 0.1 * np.random.rand()
        el = 0.0 + 0.1 * np.random.rand()
        (
            model,
            geom_model,
            visual_model,
            data,
            geom_data,
            visual_data,
            actuation,
        ) = addSolo(model, geom_model, visual_model, actuation, mu, el, comp)
    return model, geom_model, visual_model, data, geom_data, visual_data, actuation


def create_random_scene(N=1):
    """Randomly creates a scene with various objects.

    Args:
        N (int): number of objects
    """
    a = 0.2 + 0.1 * np.random.rand()
    m = 1.0 + 0.1 * np.random.rand()
    mu = 0.9 + 0.1 * np.random.rand()
    el = 0.2 + 0.1 * np.random.rand()
    (
        model,
        geom_model,
        visual_model,
        data,
        geom_data,
        visual_data,
        actuation,
    ) = create_balls([a], [m], mu, el)
    for i in range(N - 1):
        (
            model,
            geom_model,
            visual_model,
            data,
            geom_data,
            visual_data,
            actuation,
        ) = addRandomObject(model, geom_model, visual_model, actuation)
    return model, geom_model, visual_model, data, geom_data, visual_data, actuation


def build_cube_problem(dense=False, drag=False, slide=False, ccp_reg=False):
    a = 0.2  # size of cube
    m = 1.0  # mass of cube
    mu = 0.95  # friction parameter
    eps = 0.0  # elasticity
    model, geom_model, visual_model, data, geom_data, visual_data, actuation = (
        create_cubes([a], [m], mu, eps)
    )
    # time steps
    dt = 1e-3

    # Physical parameters of the contact problem
    Kb = 1e-4 * 0.0  # Baumgarte

    # initial state
    q0 = model.qinit
    v0 = np.zeros(model.nv)
    if slide:
        v0[1] = 3.0
    q, v = q0.copy(), v0.copy()
    tau = np.zeros(model.nv)
    fext = [pin.Force(np.zeros(6)) for i in range(model.njoints)]
    # simulator
    simulator = CCPNewtonPrimalSimulator(
        model, geom_model, data, geom_data, regularize=ccp_reg, warm_start=False
    )
    simulator.step(model, data, geom_model, geom_data, q, v, tau, fext, dt, Kb, 1, 1e-7)
    Del = simulator.Del
    g = simulator.g
    mus = simulator.mus
    M = simulator.M
    J = simulator.J
    vstar = simulator.vstar
    dqf = simulator.dqf
    if dense:
        Del.evaluateDel()
        prob = ContactProblem(Del.G_, g, M, J, dqf, vstar, mus)
    else:
        prob = ContactProblem(Del, g, M, J, dqf, vstar, mus)
    return prob


def addBallsToGo2(rgeomModel):
    """
    在 Go2 的足端添加球体几何对象。
    通常用于替代原始的复杂足端网格，以获得更稳定的点接触模拟。
    """
    ball_radius = 0.025  # Go2 足端半径约为 2cm

    # Go2 的足端名称通常包含这些关键字
    # 在 example_robot_data 中，几何体名字通常就是 Link 名字
    foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    # 颜色：红色，半透明
    ball_color = np.array([1.0, 0.2, 0.2, 0.8])
    shape = Sphere(ball_radius)

    # 我们需要遍历现有的几何体来找到足端的父关节和位姿
    # 这里通过名字匹配来寻找原始的足端几何体
    for i, geom in enumerate(rgeomModel.geometryObjects):
        for foot_name in foot_names:
            # 匹配几何体名称 (ERD中通常是 "FL_foot_0", "FL_foot_1" 等，或者是直接 "FL_foot")
            if foot_name in geom.name:
                parent_joint = geom.parentJoint
                placement = geom.placement.copy()

                # 创建新的球体几何对象
                # 位置通常就在原始足端几何体的原点 (因为 Go2 的 foot frame 就在球心)
                # 如果需要偏移，可以修改 placement.translation
                shape_name = f"{foot_name}_ball"
                ball_geometry = pin.GeometryObject(
                    shape_name, parent_joint, placement, shape
                )
                ball_geometry.meshColor = ball_color

                # 添加到模型中
                rgeomModel.addGeometryObject(ball_geometry)

                # 找到一个后跳出内层循环，防止重复添加（如果一个足端有多个几何体）
                # 这里假设每个足端主要由一个核心几何体代表
                break

    return rgeomModel


# def create_quadruped_robot(mu=0.9, el=0.0, comp=0., reduced=False, add_balls=False):
#     """
#     创建一个包含 Go2 机器人和地面的完整仿真环境。
#     """
#     # --- 1. 加载机器人模型 ---
#     robot = erd.load("go2")
#     rmodel = robot.model.copy()

#     # 设置参考姿态 (站立)
#     rmodel.qref = robot.q0.copy()
#     rmodel.qinit = robot.q0.copy()

#     # 稍微抬高一点，避免初始时刻嵌入地面
#     # Go2 站立高度大约 0.3m 左右，q0 通常已经包含了合理高度，这里微调确保安全
#     rmodel.qinit[2] += 0.02

#     # --- 2. 获取几何模型 ---
#     rgeomModel = robot.collision_model.copy()
#     visual_model = robot.visual_model.copy()

#     # --- 3. (可选) 添加足端球体 ---
#     # 如果添加了球体，我们通常希望使用球体进行碰撞，而不是原始网格
#     if add_balls:
#         rgeomModel = addBallsToGo2(rgeomModel)

#     # --- 4. (可选) 缩减模型 ---
#     if reduced:
#         # 锁定所有关节，只保留浮动基座 (通常用于调试)
#         # Go2 有 12 个驱动关节，索引通常从 1 (Universe) -> 2 (Root) 之后开始
#         # 这里的范围需要根据实际关节数调整，Go2 njoints=14 (Universe+Root+12)
#         joints_to_lock = [i for i in range(2, rmodel.njoints)]
#         model_red, geom_visual_models = pin.buildReducedModel(
#             rmodel, [rgeomModel, visual_model], joints_to_lock, rmodel.qref
#         )
#         geom_model_red, visual_model_red = geom_visual_models[0], geom_visual_models[1]

#         model_red.qref = rmodel.qref[:7].copy()
#         model_red.qinit = rmodel.qinit[:7].copy()

#         rmodel = model_red.copy()
#         rmodel.qref, rmodel.qinit = model_red.qref, model_red.qinit
#         rgeomModel = geom_model_red.copy()
#         visual_model = visual_model_red.copy()

#     # --- 5. 添加地面 (Plane) ---
#     n = np.array([0.0, 0.0, 1]) # 法向量
#     plane_shape = Halfspace(n, 0)
#     T = pin.SE3(np.eye(3), np.zeros(3))
#     plane = pin.GeometryObject("plane", 0, 0, T, plane_shape)
#     plane.meshColor = np.array([0.5, 0.5, 0.5, 1.0])

#     # 添加地面并记录其 ID
#     plane_id = rgeomModel.addGeometryObject(plane)

#     # --- 6. 设置碰撞对与物理属性 ---
#     rgeomModel.removeAllCollisionPairs()

#     # 初始化自定义物理属性列表
#     rgeomModel.frictions = []
#     rgeomModel.compliances = []
#     rgeomModel.elasticities = []

#     # 识别原始足端几何体的 ID，如果启用了 add_balls，我们通常想忽略原始网格
#     original_foot_ids = []
#     if add_balls:
#         foot_keywords = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
#         for i, geom in enumerate(rgeomModel.geometryObjects):
#             # 如果名字包含 foot 但不包含 ball (那是我们新加的)，则视为原始网格
#             if any(k in geom.name for k in foot_keywords) and "ball" not in geom.name:
#                 original_foot_ids.append(i)

#     # 遍历所有几何体 (除了最后一个是 Plane)
#     for id in range(len(rgeomModel.geometryObjects) - 1):
#         # 如果启用了球体碰撞，跳过原始的足端网格，只让球体与地面碰撞
#         if add_balls and id in original_foot_ids:
#             continue

#         # 建立与地面的碰撞对
#         col_pair = pin.CollisionPair(id, plane_id)
#         rgeomModel.addCollisionPair(col_pair)

#         # 添加物理属性
#         rgeomModel.frictions += [mu]
#         rgeomModel.compliances += [comp]
#         rgeomModel.elasticities += [el]

#     # --- 7. 创建数据对象 ---
#     rdata = rmodel.createData()
#     rgeom_data = rgeomModel.createData()
#     visual_data = visual_model.createData()

#     # 设置碰撞检测的安全边距
#     for req in rgeom_data.collisionRequests:
#         req.security_margin = 1e-3

#     # --- 8. 创建驱动矩阵 ---
#     # 浮动基座：nv = 18 (6 + 12)
#     # 前 6 维 (基座) 为 0，后 12 维为 Identity
#     actuation = np.zeros((rmodel.nv, rmodel.nv - 6))
#     actuation[6:, :] = np.eye(rmodel.nv - 6)

#     return rmodel, rgeomModel, visual_model, rdata, rgeom_data, visual_data, actuation
