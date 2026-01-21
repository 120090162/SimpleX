import hppfcl
import numpy as np
import meshcat
import meshcat.geometry as mg
import pinocchio as pin
import warnings

from pinocchio.visualize import MeshcatVisualizer
from pinocchio.visualize.meshcat_visualizer import MsgType
from typing import Any
from simplex_sandbox.utils.viz_common import BLACK


def sub_sample(xs, duration, fps):
    nb_frames = len(xs)
    nb_subframes = int(duration * fps)
    if nb_frames < nb_subframes:
        return xs
    else:
        step = nb_frames // nb_subframes
        xs_sub = [xs[i] for i in range(0, nb_frames, step)]
        return xs_sub


def rgbToHex(color):
    if len(color) == 4:
        c = color[:3]
        opacity = color[3]
    else:
        c = color
        opacity = 1.0
    hex_color = "0x%02x%02x%02x" % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
    return hex_color, opacity


def register_object(
    viz: MeshcatVisualizer,
    shape: hppfcl.ShapeBase,
    shape_name: str,
    M: pin.SE3,
    shape_color=np.ones(4),
) -> int:
    meshcat_shape = load_primitive(shape)
    if isinstance(shape, (hppfcl.Plane, hppfcl.Halfspace)):
        T = M.copy()
        T.translation += M.rotation @ (shape.d * shape.n)
        T = T.homogeneous
    else:
        T = M.homogeneous

    # Update viewer configuration.
    viz.viewer[shape_name].set_object(meshcat_shape, meshcat_material(*shape_color))
    viz.viewer[shape_name].set_transform(T)


def register_line(
    viz: MeshcatVisualizer,
    pt1: np.ndarray,
    pt2: np.ndarray,
    line_name: str,
    linewidth: float = 0.01,
    color: np.ndarray = BLACK,
) -> int:
    height = np.linalg.norm(pt2 - pt1)
    if height > 1e-6:
        axis_ref = np.array([0.0, 0.0, 1.0])
        axis = (pt2 - pt1) / height  # - np.array([0., 0., 1.])
        num = np.outer(axis_ref + axis, axis_ref + axis)
        den = np.dot(axis_ref + axis, axis_ref + axis)
        if den > 1e-6:
            R = 2 * num / den - np.eye(3)
        else:
            R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        t = pt1
        translation = pin.SE3(np.eye(3), np.array([0.0, 0.0, height / 2]))
        M = pin.SE3(R, t)
        Mtot = M * translation
        cylinder = hppfcl.Cylinder(linewidth, height)
        meshcat_shape = load_primitive(cylinder)
        viz.viewer[line_name].set_object(meshcat_shape, meshcat_material(*color))
        viz.viewer[line_name].set_transform(Mtot.homogeneous)


def register_arrowed_line(
    viz: MeshcatVisualizer,
    pt1: np.ndarray,
    pt2: np.ndarray,
    line_name: str,
    linewidth: float = 0.01,
    color: np.ndarray = BLACK,
) -> int:
    height = np.linalg.norm(pt2 - pt1)
    if height > 1e-6:
        register_line(viz, pt1, pt2, line_name, linewidth, color)
        arrow: hppfcl.Convex = create_arrow_head(4 * linewidth)
        M = pin.SE3(np.eye(3), np.array([0.0, 0.0, height / 2]))
        arrow_shape = load_primitive(arrow)
        arrow_name = line_name + "/arrow"
        viz.viewer[arrow_name].set_object(arrow_shape, meshcat_material(*color))
        viz.viewer[arrow_name].set_transform(M.homogeneous)


def transform_object(
    viz: MeshcatVisualizer, shape: hppfcl.ShapeBase, shape_name: str, M: pin.SE3
) -> None:
    if isinstance(shape, (hppfcl.Plane, hppfcl.Halfspace)):
        T = M.copy()
        T.translation += M.rotation @ (shape.d * shape.n)
        T = T.homogeneous
    else:
        T = M.homogeneous

    # Update viewer configuration.
    viz.viewer[shape_name].set_transform(T)
    return


def delete_object(viz: MeshcatVisualizer, name: str):
    viz.viewer[name].delete()


TWOPI = 2 * np.pi


def create_arrow_head(scale_: float = 1.0, n: int = 10) -> hppfcl.Convex:
    scale = scale_ / 2
    pts = hppfcl.StdVec_Vec3f()
    assert n > 3
    center = np.zeros(3)
    for i in range(n):
        pt = scale * np.array([np.cos(TWOPI * i / n), np.sin(TWOPI * i / n), 0.0])
        center += pt
        pts.append(pt)
    pts.append(center)
    pts.append(scale * np.array([0.0, 0.0, 2.0]))

    tris = hppfcl.StdVec_Triangle()
    for i in range(n):
        # Base triangle
        tris.append(hppfcl.Triangle(i, n, (i + 1) % n))
        # Side triangle
        tris.append(hppfcl.Triangle(i, (i + 1) % n, n + 1))

    cvx = hppfcl.Convex(pts, tris)
    return cvx


def npToTTuple(M):
    L = M.tolist()
    for i in range(len(L)):
        L[i] = tuple(L[i])
    return tuple(L)


def npToTuple(M):
    if len(M.shape) == 1:
        return tuple(M.tolist())
    if M.shape[0] == 1:
        return tuple(M.tolist()[0])
    if M.shape[1] == 1:
        return tuple(M.T.tolist()[0])
    return npToTTuple(M)


def load_primitive(geom: hppfcl.ShapeBase):
    import meshcat.geometry as mg

    # Cylinders need to be rotated
    basic_three_js_transform = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    RotatedCylinder = type(
        "RotatedCylinder",
        (mg.Cylinder,),
        {"intrinsic_transform": lambda self: basic_three_js_transform},
    )

    # Cones need to be rotated

    if isinstance(geom, hppfcl.Capsule):
        if hasattr(mg, "TriangularMeshGeometry"):
            obj = createCapsule(2.0 * geom.halfLength, geom.radius)
        else:
            obj = RotatedCylinder(2.0 * geom.halfLength, geom.radius)
    elif isinstance(geom, hppfcl.Cylinder):
        obj = RotatedCylinder(2.0 * geom.halfLength, geom.radius)
    elif isinstance(geom, hppfcl.Cone):
        obj = RotatedCylinder(2.0 * geom.halfLength, 0, geom.radius, 0)
    elif isinstance(geom, hppfcl.Box):
        obj = mg.Box(npToTuple(2.0 * geom.halfSide))
    elif isinstance(geom, hppfcl.Sphere):
        obj = mg.Sphere(geom.radius)
    elif isinstance(geom, hppfcl.TriangleP):
        faces = np.empty((1, 3), dtype=int)
        vertices = np.empty((3, 3))
        vertices[0] = geom.a
        vertices[1] = geom.b
        vertices[2] = geom.c
        faces[0] = [0, 1, 2]
        obj = mg.TriangularMeshGeometry(vertices, faces)
    elif isinstance(geom, hppfcl.Ellipsoid):
        obj = mg.Ellipsoid(geom.radii)
    elif isinstance(geom, (hppfcl.Plane, hppfcl.Halfspace)):
        plane_transform: pin.SE3 = pin.SE3.Identity()
        # plane_transform.translation[:] = geom.d # Does not work
        plane_transform.rotation = pin.Quaternion.FromTwoVectors(
            geom.n, pin.ZAxis
        ).toRotationMatrix()
        TransformedPlane = type(
            "TransformedPlane",
            (Plane,),
            {"intrinsic_transform": lambda self: plane_transform.homogeneous},
        )
        obj = TransformedPlane(10, 10)
    elif isinstance(geom, hppfcl.ConvexBase):
        obj = loadMesh(geom)
    else:
        msg = "Unsupported geometry type for (%s)" % (type(geom))
        warnings.warn(msg, category=UserWarning, stacklevel=2)
        obj = None

    return obj


def loadMesh(mesh):
    if isinstance(mesh, (hppfcl.Convex, hppfcl.BVHModelBase)):
        if isinstance(mesh, hppfcl.BVHModelBase):
            num_vertices = mesh.num_vertices
            num_tris = mesh.num_tris

            call_triangles = mesh.tri_indices
            call_vertices = mesh.vertices

        elif isinstance(mesh, hppfcl.Convex):
            num_vertices = mesh.num_points
            num_tris = mesh.num_polygons

            call_triangles = mesh.polygons
            call_vertices = mesh.points

        faces = np.empty((num_tris, 3), dtype=int)
        for k in range(num_tris):
            tri = call_triangles(k)
            faces[k] = [tri[i] for i in range(3)]

        vertices = call_vertices()

        vertices = vertices.astype(np.float32)

    if num_tris > 0:
        mesh = mg.TriangularMeshGeometry(vertices, faces)
    else:
        mesh = mg.Points(
            mg.PointsGeometry(
                vertices.T, color=np.repeat(np.ones((3, 1)), num_vertices, axis=1)
            ),
            mg.PointsMaterial(size=0.002),
        )

    return mesh


def createCapsule(length, radius, radial_resolution=30, cap_resolution=10):
    nbv = np.array([max(radial_resolution, 4), max(cap_resolution, 4)])
    h = length
    r = radius
    position = 0
    vertices = np.zeros((nbv[0] * (2 * nbv[1]) + 2, 3))
    for j in range(nbv[0]):
        phi = (2 * np.pi * j) / nbv[0]
        for i in range(nbv[1]):
            theta = (np.pi / 2 * i) / nbv[1]
            vertices[position + i, :] = np.array(
                [
                    np.cos(theta) * np.cos(phi) * r,
                    np.cos(theta) * np.sin(phi) * r,
                    -h / 2 - np.sin(theta) * r,
                ]
            )
            vertices[position + i + nbv[1], :] = np.array(
                [
                    np.cos(theta) * np.cos(phi) * r,
                    np.cos(theta) * np.sin(phi) * r,
                    h / 2 + np.sin(theta) * r,
                ]
            )
        position += nbv[1] * 2
    vertices[-2, :] = np.array([0, 0, -h / 2 - r])
    vertices[-1, :] = np.array([0, 0, h / 2 + r])
    indexes = np.zeros((nbv[0] * (4 * (nbv[1] - 1) + 4), 3))
    index = 0
    stride = nbv[1] * 2
    last = nbv[0] * (2 * nbv[1]) + 1
    for j in range(nbv[0]):
        j_next = (j + 1) % nbv[0]
        indexes[index + 0] = np.array(
            [j_next * stride + nbv[1], j_next * stride, j * stride]
        )
        indexes[index + 1] = np.array(
            [j * stride + nbv[1], j_next * stride + nbv[1], j * stride]
        )
        indexes[index + 2] = np.array(
            [j * stride + nbv[1] - 1, j_next * stride + nbv[1] - 1, last - 1]
        )
        indexes[index + 3] = np.array(
            [j_next * stride + 2 * nbv[1] - 1, j * stride + 2 * nbv[1] - 1, last]
        )
        for i in range(nbv[1] - 1):
            indexes[index + 4 + i * 4 + 0] = np.array(
                [j_next * stride + i, j_next * stride + i + 1, j * stride + i]
            )
            indexes[index + 4 + i * 4 + 1] = np.array(
                [j_next * stride + i + 1, j * stride + i + 1, j * stride + i]
            )
            indexes[index + 4 + i * 4 + 2] = np.array(
                [
                    j_next * stride + nbv[1] + i + 1,
                    j_next * stride + nbv[1] + i,
                    j * stride + nbv[1] + i,
                ]
            )
            indexes[index + 4 + i * 4 + 3] = np.array(
                [
                    j_next * stride + nbv[1] + i + 1,
                    j * stride + nbv[1] + i,
                    j * stride + nbv[1] + i + 1,
                ]
            )
        index += 4 * (nbv[1] - 1) + 4
    return mg.TriangularMeshGeometry(vertices, indexes)


class Plane(mg.Geometry):
    """A plane of the given width and height."""

    def __init__(
        self,
        width: float,
        height: float,
        widthSegments: float = 1,
        heightSegments: float = 1,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.widthSegments = widthSegments
        self.heightSegments = heightSegments

    def lower(self, object_data: Any) -> MsgType:
        return {
            "uuid": self.uuid,
            "type": "PlaneGeometry",
            "width": self.width,
            "height": self.height,
            "widthSegments": self.widthSegments,
            "heightSegments": self.heightSegments,
        }


def meshcat_material(r, g, b, a):
    material = mg.MeshPhongMaterial()
    material.color = int(r * 255) * 256**2 + int(g * 255) * 256 + int(b * 255)
    material.opacity = a
    return material


def createVisualizer(
    model: pin.GeometryModel,
    geom_model: pin.GeometryModel,
    visual_model: pin.GeometryModel,
    zmq_url: str = "tcp://127.0.0.1:6000",
    spotlight_pos: np.ndarray = np.array([0, 0, 3]),
    positive_x_pos: np.ndarray = np.array([2, 0, 3]),
    negative_x_pos: np.ndarray = np.array([-2, 2, 3]),
):
    viewer = meshcat.Visualizer(zmq_url=zmq_url)
    viewer.delete()
    # for obj in visual_model.geometryObjects:
    #     if obj.name != "floor":
    #         color = np.random.rand(4)
    #         color[3] = 1.0
    #         obj.meshColor = color
    # --> background
    viewer["/Background"].set_property("top_color", BLACK[:3].tolist())
    viewer["/Background"].set_property("bottom_color", BLACK[:3].tolist())
    # --> spotlight
    viewer["/Lights/SpotLight"].set_property("visible", True)
    viewer["/Lights/SpotLight/<object>"].set_property(
        "position", spotlight_pos.tolist()
    )
    viewer["/Lights/SpotLight/<object>"].set_property("castShadow", True)
    viewer["/Lights/SpotLight/<object>"].set_property("intensity", 10)
    # --> point light positive x
    viewer["/Lights/PointLightPositiveX/<object>"].set_property(
        "position", positive_x_pos.tolist()
    )
    viewer["/Lights/PointLightPositiveX/<object>"].set_property("castShadow", True)
    viewer["/Lights/PointLightPositiveX/<object>"].set_property("intensity", 10)
    # --> point light negative x
    viewer["/Lights/PointLightNegativeX/<object>"].set_property(
        "position", negative_x_pos.tolist()
    )
    viewer["/Lights/PointLightNegativeX/<object>"].set_property("castShadow", True)
    viewer["/Lights/PointLightNegativeX/<object>"].set_property("intensity", 0)

    vizer: MeshcatVisualizer = MeshcatVisualizer(model, geom_model, visual_model)
    vizer.initViewer(viewer=viewer, open=False, loadModel=True)
    return vizer, viewer


def subSample(xs: np.ndarray, duration: float, fps: float):
    nb_frames = len(xs)
    nb_subframes = int(duration * fps)
    if nb_frames < nb_subframes:
        return xs, [i for i in range(0, len(xs))]
    else:
        step = nb_frames // nb_subframes
        xs_sub = [xs[i] for i in range(0, nb_frames, step)]
        ts = [i for i in range(0, nb_frames, step)]
        return xs_sub, ts
