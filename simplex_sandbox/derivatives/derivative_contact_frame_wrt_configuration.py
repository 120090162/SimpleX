import numpy as np
import pinocchio as pin
import hppfcl
from viz_utils import RED, GREEN
from pinocchio.visualize import MeshcatVisualizer
import meshcat
from contact_frame_derivatives import (
    constructContactFrame,
    contactFrameDerivativeWrtConfiguration,
    getContactJacobian,
    contactVelocityDerivativeWrtConfiguration,
    contactForceDerivativeWrtConfiguration,
)

np.set_printoptions(suppress=True)
seed = 1
np.random.seed(seed)
pin.seed(seed)

# Create a model with a sphere and a hspace.
# This is the simplest collision pair and we are 100% sure diffcoal gives the right derivatives.
model = pin.Model()
gmodel = pin.GeometryModel()

freeflyer = pin.JointModelFreeFlyer()

# TODO: test with random parent placements when defining joint and geom placements

# Adding sphere
joint_sphere = model.addJoint(0, freeflyer, pin.SE3.Identity(), "joint_sphere")
model.appendBodyToJoint(
    joint_sphere, pin.Inertia.FromSphere(0.1, 0.1), pin.SE3.Identity()
)
sphere = hppfcl.Ellipsoid(0.1, 0.2, 0.3)
geom_sphere = pin.GeometryObject(
    "sphere", joint_sphere, joint_sphere, pin.SE3.Identity(), sphere
)
geom_sphere.meshColor = RED
gmodel.addGeometryObject(geom_sphere)

# Adding hspace
joint_hspace = model.addJoint(0, freeflyer, pin.SE3.Identity(), "joint_hspace")
model.appendBodyToJoint(
    joint_hspace, pin.Inertia.FromSphere(0.1, 0.1), pin.SE3.Identity()
)
hspace = hppfcl.Halfspace(0.0, 0.0, 1.0, 0.0)
geom_hspace = pin.GeometryObject(
    "hspace", joint_hspace, joint_hspace, pin.SE3.Identity(), hspace
)
geom_hspace.meshColor = GREEN
gmodel.addGeometryObject(geom_hspace)

# Add the collision pair
gmodel.addAllCollisionPairs()
print("Number of collision pairs = ", len(gmodel.collisionPairs))
assert len(gmodel.collisionPairs) > 0
col_pair_id = 0

data = model.createData()
gdata = gmodel.createData()

for creq in gdata.collisionRequests:
    creq.security_margin = 10000  # make sure there is a collision

model.lowerPositionLimit = -np.ones((model.nq, 1))
model.upperPositionLimit = np.ones((model.nq, 1))
q = pin.randomConfiguration(model)

# visualize the pose of the objects
viewer = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
viewer.delete()
vizer: MeshcatVisualizer = MeshcatVisualizer(model, gmodel, gmodel)
vizer.initViewer(viewer=viewer, open=False, loadModel=True)
vizer.display(q)

# Collision detection
pin.computeCollisions(model, data, gmodel, gdata, q)
assert len(gdata.collisionResults) == 1
cres = gdata.collisionResults[0]
assert cres.isCollision()
contact: hppfcl.Contact = cres.getContact(0)
print(f"Normal = {contact.normal}")
print(f"Distance = {contact.penetration_depth}")
print(f"Position = {contact.pos}")


# ============================================================================
# Testing doMc/dq, where oMc is the contact frame w.r.t WORLD
# ============================================================================
# Comparing analytic vs finite differences
# -- Analytic
pin.computeCollisions(model, data, gmodel, gdata, q)
pin.computeJointJacobians(model, data)
Mc = constructContactFrame(gmodel, gdata, col_pair_id)
J = contactFrameDerivativeWrtConfiguration(model, data, gmodel, gdata, col_pair_id, Mc)

# -- Finite derivatives
Jfd = np.zeros((6, model.nv))
eps = 1e-6
for i in range(model.nv):
    dq = np.zeros(model.nv)
    dq[i] = eps

    qplus = pin.integrate(model, q, dq)
    pin.computeCollisions(model, data, gmodel, gdata, qplus)
    Mplus = constructContactFrame(gmodel, gdata, col_pair_id)

    qminus = pin.integrate(model, q, -dq)
    pin.computeCollisions(model, data, gmodel, gdata, qminus)
    Mminus = constructContactFrame(gmodel, gdata, col_pair_id)
    dM = pin.log6(Mminus.inverse() * Mplus)
    Jfd[:, i] = dM / (2 * eps)

print("\n")
print(f"Jfd = \n{Jfd}")
print(f"J = \n{J}")
print(f"Diff = {np.linalg.norm(J - Jfd)}")
assert np.linalg.norm(J - Jfd) <= 1e-5


# ============================================================================
# Testing dJc*v/dq, where Jc is the contact jacobian and v is random velocity
# ============================================================================


# Comparing analytic vs finite differences
# -- Analytic
pin.computeCollisions(model, data, gmodel, gdata, q)
pin.computeJointJacobians(model, data)
Mc = constructContactFrame(gmodel, gdata, col_pair_id)
Jc = getContactJacobian(model, data, gmodel, gdata, col_pair_id, Mc)
v = 2 * (np.random.rand(model.nv) - 0.5)
J = contactVelocityDerivativeWrtConfiguration(
    model, data, gmodel, gdata, col_pair_id, Jc, Mc, v
)

# -- Finite differences
Jfd = np.zeros((6, model.nv))
eps = 1e-6
for i in range(model.nv):
    dq = np.zeros(model.nv)
    dq[i] = eps

    qplus = pin.integrate(model, q, dq)
    pin.computeCollisions(model, data, gmodel, gdata, qplus)
    pin.computeJointJacobians(model, data)
    Mcplus = constructContactFrame(gmodel, gdata, col_pair_id)
    Jcplus = getContactJacobian(model, data, gmodel, gdata, col_pair_id, Mcplus)

    qminus = pin.integrate(model, q, -dq)
    pin.computeCollisions(model, data, gmodel, gdata, qminus)
    pin.computeJointJacobians(model, data)
    Mcminus = constructContactFrame(gmodel, gdata, col_pair_id)
    Jcminus = getContactJacobian(model, data, gmodel, gdata, col_pair_id, Mcminus)
    Jfd[:, i] = ((Jcplus - Jcminus) @ v) / (2 * eps)

print("\n")
print(f"Jfd = \n{Jfd}")
print(f"J = \n{J}")
print(f"Diff = {np.linalg.norm(J - Jfd)}")
assert np.linalg.norm(J - Jfd) <= 1e-5


# ============================================================================
# Testing dJcT*lam/dq, where Jc is the contact jacobian and v is random velocity
# ============================================================================

# print("\n---------- COADJOINT STUFF ---------------")
# a = np.random.rand(6)
# b = np.random.rand(6)
# mb = pin.Motion(b)
# fa = pin.Force(a)
# coswap = computeCoadjointSwap(fa)
# print("diff = ",  coswap @ b - (mb.action).transpose() @ a)
# print()
# print("------- MOTION -------")
# print("motion = ", b)
# print("twist = \n", mb)
# print("twist linear = \n", mb.linear)
# print("twist angular = \n", mb.angular)
# print("skew linear = \n", pin.skew(mb.linear))
# print("skew angular = \n", pin.skew(mb.angular))
# print("ad = \n", mb.action)
# print("ad transpose = \n", (mb.action).transpose())
# print()
# print("------- FORCE -------")
# print("force = ", a)
# print("wrench = \n", fa)
# print("wrench linear = ", fa.linear)
# print("wrench angular = ", fa.angular)
# print("skew linear = \n", pin.skew(fa.linear))
# print("skew angular = \n", pin.skew(fa.angular))
# print("coswap = \n", coswap)
# print()
# print("------ COMPARISION ----------")
# print("coswap * motion = \n", coswap @ b)
# print("adT * force = \n", (mb.action).transpose() @ a)
# input()


lam = 2 * (np.random.rand(6) - 0.5)
pin.computeCollisions(model, data, gmodel, gdata, q)
pin.computeJointJacobians(model, data)
Mc = constructContactFrame(gmodel, gdata, col_pair_id)
Jc = getContactJacobian(model, data, gmodel, gdata, col_pair_id, Mc)
J = contactForceDerivativeWrtConfiguration(
    model, data, gmodel, gdata, col_pair_id, Jc, Mc, lam
)

# -- Finite differences
Jfd = np.zeros((model.nv, model.nv))
eps = 1e-6
for i in range(model.nv):
    dq = np.zeros(model.nv)
    dq[i] = eps

    qplus = pin.integrate(model, q, dq)
    pin.computeCollisions(model, data, gmodel, gdata, qplus)
    pin.computeJointJacobians(model, data)
    Mcplus = constructContactFrame(gmodel, gdata, col_pair_id)
    Jcplus = getContactJacobian(model, data, gmodel, gdata, col_pair_id, Mcplus)

    qminus = pin.integrate(model, q, -dq)
    pin.computeCollisions(model, data, gmodel, gdata, qminus)
    pin.computeJointJacobians(model, data)
    Mcminus = constructContactFrame(gmodel, gdata, col_pair_id)
    Jcminus = getContactJacobian(model, data, gmodel, gdata, col_pair_id, Mcminus)
    Jfd[:, i] = (Jcplus.transpose() @ lam - Jcminus.transpose() @ lam) / (2 * eps)

print("\n")
print(f"Jfd = \n{Jfd}")
print(f"J = \n{J}")
print(f"Diff = {np.linalg.norm(J - Jfd)}")
assert np.linalg.norm(J - Jfd) <= 1e-5
