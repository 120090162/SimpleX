import numpy as np
import pinocchio as pin
import hppfcl
import pydiffcoal as dcoal
import simple


# WARNING: this function assumes `pin.computeCollisions` was called before on configuration `q`!
# WARNING: this function assumes that there exists at least a contact between the shapes involved in the collision pair!
def constructContactFrame(
    gmodel: pin.GeometryModel,
    gdata: pin.GeometryData,
    col_pair: int,
):
    """
    This function computes oMc from the `col_pair`-th collision pair of `gmodel`.
    """
    # Assumes that there is only one collision pair in gmodel
    assert col_pair <= len(gmodel.collisionPairs)
    cres: hppfcl.CollisionResult = gdata.collisionResults[col_pair]
    assert cres.isCollision()
    contact: hppfcl.Contact = cres.getContact(0)
    oMc = simple.placementFromNormalAndPosition(contact.normal, contact.pos)
    return oMc


def contactFrameDerivativeWrtConfiguration(
    model: pin.Model,
    data: pin.Data,
    gmodel: pin.GeometryModel,
    gdata: pin.GeometryData,
    col_pair: int,
    oMc: pin.SE3,
):
    """
    This function computes doMc/dq.
    oMc is a contact frame between geometries of the `col_pair`-th collision pair of `gmodel`.

    WARNING: this function assumes `pin.computeCollisions` and `pin.computeJointJacobians` were called before on a configuration `q`! This configuration must be the same that was used to compute the contact frame!
    WARNING: this function assumes that there exists at least a contact between the shapes involved in the collision pair!
    """
    # Contact frame computation derivatives
    assert col_pair <= len(gmodel.collisionPairs)
    dM_dn, dM_dp = simple.placementFromNormalAndPositionDerivative(oMc)

    geom1_id = gmodel.collisionPairs[col_pair].first
    geom2_id = gmodel.collisionPairs[col_pair].second
    geom1: pin.GeometryObject = gmodel.geometryObjects[geom1_id]
    geom2: pin.GeometryObject = gmodel.geometryObjects[geom2_id]
    shape1: hppfcl.CollisionGeometry = geom1.geometry
    shape2: hppfcl.CollisionGeometry = geom2.geometry
    oMg1: pin.SE3 = gdata.oMg[geom1_id]
    oMg2: pin.SE3 = gdata.oMg[geom2_id]
    cres: hppfcl.CollisionResult = gdata.collisionResults[col_pair]
    assert cres.isCollision()
    contact: hppfcl.Contact = cres.getContact(0)
    dreq = dcoal.ContactDerivativeRequest()
    dcontact = dcoal.ContactDerivative()

    # Collision detection derivatives
    dcoal.computeContactDerivative(shape1, oMg1, shape2, oMg2, contact, dreq, dcontact)

    # Chaining the derivatives to get Jc = Jc1 - Jc2 = doMc/dq
    Jg1 = pin.getFrameJacobian(
        model, data, geom1.parentJoint, geom1.placement, pin.LOCAL
    )
    Jg2 = pin.getFrameJacobian(
        model, data, geom2.parentJoint, geom2.placement, pin.LOCAL
    )

    doMc_dq = np.zeros((6, model.nv))
    doMc_dq[:3, :] = (
        dM_dp[:3] @ dcontact.dcpos_dM1 @ Jg1 + dM_dp[:3] @ dcontact.dcpos_dM2 @ Jg2
    )
    doMc_dq[3:, :] = (
        dM_dn[3:] @ dcontact.dnormal_dM1 @ Jg1 + dM_dn[3:] @ dcontact.dnormal_dM2 @ Jg2
    )
    return doMc_dq


# WARNING: this function assumes `pin.computeCollisions` and `pin.computeJointJacobians` were called before on a configuration `q`!
# WARNING: this function assumes that there exists at least a contact between the shapes involved in the collision pair!
def getContactJacobian(
    model: pin.Model,
    data: pin.Data,
    gmodel: pin.GeometryModel,
    gdata: pin.GeometryData,
    col_pair: int,
    oMc: pin.SE3,
):
    """
    Constructs Jc = Jc1 - Jc2 = Jacobian of the contact frame between geom1 and geom2, the geometries of the `col_pair`-th collision pair of `gmodel`.
    """
    # Assumes that there is only one collision pair in gmodel
    cres: hppfcl.CollisionResult = gdata.collisionResults[col_pair]
    assert cres.isCollision()

    geom1_id = gmodel.collisionPairs[col_pair].first
    geom1: pin.GeometryObject = gmodel.geometryObjects[geom1_id]
    i1Mc = data.oMi[geom1.parentJoint].inverse() * oMc
    Jc1 = pin.getFrameJacobian(model, data, geom1.parentJoint, i1Mc, pin.LOCAL)

    geom2_id = gmodel.collisionPairs[col_pair].second
    geom2: pin.GeometryObject = gmodel.geometryObjects[geom2_id]
    i2Mc = data.oMi[geom2.parentJoint].inverse() * oMc
    Jc2 = pin.getFrameJacobian(model, data, geom2.parentJoint, i2Mc, pin.LOCAL)

    Jc = Jc1 - Jc2
    return Jc


# WARNING: this function assumes `pin.computeCollisions` and `pin.computeJointJacobians` were called before on a configuration `q`! This configuration must be the same that was used to compute the contact frame!
# WARNING: this function assumes that there exists at least a contact between the shapes involved in the collision pair!
def contactVelocityDerivativeWrtConfiguration(
    model: pin.Model,
    data: pin.Data,
    gmodel: pin.GeometryModel,
    gdata: pin.GeometryData,
    col_pair: int,
    Jc: np.ndarray,
    oMc: pin.SE3,
    v: np.ndarray,
):
    """
    Computes dJc*v/dq, where v is considered constant and Jc is the contact jacobian related to the `col_pair`-th collision pair of gmodel. Jc is a function of q, the configuration of the system.
    """
    dMcdq = contactFrameDerivativeWrtConfiguration(
        model, data, gmodel, gdata, col_pair, oMc
    )

    geom1_id = gmodel.collisionPairs[col_pair].first
    geom1: pin.GeometryObject = gmodel.geometryObjects[geom1_id]
    i1Mc: pin.SE3 = data.oMi[geom1.parentJoint].inverse() * oMc
    Jjoint1 = pin.getJointJacobian(model, data, geom1.parentJoint, pin.LOCAL)

    geom2_id = gmodel.collisionPairs[col_pair].second
    geom2: pin.GeometryObject = gmodel.geometryObjects[geom2_id]
    i2Mc: pin.SE3 = data.oMi[geom2.parentJoint].inverse() * oMc
    Jjoint2 = pin.getJointJacobian(model, data, geom2.parentJoint, pin.LOCAL)

    cvel: pin.Motion = pin.Motion(Jc @ v)
    c1vel = pin.Motion(Jjoint1 @ v)
    c2vel = pin.Motion(Jjoint2 @ v)
    dJcv_dq = cvel.action @ dMcdq - (
        i1Mc.toActionMatrixInverse() @ c1vel.action @ Jjoint1
        - i2Mc.toActionMatrixInverse() @ c2vel.action @ Jjoint2
    )
    return dJcv_dq


# The dualSmallAdSwap has the following property:
# Let:
# mb = pin.Motion(b)
# fa = pin.Force(a)
# coswap = computeCoadjointSwap(fa)
# Then:
# coswap @ b - (mb.action).transpose() @ a = [0, 0, 0, 0, 0, 0]
def dualSmallAdSwap(wrench: pin.Force):
    res = np.zeros((6, 6))
    res[:3, 3:] = pin.skew(wrench.linear)
    res[3:, :3] = pin.skew(wrench.linear)
    res[3:, 3:] = pin.skew(wrench.angular)
    return res


# WARNING: this function assumes `pin.computeCollisions` and `pin.computeJointJacobians` were called before on a configuration `q`!
# WARNING: this function assumes that there exists at least a contact between the shapes involved in the collision pair!
def contactForceDerivativeWrtConfiguration(
    model: pin.Model,
    data: pin.Data,
    gmodel: pin.GeometryModel,
    gdata: pin.GeometryData,
    col_pair: np.ndarray,
    Jc: np.ndarray,
    oMc: pin.SE3,
    lam: np.ndarray,
):
    """
    Computes dJcT*lam/dq, where v is considered constant and Jc is the contact jacobian related to the `col_pair`-th collision pair of gmodel. Jc is a function of q, the configuration of the system.
    JcT denotes the transpose of the contact jacobian.
    """
    dMcdq = contactFrameDerivativeWrtConfiguration(
        model, data, gmodel, gdata, col_pair, oMc
    )
    force = pin.Force(lam)
    lamM = dualSmallAdSwap(force) @ dMcdq
    # In case we only need the linear part, simplification:
    # PElamM[:3, :] = pin.skew(lam) @ dMcdq[3:, :]
    # PElamM[3:, :] = pin.skew(lam) @ dMcdq[:3, :]

    geom1_id = gmodel.collisionPairs[col_pair].first
    geom1: pin.GeometryObject = gmodel.geometryObjects[geom1_id]
    Mc1: pin.SE3 = oMc.inverse() * data.oMi[geom1.parentJoint]
    Jjoint1 = pin.getJointJacobian(model, data, geom1.parentJoint, pin.LOCAL)
    wrench1 = pin.Force(Mc1.toActionMatrix().transpose() @ lam)
    coswap1 = dualSmallAdSwap(wrench1)

    geom2_id = gmodel.collisionPairs[col_pair].second
    geom2: pin.GeometryObject = gmodel.geometryObjects[geom2_id]
    Mc2: pin.SE3 = oMc.inverse() * data.oMi[geom2.parentJoint]
    Jjoint2 = pin.getJointJacobian(model, data, geom2.parentJoint, pin.LOCAL)
    wrench2 = pin.Force(Mc2.toActionMatrix().transpose() @ lam)
    coswap2 = dualSmallAdSwap(wrench2)

    dJcTlam_dq = (
        (-Jc.transpose() @ lamM)
        + (Jjoint1.transpose() @ coswap1 @ Jjoint1)
        - (Jjoint2.transpose() @ coswap2 @ Jjoint2)
    )
    return dJcTlam_dq
