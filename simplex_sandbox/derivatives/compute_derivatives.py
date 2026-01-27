import numpy as np
import pinocchio as pin
from contact_frame_derivatives import (
    getContactJacobian,
    contactVelocityDerivativeWrtConfiguration,
    contactForceDerivativeWrtConfiguration,
)
import simplex
import hppfcl
# import pydiffcoal as dcoal

np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.nan)
np.set_printoptions(precision=2)


def __computePrimalDualCollisionCorrection(sim: simplex.SimulatorX, v: np.ndarray):
    """
    This function is a backup, such that **all** the terms that are involved in it are tested in
    the script `derivative_contact_frame_wrt_configuration.py`.

    Computes the Primal and Dual Collision detection Corrective terms
    Primal: (dJc*v/dq).
    Dual: (Jc * dJcT*lam/dq), where lam are the contact forces computed by the simulator.
    """
    assert (
        sim.constraint_problem.getNumberOfContacts() > 0
    )  # otherwise this function should not be called.
    PCC = np.zeros((3 * sim.constraint_problem.getNumberOfContacts(), sim.model.nv))
    DCC = np.zeros((sim.model.nv, sim.model.nv))
    nc = 0  # count number of contact points; by the end, there should be as many as in the contact problem.
    for i in range(len(sim.constraint_problem.pairs_in_collision)):
        col_pair: int = sim.constraint_problem.pairs_in_collision[i]
        contact_mapper: simpl额x.ContactMapper = sim.constraint_problem.contact_mappers[
            col_pair
        ]
        for j in range(contact_mapper.count):
            idc = contact_mapper.begin + j
            oMc: pin.SE3 = sim.constraint_problem.point_contact_constraint_placements()[
                idc
            ]
            lam = pin.Force(
                sim.constraint_problem.point_contact_constraint_forces()[
                    3 * idc : 3 * (idc + 1)
                ],
                np.zeros(3),
            )
            Jc: np.ndarray = getContactJacobian(
                sim.model, sim.data, sim.geom_model, sim.geom_data, col_pair, oMc
            )
            dJcv_dq: np.ndarray = contactVelocityDerivativeWrtConfiguration(
                sim.model, sim.data, sim.geom_model, sim.geom_data, col_pair, Jc, oMc, v
            )
            dJcTlam_dq = contactForceDerivativeWrtConfiguration(
                sim.model,
                sim.data,
                sim.geom_model,
                sim.geom_data,
                col_pair,
                Jc,
                oMc,
                lam,
            )
            PCC[3 * nc : 3 * (nc + 1), :] = dJcv_dq[:3]
            DCC = DCC + dJcTlam_dq
            nc += 1
    Minv = (
        sim.constraint_problem.constraint_cholesky_decomposition.getInverseMassMatrix()
    )
    DCC = Minv @ DCC
    assert nc == sim.constraint_problem.getNumberOfContacts()
    return PCC, DCC


# The dualSmallAdSwap has the following property:
#     Let:
#     mb = pin.Motion(b)
#     fa = pin.Force(a)
#     coswap = computeCoadjointSwap(fa)
#     Then:
#     coswap @ b - (mb.action).transpose() @ a = [0, 0, 0, 0, 0, 0]
def dualSmallAdSwap(wrench: pin.Force):
    res = np.zeros((6, 6))
    res[:3, 3:] = pin.skew(wrench.linear)
    res[3:, :3] = pin.skew(wrench.linear)
    res[3:, 3:] = pin.skew(wrench.angular)
    return res


def computePrimalDualCollisionCorrection(sim: simplex.SimulatorX, v: np.ndarray):
    """
    Computes the Primal and Dual Collision detection Corrective terms
    Primal: (dJc*v/dq).
    Dual: Minv * (dJcT*lam/dq), where lam are the contact forces computed by the simulator.
    """
    assert (
        sim.constraint_problem.getNumberOfContacts() > 0
    )  # otherwise this function should not be called.
    PCC = np.zeros((3 * sim.constraint_problem.getNumberOfContacts(), sim.model.nv))
    DCC = np.zeros((sim.model.nv, sim.model.nv))
    dJcTlam_dq = np.zeros((sim.model.nv, sim.model.nv))
    for i in range(len(sim.constraint_problem.pairs_in_collision)):
        col_pair: int = sim.constraint_problem.pairs_in_collision[i]
        contact_mapper: simplex.ContactMapper = sim.constraint_problem.contact_mappers[
            col_pair
        ]

        geom1_id = sim.geom_model.collisionPairs[col_pair].first
        geom1: pin.GeometryObject = sim.geom_model.geometryObjects[geom1_id]
        shape1: hppfcl.CollisionGeometry = geom1.geometry
        oMg1: pin.SE3 = sim.geom_data.oMg[geom1_id]

        geom2_id = sim.geom_model.collisionPairs[col_pair].second
        geom2: pin.GeometryObject = sim.geom_model.geometryObjects[geom2_id]
        shape2: hppfcl.CollisionGeometry = geom2.geometry
        oMg2: pin.SE3 = sim.geom_data.oMg[geom2_id]

        # All the contact points of the collision pair share the same normal.
        cres: hppfcl.CollisionResult = sim.geom_data.collisionResults[col_pair]
        assert cres.isCollision()
        patch_res: hppfcl.ContactPatchResult = sim.geom_data.contactPatchResults[
            col_pair
        ]
        cpatch: hppfcl.ContactPatch = patch_res.getContactPatch(0)

        # Compute collision detection derivatives
        dreq = dcoal.ContactDerivativeRequest()
        dpatch = dcoal.ContactPatchDerivative()
        calc_contact_derivative = dcoal.ComputeContactPatchDerivative(shape1, shape2)
        calc_contact_derivative(oMg1, oMg2, cpatch, dreq, dpatch)

        for j in range(contact_mapper.count):
            idc = contact_mapper.begin + j
            oMc: pin.SE3 = sim.constraint_problem.point_contact_constraint_placements()[
                idc
            ]

            # Compute doMc/dn and doMc/dp
            doMc_dn, doMc_dp = simplex.placementFromNormalAndPositionDerivative(oMc)

            # Get derivative of contact patch point.
            cpos = oMc.translation
            dcpos_dM1, dcpos_dM2 = dcoal.getDerivativeOfContactPatchPoint(
                cpatch, dpatch, oMg1, oMg2, cpos
            )

            # Chaining the derivatives to get doMc/dq
            Jg1 = pin.getFrameJacobian(
                sim.model, sim.data, geom1.parentJoint, geom1.placement, pin.LOCAL
            )

            Jg2 = pin.getFrameJacobian(
                sim.model, sim.data, geom2.parentJoint, geom2.placement, pin.LOCAL
            )

            doMc_dq = np.zeros((6, sim.model.nv))
            doMc_dp = doMc_dp[:3]
            doMc_dq[:3, :] = doMc_dp @ dcpos_dM1 @ Jg1 + doMc_dp @ dcpos_dM2 @ Jg2
            doMc_dn = doMc_dn[3:]
            doMc_dq[3:, :] = (
                doMc_dn @ dpatch.dnormal_dM1() @ Jg1
                + doMc_dn @ dpatch.dnormal_dM2() @ Jg2
            )

            # ================================================================
            # Compute Primal Corrective term
            # Compute Jc1
            i1Mc: pin.SE3 = sim.data.oMi[geom1.parentJoint].inverse() * oMc
            Xc1 = i1Mc.toActionMatrixInverse()
            Jc1 = pin.getFrameJacobian(
                sim.model, sim.data, geom1.parentJoint, i1Mc, pin.LOCAL
            )

            # Compute Jc2
            i2Mc: pin.SE3 = sim.data.oMi[geom2.parentJoint].inverse() * oMc
            Xc2 = i2Mc.toActionMatrixInverse()
            Jc2 = pin.getFrameJacobian(
                sim.model, sim.data, geom2.parentJoint, i2Mc, pin.LOCAL
            )

            # Contact jacobian
            Jc = Jc1 - Jc2

            Jjoint1 = pin.getJointJacobian(
                sim.model, sim.data, geom1.parentJoint, pin.LOCAL
            )
            Jjoint2 = pin.getJointJacobian(
                sim.model, sim.data, geom2.parentJoint, pin.LOCAL
            )

            cvel: pin.Motion = pin.Motion(Jc @ v)
            c1vel = pin.Motion(Jjoint1 @ v)
            c2vel = pin.Motion(Jjoint2 @ v)
            dJcv_dq = cvel.action @ doMc_dq - (
                Xc1 @ c1vel.action @ Jjoint1 - Xc2 @ c2vel.action @ Jjoint2
            )
            PCC[3 * idc : 3 * (idc + 1), :] = dJcv_dq[:3]

            # ================================================================
            # Compute Dual Corrective term
            wrench = pin.Force(
                sim.constraint_problem.point_contact_constraint_forces()[
                    3 * idc : 3 * (idc + 1)
                ],
                np.zeros(3),
            )
            coswap = dualSmallAdSwap(wrench)
            coswap_doMc_dq = coswap @ doMc_dq

            wrench1 = i1Mc.act(wrench)
            coswap1 = dualSmallAdSwap(wrench1)

            wrench2 = i2Mc.act(wrench)
            coswap2 = dualSmallAdSwap(wrench2)

            dJcTlam_dq += (
                (-Jc.transpose() @ coswap_doMc_dq)
                + (Jjoint1.transpose() @ coswap1 @ Jjoint1)
                - (Jjoint2.transpose() @ coswap2 @ Jjoint2)
            )

    Minv = (
        sim.constraint_problem.constraint_cholesky_decomposition.getInverseMassMatrix()
    )
    DCC = Minv @ dJcTlam_dq
    return PCC, DCC


def computeStepDerivatives(sim, q, v, tau, fext, dt):
    ndtheta = 3 * sim.model.nv
    # first call to aba
    danew_dq, danew_dv, danew_dtau = pin.computeABADerivatives(
        sim.model, sim.data, q, v, tau, sim.ftotal
    )  # this should include derivatives from diffcoal
    # contact derivatives
    nc = sim.constraint_problem.getNumberOfContacts()
    if nc > 0:
        pin.framesForwardKinematics(sim.model, sim.data, q)
        pin.updateGeometryPlacements(sim.model, sim.data, sim.geom_model, sim.geom_data)
        pin.computeForwardKinematicsDerivatives(sim.model, sim.data, q, sim.vnew)
        dGlamgdtheta = np.zeros((3 * nc, ndtheta))
        PCC, DCC = computePrimalDualCollisionCorrection(sim, sim.vnew)
        for i in range(nc):
            joint1_id = sim.constraint_problem.getConstraintModel(i).joint1_id
            placement1 = sim.constraint_problem.getConstraintModel(i).joint1_placement
            dsigma1dq, dsigma1dv = pin.getFrameVelocityDerivatives(
                sim.model, sim.data, joint1_id, placement1, pin.LOCAL
            )

            joint2_id = sim.constraint_problem.getConstraintModel(i).joint2_id
            placement2 = sim.constraint_problem.getConstraintModel(i).joint2_placement
            dsigma2dq, dsigma2dv = pin.getFrameVelocityDerivatives(
                sim.model, sim.data, joint2_id, placement2, pin.LOCAL
            )

            # FVD
            dGlamgdq = (
                dsigma1dq - dsigma2dq
            ) / dt  # note: we divide by dt because the input of the NCP solver is g/dt
            dGlamgdv = (dsigma1dv - dsigma2dv) / dt

            # ABA terms
            # -- dq
            dGlamgdtheta[3 * i : 3 * i + 3, : sim.model.nv] = dGlamgdq[:3]
            dGlamgdtheta[3 * i : 3 * i + 3, : sim.model.nv] += dGlamgdv[:3] @ (
                dt * danew_dq
            )
            dGlamgdtheta[3 * i : 3 * i + 3, : sim.model.nv] += (
                PCC[3 * i : 3 * i + 3, :] / dt
            )
            dGlamgdtheta[3 * i : 3 * i + 3, : sim.model.nv] += dGlamgdv[:3] @ (dt * DCC)

            # -- dv
            dGlamgdtheta[3 * i : 3 * i + 3, sim.model.nv : 2 * sim.model.nv] = dGlamgdv[
                :3
            ] @ (np.eye(sim.model.nv) + dt * danew_dv)

            # -- dtau
            dGlamgdtheta[3 * i : 3 * i + 3, 2 * sim.model.nv :] = dGlamgdv[:3] @ (
                dt * danew_dtau
            )

        dlam_dtheta = computeNCPDerivatives(sim, dGlamgdtheta)
        MinvJT = (
            sim.constraint_problem.constraint_cholesky_decomposition.getInverseMassMatrix()
            @ (
                (
                    sim.constraint_problem.constraint_cholesky_decomposition.matrix()[
                        : 3 * nc, -sim.model.nv :
                    ]
                ).transpose()
            )
        )
        dlamdq = dlam_dtheta[:, : sim.model.nv]
        dlamdv = dlam_dtheta[:, sim.model.nv : sim.model.nv * 2]
        dlamdtau = dlam_dtheta[:, sim.model.nv * 2 :]

        danew_dq += MinvJT @ dlamdq
        danew_dq += DCC
        danew_dv += MinvJT @ dlamdv
        danew_dtau += MinvJT @ dlamdtau
    dvnew_dq = dt * danew_dq
    dvnew_dv = np.eye(sim.model.nv) + dt * danew_dv
    dvnew_dtau = dt * danew_dtau
    dqnew_dq, dqnew_dvnew = pin.dIntegrate(sim.model, q, dt * sim.vnew)
    dqnew_dq += dqnew_dvnew @ (dt * dvnew_dq)
    dqnew_dv = dqnew_dvnew @ (dt * dvnew_dv)
    dqnewdtau = dqnew_dvnew @ (dt * dvnew_dtau)
    return dqnew_dq, dqnew_dv, dqnewdtau, dvnew_dq, dvnew_dv, dvnew_dtau


def collectActiveSet(lam, sig, epsilon=1e-6):
    index_breaking = []
    index_sticking = []
    index_sliding = []
    nc = len(lam) // 3
    for i in range(nc):
        lami = lam[3 * i : 3 * i + 3]
        sigi = sig[3 * i : 3 * i + 3]
        if np.linalg.norm(lami) < epsilon:
            index_breaking.append(i)
        elif np.linalg.norm(sigi) < epsilon:
            index_sticking.append(i)
        else:
            index_sliding.append(i)
    return index_breaking, index_sticking, index_sliding


def computeNCPDerivatives(sim, dGlamgdtheta):
    ndtheta = dGlamgdtheta.shape[1]
    lam = sim.constraint_problem.frictional_point_constraints_forces()
    contact_chol = sim.constraint_problem.constraint_cholesky_decomposition
    mu = 1e-12
    contact_chol.updateDamping(mu)
    nc = contact_chol.numContacts()
    assert 3 * nc == dGlamgdtheta.shape[0]
    Del = contact_chol.getDelassusCholeskyExpression().matrix() - np.eye(3 * nc) * mu
    sig = Del @ lam + sim.constraint_problem.g()
    index_breaking, index_sticking, index_sliding = collectActiveSet(lam, sig)
    n_brk = len(index_breaking)
    n_stk = len(index_sticking)
    n_sld = len(index_sliding)
    G_tilde = np.zeros((3 * n_stk + 2 * n_sld, 3 * n_stk + 2 * n_sld))
    E = np.zeros((3 * n_sld, 2))
    rhs = np.zeros((3 * n_stk + 2 * n_sld, ndtheta))
    # Compute new E basis for minimal system
    for i in range(n_sld):
        id_sld = index_sliding[i]
        # compute E matrix
        E[3 * i : 3 * i + 3, 0] = lam[3 * id_sld : 3 * id_sld + 3] / np.linalg.norm(
            lam[3 * id_sld : 3 * id_sld + 3]
        )
        ez = np.array([0, 0, 1])
        E[3 * i : 3 * i + 3, 1] = np.cross(
            ez,
            sig[3 * id_sld : 3 * id_sld + 3]
            / np.linalg.norm(sig[3 * id_sld : 3 * id_sld + 3]),
        )
    # compute G_tilde and rhs for sticking contacts
    for i in range(n_stk):
        id_stk = index_sticking[i]
        for j in range(n_stk):
            id_stk_j = index_sticking[j]
            G_tilde[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = Del[
                3 * id_stk : 3 * id_stk + 3, 3 * id_stk_j : 3 * id_stk_j + 3
            ]
        for j in range(n_sld):
            id_sld_j = index_sliding[j]
            G_tilde[3 * i : 3 * i + 3, 3 * n_stk + 2 * j : 3 * n_stk + 2 * j + 2] = (
                Del[3 * id_stk : 3 * id_stk + 3, 3 * id_sld_j : 3 * id_sld_j + 3]
                @ E[3 * j : 3 * j + 3, 0:2]
            )
        rhs[3 * i : 3 * i + 3] = -dGlamgdtheta[3 * id_stk : 3 * id_stk + 3]

    # compute G_tilde and rhs for sliding contacts
    for i in range(n_sld):
        id_sld = index_sliding[i]
        # compute P matrix
        lam_n = lam[3 * id_sld + 2]
        sig_T = sig[3 * id_sld : 3 * id_sld + 2]
        mu = sim.constraint_problem.cones[id_sld].mu
        alpha = np.linalg.norm(sig_T) / (lam_n * mu)
        ETP = np.zeros((2, 3))
        ETP[0, 2] = alpha * E[3 * i + 2, 0]
        ETP[1, 0:2] = E[3 * i : 3 * i + 2, 1]
        ETP *= 1 / alpha
        # compute G_tilde
        for j in range(n_stk):
            id_stk_j = index_sticking[j]
            G_tilde[3 * n_stk + 2 * i : 3 * n_stk + 2 * i + 2, 3 * j : 3 * j + 3] = (
                E[3 * i : 3 * i + 3, 0:2].T
                @ Del[3 * id_sld : 3 * id_sld + 3, 3 * id_stk_j : 3 * id_stk_j + 3]
            )
        for j in range(n_sld):
            id_sld_j = index_sliding[j]
            G_tilde[
                3 * n_stk + 2 * i : 3 * n_stk + 2 * i + 2,
                3 * n_stk + 2 * j : 3 * n_stk + 2 * j + 2,
            ] = (
                ETP
                @ Del[3 * id_sld : 3 * id_sld + 3, 3 * id_sld_j : 3 * id_sld_j + 3]
                @ E[3 * j : 3 * j + 3, 0:2]
            )
        G_tilde[3 * n_stk + 2 * i + 1, 3 * n_stk + 2 * i + 1] += 1.0
        # compute rhs
        rhs[3 * n_stk + 2 * i : 3 * n_stk + 2 * i + 2] = -(
            ETP @ dGlamgdtheta[3 * id_sld : 3 * id_sld + 3]
        )
    # solve the system
    Q, R = np.linalg.qr(G_tilde)
    dlammindtheta = np.linalg.lstsq(R, Q.T @ rhs, rcond=None)[0]

    #  retrieving derivatives from result of the minimal system
    dlam_dtheta = np.zeros_like(dGlamgdtheta)
    for i in range(n_stk):
        dlam_dtheta[3 * index_sticking[i] : 3 * index_sticking[i] + 3] = dlammindtheta[
            3 * i : 3 * i + 3
        ]
    for i in range(n_sld):
        dlam_dtheta[3 * index_sliding[i] : 3 * index_sliding[i] + 3] = (
            E[3 * i : 3 * i + 3, 0:2]
            @ dlammindtheta[3 * n_stk + 2 * i : 3 * n_stk + 2 * i + 2]
        )
    return dlam_dtheta


def finiteDifferencesStep(simulator: simplex.SimulatorX, q, v, tau, dt, eps=1e-6):
    """
    Finite differences step for the simulator
    """
    vdot = np.zeros_like(v)
    dvnew_dq = np.zeros((simulator.model.nv, simulator.model.nv))
    dvnew_dv = np.zeros((simulator.model.nv, simulator.model.nv))
    dvnew_dtau = np.zeros((simulator.model.nv, simulator.model.nv))

    simulator.reset()
    simulator.step(q, v, tau, dt)

    for i in range(simulator.model.nv):
        qdot = np.zeros(simulator.model.nv)
        qdot[i] = eps

        qplus = pin.integrate(simulator.model, q, qdot)
        simulator.reset()
        simulator.step(qplus, v, tau, dt)
        vnewplus = simulator.state.vnew.copy()

        qminus = pin.integrate(simulator.model, q, -qdot)
        simulator.reset()
        simulator.step(qminus, v, tau, dt)
        vnewminus = simulator.state.vnew.copy()

        dvnew_dq[:, i] = (vnewplus - vnewminus) / (2 * eps)

    for i in range(simulator.model.nv):
        vdot = np.zeros(simulator.model.nv)
        vdot[i] = eps

        simulator.reset()
        simulator.step(q, v + vdot, tau, dt)
        vnewplus = simulator.state.vnew.copy()

        simulator.reset()
        simulator.step(q, v - vdot, tau, dt)
        vnewminus = simulator.state.vnew.copy()

        dvnew_dv[:, i] = (vnewplus - vnewminus) / (2 * eps)

    for i in range(simulator.model.nv):
        taudot = np.zeros(simulator.model.nv)
        taudot[i] = eps

        simulator.reset()
        simulator.step(q, v, tau + taudot, dt)
        vnewplus = simulator.state.vnew.copy()

        simulator.reset()
        simulator.step(q, v, tau - taudot, dt)
        vnewminus = simulator.state.vnew.copy()

        dvnew_dtau[:, i] = (vnewplus - vnewminus) / (2 * eps)
    return dvnew_dq, dvnew_dv, dvnew_dtau
