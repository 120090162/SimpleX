import numpy as np
import pinocchio as pin


def computeSmoothStepDerivatives(
    sim, q, v, tau, fext, dt, dqdtheta, dvdtheta, dtaudtheta
):
    # We assume dfextdtheta to be zero
    ndtheta = dqdtheta.shape[1]
    # first call to aba
    pin.computeABADerivatives(sim.model, sim.data, q, v, tau, fext)
    dafdtheta = sim.data.ddq_qd @ dqdtheta
    dafdtheta += sim.data.ddq_dv @ dvdtheta
    dafdtheta += sim.data.Minv @ dtaudtheta
    # contact derivatives
    if sim.constraint_problem.getNumberOfContacts() > 0:
        # TODO: compute d(Glam + g)/dtheta directly
        dGdthetaLam = np.zeros((3 * sim.nc, ndtheta))
        dgdtheta = np.zeros((3 * sim.nc, ndtheta))
        dlamdtheta = computeSmoothNCPDerivatives(sim, dGdthetaLam, dgdtheta)
        # second call to aba
        danewdtheta = np.zeros((sim.model.nv, ndtheta))
    else:
        danewdtheta = dafdtheta
    dvnewdtheta = dvdtheta + dt * danewdtheta
    dqnewdq, dqnewdvnew = pin.dIntegrate(sim.model, q, dt * sim.vnew)
    dqnewdtheta = dqnewdq @ dqdtheta
    dqnewdtheta += dqnewdvnew @ (dt * dvnewdtheta)
    return dqnewdtheta, dvnewdtheta


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


def computeSmoothNCPDerivatives(sim, dGdthetaLam, dgdtheta, eps_smooth=1e-3):
    ndtheta = dGdthetaLam.shape[1]
    lam = sim.constraint_problem.point_contact_constraint_forces()
    contact_chol = sim.constraint_problem.constraint_cholesky_decomposition
    mu = 1e-12
    contact_chol.updateDamping(mu)
    nc = contact_chol.numContacts()
    assert 3 * nc == dgdtheta.shape[0]
    assert 3 * nc == dGdthetaLam.shape[0]
    Del = contact_chol.getDelassusCholeskyExpression().matrix() - np.eye(3 * nc) * mu
    sig = Del @ lam + sim.constraint_problem.g()
    index_breaking, index_sticking, index_sliding = collectActiveSet(lam, sig)
    n_brk = len(index_breaking)
    n_stk = len(index_sticking)
    n_sld = len(index_sliding)
    G_tilde = np.zeros(
        (3 * n_stk + 2 * n_sld + 3 * n_brk, 3 * n_stk + 2 * n_sld + 3 * n_brk)
    )
    E = np.zeros((3 * n_sld, 2))
    daccdtheta = dGdthetaLam + dgdtheta
    rhs = np.zeros((3 * n_stk + 2 * n_sld + 3 * n_brk, ndtheta))
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
        # add smoothing
        G_tilde[3 * i + 2, 3 * i + 2] += eps_smooth / (lam[3 * i + 2] ** 2)
        rhs[3 * i : 3 * i + 3] = -daccdtheta[3 * id_stk : 3 * id_stk + 3]

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
        for j in range(n_brk):
            G_tilde
        G_tilde[3 * n_stk + 2 * i + 1, 3 * n_stk + 2 * i + 1] += 1.0
        # compute rhs
        rhs[3 * n_stk + 2 * i : 3 * n_stk + 2 * i + 2] = -(
            ETP @ daccdtheta[3 * id_sld : 3 * id_sld + 3]
        )

    for i in range(n_brk):
        id_brk = index_breaking[i]
        for j in range(n_stk):
            id_stk_j = index_sticking[j]
            G_tilde[
                3 * n_stk + 2 * n_sld + 3 * i : 3 * n_stk + 2 * n_sld + 3 * i + 3,
                3 * j : 3 * j + 3,
            ] = Del[3 * id_stk : 3 * id_stk + 3, 3 * id_stk_j : 3 * id_stk_j + 3]
        for j in range(n_sld):
            id_sld_j = index_sliding[j]
            G_tilde[3 * i : 3 * i + 3, 3 * n_stk + 2 * j : 3 * n_stk + 2 * j + 2] = (
                Del[3 * id_stk : 3 * id_stk + 3, 3 * id_sld_j : 3 * id_sld_j + 3]
                @ E[3 * j : 3 * j + 3, 0:2]
            )
        rhs[
            3 * n_stk + 2 * n_sld + 3 * i : 3 * n_stk + 2 * n_sld + 3 * i + 3
        ] = -daccdtheta[3 * id_brk : 3 * id_brk + 3]

    # solve the system
    Q, R = np.linalg.qr(G_tilde)
    dlammindtheta = np.linalg.lstsq(R, Q.T @ rhs, rcond=None)[0]

    #  retrieving derivatives from result of the minimal system
    dlamdtheta = np.zeros_like(dgdtheta)
    for i in range(n_stk):
        dlamdtheta[3 * index_sticking[i] : 3 * index_sticking[i] + 3] = dlammindtheta[
            3 * i : 3 * i + 3
        ]
    for i in range(n_sld):
        dlamdtheta[3 * index_sliding[i] : 3 * index_sliding[i] + 3] = (
            E[3 * i : 3 * i + 3, 0:2]
            @ dlammindtheta[3 * n_stk + 2 * i : 3 * n_stk + 2 * i + 2]
        )
    return dlamdtheta
