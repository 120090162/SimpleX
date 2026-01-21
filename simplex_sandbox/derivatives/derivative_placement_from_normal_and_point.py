import numpy as np
import pinocchio as pin
import simple

np.set_printoptions(suppress=True)
np.random.seed(0)


def cross(a: np.ndarray, b: np.ndarray):
    res = np.zeros(3)
    res[0] = a[1] * b[2] - a[2] * b[1]
    res[1] = a[2] * b[0] - a[0] * b[2]
    res[2] = a[0] * b[1] - a[1] * b[0]
    return res


def normalized(a: np.ndarray):
    return a / np.linalg.norm(a)


def rotationFromNormal(n: np.ndarray):
    eref1 = np.array([1.0, 1.0, 1.0])
    eref2 = np.array([1.0, 0.0, -1.0])
    eref1 = eref1 / np.linalg.norm(eref1)
    eref2 = eref2 / np.linalg.norm(eref2)

    eref = eref1
    n_ = normalized(n)
    if np.linalg.norm(cross(n_, eref)) <= 1e-6:
        eref = eref2

    R = np.zeros((3, 3))
    R[:, 2] = n_
    u = normalized(cross(eref, n))
    R[:, 0] = u
    R[:, 1] = cross(n, u)
    return R


def frameFromNormalAndPoint(n, p):
    # e is the reference normal
    R = rotationFromNormal(n)
    M = pin.SE3(R, p)
    return M


def dFrameFromNormalAndPoint(M: pin.SE3):
    eref1 = np.array([1.0, 1.0, 1.0])
    eref2 = np.array([1.0, 0.0, -1.0])

    R = M.rotation
    u, v, n = R[:3, 0], R[:3, 1], R[:3, 2]
    eref = eref1
    if np.linalg.norm(cross(n, eref)) <= 1e-6:
        eref = eref2

    J = np.zeros((6, 6))
    J[:3, :3] = R.transpose()
    A = np.zeros((3, 3))
    A[:, 0] = -v
    A[:, 1] = u
    A[:, 2] = cross(v, eref) / (np.linalg.norm(cross(eref, n)))
    J[3:, 3:] = A.transpose()
    return J


# normal vector
n = np.random.rand(3)
n /= np.linalg.norm(n)

# reference vector
R = rotationFromNormal(n)
print(f"{n=}")
print(f"R = \n{R}")
print(f"det(R) = {np.linalg.det(R)}")
print(f"RT * R = \n{R.transpose() @ R}")

p = np.random.rand(3)
M = frameFromNormalAndPoint(n, p)
print(f"M = \n{M}")
print(f"Minv * M = \n{np.linalg.inv(M) @ M}")

J = dFrameFromNormalAndPoint(M)

eps = 1e-6
Jfd = np.zeros((6, 6))
for i in range(6):
    ei = np.zeros(3)
    ei[i % 3] = eps
    if i <= 2:
        pplus = p + ei
        Mplus = frameFromNormalAndPoint(n, pplus)
        pminus = p - ei
        Mminus = frameFromNormalAndPoint(n, pminus)
        dM = pin.log6(Mminus.inverse() * Mplus)
        Jfd[:, i] = dM / (2 * eps)
    else:
        v = n + ei
        dn = v - n * (n.dot(v))
        Mplus = frameFromNormalAndPoint(n + dn, p)
        Mminus = frameFromNormalAndPoint(n - dn, p)
        dM = pin.log6(Mminus.inverse() * Mplus)
        Jfd[:, i] = dM / (2 * eps)

print("\n")
print(f"J = \n{J}")
print(f"Jfd = \n{Jfd}")
print(f"Diff = {np.linalg.norm(J - Jfd)}")
assert np.linalg.norm(J - Jfd) <= 1e-8

# Testing simple's cpp version
print("----- TESTING SIMPLE's placementFromNormalAndPosition ------")
M = simple.placementFromNormalAndPosition(n, p)
MinvM = M.inverse() * M
print(f"M = \n{M}")
print(f"Minv * M = \n{M.inverse() * M}")
print(simple.placementFromNormalAndPositionDerivative(M))
dM_dn, dM_dp = simple.placementFromNormalAndPositionDerivative(M)

eps = 1e-6
dM_dn_fd = np.zeros((6, 3))
dM_dp_fd = np.zeros((6, 3))
for i in range(3):
    ei = np.zeros(3)
    ei[i % 3] = eps
    # dM_dn
    v = n + ei
    dn = v - n * (n.dot(v))
    Mplus = simple.placementFromNormalAndPosition(n + dn, p)
    Mminus = simple.placementFromNormalAndPosition(n - dn, p)
    dM = pin.log6(Mminus.inverse() * Mplus)
    dM_dn_fd[:, i] = dM / (2 * eps)

    # dM_dp
    Mplus = simple.placementFromNormalAndPosition(n, p + ei)
    Mminus = simple.placementFromNormalAndPosition(n, p - ei)
    dM = pin.log6(Mminus.inverse() * Mplus)
    dM_dp_fd[:, i] = dM / (2 * eps)
print("Diff dM_dn = ", np.linalg.norm(dM_dn - dM_dn_fd))
print("Diff dM_dp = ", np.linalg.norm(dM_dp - dM_dp_fd))
