import pinocchio as pin
import numpy as np

np.set_printoptions(suppress=True)

# Testing the derivatives of M.act(f), where M is an SE3 element, f is a spatial force.
# Derivative w.r.t M.
for i in range(1000):
    M: pin.SE3 = pin.SE3.Random()
    lam: pin.Force = pin.Force.Random()
    # res: pin.Force = M.act(lam)

    # Analytic jacobian
    P = np.zeros((6, 6))
    P[:3, 3:] = pin.skew(lam.linear)
    P[3:, :3] = pin.skew(lam.linear)
    P[3:, 3:] = pin.skew(lam.angular)
    J = -M.toDualActionMatrix() @ P

    # Finite diff jacobian
    Jfd = np.zeros((6, 6))
    eps = 1e-6
    ei = np.zeros(6)
    for i in range(6):
        ei[i] = eps
        M_plus = M * pin.exp6(ei)
        M_minus = M * pin.exp6(-ei)
        Jfd[:, i] = (M_plus.act(lam) - M_minus.act(lam)) / (2 * eps)
        ei[i] = 0

    assert np.linalg.norm(Jfd - J) <= 1e-8

v = pin.Motion.Random()
A = np.zeros((6, 6))
A[:3, :3] = pin.skew(v.angular)
A[:3, 3:] = pin.skew(v.linear)
A[3:, 3:] = pin.skew(v.angular)
print(f"Action = \n{v.action}")
print(f"Action alamano = \n{A}")
