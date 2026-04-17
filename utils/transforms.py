import numpy as np
from scipy.spatial.transform import Rotation as R


def euler_from_quaternion_np(quat, scalar_first=True):
    if scalar_first:
        quat = quat[..., [1, 2, 3, 0]]
    else:
        quat = quat[..., [0, 1, 2, 3]]

    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]
    w = quat[:, 3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1, 1)
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z


def quaternion_from_euler_np(roll_x, pitch_y, yaw_z, scalar_first=True):
    cy = np.cos(yaw_z * 0.5)
    sy = np.sin(yaw_z * 0.5)
    cp = np.cos(pitch_y * 0.5)
    sp = np.sin(pitch_y * 0.5)
    cr = np.cos(roll_x * 0.5)
    sr = np.sin(roll_x * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    if scalar_first:
        quat = np.stack([w, x, y, z], axis=-1)
    else:
        quat = np.stack([x, y, z, w], axis=-1)

    return quat
