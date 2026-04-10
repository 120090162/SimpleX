import mujoco
import mujoco.viewer
import numpy as np

np.set_printoptions(suppress=True, precision=5, linewidth=200)

from loop_rate_limiters import RateLimiter
import time
import os
import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

# Add the current directory to Python path to find module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.params import _ASSETS_DIR
from utils.logger import LOGGER

from absl import app
from absl import flags

os.environ["MUJOCO_GL"] = "egl"

import pinocchio as pin
from pycontact import (
    ContactProblem,
    RaisimSolver,
    RaisimCorrectedSolver,
    ContactSolverSettings,
    DelassusPinocchio,
)


def solve_contact_force(
    model,
    data,
    q,
    v,
    tau,
    contact_frame_ids,
    mu,
    n_contacts,
    dt,
):
    # 1. 计算动力学项
    pin.computeAllTerms(model, data, q, v)
    pin.computeMinverse(model, data, q)

    Minv = data.Minv.copy()  # shape (nv, nv)
    h = data.nle.copy()  # shape (nv, )

    # 2. 计算接触雅可比 (World Frame)
    Jk = [
        # shape (3, nv)
        pin.getFrameJacobian(
            model, data, contact_frame_ids[i], pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3, :]
        for i in range(n_contacts)
    ]
    J = np.vstack(Jk)  # shape (3*n_contacts, nv)

    # 3. Delassus Matrix
    G = J @ Minv @ J.T

    # 4. Free velocity
    # v_next = v + Minv @ (tau - h) * dt + Minv @ J.T @ impulse
    # g = J @ v_next
    v_free_update = v + Minv @ (tau - h) * dt
    g = J @ v_free_update  # 自由速度在接触点的投影

    # 5. 使用 ContactBench 求解 NCP
    # 每个接触点的摩擦系数列表
    mus = [mu] * n_contacts

    # 创建接触问题
    prob = ContactProblem(G, g, mus)

    # 选择求解器
    solver = RaisimCorrectedSolver()
    # solver = RaisimSolver()
    solver.setProblem(prob)

    # 求解器设置
    settings = ContactSolverSettings()
    settings.max_iter_ = 100
    settings.th_stop_ = 1e-6
    settings.rel_th_stop_ = 1e-12

    # 初始猜测
    impulse0 = np.zeros((3 * n_contacts, 1))

    # 求解
    is_converged = solver.solve(prob, impulse0, settings)
    # assert is_converged, "Solver did not converge"
    impulse = solver.getSolution().copy()  # 冲量

    # 6. 计算最终结果
    # 接触速度: v_contact = G @ impulse + g
    v_contact = solver.getDualSolution().copy()
    # v_contact = G @ impulse + g

    # 7. 检查 LCP 互补条件
    # LCP 条件: v_contact >= 0, impulse >= 0, v_contact^T * impulse = 0 (对于法向分量)
    # 对于摩擦锥问题，条件更复杂
    lcp_info = {
        "v_contact": v_contact.flatten(),
        "impulse": impulse.flatten(),
        "is_converged": is_converged,
    }

    # 检查每个接触点的互补条件
    for i in range(n_contacts):
        # 每个接触点有 3 个分量: [tangent1, tangent2, normal]
        idx = i * 3
        v_n = v_contact[idx + 2]  # 法向速度
        lambda_n = impulse[idx + 2]  # 法向冲量

        v_t = v_contact[idx : idx + 2]  # 切向速度
        lambda_t = impulse[idx : idx + 2]  # 切向冲量

        # 法向互补: v_n >= 0, lambda_n >= 0, v_n * lambda_n = 0
        lcp_info[f"contact_{i}_v_n"] = float(v_n)
        lcp_info[f"contact_{i}_lambda_n"] = float(lambda_n)
        lcp_info[f"contact_{i}_complementarity_n"] = float(v_n * lambda_n)

        # 切向: ||lambda_t|| <= mu * lambda_n (摩擦锥约束)
        lambda_t_norm = np.linalg.norm(lambda_t)
        lcp_info[f"contact_{i}_friction_cone"] = float(lambda_t_norm - mu * lambda_n)

    return impulse / dt, v_contact, lcp_info


def main(argv):
    del argv  # Unused.

    # Load robot model
    pin_model_path = _ASSETS_DIR / "unitree_go2/go2.xml"
    pin_model = pin.buildModelFromMJCF(pin_model_path.as_posix())
    pin_data = pin_model.createData()

    # Get Pinocchio foot frame IDs
    foot_names = ["FR", "FL", "RR", "RL"]
    pin_foot_frame_names = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]
    pin_foot_frame_ids = {}
    for name, frame_name in zip(foot_names, pin_foot_frame_names):
        frame_id = pin_model.getFrameId(frame_name)
        if frame_id < pin_model.nframes:
            pin_foot_frame_ids[name] = frame_id
        else:
            print(LOGGER.WARNING + f"Pinocchio frame {frame_name} not found!")

    # Get contact frame IDs as a list (in order of foot_names)
    contact_frame_ids = [pin_foot_frame_ids[name] for name in foot_names]
    n_contacts = len(contact_frame_ids)

    # physics parameters
    dt = 0.001
    mu = 0.8
    epsilon = 1e-6

    # Load Mujoco model
    model_path = _ASSETS_DIR / "unitree_go2/scene_display.xml"
    print(LOGGER.INFO + f"Loading Mujoco model from: {model_path}")

    try:
        model = mujoco.MjModel.from_xml_path(model_path.as_posix())
        data = mujoco.MjData(model)

        model.opt.timestep = dt
        sim_fps = 1.0 / dt

        render_fps = 30.0
        render_substeps = int(sim_fps / render_fps)

        # Reset to keyframe home
        mujoco.mj_resetDataKeyframe(model, data, 0)
        print(LOGGER.INFO + "Reset to keyframe 0")

        # Get geom IDs for feet and floor
        foot_names = [
            "FR_foot_collision",
            "FL_foot_collision",
            "RR_foot_collision",
            "RL_foot_collision",
        ]
        # Mapping: geom_id -> name and name -> geom_id for later lookups
        foot_id_to_name = {}
        foot_name_to_id = {}
        for name in foot_names:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid != -1:
                foot_id_to_name[gid] = name
                foot_name_to_id[name] = gid
            else:
                print(LOGGER.WARNING + f"Geom {name} not found!")

        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        rate_limiter = RateLimiter(frequency=sim_fps, warn=False)

        step_counter = 0

        print(LOGGER.INFO + "Successfully loaded model!")
        # launch_passive allows the script to continue running so we can step the simulation
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Enable contact force visualization
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

            while viewer.is_running():
                # Step the simulation
                mujoco.mj_step(model, data)

                # Sync the viewer
                if step_counter % render_substeps == 0:
                    viewer.sync()

                    # analyze data
                    foot_forces = {name: np.zeros(3) for name in foot_names}

                    for i in range(data.ncon):
                        contact = data.contact[i]

                        # Check if contact involves floor and one of the feet
                        g1 = contact.geom1
                        g2 = contact.geom2

                        foot_name = None
                        if g1 == floor_id and g2 in foot_id_to_name:
                            foot_name = foot_id_to_name[g2]
                        elif g2 == floor_id and g1 in foot_id_to_name:
                            foot_name = foot_id_to_name[g1]

                        if foot_name is not None:
                            # Get contact force
                            c_force = np.zeros(6)
                            mujoco.mj_contactForce(model, data, i, c_force)

                            # Transform force to world frame
                            # Contact frame: x=normal, y,z=tangent
                            # We need the contact frame rotation matrix
                            contact_frame = contact.frame.reshape(3, 3)

                            # Force in world frame = R * F_local
                            # Note: c_force[:3] is [normal, tangent1, tangent2]
                            # But mj_contactForce returns [normal, tangent1, tangent2, torque...]
                            # The contact frame matrix columns are [normal, tangent1, tangent2]

                            f_local = c_force[:3]
                            f_world = contact_frame.T @ f_local

                            foot_forces[foot_name] += f_world  # [Fx, Fy, Fz]
                    # Prepare inputs for custom force estimation
                    q = data.qpos.copy()
                    quat = q[3:7]
                    # Convert quat from [w, x, y, z] to [x, y, z, w]
                    q[3:7] = np.array([quat[1], quat[2], quat[3], quat[0]])
                    v = data.qvel.copy()
                    tau = data.qfrc_actuator.copy()

                    # Custom forward-dynamics-based estimate
                    est_forces_array, v_contact, lcp_info = solve_contact_force(
                        pin_model,
                        pin_data,
                        q,
                        v,
                        tau,
                        contact_frame_ids,
                        mu,
                        n_contacts,
                        dt,
                    )

                    # Convert array to dict for printing
                    est_forces = {}
                    for i, name in enumerate(foot_names):
                        est_forces[name] = est_forces_array[
                            i * 3 : (i + 1) * 3
                        ].flatten()

                    # Print formatted forces
                    parts = []
                    for k in foot_names:
                        mj = np.array2string(
                            foot_forces[k], precision=2, suppress_small=True
                        )
                        est = np.array2string(
                            est_forces.get(k, np.zeros(3)),
                            precision=2,
                            suppress_small=True,
                        )
                        parts.append(f"{k} mj: {mj} | est: {est}")

                    print(LOGGER.DEBUG + " | ".join(parts))

                    # Print LCP verification info
                    print(LOGGER.DEBUG + f"LCP converged: {lcp_info['is_converged']}")
                    print(
                        LOGGER.DEBUG
                        + f"v_contact: {np.array2string(lcp_info['v_contact'], precision=4, suppress_small=True)}"
                    )
                    print(
                        LOGGER.DEBUG
                        + f"impulse:   {np.array2string(lcp_info['impulse'], precision=4, suppress_small=True)}"
                    )

                    # Print per-contact LCP conditions
                    lcp_parts = []
                    for i in range(n_contacts):
                        v_n = lcp_info[f"contact_{i}_v_n"]
                        lambda_n = lcp_info[f"contact_{i}_lambda_n"]
                        comp_n = lcp_info[f"contact_{i}_complementarity_n"]
                        friction = lcp_info[f"contact_{i}_friction_cone"]
                        lcp_parts.append(
                            f"C{i}: v_n={v_n:.4f}, λ_n={lambda_n:.4f}, v_n*λ_n={comp_n:.6f}, friction_slack={friction:.4f}"
                        )
                    print(LOGGER.DEBUG + " | ".join(lcp_parts))

                step_counter += 1
                rate_limiter.sleep()

    except Exception as e:
        print(LOGGER.ERROR + f"Failed to load/visualize: {e}")


if __name__ == "__main__":
    app.run(main)
