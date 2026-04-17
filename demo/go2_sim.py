import mujoco
import mujoco.viewer

import simplex
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from loop_rate_limiters import RateLimiter
import numpy as np
from threading import Thread
import threading

import os
import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

# Add the current directory to Python path to find module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.params import *
from utils.logger import LOGGER
from utils.viz_utils import *
from utils.sim_utils import *

from absl import app
from absl import flags

os.environ["MUJOCO_GL"] = "egl"

_XML_PATH = flags.DEFINE_string(
    "xml_path",
    "unitree_go2/scene.xml",
    f"Path to the XML model file.",
)

_ROBOT_PATH = flags.DEFINE_string(
    "robot_path",
    "unitree_go2/go2.xml",
    f"Path to the robot XML file for Pinocchio.",
)

_CONFIG_PATH = flags.DEFINE_string(
    "config_path",
    "go2_sim_config.yaml",
    f"Path to the simulation config YAML file.",
)


class MPC_Sim:
    def __init__(
        self,
        mujoco_model_path: str,
        pin_model_path: str,
        args: SimulationArgs = None,
        config_path: str = None,
    ):
        print(LOGGER.INFO + "MPC_Sim start")
        self.mujoco_model_path = mujoco_model_path
        self.pin_model_path = pin_model_path
        self.mujoco_model = None
        self.mujoco_data = None
        self.pin_model = None
        self.pin_data = None
        self.collision_model = None
        self.visual_model = None
        self.viz = None
        self.is_running = True

        self.simplex_sim = None
        self.simplex_dsim = None
        self.solver = None
        self.solver_type = None
        self.q0 = None
        self.v0 = None
        self.qs = []
        self.vs = []

        # parse sim config
        self.sim_args = read_sim_config(args, config_path, is_show=False)
        # config random seeds for reproducibility
        np.random.seed(self.sim_args.seed)
        pin.seed(self.sim_args.seed)

        # Initialize Mujoco and Pinocchio models
        try:
            self.init_mujoco()
        except Exception as e:
            raise RuntimeError(LOGGER.ERROR + f"Failed to load Mujoco: {e}") from e

        try:
            self.init_pinocchio()
        except Exception as e:
            raise RuntimeError(LOGGER.ERROR + f"Failed to load Pinocchio: {e}") from e

    def __del__(self):
        print("\r\n")
        if self.viz is not None:
            print(LOGGER.INFO + "Generating Meshcat animation...")
            generate_meshcat_animation(self.viz, self.qs, 0.01)
            print(
                LOGGER.INFO
                + f"Animation uploaded! Open the Meshcat URL and look for 'Animations' controls."
            )

        print(LOGGER.INFO + "MPC_Sim exit")

    def init_mujoco(self):
        print(LOGGER.INFO + f"Loading Mujoco model from: {self.mujoco_model_path}")
        self.mujoco_model = mujoco.MjModel.from_xml_path(
            self.mujoco_model_path.as_posix()
        )
        self.mujoco_data = mujoco.MjData(self.mujoco_model)

        self.mujoco_model.opt.timestep = 0.001

        if self.mujoco_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.mujoco_model, self.mujoco_data, 0)
            print(LOGGER.INFO + "Reset to keyframe 0")

        print(LOGGER.INFO + "Successfully loaded Mujoco model!")

    def init_pinocchio(self):
        print(LOGGER.INFO + f"Loading Pinocchio model from: {self.pin_model_path}")
        pin_model_path_str = self.pin_model_path.as_posix()
        self.pin_model = pin.buildModelFromMJCF(pin_model_path_str)
        self.pin_data = self.pin_model.createData()
        self.collision_model = pin.buildGeomFromMJCF(
            self.pin_model, pin_model_path_str, pin.GeometryType.COLLISION
        )
        self.visual_model = pin.buildGeomFromMJCF(
            self.pin_model, pin_model_path_str, pin.GeometryType.VISUAL
        )

        self.q0 = self.pin_model.referenceConfigurations["home"]
        self.v0 = np.zeros(self.pin_model.nv)

        # Finalize setup of model
        # add floor first
        add_floor(
            self.collision_model,
            self.visual_model,
            BLACK,
        )
        # add model collision pairs based on q0
        add_system_collision_pairs(
            self.pin_model,
            self.collision_model,
            self.q0,
            self.sim_args.collision_pairs_init_margin,
        )
        # --> remove BVH and replace by convex
        remove_BVH_models(self.collision_model)
        # --> set material properties for contacts
        add_material_and_compliance(
            self.collision_model, self.sim_args.material, self.sim_args.compliance
        )
        # --> set joint limits, joint friction, joint damping
        setup_joint_constraints(
            self.pin_model,
            self.sim_args.joint_limit,
            self.sim_args.joint_friction,
            self.sim_args.damping,
        )

        # Simulation setup and get derivatives
        self.simplex_sim = simplex.SimulatorX(self.pin_model, self.collision_model)
        setup_simplex_simulator(self.simplex_sim, self.sim_args)
        self.simplex_dsim = simplex.SimulatorDerivatives(self.simplex_sim)
        self.simplex_sim.reset()

        if self.sim_args.contact_solver == "admm":
            self.solver = self.simplex_sim.workspace.constraint_solvers.admm_solver
            self.solver_type = simplex.ConstraintSolverType.ADMM
        elif self.sim_args.contact_solver == "pgs":
            self.solver = self.simplex_sim.workspace.constraint_solvers.pgs_solver
            self.solver_type = simplex.ConstraintSolverType.PGS
        elif self.sim_args.contact_solver == "clarabel":
            self.solver = self.simplex_sim.workspace.constraint_solvers.clarabel_solver
            self.solver_type = simplex.ConstraintSolverType.CLARABEL
        else:
            raise NotImplementedError(
                f"Unsupported contact solver: {self.sim_args.contact_solver}"
            )

        # Visualization setup
        self.viz = MeshcatVisualizer(
            self.pin_model, self.collision_model, self.visual_model
        )
        self.viz.initViewer(open=False)
        print(
            LOGGER.INFO
            + "Successfully loaded Pinocchio model and init MeshcatVisualizer!"
        )

    def simulate_mujoco(self):
        render_substeps = int(self.sim_args.sim_fps / self.sim_args.render_fps)
        rate_limiter = RateLimiter(frequency=self.sim_args.sim_fps, warn=False)
        step_counter = 0

        with mujoco.viewer.launch_passive(
            self.mujoco_model, self.mujoco_data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1  # 显示接触力

            while viewer.is_running() and self.is_running:
                mujoco.mj_step(self.mujoco_model, self.mujoco_data)

                if step_counter % render_substeps == 0:
                    viewer.sync()

                step_counter += 1
                rate_limiter.sleep()

        self.is_running = False

    def simulate_pinocchio(self):
        self.viz.loadViewerModel()
        self.viz.display(self.q0)
        self.viz.displayVisuals(True)

        # 储存历史数据以便动画展示
        self.qs = [self.q0]
        self.vs = [self.v0]

        # 力矩输入
        zero_torque = np.zeros(self.pin_model.nv)

        render_substeps = int(self.sim_args.sim_fps / self.sim_args.render_fps)
        rate_limiter = RateLimiter(
            frequency=self.sim_args.sim_fps, warn=False
        )  # 控制频率
        step_counter = 0
        while self.is_running:
            q = self.qs[step_counter]
            v = self.vs[step_counter]
            self.simplex_sim.step(q, v, zero_torque, self.sim_args.dt, self.solver_type)
            # self.simplex_sim.step(q, v, zero_torque, 0.025, self.solver_type)

            qnext = self.simplex_sim.state.qnew.copy()
            vnext = self.simplex_sim.state.vnew.copy()
            # pinocchio visualization sync
            if step_counter % render_substeps == 0:
                self.viz.display(qnext)

                self.simplex_dsim.stepDerivatives(
                    self.simplex_sim, q, v, zero_torque, self.sim_args.dt
                )
                # print(
                #     LOGGER.DEBUG
                #     + f"Contact count: {self.simplex_sim.workspace.constraint_problem.getNumberOfContacts()}"
                # )

            self.qs.append(qnext)
            self.vs.append(vnext)

            step_counter += 1
            rate_limiter.sleep()

    def run(self):
        t_mujoco = Thread(target=self.simulate_mujoco)
        t_pinocchio = Thread(target=self.simulate_pinocchio)

        # 设置为守护线程防止主线程被hang住
        t_mujoco.daemon = True
        t_pinocchio.daemon = True

        t_mujoco.start()
        t_pinocchio.start()

        try:
            # 使用带timeout的join轮询，这样主线程可以响应KeyboardInterrupt
            while t_mujoco.is_alive() or t_pinocchio.is_alive():
                t_mujoco.join(timeout=0.1)
                t_pinocchio.join(timeout=0.1)
        except KeyboardInterrupt:
            print("\n" + LOGGER.INFO + "Ctrl+C detected, shutting down gracefully...")
        finally:
            # 触发所有子线程的退出条件
            self.is_running = False
            # 给定一定时间让子线程进行清理并退出
            t_mujoco.join(timeout=1.0)
            t_pinocchio.join(timeout=1.0)


def main(argv):
    # del argv  # Unused.

    mujoco_model_path = _ASSETS_DIR / f"{_XML_PATH.value}"
    pin_model_path = _ASSETS_DIR / f"{_ROBOT_PATH.value}"  # 可是使用简化模型方便仿真
    config_path = _CONFIGS_DIR / f"{_CONFIG_PATH.value}"

    if not mujoco_model_path.exists():
        raise FileNotFoundError(f"Mujoco Model file not found: {mujoco_model_path}")

    if not pin_model_path.exists():
        raise FileNotFoundError(f"Pinocchio Model file not found: {pin_model_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Sim config file not found: {config_path}")

    sim = MPC_Sim(
        mujoco_model_path,
        pin_model_path,
        SimulationArgs().parse_args(args=argv[1:]),  # 解析命令行参数并覆盖默认值
        config_path,
    )
    sim.run()


if __name__ == "__main__":
    app.run(
        main, flags_parser=lambda a: flags.FLAGS(a, known_only=True)
    )  # 允许传递未定义的命令行参数（如 --sim_arg=value）给 SimulationArgs 解析
