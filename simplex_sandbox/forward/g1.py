import pinocchio as pin
import numpy as np
from robot_descriptions.loaders.pinocchio import load_robot_description as plr
from simplex_sandbox.utils.sim_utils import (
    SimulationArgs,
    runSimpleXSimulationFromModel,
    runMujocoXML,
)

class ScriptArgs(SimulationArgs):
    remove_anchor_constraints: bool = False
    display: bool = True  # display the simulation
    backend: str = "simplex"
    display_traj: bool = True  # display the trajectory
    pd_controller: bool = True  # use a PD controller
    random_init_vel: bool = True  # random initial velocity


args = ScriptArgs().parse_args()
np.random.seed(args.seed)
pin.seed(args.seed)
model_path = "g1_mj_description"

if args.backend == "simplex":
    # Create model
    print("Loading mj robot description...")
    robot_mj = plr(model_path)
    model = robot_mj.model
    geom_model = robot_mj.collision_model
    visual_model = robot_mj.visual_model

    for joint in model.joints:
        print("=============================")
        print(joint)
        if joint.shortname() == "JointModelComposite":
            print(joint.extract())

    # Initial state
    # q0 = model.referenceConfigurations["home"]
    q0 = pin.neutral(model)
    q0[2] += 0.8
    if args.random_init_vel:
        v0 = np.random.randn(model.nv)
    else:
        v0 = np.zeros(model.nv)

    if args.pd_controller:
        args.pd_controller_qtar = q0.copy()
        args.pd_controller_vtar = np.zeros(model.nv - 6)

    runSimpleXSimulationFromModel(
        model, geom_model, visual_model, q0, v0, args, add_floor=True
    )
elif args.backend == "mujoco":
    runMujocoXML(model_path, args)
