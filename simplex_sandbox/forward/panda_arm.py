import pinocchio as pin
import numpy as np
from robot_descriptions.loaders.pinocchio import load_robot_description as plr
from simple_sandbox.utils.sim_utils import (
    SimulationArgs,
    runSimpleSimulationFromModel,
    runMujocoXML,
)


class ScriptArgs(SimulationArgs):
    add_floor: bool = False


args = ScriptArgs().parse_args()
np.random.seed(args.seed)
pin.seed(args.seed)
model_path = "panda_mj_description"

if args.backend == "simple":
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
    q0 = model.referenceConfigurations["home"]
    if args.random_init_vel:
        v0 = np.random.randn(model.nv) * 10
    else:
        v0 = np.zeros(model.nv)

    if args.pd_controller:
        args.pd_controller_qtar = q0.copy()
        args.pd_controller_vtar = np.zeros(model.nv)

    runSimpleSimulationFromModel(
        model, geom_model, visual_model, q0, v0, args, add_floor=args.add_floor
    )
elif args.backend == "mujoco":
    runMujocoXML(model_path, args)
