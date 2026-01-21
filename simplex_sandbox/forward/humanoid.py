import pinocchio as pin
import numpy as np
from simple_sandbox.utils.sim_utils import (
    # addFloor,
    runSimpleSimulationFromModel,
    runMujocoXML,
    SimulationArgs,
)

args = SimulationArgs().parse_args()
np.random.seed(args.seed)
pin.seed(args.seed)
model_path = "./simple_sandbox/robots/humanoid.xml"

if args.backend == "simple":
    # Create model
    print("Loading mj robot description...")
    model = pin.buildModelFromMJCF(model_path)
    geom_model = pin.buildGeomFromMJCF(model, model_path, pin.COLLISION)
    visual_model = pin.buildGeomFromMJCF(model, model_path, pin.VISUAL)

    for joint in model.joints:
        print("=============================")
        print(joint)
        if joint.shortname() == "JointModelComposite":
            print(joint.extract())

    # Initial state
    q0 = model.referenceConfigurations["qpos0"]
    if args.random_init_vel:
        v0 = np.random.randn(model.nv)
    else:
        v0 = np.zeros(model.nv)

    if args.pd_controller:
        args.pd_controller_qtar = q0.copy()
        args.pd_controller_vtar = np.zeros(model.nv - 6)  # control all except freeflyer

    runSimpleSimulationFromModel(
        model, geom_model, visual_model, q0, v0, args, add_floor=False
    )
elif args.backend == "mujoco":
    runMujocoXML(model_path, args)
