import pinocchio as pin
import numpy as np
from simplex_sandbox.utils.sim_utils import (
    SimulationArgs,
    runSimpleXSimulationFromModel,
    runMujocoXML,
)


class ScriptArgs(SimulationArgs):
    bilateral_to_joint: bool = False  # by default the point anchor constraint is between two components of the same body part. With this option the anchor constraint is on the upper right joint
    display: bool = True  # display the simulation
    backend: str = "simplex"
    display_traj: bool = True  # display the trajectory
    pd_controller: bool = False  # use a PD controller
    random_init_vel: bool = True  # random initial velocity

args = ScriptArgs().parse_args()
if args.bilateral_to_joint:
    model_path = "./simplex_sandbox/robots/four_five_bar_linkage.xml"
else:
    model_path = "./simplex_sandbox/robots/four_bar_linkage.xml"

if args.backend == "simplex":
    # Create model
    model, anchor_constraints_dict, geom_model, visual_model = pin.buildModelsFromMJCF(
        model_path
    )

    # Initial state
    q0 = model.referenceConfigurations["qpos0"]
    if args.random_init_vel:
        v0 = np.random.randn(model.nv)
    else:
        v0 = np.zeros(model.nv)

    runSimpleXSimulationFromModel(
        model,
        geom_model,
        visual_model,
        q0,
        v0,
        args,
        add_floor=False,
        point_and_frame_anchor_constraints_dict=anchor_constraints_dict,
    )
elif args.backend == "mujoco":
    runMujocoXML(model_path, args)
