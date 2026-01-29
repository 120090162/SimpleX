import pinocchio as pin
import numpy as np
from robot_descriptions.loaders.pinocchio import load_robot_description as plr
from simplex_sandbox.utils.sim_utils import (
    addFloor,
    runSimpleXSimulationFromModel,
    SimulationArgs,
)


class ScriptArgs(SimulationArgs):
    remove_anchor_constraints: bool = False
    display: bool = True  # display the simulation
    backend: str = "simplex"
    display_traj: bool = True  # display the trajectory
    pd_controller: bool = False  # use a PD controller
    random_init_vel: bool = False  # random initial velocity


args = ScriptArgs().parse_args()
np.random.seed(args.seed)
pin.seed(args.seed)

print("Loading mj robot description...")
robot_mj = plr("go2_mj_description")
model = robot_mj.model
geom_model = robot_mj.collision_model
visual_model = robot_mj.visual_model

# define some specifc collision pairs, need to add floor first
addFloor(geom_model, visual_model)
# geom_model.removeAllCollisionPairs()
# foot_names = ["RL", "FR", "FL", "RR"]
# ground_name = "floor"

# for foot_name in foot_names:
#     foot_id = geom_model.getGeometryId(foot_name)
#     ground_id = geom_model.getGeometryId(ground_name)
#     if foot_id >= geom_model.ngeoms or ground_id >= geom_model.ngeoms:
#         print(
#             f"Warning: {foot_name} or {ground_name} not found in GeometryModel. Skipping..."
#         )
#         continue
#     col_pair = pin.CollisionPair(foot_id, ground_id)
#     print(
#         f"Adding collision pair between {foot_name} [id = {foot_id}] and {ground_name} [id = {ground_id}]"
#     )
#     geom_model.addCollisionPair(col_pair)

# for joint in model.joints:
#     print("=============================")
#     print(joint)
# if joint.shortname() == "JointModelComposite":
#     print(joint.extract())

# Initial state
q0 = model.referenceConfigurations["qpos0"]
if args.random_init_vel:
    v0 = np.random.randn(model.nv)
else:
    v0 = np.zeros(model.nv)

if args.pd_controller:
    args.pd_controller_qtar = q0.copy()
    args.pd_controller_vtar = np.zeros(
        model.nv - 6
    )  # control everything except freeflyer
    args.pd_controller_kp = 60.0
    args.pd_controller_kd = 1.0

runSimpleXSimulationFromModel(
    model,
    geom_model,
    visual_model,
    q0,
    v0,
    args,
    add_floor=False,
    add_system_collision_pairs=True,
)
