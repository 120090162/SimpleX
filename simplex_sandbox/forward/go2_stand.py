import pinocchio as pin
import numpy as np
from robot_descriptions.loaders.pinocchio import load_robot_description as plr
from simplex_sandbox.utils.sim_utils import addFloor

from absl import app
from absl import flags

_SEED = flags.DEFINE_integer(
    "seed",
    42,
    "random seed",
)


def main(_):
    np.random.seed(_SEED.value)
    pin.seed(_SEED.value)

    print("Loading mj robot description...")
    robot_mj = plr("go2_mj_description")
    model = robot_mj.model
    geom_model = robot_mj.collision_model
    visual_model = robot_mj.visual_model

    # define some specifc collision pairs, need to add floor first
    addFloor(geom_model, visual_model)

    geom_model.removeAllCollisionPairs()
    foot_names = ["RL", "FR", "FL", "RR"]
    ground_name = "floor"

    for foot_name in foot_names:
        foot_id = geom_model.getGeometryId(foot_name)
        ground_id = geom_model.getGeometryId(ground_name)
        if foot_id >= geom_model.ngeoms or ground_id >= geom_model.ngeoms:
            print(
                f"Warning: {foot_name} or {ground_name} not found in GeometryModel. Skipping..."
            )
            continue
        col_pair = pin.CollisionPair(foot_id, ground_id)
        print(
            f"Adding collision pair between {foot_name} [id = {foot_id}] and {ground_name} [id = {ground_id}]"
        )
        geom_model.addCollisionPair(col_pair)

    # Get and print all collision objects to identify feet
    print("All collision geometry objects:")
    for gobj in geom_model.geometryObjects:
        print(f"  Name: {gobj.name}, Parent Joint ID: {gobj.parentJoint}")

    # Initial state
    q0 = model.referenceConfigurations["qpos0"]
    v0 = np.zeros(model.nv)

    print("q0:", q0)


if __name__ == "__main__":
    app.run(main)
