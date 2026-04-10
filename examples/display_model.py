import mujoco
import mujoco.viewer
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

_XML_PATH = flags.DEFINE_string(
    "xml_path",
    None,
    f"Path to the XML model file.",
)


def main(argv):
    del argv  # Unused.

    if _XML_PATH.value is None:
        raise ValueError("Please provide an XML path with --xml_path flag.")

    model_path = _ASSETS_DIR / f"{_XML_PATH.value}"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(LOGGER.INFO + f"Loading Mujoco model from: {model_path}")

    try:
        model = mujoco.MjModel.from_xml_path(model_path.as_posix())
        data = mujoco.MjData(model)

        model.opt.timestep = 0.001
        sim_fps = 1000.0
        render_fps = 30.0
        render_substeps = int(sim_fps / render_fps)

        # Reset to keyframe 0 if available (usually standing pose)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
            print(LOGGER.INFO + "Reset to keyframe 0")

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

                if step_counter % render_substeps == 0:
                    viewer.sync()

                step_counter += 1
                rate_limiter.sleep()

    except Exception as e:
        print(LOGGER.ERROR + f"Failed to load/visualize: {e}")


if __name__ == "__main__":
    app.run(main)
