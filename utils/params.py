from etils import epath
import numpy as np

_ROOT_PATH = epath.Path(__file__).parent.parent

_ASSETS_DIR = _ROOT_PATH / "assets"
_LOGS_DIR = _ROOT_PATH / "logs"
_CONFIGS_DIR = _ROOT_PATH / "configs"

RED = np.array([232, 114, 84, 255]) / 255
GREEN = np.array([84, 232, 121, 255]) / 255
BLUE = np.array([96, 86, 232, 255]) / 255
BLACK = np.array([58, 60, 69, 255]) / 255
PINK = np.array([239, 47, 201, 255]) / 255
GREY = np.array([192, 201, 229, 255]) / 255
BEIGE = np.array([252, 247, 234, 255]) / 255
PURPLE = np.array([161, 34, 183, 255]) / 255

__all__ = [
    "RED",
    "GREEN",
    "BLUE",
    "BLACK",
    "PINK",
    "GREY",
    "BEIGE",
    "PURPLE",
    "_ASSETS_DIR",
    "_LOGS_DIR",
    "_CONFIGS_DIR",
]
