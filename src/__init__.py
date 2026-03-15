"""Core simulation and control modules.

MuJoCoViewer requires mujoco to be installed.
PIDController and MPCController can be used independently.
"""
from .pid_controller import PIDController
from .mpc_controller import MPCController

__all__ = ["PIDController", "MPCController"]

try:
    from .mujoco_viewer import MuJoCoViewer
    from .kinematics_vis import KinematicsVisualizer
    __all__ += ["MuJoCoViewer", "KinematicsVisualizer"]
except ImportError:
    pass
