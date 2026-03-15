"""Core simulation and control modules.

MuJoCoViewer requires mujoco to be installed.
PIDController and MPCController can be used independently.
"""
from .pid_controller import PIDController
from .mpc_controller import MPCController

__all__ = ["PIDController", "MPCController"]

try:
    from .mujoco_viewer import MuJoCoViewer
    __all__.append("MuJoCoViewer")
except ImportError:
    pass
