"""Environment registry – maps (robot, task) string keys to env factory functions.

Usage:
    from learning.envs.registry import make_env
    env = make_env("panda", "reach", render_mode="human")
"""
from __future__ import annotations

from typing import Any

from learning.envs.tasks.reach import ReachTask
from learning.envs.tasks.push import PushTask
from learning.envs.tasks.pick_place import PickPlaceTask


# -------------------- task factory --------------------

_TASK_MAP = {
    "reach": ReachTask,
    "push": PushTask,
    "pick_place": PickPlaceTask,
}


def _make_task(task_name: str, task_kwargs: dict | None = None):
    key = task_name.replace("-", "_").lower()
    if key not in _TASK_MAP:
        raise ValueError(f"Unknown task '{task_name}'. Available: {list(_TASK_MAP)}")
    return _TASK_MAP[key](**(task_kwargs or {}))


# -------------------- robot factory --------------------

def make_env(
    robot: str,
    task: str,
    *,
    render_mode: str | None = None,
    task_kwargs: dict | None = None,
    **env_kwargs: Any,
):
    """Create a (robot, task) environment.

    Args:
        robot:        "panda" | "so_arm"
        task:         "reach" | "push" | "pick_place"
        render_mode:  None (headless) | "human"
        task_kwargs:  extra kwargs forwarded to the task constructor
        **env_kwargs: extra kwargs forwarded to the env constructor

    Returns:
        A gymnasium.Env instance.
    """
    robot_key = robot.lower().replace("-", "_")
    task_obj = _make_task(task, task_kwargs)

    if robot_key == "panda":
        from learning.robots.panda import FrankaPandaEnv
        return FrankaPandaEnv(
            task=task_obj,
            task_name=task.replace("-", "_"),
            render_mode=render_mode,
            **env_kwargs,
        )
    elif robot_key in {"so_arm", "so_arm100"}:
        from learning.robots.so_arm import SoArm100Env
        return SoArm100Env(
            task=task_obj,
            render_mode=render_mode,
            **env_kwargs,
        )
    else:
        raise ValueError(f"Unknown robot '{robot}'. Available: panda, so_arm")
