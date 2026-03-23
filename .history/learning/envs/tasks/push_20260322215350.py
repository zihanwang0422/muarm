"""Push task: push a box to a goal position on a table surface."""
from __future__ import annotations

import numpy as np

from learning.envs.tasks.base_task import BaseTask


class PushTask(BaseTask):
    """Push a free rigid body to a goal position."""

    def __init__(
        self,
        object_body: str = "push_box",
        goal_center: np.ndarray | None = None,
        goal_range: np.ndarray | None = None,
        distance_threshold: float = 0.05,
    ):
        self.object_body = object_body
        self.distance_threshold = distance_threshold
        self._center = np.asarray(goal_center or [0.58, 0.0, 0.025], dtype=np.float64)
        self._range = np.asarray(goal_range or [0.08, 0.18, 0.0], dtype=np.float64)

    def reset(self, model, data, np_random) -> None:
        x = 0.48 + float(np_random.uniform(-0.05, 0.05))
        y = float(np_random.uniform(-0.15, 0.15))
        z = 0.025
        try:
            body_id = data.body(self.object_body).id
            jnt_adr = data.model.body_jntadr[body_id]
            if jnt_adr >= 0:
                qpos_adr = data.model.jnt_qposadr[jnt_adr]
                data.qpos[qpos_adr: qpos_adr + 7] = [x, y, z, 1.0, 0.0, 0.0, 0.0]
        except Exception:
            pass

    def sample_goal(self, np_random) -> np.ndarray:
        delta = np_random.uniform(-self._range, self._range)
        return self._center + delta

    def get_obs(self, model, data, env) -> np.ndarray:
        ee = env.get_ee_pos()
        try:
            obj = data.body(self.object_body).xpos.copy()
        except Exception:
            obj = np.zeros(3)
        return np.concatenate([ee, obj])

    def achieved_goal(self, model, data, env) -> np.ndarray:
        try:
            return data.body(self.object_body).xpos.copy()
        except Exception:
            return np.zeros(3)

    def compute_reward(self, achieved_goal, desired_goal, info) -> float:
        ee_to_obj = info.get("ee_to_obj", 0.0)
        obj_to_goal = float(np.linalg.norm(achieved_goal - desired_goal))
        return -(obj_to_goal + 0.2 * ee_to_obj)

    def is_success(self, achieved_goal, desired_goal) -> bool:
        return np.linalg.norm(achieved_goal - desired_goal) < self.distance_threshold
