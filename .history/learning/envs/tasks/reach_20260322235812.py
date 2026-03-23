"""Reach task: move the robot EE to a random 3-D goal position."""
from __future__ import annotations

import numpy as np

from learning.envs.tasks.base_task import BaseTask


class ReachTask(BaseTask):
    """Move end-effector to a sampled goal position."""

    def __init__(
        self,
        goal_center: np.ndarray | None = None,
        goal_range: np.ndarray | None = None,
        distance_threshold: float = 0.04,
        ee_body: str = "hand",
    ):
        self.distance_threshold = distance_threshold
        self.ee_body = ee_body
        self._center = np.asarray(goal_center or [0.5, 0.0, 0.32], dtype=np.float64)
        self._range = np.asarray(goal_range or [0.12, 0.18, 0.10], dtype=np.float64)

    def reset(self, model, data, np_random) -> None:
        pass

    def sample_goal(self, np_random) -> np.ndarray:
        delta = np_random.uniform(-self._range, self._range)
        return self._center + delta

    def get_obs(self, model, data, env) -> np.ndarray:
        return env.get_ee_pos()

    def achieved_goal(self, model, data, env) -> np.ndarray:
        return env.get_ee_pos()

    def compute_reward(self, achieved_goal, desired_goal, info) -> float:
        dist = float(np.linalg.norm(achieved_goal - desired_goal))
        return -dist

    def is_success(self, achieved_goal, desired_goal) -> bool:
        return np.linalg.norm(achieved_goal - desired_goal) < self.distance_threshold
