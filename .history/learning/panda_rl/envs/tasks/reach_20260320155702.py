"""Reach task logic only (robot-agnostic)."""
from __future__ import annotations

import numpy as np


class ReachTask:
    def __init__(self, distance_threshold=0.04):
        self.distance_threshold = distance_threshold

    def reset(self, model, data, np_random) -> None:
        del model, data, np_random

    def get_obs(self, model, data, utils) -> np.ndarray:
        del model, utils
        ee = data.body("hand").xpos.copy()
        return ee

    def achieved_goal(self, model, data, utils) -> np.ndarray:
        return self.get_obs(model, data, utils)

    def sample_goal(self, np_random) -> np.ndarray:
        center = np.array([0.5, 0.0, 0.32], dtype=np.float64)
        return center + np_random.uniform(low=[-0.12, -0.18, -0.10], high=[0.12, 0.18, 0.12])

    def compute_reward(self, achieved_goal, desired_goal, info) -> float:
        del info
        return -float(np.linalg.norm(achieved_goal - desired_goal))

    def is_success(self, achieved_goal, desired_goal) -> bool:
        return np.linalg.norm(achieved_goal - desired_goal) < self.distance_threshold
