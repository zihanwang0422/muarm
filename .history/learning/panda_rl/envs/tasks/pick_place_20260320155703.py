"""Pick-place task logic only (robot-agnostic)."""
from __future__ import annotations

import numpy as np


class PickPlaceTask:
    def __init__(self, object_body="pick_obj", distance_threshold=0.06):
        self.object_body = object_body
        self.distance_threshold = distance_threshold

    def reset(self, model, data, np_random) -> None:
        del model
        x = 0.5 + np_random.uniform(-0.05, 0.05)
        y = np_random.uniform(-0.12, 0.12)
        z = 0.025
        body_id = data.body(self.object_body).id
        jnt_adr = data.model.body_jntadr[body_id]
        if jnt_adr >= 0:
            qpos_adr = data.model.jnt_qposadr[jnt_adr]
            data.qpos[qpos_adr:qpos_adr + 7] = np.array([x, y, z, 1.0, 0.0, 0.0, 0.0])

    def get_obs(self, model, data, utils) -> np.ndarray:
        del model, utils
        ee = data.body("hand").xpos.copy()
        obj = data.body(self.object_body).xpos.copy()
        return np.concatenate([ee, obj])

    def achieved_goal(self, model, data, utils) -> np.ndarray:
        del model, utils
        return data.body(self.object_body).xpos.copy()

    def sample_goal(self, np_random) -> np.ndarray:
        return np.array([0.55, 0.0, 0.25]) + np_random.uniform(
            low=[-0.10, -0.15, -0.02], high=[0.10, 0.15, 0.12]
        )

    def compute_reward(self, achieved_goal, desired_goal, info) -> float:
        ee_to_obj = info.get("ee_to_obj", 0.0)
        obj_to_goal = np.linalg.norm(achieved_goal - desired_goal)
        height_bonus = 0.2 if achieved_goal[2] > 0.08 else 0.0
        return -float(obj_to_goal + 0.1 * ee_to_obj) + height_bonus

    def is_success(self, achieved_goal, desired_goal) -> bool:
        return np.linalg.norm(achieved_goal - desired_goal) < self.distance_threshold
