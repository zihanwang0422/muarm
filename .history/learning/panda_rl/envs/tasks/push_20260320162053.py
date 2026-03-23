"""Push task logic only (robot-agnostic)."""
from __future__ import annotations

import numpy as np


class PushTask:
    def __init__(self, object_body="push_box", distance_threshold=0.05):
        self.object_body = object_body
        self.distance_threshold = distance_threshold

    def reset(self, model, data, np_random) -> None:
        del model
        x = 0.48 + np_random.uniform(-0.05, 0.05)
        y = np_random.uniform(-0.15, 0.15)
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
        return np.array([0.58, 0.0, 0.025]) + np_random.uniform(
            low=[-0.08, -0.18, 0.0], high=[0.08, 0.18, 0.0]
        )

    def compute_reward(self, achieved_goal, desired_goal, info) -> float:
        ee_to_obj = info.get("ee_to_obj", 0.0)
        obj_to_goal = np.linalg.norm(achieved_goal - desired_goal)
        return -float(obj_to_goal + 0.2 * ee_to_obj)

    def is_success(self, achieved_goal, desired_goal) -> bool:
        return np.linalg.norm(achieved_goal - desired_goal) < self.distance_threshold
