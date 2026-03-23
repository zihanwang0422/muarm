"""Franka Panda envs that bind robot control with task modules."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from learning.panda_rl.envs.base_env import BaseRobotEnv
from learning.panda_rl.envs.tasks.pick_place import PickPlaceTask
from learning.panda_rl.envs.tasks.push import PushTask
from learning.panda_rl.envs.tasks.reach import ReachTask


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "learning" / "panda_rl" / "assets"
PANDA_XML = ROOT / "models" / "franka_emika_panda" / "panda.xml"


class FrankaPandaEnv(BaseRobotEnv):
    def __init__(self, task_name="reach", render_mode=None, **kwargs):
        task_name = task_name.lower()
        if task_name == "reach":
            task = ReachTask()
            model_path = ASSETS / "reach.xml"
        elif task_name == "push":
            task = PushTask()
            model_path = ASSETS / "push.xml"
        elif task_name in {"pick_place", "pick-place", "pickplace"}:
            task = PickPlaceTask()
            model_path = ASSETS / "pick_place.xml"
        else:
            raise ValueError(f"Unsupported task: {task_name}")

        self.task_name = task_name
        super().__init__(
            model_path=str(model_path),
            panda_model_path=str(PANDA_XML),
            task=task,
            render_mode=render_mode,
            **kwargs,
        )

    def _get_obs(self):
        ee_pos = self.data.body("hand").xpos.copy()
        qpos = self.data.qpos[:9].copy()
        qvel = self.data.qvel[:9].copy()

        task_obs = self.task.get_obs(self.model, self.data, self)
        achieved = self.task.achieved_goal(self.model, self.data, self)

        obs = np.concatenate([ee_pos, qpos, qvel, task_obs]).astype(np.float32)
        out = {
            "observation": np.zeros(32, dtype=np.float32),
            "achieved_goal": achieved.astype(np.float32),
            "desired_goal": self.goal.astype(np.float32),
        }

        n = min(len(obs), 32)
        out["observation"][:n] = obs[:n]
        return out
