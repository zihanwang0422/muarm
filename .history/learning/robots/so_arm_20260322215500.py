"""SO-ARM100 (TRS) robot wrapper.

6-DOF serial arm. Action space is 6-D joint velocity delta (normalized).
EE body: Fixed_Jaw.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning.envs.base_env import MuJocoRobotEnv

_ROOT_DIR = Path(__file__).resolve().parents[2]
_SOARM_SCENE = _ROOT_DIR / "models" / "trs_so_arm100" / "scene_sim.xml"

# Joint velocity limits (rad/s, normalized action maps to ±this)
_VEL_LIMIT = 1.0  # rad/step
_N_JOINTS = 6


class SoArm100Env(MuJocoRobotEnv):
    """TRS SO-ARM100 manipulation environment.

    Action: 6-D joint velocity delta in [-1, 1] (maps to ±_VEL_LIMIT)
    EE: Fixed_Jaw body
    """

    def __init__(
        self,
        task,
        model_path: str | None = None,
        render_mode: str | None = None,
        n_substeps: int = 10,
        horizon: int = 200,
        action_scale: float = 0.05,
        obs_dim: int = 32,
    ):
        self.action_scale = action_scale
        self.obs_dim = obs_dim
        super().__init__(
            model_path=model_path or str(_SOARM_SCENE),
            task=task,
            render_mode=render_mode,
            n_substeps=n_substeps,
            horizon=horizon,
        )

    # ------------------------------------------------------------------
    # Abstract implementations
    # ------------------------------------------------------------------

    def _build_spaces(self) -> None:
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(_N_JOINTS,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        })

    def _get_obs(self) -> dict:
        ee_pos = self.get_ee_pos()
        qpos = self.data.qpos[:_N_JOINTS].copy()
        qvel = self.data.qvel[:_N_JOINTS].copy()
        task_obs = self.task.get_obs(self.model, self.data, self)
        achieved = self.task.achieved_goal(self.model, self.data, self)

        raw = np.concatenate([ee_pos, qpos, qvel, task_obs]).astype(np.float32)
        obs_vec = np.zeros(self.obs_dim, dtype=np.float32)
        n = min(len(raw), self.obs_dim)
        obs_vec[:n] = raw[:n]

        return {
            "observation": obs_vec,
            "achieved_goal": achieved.astype(np.float32),
            "desired_goal": self._goal.astype(np.float32),
        }

    def _apply_action(self, action: np.ndarray) -> None:
        """Apply joint velocity commands via position integration."""
        action = np.asarray(action, dtype=np.float64)
        q = self.data.qpos[:_N_JOINTS].copy()
        q_new = q + self.action_scale * action[:_N_JOINTS]

        # Clip to joint limits
        lb = self.model.jnt_range[:_N_JOINTS, 0]
        ub = self.model.jnt_range[:_N_JOINTS, 1]
        valid_limits = ub > lb
        q_new = np.where(valid_limits, np.clip(q_new, lb, ub), q_new)

        self.data.ctrl[:_N_JOINTS] = q_new

    def get_ee_pos(self) -> np.ndarray:
        try:
            return self.data.body("Fixed_Jaw").xpos.copy()
        except Exception:
            return np.zeros(3)

    def get_ee_body(self) -> str:
        return "Fixed_Jaw"
