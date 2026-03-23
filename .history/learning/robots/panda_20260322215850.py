"""Franka Panda robot wrapper.

Provides:
  - Cartesian delta-action via damped Jacobian IK (pinocchio)
  - Gripper control (normalized [-1, 1] → MuJoCo actuator range)
  - 9-DOF observation (7 arm + 2 finger joints + EE pos + goal)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from gymnasium import spaces

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kinematic.panda_kinematics import PandaKinematics
from learning.envs.base_env import MuJocoRobotEnv

# Workspace limits [m]
_WS_LOW = np.array([0.20, -0.50, 0.05], dtype=np.float64)
_WS_HIGH = np.array([0.80, 0.50, 0.75], dtype=np.float64)

# Model paths
_ROOT_DIR = Path(__file__).resolve().parents[2]
_PANDA_MJCF = _ROOT_DIR / "models" / "franka_emika_panda" / "panda.xml"
_ASSETS_DIR = _ROOT_DIR / "learning" / "assets"  # panda_reach.xml, panda_push.xml, ...


def _panda_model_path(task_name: str) -> Path:
    return _ASSETS_DIR / f"panda_{task_name}.xml"


class FrankaPandaEnv(MuJocoRobotEnv):
    """Franka Emika Panda manipulation environment.

    Action: 4-D cartesian delta + gripper  (all in [-1, 1])
    Observation: Dict with 'observation' (32,), 'achieved_goal' (3,), 'desired_goal' (3,)
    """

    def __init__(
        self,
        task,
        task_name: str = "reach",
        model_path: str | None = None,
        render_mode: str | None = None,
        n_substeps: int = 20,
        horizon: int = 200,
        action_scale: float = 0.03,
        obs_dim: int = 32,
    ):
        self.action_scale = action_scale
        self.obs_dim = obs_dim
        self.task_name = task_name

        resolved_model = model_path or str(_panda_model_path(task_name))

        # Build pinocchio kinematics from panda MJCF
        self._kin = PandaKinematics(ee_frame="hand")
        self._kin.build_from_mjcf(str(_PANDA_MJCF))

        super().__init__(
            model_path=resolved_model,
            task=task,
            render_mode=render_mode,
            n_substeps=n_substeps,
            horizon=horizon,
        )

    # ------------------------------------------------------------------
    # Abstract implementations
    # ------------------------------------------------------------------

    def _build_spaces(self) -> None:
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        })

    def _get_obs(self) -> dict:
        ee_pos = self.get_ee_pos()
        qpos = self.data.qpos[:9].copy()  # 7 arm + 2 finger
        qvel = self.data.qvel[:9].copy()
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
        action = np.asarray(action, dtype=np.float64)
        ee = self.get_ee_pos()
        target = np.clip(ee + self.action_scale * action[:3], _WS_LOW, _WS_HIGH)

        tf = np.eye(4, dtype=np.float64)
        tf[:3, :3] = np.diag([1.0, -1.0, -1.0])
        tf[:3, 3] = target

        q_seed = self.data.qpos[:7].copy()
        q_sol, _ = self._kin.ik(tf, q_init=q_seed, max_iters=100)
        self.data.ctrl[:7] = q_sol[:7]

        # Gripper: actuator index 7 (if present), 0=close 255=open
        if self.model.nu > 7:
            grip = float(np.clip((float(action[3]) + 1.0) * 0.5 * 255.0, 0.0, 255.0))
            self.data.ctrl[7] = grip

    def get_ee_pos(self) -> np.ndarray:
        return self.data.body("hand").xpos.copy()

    def get_ee_body(self) -> str:
        return "hand"
