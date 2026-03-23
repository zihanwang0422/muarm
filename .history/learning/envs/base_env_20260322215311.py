"""Generic MuJoCo manipulation environment base class.

Supports any robot described by an MJCF file. The concrete robot wrapper
provides robot-specific helpers (action application, observation building,
EE position query). The task object provides episode logic.

Design:
  MuJocoRobotEnv  (this file)  ← generic simulation loop + rendering
      └── RobotWrapper         ← robot-specific ctrl / obs / kin  (robots/)
              └── Task          ← episode reward / reset / success  (tasks/)
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from learning.utils.visualize import clear_overlay, draw_polyline, draw_sphere


class MuJocoRobotEnv(gym.Env, ABC):
    """Robot-agnostic MuJoCo Gymnasium environment.

    Subclasses must implement:
      - _build_spaces()       → set self.action_space / self.observation_space
      - _get_obs()            → return observation dict
      - _apply_action(action) → apply action to ctrl
      - get_ee_pos()          → return EE position as (3,) array
      - get_ee_body()         → return MuJoCo body name string for EE

    Optionally override:
      - _render_callback()    → custom per-frame overlays
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        model_path: str,
        task,
        *,
        render_mode: str | None = None,
        n_substeps: int = 20,
        horizon: int = 200,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.task = task
        self.render_mode = render_mode
        self.n_substeps = n_substeps
        self.horizon = horizon
        self.dt = self.model.opt.timestep * n_substeps  # wall-clock step dt

        self._viewer = None
        self._goal = np.zeros(3, dtype=np.float64)
        self._step_count = 0
        self._ee_traj: list[np.ndarray] = []

        self._build_spaces()

    # ------------------------------------------------------------------
    # Abstract interface – robot wrappers implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_spaces(self) -> None:
        """Set self.action_space and self.observation_space."""

    @abstractmethod
    def _get_obs(self) -> dict:
        """Return Gymnasium observation dict."""

    @abstractmethod
    def _apply_action(self, action: np.ndarray) -> None:
        """Write action into self.data.ctrl / qpos."""

    @abstractmethod
    def get_ee_pos(self) -> np.ndarray:
        """Return end-effector world position (3,)."""

    @abstractmethod
    def get_ee_body(self) -> str:
        """Return MuJoCo body name used as end-effector."""

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)

        # Reset to keyframe 0 if available, else zeros
        if self.model.nkey > 0:
            self.data.qpos[:] = self.model.key_qpos[0].copy()
            if hasattr(self.model, "key_ctrl") and self.model.nkey > 0:
                try:
                    self.data.ctrl[:] = self.model.key_ctrl[0].copy()
                except Exception:
                    self.data.ctrl[:] = 0.0
        else:
            self.data.qpos[:] = 0.0
            self.data.ctrl[:] = 0.0
        self.data.qvel[:] = 0.0

        self.task.reset(self.model, self.data, self.np_random)
        self._goal = self.task.sample_goal(self.np_random)
        self._step_count = 0
        self._ee_traj = []

        mujoco.mj_forward(self.model, self.data)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._apply_action(action)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        achieved = self.task.achieved_goal(self.model, self.data, self)
        ee = self.get_ee_pos()
        info = {"ee_to_obj": float(np.linalg.norm(ee - achieved))}
        reward = float(self.task.compute_reward(achieved, self._goal, info))

        self._step_count += 1
        terminated = bool(self.task.is_success(achieved, self._goal))
        truncated = self._step_count >= self.horizon

        if self.render_mode == "human":
            self.render()

        info["is_success"] = terminated
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return None
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.cam.distance = 2.2
            self._viewer.cam.azimuth = -40
            self._viewer.cam.elevation = -24
        self._render_callback()
        self._viewer.sync()
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------------
    # Rendering helper – subclasses may override
    # ------------------------------------------------------------------

    def _render_callback(self) -> None:
        if self._viewer is None:
            return
        clear_overlay(self._viewer)
        draw_sphere(self._viewer, self._goal, radius=0.025, rgba=[1.0, 0.1, 0.1, 0.6])

        ee = self.get_ee_pos()
        if not self._ee_traj or np.linalg.norm(ee - self._ee_traj[-1]) > 0.002:
            self._ee_traj.append(ee.copy())
        if len(self._ee_traj) > 300:
            self._ee_traj.pop(0)
        draw_polyline(self._viewer, self._ee_traj, width=2.0, rgba=[0.1, 0.8, 1.0, 0.9])

        # Sync target site if it exists in the model
        try:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
            if site_id >= 0:
                self.model.site_pos[site_id] = self._goal
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers shared by robot wrappers
    # ------------------------------------------------------------------

    def workspace_clip(self, xyz: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        return np.clip(xyz, low, high)

    @property
    def goal(self) -> np.ndarray:
        return self._goal
