"""Base Panda MuJoCo environment with task-env decoupling and viewer callback."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kinematic.panda_kinematics import PandaKinematics
from learning.panda_rl.utils.visualize import clear_overlay, draw_polyline, draw_sphere


class BaseRobotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        model_path: str,
        panda_model_path: str,
        task,
        render_mode: str | None = None,
        n_substeps: int = 20,
        horizon: int = 150,
        action_scale: float = 0.03,
    ):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.task = task
        self.render_mode = render_mode
        self.n_substeps = n_substeps
        self.horizon = horizon
        self.action_scale = action_scale

        self.kin = PandaKinematics(ee_frame="hand")
        self.kin.build_from_mjcf(str(panda_model_path))

        self.viewer = None
        self.goal = np.zeros(3, dtype=np.float64)
        self.step_count = 0
        self._ee_traj = []

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(-np.inf, np.inf, shape=(32,), dtype=np.float32),
                "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        )

    def _target_tf(self, xyz):
        tf = np.eye(4, dtype=np.float64)
        tf[:3, :3] = np.diag([1.0, -1.0, -1.0])
        tf[:3, 3] = xyz
        return tf

    def _workspace_clip(self, xyz):
        low = np.array([0.25, -0.45, 0.10], dtype=np.float64)
        high = np.array([0.75, 0.45, 0.70], dtype=np.float64)
        return np.clip(xyz, low, high)

    def _apply_cartesian_action(self, action):
        action = np.asarray(action, dtype=np.float64)
        ee = self.data.body("hand").xpos.copy()
        target = self._workspace_clip(ee + self.action_scale * action[:3])

        q_seed = self.data.qpos[:7].copy()
        q_sol, _ = self.kin.ik(self._target_tf(target), q_init=q_seed, max_iters=100)
        self.data.ctrl[:7] = q_sol[:7]

        # actuator8: 0 close, 255 open
        grip = float(np.clip((action[3] + 1.0) * 0.5 * 255.0, 0.0, 255.0))
        if self.model.nu > 7:
            self.data.ctrl[7] = grip

    def _render_callback(self):
        """Called every frame to update target visualization and trajectory line."""
        if self.viewer is None:
            return

        clear_overlay(self.viewer)
        draw_sphere(self.viewer, self.goal, radius=0.025, rgba=[1.0, 0.1, 0.1, 0.6])

        ee = self.data.body("hand").xpos.copy()
        if not self._ee_traj or np.linalg.norm(ee - self._ee_traj[-1]) > 0.002:
            self._ee_traj.append(ee)
        if len(self._ee_traj) > 250:
            self._ee_traj.pop(0)
        draw_polyline(self.viewer, self._ee_traj, width=2.0, rgba=[0.1, 0.8, 1.0, 0.9])

        try:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
            self.model.site_pos[site_id] = self.goal
        except Exception:
            pass

        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self):
        raise NotImplementedError

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        del options

        self.data.qpos[:] = self.model.key_qpos[0].copy()
        self.data.ctrl[:] = self.model.key_ctrl[0].copy()
        self.data.qvel[:] = 0.0
        self.task.reset(self.model, self.data, self.np_random)

        self.goal = self.task.sample_goal(self.np_random)
        self.step_count = 0
        self._ee_traj = []

        mujoco.mj_forward(self.model, self.data)
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def step(self, action):
        self._apply_cartesian_action(action)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        achieved = self.task.achieved_goal(self.model, self.data, self)
        ee = self.data.body("hand").xpos.copy()
        reward = self.task.compute_reward(
            achieved,
            self.goal,
            info={"ee_to_obj": float(np.linalg.norm(ee - achieved))},
        )

        self.step_count += 1
        terminated = bool(self.task.is_success(achieved, self.goal))
        truncated = self.step_count >= self.horizon

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return None

        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 2.2
            self.viewer.cam.azimuth = -40
            self.viewer.cam.elevation = -24

        self._render_callback()
        self.viewer.sync()
        time.sleep(self.model.opt.timestep)
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
