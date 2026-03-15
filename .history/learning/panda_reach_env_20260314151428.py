"""Gymnasium environment for Panda end-effector reaching with PPO.

The agent controls 7 joint positions to reach a randomly sampled target
position in the workspace.
"""
import numpy as np
import mujoco
import mujoco.viewer
import time
import os
from typing import Optional

import gymnasium as gym
from gymnasium import spaces


def _flag_path(name="rl_visu_flag"):
    return os.path.join("/tmp", name)


class PandaReachEnv(gym.Env):
    """Gymnasium environment: Panda reaches a target 3D position.

    Observation: [joint_pos(7), goal_pos(3)]  dim=10
    Action: [-1, 1]^7 mapped to joint position ranges
    Reward: distance-based + orientation + smoothness penalties

    Example::

        env = PandaReachEnv(render_mode="human")
        obs, info = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                obs, info = env.reset()
        env.close()
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None,
                 scene_xml="models/franka_emika_panda/scene.xml"):
        super().__init__()
        self.render_mode = render_mode
        self.visualize = render_mode == "human"

        # Only allow one visualized env when using SubprocVecEnv
        if self.visualize:
            flag = _flag_path()
            if os.path.exists(flag):
                self.visualize = False
            else:
                with open(flag, "w") as f:
                    f.write("1")

        self.handle = None
        self.scene_xml = scene_xml
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data = mujoco.MjData(self.model)

        if self.visualize:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3.0
            self.handle.cam.azimuth = -45.0
            self.handle.cam.elevation = -30.0

        self.ee_body_name = "hand"
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name)

        # Home position from keyframe
        if self.model.nkey > 0:
            self.home_qpos = np.array(self.model.key_qpos[0][:7], dtype=np.float32)
        else:
            self.home_qpos = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4], dtype=np.float32)

        self.goal_threshold = 0.05
        self.workspace = {'x': [0.2, 0.7], 'y': [-0.4, 0.4], 'z': [0.05, 0.6]}

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self.goal = np.zeros(3, dtype=np.float32)
        self.np_random = np.random.default_rng()
        self.prev_action = np.zeros(7, dtype=np.float32)
        self.start_time = 0.0

    def _sample_goal(self):
        """Sample a valid target position in workspace."""
        for _ in range(100):
            goal = self.np_random.uniform(
                low=[self.workspace['x'][0], self.workspace['y'][0], self.workspace['z'][0]],
                high=[self.workspace['x'][1], self.workspace['y'][1], self.workspace['z'][1]],
            )
            return goal.astype(np.float32)
        return np.array([0.5, 0.0, 0.3], dtype=np.float32)

    def _render_goal(self):
        if not self.visualize or self.handle is None:
            return
        self.handle.user_scn.ngeom = 1
        mujoco.mjv_initGeom(
            self.handle.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.03, 0, 0],
            pos=self.goal.astype(np.float64),
            mat=np.eye(3).flatten(),
            rgba=np.array([0.1, 0.1, 0.9, 0.9], dtype=np.float32),
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.home_qpos
        mujoco.mj_forward(self.model, self.data)

        self.goal = self._sample_goal()
        self.prev_action = np.zeros(7, dtype=np.float32)
        self.start_time = time.time()
        self._render_goal()

        return self._get_obs(), {}

    def _get_obs(self):
        joint_pos = self.data.qpos[:7].copy().astype(np.float32)
        return np.concatenate([joint_pos, self.goal])

    def step(self, action):
        # Scale action from [-1,1] to joint ranges
        joint_ranges = self.model.jnt_range[:7]
        scaled = np.zeros(7, dtype=np.float32)
        for i in range(7):
            lo, hi = joint_ranges[i]
            scaled[i] = lo + (action[i] + 1) * 0.5 * (hi - lo)

        self.data.ctrl[:7] = scaled
        mujoco.mj_step(self.model, self.data)

        ee_pos = self.data.body(self.ee_id).xpos.copy()
        dist = np.linalg.norm(ee_pos - self.goal)

        # Reward
        if dist < self.goal_threshold:
            reward = 100.0
        elif dist < 2 * self.goal_threshold:
            reward = 50.0
        else:
            reward = 1.0 / (1.0 + dist)

        # Smoothness penalty
        reward -= 0.1 * np.linalg.norm(action - self.prev_action)

        # Joint limit penalty
        for i in range(7):
            lo, hi = self.model.jnt_range[:7][i]
            q = self.data.qpos[i]
            if q < lo:
                reward -= 0.5 * (lo - q)
            elif q > hi:
                reward -= 0.5 * (q - hi)

        self.prev_action = action.copy()

        terminated = dist < self.goal_threshold
        if not terminated and time.time() - self.start_time > 20.0:
            reward -= 10.0
            terminated = True

        if self.visualize and self.handle is not None:
            self.handle.sync()
            time.sleep(0.01)

        info = {
            'is_success': dist < self.goal_threshold,
            'distance_to_goal': float(dist),
        }
        return self._get_obs(), float(reward), terminated, False, info

    def close(self):
        if self.visualize and self.handle is not None:
            self.handle.close()
            self.handle = None
        flag = _flag_path()
        if os.path.exists(flag):
            try:
                os.remove(flag)
            except OSError:
                pass
