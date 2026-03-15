"""Gymnasium environment for Panda obstacle avoidance with PPO.

The agent controls 7 joint positions to reach a target position
while avoiding a spherical obstacle in the workspace.
"""
import numpy as np
import mujoco
import mujoco.viewer
import time
import os
from typing import Optional

import gymnasium as gym
from gymnasium import spaces


def _flag_path(name="rl_obstacle_visu_flag"):
    return os.path.join("/tmp", name)


class PandaObstacleAvoidEnv(gym.Env):
    """Gymnasium environment: Panda reaches target while avoiding obstacles.

    Observation: [joint_pos(7), goal_pos(3), obstacle_pos(3), obstacle_radius(1)]  dim=14
    Action: [-1, 1]^7 mapped to joint position ranges
    Reward: distance-based + collision penalty + smoothness penalty

    Example::

        env = PandaObstacleAvoidEnv(render_mode="human")
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.close()
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None,
                 scene_xml="models/franka_emika_panda/scene.xml",
                 obstacle_radius=0.06):
        super().__init__()
        self.render_mode = render_mode
        self.visualize = render_mode == "human"

        if self.visualize:
            flag = _flag_path()
            if os.path.exists(flag):
                self.visualize = False
            else:
                with open(flag, "w") as f:
                    f.write("1")

        self.handle = None
        self.scene_xml = scene_xml
        self.obstacle_radius = obstacle_radius

        # Inject an obstacle into the model
        self._obstacle_pos = np.array([0.3, 0.0, 0.4], dtype=np.float32)
        self._setup_model()

        self.ee_body_name = "hand"
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name)

        if self.model.nkey > 0:
            self.home_qpos = np.array(self.model.key_qpos[0][:7], dtype=np.float32)
        else:
            self.home_qpos = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4], dtype=np.float32)

        self.goal_threshold = 0.05
        self.goal_position = np.array([0.4, -0.3, 0.4], dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

        self.np_random = np.random.default_rng()
        self.last_action = self.home_qpos.copy()
        self.start_time = 0.0

    def _setup_model(self):
        """Load model and inject obstacle geom."""
        from xml.etree import ElementTree as ET

        tree = ET.parse(self.scene_xml)
        root = tree.getroot()
        worldbody = root.find("worldbody")

        obstacle = ET.SubElement(worldbody, "geom")
        obstacle.set("name", "obstacle_0")
        obstacle.set("type", "sphere")
        obstacle.set("size", f"{self.obstacle_radius:.3f}")
        obstacle.set("pos", f"{self._obstacle_pos[0]:.3f} {self._obstacle_pos[1]:.3f} {self._obstacle_pos[2]:.3f}")
        obstacle.set("contype", "1")
        obstacle.set("conaffinity", "1")
        obstacle.set("mass", "0.0")
        obstacle.set("rgba", "0.8 0.2 0.2 0.8")

        new_path = self.scene_xml.replace(".xml", "_obstacle_env.xml")
        tree.write(new_path, encoding="utf-8", xml_declaration=True)

        self.model = mujoco.MjModel.from_xml_path(new_path)
        self.data = mujoco.MjData(self.model)

        # Find obstacle geom id
        self.obstacle_geom_id = -1
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name == "obstacle_0":
                self.obstacle_geom_id = i
                break

        if self.visualize:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3.0
            self.handle.cam.azimuth = -45.0
            self.handle.cam.elevation = -30.0

    def _render_goal(self):
        if not self.visualize or self.handle is None:
            return
        self.handle.user_scn.ngeom = 1
        mujoco.mjv_initGeom(
            self.handle.user_scn.geoms[0],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.03, 0, 0],
            pos=self.goal_position.astype(np.float64),
            mat=np.eye(3).flatten(),
            rgba=np.array([0.1, 0.3, 0.9, 0.8], dtype=np.float32),
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.home_qpos
        if self.model.nq > 7:
            self.data.qpos[7:9] = [0.04, 0.04]
        mujoco.mj_forward(self.model, self.data)

        # Randomize goal y-position
        self.goal_position[1] = self.np_random.uniform(-0.3, 0.3)

        # Randomize obstacle y-position
        self._obstacle_pos[1] = self.np_random.uniform(-0.3, 0.3)
        if self.obstacle_geom_id >= 0:
            self.model.geom_pos[self.obstacle_geom_id] = self._obstacle_pos
        mujoco.mj_step(self.model, self.data)

        self.last_action = self.home_qpos.copy()
        self.start_time = time.time()
        self._render_goal()

        return self._get_obs(), {}

    def _get_obs(self):
        joint_pos = self.data.qpos[:7].copy().astype(np.float32)
        return np.concatenate([
            joint_pos,
            self.goal_position,
            self._obstacle_pos,
            np.array([self.obstacle_radius], dtype=np.float32),
        ])

    def step(self, action):
        # Scale action
        joint_ranges = self.model.jnt_range[:7]
        scaled = np.zeros(7, dtype=np.float32)
        for i in range(7):
            lo, hi = joint_ranges[i]
            scaled[i] = lo + (action[i] + 1) * 0.5 * (hi - lo)

        self.data.ctrl[:7] = scaled
        if self.model.nq > 7:
            self.data.qpos[7:9] = [0.04, 0.04]
        mujoco.mj_step(self.model, self.data)

        ee_pos = self.data.body(self.ee_id).xpos.copy()
        dist = np.linalg.norm(ee_pos - self.goal_position)
        has_collision = self.data.ncon > 0

        # Distance reward
        if dist < self.goal_threshold:
            reward = 20.0
        elif dist < 2 * self.goal_threshold:
            reward = 10.0
        else:
            reward = 1.0 / (1.0 + dist)

        # Collision penalty
        if has_collision:
            reward -= 10.0 * self.data.ncon

        # Smoothness penalty
        reward -= 0.001 * np.linalg.norm(action - self.last_action)

        # Joint limit penalty
        for i in range(7):
            lo, hi = self.model.jnt_range[:7][i]
            q = self.data.qpos[i]
            if q < lo:
                reward -= 0.5 * (lo - q)
            elif q > hi:
                reward -= 0.5 * (q - hi)

        # Time penalty
        reward -= 0.001 * (time.time() - self.start_time)

        self.last_action = action.copy()

        terminated = False
        if has_collision:
            reward -= 10.0
            terminated = True
        elif dist < self.goal_threshold:
            terminated = True
        elif time.time() - self.start_time > 20.0:
            reward -= 10.0
            terminated = True

        if self.visualize and self.handle is not None:
            self.handle.sync()
            time.sleep(0.01)

        info = {
            'is_success': not has_collision and dist < self.goal_threshold,
            'distance_to_goal': float(dist),
            'collision': has_collision,
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
