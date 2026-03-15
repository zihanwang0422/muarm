"""
SO-ARM100 MuJoCo Simulator

Wraps the official TRS MJCF model + our enhanced scene (scene_sim.xml) with:
  - Position-actuator control (matching the official model)
  - Sensor readout (joint pos/vel, EE pose)
  - Trajectory execution with real-time viewer sync
  - Interactive 3D viewer
"""

import os
import numpy as np
from typing import Dict, List, Optional

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    raise ImportError("Install mujoco:  pip install mujoco")


class MuJoCoSimulator:
    """
    MuJoCo simulator for the SO-ARM100 using the official TRS model.

    The official model uses **position actuators** (kp=50, dampratio=1)
    so ctrl values are *desired joint angles* in radians.
    """

    MODEL_PATH = os.path.join(
        os.path.dirname(__file__), "models", "trs_so_arm100", "scene_sim.xml"
    )

    # 5 arm joints + 1 gripper
    JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    ACTUATOR_NAMES = JOINT_NAMES  # same names in official model
    N_ARM_JOINTS = 5  # first 5 are arm, last is gripper

    def __init__(self, model_path: Optional[str] = None, render: bool = True):
        path = model_path or self.MODEL_PATH
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model not found: {path}")

        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self._render = render
        self._viewer = None

        # Cache IDs
        self._jnt_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                         for n in self.JOINT_NAMES]
        self._act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                         for n in self.ACTUATOR_NAMES]

        # Sensor address cache
        self._sensor = {}
        sensor_names = (
            [f"pos_{n}" for n in self.JOINT_NAMES]
            + [f"vel_{n}" for n in self.JOINT_NAMES]
            + ["ee_pos", "ee_quat"]
        )
        for sname in sensor_names:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sname)
            if sid >= 0:
                self._sensor[sname] = (self.model.sensor_adr[sid], self.model.sensor_dim[sid])

        self.reset()

    # ─── core ────────────────────────────────────────────────────────

    def reset(self, arm_angles: Optional[List[float]] = None, jaw: float = 0.0):
        """Reset simulation. Optionally set initial arm angles and jaw."""
        mujoco.mj_resetData(self.model, self.data)
        if arm_angles is not None:
            for i in range(min(len(arm_angles), self.N_ARM_JOINTS)):
                qadr = self.model.jnt_qposadr[self._jnt_ids[i]]
                self.data.qpos[qadr] = arm_angles[i]
                self.data.ctrl[self._act_ids[i]] = arm_angles[i]
            # Jaw
            qadr = self.model.jnt_qposadr[self._jnt_ids[5]]
            self.data.qpos[qadr] = jaw
            self.data.ctrl[self._act_ids[5]] = jaw
        mujoco.mj_forward(self.model, self.data)

    def reset_to_keyframe(self, name: str = "home"):
        """Reset to a named keyframe defined in the MJCF."""
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        if kid < 0:
            raise ValueError(f"Keyframe '{name}' not found.")
        mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        # Sync ctrl to qpos so actuators track the keyframe
        for i in range(len(self.ACTUATOR_NAMES)):
            qadr = self.model.jnt_qposadr[self._jnt_ids[i]]
            self.data.ctrl[self._act_ids[i]] = self.data.qpos[qadr]
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def set_arm_target(self, angles: List[float]):
        """Set position-actuator targets for the 5 arm joints (radians)."""
        for i in range(min(len(angles), self.N_ARM_JOINTS)):
            self.data.ctrl[self._act_ids[i]] = angles[i]

    def set_jaw(self, angle: float):
        """Set jaw (gripper) target angle."""
        self.data.ctrl[self._act_ids[5]] = angle

    # ─── sensors ─────────────────────────────────────────────────────

    def _read(self, name: str) -> np.ndarray:
        adr, dim = self._sensor[name]
        return self.data.sensordata[adr:adr+dim].copy()

    def get_arm_positions(self) -> np.ndarray:
        """5 arm joint positions (rad)."""
        return np.array([self._read(f"pos_{n}")[0] for n in self.JOINT_NAMES[:5]])

    def get_arm_velocities(self) -> np.ndarray:
        return np.array([self._read(f"vel_{n}")[0] for n in self.JOINT_NAMES[:5]])

    def get_jaw_position(self) -> float:
        return float(self._read("pos_Jaw")[0])

    def get_ee_position(self) -> np.ndarray:
        """EE position [x,y,z] in metres."""
        return self._read("ee_pos")

    def get_ee_quaternion(self) -> np.ndarray:
        """EE quaternion [w,x,y,z] (MuJoCo convention)."""
        return self._read("ee_quat")

    def get_state(self) -> Dict:
        return {
            "time": self.data.time,
            "arm_pos": self.get_arm_positions(),
            "arm_vel": self.get_arm_velocities(),
            "jaw": self.get_jaw_position(),
            "ee_pos": self.get_ee_position(),
            "ee_quat": self.get_ee_quaternion(),
        }

    # ─── trajectory execution ────────────────────────────────────────

    def execute_trajectory(
        self,
        trajectory: List[List[float]],
        frequency: float = 50.0,
        jaw_angle: float = 0.0,
        record: bool = False,
    ) -> Optional[List[Dict]]:
        """
        Execute a joint-space trajectory.

        Position actuators track the target at each time step.

        Args:
            trajectory:  List of 5-element arm joint arrays.
            frequency:   Trajectory rate (Hz).
            jaw_angle:   Fixed jaw angle during execution.
            record:      If True, record states.

        Returns:
            List of state dicts if record=True.
        """
        dt_traj = 1.0 / frequency
        dt_sim = self.model.opt.timestep
        steps = max(1, int(dt_traj / dt_sim))
        states = [] if record else None

        if self._render and self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

        for wp in trajectory:
            self.set_arm_target(wp)
            self.set_jaw(jaw_angle)
            self.step(steps)

            if self._render and self._viewer is not None:
                self._viewer.sync()

            if record:
                states.append(self.get_state())

        return states

    # ─── viewer ──────────────────────────────────────────────────────

    def launch_viewer(self):
        """Blocking interactive viewer."""
        mujoco.viewer.launch(self.model, self.data)

    def launch_passive_viewer(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self._viewer

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
            import time
            time.sleep(0.1)  # Allow viewer thread to terminate gracefully before main thread exits

    # Do not use __del__ to close the viewer. 
    # Python GC tearing down C/C++ threads during interpreter exit often causes Segfaults.


if __name__ == "__main__":
    sim = MuJoCoSimulator(render=False)
    sim.reset_to_keyframe("home")
    print("Home state:", sim.get_state())
