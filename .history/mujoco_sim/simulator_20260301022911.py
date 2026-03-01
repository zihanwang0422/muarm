"""
SO-ARM100 MuJoCo Simulator

Wraps the MuJoCo physics engine to:
  - Load the MJCF arm model
  - Step the simulation with joint commands
  - Provide interactive 3D visualization
  - Execute joint-space trajectories
  - Read sensor data (joint pos/vel, EE pose)
"""

import os
import time
import numpy as np
from typing import Dict, List, Optional

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    raise ImportError(
        "MuJoCo Python bindings not found. Install with:\n"
        "  pip install mujoco"
    )


class MuJoCoSimulator:
    """
    MuJoCo-based simulator for the SO-ARM100 robotic arm.

    Provides both headless stepping and interactive visualization.
    """

    MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
    DEFAULT_MODEL = "so_arm100.xml"

    JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)]
    MOTOR_NAMES = [f"motor_{i}" for i in range(1, 7)]

    def __init__(self, model_path: Optional[str] = None, render: bool = True):
        """
        Args:
            model_path: Path to MJCF XML. Defaults to built-in model.
            render: Whether to open the interactive viewer.
        """
        if model_path is None:
            model_path = os.path.join(self.MODEL_DIR, self.DEFAULT_MODEL)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._render = render
        self._viewer = None

        # Cache actuator and sensor indices
        self._motor_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                           for n in self.MOTOR_NAMES]
        self._joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                           for n in self.JOINT_NAMES]

        # Sensor address cache
        self._sensor_addr = {}
        for name in (
            [f"pos_joint_{i}" for i in range(1, 7)]
            + [f"vel_joint_{i}" for i in range(1, 7)]
            + ["ee_pos", "ee_quat"]
        ):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            addr = self.model.sensor_adr[sid]
            dim = self.model.sensor_dim[sid]
            self._sensor_addr[name] = (addr, dim)

        # Reset to home
        self.reset()

    # ─────────────────────────────────────────────────────────────────
    # Core
    # ─────────────────────────────────────────────────────────────────

    def reset(self, angles: Optional[List[float]] = None):
        """Reset simulation. Optionally set initial joint positions."""
        mujoco.mj_resetData(self.model, self.data)
        if angles is not None:
            for i, jid in enumerate(self._joint_ids):
                qadr = self.model.jnt_qposadr[jid]
                self.data.qpos[qadr] = angles[i]
        mujoco.mj_forward(self.model, self.data)

    def step(self, n: int = 1):
        """Advance simulation by n steps."""
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def set_joint_positions(self, angles: List[float]):
        """
        Set target joint positions via position-like motor control.
        The actuators are torque motors, so we use a PD-style command.
        """
        for i, mid in enumerate(self._motor_ids):
            self.data.ctrl[mid] = angles[i]

    def set_joint_targets_pd(
        self,
        target_angles: List[float],
        kp: float = 50.0,
        kd: float = 5.0,
    ):
        """
        PD position control: compute torque = kp*(target-q) - kd*qdot.
        """
        for i in range(6):
            jid = self._joint_ids[i]
            qadr = self.model.jnt_qposadr[jid]
            vadr = self.model.jnt_dofadr[jid]
            q = self.data.qpos[qadr]
            qd = self.data.qvel[vadr]
            torque = kp * (target_angles[i] - q) - kd * qd
            self.data.ctrl[self._motor_ids[i]] = torque

    # ─────────────────────────────────────────────────────────────────
    # Sensor readout
    # ─────────────────────────────────────────────────────────────────

    def _read_sensor(self, name: str) -> np.ndarray:
        addr, dim = self._sensor_addr[name]
        return self.data.sensordata[addr: addr + dim].copy()

    def get_joint_positions(self) -> np.ndarray:
        """Current joint positions (6,) in radians."""
        return np.array([self._read_sensor(f"pos_joint_{i+1}")[0] for i in range(6)])

    def get_joint_velocities(self) -> np.ndarray:
        """Current joint velocities (6,) in rad/s."""
        return np.array([self._read_sensor(f"vel_joint_{i+1}")[0] for i in range(6)])

    def get_ee_position(self) -> np.ndarray:
        """End-effector position [x, y, z] in metres (MuJoCo units)."""
        return self._read_sensor("ee_pos")

    def get_ee_quaternion(self) -> np.ndarray:
        """End-effector quaternion [w, x, y, z] (MuJoCo convention)."""
        return self._read_sensor("ee_quat")

    def get_state(self) -> Dict:
        """Full state snapshot."""
        return {
            "joint_pos": self.get_joint_positions(),
            "joint_vel": self.get_joint_velocities(),
            "ee_pos": self.get_ee_position(),
            "ee_quat": self.get_ee_quaternion(),
            "time": self.data.time,
        }

    # ─────────────────────────────────────────────────────────────────
    # Trajectory execution
    # ─────────────────────────────────────────────────────────────────

    def execute_trajectory(
        self,
        trajectory: List[List[float]],
        frequency: float = 50.0,
        kp: float = 80.0,
        kd: float = 8.0,
        record: bool = False,
    ) -> Optional[List[Dict]]:
        """
        Execute a joint-space trajectory in the simulator.

        At each trajectory waypoint, applies PD control and steps
        the simulation to match real-time at the given frequency.

        Args:
            trajectory: List of joint angle arrays (6 each).
            frequency: Trajectory publish rate in Hz.
            kp: Proportional gain.
            kd: Derivative gain.
            record: If True, returns list of state dicts.

        Returns:
            Recorded states if record=True, else None.
        """
        dt_traj = 1.0 / frequency
        dt_sim = self.model.opt.timestep
        steps_per_point = max(1, int(dt_traj / dt_sim))

        states = [] if record else None

        if self._render and self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

        for waypoint in trajectory:
            self.set_joint_targets_pd(waypoint, kp=kp, kd=kd)

            for _ in range(steps_per_point):
                # Reapply PD each sim step for stability
                self.set_joint_targets_pd(waypoint, kp=kp, kd=kd)
                self.step()

            if self._render and self._viewer is not None:
                self._viewer.sync()

            if record:
                states.append(self.get_state())

        return states

    # ─────────────────────────────────────────────────────────────────
    # Interactive viewer
    # ─────────────────────────────────────────────────────────────────

    def launch_viewer(self):
        """Launch interactive MuJoCo viewer (blocking)."""
        mujoco.viewer.launch(self.model, self.data)

    def launch_passive_viewer(self):
        """Launch passive (non-blocking) viewer. Call viewer.sync() to update."""
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self._viewer

    def close(self):
        """Clean up viewer resources."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def __del__(self):
        self.close()


if __name__ == "__main__":
    sim = MuJoCoSimulator(render=False)
    sim.reset()
    state = sim.get_state()
    print("Initial state:")
    for k, v in state.items():
        print(f"  {k}: {v}")

    # Move to a pose
    target = [0.3, 1.0, -1.2, -0.5, 0.2, 0.1]
    for _ in range(2000):
        sim.set_joint_targets_pd(target)
        sim.step()
    state = sim.get_state()
    print("\nAfter moving:")
    for k, v in state.items():
        print(f"  {k}: {v}")
    print(f"  Target was: {target}")
    print(f"  Joint error: {np.abs(state['joint_pos'] - target)}")
