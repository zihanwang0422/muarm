"""
SO-ARM100 Trajectory Generation Module

Provides cubic polynomial trajectory generation in both joint space
and task space (Cartesian), with optional IK conversion.
"""

import numpy as np
from typing import List, Optional, Tuple
from .kinematics import ForwardKinematics, InverseKinematics


class TrajectoryGenerator:
    """
    Cubic polynomial trajectory generator for the SO-ARM100.

    Supports:
      - Joint-space cubic trajectories
      - Task-space (Cartesian) cubic trajectories with IK conversion
      - Multi-waypoint trajectory chaining
    """

    def __init__(self):
        self.fk = ForwardKinematics()
        self.ik = InverseKinematics()

    # ─────────────────────────────────────────────────────────────────
    # Joint-space trajectory
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def cubic_coefficients(
        p0: float, pf: float, v0: float, vf: float, T: float
    ) -> Tuple[float, float, float, float]:
        """
        Compute cubic polynomial coefficients: q(t) = a0 + a1*t + a2*t² + a3*t³.

        Args:
            p0: Start position/angle.
            pf: End position/angle.
            v0: Start velocity.
            vf: End velocity.
            T: Duration.

        Returns:
            Coefficients (a0, a1, a2, a3).
        """
        if T <= 0:
            return p0, 0.0, 0.0, 0.0
        a0 = p0
        a1 = v0
        a2 = (3 * (pf - p0) / T ** 2) - (2 * v0 / T) - (vf / T)
        a3 = (2 * (p0 - pf) / T ** 3) + ((v0 + vf) / T ** 2)
        return a0, a1, a2, a3

    def joint_space_trajectory(
        self,
        start_angles: List[float],
        end_angles: List[float],
        duration: float,
        frequency: float,
        start_vel: Optional[List[float]] = None,
        end_vel: Optional[List[float]] = None,
    ) -> List[List[float]]:
        """
        Generate cubic trajectory in joint space.

        Args:
            start_angles: Start joint angles (6) in radians.
            end_angles: End joint angles (6) in radians.
            duration: Duration in seconds.
            frequency: Sampling frequency in Hz.
            start_vel: Start joint velocities (6). Defaults to zeros.
            end_vel: End joint velocities (6). Defaults to zeros.

        Returns:
            List of joint angle arrays, one per time step.
        """
        n_joints = 6
        if start_vel is None:
            start_vel = [0.0] * n_joints
        if end_vel is None:
            end_vel = [0.0] * n_joints

        num_points = int(duration * frequency)
        dt = 1.0 / frequency

        # Compute coefficients for each joint
        coeffs = []
        for j in range(n_joints):
            coeffs.append(
                self.cubic_coefficients(start_angles[j], end_angles[j],
                                        start_vel[j], end_vel[j], duration)
            )

        trajectory = []
        for i in range(num_points):
            t = min(i * dt, duration)
            angles = []
            for j in range(n_joints):
                a0, a1, a2, a3 = coeffs[j]
                q = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3
                angles.append(q)
            trajectory.append(angles)

        return trajectory

    # ─────────────────────────────────────────────────────────────────
    # Task-space trajectory
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def euler_to_rotation_matrix(euler_xyz: List[float]) -> np.ndarray:
        """
        Convert intrinsic XYZ Euler angles to 3x3 rotation matrix.
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
        """
        roll, pitch, yaw = euler_xyz
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    @staticmethod
    def _unwrap_euler(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        """Shortest-path unwrap for Euler angles."""
        unwrapped = np.array(curr, dtype=float)
        for i in range(3):
            diff = curr[i] - prev[i]
            if diff > np.pi:
                unwrapped[i] -= 2 * np.pi
            elif diff < -np.pi:
                unwrapped[i] += 2 * np.pi
        return unwrapped

    def task_space_trajectory(
        self,
        start_pos: List[float],
        end_pos: List[float],
        start_euler: List[float],
        end_euler: List[float],
        duration: float,
        frequency: float,
        start_vel: Optional[List[float]] = None,
        end_vel: Optional[List[float]] = None,
        w_start: Optional[List[float]] = None,
        w_end: Optional[List[float]] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate cubic trajectory in task space (position + orientation).

        Args:
            start_pos: Start [x, y, z] in mm.
            end_pos: End [x, y, z] in mm.
            start_euler: Start [roll, pitch, yaw] in radians.
            end_euler: End [roll, pitch, yaw] in radians.
            duration: Duration in seconds.
            frequency: Sampling frequency in Hz.
            start_vel: Start linear velocity [vx, vy, vz]. Defaults to zeros.
            end_vel: End linear velocity. Defaults to zeros.
            w_start: Start angular velocity [wx, wy, wz]. Defaults to zeros.
            w_end: End angular velocity. Defaults to zeros.

        Returns:
            List of (position, quaternion) tuples per time step.
        """
        if start_vel is None:
            start_vel = [0.0, 0.0, 0.0]
        if end_vel is None:
            end_vel = [0.0, 0.0, 0.0]
        if w_start is None:
            w_start = [0.0, 0.0, 0.0]
        if w_end is None:
            w_end = [0.0, 0.0, 0.0]

        num_points = int(duration * frequency)
        dt = 1.0 / frequency

        end_euler_unwrapped = self._unwrap_euler(np.array(start_euler), np.array(end_euler))

        # Position coefficients (3 dims)
        pos_coeffs = [
            self.cubic_coefficients(start_pos[d], end_pos[d], start_vel[d], end_vel[d], duration)
            for d in range(3)
        ]
        # Euler coefficients (3 dims)
        eul_coeffs = [
            self.cubic_coefficients(start_euler[d], end_euler_unwrapped[d], w_start[d], w_end[d], duration)
            for d in range(3)
        ]

        trajectory = []
        prev_euler = np.array(start_euler, dtype=float)

        for i in range(num_points):
            t = min(i * dt, duration)

            pos = np.array([
                sum(c * t ** p for p, c in enumerate(pos_coeffs[d])) for d in range(3)
            ])
            euler = np.array([
                sum(c * t ** p for p, c in enumerate(eul_coeffs[d])) for d in range(3)
            ])

            if i > 0:
                euler = self._unwrap_euler(prev_euler, euler)
            prev_euler = euler.copy()

            R = self.euler_to_rotation_matrix(euler)
            quat = ForwardKinematics.rotation_matrix_to_quaternion(R)
            trajectory.append((pos, quat))

        return trajectory

    def task_to_joint_trajectory(
        self,
        task_trajectory: List[Tuple[np.ndarray, np.ndarray]],
        method: str = "numerical",
        initial_angles: Optional[List[float]] = None,
    ) -> List[List[float]]:
        """
        Convert a task-space trajectory to joint-space using IK.

        Args:
            task_trajectory: List of (position, quaternion) tuples.
            method: IK method — "analytical", "numerical", or "nn".
            initial_angles: Initial joint angles for the first IK call.

        Returns:
            List of joint angle arrays.
        """
        joint_traj = []
        prev_angles = initial_angles or [0.0] * 6

        for pos, quat in task_trajectory:
            if method == "analytical":
                angles = self.ik.analytical(quat, pos)
            elif method == "numerical":
                angles = self.ik.numerical(quat, pos, initial_angles=prev_angles)
            elif method == "nn":
                angles = self.ik.neural_network(quat, pos, reference_angles=prev_angles)
            else:
                raise ValueError(f"Unknown IK method: {method}")
            joint_traj.append(angles)
            prev_angles = angles

        return joint_traj

    # ─────────────────────────────────────────────────────────────────
    # Multi-waypoint
    # ─────────────────────────────────────────────────────────────────

    def multi_waypoint_joint_trajectory(
        self,
        waypoints: List[List[float]],
        durations: List[float],
        frequency: float,
    ) -> List[List[float]]:
        """
        Generate smooth trajectory through multiple waypoints in joint space.
        Velocities at interior waypoints are zero (stop at each waypoint).

        Args:
            waypoints: List of joint angle arrays (N waypoints).
            durations: Segment durations (N-1 values).
            frequency: Sampling frequency in Hz.

        Returns:
            Full joint-space trajectory.
        """
        if len(durations) != len(waypoints) - 1:
            raise ValueError("Need exactly len(waypoints)-1 durations.")

        full_traj = []
        for i in range(len(waypoints) - 1):
            seg = self.joint_space_trajectory(
                waypoints[i], waypoints[i + 1], durations[i], frequency
            )
            # Avoid duplicating the boundary point
            if i > 0 and len(seg) > 0:
                seg = seg[1:]
            full_traj.extend(seg)

        return full_traj


if __name__ == "__main__":
    tg = TrajectoryGenerator()

    # Joint-space trajectory demo
    start = [0, 0.8, -1.5, -1.0, 0, 0]
    end = [0.5, 1.2, -1.0, -0.5, 0.3, 0.1]
    traj = tg.joint_space_trajectory(start, end, duration=3.0, frequency=50)
    print(f"Joint-space trajectory: {len(traj)} points")
    print(f"  Start: {np.rad2deg(traj[0])}")
    print(f"  End:   {np.rad2deg(traj[-1])}")
