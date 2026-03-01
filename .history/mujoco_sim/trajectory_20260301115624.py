"""
SO-ARM100 Cubic Trajectory Generation

Provides joint-space and task-space cubic polynomial interpolation,
plus multi-waypoint chaining.
"""

import numpy as np
from typing import List, Optional, Tuple
from .kinematics import ForwardKinematics, InverseKinematics


class TrajectoryGenerator:
    """Cubic polynomial trajectory generator for the SO-ARM100."""

    def __init__(self):
        self.fk = ForwardKinematics()
        self.ik = InverseKinematics()

    # ─── cubic math ──────────────────────────────────────────────────

    @staticmethod
    def _cubic_coeffs(p0, pf, v0, vf, T):
        """q(t) = a0 + a1*t + a2*t² + a3*t³"""
        if T <= 0:
            return p0, 0.0, 0.0, 0.0
        a0 = p0
        a1 = v0
        a2 = (3*(pf - p0)/T**2) - (2*v0/T) - (vf/T)
        a3 = (2*(p0 - pf)/T**3) + ((v0 + vf)/T**2)
        return a0, a1, a2, a3

    # ─── joint-space trajectory ──────────────────────────────────────

    def joint_trajectory(
        self,
        start: List[float],
        end: List[float],
        duration: float,
        frequency: float,
        start_vel: Optional[List[float]] = None,
        end_vel: Optional[List[float]] = None,
    ) -> List[List[float]]:
        """
        Cubic trajectory in joint space.

        Args:
            start / end:   Joint angles (5 each), radians.
            duration:      Seconds.
            frequency:     Hz.
            start_vel / end_vel:  Joint velocities. Defaults to zeros.

        Returns:
            List of joint-angle arrays, one per time step.
        """
        nj = len(start)
        sv = start_vel or [0.0]*nj
        ev = end_vel or [0.0]*nj

        coeffs = [self._cubic_coeffs(start[j], end[j], sv[j], ev[j], duration)
                  for j in range(nj)]

        n_pts = int(duration * frequency)
        dt = 1.0 / frequency
        traj = []
        for i in range(n_pts):
            t = min(i * dt, duration)
            traj.append([sum(c * t**p for p, c in enumerate(coeffs[j]))
                         for j in range(nj)])
        return traj

    # ─── task-space trajectory ───────────────────────────────────────

    def task_trajectory(
        self,
        start_pos: List[float],
        end_pos: List[float],
        start_euler: List[float],
        end_euler: List[float],
        duration: float,
        frequency: float,
        start_vel=None, end_vel=None,
        w_start=None, w_end=None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Cubic trajectory in Cartesian space (position + Euler angles).

        Returns list of (position, quaternion_wxyz) per step.
        """
        sv = start_vel or [0]*3
        ev = end_vel or [0]*3
        ws = w_start or [0]*3
        we = w_end or [0]*3

        # Unwrap end euler for shortest path
        end_e = np.array(end_euler, dtype=float)
        for i in range(3):
            d = end_e[i] - start_euler[i]
            if d > np.pi:  end_e[i] -= 2*np.pi
            elif d < -np.pi: end_e[i] += 2*np.pi

        pc = [self._cubic_coeffs(start_pos[d], end_pos[d], sv[d], ev[d], duration) for d in range(3)]
        ec = [self._cubic_coeffs(start_euler[d], end_e[d], ws[d], we[d], duration) for d in range(3)]

        n_pts = int(duration * frequency)
        dt = 1.0 / frequency
        traj = []

        for i in range(n_pts):
            t = min(i * dt, duration)
            pos = np.array([sum(c * t**p for p, c in enumerate(pc[d])) for d in range(3)])
            euler = np.array([sum(c * t**p for p, c in enumerate(ec[d])) for d in range(3)])
            R = self._euler_to_R(euler)
            q = ForwardKinematics.rotmat_to_quat_wxyz(R)
            traj.append((pos, q))
        return traj

    def task_to_joint(
        self,
        task_traj: List[Tuple[np.ndarray, np.ndarray]],
        initial_angles: Optional[List[float]] = None,
    ) -> List[List[float]]:
        """Convert task-space trajectory to joint-space via numerical IK."""
        prev = initial_angles or [0.0] * self.fk.NUM_JOINTS
        out = []
        for pos, quat in task_traj:
            angles = self.ik.numerical(pos, quat, initial_angles=prev)
            out.append(angles)
            prev = angles
        return out

    # ─── multi-waypoint ──────────────────────────────────────────────

    def multi_waypoint(
        self,
        waypoints: List[List[float]],
        durations: List[float],
        frequency: float,
    ) -> List[List[float]]:
        """Chain cubic segments through multiple waypoints (zero vel at each)."""
        if len(durations) != len(waypoints) - 1:
            raise ValueError("Need len(waypoints)-1 durations.")
        full = []
        for i in range(len(waypoints) - 1):
            seg = self.joint_trajectory(waypoints[i], waypoints[i+1], durations[i], frequency)
            if i > 0 and seg:
                seg = seg[1:]
            full.extend(seg)
        return full

    # ─── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _euler_to_R(euler_xyz):
        """Intrinsic XYZ Euler → rotation matrix."""
        r, p, y = euler_xyz
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
        Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
        return Rz @ Ry @ Rx
