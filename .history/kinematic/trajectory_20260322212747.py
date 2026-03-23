"""Trajectory generation for robot manipulation.

Provides cubic polynomial, linear, and Cartesian trajectory generators
for joint-space and task-space planning.
"""
import numpy as np


class TrajectoryGenerator:
    """Generate smooth trajectories in joint space or task space.

    Supports:
    - Cubic polynomial interpolation (smooth start/stop)
    - Linear interpolation
    - Multi-waypoint trajectory with blending

    Example::

        traj = TrajectoryGenerator()
        # Joint-space cubic trajectory
        points = traj.cubic(q_start, q_end, duration=2.0, dt=0.002)
        # Cartesian linear trajectory
        cart_points = traj.cartesian_linear(pos_start, pos_end, steps=100)
    """

    def cubic(self, q_start, q_end, duration, dt=0.002):
        """Generate a cubic polynomial trajectory between two joint configurations.

        Zero velocity at start and end (rest-to-rest motion).

        Args:
            q_start: Start joint positions (n_joints,)
            q_end: End joint positions (n_joints,)
            duration: Total trajectory duration in seconds
            dt: Time step in seconds

        Returns:
            dict with keys:
                - "positions": (T, n_joints) joint positions
                - "velocities": (T, n_joints) joint velocities
                - "times": (T,) time stamps
        """
        q_start = np.array(q_start)
        q_end = np.array(q_end)
        n_joints = len(q_start)
        steps = int(duration / dt)
        times = np.linspace(0, duration, steps)

        # Cubic coefficients: q(t) = a0 + a1*t + a2*t^2 + a3*t^3
        # Boundary: q(0)=q_start, q(T)=q_end, q'(0)=0, q'(T)=0
        a0 = q_start
        a1 = np.zeros(n_joints)
        a2 = 3.0 * (q_end - q_start) / (duration**2)
        a3 = -2.0 * (q_end - q_start) / (duration**3)

        positions = np.zeros((steps, n_joints))
        velocities = np.zeros((steps, n_joints))

        for i, t in enumerate(times):
            positions[i] = a0 + a1 * t + a2 * t**2 + a3 * t**3
            velocities[i] = a1 + 2 * a2 * t + 3 * a3 * t**2

        return {"positions": positions, "velocities": velocities, "times": times}

    def linear(self, q_start, q_end, steps=100):
        """Generate a linear interpolation in joint space.

        Args:
            q_start: Start configuration (n,)
            q_end: End configuration (n,)
            steps: Number of interpolation steps

        Returns:
            (steps, n) array of joint positions
        """
        q_start = np.array(q_start)
        q_end = np.array(q_end)
        return np.array([
            q_start + (q_end - q_start) * t / (steps - 1)
            for t in range(steps)
        ])

    def multi_waypoint(self, waypoints, segment_duration, dt=0.002):
        """Generate a cubic trajectory through multiple waypoints.

        Args:
            waypoints: List of joint configurations, shape (M, n_joints)
            segment_duration: Duration for each segment in seconds (scalar or list)
            dt: Time step

        Returns:
            dict with "positions", "velocities", "times"
        """
        if np.isscalar(segment_duration):
            segment_duration = [segment_duration] * (len(waypoints) - 1)

        all_positions = []
        all_velocities = []
        all_times = []
        t_offset = 0.0

        for i in range(len(waypoints) - 1):
            seg = self.cubic(waypoints[i], waypoints[i + 1], segment_duration[i], dt)
            all_positions.append(seg["positions"])
            all_velocities.append(seg["velocities"])
            all_times.append(seg["times"] + t_offset)
            t_offset += segment_duration[i]

        return {
            "positions": np.vstack(all_positions),
            "velocities": np.vstack(all_velocities),
            "times": np.concatenate(all_times),
        }

    def catmull_rom(self, waypoints, segment_duration, dt=0.002, closed=True):
        """Generate a smooth Catmull-Rom spline trajectory.

        Unlike ``multi_waypoint``, intermediate waypoints carry non-zero
        velocity (continuous first derivative), so the robot never pauses
        between segments — ideal for periodic paths like a figure-8.

        Args:
            waypoints: (M, n_joints) array-like of joint configurations.
            segment_duration: Duration of each segment in seconds (scalar).
            dt: Simulation time step.
            closed: If True the path is treated as a closed loop; the last
                segment connects waypoints[M-1] → waypoints[0].

        Returns:
            dict with "positions" (N, n), "velocities" (N, n), "times" (N,).
        """
        wps = np.array(waypoints, dtype=np.float64)   # (M, n_joints)
        M = len(wps)
        T = float(segment_duration)
        n_segs = M if closed else M - 1

        # Catmull-Rom tangents in position-space (normalised t ∈ [0,1]):
        #   m_i = (q_{i+1} - q_{i-1}) / 2
        # Physical velocity = m_i / T
        m = np.zeros_like(wps)
        for i in range(M):
            ip = (i + 1) % M if closed else min(i + 1, M - 1)
            im = (i - 1) % M if closed else max(i - 1, 0)
            m[i] = (wps[ip] - wps[im]) / 2.0

        all_pos, all_vel, all_t = [], [], []
        t_offset = 0.0

        for seg in range(n_segs):
            i0 = seg % M
            i1 = (seg + 1) % M
            p0, p1 = wps[i0], wps[i1]
            m0, m1 = m[i0], m[i1]

            steps = max(2, int(T / dt))
            t_ = np.linspace(0.0, 1.0, steps)   # normalised ∈ [0,1]
            t2, t3 = t_ ** 2, t_ ** 3

            # Cubic Hermite basis  ─────────
            h00 =  2*t3 - 3*t2 + 1
            h10 =    t3 - 2*t2 + t_
            h01 = -2*t3 + 3*t2
            h11 =    t3 -   t2

            # Derivative bases (w.r.t. normalised t; divide by T for rad/s)
            dh00 =  6*t2 - 6*t_
            dh10 =  3*t2 - 4*t_ + 1
            dh01 = -6*t2 + 6*t_
            dh11 =  3*t2 - 2*t_

            pos = (np.outer(h00, p0) + np.outer(h10, m0) +
                   np.outer(h01, p1) + np.outer(h11, m1))
            vel = (np.outer(dh00, p0) + np.outer(dh10, m0) +
                   np.outer(dh01, p1) + np.outer(dh11, m1)) / T

            all_pos.append(pos)
            all_vel.append(vel)
            all_t.append(t_ * T + t_offset)
            t_offset += T

        return {
            "positions":  np.vstack(all_pos),
            "velocities": np.vstack(all_vel),
            "times":      np.concatenate(all_t),
        }

    def cartesian_linear(self, pos_start, pos_end, steps=100):
        """Linear interpolation in Cartesian space.

        Args:
            pos_start: Start position [x, y, z] (or [x,y,z,roll,pitch,yaw])
            pos_end: End position (same shape as start)
            steps: Number of interpolation steps

        Returns:
            (steps, dim) array of interpolated positions
        """
        pos_start = np.array(pos_start)
        pos_end = np.array(pos_end)
        return np.array([
            pos_start + (pos_end - pos_start) * t / (steps - 1)
            for t in range(steps)
        ])

    def cartesian_arc(self, center, radius, start_angle, end_angle, axis='z', steps=100):
        """Generate a circular arc trajectory in Cartesian space.

        Args:
            center: Center of the arc [x, y, z]
            radius: Radius of the arc
            start_angle: Start angle in radians
            end_angle: End angle in radians
            axis: Rotation axis ('x', 'y', or 'z')
            steps: Number of points

        Returns:
            (steps, 3) array of Cartesian positions
        """
        center = np.array(center)
        angles = np.linspace(start_angle, end_angle, steps)
        points = np.zeros((steps, 3))

        for i, theta in enumerate(angles):
            if axis == 'z':
                points[i] = center + [radius * np.cos(theta), radius * np.sin(theta), 0]
            elif axis == 'y':
                points[i] = center + [radius * np.cos(theta), 0, radius * np.sin(theta)]
            elif axis == 'x':
                points[i] = center + [0, radius * np.cos(theta), radius * np.sin(theta)]

        return points
