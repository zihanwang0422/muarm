"""Cartesian admittance controller for Panda robot.

Implements the admittance control law:
    M_d * ddx + B_d * dx + K_d * (x - x_des) = F_ext

Converts external forces into desired position modifications,
then uses IK to track the modified trajectory.
"""
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.transform import transform2mat


class AdmittanceController:
    """Cartesian admittance controller.

    Models a virtual mass-spring-damper system that determines how the
    end-effector responds to external forces.

    Example::

        ctrl = AdmittanceController(M=10.0, B=50.0, K=100.0)
        # In the control loop:
        desired_pos = ctrl.update(F_ext, ee_pos, ee_vel, dt)
    """

    def __init__(self, M=10.0, B=50.0, K=100.0, n_dims=6):
        """
        Args:
            M: Virtual inertia (scalar or array of size n_dims)
            B: Virtual damping (scalar or array of size n_dims)
            K: Virtual stiffness (scalar or array of size n_dims)
            n_dims: Number of Cartesian dimensions (6 = pos + orient)
        """
        self.n_dims = n_dims
        self.M_d = np.diag(np.full(n_dims, M) if np.isscalar(M) else np.array(M))
        self.B_d = np.diag(np.full(n_dims, B) if np.isscalar(B) else np.array(B))
        self.K_d = np.diag(np.full(n_dims, K) if np.isscalar(K) else np.array(K))

        self.M_inv = np.linalg.inv(self.M_d)
        self.delta_vel = np.zeros(n_dims)
        self.delta_pos = np.zeros(n_dims)
        self.reference_pose = None

    def set_reference(self, pose):
        """Set the equilibrium/reference pose [x, y, z, roll, pitch, yaw].

        Args:
            pose: Reference pose (n_dims,)
        """
        self.reference_pose = np.array(pose[:self.n_dims])
        self.delta_vel = np.zeros(self.n_dims)
        self.delta_pos = np.zeros(self.n_dims)

    def update(self, F_ext, current_pose, current_vel, dt):
        """Compute the desired pose based on admittance dynamics.

        M_d * dd_delta + B_d * d_delta + K_d * delta = F_ext

        where delta = desired_pose - reference_pose

        Args:
            F_ext: External force/torque vector (n_dims,)
            current_pose: Current EE pose [x,y,z,roll,pitch,yaw] (n_dims,)
            current_vel: Current EE velocity (n_dims,)
            dt: Time step

        Returns:
            Desired pose (n_dims,) that the robot should track
        """
        if self.reference_pose is None:
            self.reference_pose = np.array(current_pose[:self.n_dims])

        F_ext = np.array(F_ext[:self.n_dims])
        pose_error = np.array(current_pose[:self.n_dims]) - self.reference_pose

        # Admittance dynamics: M*ddx = F_ext - B*dx - K*x_err
        ddelta = self.M_inv @ (F_ext - self.B_d @ current_vel[:self.n_dims] - self.K_d @ pose_error)

        self.delta_vel += ddelta * dt
        self.delta_pos += self.delta_vel * dt

        desired_pose = self.reference_pose + self.delta_pos
        return desired_pose

    def get_transform(self, desired_pose):
        """Convert a 6D pose to a 4x4 homogeneous transformation matrix.

        Args:
            desired_pose: [x, y, z, roll, pitch, yaw]

        Returns:
            4x4 transformation matrix
        """
        return transform2mat(*desired_pose[:6])

    def reset(self):
        """Reset the controller state."""
        self.delta_vel = np.zeros(self.n_dims)
        self.delta_pos = np.zeros(self.n_dims)
        self.reference_pose = None
