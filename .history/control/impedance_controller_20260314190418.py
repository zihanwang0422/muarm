"""Joint-space and Cartesian impedance controller for Panda robot.

Implements the impedance control law:
    tau = Kp * (q_des - q) - Kd * q_dot

Can be used in joint space (directly on joint errors) or Cartesian space
(with Jacobian transpose mapping).
"""
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.transform import damped_pinv, quat2rotmat, euler2rotmat


class ImpedanceController:
    """Joint-space or Cartesian impedance controller.

    In joint mode: tau = Kp * (q_des - q) - Kd * q_dot
    In Cartesian mode: tau = J^T * (Kp_cart * x_err - Kd_cart * x_dot)

    Example (joint space)::

        ctrl = ImpedanceController(n_joints=7, kp=100.0, kd=10.0)
        tau = ctrl.compute_joint(q_desired, q_current, q_dot_current)

    Example (Cartesian space)::

        ctrl = ImpedanceController(n_joints=7, kp=100.0, kd=10.0, mode='cartesian')
        tau = ctrl.compute_cartesian(
            x_desired, x_current, x_dot_current, jacobian,
            R_desired=R_des, R_current=R_cur
        )
    """

    def __init__(self, n_joints=7, kp=100.0, kd=10.0, mode='joint'):
        """
        Args:
            n_joints: Number of robot joints
            kp: Stiffness (scalar or array)
            kd: Damping (scalar or array)
            mode: 'joint' for joint-space, 'cartesian' for task-space
        """
        self.n_joints = n_joints
        self.mode = mode

        if mode == 'joint':
            dim = n_joints
        else:
            dim = 6  # 3 translational + 3 rotational
        self.Kp = np.diag(np.full(dim, kp) if np.isscalar(kp) else np.array(kp))
        self.Kd = np.diag(np.full(dim, kd) if np.isscalar(kd) else np.array(kd))

    def compute_joint(self, q_desired, q_current, q_dot_current):
        """Compute joint-space impedance torque.

        Args:
            q_desired: Desired joint positions (n_joints,)
            q_current: Current joint positions (n_joints,)
            q_dot_current: Current joint velocities (n_joints,)

        Returns:
            Torque command (n_joints,)
        """
        error = np.array(q_desired) - np.array(q_current)
        qdot = np.array(q_dot_current)
        return self.Kp @ error - self.Kd @ qdot

    def compute_cartesian(self, pos_desired, pos_current, vel_current, jacobian,
                          R_desired=None, R_current=None):
        """Compute Cartesian impedance torque mapped to joint torques via J^T.

        Args:
            pos_desired: Desired position [x, y, z]
            pos_current: Current position [x, y, z]
            vel_current: Current Cartesian velocity (6,) [v_lin; v_ang]
            jacobian: 6xN Jacobian matrix
            R_desired: Desired rotation matrix (3x3). If None, no orientation control.
            R_current: Current rotation matrix (3x3). If None, no orientation control.

        Returns:
            Joint torque command (n_joints,)
        """
        # Position error
        e_pos = np.array(pos_current) - np.array(pos_desired)

        # Orientation error
        if R_desired is not None and R_current is not None:
            e_rot = 0.5 * (
                np.cross(R_desired[:, 0], R_current[:, 0]) +
                np.cross(R_desired[:, 1], R_current[:, 1]) +
                np.cross(R_desired[:, 2], R_current[:, 2])
            )
        else:
            e_rot = np.zeros(3)

        e = np.concatenate([e_pos, e_rot])
        F = self.Kp @ e - self.Kd @ vel_current

        # Map to joint torques
        tau = jacobian.T @ F
        return tau[:self.n_joints]
