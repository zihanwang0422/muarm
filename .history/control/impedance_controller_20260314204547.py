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
        # 1. 位置误差：目标 - 当前 (确保是负反馈)
        e_pos = np.array(pos_desired) - np.array(pos_current)

        # 2. 姿态误差
        if R_desired is not None and R_current is not None:
            # 这个公式计算的是从 R_current 到 R_desired 的旋转差
            e_rot = 0.5 * (
                np.cross(R_current[:, 0], R_desired[:, 0]) +
                np.cross(R_current[:, 1], R_desired[:, 1]) +
                np.cross(R_current[:, 2], R_desired[:, 2])
            )
        else:
            e_rot = np.zeros(3)

        e = np.concatenate([e_pos, e_rot])
        
        # 3. 计算笛卡尔力: F = Kp * e + Kd * (0 - v)
        # 这里的 vel_current 应该是 6x1 的末端速度
        F = self.Kp @ e - self.Kd @ np.array(vel_current)

        # 4. 映射到力矩
        # 注意：这里的 tau 只是阻抗部分，外面调用时必须加 g_tau (重力补偿)
        tau = jacobian.T @ F
        return tau[:self.n_joints]