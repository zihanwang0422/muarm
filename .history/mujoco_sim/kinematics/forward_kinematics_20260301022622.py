"""
SO-ARM100 Forward Kinematics Module

Implements FK using DH-style transformation matrices.
Rotation axes: [Z, X, X, X, Y, X]
Units: millimeters (consistent with original SOARM101)
"""

import numpy as np
from typing import List, Optional, Tuple


class ForwardKinematics:
    """
    Forward Kinematics solver for the SO-ARM100 6-DOF robot arm.
    
    The arm has 6 revolute joints with rotation axes [Z, X, X, X, Y, X]
    and specific link offsets defined by the hardware geometry.
    
    Units: mm for position, radians for angles.
    """

    def __init__(self):
        self.num_joints = 6
        self.rotation_axes = [2, 0, 0, 0, 1, 0]  # z=2, x=0, y=1
        
        # Link offsets in mm (from joint i frame to joint i+1 frame)
        self.link_offsets = [
            [0.0, 0.0, 0.0],       # Base → J1
            [0.0, 31.15, 119.7],    # J1 → J2
            [0.0, 112.35, -28.0],   # J2 → J3
            [0.0, 134.9, 4.85],     # J3 → J4
            [0.0, 54.8, 0.0],       # J4 → J5
            [0.0, 31.5, 20.0],      # J5 → J6
        ]
        self.grab_offset = [0.0, 76.0, -11.5]  # J6 → end-effector

        # Joint limits in radians
        self.joint_limits = [
            (-np.pi / 2, np.pi / 2),       # Joint 1: ±90°
            (0.0, np.pi),                   # Joint 2: 0° to 180°
            (-170 * np.pi / 180, 25 * np.pi / 180),  # Joint 3
            (-np.pi, np.pi),                # Joint 4: ±180°
            (-np.pi / 2, np.pi / 2),        # Joint 5: ±90°
            (-np.pi, np.pi),                # Joint 6: ±180°
        ]

    @staticmethod
    def _rotation_x(angle: float) -> np.ndarray:
        """Rotation matrix about X axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [1, 0,  0, 0],
            [0, c, -s, 0],
            [0, s,  c, 0],
            [0, 0,  0, 1]
        ])

    @staticmethod
    def _rotation_y(angle: float) -> np.ndarray:
        """Rotation matrix about Y axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [ c, 0, s, 0],
            [ 0, 1, 0, 0],
            [-s, 0, c, 0],
            [ 0, 0, 0, 1]
        ])

    @staticmethod
    def _rotation_z(angle: float) -> np.ndarray:
        """Rotation matrix about Z axis."""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])

    def transformation_matrix(self, angle: float, axis: int, offset: List[float]) -> np.ndarray:
        """
        Compute 4x4 homogeneous transformation matrix for a single joint.

        Args:
            angle: Joint angle in radians.
            axis: Rotation axis index (0=X, 1=Y, 2=Z).
            offset: [x, y, z] translation offset in mm.

        Returns:
            4x4 homogeneous transformation matrix.
        """
        if axis == 0:
            T = self._rotation_x(angle)
        elif axis == 1:
            T = self._rotation_y(angle)
        elif axis == 2:
            T = self._rotation_z(angle)
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 0 (X), 1 (Y), or 2 (Z).")

        T[0, 3] = offset[0]
        T[1, 3] = offset[1]
        T[2, 3] = offset[2]
        return T

    def compute(self, angles: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics: joint angles → end-effector pose.

        Args:
            angles: List of 6 joint angles in radians. Defaults to zeros.

        Returns:
            Tuple of (quaternion [x,y,z,w], position [x,y,z] in mm).
        """
        if angles is None:
            angles = [0.0] * self.num_joints

        if len(angles) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} angles, got {len(angles)}.")

        T = np.eye(4)
        for i in range(self.num_joints):
            Ti = self.transformation_matrix(angles[i], self.rotation_axes[i], self.link_offsets[i])
            T = T @ Ti

        # Apply end-effector (grab) offset
        T_ee = self.transformation_matrix(0.0, 0, self.grab_offset)
        T = T @ T_ee

        position = T[:3, 3]
        quaternion = self.rotation_matrix_to_quaternion(T[:3, :3])
        return quaternion, position

    def compute_all_transforms(self, angles: Optional[List[float]] = None) -> List[np.ndarray]:
        """
        Compute cumulative transforms for all joints (useful for visualization).

        Args:
            angles: List of 6 joint angles in radians.

        Returns:
            List of 4x4 transforms from base to each joint frame (7 total, including EE).
        """
        if angles is None:
            angles = [0.0] * self.num_joints

        transforms = [np.eye(4)]
        T = np.eye(4)
        for i in range(self.num_joints):
            Ti = self.transformation_matrix(angles[i], self.rotation_axes[i], self.link_offsets[i])
            T = T @ Ti
            transforms.append(T.copy())

        # End-effector
        T_ee = self.transformation_matrix(0.0, 0, self.grab_offset)
        T = T @ T_ee
        transforms.append(T.copy())

        return transforms

    @staticmethod
    def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
        """
        Convert 3x3 rotation matrix to quaternion [x, y, z, w].

        Args:
            R: 3x3 rotation matrix.

        Returns:
            Quaternion as [x, y, z, w].
        """
        R = np.asarray(R, dtype=float)
        trace = np.trace(R)

        if trace > 0:
            s = 2.0 * np.sqrt(trace + 1.0)
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([x, y, z, w])

    @staticmethod
    def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
        """
        Convert quaternion [x, y, z, w] to 3x3 rotation matrix.

        Args:
            quat: Quaternion as [x, y, z, w].

        Returns:
            3x3 rotation matrix.
        """
        x, y, z, w = quat
        norm = np.sqrt(x * x + y * y + z * z + w * w)
        if norm > 0:
            x, y, z, w = x / norm, y / norm, z / norm, w / norm

        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
        ])


if __name__ == "__main__":
    fk = ForwardKinematics()

    # Zero pose
    q0, p0 = fk.compute([0, 0, 0, 0, 0, 0])
    print("Zero pose:")
    print(f"  Position (mm): {p0}")
    print(f"  Quaternion [x,y,z,w]: {q0}")

    # Test pose
    angles = [np.pi / 3, np.pi / 2, 0, -np.pi / 4, 0, 0]
    q, p = fk.compute(angles)
    print(f"\nTest pose {np.rad2deg(angles)}°:")
    print(f"  Position (mm): {p}")
    print(f"  Quaternion [x,y,z,w]: {q}")
