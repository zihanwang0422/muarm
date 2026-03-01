"""
SO-ARM100 Forward Kinematics Module

Implements FK matching the official TRS MuJoCo MJCF model exactly.

The official model body chain:
  Base → Rotation_Pitch → Upper_Arm → Lower_Arm → Wrist_Pitch_Roll → Fixed_Jaw (EE)

Each body-to-body transition consists of:
  1. A fixed translation (body pos) in the parent frame
  2. A fixed rotation (body quat) reorienting the child frame
  3. A joint rotation about the joint axis in the child's local frame

Joint axes (in local body frame):
  Rotation:     Y-axis [0,1,0]   (base rotation)
  Pitch:        X-axis [1,0,0]   (shoulder)
  Elbow:        X-axis [1,0,0]   (elbow)
  Wrist_Pitch:  X-axis [1,0,0]   (wrist pitch)
  Wrist_Roll:   Y-axis [0,1,0]   (wrist roll)

Units: metres (matching MuJoCo convention).
"""

import numpy as np
from typing import List, Optional, Tuple


class ForwardKinematics:
    """
    Forward Kinematics for the SO-ARM100, matching the official TRS MJCF model.
    """

    # Number of arm joints (excluding Jaw gripper)
    NUM_JOINTS = 5

    # Body chain: parent → child transforms (pos, quat in wxyz)
    # Extracted directly from official so_arm100.xml
    BODY_CHAIN = [
        {   # Base → Rotation_Pitch
            "name": "Rotation_Pitch",
            "pos": np.array([0.0, -0.0452, 0.0165]),
            "quat_wxyz": np.array([0.70710528, 0.70710828, 0.0, 0.0]),
            "joint_axis": np.array([0.0, 1.0, 0.0]),   # Rotation (Y)
        },
        {   # Rotation_Pitch → Upper_Arm
            "name": "Upper_Arm",
            "pos": np.array([0.0, 0.1025, 0.0306]),
            "quat_wxyz": np.array([0.70710902, 0.70710454, 0.0, 0.0]),
            "joint_axis": np.array([1.0, 0.0, 0.0]),   # Pitch (X)
        },
        {   # Upper_Arm → Lower_Arm
            "name": "Lower_Arm",
            "pos": np.array([0.0, 0.11257, 0.028]),
            "quat_wxyz": np.array([0.70710902, -0.70710454, 0.0, 0.0]),
            "joint_axis": np.array([1.0, 0.0, 0.0]),   # Elbow (X)
        },
        {   # Lower_Arm → Wrist_Pitch_Roll
            "name": "Wrist_Pitch_Roll",
            "pos": np.array([0.0, 0.0052, 0.1349]),
            "quat_wxyz": np.array([0.70710902, -0.70710454, 0.0, 0.0]),
            "joint_axis": np.array([1.0, 0.0, 0.0]),   # Wrist_Pitch (X)
        },
        {   # Wrist_Pitch_Roll → Fixed_Jaw (EE)
            "name": "Fixed_Jaw",
            "pos": np.array([0.0, -0.0601, 0.0]),
            "quat_wxyz": np.array([0.70710902, 0.0, 0.70710454, 0.0]),
            "joint_axis": np.array([0.0, 1.0, 0.0]),   # Wrist_Roll (Y)
        },
    ]

    # Joint limits in radians (from official MJCF)
    JOINT_LIMITS = [
        (-1.92, 1.92),      # Rotation:    ±110°
        (-3.32, 0.174),     # Pitch:       -190°~10°
        (-0.174, 3.14),     # Elbow:       -10°~180°
        (-1.66, 1.66),      # Wrist_Pitch: ±95°
        (-2.79, 2.79),      # Wrist_Roll:  ±160°
    ]

    JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]

    @staticmethod
    def quat_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
        """Convert quaternion [w,x,y,z] → 3x3 rotation matrix."""
        w, x, y, z = q_wxyz
        n = np.sqrt(w*w + x*x + y*y + z*z)
        if n > 0:
            w, x, y, z = w/n, x/n, y/n, z/n
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
        ])

    @staticmethod
    def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix → quaternion [w,x,y,z]."""
        trace = np.trace(R)
        if trace > 0:
            s = 2.0 * np.sqrt(trace + 1.0)
            w, x, y, z = 0.25 * s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            w, x, y, z = (R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            w, x, y, z = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            w, x, y, z = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s
        return np.array([w, x, y, z])

    @staticmethod
    def axis_angle_to_rotmat(axis: np.ndarray, angle: float) -> np.ndarray:
        """Rotation matrix from axis-angle (Rodrigues formula)."""
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    def compute(self, angles: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FK: joint angles → end-effector (Fixed_Jaw) pose.

        Args:
            angles: 5 arm joint angles [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll]
                    in radians. Defaults to zeros.

        Returns:
            (position [x,y,z] in metres, quaternion [w,x,y,z]).
        """
        if angles is None:
            angles = [0.0] * self.NUM_JOINTS
        angles = list(angles)
        while len(angles) < self.NUM_JOINTS:
            angles.append(0.0)

        R = np.eye(3)
        p = np.zeros(3)

        for i, link in enumerate(self.BODY_CHAIN):
            p = p + R @ link["pos"]
            R = R @ self.quat_to_rotmat(link["quat_wxyz"])
            R = R @ self.axis_angle_to_rotmat(link["joint_axis"], angles[i])

        return p, self.rotmat_to_quat_wxyz(R)

    def compute_all_frames(self, angles: Optional[List[float]] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Compute world-frame pose of every body from Base to Fixed_Jaw.

        Returns:
            List of (position, rotation_matrix) per body frame.
        """
        if angles is None:
            angles = [0.0] * self.NUM_JOINTS
        angles = list(angles)
        while len(angles) < self.NUM_JOINTS:
            angles.append(0.0)

        frames = []
        R = np.eye(3)
        p = np.zeros(3)
        frames.append((p.copy(), R.copy()))

        for i, link in enumerate(self.BODY_CHAIN):
            p = p + R @ link["pos"]
            R = R @ self.quat_to_rotmat(link["quat_wxyz"])
            R = R @ self.axis_angle_to_rotmat(link["joint_axis"], angles[i])
            frames.append((p.copy(), R.copy()))

        return frames


if __name__ == "__main__":
    fk = ForwardKinematics()
    p, q = fk.compute([0, 0, 0, 0, 0])
    print(f"Zero:  pos={p}  quat={q}")
    p, q = fk.compute([0, -1.57, 1.57, 1.57, -1.57])
    print(f"Home:  pos={p}  quat={q}")
