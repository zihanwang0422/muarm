"""Math and transformation utilities for robotics manipulation."""
import numpy as np


def quat2rotmat(quat):
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x**2 - 2*z**2,  2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,         1 - 2*x**2 - 2*y**2],
    ])


def rotmat2quat(R):
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
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
    return np.array([w, x, y, z])


def euler2rotmat(roll, pitch, yaw):
    """Convert euler angles (Z-Y-X order) to 3x3 rotation matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ])


def euler2quat(roll, pitch, yaw):
    """Convert euler angles (roll, pitch, yaw) to quaternion [w, x, y, z]."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = cy * sp * cr + sy * cp * sr
    z = -cy * sp * sr + sy * cp * cr
    return np.array([w, x, y, z])


def quat2euler(quat):
    """Convert quaternion [w, x, y, z] to euler angles (roll, pitch, yaw)."""
    w, x, y, z = map(float, quat)
    sin_pitch = np.clip(2 * (w * y - x * z), -1.0, 1.0)
    if np.abs(sin_pitch) < 0.9999999:
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
        pitch = np.arcsin(sin_pitch)
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    else:
        pitch = np.pi / 2 if sin_pitch > 0 else -np.pi / 2
        yaw = np.arctan2(2 * (x * y + w * z), 1 - 2 * (x**2 + z**2))
        roll = 0.0
    return roll, pitch, yaw


def transform2mat(x, y, z, roll, pitch, yaw):
    """Create 4x4 homogeneous transformation matrix from position and euler angles."""
    R = euler2rotmat(roll, pitch, yaw)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def mat2transform(mat):
    """Extract (x, y, z, roll, pitch, yaw) from 4x4 homogeneous matrix."""
    x, y, z = mat[0:3, 3]
    roll = np.arctan2(mat[2, 1], mat[2, 2])
    pitch = np.arctan2(-mat[2, 0], np.sqrt(mat[2, 1]**2 + mat[2, 2]**2))
    yaw = np.arctan2(mat[1, 0], mat[0, 0])
    return x, y, z, roll, pitch, yaw


def damped_pinv(J, lambda_d=0.1):
    """Compute the damped pseudo-inverse of a Jacobian matrix."""
    JT = J.T
    damping = lambda_d**2 * np.eye(J.shape[0])
    return JT @ np.linalg.inv(J @ JT + damping)
