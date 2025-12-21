# DO NOT TOUCH!

import numpy as np


class SOARM101:
    """
    Forward Kinematics for SO Arm 101
    units: radians/cm
    use SOARM101() to initialize the class
    use arm.forward_kinematics(angles) to get the end effector transformation matrix
    """

    def __init__(self):
        self.angles = [0, 0, 0, 0, 0, 0]
        self.rotation_axis = [2, 0, 0, 0, 1, 0]  # z, x, x, x, y, x
        self.link_offsets = [
            [0, 0, 0],
            [0, 3.115, 11.97],
            [0, 11.235, -2.8],
            [0, 13.49, 0.485],
            [0, 5.48, 0],
            [0, 3.15, 2],
        ]
        self.grab_position_offset = [0, 7.6, -1.15]

    def transformation_matrix(self, angle, axis, offset):
        """
        Generate Transformation matrix for each joint
        Args:
            angle: angle of the joint, in radians
            axis: axis of the joint, x:0, y:1, z:2
            offset: offset of the joint, in cm
        Returns:
            Transformation matrix
        """
        if axis == 0:  # x-axis
            return np.array(
                [
                    [1, 0, 0, offset[0]],
                    [0, np.cos(angle), -np.sin(angle), offset[1]],
                    [0, np.sin(angle), np.cos(angle), offset[2]],
                    [0, 0, 0, 1],
                ]
            )
        if axis == 1:  # y-axis
            return np.array(
                [
                    [np.cos(angle), 0, np.sin(angle), offset[0]],
                    [0, 1, 0, offset[1]],
                    [-np.sin(angle), 0, np.cos(angle), offset[2]],
                    [0, 0, 0, 1],
                ]
            )
        if axis == 2:
            return np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0, offset[0]],
                    [np.sin(angle), np.cos(angle), 0, offset[1]],
                    [0, 0, 1, offset[2]],
                    [0, 0, 0, 1],
                ]
            )
        print(f"Invalid axis: {axis}")
        return None

    def rotation_matrix_to_quaternion(self, R):
        """
        Convert a 3x3 rotation matrix to a quaternion [x, y, z, w].
        """
        # Ensure the matrix is numpy array
        R = np.array(R, dtype=float)

        trace = np.trace(R)
        if trace > 0:
            s = 2.0 * np.sqrt(trace + 1.0)
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
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

    def forward_kinematics(self, angles=None):
        """
        Forward Kinematics for SO Arm 101
        Args:
            None
        Returns:
            End effector transformation matrix, in radians/cm
        """
        if angles is None:
            angles = self.angles
        result = np.diag([1, 1, 1, 1])
        for index, angle in enumerate(angles):
            result = result @ self.transformation_matrix(
                angle, self.rotation_axis[index], self.link_offsets[index]
            )
        result = result @ self.transformation_matrix(0, 0, self.grab_position_offset)
        quaternion = self.rotation_matrix_to_quaternion(result[:3, :3])
        position = result[:3, 3]
        return [quaternion, position]

    def inverse_kinematics_analytical(self, euler_angles, position):
        """
        Inverse Kinematics for SO Arm 101 (Analytical)
        Args:
            euler_angles: euler angles of the end effector
            position: position of the end effector
        Returns:
            Angles of the joints
        """
        angles = [0, 0, 0, 0, 0, 0]
        # Your code here
        return angles

    def inverse_kinematics_numerical(self, euler_angles, position):
        """
        Inverse Kinematics for SO Arm 101 (Numerical)
        Args:
            euler_angles: euler angles of the end effector
            position: position of the end effector
        Returns:
            Angles of the joints
        """
        angles = [0, 0, 0, 0, 0, 0]
        # Your code here
        return angles


if __name__ == "__main__":
    arm = SOARM101()
    transformation_matrix = arm.forward_kinematics(
        angles=[np.pi / 3, np.pi / 2, 0, -np.pi / 4, 0, 0]
    )
    quaternion = arm.rotation_matrix_to_quaternion(transformation_matrix[:3, :3])
    position = transformation_matrix[:3, 3]
    print(position)
    print(quaternion)
