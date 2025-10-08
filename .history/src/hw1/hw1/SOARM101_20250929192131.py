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
        self.link_offsets = [ # x, y, z, in local frames, in cm
            [0, 3.115, 11.97],
            [0, 11.235, -2.8],
            [0, 13.49, 0.485],
            [0, 5.48, 0],
            [0, 3.15, 2],
            [0, 7.6, -1.15]
        ]


    def forward_kinematics(self, angles=None):
        """
        Forward Kinematics for SO Arm 101
        Args:
            angles: List of 6 joint angles in radians
        Returns:
            End effector transformation matrix, in cm
        """
        if angles is None:
            angles = [0, 0, 0, 0, 0, 0]
        
        if len(angles) != 6:
            return None
        
        # Initialize transformation matrix as identity
        T = np.eye(4)
        
        # For each joint, compute transformation matrix
        for i in range(6):
            # Get joint angle and link offset
            theta = angles[i]
            offset = self.link_offsets[i]
            
            # Create transformation matrix for this joint
            # Assuming Z-axis rotation joints (common for robotic arms)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            
            # Individual transformation matrix
            Ti = np.array([
                [cos_theta, -sin_theta, 0, offset[0]],
                [sin_theta, cos_theta, 0, offset[1]],
                [0, 0, 1, offset[2]],
                [0, 0, 0, 1]
            ])
            
            # Accumulate transformations
            T = np.dot(T, Ti)
        
        return T


if __name__ == "__main__":
    arm = SOARM101()
    print(arm.forward_kinematics(angles=[0, 0, 0, 0, 0, 0]))
