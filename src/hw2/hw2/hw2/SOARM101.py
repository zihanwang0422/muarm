# DO NOT TOUCH!

import numpy as np

class SOARM101:
    """
    Forward Kinematics for SO Arm 101
    units: radians/mm
    use SOARM101() to initialize the class
    use arm.forward_kinematics(angles) to get the end effector transformation matrix
    """
    def __init__(self):
        self.angles = [0, 0, 0, 0, 0, 0]
        self.rotation_axis = [2, 0, 0, 0, 1, 0] # z, x, x, x, y, x
        # Convert from cm to mm: multiply by 10
        self.link_offsets = [
            [0, 0, 0],
            [0, 31.15, 119.7],
            [0, 112.35, -28.0],
            [0, 134.9, 4.85],
            [0, 54.8, 0],
            [0, 31.5, 20.0]
        ]
        self.grab_position_offset = [0, 76.0, -11.5]

    def transformation_matrix(self, angle, axis, offset):
        """
        Generate Transformation matrix for each joint
        Args:
            angle: angle of the joint, in radians
            axis: axis of the joint, x:0, y:1, z:2
            offset: offset of the joint, in mm
        Returns:
            Transformation matrix
        """
        if axis == 0: # x-axis
            return np.array([[1, 0, 0, offset[0]],
                             [0, np.cos(angle), -np.sin(angle), offset[1]],
                             [0, np.sin(angle), np.cos(angle), offset[2]],
                             [0, 0, 0, 1]])
        if axis == 1: # y-axis
            return np.array([[np.cos(angle), 0, np.sin(angle), offset[0]],
                             [0, 1, 0, offset[1]],
                             [-np.sin(angle), 0, np.cos(angle), offset[2]],
                             [0, 0, 0, 1]])
        if axis == 2:
            return np.array([[np.cos(angle), -np.sin(angle), 0, offset[0]],
                             [np.sin(angle), np.cos(angle), 0, offset[1]],
                             [0, 0, 1, offset[2]],
                             [0, 0, 0, 1]])
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
            End effector transformation matrix, in radians/mm
        """
        if angles is None:
            angles = self.angles
        result = np.diag([1, 1, 1, 1])
        for index, angle in enumerate(angles):
            result = result @ self.transformation_matrix(angle, self.rotation_axis[index], self.link_offsets[index])
        result = result @ self.transformation_matrix(0, 0, self.grab_position_offset)
        quaternion = self.rotation_matrix_to_quaternion(result[:3, :3])
        position = result[:3, 3]
        return [quaternion, position]

    def quaternion_to_rotation_matrix(self, quaternion):
        """
        Convert quaternion [x, y, z, w] to 3x3 rotation matrix.
        Args:
            quaternion: quaternion as [x, y, z, w] or geometry_msgs.msg.Quaternion
        Returns:
            3x3 rotation matrix
        """
        # Handle both numpy array and geometry_msgs.msg.Quaternion
        if hasattr(quaternion, 'x'):
            x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
        else:
            x, y, z, w = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
        
        # Normalize quaternion
        norm = np.sqrt(x*x + y*y + z*z + w*w)
        if norm > 0:
            x, y, z, w = x/norm, y/norm, z/norm, w/norm
        
        # Convert to rotation matrix
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])
        return R
    
    def compute_jacobian(self, angles):
        """
        Compute the Jacobian matrix for the end effector.
        Args:
            angles: joint angles [6]
        Returns:
            6x6 Jacobian matrix (3 for position, 3 for orientation)
        """
        # Numerical differentiation step
        delta = 1e-6
        
        # Get current end effector pose
        quat_current, pos_current = self.forward_kinematics(angles)
        R_current = self.quaternion_to_rotation_matrix(quat_current)
        
        # Convert rotation matrix to axis-angle representation for error computation
        # We'll use a simpler approach: compute position and orientation errors separately
        jacobian = np.zeros((6, 6))
        
        for i in range(6):
            # Perturb joint i
            angles_perturbed = angles.copy()
            angles_perturbed[i] += delta
            
            # Compute forward kinematics with perturbation
            quat_perturbed, pos_perturbed = self.forward_kinematics(angles_perturbed)
            R_perturbed = self.quaternion_to_rotation_matrix(quat_perturbed)
            
            # Position Jacobian (linear velocity)
            jacobian[:3, i] = (pos_perturbed - pos_current) / delta
            
            # Orientation Jacobian (angular velocity)
            # Compute rotation difference: R_diff = R_perturbed @ R_current^T
            R_diff = R_perturbed @ R_current.T
            # Extract axis-angle from rotation matrix
            trace = np.trace(R_diff)
            angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            if angle > 1e-6:
                # Compute axis
                axis = np.array([
                    R_diff[2, 1] - R_diff[1, 2],
                    R_diff[0, 2] - R_diff[2, 0],
                    R_diff[1, 0] - R_diff[0, 1]
                ]) / (2 * np.sin(angle))
                angular_velocity = axis * angle / delta
            else:
                angular_velocity = np.zeros(3)
            
            jacobian[3:, i] = angular_velocity
        
        return jacobian
    
    def rotation_matrix_error(self, R_desired, R_current):
        """
        Compute orientation error as axis-angle representation.
        Args:
            R_desired: desired rotation matrix
            R_current: current rotation matrix
        Returns:
            3D orientation error vector
        """
        R_error = R_desired @ R_current.T
        trace = np.trace(R_error)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        if angle < 1e-6:
            return np.zeros(3)
        
        # Compute axis
        axis = np.array([
            R_error[2, 1] - R_error[1, 2],
            R_error[0, 2] - R_error[2, 0],
            R_error[1, 0] - R_error[0, 1]
        ]) / (2 * np.sin(angle))
        
        return axis * angle

    def inverse_kinematics_analytical(self, quaternion, position):
        """
        Inverse Kinematics for SO Arm 101 (Analytical with iterative refinement)
        Based on robot structure: rotation axes [Z, X, X, X, Y, X]
        
        Due to complex wrist offsets coupling position and orientation,
        this uses an iterative approach for wrist center computation.
        
        Args:
            quaternion: quaternion of the end effector (geometry_msgs.msg.Quaternion or [x,y,z,w])
            position: position of the end effector [x, y, z] in mm (geometry_msgs.msg.Vector3 or array)
        Returns:
            Angles of the joints [6] in radians
        """
        # Extract position
        if hasattr(position, 'x'):
            target_pos = np.array([position.x, position.y, position.z])
        else:
            target_pos = np.array([position[0], position[1], position[2]])
        
        # Convert quaternion to rotation matrix
        R_target = self.quaternion_to_rotation_matrix(quaternion)
        
        # Initialize angles
        angles = np.zeros(6)
        
        try:
            # Iterative approach: estimate wrist center considering wrist joint offsets
            max_iter = 10
            wrist_center = target_pos.copy()
            
            for iteration in range(max_iter):
                # Step 1: Estimate wrist center by subtracting end effector and wrist offsets
                # Total wrist offset (joints 4, 5, 6 + grab)
                total_wrist_offset = np.array([
                    0,
                    self.link_offsets[4][1] + self.link_offsets[5][1] + self.grab_position_offset[1],
                    self.link_offsets[4][2] + self.link_offsets[5][2] + self.grab_position_offset[2]
                ])
                
                # Rotate by target orientation
                wrist_center = target_pos - R_target @ total_wrist_offset
                
                # Step 2: Solve Joint 1 (Z-axis rotation)
                angles[0] = np.arctan2(wrist_center[1], wrist_center[0])
                
                # Step 3: Transform to frame after joint 1
                r_xy = np.sqrt(wrist_center[0]**2 + wrist_center[1]**2)
                p1_y = r_xy
                p1_z = wrist_center[2]
                
                # Step 4: Account for joint 2 offset
                p1_y -= self.link_offsets[1][1]
                p1_z -= self.link_offsets[1][2]
                
                # Step 5: 2-link IK for joints 2 and 3
                L2 = np.sqrt(self.link_offsets[2][1]**2 + self.link_offsets[2][2]**2)
                L3 = np.sqrt(self.link_offsets[3][1]**2 + self.link_offsets[3][2]**2)
                
                alpha2 = np.arctan2(-self.link_offsets[2][2], self.link_offsets[2][1])
                alpha3 = np.arctan2(self.link_offsets[3][2], self.link_offsets[3][1])
                
                D = np.sqrt(p1_y**2 + p1_z**2)
                
                # Check if target is reachable
                if D > L2 + L3 or D < abs(L2 - L3):
                    raise ValueError("Target position unreachable")
                
                cos_theta3 = (D**2 - L2**2 - L3**2) / (2 * L2 * L3)
                cos_theta3 = np.clip(cos_theta3, -1, 1)
                
                theta3_geom = np.arccos(cos_theta3)
                angles[2] = theta3_geom - alpha2 - alpha3
                
                phi = np.arctan2(p1_z, p1_y)
                psi = np.arctan2(L3 * np.sin(theta3_geom), L2 + L3 * np.cos(theta3_geom))
                angles[1] = phi - psi + alpha2
                
                # Step 6: Compute orientation for wrist joints 4, 5, 6
                T01 = self.transformation_matrix(angles[0], self.rotation_axis[0], self.link_offsets[0])
                T12 = self.transformation_matrix(angles[1], self.rotation_axis[1], self.link_offsets[1])
                T23 = self.transformation_matrix(angles[2], self.rotation_axis[2], self.link_offsets[2])
                T34 = self.transformation_matrix(0, self.rotation_axis[3], self.link_offsets[3])
                
                T04 = T01 @ T12 @ T23 @ T34
                R04 = T04[:3, :3]
                
                # Update target rotation with current estimate
                R46_desired = R04.T @ R_target
                
                # Step 7: Extract X-Y-X Euler angles for joints 4, 5, 6
                angles[4] = np.arctan2(np.sqrt(R46_desired[0,1]**2 + R46_desired[0,2]**2), R46_desired[0,0])
                
                s5 = np.sin(angles[4])
                
                if abs(s5) > 1e-6:
                    angles[3] = np.arctan2(R46_desired[0,1] / s5, R46_desired[0,2] / s5)
                    angles[5] = np.arctan2(R46_desired[1,0] / s5, -R46_desired[2,0] / s5)
                else:
                    angles[3] = 0
                    angles[5] = np.arctan2(-R46_desired[2,1], R46_desired[1,1])
                
                # Check convergence by FK
                quat_check, pos_check = self.forward_kinematics(angles)
                error = np.linalg.norm(pos_check - target_pos)
                
                if error < 1.0:  # Good enough
                    break
                    
        except Exception as e:
            # If analytical solution fails, fallback to numerical IK
            return self.inverse_kinematics_numerical(quaternion, position)
        
        return angles.tolist()

    def inverse_kinematics_numerical(self, quaternion, position, max_iterations=100, tolerance=1e-4):
        """
        Inverse Kinematics for SO Arm 101 (Numerical using Jacobian inverse)
        Uses iterative Newton-Raphson method with Jacobian pseudo-inverse.
        
        Args:
            quaternion: quaternion of the end effector (geometry_msgs.msg.Quaternion or [x,y,z,w])
            position: position of the end effector [x, y, z] in mm (geometry_msgs.msg.Vector3 or array)
            max_iterations: maximum number of iterations
            tolerance: convergence tolerance
        Returns:
            Angles of the joints [6] in radians
        """
        # Extract target position
        if hasattr(position, 'x'):
            target_pos = np.array([position.x, position.y, position.z])
        else:
            target_pos = np.array([position[0], position[1], position[2]])
        
        # Convert quaternion to rotation matrix
        R_target = self.quaternion_to_rotation_matrix(quaternion)
        
        # Initialize joint angles (use current angles or zeros)
        angles = np.array(self.angles.copy(), dtype=float)
        
        # Damping factor for Levenberg-Marquardt style regularization
        lambda_reg = 0.1
        
        for iteration in range(max_iterations):
            # Compute current forward kinematics
            quat_current, pos_current = self.forward_kinematics(angles)
            R_current = self.quaternion_to_rotation_matrix(quat_current)
            
            # Compute error
            pos_error = target_pos - pos_current
            orient_error = self.rotation_matrix_error(R_target, R_current)
            
            # Combined error vector [position(3), orientation(3)]
            error = np.concatenate([pos_error, orient_error])
            
            # Check convergence
            error_norm = np.linalg.norm(error)
            if error_norm < tolerance:
                break
            
            # Compute Jacobian
            J = self.compute_jacobian(angles)
            
            # Compute Jacobian pseudo-inverse with regularization (damped least squares)
            # J_pinv = J^T (J J^T + lambda I)^(-1)
            J_reg = J @ J.T + lambda_reg * np.eye(6)
            try:
                J_pinv = J.T @ np.linalg.inv(J_reg)
            except:
                # If singular, use Moore-Penrose pseudo-inverse
                J_pinv = np.linalg.pinv(J)
            
            # Update joint angles
            delta_angles = J_pinv @ error
            
            # Step size control (can be adaptive)
            step_size = 1.0
            angles_new = angles + step_size * delta_angles
            
            # Optional: clamp angles to reasonable ranges
            # angles_new = np.clip(angles_new, -np.pi, np.pi)
            
            angles = angles_new
        
        # Normalize angles to [-π, π] range
        angles = np.arctan2(np.sin(angles), np.cos(angles))
        
        return angles.tolist()

if __name__ == "__main__":
    arm = SOARM101()
    transformation_matrix = arm.forward_kinematics(angles=[np.pi/3, np.pi/2, 0, -np.pi/4, 0, 0])
    quaternion = arm.rotation_matrix_to_quaternion(transformation_matrix[:3, :3])
    position = transformation_matrix[:3, 3]
    print(position)
    print(quaternion)
