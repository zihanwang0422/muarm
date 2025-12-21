import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from hw2_msg.msg import TrajectoryCommand
from hw2.SOARM101 import SOARM101
from hw2.save_trajectory_to_file import save_trajectory_to_file
import math


class CubicTrajectoryGenerator(Node):
    def __init__(self):
        super().__init__('cubic_trajectory_generator')
        self.arm = SOARM101()
        self.subscription = self.create_subscription(
            TrajectoryCommand,
            '/hw2_q2_1155249290',
            self.trajectory_callback,
            10)
        self.publisher = self.create_publisher(
            Float64MultiArray,
            # '/hw2_q2_joint_cmd_1155249290',
            '/joint_angles_command',
            10)
        
        # Trajectory state
        self.current_trajectory = None
        self.trajectory_index = 0
        self.timer = None
        self.trajectory_joint_space = None
        self.trajectory_to_save = None

    # SOARM101 Joint Limits (from simulator)
    JOINT_LIMITS = [
        (-90, 90),      # Joint 1: -90° to 90°
        (0, 180),       # Joint 2: 0° to 180°
        (-170, 25),     # Joint 3: -170° to 25°
        (-180, 180),    # Joint 4: -180° to 180°
        (-90, 90),      # Joint 5: -90° to 90°
        (-180, 180),    # Joint 6: -180° to 180°
    ]

    def normalize_joint_angles(self, angles):
        """
        Normalize joint angles to simulator-specific ranges.
        Args:
            angles: list of 6 joint angles in radians
        Returns:
            list of 6 normalized joint angles in radians
        """
        normalized = []
        for i, angle in enumerate(angles):
            min_deg, max_deg = self.JOINT_LIMITS[i]
            min_rad = np.deg2rad(min_deg)
            max_rad = np.deg2rad(max_deg)
            
            # First normalize to [-π, π]
            norm_angle = np.arctan2(np.sin(angle), np.cos(angle))
            
            # Clamp to joint-specific range
            if norm_angle < min_rad:
                norm_angle = min_rad
            elif norm_angle > max_rad:
                norm_angle = max_rad
                
            normalized.append(norm_angle)
        
        return normalized


    def euler_to_rotation_matrix(self, euler_xyz):
        """
        Convert intrinsic Euler angles (XYZ) to rotation matrix.
        Args:
            euler_xyz: [roll, pitch, yaw] in radians
        Returns:
            3x3 rotation matrix
        """
        roll, pitch, yaw = euler_xyz[0], euler_xyz[1], euler_xyz[2]
        
        # Intrinsic XYZ rotation: R = R_z(yaw) * R_y(pitch) * R_x(roll)
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        
        R_x = np.array([[1, 0, 0],
                       [0, cos_r, -sin_r],
                       [0, sin_r, cos_r]])
        
        R_y = np.array([[cos_p, 0, sin_p],
                       [0, 1, 0],
                       [-sin_p, 0, cos_p]])
        
        R_z = np.array([[cos_y, -sin_y, 0],
                       [sin_y, cos_y, 0],
                       [0, 0, 1]])
        
        R = R_z @ R_y @ R_x
        return R

    def rotation_matrix_to_quaternion(self, R):
        """
        Convert rotation matrix to quaternion [x, y, z, w].
        """
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

    def unwrap_euler_angles(self, angles_prev, angles_curr):
        """
        Apply shortest path unwrap to Euler angles to avoid jumps across ±π.
        Args:
            angles_prev: previous Euler angles [roll, pitch, yaw]
            angles_curr: current Euler angles [roll, pitch, yaw]
        Returns:
            unwrapped current angles
        """
        unwrapped = np.array(angles_curr)
        for i in range(3):
            diff = angles_curr[i] - angles_prev[i]
            # If difference is greater than π, wrap it
            if diff > np.pi:
                unwrapped[i] -= 2 * np.pi
            elif diff < -np.pi:
                unwrapped[i] += 2 * np.pi
        return unwrapped

    def generate_cubic_trajectory(self, start_pos, end_pos, start_vel, end_vel, 
                                   start_euler, end_euler, w_start, w_end, duration, frequency):
        """
        Generate cubic trajectory in task space.
        Args:
            start_pos: [x, y, z] in mm
            end_pos: [x, y, z] in mm
            start_vel: [vx, vy, vz] in mm/s
            end_vel: [vx, vy, vz] in mm/s
            start_euler: [roll, pitch, yaw] in radians
            end_euler: [roll, pitch, yaw] in radians
            w_start: [wx, wy, wz] in rad/s
            w_end: [wx, wy, wz] in rad/s
            duration: trajectory duration in seconds
            frequency: publishing frequency in Hz
        Returns:
            task_space_trajectory: list of [position, quaternion] for each time step
        """
        num_points = int(duration * frequency)
        dt = 1.0 / frequency
        trajectory = []
        
        # Apply unwrapping to end_euler relative to start_euler for shortest path
        end_euler_unwrapped = self.unwrap_euler_angles(start_euler, end_euler)
        
        # Compute cubic polynomial coefficients for position (once)
        a0_pos = np.array(start_pos)
        a1_pos = np.array(start_vel)
        a2_pos = (3 * (np.array(end_pos) - np.array(start_pos)) / (duration**2) - 
                  2 * np.array(start_vel) / duration - np.array(end_vel) / duration) if duration > 0 else np.zeros(3)
        a3_pos = (2 * (np.array(start_pos) - np.array(end_pos)) / (duration**3) + 
                  (np.array(start_vel) + np.array(end_vel)) / (duration**2)) if duration > 0 else np.zeros(3)
        
        # Compute cubic polynomial coefficients for Euler angles (once)
        a0_euler = np.array(start_euler)
        a1_euler = np.array(w_start)
        a2_euler = (3 * (end_euler_unwrapped - np.array(start_euler)) / (duration**2) - 
                   2 * np.array(w_start) / duration - np.array(w_end) / duration) if duration > 0 else np.zeros(3)
        a3_euler = (2 * (np.array(start_euler) - end_euler_unwrapped) / (duration**3) + 
                   (np.array(w_start) + np.array(w_end)) / (duration**2)) if duration > 0 else np.zeros(3)
        
        # Previous Euler angles for unwrapping during trajectory
        prev_euler = np.array(start_euler)
        
        for i in range(num_points):
            t = i * dt
            if t > duration:
                t = duration
            
            # Position at time t
            position = a0_pos + a1_pos * t + a2_pos * t**2 + a3_pos * t**3
            
            # Euler angles at time t
            euler = a0_euler + a1_euler * t + a2_euler * t**2 + a3_euler * t**3
            
            # Apply unwrapping to current euler to avoid jumps across ±π
            if i > 0:
                euler = self.unwrap_euler_angles(prev_euler, euler)
            
            # Update previous euler for next iteration
            prev_euler = euler.copy()
            
            # Convert Euler angles to rotation matrix, then to quaternion
            R = self.euler_to_rotation_matrix(euler)
            quaternion = self.rotation_matrix_to_quaternion(R)
            
            trajectory.append([position, quaternion])
        
        return trajectory

    def trajectory_callback(self, msg):
        """
        Callback when receiving TrajectoryCommand message.
        """
        self.get_logger().info('Received trajectory command')
        
        # Extract trajectory parameters
        start_pos = np.array(msg.start_position)
        end_pos = np.array(msg.end_position)
        start_vel = np.array(msg.v_start_pos)
        end_vel = np.array(msg.v_end_pos)
        start_euler = np.array(msg.start_euler_xyz)
        end_euler = np.array(msg.end_euler_xyz)
        w_start = np.array(msg.w_start_euler)
        w_end = np.array(msg.w_end_euler)
        duration = msg.duration
        frequency = msg.frequency
        
        # Generate task space trajectory
        task_trajectory = self.generate_cubic_trajectory(
            start_pos, end_pos, start_vel, end_vel,
            start_euler, end_euler, w_start, w_end,
            duration, frequency
        )
        
        # Convert to joint space using IK solver
        self.get_logger().info(f'Converting {len(task_trajectory)} task space points to joint space using numerical IK...')
        joint_trajectory = []
        previous_angles = None  # Use previous solution as initial guess for better convergence
        for i, (pos, quat) in enumerate(task_trajectory):
            # Use numerical IK (more reliable and accurate)
            # Use previous angles as initial guess for faster and more accurate convergence
            if previous_angles is not None:
                # Set current angles to previous solution as initial guess
                self.arm.angles = previous_angles.copy()
            
            joint_angles = self.arm.inverse_kinematics_numerical(quat, pos)
            # Ensure angles are normalized to [-π, π] range
            joint_angles = [np.arctan2(np.sin(a), np.cos(a)) for a in joint_angles]
            joint_trajectory.append(joint_angles)
            previous_angles = joint_angles  # Store for next iteration
            
            if (i + 1) % 100 == 0:
                self.get_logger().info(f'  Converted {i + 1}/{len(task_trajectory)} points')
        
        self.get_logger().info(f'Joint space conversion completed: {len(joint_trajectory)} points')
        
        # Store trajectory for publishing (file saving will be done after publishing completes)
        self.trajectory_joint_space = joint_trajectory
        self.trajectory_index = 0
        self.trajectory_to_save = joint_trajectory.copy()  # Keep a copy for saving later
        
        # Cancel existing timer if any
        if self.timer is not None:
            self.timer.cancel()
        
        # Create timer to publish at given frequency
        # Reduce frequency for simulator (use 20Hz instead of original frequency)
        timer_period = 1.0 / frequency  # seconds
        self.timer = self.create_timer(timer_period, self.publish_trajectory)
        
        self.get_logger().info(f'Generated trajectory with {len(joint_trajectory)} points, publishing at {frequency} Hz')

    def publish_trajectory(self):
        """
        Publish current joint angles from trajectory.
        """
        if self.trajectory_joint_space is None or self.trajectory_index >= len(self.trajectory_joint_space):
            # Trajectory finished, cancel timer
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None
            self.get_logger().info('Trajectory publishing completed')
            
            # Save trajectory to file after publishing is complete
            if hasattr(self, 'trajectory_to_save') and self.trajectory_to_save is not None:
                try:
                    # Save trajectory in radians (as per homework requirements)
                    save_trajectory_to_file(self.trajectory_to_save, self.get_logger())
                except Exception as e:
                    self.get_logger().error(f'Failed to save trajectory to file: {str(e)}')
                finally:
                    self.trajectory_to_save = None  # Clear after saving
            return
        
        # Get current joint angles (6×1 vector)
        current_joint_vector = self.trajectory_joint_space[self.trajectory_index]
        
        # Print 6×1 joint vector at current t_i
        self.get_logger().info(f't_i={self.trajectory_index}: [{current_joint_vector[0]:.6f}, {current_joint_vector[1]:.6f}, {current_joint_vector[2]:.6f}, {current_joint_vector[3]:.6f}, {current_joint_vector[4]:.6f}, {current_joint_vector[5]:.6f}]')
        
        # Normalize joint angles to simulator-specific ranges (keep in radians)
        normalized_angles = self.normalize_joint_angles(current_joint_vector)
        
        # Publish joint angles in radians (connector handles conversion to degrees)
        msg = Float64MultiArray()
        msg.data = [float(angle) for angle in normalized_angles]
        self.publisher.publish(msg)
        
        self.trajectory_index += 1


def main(args=None):
    rclpy.init(args=args)
    node = CubicTrajectoryGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
