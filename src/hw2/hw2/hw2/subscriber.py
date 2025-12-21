import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math
from geometry_msgs.msg import TransformStamped

# Import SOARM101 class
from hw2.SOARM101 import SOARM101  

class IKSubscriber(Node):
    def __init__(self):
        super().__init__('ik_subscriber')
        self.subscription = self.create_subscription(
            TransformStamped,
            '/hw2_q1_tf2_1155249290',
            self.listener_callback,
            10)
        self.publisher_analytical = self.create_publisher(Float64MultiArray, '/hw2_q1_ana_1155249290', 10)
        self.publisher_numerical = self.create_publisher(Float64MultiArray, '/hw2_q1_num_1155249290', 10)
        # Initialize SOARM101
        self.arm = SOARM101()

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
            min_rad = math.radians(min_deg)
            max_rad = math.radians(max_deg)
            
            # First normalize to [-π, π]
            norm_angle = math.atan2(math.sin(angle), math.cos(angle))
            
            # Clamp to joint-specific range
            if norm_angle < min_rad:
                norm_angle = min_rad
            elif norm_angle > max_rad:
                norm_angle = max_rad
                
            normalized.append(norm_angle)
        
        return normalized



    def listener_callback(self, msg):
        """
        Callback function to process incoming TransformStamped messages.
        Computes both analytical and numerical IK solutions and publishes the results.
        """
        quaternion = msg.transform.rotation
        position = msg.transform.translation
        
        # Compute analytical IK
        angles_analytical = self.arm.inverse_kinematics_analytical(quaternion, position)
        # Normalize to simulator-specific joint limits
        angles_analytical_norm = self.normalize_joint_angles(angles_analytical)
        msg_ana = Float64MultiArray()
        msg_ana.data = [float(angle) for angle in angles_analytical_norm]
        self.publisher_analytical.publish(msg_ana)

        # Compute numerical IK
        angles_numerical = self.arm.inverse_kinematics_numerical(quaternion, position)
        # Normalize to simulator-specific joint limits
        angles_numerical_norm = self.normalize_joint_angles(angles_numerical)
        msg_num = Float64MultiArray()
        msg_num.data = [float(angle) for angle in angles_numerical_norm]
        self.publisher_numerical.publish(msg_num)

        # Print output information
        self.get_logger().info("=" * 60)
        self.get_logger().info("Output:")
        self.get_logger().info(f"  Analytical IK joint angles (rad): [{angles_analytical[0]:.6f}, {angles_analytical[1]:.6f}, {angles_analytical[2]:.6f}, {angles_analytical[3]:.6f}, {angles_analytical[4]:.6f}, {angles_analytical[5]:.6f}]")
        self.get_logger().info(f"  Numerical IK joint angles (rad):  [{angles_numerical[0]:.6f}, {angles_numerical[1]:.6f}, {angles_numerical[2]:.6f}, {angles_numerical[3]:.6f}, {angles_numerical[4]:.6f}, {angles_numerical[5]:.6f}]")
        self.get_logger().info("=" * 60)

def main(args=None):
    rclpy.init(args=args)
    node = IKSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()