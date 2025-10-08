import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math
from tf2_msgs.msg import TransformStamped

# Import SOARM101 class
from hw1.SOARM101 import SOARM101


class FKSubscriber(Node):
    def __init__(self):
        super().__init__('fk_subscriber')
        # Initialize SOARM101
        self.arm = SOARM101()

        # Create subscriber for joint angles from Problem 1
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/hw1_q1_1155249290',
            self.listener_callback,
            10
        )
        
        # Create publisher for transformation data
        self.publisher_ = self.create_publisher(
            TransformStamped,
            '/hw1_q2_1155249290',
            10
        )
        
        self.get_logger().info('FK Subscriber node started')

    def matrix_to_quaternion(self, rotation_matrix):
        """Convert rotation matrix to quaternion"""
        R = rotation_matrix
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2  # s=4*qw
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s=4*qx
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s=4*qy
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s=4*qz
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
            
        return [x, y, z, w]

    def listener_callback(self, msg):
        joints = msg.data  # 6 joint angles in radians
        if len(joints) != 6:
            self.get_logger().warn(
                'Received incorrect number of joint angles!')
            return

        # Compute forward kinematics
        transform = self.arm.forward_kinematics(joints)
        
        if transform is None:
            self.get_logger().error('Forward kinematics failed!')
            return

        # Extract position (translation) from transformation matrix
        position = [
            round(float(transform[0, 3]), 4),
            round(float(transform[1, 3]), 4),
            round(float(transform[2, 3]), 4)
        ]
        
        # Extract rotation matrix and convert to quaternion
        rotation_matrix = transform[:3, :3]
        quat = self.matrix_to_quaternion(rotation_matrix)
        quaternion = [round(q, 4) for q in quat]
        
        # Create TransformStamped message
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = 'base_link'
        tf_msg.child_frame_id = 'end_effector'
        
        # Set translation
        tf_msg.transform.translation.x = position[0]
        tf_msg.transform.translation.y = position[1]
        tf_msg.transform.translation.z = position[2]
        
        # Set rotation (quaternion)
        tf_msg.transform.rotation.x = quaternion[0]
        tf_msg.transform.rotation.y = quaternion[1]
        tf_msg.transform.rotation.z = quaternion[2]
        tf_msg.transform.rotation.w = quaternion[3]
        
        # Publish the transformation
        self.publisher_.publish(tf_msg)
        
        self.get_logger().info(
            f'Published TF: pos=[{position[0]}, {position[1]}, '
            f'{position[2]}], quat=[{quaternion[0]}, {quaternion[1]}, '
            f'{quaternion[2]}, {quaternion[3]}]')


def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create the subscriber node
    fk_subscriber = FKSubscriber()
    
    try:
        # Keep the node alive and responsive
        rclpy.spin(fk_subscriber)
    except KeyboardInterrupt:
        fk_subscriber.get_logger().info('FK Subscriber node stopped by user')
    finally:
        # Clean shutdown
        fk_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
