import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import random
import math

class JointPublisher(Node):
    def __init__(self):
        super().__init__('joint_publisher')

        # Create publisher for joint angles with topic name including SID
        self.publisher_ = self.create_publisher(
            Float64MultiArray,
            '/hw1_q1_1155249290',
            10
        )
        
        # Create timer with 50Hz frequency (0.02 seconds period)
        self.timer = self.create_timer(0.02, self.timer_callback)
        
        self.get_logger().info(
            'Joint Publisher node started, publishing at 50Hz')

    def timer_callback(self):
        # Generate random joint angles (fix the random function call)
        joint_angles = [random.uniform(0, math.pi) for _ in range(6)]

        # Create Float64MultiArray message
        msg = Float64MultiArray()
        msg.data = joint_angles
        
        # Publish the joint angles
        self.publisher_.publish(msg)
        
        # Log the published data
        angles_str = [f"{angle:.3f}" for angle in joint_angles]
        self.get_logger().info(f'Published joint angles: {angles_str}')


def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create the publisher node
    joint_publisher = JointPublisher()
    
    try:
        # Keep the node alive and responsive
        rclpy.spin(joint_publisher)
    except KeyboardInterrupt:
        joint_publisher.get_logger().info('Publisher node stopped by user')
    finally:
        # Clean shutdown
        joint_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
