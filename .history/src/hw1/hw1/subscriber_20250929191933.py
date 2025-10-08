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


    def listener_callback(self, msg):
        joints = msg.data  # 6 joint angles in radians
        if len(joints) != 6:
            self.get_logger().warn('Received incorrect number of joint angles!')
            return

        # Compute forward kinematics
        transform = self.arm.forward_kinematics(joints)


        # TODO: Extract end-effector position and rotation from transformation matrix to tf2_msgs/msg/TransformStamped
        position = ...
        quaternion = ...
        tf_msg = ...
        
        # TODO: publish the tf_msg
        ...
        

def main(args=None):
    # TODO: Initialize ROS2 node and do the main loop
    ...

if __name__ == '__main__':
    main()