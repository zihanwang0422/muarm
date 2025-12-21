import rclpy
from rclpy.node import Node
from hw2.SOARM101 import SOARM101
from geometry_msgs.msg import TransformStamped
import random
import math
from hw2_msg.msg import TrajectoryCommand   
class JointPublisher(Node):
    def __init__(self):
        super().__init__('joint_publisher')
        self.arm = SOARM101()
        self.publisher_ = self.create_publisher(TransformStamped, '/hw2_q1_tf2_1155249290', 10)
        timer_period = 1  # seconds, for 1 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.ctrl_msg_publisher = self.create_publisher(TrajectoryCommand, '/hw2_q1_ctrl_msg', 10)

    def timer_callback(self):
        msg = TransformStamped()
        angles = [random.uniform(0, math.pi) for _ in range(6)]
        quaternion, position = self.arm.forward_kinematics(angles)
        msg.transform.translation.x = position[0]
        msg.transform.translation.y = position[1]
        msg.transform.translation.z = position[2]
        msg.transform.rotation.x = quaternion[0]
        msg.transform.rotation.y = quaternion[1]
        msg.transform.rotation.z = quaternion[2]
        msg.transform.rotation.w = quaternion[3]
        
        # Print input information
        self.get_logger().info("=" * 60)
        self.get_logger().info("Input:")
        self.get_logger().info(f"  Desired end-effector position [x, y, z] (mm): [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
        self.get_logger().info(f"  Desired end-effector quaternion [x, y, z, w]: [{quaternion[0]:.6f}, {quaternion[1]:.6f}, {quaternion[2]:.6f}, {quaternion[3]:.6f}]")
        
        self.publisher_.publish(msg)

        ctrl_msg = TrajectoryCommand()
        ctrl_msg.start_position = [0.0, 0.0, 0.0]
        ctrl_msg.start_euler_xyz = [0.0, 0.0, 0.0]
        ctrl_msg.end_position = [100.0, 100.0, 100.0]
        ctrl_msg.end_euler_xyz = [math.pi, math.pi, math.pi]
        ctrl_msg.v_start_pos = [0.0, 0.0, 0.0]
        ctrl_msg.v_end_pos = [0.0, 0.0, 0.0] 
        ctrl_msg.w_start_euler = [0.0, 0.0, 0.0]
        ctrl_msg.w_end_euler = [0.0, 0.0, 0.0]
        ctrl_msg.duration = 10.0
        ctrl_msg.frequency = 100.0
        self.ctrl_msg_publisher.publish(ctrl_msg)



def main(args=None):
    rclpy.init(args=args)
    node = JointPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()