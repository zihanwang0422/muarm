## ROSE 5710 HW1

### Problem 1: Publisher Node

```python
#~/ros2_ws/src/hw1/publisher.py
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

```





```bash
cd ~/ros2_ws

colcon build --packages-select hw1

source install/setup.bash

ros2 run hw1 publisher
```

![image-20250929194332481](/home/wzh/.config/Typora/typora-user-images/image-20250929194332481.png)



```bash
ros2 topic echo /hw1_q1_1155249290
```

![image-20250929195428024](/home/wzh/.config/Typora/typora-user-images/image-20250929195428024.png)





### Problem 2: Subscriber + Publisher Node

```python
#~/ros2_ws/src/hw1/subscriber.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math
from geometry_msgs.msg import TransformStamped

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

```



```python
#~/ros2_ws/src/hw1/SOARM101.py
# DO NOT TOUCH!

import numpy as np


class SOARM101:
    """
    Forward Kinematics for SO Arm 101
    units: radians/cm
    use SOARM101() to initialize the class
    use arm.forward_kinematics(angles) to get the end effector
    transformation matrix
    """
    def __init__(self):
        # x, y, z, in local frames, in cm
        self.link_offsets = [
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

```



```bash
cd /home/wzh/Downloads/ros2_ws && colcon build --packages-select hw1

#bash 1
source install/setup.bash && ros2 run hw1 publisher

#bash 2
source install/setup.bash && ros2 run hw1 subscriber
```



* bash 1：publisher

![image-20250929200408062](/home/wzh/.config/Typora/typora-user-images/image-20250929200408062.png)



* bash 2：subscriber

![image-20250929200347232](/home/wzh/.config/Typora/typora-user-images/image-20250929200347232.png)

-

