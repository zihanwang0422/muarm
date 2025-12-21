import threading
import time
import struct
import socket
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def bytes_to_float(byte_array, byte_order='<'):
    # Assuming the byte array represents a 32-bit floating-point number (single precision)
    float_size = 4

    # Ensure the byte array length matches the float size
    if len(byte_array) != float_size:
        raise ValueError(f"Invalid byte array length. Expected {float_size} bytes.")

    # Convert byte array to float
    format_str = f"{byte_order}f"
    return struct.unpack(format_str, byte_array)[0]


def float_to_bytes(float_num, byte_order='<'):
    # Convert float to bytes
    # print(float_num)
    byte_array = struct.pack(f"{byte_order}f", float_num)
    return byte_array


def bytes_to_int(byte_array, byte_order='little'):
    return int.from_bytes(byte_array, byte_order)


def int_to_bytes(integer, byte_order='little', signed=True):
    # Convert integer to bytes
    byte_size = 2  # Calculate the number of bytes needed
    byte_array = integer.to_bytes(byte_size, byte_order, signed=signed)
    return byte_array


class RobotArm:
    def __init__(self):
        # internal variables
        self.joint_angles = [0.0 for _ in range(6)]
        self.joint_ids = [1, 2, 3, 4, 5, 6]

    def set_joint_angle(self, joint_id, angle):
        ''' Set the joint angle of a single joint '''
        self.joint_angles[joint_id - 1] = angle
        return True

    def get_joint_angle(self, joint_id):
        ''' Get the joint angle of a single joint '''
        return self.joint_angles[joint_id - 1]

    def set_joint_angle_group(self, joint_ids, angles):
        ''' Set the joint angles of a group of joints '''
        for i, joint_id in enumerate(joint_ids):
            self.joint_angles[joint_id - 1] = angles[i]
        return True

    def get_joint_angle_group(self, joint_ids):
        ''' Get the joint angles of a group of joints '''
        result = []
        for joint_id in joint_ids:
            result.append(self.joint_angles[joint_id - 1])
        return result

class UDPClient:
    def __init__(self):
        self.udp_socket = None
        self.is_connected = False
        self.client_addr = None
        self.is_locked = False
        self.servo_angle_list = [0 for _ in range(6)]

        self.sender_thread = None
        self.receiver_thread = None

        self.arm = None

    def start_server(self, server_host, server_port, arm):
        ''' Start the server '''
        print("Preparing server")
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.bind((server_host, server_port))
            self.arm = arm
            self.servo_angle_list = self.arm.get_joint_angle_group(self.arm.joint_ids)
            self.server_thread = threading.Thread(target=self.start_receiver)
            self.server_thread.start()
        except Exception as e:
            print(f"Server start failed. Error: {e}")
            return False
        return True

    def start_sender(self):
        print("Starting sender")
        while True:
            try:
                time.sleep(0.02)
                if not self.is_connected or self.client_addr is None or self.is_locked:
                    break
                send_data = []
                send_data += int_to_bytes(len(self.arm.joint_ids)) # number of joints
                send_data += int_to_bytes(0) # no PWM
                # add angle data
                for angle in self.arm.joint_angles:
                    send_data += float_to_bytes(angle)
                # print(send_data)
                # send data
                self.udp_socket.sendto(bytes(send_data), self.client_addr)
            except Exception as e:
                print(f"Sender failed. Error: {e}")
                break
        print("sender stopped")
  
    def start_receiver(self):
        print("Server started, waiting for client...")
        
        # start receiver loop
        while True:
            try:
                # receive data
                data, addr = self.udp_socket.recvfrom(1024)  
                
                # check command type
                cmd_type = data[0]
                if cmd_type == 0: # connect message
                    print("Connect Message from client: ", addr)
                    cmd_data = data[1]
                    if cmd_data == 0: # disconnect
                        print("Disconnect")
                        self.is_connected = False
                        if self.sender_thread is not None:
                            self.sender_thread.join()
                    elif cmd_data == 1: # connect
                        # close the last connection if any
                        if self.is_connected:
                            print("Already connected, restarting sender")
                            self.is_connected = False
                            if self.sender_thread is not None:
                                self.sender_thread.join()

                        # prepare to connect
                        self.is_connected = True
                        self.client_addr = addr
                        self.servo_angle_list = self.arm.joint_angles.copy()
                        
                        # start sender thread
                        self.sender_thread = threading.Thread(target=self.start_sender)
                        self.sender_thread.start()
                        print("connect success, sender started`")
                    else: # invalid command
                        print("Invalid connect command")
                        continue
                elif cmd_type == 1: # unlock/lock message
                    print("Unlock/Lock Message from client: ", addr)
                    cmd_data = data[1]
                    if cmd_data == 0: # unlock
                        print("Unlock")
                        self.is_locked = False

                        # close sender thread
                        if self.sender_thread is not None:
                            self.sender_thread.join()
                    elif cmd_data == 1: # lock
                        print("Lock")
                        # start sender thread
                        try:
                            self.is_locked = True   
                            self.sender_thread = threading.Thread(target=self.start_sender)
                            self.sender_thread.start()
                        except Exception as e:
                            print("Failed to start sender thread: ", e)
                            self.is_locked = False
                            continue
                        print("lock success, sender started")
                    else: # invalid command
                        print("Invalid unlock/lock command")
                        continue
                elif cmd_type == 2: # angle command
                    print("Angle Command from client: ", addr)
                    # parse angle command
                    try:
                        for i, joint_id in enumerate(self.arm.joint_ids):   
                            tmp_angle = bytes_to_float(data[1 + i * 4: 1 + (i + 1) * 4], '<')
                            self.servo_angle_list[joint_id - 1] = tmp_angle
                        print("Angle command received: ", self.servo_angle_list)
                        self.arm.set_joint_angle_group(self.arm.joint_ids, self.servo_angle_list)
                    except Exception as e:
                        print("Failed to parse angle command: ", e)
                        continue
            except Exception as e:
                pass

class SOArmNode(Node):
    def __init__(self, arm):
        super().__init__("so_arm_node")
        self.publisher = self.create_publisher(Float64MultiArray, "joint_angles", 10)
        self.subscription = self.create_subscription(Float64MultiArray, "joint_angles_command", self.joint_angles_callback, 10)
        self.timer = self.create_timer(0.02, self.timer_callback)
        self.arm = arm

    def timer_callback(self):
        msg = Float64MultiArray()
        # print("Publishing joint angles: ", self.arm.get_joint_angle_group(self.arm.joint_ids))
        msg.data = self.arm.get_joint_angle_group(self.arm.joint_ids)
        self.publisher.publish(msg)

    def joint_angles_callback(self, msg):
        for idx, angle in enumerate(msg.data):
            msg.data[idx] = np.rad2deg(angle)
        self.arm.set_joint_angle_group(self.arm.joint_ids, msg.data)

class ROSClient:
    def __init__(self, arm):
        self.node = None
        self.arm = arm

    def connect(self):
        print("Connecting to ROS...")
        rclpy.init()
        self.node = SOArmNode(self.arm)
        try:
            rclpy.spin(self.node)
        except KeyboardInterrupt:
            print("ROS disconnected")
            self.node.destroy_node()
            rclpy.shutdown()
            return True
        except Exception as e:
            print("ROS connection failed: ", e)
            self.node.destroy_node()
            rclpy.shutdown()
            return False
        finally:
            print("ROS disconnected")
            self.node.destroy_node()
            rclpy.shutdown()
            return False


UDP_SERVER_HOST = "0.0.0.0"
UDP_SERVER_PORT = 1234

# init robot arm
arm = RobotArm()

# init socket
udp_client = UDPClient()
try:
    udp_client.start_server(UDP_SERVER_HOST, UDP_SERVER_PORT, arm)
    print(f"UDP receiver started on {UDP_SERVER_HOST}:{UDP_SERVER_PORT}")
except Exception as e:
    print(f"UDP receiver start failed. Error: {e}")
    exit(-1)

# init ros client
ros_client = ROSClient(arm)
ros_client.connect()
