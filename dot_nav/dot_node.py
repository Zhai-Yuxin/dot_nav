import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, PoseStamped
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid

import math
import threading
from time import sleep
from cv_bridge import CvBridge
import cv2
import numpy as np

from openai import OpenAI
import os
import re
import base64

class DotNode(Node):
    def __init__(self):
        super().__init__('dot_node')
        self.gt_sub = self.create_subscription(TFMessage, '/ground_truth', self.gt_callback, 10)
        self.gt = None
        self.get_gt = False

        self.bridge = CvBridge()
        self.img_sub = self.create_subscription(Image, '/camera/image', self.image_callback, 1)
        self.view = None
        self.viewcount = 0
        self.get_view = False
        self.image_dir = '/root/Workspaces/images/'

        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 1)
        self.map = None
        self.get_map = False

        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.nav_flag = False

        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )

        self._stop_event = threading.Event()
        self.task_thread = threading.Thread(target=self.task)
        self.task_thread.start()

    def map_callback(self, msg):
        if self.get_map:
            self.map = np.array(msg.data, dtype=np.int8)  # occupancy probabilities in range [0,100], unknown is -1
            self.map = self.map.reshape((msg.info.height, msg.info.width))
            self.map = np.flipud(self.map)
            print("Map received: size =", self.map.shape)

            self.resolution = msg.info.resolution
            self.origin = msg.info.origin  # geometry_msgs/Pose
            print(f"resolution: {self.resolution}, origin: {self.origin}")

            img = np.zeros_like(self.map, dtype=np.uint8)
            img[self.map == 0] = 255        # free
            img[self.map == 100] = 0        # occupied
            img[self.map == -1] = 127       # unknown
            cv2.imwrite(self.image_dir + "map.png", img)

            self.get_map = False

    def image_callback(self, msg):
        try:
            if self.get_view:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                self.view = cv_image
                self.get_view = False
        except Exception as e:
            print(f'Failed to process image: {e}')

    def gt_callback(self, msg):
        if not msg.transforms:
            return
        if self.get_gt:
            for tf in msg.transforms:
                if tf.child_frame_id == 'simple_car':
                    transform = tf.transform
                    self.gt = Pose()
                    self.gt.position.x = transform.translation.x
                    self.gt.position.y = transform.translation.y
                    self.gt.position.z = transform.translation.z
                    self.gt.orientation = transform.rotation
                    break
            self.get_gt = False

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def task(self):
        while rclpy.ok() and not self._stop_event.is_set():
            user_input = input("Instruction: ")
            if user_input.lower() == 'exit':
                print("ctrl-c to exit")
                break
            print("Received user input: ", user_input)

            self.get_view = True
            self.get_gt = True
            self.get_map = True
            while self.gt is None or self.view is None or self.map is None:
                sleep(1)

            print(self.gt)
            filepath = self.image_dir + 'camera_image_' + str(self.viewcount) + '.png'
            cv2.imwrite(filepath, self.view)
            camera_image = self.encode_image(filepath)
            filepath = self.image_dir + 'map.png'
            map_image = self.encode_image(filepath)

            # f"The robot has a current state: {self.gt}.\n"
            instr_map = (
                "The image shows the occupancy grid map of the environment, with white as free space, black as obstacles, and gray as unknown areas.\n"
                f"The map has a resolution of {self.resolution} meters per pixel, coordinate originate from the bottom left of the image, and the pose at origin is {self.origin}\n"
            )
            instr = (
                f"Instruction: {user_input}.\n"
                "The image is from the camera at the front of the robot, with horizontal_fov set to 1.5708 and clip distance from 0.1 to 10.\n"
                "Based on the current position's view, analyse carefully and output the next targeted movement needed to achieve the instruction with some helpful short descriptions.\n"
                "Example output format: <description>There is a box about 3m ahead and a wall on the right about 2m. Turn 30 degrees to the left from current position and then move forward 1.5m</description> <target>turn=30 dir=left move=1.5</target>\n"
                "Example output format: <description>It is empty ahead, a turn is needed to inspect the other direction. Turn 90 degrees to the right from current position</description> <target>turn=90 dir=right move=0</target>\n"
                "If the instruction cannot be accomplished in one step, output the next best step.\n"
                "If the instruction is already accomplished by the current state, output 'Instruction accomplished.'."
            )
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{map_image}"},},
                        {"type": "text", "text": instr_map},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{camera_image}"},},
                        {"type": "text", "text": instr},
                    ],
                },
            ]

            # try:
            #     completion = self.client.chat.completions.create(
            #         model="qwen3-vl-plus",  # For a list of models, see https://www.alibabacloud.com/help/model-studio/getting-started/models
            #         messages=messages
            #     )
            #     output = completion.choices[0].message.content
            #     print("Output: ", output)
            # except Exception as e:
            #     print(f"API call failed: {e}")
            output = "<target>turn=0 dir=right move=1</target>"  # Placeholder for testing without API call

            x, y, yaw = None, None, None
            match = re.search(r"<target>(.*?)</target>", output)
            if match:
                content = match.group(1)
                vals = {k: float(v) for k, v in re.findall(r"(turn|move)\s*=\s*(-?\d+(?:\.\d+)?)", content)}
                turn = vals.get("turn")
                move = vals.get("move")
                dir_match = re.search(r"dir\s*=\s*(left|right)", content)
                dire = dir_match.group(1) if dir_match else "right"
                if dire == "right":
                    turn = -turn
                yaw = math.atan2(2.0 * (self.gt.orientation.w * self.gt.orientation.z), 1.0 - 2.0 * (self.gt.orientation.z ** 2))
                yaw_new = yaw + math.radians(turn)
                x = self.gt.position.x + move * math.cos(yaw_new)
                y = self.gt.position.y + move * math.sin(yaw_new)
            elif output.strip() == "Instruction accomplished.":
                print("Instruction accomplished.")
            else:
                print("No valid tag found, no motion for now.")

            if x is not None:
                print(f"target: x={x}, y={y}, deg={math.degrees(yaw_new)}")
                self.nav_flag = True
                self.send_nav_goal(x, y, yaw_new)

            self.viewcount += 1
            self.gt = None
            self.view = None
            self.map = None

    # ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
    # "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 1.0, y: 2.0, z: 0.0}, orientation: {z: 0.0, w: 1.0}}}}"
    # 
    #  x = 0, y = 0, z = sin(θ/2), w = cos(θ/2)
    def send_nav_goal(self, x, y, yaw=0.0):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        # goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        # goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.nav_client.wait_for_server()
        print(f"Sending NAV2 goal: (x={x}, y={y}, yaw={math.degrees(yaw)} deg)")
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            print('Goal rejected by NAV2.')
            self.nav_flag = False
            return
        print('Goal accepted — navigating...')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        status = future.result().status
        print(f'Navigation completed.')
        print(f'Navigation completed with status: {status}')
        self.nav_flag = False

    def stop(self):
        self._stop_event.set()
        self.task_thread.join()

def main(args=None):
    rclpy.init(args=args)
    node = DotNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
