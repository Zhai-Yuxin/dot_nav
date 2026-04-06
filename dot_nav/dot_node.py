import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, PoseStamped
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool

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

        self.explore_pub = self.create_publisher(Bool, '/explore/resume', 10)

        self._stop_event = threading.Event()
        self.task_thread = threading.Thread(target=self.task)
        self.task_thread.start()

    def map_callback(self, msg):
        if self.get_map:
            self.map = np.array(msg.data, dtype=np.int8)  # occupancy probabilities in range [0,100], unknown is -1
            self.map = self.map.reshape((msg.info.height, msg.info.width))
            self.map = np.flipud(self.map)
            # print("Map received: size =", self.map.shape)  ### eg (223, 375)

            self.resolution = msg.info.resolution
            self.origin = msg.info.origin  # geometry_msgs/Pose
            # print(f"resolution: {self.resolution}, origin: {self.origin}")   ### eg 0.05 geometry_msgs.msg.Pose(position=geometry_msgs.msg.Point(x=-9.355363191853257, y=-5.596854321969525, z=0.0), orientation=geometry_msgs.msg.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))

            img = np.zeros_like(self.map, dtype=np.uint8)
            img[self.map == 0] = 255        # free
            img[self.map == 100] = 0        # occupied
            img[self.map == -1] = 127       # unknown

            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            px_per_meter = int(1 / self.resolution)
            for x in range(0, self.map.shape[1], px_per_meter):  # draw vertical grid lines
                cv2.line(img_color, (x, 0), (x, self.map.shape[0]), (200, 200, 200), 1)
            for y in range(0, self.map.shape[0], px_per_meter):  # draw horizontal grid lines
                cv2.line(img_color, (0, y), (self.map.shape[1], y), (200, 200, 200), 1)
            s=5
            pixel_x = int((self.gt.position.x - self.origin.position.x) / self.resolution)
            pixel_y = int((self.gt.position.y - self.origin.position.y) / self.resolution)
            pixel_y = self.map.shape[0] - pixel_y  # flip y for image coordinates
            pts = np.array([[pixel_x, pixel_y - s], [pixel_x - s, pixel_y + s], [pixel_x + s, pixel_y + s]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(img_color, [pts], (0, 0, 255))
            yaw = math.atan2(2.0 * (self.gt.orientation.w * self.gt.orientation.z), 1.0 - 2.0 * (self.gt.orientation.z ** 2))
            arrow_length = 20
            end_x = int(pixel_x + arrow_length * math.cos(yaw))
            end_y = int(pixel_y - arrow_length * math.sin(yaw))  # flip y for image coordinates
            cv2.arrowedLine(img_color, (pixel_x, pixel_y), (end_x, end_y), (255, 0, 0), 2, tipLength=0.3)

            ring_step_m = 1.0   # distance between rings in meters
            max_range_m = 15.0  # how far you want rings
            px_per_meter = int(1 / self.resolution)
            for d in np.arange(ring_step_m, max_range_m + ring_step_m, ring_step_m):
                radius_px = int(d * px_per_meter)
                cv2.circle(img_color, (pixel_x, pixel_y), radius_px, (0, 255, 255), 1)
                label = f"{d:.0f}"
                label_x = pixel_x + radius_px + 2
                label_y = pixel_y
                if 0 <= label_x < self.map.shape[1] and 0 <= label_y < self.map.shape[0]: # avoid drawing outside image
                    cv2.putText(img_color, label, (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            cv2.imwrite(self.image_dir + 'map_' + str(self.viewcount) + '.png', img_color)
            self.get_map = False

    def image_callback(self, msg):
        try:
            if self.get_view:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                self.view = cv_image
                cv2.imwrite(self.image_dir + 'camera_image_' + str(self.viewcount) + '.png', self.view)
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
            print("Received user input:", user_input)

            self.get_view = True
            self.get_gt = True
            self.get_map = True
            while self.get_gt or self.get_view or self.get_map:
                sleep(1)

            print(self.gt)
            filepath = self.image_dir + 'camera_image_' + str(self.viewcount) + '.png'
            camera_image = self.encode_image(filepath)
            filepath = self.image_dir + 'map_' + str(self.viewcount) + '.png'
            map_image = self.encode_image(filepath)
            print(f"round {self.viewcount} --------")

            instr_init = (
                "You are an intelligent navigation assistant for a robot tasked to analyze the provided information for the next action step to fulfill instruction from the user.\n"
            )
            instr_map = (
                f"The map image shows the occupancy grid map of the environment, with white as free space, black as obstacles, and darkgray as unknown areas.\n"
                # f"The map's origin has world coordinates ({self.origin.position.x}, {self.origin.position.y}) that is at the bottom left of the image.\n"
                "Metre grids with horizontal and vertical lines are drawn on the map image in light gray color (200, 200, 200) to assist better distance estimation, with grid lines every 1 meter.\n"
                "Distance rings are also drawn on the map image in yellow color (0, 200, 200) to assist better distance estimation, with grid lines every 1 meter and numerical labels to the right of each ring indicating the distance (in meters) from the robot that is at the center of the rings.\n"
                "The horizontal right direction of the map image is the 0 degree direction for the robot in world frame.\n"
                "The current position of the robot is indicated with a small red triangle on the map image, and current facing direction is indicated by a blue arrow from the triangle.\n"
                "x-axis is increasing to the right, and y-axis is increasing upwards on the map image.\n"
            )
            instr = (
                "The image is from the camera at the front of the robot, with camera sensor details horizontal_fov set to 1.5708 and clip distance from 0.1m to 10m.\n"
                f"Instruction: {user_input}.\n"
                f"The robot is at currently (ground truth): {self.gt}\n"
                "Based on the current position and view, analyse carefully and output the next targeted destination to achieve the instruction.\n"
                "Take note of the relative current position and facing direction of the robot reflected on the map image, the camera view corresponds to the robot's field of view at the current position.\n"
                "Navigation will be followed up with nav2 stack, so obstacle avoidance is not required to be taken into consideration, the output should be in a format that can be easily parsed for the next navigation goal.\n"
                "There are 2 output formats that are not to be confused or mixed. The output should only follow one of the below formats that best fits the need.\n"
                "Format 1: <description>...</description> <target>move_x=1.2 move_y=2.3 turn=90 dir=right</target>\n"
                "Format 2: <description>Instruction accomplished.</description>\n"
                "The <description> tag contains short description of the current view and position and intended destination.\n"
                "The <target> tag contains information about the next targeted position. Treat the current position (red triangle) as the origin in the map image, x is increasing positive to the horizontal right direction and y is increasing positive vertically upwards. Locate the targetted destination, output the 'move_x' and 'move_y' in meters with respect to the map, and 'turn' in degrees and 'dir' as the turning direction from the current facing direction.\n"
                "For example, <target>move_x=-1.2 move_y=2.3 turn=90 dir=right</target> means the targeted destination is 1.2 meters to the left and 2.3 meters upwards on the map image from the current position, and the robot should turn 90 degrees to the right from its current facing direction when it reaches the destination.\n"
                "If the instruction is already accomplished by the current state, use Format 2.\n"
                "If the instruction cannot be accomplished in one step, output the next best step.\n"
                "Take note of a radius of 0.5m around the robot as its body size when planning the destination.\n"
                "IMPORTANT: Distance should be estimated in combination with the map instead of estimating solely from camera view. Use the grid lines to estimate distances instead of pixels.\n"
                "IMPORTANT: Do NOT output multiple <target> tags in single response.\n"
                "If the instruction have similar meaning to 'explore the area', 'explore the map', 'go around', etc, ignore the above requirements and output just <explore>true</explore>\n"
                "If the instruction have similar meaning to 'stop exploring', 'cease exploration', etc, ignore the above requirements and output just <explore>false</explore>\n"
            )
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instr_init},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{map_image}"},},
                        {"type": "text", "text": instr_map},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{camera_image}"},},
                        {"type": "text", "text": instr},
                    ],
                },
            ]

            complete = False
            while not complete:
                try:
                    time = self.get_clock().now().to_msg()
                    completion = self.client.chat.completions.create(
                        model="qwen3-vl-plus",  # For a list of models, see https://www.alibabacloud.com/help/model-studio/getting-started/models
                        messages=messages
                    )
                    time_diff = self.get_clock().now().to_msg().sec - time.sec
                    print(f"API call latency: {time_diff} seconds")
                    output = completion.choices[0].message.content
                    print("Output: ", output)
                except Exception as e:
                    print(f"API call failed: {e}")
                # output = "<move>turn=180 dir=left move=0</move>"  # Placeholder for testing without API call
                # output = "tmp"

                x, y, z, w = None, None, None, None
                # match = re.search(r"<move>(.*?)</move>", output)
                # if match:
                #     content = match.group(1)
                #     vals = {k: float(v) for k, v in re.findall(r"(move|turn)\s*=\s*(-?\d+(?:\.\d+)?)", content)}
                #     turn = vals.get("turn", 0)
                #     move = vals.get("move", 0)
                #     dir_match = re.search(r"dir\s*=\s*(left|right)", content)
                #     dire = dir_match.group(1) if dir_match else "right"
                #     if dire == "right":
                #         turn = -turn
                #     yaw = math.atan2(2.0 * (self.gt.orientation.w * self.gt.orientation.z), 1.0 - 2.0 * (self.gt.orientation.z ** 2))
                #     yaw_new = yaw + math.radians(turn)
                #     x = self.gt.position.x + move * math.cos(yaw_new)
                #     y = self.gt.position.y + move * math.sin(yaw_new)
                #     z = math.sin(yaw_new / 2.0)
                #     w = math.cos(yaw_new / 2.0)
                # else:
                match = re.search(r"<target>(.*?)</target>", output)
                if match:
                    content = match.group(1)
                    vals = {k: float(v) for k, v in re.findall(r"(x|y|turn)\s*=\s*(-?\d+(?:\.\d+)?)", content)}
                    x = self.gt.position.x + vals.get("x")
                    y = self.gt.position.y + vals.get("y")
                    turn = vals.get("turn", 0)
                    dir_match = re.search(r"dir\s*=\s*(left|right)", content)
                    dire = dir_match.group(1) if dir_match else "right"
                    if dire == "right":
                        turn = -turn
                    yaw = math.atan2(2.0 * (self.gt.orientation.w * self.gt.orientation.z), 1.0 - 2.0 * (self.gt.orientation.z ** 2))
                    yaw_new = yaw + math.radians(turn)
                    z = math.sin(yaw_new / 2.0)
                    w = math.cos(yaw_new / 2.0)
                else:
                    match = re.search(r"<explore>(.*?)</explore>", output)
                    if match:
                        content = match.group(1).strip().lower()
                        if content == "true":
                            print("Start exploring...")
                            self.explore_pub.publish(Bool(data=True))
                        elif content == "false":
                            print("Stop exploring...")
                            self.explore_pub.publish(Bool(data=False))
                        else:
                            print("Invalid explore command, ignoring.")
                    else:
                        match = re.search(r"<description>(.*?)</description>", output)
                        if match:
                            content = match.group(1)
                            if content.strip() == "Instruction accomplished.":
                                print("Instruction accomplished.")
                        else:
                            print("No valid tag found, no motion for now.")
                    complete = True

                if x is not None:
                    print(f"target: x={x}, y={y}, deg={math.degrees(yaw_new)}")
                    self.nav_flag = True
                    self.send_nav_goal(x, y, z, w)

                self.viewcount += 1

                while self.nav_flag:
                    sleep(1)

                if not complete:
                    user_input = input("Whether to continue (y/n): ")
                    if user_input.lower() not in ['y', '', 'yes']:
                        complete = True
                    else:
                        self.get_view = True
                        self.get_gt = True
                        self.get_map = True
                        while self.get_gt or self.get_view or self.get_map:
                            sleep(1)
                        print(self.gt)
                        filepath = self.image_dir + 'camera_image_' + str(self.viewcount) + '.png'
                        camera_image = self.encode_image(filepath)
                        filepath = self.image_dir + 'map_' + str(self.viewcount) + '.png'
                        map_image = self.encode_image(filepath)

                        instr_cont = (
                            f"Current pose of the robot (ground truth): {self.gt}\n"
                            "The updated camera view and map images are provided, take note of the current view and position reflected on the map image, refer to the past messages for progress tracking.\n"
                            "Continue to analyze and provide the next targeted destination or confirm if the instruction is accomplished.\n"
                            "IMPT: Guidelines and outputs should follow the requirements from the initial message."
                        )
                        messages.append(completion.choices[0].message.model_dump())
                        messages.append({
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{camera_image}"},},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{map_image}"},},
                                    {"type": "text", "text": instr_cont},
                                ]
                            })

    # ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
    # "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 1.0, y: 2.0, z: 0.0}, orientation: {z: 0.0, w: 1.0}}}}"
    # 
    #  x = 0, y = 0, z = sin(θ/2), w = cos(θ/2)
    def send_nav_goal(self, x, y, z=0.0, w=1.0):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        # goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        # goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.z = z
        goal_msg.pose.pose.orientation.w = w

        self.nav_client.wait_for_server()
        print(f"Sending NAV2 goal...")
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
