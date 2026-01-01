import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, PoseStamped
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import Image
# from sensor_msgs.msg import PointCloud2
# from sensor_msgs_py import point_cloud2

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
        self.image_dir = '/root/Workspaces/camera_images/'

        # self.img_depth_sub = self.create_subscription(PointCloud2, '/depth_camera/points', self.image_depth_callback, 1)
        # self.view_depth = None
        # self.get_view_depth = False

        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )

        self._stop_event = threading.Event()
        self.task_thread = threading.Thread(target=self.task)
        self.task_thread.start()

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
            while self.gt is None or self.view is None:
                sleep(1)

            print(self.gt)
            filepath = self.image_dir + 'camera_image_' + str(self.viewcount) + '.png'
            cv2.imwrite(filepath, self.view)

            base64_image = self.encode_image(filepath)
            instr = (
                f"Instruction: {user_input}.\n"
                f"The robot has a current state: {self.gt}.\n"
                "The image is from the camera at the front of the robot, with horizontal_fov set to 1.5708 and clip distance from 0.1 to 10.\n"
                "Based on the current position, and view, output the single line next targeted position in terms of x, y and yaw in degrees with a short description to achieve the instruction.\n"
                "Example output: <description>forward 2m and left 1m, face 45 degrees</description> <target>x=1.0, y=1.0, yaw=45.0</target>\n"
                "If the instruction cannot be accomplished in one step, output the next best step.\n"
                "If the instruction is already accomplished by the current state, just output 'Instruction accomplished.'."
            )
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"},},
                        {"type": "text", "text": instr},
                    ],
                },
            ]

            ## add try except block warp for api call
            # completion = self.client.chat.completions.create(
            #     model="qwen3-vl-plus",  # For a list of models, see https://www.alibabacloud.com/help/model-studio/getting-started/models
            #     messages=messages
            # )
            # output = completion.choices[0].message.content
            # print("Output: ", output)
            output = "1"  # Placeholder for testing without API call

            match = re.search(r"<target>(.*?)</target>", output)
            if match:
                content = match.group(1)
                vals = {k: float(v) for k, v in re.findall(r"(x|y|yaw)\s*=\s*([\d\.\-]+)", content)}
                x = vals.get("x")
                y = vals.get("y")
                yaw = vals.get("yaw") # + math.degrees(2*math.atan2(self.gt.orientation.z, self.gt.orientation.w))
            elif output.strip() == "Instruction accomplished.":
                print("Instruction accomplished.")
                x = None
                y = None
                yaw = None
            else:
                print("No valid tag found, no motion for now.")
                x = None
                y = None
                yaw = None

            if x is not None:
                print(f"target: x={x}, y={y}, yaw={yaw}")
                # send_nav_goal(-2.0, 1.0, math.radians(180.0))
                self.send_nav_goal(x, y, math.radians(yaw))  ## need wait for complete or no ?

            self.viewcount += 1
            self.gt = None
            self.view = None

    # def image_depth_callback(self, msg: PointCloud2):
    #     if self.get_view_depth:
    #         print("0")
    #         # PointCloud2 → list of (x, y, z)
    #         points_list = list(point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True))
    #         if len(points_list) == 0:
    #             return
    #         print("0.5")
    #         points = np.array(points_list, dtype=np.float32)
    #         print("0.7")
    #         if points.ndim != 2 or points.shape[1] != 3:
    #             print("Invalid point cloud shape:", points.shape)
    #             return
    #         print("1")
    #         fx = 320  #cam_info.k[0]
    #         fy = 320  #cam_info.k[4]
    #         cx = 320  #cam_info.k[2]
    #         cy = 240  #cam_info.k[5]
    #         width = 640
    #         height = 480
    #         self.view_depth = np.zeros((height, width), dtype=np.float32)
    #         print("2")
    #         # Project points to image plane
    #         X = points[:, 0]
    #         Y = points[:, 1]
    #         Z = points[:, 2]
    #         # Filter Z ≤ 0 (behind camera or invalid)
    #         valid = Z > 0
    #         X = X[valid]; Y = Y[valid]; Z = Z[valid]
    #         print("3")
    #         # Pixel coordinates
    #         u = (X * fx / Z + cx).astype(np.int32)
    #         v = (Y * fy / Z + cy).astype(np.int32)
    #         print("4")
    #         # Keep only pixels inside the image range
    #         in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    #         u = u[in_bounds]
    #         v = v[in_bounds]
    #         Z = Z[in_bounds]
    #         print("5")
    #         # Fill depth image (keep nearest point for each pixel)
    #         for uu, vv, zz in zip(u, v, Z):
    #             if self.view_depth[vv, uu] == 0 or zz < self.view_depth[vv, uu]:
    #                 self.view_depth[vv, uu] = zz
    #         print("6")
    #         self.get_view_depth = False

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

        print("Waiting for NAV2 action server...")
        self.nav_client.wait_for_server()

        print(f"Sending NAV2 goal: (x={x}, y={y}, yaw={math.degrees(yaw)} deg)")
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            print('Goal rejected by NAV2.')
            return

        print('Goal accepted — navigating...')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        status = future.result().status
        print(f'Navigation completed.')
        # print(f'Navigation completed with status: {status}')

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
