import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, PoseStamped
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import Image

import math
import threading
from time import sleep
from cv_bridge import CvBridge
import cv2

class DotNode(Node):
    def __init__(self):
        super().__init__('dot_node')
        self.gt_sub = self.create_subscription(TFMessage, '/ground_truth', self.gt_callback, 10)
        self.gt = None
        self.get_gt = False

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image', self.image_callback, 1)
        self.view = None
        self.viewcount = 0
        self.get_view = False

        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        ##llm setup

        self._stop_event = threading.Event()
        self.task_thread = threading.Thread(target=self.task)
        self.task_thread.start()

    def task(self):
        while rclpy.ok() and not self._stop_event.is_set():
            user_input = input("Instruction: ")
            if user_input.lower() == 'exit':
                print("ctrl-c to exit")
                break
            self.get_view = True
            self.get_gt = True
            while self.gt is None or self.view is None:
                sleep(0.1)
            cv2.imwrite('/root/Workspaces/camera_images/camera_image_'+str(self.viewcount)+'.png', self.view)
            self.viewcount += 1
            print(self.gt)
            sleep(2)
            ##llm call + output parse
            ##send goal to nav2
            # node.send_nav_goal(-2.0, 1.0, math.radians(180.0))
            # node.send_nav_goal(x, y, math.radians(deg))
            self.gt = None
            self.view = None

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
