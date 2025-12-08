import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, PoseStamped
from tf2_msgs.msg import TFMessage

import math
import threading
from time import sleep

class DotNode(Node):
    def __init__(self):
        super().__init__('dot_node')
        self.gt_sub = self.create_subscription(TFMessage, '/ground_truth', self.gt_callback, 10)
        self.gt = Pose()

        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        self.task_thread = threading.Thread(target=self.task)
        self.task_thread.start()

    def task(self):
        while rclpy.ok():
            user_input = input("Instruction: ")
            if user_input.lower() == 'exit':
                print("ctrl-c to exit")
                break
            sleep(2)
            ##llm set up + call + output parse
            ##send goal to nav2
            # node.send_nav_goal(-2.0, 1.0, math.radians(180.0))
            # node.send_nav_goal(x, y, math.radians(deg))

    def gt_callback(self, msg):
        if not msg.transforms:
            return

        for tf in msg.transforms:
            if tf.child_frame_id == 'simple_car':
                transform = tf.transform
                self.gt.position.x = transform.translation.x
                self.gt.position.y = transform.translation.y
                self.gt.position.z = transform.translation.z
                self.gt.orientation = transform.rotation
                break

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

def main(args=None):
    rclpy.init(args=args)
    node = DotNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
