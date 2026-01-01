from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    robot_name = 'simple_car'
    gz_world = '/root/Workspaces/proj2_ws/src/dot_nav/worlds/simple_shapes.sdf'
    robot_urdf = '/root/Workspaces/proj2_ws/src/dot_nav/descriptions/simple_car.urdf'
    bridge_params = "/root/Workspaces/proj2_ws/src/dot_nav/configs/bridge_params.yaml"
    rviz_world = '/root/Workspaces/proj2_ws/src/dot_nav/configs/rviz_robot_config.rviz'

    gazebo_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
            ),
            launch_arguments={'gz_args': gz_world}.items()
        )

    spawn_entity = Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                    '-name', robot_name,
                    '-topic', 'robot_description'],
            output='screen',
            parameters=[{"use_sim_time": True}],
        )

    robot_state_publisher = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': open(robot_urdf).read(), 
                    'use_sim_time': True}],
        )

    bridge = Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='ros_gz_bridge',
            output='screen',
            arguments=[
                '--ros-args',
                '-p',
                f'config_file:={bridge_params}'
            ],
            parameters=[{"use_sim_time": True}],
        )

    rviz_launch = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_world],
            parameters=[{"use_sim_time": True}],
        )

    return LaunchDescription([
        gazebo_launch,
        spawn_entity,
        robot_state_publisher,
        bridge,
        rviz_launch,
    ])


    # dot_node = Node(
    #         package='dot_nav',
    #         executable='dot_node',
    #         name='dot_node',
    #         output='screen',
    #     )

    # ekf_node = Node(
    #     package='robot_localization',
    #     executable='ekf_node',
    #     name='ekf_filter_node',
    #     output='screen',
    #     parameters=['/root/Workspaces/proj2_ws/src/dot_nav/configs/ekf.yaml', {'use_sim_time': True}]
    # )

    # nav2_dir = get_package_share_directory('nav2_bringup')
    # map_file = '/root/Workspaces/proj2_ws/src/dot_nav/maps/layout1.yaml'
    # nav2_launch = os.path.join(nav2_dir, 'launch', 'bringup_launch.py')
    # nav2_params = '/root/Workspaces/proj2_ws/src/dot_nav/config/nav2_params.yaml'

    # map_launch = Node(
    #         package='nav2_map_server',
    #         executable='map_server',
    #         name='map_server',
    #         output='screen',
    #         parameters=[{'yaml_filename': '/root/Workspaces/proj2_ws/src/dot_nav/maps/layout1.yaml'}]
    #     )

        # joint_state_publisher,
        # map_launch,
        # dot_node,
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(nav2_launch),
        #     launch_arguments={'map': map_file, 'use_sim_time': 'false', 'params_file': nav2_params}.items(),
        # ),

    # Static TF to map the laser frame produced by the gz bridge
    # static_tf = Node(
    #         package='tf2_ros',
    #         executable='static_transform_publisher',
    #         name='static_tf_lidar',
    #         output='screen',
    #         parameters=[{"use_sim_time": True}],
    #         arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'simple_car/base_footprint/gpu_lidar'] # args: x y z yaw pitch roll parent_frame child_frame
    #     )