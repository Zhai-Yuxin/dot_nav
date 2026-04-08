from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    robot_name = 'simple_car'
    gz_world = '/root/Workspaces/proj2_ws/src/dot_nav/worlds/simple_house.sdf'  # change for different worlds
    robot_urdf = '/root/Workspaces/proj2_ws/src/dot_nav/descriptions/simple_car.urdf'
    bridge_params = '/root/Workspaces/proj2_ws/src/dot_nav/configs/bridge_params.yaml'
    pointcloud_to_scan_params = '/root/Workspaces/proj2_ws/src/dot_nav/configs/pointcloud_to_laserscan.yaml'
    ekf_params = '/root/Workspaces/proj2_ws/src/dot_nav/configs/ekf.yaml'
    rviz_world = '/root/Workspaces/proj2_ws/src/dot_nav/configs/rviz_robot_config.rviz'
    nav2_params = '/root/Workspaces/proj2_ws/src/dot_nav/configs/nav2_params.yaml'
    slam_params = '/root/Workspaces/proj2_ws/src/dot_nav/configs/mapper_params_online_async.yaml'

    gazebo_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
            ),
            launch_arguments={'gz_args': f'-r {gz_world}'}.items()
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

    pointcloud_to_scan = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        output='screen',
        parameters=[pointcloud_to_scan_params],
        remappings=[('cloud_in', '/lidar/points'), ('scan', '/scan'),],
    )

    robot_localization = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_params, {'use_sim_time': True}],
    )

    rviz_launch = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_world],
            parameters=[{"use_sim_time": True}],
        )

    slam_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('slam_toolbox'), 'launch', 'online_async_launch.py')   # localization_launch.py for localization, online_async_launch.py for mapping
            ),
            launch_arguments={'use_sim_time': 'true',
                              'slam_params_file': slam_params}.items(),
        )

    nav2_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py')
            ),
            launch_arguments={'use_sim_time': 'true',
                              'params_file': nav2_params}.items(),
        )

    explore_lite_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('explore_lite'), 'launch', 'explore.launch.py')
            ),
            launch_arguments={'use_sim_time': 'true'}.items(),
        )

    explore_status = ExecuteProcess(
        cmd=[
            "ros2",
            "topic",
            "pub",
            "--once",
            "/explore/resume",
            "std_msgs/msg/Bool",
            "{data: false}",
        ]
    )

    return LaunchDescription([
        gazebo_launch,
        spawn_entity,
        robot_state_publisher,
        bridge,
        pointcloud_to_scan,
        robot_localization,
        rviz_launch,
        slam_launch,
        nav2_launch,
        explore_lite_launch,
        explore_status
    ])


    # dot_node = Node(
    #         package='dot_nav',
    #         executable='dot_node',
    #         name='dot_node',
    #         output='screen',
    #     )