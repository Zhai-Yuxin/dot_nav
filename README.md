# DotNav

This project is developed in ROS2 Jazzy environment, using docker image provided by OSRF (Open Source Robotics Foundation) with support to ROS2 Jazzy and Gz-sim Harmonic.

## Model
Model used: **qwen3-vl-plus**

Model choice can be changed at dot_node.py, see see https://www.alibabacloud.com/help/model-studio/getting-started/models for list of qwen models)

## Initial setup
Docker environment:
```
-- Download and install docker desktop, wsl ubuntu 24 --
docker pull osrf/ros:jazzy-desktop-full  # download prebuilt container image from remote
docker run -it --name ros2_container --net=host --privileged osrf/ros:jazzy-desktop-full bash  # create container
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
apt update && apt install -y git
```
Python venv:
```
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -U openai
-- get api key and set as env variable --
```
Repo cloning:
```
cd Workspaces/proj2_ws/src
git clone https://github.com/Zhai-Yuxin/dot_nav.git
git clone https://github.com/robo-friends/m-explore-ros2.git
cd m-explore-ros2/
rm -rf map_merge/
```
Build:
```
cd Workspaces/proj2_ws
colcon build --symlink-install
source ./install/setup.bash
```

## Run
In terminal 1:
```
cd Workspaces/proj2_ws/
source ./install/setup.bash
export GZ_SIM_RESOURCE_PATH=/root/Workspaces/proj2_ws/src/dot_nav/worlds/models
ros2 launch dot_nav sim.launch.py
```
In terminal 2:
```
source venv/bin/activate
python3 Workspaces/proj2_ws/src/dot_nav/dot_nav/dot_node.py
```
Optional: switch worlds at sim.launch.py
```
gz_world = '/root/Workspaces/proj2_ws/src/dot_nav/worlds/simple_house.sdf'
```

## Save
```
rviz2 window -> panel -> add new panel for slam_toolbox -> save & serialize map
```
Remember to update mapper_params_online_async.yaml and sim.launch.py for switch between mapping and localisation mode for SLAM.
```
# mapper_params_online_async.yaml
mode: localization
map_file_name: /root/Workspaces/proj2_ws/src/dot_nav/map/simple_house

# sim.launch.py
os.path.join(get_package_share_directory('slam_toolbox'), 'launch', 'localization_launch.py')
```

## Miscellaneous
To control the robot with keyboard input:
```
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p stamped:=true
```

To launch and test exploration node alone:
```
ros2 launch explore_lite explore.launch.py
ros2 topic pub --once /explore/resume std_msgs/msg/Bool "{data: false}"
ros2 topic pub --once /explore/resume std_msgs/msg/Bool "{data: true}"
```