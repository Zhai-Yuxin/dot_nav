model used: qwen vl series

setup:
- python3 -m venv --system-site-packages venv
- source venv/bin/activate
- pip install -U openai
- get api key and set as env variable

(paths just for reference)
- cd Workspaces/proj2_ws/src
- git clone ...
- cd Workspaces/proj2_ws
- colcon build --symlink-install
- source ./install/setup.bash

Commands (as of now):
- ros2 launch dot_nav sim.launch.py
- ros2 launch slam_toolbox online_async_launch.py use_sim_time:=true
- ros2 launch nav2_bringup navigation_launch.py params_file:=/root/Workspaces/proj2_ws/src/dot_nav/configs/nav2_params.yaml use_sim_time:=true
- python3 ./proj2_ws/src/dot_nav/dot_nav/dot_node.py