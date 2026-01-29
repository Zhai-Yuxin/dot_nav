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

Commands to run:
- ros2 launch dot_nav sim.launch.py
- python3 ./proj2_ws/src/dot_nav/dot_nav/dot_node.py

(save map) rvix2 -> panel -> add new panel for slam_toolbox -> save & serialize map
- rmb to update mapper_params_online_async.yaml and sim.launch.py for switch between mapping and localisation mode for slam
