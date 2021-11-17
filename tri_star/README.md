# TRI-STAR

## libs needed
1. Open3d: `pip install open3d`
2. ros-numpy: `sudo apt install ros-melodic-ros-numpy`
3. meshlabxml: get the repo from <https://github.com/ScazLab/MeshLabXML>, and then go the the parent folder of MeshLabXML, and run: `pip install -e MeshLabXML`

## configuration
1. most configs are set in the config file `config.xml`. The important parameters in the file is:
- robot_platform_type: which robot is using.
- robot_platform_side: some robot has two arms like Baxter. Set which arm is used. Default is left.
- data_learning_robot_platform_type: the training data is from which platform. this is for star 3. 
- is_testing: for the trajectory generated, whether to return key frames only or the entire trajectory.
- is_multiple_computers: whether the computer that controls the robot and the computer running the tri_star are different computers.
- candidate_tasks: the tool-use tasks.
- learned_data_dir: star 1 training data.

2. Robot related configuration are in the file `[robot]_config.xml`. It includes the following parameters:
- sub_cam_matrix and master_cam_matrix: the hand-eye calibration matrices of the two cameras.
- workspace_min_boundary and workspace_max_boundary: the boundary of the workspace.
- joint_names: the names of the joints of the robot.
- scan_tool_perception_workspace_min_boundary, scan_tool_perception_workspace_max_boundary, scan_tool_robot_boundary: the parameters used in scanning tools.

## steps to run
1. hand-eye calibration: `roslaunch tri_star single_azure_driver.launch` and `roslaunch tri_star stage_zero_calibrate_cam.launch`. We used the azure sensor, and a aruco code is attached to the robot's gripper.
2. scan 3D models using UR5e: `roslaunch tri_star stage_one_scan_object.launch` to scan the models. `roslaunch tri_star stage_one_process_scans.launch` to post-processing the scans, and `roslaunch tri_star step_one_normalize_pc.launch` to normalize the scans. After you scanned the point clouds, create a directory called `pointcloud` and two sub-directoreis in `goals` and `tools`, which save the point clouds of the manipulanda and tools.
3. train and test star 1: `roslaunch tri_star crow_tool_pointcloud.launch`. Create a directory called learned_samples to save the training samples. And follow the intructions.
4. test star 1 control: `roslaunch tri_star crow_tool_star_1_control.launch`.
5. test star 2: `roslaunch tri_star crow_tool_star_2.launch`.
6. test star 2 control: `roslaunch tri_star crow_tool_star_2_control.launch`.
7. test star 3: `roslaunch tri_star crow_tool_star_3.launch`.

For star 1 control, star 2 and its control, and star 3, the task, tool and goal (i.e., the manipulandum in the paper) needs to set in the python file to make it simple.

## to run on a simulator
In the launch file: 
1. replace `object_monitor.py` with `object_monitor_simulator.py`.
2. add `simulated_world_perception.py`.
3. set `is_simulator` to `True`
To manipulate the objects in the simulated world, run `rosrun tri_star simulator_interaction.py`.

