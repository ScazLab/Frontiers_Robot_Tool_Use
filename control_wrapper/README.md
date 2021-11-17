This is a ros wrapper to control several robot arm, including: 1)The UR arm with the Robotiq 2F 85 gripper, with moveit. 2)The Baxter robot arm with moveit. 3)(To come) The Kuka Youbot arm.

Please note that this is a wrap up for the robot controllers, not a replacement of a simulator.

# Usage

The control currently include the following functionalities:

- get/set the joint angles
- get/set the end effector pose (e.g., kinematics and inverse kinematics)
- open/close the gripper
- start/end the freedrive mode
- follow a trajectory
- add/remove, attach/detach an object/mesh
- check whether a pose could be reached, and return the joint changes if it could
- allow/disabllow the collision with an object/mesh.

The package has the following topics:

It has the following topics, in which `<robot>` should be replaced with the name of the robot you plan to use, and `<side>` is to control which robot as some robot has multiple arms. Currently supported robot names are: `ur`, `baxter` and `kuka`. The sides are: `left` and `right`. When a robot has only one arm like the UR arm, then just use left which is the default one.

- /control_wrapper/scene_objects: It provides a list of objects (msg: SceneObject) with its poses in the world. The objects can either attached to the end-effector, or free in the world. If it is attached, then its type is 1. Otherwise, it is 2.
- /\<robot\>/control_wrapper/\<side\>/connect: This is for UR specifically. It takes `Bool`, to start the externalcontrol program to connect the driver to the robot to enable robot arm control.
- /\<robot\>/control_wrapper/\<side\>/enable_freedrive: This is for UR and Kuka. It takes `Bool`. For UR, it starts or ends the freedrive mode. It will stop the externalcontrol program. If you send True, then you will need to manually restart the externalcontrol program by send true to the connect topic (as it is not known how long it would need to be in this state). However, if you exit the mode by sending False, you don't need to reconnect it manually, as the program already do it for you. For Kuka, it will turn off the stiffness of the joint. Completely it doesn't support gravity compensation.
- /\<robot\>/control_wrapper/\<side\>/gripper: It takes `Bool` to open or close the gripper. For UR, It will also stop the externalcontrol program. But you don't need to start externalcontrol in this case, since the program already does it for you.
- /\<robot\>/control_wrapper/\<side\>/scene_allow_collosion: It allows (True) or disallow (False) the collision of the object/mesh in the environment. It takes `SceneObjectAllowCollision`. To take effect, you may need to wait for a few seconds.

It runs the folliwng services:

- /\<robot\>/control_wrapper/\<side\>/get_joints: get the joint angles
- /\<robot\>/control_wrapper/\<side\>/get_pose: get the end effector pose
- /\<robot\>/control_wrapper/\<side\>/set_joints: set the joint angles
- /\<robot\>/control_wrapper/\<side\>/set_pose: set the end effector pose
- /\<robot\>/control_wrapper/\<side\>/check_pose: check whether a pose could be reached
- /\<robot\>/control_wrapper/\<side\>/follow_trajectory: follow a trajectory. Make sure all the points on the trajectory are difference. NOTE: do not add the starting pose if it is the same as the current pose. This is a moveit bug. <https://answers.ros.org/question/253004/moveit-problem-error-trajectory-message-contains-waypoints-that-are-not-strictly-increasing-in-time/>
- /\<robot\>/control_wrapper/\<side\>/add_object: add an object to scene. Currently support Box and Cylinder. For cylinder, the size is (height, radius, not use)
- /\<robot\>/control_wrapper/\<side\>/attach_object: attach an object to the end effector (there is no need to call add object as it already handles it). Currently support Box and Cylinder.
- /\<robot\>/control_wrapper/\<side\>/detach_object: attach an object from the end effector. If to_remove is set to True, it also remove the object from the scene.
- /\<robot\>/control_wrapper/\<side\>/remove_object: remove an object form the scene.
- /\<robot\>/control_wrapper/\<side\>/forward_kinematics: get the pose with joint angles.
- /\<robot\>/control_wrapper/\<side\>/get_interpolate_points: get the waypoints.

They are all blocking calls.

# Install the package

```bash
git clone https://github.com/ScazLab/control_wrapper.git
```

And then build the package

# Visualize in rviz

```bash
rosrun rviz rviz
```

If you don't see the robot,  add the planning scene.

You can visualize the robot even if you are running a simulator like Gazebo, and you should see the same result. 

# UR5e arm

## Installation

UR5e lib installation can be found (Using Ros Kinetic): <https://scazlab.github.io/ur5e_setup_guide.html> (**Note**: if you followed the instructions before May 20th 2020, please get the udpated files in ur_extra_changes).


The UR ros driver is the latest one found here: <https://github.com/UniversalRobots/Universal_Robots_ROS_Driver>

## Instructions:

you can roslaunch 
- `ur5e_cam_2f85_control.launch` for a real robot. 
- Or `sim_ur5e_cam_2f85_control.launch` if you works with the simulator.

**Or** If you wish to run the launch files separately
- Make sure in the robot driver launch file, the arg headless_mode is set to True
- Launch the ros driver launch file and moveit launch file
- Then launch `ur_control_wrapper.launch` 

## Demo:

To test it on the robot, you can rosrun the ur_demo.py file:

```bash
rosrun control_wrapper ur_demo.py
```

## Notes:

local_ip in connect.py needs to be updated if used on a different computer with the real robot.

# Baxter arm

## Installation

### Install Baxter libs

After ROS Indigo has been installed, here is the Baxter installation guide (adapted from <https://sdk.rethinkrobotics.com/wiki/Workstation_Setup>):

```bash
sudo apt-get update
sudo apt-get install git-core python-argparse python-wstool python-vcstools python-rosdep ros-indigo-control-msgs ros-indigo-joystick-drivers
```

cd to your ROS ws folder that you would like to save the baxter ros node. On mine, it is under *ros_lib_ws/src*

```bash
wstool init .
wstool merge https://raw.githubusercontent.com/RethinkRobotics/baxter/master/baxter_sdk.rosinstall
wstool update
```

Make sure you source ROS.

```bash
source /opt/ros/indigo/setup.bash
```

Then build it
```bash
catkin build
```

Configure Baxter simulator. Instructions adapted from <https://sdk.rethinkrobotics.com/wiki/Simulator_Installation>.First get the related lib:
```bash
sudo apt-get install gazebo2 ros-indigo-qt-build ros-indigo-driver-common ros-indigo-gazebo-ros-control ros-indigo-gazebo-ros-pkgs ros-indigo-ros-control ros-indigo-control-toolbox ros-indigo-realtime-tools ros-indigo-ros-controllers ros-indigo-xacro python-wstool ros-indigo-tf-conversions ros-indigo-kdl-parser
```

### Install moveit:

```bash
sudo apt-get update
sudo apt-get install ros-indigo-moveit-full
```

Install the trackik solver if you don't have it:
```bash
sudo apt-get install ros-indigo-trac-ik-kinematics-plugin
```

Get the Baxter moveit config, and save it to your favourite ros ws, for me, it is *ros_lib_ws/src*:
```bash
git clone https://github.com/ScazLab/baxter_moveit_config.git
```

Build the file

```bash
catkin build
```

### Get Simulator:

Then get the Baxter simulator ros nodes, install it at the same location where you install the baxter node, or wherever you like. I installed it under *ros_lib_ws/src*

```bash
wstool init .
wstool merge https://raw.githubusercontent.com/RethinkRobotics/baxter_simulator/master/baxter_simulator.rosinstall
wstool update
```

Then source it and build it:

```bash
source /opt/ros/indigo/setup.bash
catkin build
```

Then copy the baxter.sh file under src/baxter to *ros_lib_ws*, or your equivalent choice, and then:

```bash
chmod u+x baxter.sh
```

## Instructions

To communicate with the simulator or a real robot(This is needed whenever you start a new terminal to control the robot):
```bash
cd ros_lib_ws
./baxter.sh sim
```

If you are using a simulator, start the simulator (note:somehow running the launch file throw an error like cannot parse certain model. Restart the computer solved the problem):

```bash
roslaunch control_wrapper sim_baxter.launch
```

Wait until you see the following lines:
```bash
[ INFO] [1400513321.531488283, 34.216000000]: Simulator is loaded and started successfully
[ INFO] [1400513321.535040726, 34.219000000]: Robot is disabled
[ INFO] [1400513321.535125386, 34.220000000]: Gravity compensation was turned off
```

If you are using a simulator, enable the robot, untuck the robot (it sometimes doesn't work without untuck the robot first) and start the joint trajectory action servier, and of course run execute baxter.sh first:

```bash
./baxter.sh sim
rosrun baxter_tools enable_robot.py -e
rosrun baxter_tools tuck_arms.py -u
rosrun baxter_interface joint_trajectory_action_server.py
```

For both a real robot and a simulator, start our controller wrapper:
```bash
roslaunch control_wrapper baxter_control.launch
```

Other Baxter commands:

To disable the robot:

```bash
rosrun baxter_tools enable_robot.py -d
```

To untuck the robot:
```bash
rosrun baxter_tools tuck_arms.py -u
```

To tuck the robot:
```bash
rosrun baxter_tools tuck_arms.py -t
```

## Demo

To test the left arm or the right arm on a robot, rosrun either the `baxter_left_demo.py` or the `baxter_right_demo.py`

# Kuka arm

## Installation

The installation instructions can be found (ROS Indigo): <https://scazlab.github.io/kuka_setup_guide.html>

## Instruction

To work with a simulator, launch (This will handle everything for you, e.g., starting the simulator, start moveit and this control wrapper):

```bash
roslaunch control_wrapper sim_kuka_control.launch 
```

Currently the gripper doesn't work on simulator, but works with the real robot. I do not plan to fix this as this is minor.

To work with a real robot (it will launch the ros driver for you, so you can skip everything in the [kuka setup guide](https://scazlab.github.io/kuka_setup_guide.html), and just follow the instructions here):

```bash
sudo ldconfig /opt/ros/indigo/lib
roslaunch control_wrapper kuka_control.launch 
```

## Demo

Run:

```bash
rosrun control_wrapper kuka_demo.py
```
