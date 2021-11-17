#!/usr/bin/env python

# Software License Agreement (MIT License)
#
# Copyright (c) 2020, tri_star
# All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Meiying Qin, Jake Brawer

import os
import glob
import numpy as np

import rospy

from tri_star.msg import TargetPosition

PERCEPTION_METHOD_ARUCO = "aruco"
PERCEPTION_METHOD_POINTCLOUD = "pointcloud"

CONDITION_SIMULATOR = "simulator"
CONDITION_REAL = "real_robot"

ROBOT_PLATFORM_KUKA = "kuka"
ROBOT_PLATFORM_UR5E = "ur"
ROBOT_PLATFORM_BAXTER = "baxter"

TOOL_TYPE_SOURCE     = "tool_source"
TOOL_TYPE_SUBSTITUTE = "tool_substitute"
GOAL_TYPE_SOURCE     = "goal_source"
GOAL_TYPE_SUBSTITUTE = "goal_substitute"

TASK_TYPE_POSE_CHANGE = "pose_change"
TASK_SUBTYPE_POSE_CHANGE_GENERAL = "pose_change_type_general" # e.g., push
TASK_SUBTYPE_POSE_CHANGE_SPECIFIC = "pose_change_type_specific"
TASK_TYPE_OTHER = "other"

DIRNAME_ARCHIVE = "archive_{}"

SYMMETRY_TOOL_FILE_NAME = "tools.json"
SYMMETRY_GOAL_FILE_NAME = "goals.json"

OBJECT_TYPE_GOAL = TargetPosition.NAME_GOAL
OBJECT_TYPE_TOOL = TargetPosition.NAME_TOOL

PC_MASTER_INDEX = 0
PC_SUB_INDEX = 1

STAR_1_TEE_TOOL = {}
STAR_1_TEE_TOOL["plunger"] = np.array([[-1., 0., 0., 0.],
                                       [0., -1., 0., 0.],
                                       [0., 0., 1., 0.],
                                       [0., 0., 0., 1.]])
STAR_1_TEE_TOOL["xylo_stick"] = np.array([[0., 0., 1., 0.],
                                          [0., 1., 0., 0.],
                                          [-1., 0., 0., 0.05],
                                          [0., 0., 0., 1.]])
STAR_1_TEE_TOOL["butcher_knife"] = np.array([[-1., 0., 0., 0.05],
                                             [0., 0., -1., 0.],
                                             [0., -1., 0., 0.],
                                             [0., 0., 0., 1.]])
STAR_1_TEE_TOOL["blue_scooper"] = np.array([[-1., 0., 0., 0.1],
                                            [0., 1., 0., 0.],
                                            [0., 0., -1., 0.02],
                                            [0., 0., 0., 1.]])
STAR_1_TEE_TOOL["small_blue_spatula"] = np.array([[0., 0., -1., 0.],
                                                  [0., 1., 0., 0.],
                                                  [1., 0., 0., 0.05],
                                                  [0., 0., 0., 1]])
STAR_1_TEE_TOOL["writing_brush"] = np.array([[0., 0., -1., 0.],
                                             [0., 1., 0., 0.],
                                             [1., 0., 0., 0.05],
                                             [0., 0., 0., 1]])
STAR_1_TEE_TOOL["gavel"] = np.array([[-1., 0., 0., 0.],
                                     [0., 0., -1., 0.],
                                     [0., -1., 0., 0.],
                                     [0., 0., 0., 1.]])  

# temp solution, copied directly from file_util
def get_filenames_in_dir(dir_path, ext=None):
    if ext:
        dir_path += "/*." + ext
    else:
        if not dir_path.endswith("/"):
            dir_path += "/*"
    return [os.path.basename(i) for i in glob.glob(dir_path)]

def get_azure_config_file_name(frame):
    return rospy.get_param("{}_config_file_name".format(frame))

def get_perception_method():
    return rospy.get_param("perception_method")

def get_package_dir():
    return rospy.get_param("package_dir")

def get_learned_data_dir(platform=None):
    if platform is None:
        platform = get_robot_platform()
    
    condition_name = ""
    if is_simulator():
        condition_name = CONDITION_SIMULATOR
    else:
        condition_name = CONDITION_REAL
    
    return os.path.join(rospy.get_param("learned_data_dir"), platform, condition_name, rospy.get_param("perception_method"))

def get_tool_mesh_path(tool_name):
    tool_mesh_path = os.path.join(tool_mesh_dir(), tool_name + ".ply")
    return tool_mesh_path

def get_goal_mesh_path(goal_name):
    goal_mesh_path = os.path.join(goal_mesh_dir(), goal_name + ".ply")
    return goal_mesh_path    

def tool_mesh_dir():
    return rospy.get_param("tool_mesh_dir")

def goal_mesh_dir():
    return rospy.get_param("goal_mesh_dir")

def symmetry_dir():
    return rospy.get_param("symmetry_dir")

def pointcloud_dir():
    return rospy.get_param("pointcloud_dir")

def pointcloud_tool_dir():
    return rospy.get_param("pointcloud_tool_dir")

def pointcloud_goal_dir():
    return rospy.get_param("pointcloud_goal_dir")

def pointcloud_raw_dir():
    return rospy.get_param("pointcloud_raw_dir")

def get_candidate_tools():
    tools = get_filenames_in_dir(tool_mesh_dir(), ext="ply")
    
    return [os.path.splitext(i)[0] for i in tools]

def get_candidate_goals():
    goals = get_filenames_in_dir(goal_mesh_dir(), ext="ply")
    
    return [os.path.splitext(i)[0] for i in goals]

def get_candidate_tasks():
    return rospy.get_param("candidate_tasks")

def is_simulator():
    return rospy.get_param("is_simulator")

def is_testing():
    return rospy.get_param("is_testing")

def is_multiple_computers():
    return rospy.get_param("is_multiple_computers")

def get_tool_aruco_corners(tool):
    points = np.array(rospy.get_param("{}_aruco_points".format(tool))).reshape((4, 3))
    
    return points

def get_goal_aruco_corners(goal):
    points = np.array(rospy.get_param("{}_aruco_points".format(goal))).reshape((4, 3))
    
    return points    

def get_sampling_repeatition():
    return rospy.get_param("perception_repeatition")

def get_robot_platform():
    return rospy.get_param("robot_platform_type")

def get_robot_platform_side():
    return rospy.get_param("robot_platform_side")

def get_Trobot_camera(frame_id):
    robot_usbcam_world_matrix = rospy.get_param("{}_cam_matrix".format(frame_id))
    Trobot_camera = np.array([robot_usbcam_world_matrix]).reshape((4, 4))
    return Trobot_camera

def get_work_space_boundary():
    min_boundary = rospy.get_param("workspace_min_boundary")
    max_boundary = rospy.get_param("workspace_max_boundary")
    return min_boundary, max_boundary

def get_tool_perception_work_space_boundary():
    min_boundary = rospy.get_param("tool_perception_workspace_min_boundary")
    max_boundary = rospy.get_param("tool_perception_workspace_max_boundary")
    return min_boundary, max_boundary
 
def get_tool_robot_boundary():
    return rospy.get_param("tool_robot_boundary")    

def get_scan_tool_perception_work_space_boundary():
    min_boundary = rospy.get_param("scan_tool_perception_workspace_min_boundary")
    max_boundary = rospy.get_param("scan_tool_perception_workspace_max_boundary")
    return min_boundary, max_boundary

def get_scan_tool_robot_boundary():
    return rospy.get_param("scan_tool_robot_boundary") 

def get_robot_joint_names():
    joint_names = rospy.get_param("joint_names")
    return  joint_names
