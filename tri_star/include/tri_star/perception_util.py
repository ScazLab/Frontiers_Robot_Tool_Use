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

import rospy

import numpy as np

from geometry_msgs.msg import Point, Quaternion, Pose, Vector3

from control_wrapper.srv import AddObject, AddObjectRequest
from control_wrapper.srv import RemoveObject
from control_wrapper.srv import AttachObject, AttachObjectRequest
from control_wrapper.srv import DetachObject
from control_wrapper.msg import SceneObjectAllowCollision

from tri_star import constants
from tri_star import transformation_util
from tri_star import pointcloud_util
from tri_star.robot_util import robot_topic, Robot
from tri_star.robot_util import pose_matrix_to_msg, pose_msg_to_matrix
from tri_star.msg import TargetPosition
from tri_star.srv import ObjectPose

# get the pose of the object with the name
def object_pose(name, Tstart=np.identity(4)):
    service_topic = "/tri_star/object_pose"
    rospy.wait_for_service(service_topic)
    get_object_pose = rospy.ServiceProxy(service_topic, ObjectPose)
    pose = None
    try:
        response = get_object_pose(name, list(Tstart.flatten()))
        if response.found:
            pose = pose_msg_to_matrix(response.pose)
    except rospy.ServiceException as exc:
        print "Service did not process request: " + str(exc) 
    return pose

# get the averaged pose to be more accurate.
# Especially with tools holding in baxter's hand, which is shaking all the time :( @@~~~~~
def get_averaged_object_pose(name, Tstart=np.identity(4)):
    matrices = []
    count = 0
    i = 0
    
    sample_repeatitions = constants.get_sampling_repeatition()
    
    while i < sample_repeatitions and count < 100:
        pose = object_pose(name, Tstart)
        if pose is not None:
            matrices.append(pose)
            i += 1
        count += 1
    
    matrix = transformation_util.average_transformation_matrix(matrices)
    
    return matrix

def get_Tworld_goalaruco(goal_name=None):# goal_name is not needed for aruco perception
    Tworld_goalaruco = None
    if constants.get_perception_method() == constants.PERCEPTION_METHOD_ARUCO:
        Tworld_goalaruco = get_averaged_object_pose(TargetPosition.NAME_GOAL)
    elif constants.get_perception_method() == constants.PERCEPTION_METHOD_POINTCLOUD:
        Tworld_goalpointcloud = get_Tworld_goalpointcloud(goal_name)
        Tworld_goalaruco = goalpointcloud_to_goalaruco(Tworld_goalpointcloud, goal_name)
    return Tworld_goalaruco

def get_Tworld_toolaruco(tool_name=None):# tool_name is not needed for aruco perception
    Tworld_toolaruco = None
    if constants.get_perception_method() == constants.PERCEPTION_METHOD_ARUCO:
        Tworld_toolaruco = get_averaged_object_pose(TargetPosition.NAME_TOOL)
    elif constants.get_perception_method() == constants.PERCEPTION_METHOD_POINTCLOUD:
        Tworld_toolpointcloud = get_Tworld_toolpointcloud(tool_name)
        Tworld_toolaruco = toolpointcloud_to_toolaruco(Tworld_toolpointcloud, tool_name)
    return Tworld_toolaruco

def get_Tworld_goalpointcloud(goal_name=None):# goal_name is not needed for point cloud perception
    Tworld_goalpointcloud = None
    if constants.get_perception_method() == constants.PERCEPTION_METHOD_ARUCO:
        Tworld_goalaruco = get_Tworld_goalaruco(goal_name)
        Tworld_goalpointcloud = goalaruco_to_goalpointcloud(Tworld_goalaruco, goal_name)
    elif constants.get_perception_method() == constants.PERCEPTION_METHOD_POINTCLOUD:
        Tworld_goalpointcloud = object_pose(TargetPosition.NAME_GOAL)
    return Tworld_goalpointcloud

def get_Tworld_toolpointcloud(tool_name=None): # tool_name is not needed for point cloud perception
    Tworld_toolpointcloud = None
    if constants.get_perception_method() == constants.PERCEPTION_METHOD_ARUCO:
        Tworld_toolaruco = Tworld_get_Tworld_toolaruco(tool_name)
        Tworld_toolpointcloud = toolaruco_to_toolpointcloud(Tworld_toolaruco, tool_name)
    elif constants.get_perception_method() == constants.PERCEPTION_METHOD_POINTCLOUD:
        Tworld_toolpointcloud = get_averaged_object_pose(TargetPosition.NAME_TOOL)
    return Tworld_toolpointcloud

def get_Tworld_goal(Tstart=np.identity(4)):
    return object_pose(TargetPosition.NAME_GOAL, Tstart)

def get_Tee_tool(Tstart=np.identity(4)):
    return object_pose(TargetPosition.NAME_TOOL, Tstart)

"""
aruco<->object transformation
"""
def get_Ttool_aruco(tool_name):
    Ttool_aruco = np.identity(4)
    try:
        tool_aruco_corners = constants.get_tool_aruco_corners(tool_name)
        Ttool_aruco = transformation_util.get_Tobject_aruco(tool_aruco_corners)
    except Exception:
        pass
    
    return Ttool_aruco

def toolaruco_to_toolpointcloud(Tworld_aruco, tool_name):
    Ttool_aruco = get_Ttool_aruco(tool_name)
    Tworld_tool = np.matmul(Tworld_aruco, transformation_util.get_transformation_matrix_inverse(Ttool_aruco))
    
    return Tworld_tool

def toolpointcloud_to_toolaruco(Tworld_tool, tool_name):
    Ttool_aruco = get_Ttool_aruco(tool_name)
    Tworld_aruco = np.matmul(Tworld_tool, Ttool_aruco)
    
    return Tworld_aruco

# For fake model (not scanned, generated ply), push task only
def get_Tgoal_aruco(goal_name):
    Tgoal_aruco = np.identity(4)
    
    try:
        goal_aruco_corners = constants.get_goal_aruco_corners(goal_name)
        Tgoal_aruco = transformation_util.get_Tobject_aruco(goal_aruco_corners)
    except Exception:
        pass    

    return Tgoal_aruco

def goalaruco_to_goalpointcloud(Tworld_aruco, goal_name):
    Tgoal_aruco = get_Tgoal_aruco(goal_name)
    Tworld_goal = np.matmul(Tworld_aruco, transformation_util.get_transformation_matrix_inverse(Tgoal_aruco))
    return Tworld_goal

def goalpointcloud_to_goalaruco(Tworld_goal, goal_name):
    Tgoal_aruco = get_Tgoal_aruco(goal_name)
    Tworld_aruco = np.matmul(Tworld_goal, Tgoal_aruco)
    return Tworld_aruco

"""
scene related functions
"""
def goal_mesh_filename(goal):
    return constants.get_goal_mesh_path(goal)

def tool_mesh_filename(tool):
    return constants.get_tool_mesh_path(tool)

# if the perception method is aruco, pose is the aruco code T in the world frame
# if the perception method is pointcloud, pose is the point cloud T in the world frame
def add_goal(goal_name, Tworld_pose):
    name = TargetPosition.NAME_GOAL
    pose_msg = Pose()
    size = Vector3()
    Tworld_goal = np.identity(4)
    object_type = ""
    mesh_filename = ""
    
    if constants.get_perception_method() == constants.PERCEPTION_METHOD_ARUCO:
        Tworld_goal = goalaruco_to_goalpointcloud(Tworld_pose, goal_name)
    elif constants.get_perception_method() == constants.PERCEPTION_METHOD_POINTCLOUD:
        Tworld_goal = Tworld_pose
    
    pose_msg = pose_matrix_to_msg(Tworld_goal)
    
    mesh_filename = constants.get_goal_mesh_path(goal_name)
    object_type = AddObjectRequest.TYPE_MESH    
    
    return add_scene_object(name, pose_msg, size, mesh_filename, object_type)

def remove_goal():
    return remove_scene_object(TargetPosition.NAME_GOAL)

# if the perception method is aruco, pose is the aruco code T in the world frame
# if the perception method is pointcloud, pose is the point cloud T in the world frame
def attach_goal(Tworld_goal, goal_name):
    size = Vector3()
    name = TargetPosition.NAME_GOAL
    
    if constants.get_perception_method() == constants.PERCEPTION_METHOD_ARUCO:
        Tworld_goal = goalaruco_to_goalpointcloud(Tworld_goal, goal_name)
    elif constants.get_perception_method() == constants.PERCEPTION_METHOD_POINTCLOUD:
        Tworld_goal = Tworld_goal
        
    pose_msg = pose_matrix_to_msg(Tworld_goal)
    mesh_filename = constants.get_goal_mesh_path(goal_name)
    object_type = AttachObjectRequest.TYPE_MESH  
    
    return attach_scene_object(name, pose_msg, size, mesh_filename, object_type)

def detach_goal():
    return detach_scene_object(TargetPosition.NAME_GOAL, to_remove=False)

def disable_goal_collision():
    toggle_scene_object_collision(TargetPosition.NAME_GOAL, False)

def enable_goal_collision():
    toggle_scene_object_collision(TargetPosition.NAME_GOAL, True)

def default_grasping_gesture(tool):
    return np.identity(4)

# if the perception method is aruco, pose is the aruco code T in the world frame
# if the perception method is pointcloud, pose is the point cloud T in the world frame
def attach_tool(tool, Tworld_tool=None):
    name = TargetPosition.NAME_TOOL
    if Tworld_tool is None:
        Tee_tool = default_grasping_gesture(tool)
        Tworld_ee = Robot().get_robot_pose()
        Tworld_toolaruco = np.matmul(Tworld_ee, Tee_tool)
    
    print "attach tool Tworld_tool{}: ".format(constants.get_perception_method())
    print Tworld_tool
    
    if constants.get_perception_method() == constants.PERCEPTION_METHOD_ARUCO:
        Tworld_tool = toolaruco_to_toolpointcloud(Tworld_tool, tool)
    print "Tworld_tool"
    print Tworld_tool
    
    pose_msg = pose_matrix_to_msg(Tworld_tool)
    size = Vector3()
    mesh_filename = tool_mesh_filename(tool)
    object_type = AttachObjectRequest.TYPE_MESH
    
    return attach_scene_object(name, pose_msg, size, mesh_filename, object_type)

def detach_tool():
    return detach_scene_object(TargetPosition.NAME_TOOL)

def disable_tool_collision():
    toggle_scene_object_collision(TargetPosition.NAME_TOOL, False)

def enable_tool_collision():
    toggle_scene_object_collision(TargetPosition.NAME_TOOL, True)

def add_desk():
    name = "desk"
    if constants.get_robot_platform() == constants.ROBOT_PLATFORM_UR5E:
        pose = Pose(Point(-0.34, -0.0075, -0.023-0.023), Quaternion(0.0, 0.0, 0.0, 1.0))
        size = Vector3(0.912, 1.525, 0.045)
        mesh_filename = ""
        object_type = AddObjectRequest.TYPE_BOX
        
        return add_scene_object(name, pose, size, mesh_filename, object_type)

# pose is the object's pose, not the aruco pose
def add_scene_object(name, pose_msg, size, mesh_filename, object_type):
    service_topic = robot_topic("add_object")
    rospy.wait_for_service(service_topic)
    add = rospy.ServiceProxy(service_topic, AddObject)

    try:
        if constants.is_multiple_computers():
            mesh_filename = os.path.basename(mesh_filename)        
        response = add(name, pose_msg, size, mesh_filename, object_type).is_success
    except rospy.ServiceException as exc:
        print "Service did not process request: " + str(exc)
    
    return response

def toggle_scene_object_collision(scene_object_name, is_allow_collision):
    msg = SceneObjectAllowCollision(scene_object_name, is_allow_collision)
    collision_pub = rospy.Publisher(robot_topic('scene_allow_collision'), SceneObjectAllowCollision, queue_size=10)
    collision_pub.publish(msg)
    rospy.sleep(1.0)

def remove_scene_object(name):
    service_topic = robot_topic("remove_object")
    rospy.wait_for_service(service_topic)
    remove = rospy.ServiceProxy(service_topic, RemoveObject)
    
    try:
        response = remove(name).is_success
    except rospy.ServiceException as exc:
        print "Service did not process request: " + str(exc)
    
    return response

# pose is the poind cloud in robot end-effector frame
# Tee_tool
def attach_scene_object(name, pose_msg, size, mesh_filename, object_type):
    service_topic = robot_topic("attach_object")
    rospy.wait_for_service(service_topic)
    add = rospy.ServiceProxy(service_topic, AttachObject)
    
    try:
        if constants.is_multiple_computers():
            mesh_filename = os.path.basename(mesh_filename)        
        response = add(name, pose_msg, size, mesh_filename, object_type).is_success
    except rospy.ServiceException as exc:
        print "Service did not process request: " + str(exc)
    
    return response

def detach_scene_object(name, to_remove=True):
    service_topic = robot_topic("detach_object")
    rospy.wait_for_service(service_topic)
    detach = rospy.ServiceProxy(service_topic, DetachObject)    
    
    try:
        response = detach(name, to_remove).is_success
    except rospy.ServiceException as exc:
        print "Service did not process request: " + str(exc)
    
    return response
