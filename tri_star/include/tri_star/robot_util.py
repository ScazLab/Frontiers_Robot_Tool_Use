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

import numpy as np

import rospy

from std_msgs.msg import Bool
from geometry_msgs.msg import Point, Quaternion, Pose

from sensor_msgs.msg import JointState

from control_wrapper.srv import SetPose
from control_wrapper.srv import GetPose
from control_wrapper.srv import GetJoints
from control_wrapper.srv import SetJoints
from control_wrapper.srv import Reset
from control_wrapper.srv import SetTrajectory
from control_wrapper.srv import CheckPose

from tri_star import transformation_util
from tri_star import constants

def pose_matrix_to_msg(matrix):
    translation, rotation = transformation_util.decompose_homogeneous_transformation_matrix(matrix)

    position_msg = Point(translation[0], translation[1], translation[2])
    quaternion_msg = Quaternion(rotation[0], rotation[1], rotation[2], rotation[3])
    
    return Pose(position_msg, quaternion_msg)

def pose_msg_to_matrix(pose_msg):
    position = np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
    quaternion = np.array([pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w])
    return transformation_util.get_homogeneous_transformation_matrix_from_quaternion(quaternion, position)

def robot_topic(topic):
    robot = constants.get_robot_platform()
    side = constants.get_robot_platform_side()
    
    return "/{}/control_wrapper/{}/{}".format(robot, side, topic)

class Robot(object):
    def __init__(self):
        self.free_drive_pub = rospy.Publisher(robot_topic("enable_freedrive"), Bool, queue_size=10)
        self.connect_pub    = rospy.Publisher(robot_topic("connect"),          Bool, queue_size=10)
        self.gripper_pub    = rospy.Publisher(robot_topic("gripper"),          Bool, queue_size=10)        
        
    def get_robot_pose(self):
        service_topic = robot_topic("get_pose")
        rospy.wait_for_service(service_topic)
        get_current_pose = rospy.ServiceProxy(service_topic, GetPose)
        current_pose = None
        try:
            current_pose = pose_msg_to_matrix(get_current_pose().pose)
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc) 
        return current_pose

    def get_robot_angle(self):
        service_topic = robot_topic("get_joints")
        rospy.wait_for_service(service_topic)
        get_current_joints = rospy.ServiceProxy(service_topic, GetJoints)
        current_joints = None
        try:
            current_joints = get_current_joints().joints
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc) 
        return current_joints
    
    def set_robot_angle(self, joint_names, joint_angles): # angles is in degree
        is_reached = False

        try:
            service_topic = robot_topic("set_joints")
            rospy.wait_for_service(service_topic)
            set_current_joints = rospy.ServiceProxy(service_topic, SetJoints)
                    
            joints = JointState()
            joints.name = joint_names
            joints.position = [np.deg2rad(angle) for angle in joint_angles]
            is_reached = set_current_joints(joints).is_reached
           
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc)
        
        return is_reached

    def set_robot_pose(self, pose_matrix, ask_before_move=True):
        if ask_before_move:
            raw_input("press any key to execute the trajectory")
        
        service_topic = robot_topic("set_pose")
        rospy.wait_for_service(service_topic)
        set_current_pose = rospy.ServiceProxy(service_topic, SetPose)
        is_reached = False
        
        try:
            pose_msg = pose_matrix_to_msg(pose_matrix)
            response = set_current_pose(pose_msg)
            pose = response.response_pose
            is_reached = response.is_reached
            if is_reached:
                print "[robot_util][set_robot_pose] robot reach pose SUCCEEDED"
            else:
                print "[robot_util][set_robot_pose] robot reach pose FAILED"
            print "[robot_util][set_robot_pose] pose to set"
            print pose_matrix
            print "[robot_util][set_robot_pose] pose reached"
            print self.get_robot_pose()
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc)
        
        return is_reached

    def execute_trajectory(self, ee_trajectory, ask_before_move=True):
        if ask_before_move:
            raw_input("press any key to execute trajectory...")
        
        service_topic = robot_topic("follow_trajectory")
        rospy.wait_for_service(service_topic)
        follow_trajectory = rospy.ServiceProxy(service_topic, SetTrajectory)
        is_reached = False
        
        try:
            poses = [pose_matrix_to_msg(i) for i in ee_trajectory]
            response = follow_trajectory(poses)
            current_pose = response.final_pose
            is_reached = response.is_reached
            print "is_reached: ", is_reached
            print "result pose (follow_trajectory)"
            print current_pose
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc)
        
        return is_reached

    # move robot out of work boundary
    def set_perceive_goal_pose(self, ask_before_move=True):
        joint_names = []
        joint_angles = [] # in degrees
        if constants.get_robot_platform() == constants.ROBOT_PLATFORM_BAXTER:
            joint_names = []
            joint_angles = []
        elif constants.get_robot_platform() == constants.ROBOT_PLATFORM_UR5E:
            print "MOVING TO PRE GOAL LOCATION"
            joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
            joint_angles = [-175.0, -118.0, -55.0, 115., -93,  -6.]
        elif constants.get_robot_platform() == constants.ROBOT_PLATFORM_KUKA:
            joint_names = []
            joint_angles = []            
        self.set_robot_angle(joint_names, joint_angles)

    def reset_robot(self, ask_before_move=True):
        if constants.is_simulator():
            if constants.get_robot_platform() == constants.ROBOT_PLATFORM_BAXTER:
                return True
            
        if ask_before_move:
            raw_input("press any key to reset the robot...")

        service_topic = robot_topic("reset")
        rospy.wait_for_service(service_topic)
        reset = rospy.ServiceProxy(service_topic, Reset)
        is_reached = False
        
        try:
            is_reached = reset().is_reached
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc)
        
        return is_reached
    
    def free_drive_robot(self, enable):
        self.free_drive_pub.publish(enable)

    def connect_robot(self, enable):
        rospy.sleep(0.5)
        self.connect_pub.publish(enable)
        rospy.sleep(0.5)
        
    def gripper_robot(self, enable, ask_before_move=True):
        if ask_before_move:
            if enable:
                raw_input("press any key to open the gripper")
            else:
                raw_input("press any key to close the gripper")
        self.gripper_pub.publish(enable)
        rospy.sleep(2.0)

    def check_pose(self, pose):
        service_topic = robot_topic("check_pose")
        rospy.wait_for_service(service_topic)
        check_pose = rospy.ServiceProxy(service_topic, CheckPose)
        could_reach = False
        joint_changes = np.inf
        try:
            response = check_pose(pose_matrix_to_msg(pose))
            could_reach = response.could_reach
            joint_changes = response.joint_changes
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc) 
        return could_reach, joint_changes