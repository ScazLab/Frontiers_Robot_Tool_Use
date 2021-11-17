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
import threading
import Queue
import time
import json

import copy
import open3d as o3d
import numpy as np

import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import String

from tri_star import pointcloud_util
from tri_star import robot_util
from tri_star import transformation_util
from tri_star import constants

from tri_star.srv import CurrentPointCloud, CurrentPointCloudResponse
from tri_star.srv import ObjectPose, ObjectPoseResponse

from tri_star.msg import TargetPosition

MASTER = 0
SUB = 1

class ObjectMonitor(object):
    def __init__(self):
        self.robot = robot_util.Robot()
        
        self.task_name = None
        self.tool_name = None
        
        self.task_name = None
        self.tool_name = None
        self.goal_name = None

        self.goal_backgrounds = []
        
        rospy.Subscriber("tri_star/task", String, self.get_task_name)
        rospy.Subscriber("tri_star/tool", String, self.get_tool_name)
        rospy.Subscriber("tri_star/goal", String, self.get_goal_name)
        
        rospy.Service('/tri_star/object_pose', ObjectPose, self.get_pose)

    def get_tool_name(self, data):
        if len(data.data) != 0:
            self.tool_name = data.data
        else:
            self.tool_name = None
    
    def get_goal_name(self, data):
        if len(data.data) != 0:
            self.goal_name = data.data
        else:
            self.goal_name = None
    
    def get_task_name(self, data):
        if len(data.data) != 0:
            if self.task_name != data.data:
                self.target_background = None # reset the target background.
            self.task_name = data.data
        else:
            self.task_name = None
    
    def get_tool_bbs(self): 
        """
        Get the bounding box of the tool relative to the robot end-effector
        """
        Tee = np.identity(4)
        background_pcs = []
        foreground_pcs = []

        get_tool_pcs = False
        Tee, tool_bbs, tool_pcs, Tworld_tool = None, None, None, None
        while not get_tool_pcs:
            print "Position tool to get initial bounding box."
            raw_input("press any key to start free drive...")
            self.robot.free_drive_robot(True)
            raw_input("press any key to end free drive...")
            self.robot.free_drive_robot(False)

            # Get background with no tool
            self.robot.gripper_robot(True, ask_before_move=False)
            raw_input("make sure the robot is NOT holding anything, press any key to get the tool background...")
            self.robot.gripper_robot(False, ask_before_move=False)
            self.robot.reset_robot()
            rospy.sleep(1.)
            Tee = self.robot.get_robot_pose()
            # Process pc to remove uneeded stuff like gripper.
            background_pcs = pointcloud_util.get_current_point_cloud()

            print "Place tool in endeffector.do not move the Tee"
            self.robot.gripper_robot(True, ask_before_move=True)
            self.robot.gripper_robot(False, ask_before_move=True)
            
            raw_input("press any key to get the tool foreground...")
            rospy.sleep(1.)
            foreground_pcs = pointcloud_util.get_current_point_cloud()

            Tee, tool_bbs, tool_pcs = pointcloud_util.get_tool_bbs(Tee, background_pcs, foreground_pcs)
            tool_standard = pointcloud_util.get_tool_mesh(self.tool_name)
            Tworld_tool = pointcloud_util._align_pc(tool_pcs, tool_standard, visualize=True)

            while not get_tool_pcs in ['y', 'n']:
                get_tool_pcs = raw_input("keep this example? (y/n)")

            get_tool_pcs == 'y'

        return (Tee, tool_bbs, tool_pcs), Tworld_tool

    def get_tool_pose(self, Tstart):
        bb_info, Tworld_tool = self.get_tool_bbs()
        Tee, tool_bbs, tool_pcs = bb_info

        Tee_tools = []
        if not Tworld_tool is None:
            Tee_tools.append(np.matmul(transformation_util.get_transformation_matrix_inverse(Tee), Tworld_tool))
        
        is_finish = False
        while not is_finish:
            is_finish = raw_input("finish getting the tool samples? (y/n): ") == "y"
            if not is_finish:
                keep_current_sample = False
                while not keep_current_sample:
                    raw_input("press any key to start free drive...")
                    self.robot.free_drive_robot(True)
                    raw_input("press any key to end free drive...")
                    self.robot.free_drive_robot(False)
                    
                    rospy.sleep(1.)
                    current_Tee = self.robot.get_robot_pose()
                    current_pcs = pointcloud_util.get_current_point_cloud()

                    current_Ttool = pointcloud_util._get_tool_pose(Tee, tool_bbs, self.robot.get_robot_pose(), current_pcs, self.tool_name, visualize=True)
                    
                    keep_current_sample = raw_input("keep the current tool sample? (y/n): ") == "y"
                    if keep_current_sample:
                        if not current_Ttool is None:
                            Tee_tools.append(np.matmul(transformation_util.get_transformation_matrix_inverse(current_Tee), current_Ttool))                        

        return pointcloud_util.get_main_Tee_tool(Tee_tools)
        
    def get_goal_pose(self, Tstart=np.identity(4)):
        """
        @Tstart identity matrix for tool use. For goal start Tstart is identity. for goalend its goalstart
        """
        print "Now get Tworld_goalstart"

        Tworld_goal = np.identity(4)
        self.robot.connect_robot(True)
        self.robot.reset_robot()

        self.robot.set_perceive_goal_pose()
        raw_input("get the background. DO NOT put the goal in the workspace yet.")
        self.goal_backgrounds = pointcloud_util.get_current_point_cloud()

        get_good_sample = False
        while not get_good_sample:
            self.robot.set_perceive_goal_pose()
            raw_input("put the goal in the right position. do not move the robot")
            goal_foregrounds = pointcloud_util.get_current_point_cloud()

            Tworld_goal = pointcloud_util.get_goal_pose(self.goal_name, self.goal_backgrounds, goal_foregrounds, threshold=.008, Tstart=Tstart, visualize=True)

            get_good_sample = raw_input("keep the current goal sample? (y/n): ") == "y"

        Tworld_goal = copy.deepcopy(Tworld_goal)

        return Tworld_goal
    
    def get_pose(self, data):
        object_type = data.name
        T_start = np.array([data.Tstart]).reshape((4, 4))
        object_name = ""
        
        pose = Pose()
        found = True       
        if object_type == TargetPosition.NAME_GOAL:
            pose = robot_util.pose_matrix_to_msg(self.get_goal_pose(T_start))
        else:
            pose = robot_util.pose_matrix_to_msg(self.get_tool_pose(T_start))
        
        return ObjectPoseResponse(pose, found)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        rospy.init_node('object_monitor', anonymous=True)

        monitor = ObjectMonitor()

        monitor.run()

    except rospy.ROSInterruptException:
        pass
