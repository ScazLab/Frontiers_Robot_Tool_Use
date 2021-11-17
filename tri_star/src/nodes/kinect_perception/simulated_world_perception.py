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

import random
import threading
import numpy as np

import rospy

from std_msgs.msg import String

from control_wrapper.msg import SceneObjects

from tri_star import robot_util
from tri_star import constants
from tri_star import perception_util
from tri_star.msg import TargetPositions, TargetPosition
from tri_star import transformations as tfs

class SimulatedWorld:
    def __init__(self):
        self.get_data_lock = threading.Lock()
        
        self.position_noise = rospy.get_param("noise_position")
        self.quaternion_noise = rospy.get_param("noise_orientation")
        self.quaternion_noise = self.quaternion_noise * np.pi / 180.0
        
        self.tool_name = None
        self.task_name = None
        
        self.positions_pub = rospy.Publisher('/tri_star/visual_robot_frame', TargetPositions, queue_size=10)
        
        rospy.Subscriber("tri_star/tool", String, self.get_tool_name)
        rospy.Subscriber("tri_star/task", String, self.get_task_name)        
        rospy.Subscriber(robot_util.robot_topic('scene_objects'), SceneObjects, self.object_states)
    
    def get_tool_name(self, data):
        if len(data.data) != 0:
            self.tool_name = data.data
        else:
            self.tool_name = None
    
    def get_task_name(self, data):
        if len(data.data) != 0:
            self.task_name = data.data
        else:
            self.task_name = None
    
    def get_target_position_msg(self, pose, object_type):
        position_msg = TargetPosition()
        position_msg.frame = TargetPosition.FRAME_ROBOT
        position_msg.detect_type = TargetPosition.DETECT_TYPE_ARUCO
        position_msg.group_name = object_type
        position_msg.name = object_type
        
        if constants.get_perception_method() == constants.PERCEPTION_METHOD_ARUCO:
            position_msg.detect_type = TargetPosition.DETECT_TYPE_ARUCO
            if object_type == TargetPosition.NAME_TOOL:
                pose = self.get_tool_aruco_msg(pose)
            elif object_type == TargetPosition.NAME_GOAL:
                pose = self.get_goal_aruco_msg(pose)
        elif constants.get_perception_method() == constants.PERCEPTION_METHOD_POINTCLOUD:
            position_msg.detect_type = TargetPosition.DETECT_TYPE_POINTCLOUD
            pose = pose
        
        position_msg.position.x = pose.position.x + random.gauss(0.0, self.position_noise)
        position_msg.position.y = pose.position.y + random.gauss(0.0, self.position_noise)
        position_msg.position.z = pose.position.z + random.gauss(0.0, self.position_noise)
        
        quaternion = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        alpha, beta, gamma = tfs.euler_from_quaternion(quaternion)
        alpha += random.gauss(0.0, self.quaternion_noise)
        beta  += random.gauss(0.0, self.quaternion_noise)
        gamma += random.gauss(0.0, self.quaternion_noise)
        target_quaternion = tfs.quaternion_from_euler(alpha, beta, gamma)
        
        position_msg.quaternion.x = target_quaternion[0]
        position_msg.quaternion.y = target_quaternion[1]
        position_msg.quaternion.z = target_quaternion[2]
        position_msg.quaternion.w = target_quaternion[3]
        
        return position_msg

    def get_tool_aruco_msg(self, tool_pose_msg):
        Tworld_tool = robot_util.pose_msg_to_matrix(tool_pose_msg)
        Tworld_aruco = perception_util.toolpointcloud_to_toolaruco(Tworld_tool, self.tool_name)
        return robot_util.pose_matrix_to_msg(Tworld_aruco)
    
    def get_goal_aruco_msg(self, goal_pose_msg):
        Tworld_goal = robot_util.pose_msg_to_matrix(goal_pose_msg)
        Tworld_aruco = perception_util.goalpointcloud_to_goalaruco(Tworld_goal, self.task_name)
        return robot_util.pose_matrix_to_msg(Tworld_aruco)

    # publish the aruco pose, just like the real world perception
    def object_states(self, data): # perceived is the object pose
        self.get_data_lock.acquire()
        
        positions_msg = TargetPositions()
        positions_msg.header.frame_id = "robot_frame"
        positions_msg.positions = []
        
        for scene_object in data.objects:
            tool_position_msg = self.get_target_position_msg(scene_object.pose, scene_object.name)
            positions_msg.positions.append(tool_position_msg)
            
        self.positions_pub.publish(positions_msg)
        self.get_data_lock.release()

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        rospy.init_node('simulated_world_perception', anonymous=True)

        world = SimulatedWorld()
        world.run()

    except rospy.ROSInterruptException:
        pass
