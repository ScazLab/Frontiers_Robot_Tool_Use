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

import threading

import numpy as np

import rospy

from std_msgs.msg import Header
#from aruco_msgs.msg import MarkerArray
from geometry_msgs.msg import Point, Quaternion, PoseStamped

from tri_star.msg import TargetPosition, TargetPositions

from tri_star import transformation_util
from tri_star import transformations as tfs

class PoseCentroid:
    def __init__(self, pose, centroid):
        self.pose = pose
        self.centroid = centroid

class Perception:
    def __init__(self):
        self.accept_data_lock = threading.Lock()

        self.positions_pub = rospy.Publisher('/tri_star/aruco_3d_perceptions', TargetPositions, queue_size=10)
        self.msg_seq = 0

        self.usbcam_world_goal_pose = None
        self.usbcam_world_tool_pose = None
        
        rospy.Subscriber('/aruco_single_usbcam_world_goal/pose', PoseStamped, self.get_usbcam_world_goal_aruco)
        rospy.Subscriber('/aruco_single_usbcam_world_tool/pose', PoseStamped, self.get_usbcam_world_tool_aruco)

    def rotate_quaternion_default(self, quaternion, rotation):
        default_quaternion = tfs.quaternion_multiply(transformation_util.get_quaternion_from_rotation_matrix(rotation), np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w]))
        return Quaternion(default_quaternion[0], default_quaternion[1], default_quaternion[2], default_quaternion[3])

    def get_usbcam_world_goal_aruco(self, data):
        self.usbcam_world_goal_pose = data.pose
    
    def get_usbcam_world_tool_aruco(self, data):
        self.usbcam_world_tool_pose = data.pose    

    def get_position_msg(self, msgs, pose, frame, name):
        if pose is not None:
            position_msg = TargetPosition()
            position_msg.frame = frame
            position_msg.detect_type = TargetPosition.DETECT_TYPE_ARUCO
            position_msg.group_name = name
            position_msg.name = name

            position_msg.position = Point()
            position_msg.position.x = pose.position.x
            position_msg.position.y = pose.position.y
            position_msg.position.z = pose.position.z

            position_msg.quaternion = Quaternion()
            position_msg.quaternion.x = pose.orientation.x
            position_msg.quaternion.y = pose.orientation.y
            position_msg.quaternion.z = pose.orientation.z
            position_msg.quaternion.w = pose.orientation.w

            msgs.append(position_msg)

    def publish_msg(self):
        positions_msg = TargetPositions()
        positions_msg.header = Header()
        positions_msg.header.seq = self.msg_seq
        positions_msg.header.stamp = rospy.Time.now()
        positions_msg.header.frame_id = "camera_frame"
        self.msg_seq += 1
        positions_msg.positions = []

        self.get_position_msg(positions_msg.positions, self.usbcam_world_goal_pose, TargetPosition.FRAME_WORLD, "goal")
        self.get_position_msg(positions_msg.positions, self.usbcam_world_tool_pose, TargetPosition.FRAME_WORLD, "tool")

        self.positions_pub.publish(positions_msg)

        self.usbcam_world_goal_pose = None
        self.usbcam_world_tool_pose = None

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.publish_msg()
            rate.sleep()

if __name__ == '__main__':
    try:
        rospy.init_node('aruco_3d_perception', anonymous=True)

        perception = Perception()

        perception.run()

    except rospy.ROSInterruptException:
        pass

