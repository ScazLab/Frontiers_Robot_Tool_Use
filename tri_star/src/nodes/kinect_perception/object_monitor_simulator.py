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

from geometry_msgs.msg import Pose

from tri_star.msg import TargetPositions
from tri_star.srv import ObjectPose, ObjectPoseResponse

from tri_star import robot_util
from tri_star import transformation_util

class ObjectMonitor:
    def __init__(self):
        self.get_data_lock = threading.Lock()
        self.robot = robot_util.Robot()
        
        self.position = {}       
        
        rospy.Subscriber('/tri_star/visual_robot_frame', TargetPositions, self.get_position)
        rospy.Service('/tri_star/object_pose', ObjectPose, self.get_pose)

    def get_position(self, data):
        self.get_data_lock.acquire()
        
        progress = self.position.keys()
        
        for position in data.positions:
            name = position.group_name
            self.position[name] = Pose(position.position, position.quaternion)
            if name in progress:
                progress.remove(name)
        
        for name in progress:
            self.position[name] = None
        
        self.get_data_lock.release()
    
    def get_pose(self, data):
        name = data.name
        
        pose = Pose()
        found = False
        
        if name in self.position.keys():
            pose = self.position[name]
            found = True
        
        if name == "tool":
            Tworld_ee = self.robot.get_robot_pose()
            Tworld_tool = robot_util.pose_msg_to_matrix(pose)
            Tee_tool = np.matmul(transformation_util.get_transformation_matrix_inverse(Tworld_ee), Tworld_tool)
            pose = robot_util.pose_matrix_to_msg(Tee_tool)
        
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
