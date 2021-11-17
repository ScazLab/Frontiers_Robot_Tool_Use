#!/usr/bin/env python

# Software License Agreement (MIT License)
#
# Copyright (c) 2020, control_warpper
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
from sensor_msgs.msg import JointState

from control_wrapper.srv import SetJoints

from demo import Demo

class URDemo(Demo):
    def __init__(self, robot, side):
        super(URDemo, self).__init__(robot, side)
    
    def set_goal_default_postion(self):
        rospy.wait_for_service(self.topic + "set_joints")
        set_joints = rospy.ServiceProxy(self.topic + "set_joints", SetJoints) 
        
        try:
            joints = JointState()
            joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
            joint_angles = np.deg2rad([-175.0, -118.0, -55.0, 115., -93,  -6.]).tolist()                  
            joints.name = joint_names
            joints.position = joint_angles
            is_reached = set_joints(joints).is_reached

        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc)

if __name__ == '__main__':
    try:
        rospy.init_node('ur_control_wrapper_demo_mode', anonymous=True)

        demo = URDemo("ur", "left")

        demo.run()
    except rospy.ROSInterruptException:
        pass