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
import rospkg

from std_srvs.srv import Empty
from std_msgs.msg import String, Bool
from brics_actuator.msg import JointPositions
from brics_actuator.msg import JointValue
from control_wrapper.srv import Reset, ResetResponse
from control_wrapper.srv import SetJoints, SetJointsResponse
from sensor_msgs.msg import JointState

from kinematics import Kinematics

class AccomodateUR5E:
    def __init__(self):
        rospy.Subscriber("/kuka/control_wrapper/left/connect", Bool, self.connect_to_robot)
        rospy.Subscriber("/kuka/control_wrapper/left/enable_freedrive", Bool, self.enable)       
    
    def connect_to_robot(self, data):
        pass
    
    def enable(self, data):
        pass

class Gripper:
    def __init__(self):
        rospy.Subscriber("/kuka/control_wrapper/left/gripper", Bool, self.control)
        self.gripper_pub = rospy.Publisher("arm_1/gripper_controller/position_command", JointPositions, queue_size=10)

    def control(self, data):
        finger_l = JointValue()
        finger_r = JointValue()
        
        finger_l.timeStamp = rospy.get_rostime()
        finger_r.timeStamp = rospy.get_rostime()
        
        finger_l.unit = "m"
        finger_r.unit = "m"
        
        finger_l.joint_uri = "gripper_finger_joint_l"
        finger_r.joint_uri = "gripper_finger_joint_r"
        
        if data.data:
            finger_l.value = 0.011
            finger_r.value = 0.011
        else:
            finger_l.value = 0.0
            finger_r.value = 0.0
        
        gripper_msg = JointPositions()
        gripper_msg.positions = [finger_l, finger_r]
        
        self.gripper_pub.publish(gripper_msg)
        
        rospy.sleep(0.1)

class FreeDrive:
    def __init__(self):
        #self.is_simulator = rospy.get_param("sim")
        self.is_simulator = False
        
        rospy.Subscriber("/kuka/control_wrapper/left/enable_freedrive", Bool, self.enable)
        
    def enable(self, data):
        #self.connect_pub.publish(False)
        if not self.is_simulator:
            service_topic = ""
            if data.data:
                service_topic = "arm_1/switchOffMotors"
            else:
                service_topic = "arm_1/switchOnMotors"
            
            rospy.wait_for_service(service_topic)
            free_drive_service = rospy.ServiceProxy(service_topic, Empty)
            try:
                free_drive_service()
            except rospy.ServiceException as exc:
                print "Service did not process request: " + str(exc)
                

class Kuka_Kinematics(Kinematics):
    def __init__(self):
        robot_name = "kuka"
        side = "left"
        group_name = "arm_1"
        grasping_grup = "arm_1_gripper"
        base_frame = "base_link"
        joint_names = ["arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5"]
        #goal_tolarance = (0.0001, 0.001, 0.001)
        goal_tolarance = (0.0001, 0.01, 0.05)
        super(Kuka_Kinematics, self).__init__(robot_name, side, group_name, joint_names, grasping_grup, base_frame, goal_tolarance=goal_tolarance)
    
    def reset(self, data):
        # candle position
        rospy.wait_for_service("/kuka/control_wrapper/left/set_joints")
        set_joints = rospy.ServiceProxy("/kuka/control_wrapper/left/set_joints", SetJoints)
        joints = JointState()
        joints.name = ["arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5"]
        joints.position = [np.deg2rad(260), np.deg2rad(50), np.deg2rad(-100), np.deg2rad(165), np.deg2rad(45)]
        is_reached = False
        try:
            is_reached = set_joints(joints).is_reached
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc)
        return ResetResponse(is_reached)

if __name__ == '__main__':
    try:
        rospy.init_node('kuka_control_wrapper', anonymous=True)
        
        useless = AccomodateUR5E()
        
        gripper_control = Gripper()
        
        free_drive = FreeDrive()

        kuka_kinematics = Kuka_Kinematics()

        kuka_kinematics.run()
        
    except rospy.ROSInterruptException:
        pass