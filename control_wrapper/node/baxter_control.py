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

import threading

import rospy
import rospkg
import struct

from std_msgs.msg import String, Bool
from control_wrapper.srv import Reset, ResetResponse
from control_wrapper.srv import SetPose, SetPoseResponse, SetPoseRequest
from control_wrapper.srv import SetJoints, SetJointsResponse
from control_wrapper.srv import SetTrajectory, SetTrajectoryResponse


from std_msgs.msg import (
    Header,
    Empty
)
from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest
)

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion
)

from kinematics import Kinematics

import baxter_interface

class AccomodateUR5E:
    def __init__(self):
        rospy.Subscriber("/baxter/control_wrapper/left/connect", Bool, self.connect_to_robot)
        rospy.Subscriber("/baxter/control_wrapper/left/enable_freedrive", Bool, self.enable)
        rospy.Subscriber("/baxter/control_wrapper/right/connect", Bool, self.connect_to_robot)
        rospy.Subscriber("/baxter/control_wrapper/right/enable_freedrive", Bool, self.enable)        
    
    def connect_to_robot(self, data):
        pass
    
    def enable(self, data):
        pass

class Gripper:
    def __init__(self):
        self.left_gripper = baxter_interface.Gripper('left')
        self.right_gripper = baxter_interface.Gripper('right')
        
        rospy.Subscriber("/baxter/control_wrapper/left/gripper", Bool, self.left_control)
        rospy.Subscriber("/baxter/control_wrapper/right/gripper", Bool, self.right_control)

    def left_control(self, data):
        if data.data:
            self.left_gripper.open()
        else:
            self.left_gripper.close()
        rospy.sleep(0.1)
    
    def right_control(self, data):
        if data.data:
            self.right_gripper.open()
        else:
            self.right_gripper.close()
        rospy.sleep(0.1)

class Baxter_Kinematics(Kinematics):
    def __init__(self, side, group_name, joint_names, grasping_group):
        robot_name = "baxter"
        self.limb = baxter_interface.Limb(side)
        joint_state_topic = "/robot/joint_states"
        base_frame = "world"
        super(Baxter_Kinematics, self).__init__(robot_name, side, group_name, joint_names, grasping_group, base_frame, joint_state_topic)
        ns = "ExternalTools/" + side + "/PositionKinematicsNode/IKService"
        self.iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
    

    def ik_request(self, pose):
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        try:
            resp = self.iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False
        # Check if result valid, and type of seed ultimately used to get solution
        # convert rospy's string representation of uint8[]'s to int's
        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        limb_joints = {}
        if (resp_seeds[0] != resp.RESULT_INVALID):
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        else:
            rospy.logwarn("INVALID POSE - No Valid Joint Solution Found.")
            return False
        return limb_joints


    def set_pose(self, data):
        if type(data) == SetPoseRequest:
            pose = Pose()
            pose.position = data.request_pose.position
            pose.orientation = data.request_pose.orientation
        else:
            pose = data
        joints = self.ik_request(pose)
        return self.set_joints_helper(joints)

    def set_trajectory(self, data):
        for pose in data.trajectory:
            self.set_pose(pose)

    def set_joints_helper(self, joint_angles):
        if joint_angles:
            self.limb.move_to_joint_positions(joint_angles)
            return True
        else:
            rospy.logwarn("No Joint Angles provided for move_to_joint_positions. Staying put.")
            return False
        
    def reset(self, data):
        self.limb.move_to_neutral()
        return ResetResponse(True)

class Baxter_Left_Kinematics(Baxter_Kinematics):
    def __init__(self):
        side = "left"
        joint_names = ["left_s0", "left_s1", "left_e0", "left_e1", "left_w0", "left_w1", "left_w2"]
        group_name = "left_arm"
        grapsing_group = "left_ee"
        super(Baxter_Left_Kinematics, self).__init__(side, group_name, joint_names, grapsing_group)

class Baxter_Right_Kinematics(Baxter_Kinematics):
    def __init__(self):
        side = "right"
        joint_names = ["right_s0", "right_s1", "right_e0", "right_e1", "right_w0", "right_w1", "right_w2"]
        group_name = "right_arm"
        grapsing_group = "right_ee"
        super(Baxter_Right_Kinematics, self).__init__(side, group_name, joint_names, grapsing_group)

if __name__ == '__main__':
    try:
        rospy.init_node('baxter_control_wrapper', anonymous=True)
        
        useless = AccomodateUR5E()
        
        gripper_control = Gripper()

        left_kinematics = Baxter_Left_Kinematics()
        right_kinematics = Baxter_Right_Kinematics()

        left_arm_thread = threading.Thread(target = left_kinematics.run)
        right_arm_thread = threading.Thread(target = right_kinematics.run)

        left_arm_thread.start()
        right_arm_thread.start()

        left_arm_thread.join()
        right_arm_thread.join()
        
    except rospy.ROSInterruptException:
        pass
