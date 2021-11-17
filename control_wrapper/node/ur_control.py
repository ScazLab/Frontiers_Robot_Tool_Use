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

import rospy
import rospkg

from std_msgs.msg import String, Bool
from sensor_msgs.msg import JointState

import numpy as np

from kinematics import Kinematics
from control_wrapper.srv import Reset, ResetResponse
from control_wrapper.srv import SetJoints, SetJointsResponse

class ExternalControl:
    def __init__(self):
        self.is_simulator = rospy.get_param("sim")
        
        self.connect_pub = rospy.Publisher("/ur_hardware_interface/script_command", String, queue_size=10)
        self.external_control = self.get_external_control_command()
        
        rospy.Subscriber("/ur/control_wrapper/left/connect", Bool, self.connect_to_robot)
    
    def get_external_control_command(self):
        commands = ""
        
        tool_voltage = rospy.get_param("/ur_hardware_interface/tool_voltage")
        tool_parity = rospy.get_param("/ur_hardware_interface/tool_parity")
        tool_baud_rate = rospy.get_param("/ur_hardware_interface/tool_baud_rate")
        tool_stop_bits = rospy.get_param("/ur_hardware_interface/tool_stop_bits")
        tool_rx_idle_chars = rospy.get_param("/ur_hardware_interface/tool_rx_idle_chars")
        tool_tx_idle_chars = rospy.get_param("/ur_hardware_interface/tool_tx_idle_chars")
        
        mult_jointstate = 1000000
        servo_j_replace = "lookahead_time=0.03, gain=2000"
        local_ip = "192.168.1.114" # may needs to be changed if your local host is with a different ip
        reverse_port = 50001
        
        rospack = rospkg.RosPack()
        
        with open(rospack.get_path("control_wrapper") + "/resources/ur_externalcontrol_urscript.txt", "r") as command_file:
            commands = command_file.read()
        
        commands = commands.replace("{{TOOL_VOLTAGE}}", str(tool_voltage))
        commands = commands.replace("{{TOOL_PARITY}}", str(tool_parity))
        commands = commands.replace("{{TOOL_BAUD_RATE}}", str(tool_baud_rate))
        commands = commands.replace("{{TOOL_STOP_BITS}}", str(tool_stop_bits))
        commands = commands.replace("{{TOOL_RX_IDLE_CHARS}}", str(tool_rx_idle_chars))
        commands = commands.replace("{{TOOL_TX_IDLE_CHARS}}", str(tool_tx_idle_chars))
        
        commands = commands.replace("{{JOINT_STATE_REPLACE}}", str(mult_jointstate))
        commands = commands.replace("{{SERVO_J_REPLACE}}", str(servo_j_replace))
        commands = commands.replace("{{SERVER_IP_REPLACE}}", str(local_ip))
        commands = commands.replace("{{SERVER_PORT_REPLACE}}", str(reverse_port))
        
        return commands + "\n"
    
    def connect_to_robot(self, data):
        if not self.is_simulator and data.data:
            self.connect_pub.publish(self.external_control)

class FreeDrive:
    def __init__(self):
        self.is_simulator = rospy.get_param("sim")
        
        self.free_drive_pub = rospy.Publisher("/ur_hardware_interface/script_command", String, queue_size=10)
        self.connect_pub = rospy.Publisher("/ur/control_wrapper/left/connect", Bool, queue_size=10)
        
        rospy.Subscriber("/ur/control_wrapper/left/enable_freedrive", Bool, self.enable)
        
    def enable(self, data):
        #self.connect_pub.publish(False)
        if not self.is_simulator:
            if data.data:
                self.free_drive_pub.publish('def myProg():\n\twhile (True):\n\t\tfreedrive_mode()\n\t\tsync()\n\tend\nend\n')
            else:
                self.free_drive_pub.publish('def myProg():\n\twhile (True):\n\t\tend_freedrive_mode()\n\t\tsleep(0.5)\n\tend\nend\n')
                rospy.sleep(0.1)
                self.connect_pub.publish(True)

class Gripper:
    def __init__(self):
        # to update?? open/close with https://github.com/ctaipuj/luc_control/blob/master/robotiq_85_control/src/gripper_ur_control.cpp

        self.is_simulator = rospy.get_param("sim")

        self.gripper_pub = rospy.Publisher("/ur_hardware_interface/script_command", String, queue_size=10)
        self.connect_pub = rospy.Publisher("/ur_hardware_interface/connect", String, queue_size=10)
        self.gripper_commands = self.get_gripper_command()
        self.command = "{{GRIPPER_COMMAND}}"

        self.activate_gripper()

        rospy.Subscriber("/ur/control_wrapper/left/gripper", Bool, self.control)

    def get_gripper_command(self):
        commands = ""

        rospack = rospkg.RosPack()

        with open(rospack.get_path("control_wrapper") + "/resources/ur_gripper.script", "r") as command_file:
            commands = command_file.read()

        return commands + "\n"

    def activate_gripper(self):
        if not self.is_simulator:
            command = self.gripper_commands.replace(self.command, "rq_activate_and_wait()")
            self.gripper_pub.publish(command)
            rospy.sleep(0.1)
            self.connect_pub.publish(True)

    def control(self, data):
        if not self.is_simulator:
            if data.data:
                self.open_gripper()
            else:
                self.close_gripper()
            rospy.sleep(0.1)
            self.connect_pub.publish(True)

    def open_gripper(self):
        command = self.gripper_commands.replace(self.command, "rq_open()")
        self.gripper_pub.publish(command)

    def close_gripper(self):
        command = self.gripper_commands.replace(self.command, "rq_close()")
        self.gripper_pub.publish(command)

    def deactivate_gripper(self):
        if not self.is_simulator:
            command = self.gripper_commands.replace(self.command, "")
            self.gripper_pub.publish(command)
            rospy.sleep(0.1)
            self.connect_pub.publish(True) 

class UR5e_Kinematics(Kinematics):
    def __init__(self):
        robot_name = "ur"
        side = "left"
        group_name = "manipulator"
        grasping_group = 'endeffector'
        base_frame = "world"
        joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']        
        super(UR5e_Kinematics, self).__init__(robot_name, side, group_name, joint_names, grasping_group, base_frame)
    
    def reset(self, data):
        rospy.wait_for_service("/ur/control_wrapper/left/set_joints")
        set_joints = rospy.ServiceProxy("/ur/control_wrapper/left/set_joints", SetJoints)
        joints = JointState()
        joints.name = ["elbow_joint", "shoulder_lift_joint", "shoulder_pan_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        joints.position = [-np.pi / 3.0, -2.0 * np.pi / 3, 0.0, np.pi * 2.0 / 3.0, -np.pi / 2.0, 0.0]
        is_reached = False
        try:
            is_reached = set_joints(joints).is_reached
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc)
        return ResetResponse(is_reached)

if __name__ == '__main__':
    try:
        rospy.init_node('ur_control_wrapper', anonymous=True)
        
        external_control = ExternalControl()
        
        free_drive = FreeDrive()
        
        gripper_control = Gripper()

        kinematics = UR5e_Kinematics()

        kinematics.run()
        
        gripper_control.deactivate_gripper()
        
    except rospy.ROSInterruptException:
        pass