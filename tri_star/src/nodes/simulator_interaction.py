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

from std_msgs.msg import String

from tri_star.msg import TargetPosition
from tri_star import perception_util
from tri_star import robot_util
from tri_star import transformation_util
from tri_star import constants

def get_Tx(T, theta):
    theta = np.deg2rad(theta)
    T_x = np.array([[1., 0., 0., 0.],
                    [0., np.cos(theta), -np.sin(theta), 0.],
                    [0., np.sin(theta),  np.cos(theta), 0.],
                    [0., 0., 0., 1.]])
    return np.matmul(T_x, T)

def get_Ty(T, theta):
    theta = np.deg2rad(theta)
    T_y = np.array([[np.cos(theta), 0., np.sin(theta), 0.],
                    [0., 1., 0., 0.],
                    [-np.sin(theta), 0., np.cos(theta), 0.],
                    [0., 0., 0., 1.]])
    return np.matmul(T_y, T)

def get_Tz(T, theta):
    theta = np.deg2rad(theta)
    T_z = np.array([[np.cos(theta), -np.sin(theta), 0., 0.],
                    [np.sin(theta),  np.cos(theta), 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])
    return np.matmul(T_z, T)

class SimulatorInteraction(object):
    def __init__(self):
        self.tool_name = TargetPosition.NAME_TOOL
        self.goal_name = TargetPosition.NAME_GOAL
        self.task_name = None
        self.tool_type = None
        
        #self.task_name = "push"
        #self.tool_type = "plunger"
        #self.goal_type = "blue_puck"
        
        #self.task_name = "knock"
        #self.tool_type = "gavel"
        #self.goal_type = "gavel_target"
        
        #self.task_name = "knock" #
        #self.tool_type = "xylo_stick"
        #self.goal_type = "xylophone"
        
        #self.task_name = "knock"
        #self.tool_type = "percusive_stick"
        #self.goal_type = "drum"
        
        #self.task_name = "cut" # TODO
        #self.tool_type = "butcher_knife"
        #self.goal_type = "square_playdough"
        
        #self.task_name = "scoop" # TODO
        #self.tool_type = "blue_scooper"
        #self.goal_type = "duck"
        
        #self.task_name = "stir"
        #self.tool_type = "small_blue_spatula"
        #self.goal_type = "large_bowl"
        
        #self.task_name = "draw"
        #self.tool_type = "writing_brush"
        #self.goal_type = "buddha_board"
        
        #self.task_name = "screw"
        #self.tool_type = "paint_scraper"
        #self.goal_type = "screw"
        
        self.task_name = "stir"
        self.tool_type = "small_plastic_spoon"
        self.goal_type = "ramen_bowl"
        
        self.robot = robot_util.Robot()
        
        rospy.Subscriber("tri_star/task", String, self.get_task_name)
        rospy.Subscriber("tri_star/tool", String, self.get_tool_type)
        rospy.Subscriber("tri_star/goal", String, self.get_goal_type)
    
    def get_tool_type(self, data):
        if len(data.data) != 0:
            self.tool_type = data.data
        else:
            self.tool_type = None
            
    def get_goal_type(self, data):
        if len(data.data) != 0:
            self.goal_type = data.data
        else:
            self.goal_type = None    
    
    def get_task_name(self, data):
        if len(data.data) != 0:
            self.task_name = data.data
        else:
            self.task_name = None
    
    def get_pose(self):
        pose_matrix = self.robot.get_robot_pose()
        x, y, z, ax, ay, az = transformation_util.get_euler_angle_from_transformation_matrix(pose)
        print "x: {}\ny: {}\ny: {}\nz: {}\nrx: {}\nry: {}\nrz: {}\n".format(x, y, z, transformation_util.radian_to_degrees(ax), transformation_util.radian_to_degrees(ay), transformation_util.radian_to_degrees(az))
    
    def get_angle(self):
        print self.robot.get_robot_angle()
    
    def add_goal(self):
        self.remove_goal()
        
        print "add goal: {}".format(self.goal_type)

        desk_top = -0.023 + 0.045 / 2
        
        print "get the pose of the {} of the goal object. Unit is meter".format(constants.get_perception_method())
        x = float(raw_input("x: "))
        y = float(raw_input("y: "))
        
        robot_platform = constants.get_robot_platform()
        if robot_platform == constants.ROBOT_PLATFORM_UR5E:
            z = desk_top + 0.045
        else:
            z = float(raw_input("z: "))

        rx = transformation_util.degrees_to_radian(float(raw_input("rx: ")))
        ry = transformation_util.degrees_to_radian(float(raw_input("ry: ")))
        rz = transformation_util.degrees_to_radian(float(raw_input("rz: ")))
        
        Tworld_goal = transformation_util.get_transformation_matrix_from_euler_angle(x, y, z, rx, ry, rz)
        
        perception_util.add_goal(self.goal_type, Tworld_goal)
    
    def remove_goal(self):
        perception_util.remove_goal()
    
    def attach_goal(self):
        pose = perception_util.get_Tworld_goal()
        perception_util.attach_goal(pose, self.task_name)
        
    def detach_goal(self):
        perception_util.detach_goal()
    
    def toggle_goal_collision(self, enable_collision):
        if enable_collision:
            perception_util.enable_goal_collision()
        else:
            perception_util.disable_goal_collision()

    def attach_tool(self):
        #Tee_tool = np.identity(4)
        Tee_tool = {}
        robot_platform = constants.get_robot_platform()
        print "robot platform is: ", robot_platform
        
        if robot_platform == constants.ROBOT_PLATFORM_UR5E:
            if constants.get_perception_method() == constants.PERCEPTION_METHOD_ARUCO:
                Tee_tool["plunger_normalized"] = np.array([[0, 1, 0, 0.08],
                                                           [1, 0, 0, 0],
                                                           [0, 0, -1, 0.04],
                                                           [0, 0, 0, 1]])
            elif constants.get_perception_method() == constants.PERCEPTION_METHOD_POINTCLOUD:
                Tee_tool["plunger_normalized"] = np.array([[-1., 0., 0., 0.],
                                                           [0., 1., 0., 0.],
                                                           [0., 0., -1., 0.],
                                                           [0., 0., 0., 1.]])
                Tee_tool["plunger"] = np.array([[-1., 0., 0., 0.],
                                                [0., -1., 0., 0.],
                                                [0., 0., 1., 0.],
                                                [0., 0., 0., 1.]])
                #Tee_tool["xylo_stick"] = np.array([[0., 0., 1., 0.],
                                                   #[0., 1., 0., 0.],
                                                   #[-1., 0., 0., 0.05],
                                                   #[0., 0., 0., 1.]])
                Tee_tool["xylo_stick"] = np.array([[-0.74096603,  0.65540231,  0.146346,    0.00451146],
                                                   [-0.00930174  ,0.20788756 ,-0.9781085 ,  0.00348856],
                                                   [-0.67147809 ,-0.72610645, -0.14794124 , 0.03092808],
                                                   [ 0.          ,0.        ,  0.         , 1.        ]])
                theta = 0.
                Tx = np.array([[1., 0., 0., 0.],
                               [0., np.cos(theta), -np.sin(theta), 0.],
                               [0., np.sin(theta),  np.cos(theta), 0],
                               [0., 0., 0., 1]])
                ee_theta_z = np.pi / 3. # y
                Tee = np.array([[ np.cos(ee_theta_z), 0., np.sin(ee_theta_z), 0.],
                                [0., 1., 0., 0.],
                                [-np.sin(ee_theta_z), 0., np.cos(ee_theta_z), 0],
                                [0., 0., 0., 1]])
                Tee_tool["xylo_stick"] = np.matmul(Tee_tool["xylo_stick"], Tx)
                Tee_tool["xylo_stick"] = np.matmul(Tee, Tee_tool["xylo_stick"])
                Tee_tool["wooden_knife"] = np.array([[1., 0., 0., 0.05],
                                                     [0., 0., -1., 0.],
                                                     [0., 1., 0., 0.],
                                                     [0., 0., 0., 1.]])
                Tee_tool["butcher_knife"] = np.array([[-1., 0., 0., 0.05],
                                                     [0., 0., -1., 0.],
                                                     [0., -1., 0., 0.],
                                                     [0., 0., 0., 1.]])                
                Tee_tool["blue_scooper"] = np.array([[-1., 0., 0., 0.1],
                                                     [0., 1., 0., 0.],
                                                     [0., 0., -1., 0.02],
                                                     [0., 0., 0., 1.]])
                Tee_tool["small_blue_spatula"] = np.array([[0., 0., -1., 0.],
                                                           [0., 1., 0., 0.],
                                                           [1., 0., 0., 0.05],
                                                           [0., 0., 0., 1]])
                Tee_tool["writing_brush"] = np.array([[0., 0., -1., 0.],
                                                      [0., 1., 0., 0.],
                                                      [1., 0., 0., 0.05],
                                                      [0., 0., 0., 1]])
                #Tee_tool["percusive_stick"] = np.array([[-1.000000e+00,  0.000000e+00,  0.000000e+00,  8.000000e-02],
                                                        #[ 0.000000e+00, -1.000000e+00, -6.123234e-17,  0.000000e+00],
                                                        #[ 0.000000e+00, -6.123234e-17,  1.000000e+00,  3.000000e-02],
                                                        #[ 0.000000e+00,  0.000000e+00,  0.000000e+00,  1.000000e+00]])
                Tee_tool["percusive_stick"] = np.array([[-1.0000000e+00,  0.0000000e+00,  0.0000000e+00,  8.0000000e-02],
                                                        [ 0.0000000e+00,  1.0000000e+00,  1.8369702e-16,  0.0000000e+00],
                                                        [ 0.0000000e+00,  1.8369702e-16, -1.0000000e+00,  3.0000000e-02],
                                                        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]) # sample 1                
                Tee_tool["gavel"] = np.array([[-1., 0., 0., 0.],
                                              [0., 0., -1., 0.],
                                              [0., -1., 0., 0.],
                                              [0., 0., 0., 1.]])
                Tee_tool["paint_scraper"] = np.array([[0., 0., -1., 0.],
                                                      [0., 1., 0., 0.],
                                                      [1., 0., 0., 0.05],
                                                      [0., 0., 0., 1.]])
                Tee_tool["paint_scraper"] = get_Tz(Tee_tool["paint_scraper"], 90)
                Tee_tool["paint_scraper"] = get_Ty(Tee_tool["paint_scraper"], 25)
                Tee_tool["small_plastic_spoon"] = np.array([[0., 0., 1., 0.],
                                                            [0., 1., 0., 0.],
                                                            [-1., 0., 0., 0.05],
                                                            [0., 0., 0., 1]])                
                # stir, screw tasks
                #Tee_tool["plunger_normalized"] = np.array([[0., 0., 1., 0.],
                                                           #[0., 1., 0., 0.],
                                                           #[-1., 0., 0., 0.15],
                                                           #[0., 0., 0., 1.]])
        elif robot_platform == constants.ROBOT_PLATFORM_KUKA:
            Tee_tool = np.array([[-1, 0, 0, 0.0],
                                 [0, -1, 0, 0.0],
                                 [0, 0, 1, 0.08],
                                 [0, 0, 0, 1]])
        elif robot_platform == constants.ROBOT_PLATFORM_BAXTER:
            Tee_tool = np.array([[-1, 0, 0, -0.08],
                                 [0, -1, 0, 0],
                                 [0, 0, 1, 0.04],
                                 [0, 0, 0, 1]])
        
        print "Tee_{}_{}".format(self.tool_type, constants.get_perception_method())
        print Tee_tool[self.tool_type]
        Tworld_ee = self.robot.get_robot_pose()
        print "Tworld_ee"
        print Tworld_ee
        Tworld_tool = np.matmul(Tworld_ee, Tee_tool[self.tool_type])
        print "Tworld_tool{}".format(constants.get_perception_method())
        print Tworld_tool
        perception_util.attach_tool(self.tool_type, Tworld_tool) # todo: select more tools
    
    def detach_tool(self):
        perception_util.detach_tool()
    
    def move_arm(self, direction):
        x, y, z, rx, ry, rz = transformation_util.get_euler_angle_from_transformation_matrix(self.robot.get_robot_pose())
        
        if direction.startswith("x"):
            value = float(direction[1:])
            x += value
        elif direction.startswith("y"):
            value = float(direction[1:])
            y += value
        elif direction.startswith("z"):
            value = float(direction[1:])
            z += value
        
        value = transformation_util.degrees_to_radian(float(direction[2:]))
        if direction.startswith("rx"):
            rx += value
        elif direction.startswith("ry"):
            ry += value
        elif direction.startswith("rz"):
            rz += value
        
        pose = transformation_util.get_transformation_matrix_from_euler_angle(x, y, z, rx, ry, rz)
        
        self.robot.set_robot_pose(pose, ask_before_move=False)
    
    def run(self):
        while not rospy.is_shutdown():
            print "====================================================="
            command_input = raw_input("Goals:\n\tag(add goal);\n\trg(remove goal);\n\tatg(attach goal);\n\tdtg(detach goal);\n\tegc(enable goal collision);\n\tdgc(disable goal collision)\nTools:\n\tat(attach tool);\n\tdt(detach tool);\nMove arm:\n\tx+n(e.g., x+5.13, x direction move up 5.13 cm. x could be replaced by y or z. + could be replaced by -);\n\tSimilarly, rx+n(e.g., rotate n degree around x axis)\nGet Pose: gp\nGet Joint angles; ja.\n\nYou Choice: ")
            if command_input == "gp":
                self.get_pose()
            elif command_input == "ja":
                self.get_angle()
            elif command_input == "ag":
                self.add_goal()
            elif command_input == "rg":
                self.remove_goal()
            elif command_input == "atg":
                self.attach_goal()
            elif command_input == "dtg":
                self.detach_goal()
            elif command_input == "egc":
                self.toggle_goal_collision(True)
            elif command_input == "dgc":
                self.toggle_goal_collision(False)
            elif command_input == "at":
                self.attach_tool()
            elif command_input == "dt":
                self.detach_tool()
            else: # move arm
                direction = command_input
                self.move_arm(direction)        
    
if __name__ == '__main__':
    try:
        rospy.init_node('simulator_interaction', anonymous=True)

        interaction = SimulatorInteraction()
        interaction.run()

    except rospy.ROSInterruptException:
        pass    
    
