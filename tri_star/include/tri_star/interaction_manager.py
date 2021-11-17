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
import copy

import rospy

from tri_star.constants import TASK_TYPE_POSE_CHANGE, TASK_TYPE_OTHER
from tri_star import robot_util
from tri_star import perception_util
from tri_star import transformations as tfs
from tri_star import transformation_util
from tri_star import pointcloud_util
from tri_star import constants

def get_body_Tz(T, angle):
    angle = np.deg2rad(angle)
    Tz = np.array([[np.cos(angle), -np.sin(angle), 0., 0.],
                   [np.sin(angle),  np.cos(angle), 0., 0.],
                   [0.,             0.,            1., 0.],
                   [0.,             0.,            0., 1.]])
    return np.matmul(T, Tz)

def get_body_Tx(T, angle):
    angle = np.deg2rad(angle)
    Tx = np.array([[1., 0.,             0.,            0.],
                   [0., np.cos(angle), -np.sin(angle), 0.],
                   [0., np.sin(angle),  np.cos(angle), 0.],
                   [0., 0.,             0.,            1.]])
    return np.matmul(T, Tx)

class InteractionManager(object):
    def __init__(self):
        perception_util.remove_goal()
        perception_util.detach_tool()
        self.Tee_tool = None

        self.robot = robot_util.Robot()
        self.training_sample = TrainingSample()
        self.testing_sample = None
    
    def reset_grasping_gesture(self):
        self.Tee_tool = None
        perception_util.detach_tool()
    
    def perceive_grasping_tool_gesture(self, tool):
        perception_util.detach_tool()
        finish = False
        Tworld_tool = None
        while not finish:
            self.robot.free_drive_robot(True)
            choice = raw_input("pose the tool to get the Tworld_tool")
            self.robot.free_drive_robot(False)
            self.robot.connect_robot(True)

            Tee_tool = perception_util.get_Tee_tool()
            print "Tee_tool"
            print Tee_tool
           
            if (not constants.is_simulator()) and Tee_tool is not None:
                Tworld_ee = self.robot.get_robot_pose()
                Tworld_tool = np.matmul(Tworld_ee, Tee_tool)
                perception_util.attach_tool(tool, Tworld_tool=Tworld_tool)            
            
            choice = raw_input("press f to save this, or other key to retake the parameter...")
            if choice.lower() == "f":
                self.Tee_tool = Tee_tool
                finish = True
        
        self.robot.reset_robot()

    def get_grasping_tool_gesture(self):
        return self.Tee_tool
    
    def switch_tool(self):
        self.Tee_tool = None
        perception_util.detach_tool()

        self.robot.gripper_robot(True)
        # add the tool
        # does not know the pose yet
        self.robot.gripper_robot(False)
    
    def perceive_training_sample(self, tool, goal):
        if self.Tee_tool is None:
            self.perceive_grasping_tool_gesture(tool)
            
        return self.training_sample.perceive_training_sample(self.Tee_tool, goal)
    
    def perceive_testing_sample(self, task_type, tool, goal, task):
        if self.Tee_tool is None:
            self.perceive_grasping_tool_gesture(tool)        
        
        self.testing_sample = TestingSample()
        return self.testing_sample.perceive_testing_sample(self.Tee_tool, task_type, goal, task)
    
    def find_trajectory_index(self, tool_trajectories):
        Tee_tool = self.Tee_tool
        robot = self.robot
        
        ee_trajectories = []
        for tool_trajectory in tool_trajectories:
            ee_trajectories.append([np.matmul(Tworld_tool, transformation_util.get_transformation_matrix_inverse(Tee_tool)) for Tworld_tool in tool_trajectory])
        
        min_joint_changes = np.inf
        chosen_index = 0
        current_index = 0
        for ee_trajectory in ee_trajectories:
            initial_pose = ee_trajectory[0]
            is_reach, joint_changes = robot.check_pose(initial_pose)
            if is_reach:
                if joint_changes < min_joint_changes:
                    for Tee in ee_trajectory[1:]:
                        is_reach, _ = robot.check_pose(Tee)
                        if is_reach == False:
                            break
                    if is_reach:
                        min_joint_changes = joint_changes
                        chosen_index = current_index
                        print "chose trajectory: ", chosen_index
            current_index += 1
        
        return chosen_index    
    
    def run_testing_sample(self, goal, tool_trajectories, task_type, tool, control_condition=False):
        if self.testing_sample is None:
            print "testing sample not provided yet!"
        
        if control_condition:
            ee_trajectory = tool_trajectories[0]
            self.testing_sample.run_test(ee_trajectory, task_type, goal, tool, control_condition=control_condition)
        else:
            if self.Tee_tool is None:
                self.perceive_grasping_tool_gesture(tool) 
            
            min_joint_changes = np.inf
            min_index = -1
            trajectory_index = 0
            if len(tool_trajectories) > 1:
                trajectory_index = self.find_trajectory_index(tool_trajectories)

            tool_trajectory = [copy.deepcopy(T) for T in tool_trajectories[trajectory_index]]
            
            self.testing_sample.run_test(tool_trajectory, task_type, goal, tool)
        
        self.testing_sample = None
    
class TrainingSample(object):
    def __init__(self):
        self.robot = robot_util.Robot()

    def perceive_training_sample(self, Tee_tool, goal): # return all in the world frame
        raw_input("place the target object at the START pose. press any key when done...")
        self.Tworld_goalstart = perception_util.get_Tworld_goal()
        if not constants.is_simulator():
            perception_util.remove_goal()
            perception_util.add_goal(goal, self.Tworld_goalstart)  
        
        self.robot.connect_robot(True)
        self.robot.reset_robot()
        
        tool_trajectory = []
        finish = False
        index = 1
        
        perception_util.disable_goal_collision()
        
        while not finish:
            self.robot.free_drive_robot(True)
            choice = raw_input("pose the tool to get the point on the trajectory, press f to finish")
            self.robot.free_drive_robot(False)
            self.robot.connect_robot(True)
            
            if choice.lower() == "f":
                finish = True
            else:
                Tworld_ee = self.robot.get_robot_pose()
                Tworld_tool = np.matmul(Tworld_ee, Tee_tool)
                tool_trajectory.append(Tworld_tool)
                print "tool trajectory point {}: ".format(index)
                print Tworld_tool
                index += 1
        
        perception_util.enable_goal_collision()
        
        self.robot.reset_robot()
        
        raw_input("place the target object at the TARGET/END pose. press any key when done...")
        self.Tworld_goalend = perception_util.get_Tworld_goal(self.Tworld_goalstart)
        if not constants.is_simulator():
            perception_util.remove_goal()
            perception_util.add_goal(goal, self.Tworld_goalend)
        print "Tworld_goalend:"
        print self.Tworld_goalend        
        
        return self.Tworld_goalstart, self.Tworld_goalend, tool_trajectory

class TestingSample(object):
    def __init__(self):
        self.robot = robot_util.Robot()
        self.Tee_tool = None
        self.Tworld_goalstart = None
        self.Tworld_goalend = None
    
    def perceive_testing_sample(self, Tee_tool, task_type, goal, task):
        if task_type == TASK_TYPE_POSE_CHANGE: 
            self.Tworld_goalend = np.identity(4)
            raw_input("place the target object at the TARGET/END pose. press any key when done...")
            self.Tworld_goalend = perception_util.get_Tworld_goal()
            print "Tworld_goalend:"
            print self.Tworld_goalend
            if not constants.is_simulator():
                perception_util.remove_goal()
                perception_util.add_goal(goal, self.Tworld_goalend)            
        
        print "[interaction_manager][perceive_testing_sample] self.Tworld_goalend"
        print self.Tworld_goalend
        
        raw_input("place the target object at the START pose. press any key when done...")
        if not self.Tworld_goalend is None:
            self.Tworld_goalstart = perception_util.get_Tworld_goal()
            if not constants.is_simulator():
                perception_util.remove_goal()
                perception_util.add_goal(goal, self.Tworld_goalstart)
        else: # other tasks
            self.Tworld_goalstart = perception_util.get_Tworld_goal(np.identity(4))
            if not constants.is_simulator():
                perception_util.remove_goal()
                perception_util.add_goal(goal, self.Tworld_goalstart)
        
        print "Tworld_goalstart:"
        print self.Tworld_goalstart
        print "Tworld_goalend:"
        print self.Tworld_goalend
        
        self.Tee_tool = Tee_tool.copy()
        
        return self.Tworld_goalstart, self.Tworld_goalend

    def run_test(self, tool_trajectory, task_type, goal, tool, control_condition=False):
        self.robot.connect_robot(True)
        self.robot.reset_robot()
        
        if control_condition:
            ee_trajectory = tool_trajectory
            ee_trajectory = transformation_util.clear_ee_trajectory(ee_trajectory)
        else:
            ee_trajectory = [np.matmul(Tworld_tool, transformation_util.get_transformation_matrix_inverse(self.Tee_tool)) for Tworld_tool in tool_trajectory]         
            ee_trajectory = transformation_util.clear_trajectory(ee_trajectory, tool, Tee_tool=self.Tee_tool)   
        
        if constants.get_robot_platform() != constants.ROBOT_PLATFORM_KUKA:
            self.robot.set_robot_pose(ee_trajectory[0])
        
        print "start moving"
        if constants.is_simulator():
            perception_util.enable_goal_collision()
            rospy.sleep(10.)
            
            self.robot.set_robot_pose(ee_trajectory[0], ask_before_move=True)
            self.robot.set_robot_pose(ee_trajectory[1], ask_before_move=False)
            self.robot.set_robot_pose(ee_trajectory[2], ask_before_move=False)
            perception_util.attach_goal(self.Tworld_goalstart, goal)
            self.robot.execute_trajectory(ee_trajectory[3:-1], ask_before_move=False)
            perception_util.detach_goal()
            self.robot.set_robot_pose(ee_trajectory[-1], ask_before_move=False)            

        else:
            perception_util.remove_goal()
            perception_util.detach_tool()
            perception_util.enable_goal_collision()

            print "Enabling goal collision..."
            rospy.sleep(5.5)
            self.robot.execute_trajectory(ee_trajectory)

        Tworld_ee = self.robot.get_robot_pose()
        Tworld_tool = np.matmul(Tworld_ee, self.Tee_tool)
        perception_util.attach_tool(tool, Tworld_tool=Tworld_tool)

        self.robot.reset_robot()

        Tactual_world_goal_end = perception_util.get_Tworld_goal()
        
        self.analyze_result(Tactual_world_goal_end)
        
        return Tactual_world_goal_end, self.Tworld_goalend
    
    def analyze_result(self, Tactual_world_goal_end):
        if self.Tworld_goalend is None or Tactual_world_goal_end is None:
            return
        
        target_translation, target_rotation = transformation_util.decompose_homogeneous_transformation_matrix(self.Tworld_goalend)
        actual_translation, actual_rotation = transformation_util.decompose_homogeneous_transformation_matrix(Tactual_world_goal_end)
        start_translation, start_rotation = transformation_util.decompose_homogeneous_transformation_matrix(self.Tworld_goalstart)

        print "-" * 100

        print "desired translation: ", target_translation
        print "actual translation: ", actual_translation

        print "translation difference(m): ", actual_translation - target_translation
        print "translation difference length (m): ", transformation_util.get_vector_length(actual_translation - target_translation)
        print "translation different length(m) 2D: ", transformation_util.get_vector_length(actual_translation[0:2] - target_translation[0:2])

        print
        print "start translation: ", start_translation
        print "desired translation: ", target_translation
        print "pushing length (m): ", transformation_util.get_vector_length(target_translation - start_translation)
        print "pushing length (m) 2D: ", transformation_util.get_vector_length(target_translation[0:2] - start_translation[0:2])
        print "pushing error (%): ", transformation_util.get_vector_length(actual_translation - target_translation) / transformation_util.get_vector_length(start_translation - target_translation) * 100., "%"
        print "pushing error (%) 2D: ", transformation_util.get_vector_length(actual_translation[0:2] - target_translation[0:2]) / transformation_util.get_vector_length(start_translation[0:2] - target_translation[0:2]) * 100., "%"

        target_alpha, target_beta, target_gamma = tfs.euler_from_quaternion(target_rotation)
        actual_alpha, actual_beta, actual_gamma = tfs.euler_from_quaternion(actual_rotation)
        start_alpha, start_beta, start_gamma = tfs.euler_from_quaternion(start_rotation)
        
        target_angle = np.array([target_alpha, target_beta, target_gamma])
        actual_angle = np.array([actual_alpha, actual_beta, actual_gamma])
        
        print "desired rotation: ", target_angle
        print "actual rotation: ", actual_rotation

        print "rotation difference (degrees): ", np.degrees(target_angle - actual_angle)
        print "rotation difference length (degrees): ", transformation_util.get_vector_length(np.degrees(target_angle - actual_angle))

        print
        print "start rotation: ", np.array([start_alpha, start_beta, start_gamma])
        print "push start rotation gamma, (degree): ", np.rad2deg(start_gamma)
        print "push desired rotation gamma, (degree): ", np.rad2deg(target_gamma)
        print "push rotation difference gamma (degree): ", np.rad2deg(target_gamma - start_gamma)
        if target_gamma - start_gamma != 0:
            print "push rotation difference gamma (%): ", (target_gamma - actual_gamma) / (target_gamma - start_gamma) * 100., "%"
