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
import sys
from abc import abstractmethod
from copy import deepcopy
import json
import random
import open3d as o3d

import numpy as np

import transformations as tfs

from tool_substitution.tool_substitution_controller import ToolSubstitution
from tool_substitution.tool_pointcloud import ToolPointCloud
from tool_substitution.sample_pointcloud import GeneratePointcloud

from tri_star import file_util
from tri_star import transformation_util
from tri_star import perception_util
from tri_star import pointcloud_util
from tri_star import constants
from tri_star import learning_sample

# strings to use to save files, in the format of <variable type>_<name>, :p
# task related
STR_TASK                = "task"
STR_TOOL                = "tool"
STR_GOAL                = "goal"
STR_TASK_TYPE           = "task_type"
STR_SUBTASK_TYPE        = "subtask_type"
STR_TOOL_USAGE_PATH     = "tool_usage_file_path"
STR_RAW_TOOL_USAGE_PATH = "raw_tool_usage_file_path"

# samples
LIST_STR_SAMPLE_PATH = "sample_path"

# variables
DICT_TTOOL_GOAL                      = "Ttool_goal_group"
LIST_INT_TOOL_CONTACT_AREA_INDEX     = "tool_contact_area_index"
LIST_INT_GOAL_CONTACT_AREA_INDEX     = "goal_contact_area_index"
LIST_ARRAY_TRAJECTORY_AXIS           = "trajectory_axis"
LIST_FLOAT_TRAJECTORY_THETA          = "trajectory_theta"
LIST_ARRAY_SAMPLE_TRAJECTORY_AXIS    = "sample_trajectory_axis"
LIST_FLOAT_SAMPLE_TRAJECTORY_THETA   = "sample_trajectory_theta"
MATRIX_TEND                          = "Tend"
INT_TEND_INDEX                       = "Tend_index"
BOOL_CIRCULAR                        = "circular"
MATRIX_TEFFECTIVESTART_HOVERINGSTART = "Teffectivestart_hoveringstart"
MATRIX_TEFFECTIVESTART_HOVERINGEND   = "Teffectivestart_hoveringend"
MATRIX_TEFFECTIVEEND_HOVERINGEND     = "Teffectiveend_hoveringend"

# substitute variable:
STR_SOURCE_TOOL_FILE_PATH = "source_tool_file_path"

# File template
FILENAME_TOOL_USAGE     = "tool_usage.txt"
FILENAME_RAW_TOOL_USAGE = "raw_tool_usage.txt"

class ToolUsage(object):
    def __init__(self, task, tool, goal, base_data_dir):
        self.task = task
        self.tool = tool
        self.goal = goal
        self.task_type = None
        self.subtask_type = None
        self.base_data_dir = None # to be extended in child class
        self.is_learned = False

        self.reset()
    
    def clear_learned_data(self):
        # Ttool_goal
        self.Ttool_goal_group = {} # {axis: [AngleGroup, originT]}
        
        # contact area
        self.tool_contact_area_index = []
        self.goal_contact_area_index = []
        
        # trajectory
        self.trajectory_axis = [] # [S, S, S, ...]
        self.trajectory_theta = [] # [angle, angle, angle, ...]
        self.sample_averaged_trajectory_axis = [] # [S, S, S, ...]
        self.sample_averaged_trajectory_theta = [] # [angle, angle, angle, ...]
        self.Tend = np.identity(4) # needed for more than 1 fragment
        self.Tend_index = -1
        self.is_circular = False
        
        # hovering related. In the tool effective start frame.
        self.Teffectivestart_hoveringstart = None
        self.Teffectivestart_hoveringend = None
        self.Teffectiveend_hoveringend = None

    def reset(self):
        self.samples = []
        self.clear_learned_data()

    """
    get a new sample
    """
    def process_new_sample(self, Tworld_goalstart, Tworld_goalend, tool_trajectory_world_frame):
        current_sample = learning_sample.Sample(self.task, self.tool, self.goal, self.base_data_dir)
        current_sample.process_sample(Tworld_goalstart, Tworld_goalend, tool_trajectory_world_frame)
        self.samples.append(current_sample)

    def get_task_type_from_sample(self, sample):
        if sample is not None:
            return sample.get_task_type()
        else:
            return None

    """
    learn source/substitue tool usages
    """
    def step_1_find_task_type_from_samples(self):
        num_pose_change = 0
        num_other = 0
        for sample in self.samples:
            sample_task_type = self.get_task_type_from_sample(sample)
            if sample_task_type == constants.TASK_TYPE_OTHER:
                num_other += 1
            elif sample_task_type == constants.TASK_TYPE_POSE_CHANGE:
                num_pose_change += 1
        
        task_type = constants.TASK_TYPE_OTHER
        if num_pose_change >= num_other:
            task_type = constants.TASK_TYPE_POSE_CHANGE

        return task_type
    
    def step_1_find_subtask_type_from_samples(self):
        v_goals = []
        for sample in self.samples:
            v_goal = sample.get_v_goal()
            v_goals.append(v_goal)
        
        results, result_indices = transformation_util.cluster_array_DBSCAN(v_goals, eps=0.01, min_samples=2)
        num_clusters = len(results.keys())
        
        print "step 1 num_clusters: ", num_clusters
        
        subtask_type = constants.TASK_SUBTYPE_POSE_CHANGE_SPECIFIC
        if (num_clusters == 1 and results.keys()[0] == -1) or num_clusters > 1:
            subtask_type = constants.TASK_SUBTYPE_POSE_CHANGE_GENERAL
        
        return subtask_type

    @abstractmethod
    def step_2_update_Ttool_goal(self, sample):
        pass
    
    @abstractmethod
    def step_2_categorize_Tool_goal(self, Ttool_goals, axes):
        pass
    
    @abstractmethod
    def step_3_process_contact_area(self, indices):
        pass
    
    @abstractmethod
    # trajectories: [[axes, angles, Tend], ...]
    def step_4_process_trajectory(self, trajectories):
        pass
    
    @abstractmethod
    def step_5_process_hovering(self, Teffectivestart_hoveringstarts, Teffectivestart_hoveringends, Teffectiveend_hoveringends):
        pass
    
    @abstractmethod
    # main algorithm for learning
    def process_learned_samples(self):
        pass
        
    """
    use the tool
    """
    def _get_Ttool_goals_from_group(self, Tworld_goal, is_sub=False, sub_tool_name="", sub_goal_name="", Tsourcetool_subtool=np.identity(4), Tworld_subgoal=np.identity(4)):
        axis = np.array(self.Ttool_goal_group.keys()[0])
        angle_group = self.Ttool_goal_group[tuple(axis)][0]
        origin = self.Ttool_goal_group[tuple(axis)][1]
        
        print "[tool_usage][_get_Ttool_goals_from_group] origin: "
        print origin
        
        Ttool_goals = []
        angles = []
        if angle_group.is_uniform():
            range_value = angle_group.group_range()
            min_angle, max_angle = range_value[0], range_value[1]
            interval = np.deg2rad(1.)
            num_interval = int((max_angle - min_angle) / interval) + 1
            angles = [min_angle + i * interval for i in range(num_interval)]
        else:
            angles = angle_group.group_mean()

        Ts = [transformation_util.get_transformation_matrix_from_exponential(axis, theta) for theta in angles]

        Ttool_goals = [np.matmul(T, origin) for T in Ts]
        Ttool_goals = [self._adjust_Ttool_goal(T, Tworld_goal, axis, is_sub=is_sub, sub_tool_name=sub_tool_name, sub_goal_name=sub_goal_name, Tsourcetool_subtool=Tsourcetool_subtool, Tworld_subgoal=Tworld_subgoal) for T in Ttool_goals]
        
        print "[tool_usage][_get_Ttool_goals_from_group] Ttool_goals[0]: "
        print Ttool_goals[0]       
        
        return Ttool_goals
    
    def _align_tool_goal(self, tool_name, goal_name, Tworld_tool, Tworld_goal, axis_world_frame):
        goal_pc = pointcloud_util.get_goal_mesh(goal_name, paint=True)
        goal_pc.transform(Tworld_goal)            
        tool_pc = pointcloud_util.get_tool_mesh(tool_name, paint=True)
        tool_pc.transform(Tworld_tool)
        
        tool_contact_area_index = pointcloud_util.get_contact_surface(np.asarray(tool_pc.points), np.asarray(goal_pc.points), get_goal=False)
        tool_contact_area_pc = pointcloud_util.get_tool_contact_area(tool_name, tool_contact_area_index)
        tool_contact_area_pc.paint_uniform_color(np.array([1., 0., 0.]))
        tool_contact_area_pc.transform(Tworld_tool)
        tool_center = tool_contact_area_pc.get_center()
        
        line_pc = pointcloud_util.get_line(tool_center, axis_world_frame, 1.)
        
        point_1, point_2, line_pc = pointcloud_util.get_line_pc_intersect(line_pc, goal_pc)
        
        left_point, right_point = point_1, point_2
        if not transformation_util.is_similar_vector(transformation_util.normalize(point_1 - point_2), axis_world_frame, error=np.deg2rad(1.)):
            print "[tool_usage][_align_tool_goal] flip left and right point"
            left_point, right_point = point_2, point_1
        
        move_direction = tool_center - left_point
        
        Tworld_tool[0, 3] -= move_direction[0]
        Tworld_tool[1, 3] -= move_direction[1]
        Tworld_tool[2, 3] -= move_direction[2]      

        return Tworld_tool, Tworld_goal
    
    def _adjust_Ttool_goal(self, Ttool_goal, Tworld_sourcegoal, axis, is_sub=False, sub_tool_name="", sub_goal_name="", Tsourcetool_subtool=np.identity(4), Tworld_subgoal=np.identity(4)):
        if self.task_type == constants.TASK_TYPE_POSE_CHANGE:
            Tworld_sourcetool = np.matmul(Tworld_sourcegoal, transformation_util.get_transformation_matrix_inverse(Ttool_goal))
            v_tool_frame_end = np.array([[axis[0], axis[1], axis[2], 1.]]).T # in the tool frame
            v_tool_frame_start = np.array([[0., 0., 0., 1.]]).T
            v_world_frame_end = np.matmul(Tworld_sourcetool, v_tool_frame_end).T[0][:3]
            v_world_frame_start = np.matmul(Tworld_sourcetool, v_tool_frame_start).T[0][:3]
            v_world_frame = transformation_util.normalize(v_world_frame_start - v_world_frame_end)

            goal_name = self.goal
            tool_name = self.tool
            Tcalculation_tool = deepcopy(Tworld_sourcetool)
            Tcalculation_goal = deepcopy(Tworld_sourcegoal)
            if is_sub:
                goal_name = sub_goal_name
                tool_name = sub_tool_name
                Tworld_subtool = np.matmul(Tworld_sourcetool, Tsourcetool_subtool)
                Tcalculation_tool = deepcopy(Tworld_subtool)
                Tcalculation_goal = deepcopy(Tworld_subgoal)
              
            Tcalculation_tool, Tcalculation_goal = self._align_tool_goal(tool_name, 
                                                                         goal_name, 
                                                                         deepcopy(Tcalculation_tool), 
                                                                         deepcopy(Tcalculation_goal), 
                                                                         v_world_frame)
          
            Ttool_goal = np.identity(4)
            if is_sub:
                Tworld_subtool = deepcopy(Tcalculation_tool)
                Tworld_subgoal = deepcopy(Tcalculation_goal)
                Tworld_sourcetool = np.matmul(Tworld_subtool, transformation_util.get_transformation_matrix_inverse(Tsourcetool_subtool))
                Ttool_goal = np.matmul(transformation_util.get_transformation_matrix_inverse(Tworld_sourcetool), Tworld_sourcegoal)
            else:
                Ttool_goal = np.matmul(transformation_util.get_transformation_matrix_inverse(Tcalculation_tool), Tcalculation_goal)
        
        return Ttool_goal
    
    def step_2_Ttool_goals(self, Tworld_goalstart, Tworld_goalend=None, is_sub=False, sub_tool_name="", sub_goal_name="", Tsourcetool_subtool=np.identity(4), Tworld_subgoal=np.identity(4)):
        updated_Tworld_goalstart = Tworld_goalstart
        updated_Tworld_goalend = Tworld_goalend
        if self.task_type == constants.TASK_TYPE_POSE_CHANGE:
            assert not Tworld_goalend is None, "Tworld_goalend is None!"
            if self.subtask_type == constants.TASK_SUBTYPE_POSE_CHANGE_GENERAL:
                T = transformation_util.get_body_frame_transformation(Tworld_goalstart, Tworld_goalend)
                S, theta = transformation_util.get_exponential_from_transformation_matrix(T)
                v_start_goal_frame = transformation_util.normalize(S[3:])
                v_start_goal_frame = np.array([[v_start_goal_frame[0], v_start_goal_frame[1], v_start_goal_frame[2], 1.]]).T
                v_start_world_frame_end = np.matmul(Tworld_goalstart, v_start_goal_frame)
                v_start_world_frame_end = v_start_world_frame_end.T[0][0:3]
                v_start_world_frame_start = np.matmul(Tworld_goalstart, np.array([[0., 0., 0., 1.]]).T)
                v_start_world_frame_start = v_start_world_frame_start.T[0][0:3]
                v_start_world_frame = v_start_world_frame_end - v_start_world_frame_start                
                v_start_world_frame = transformation_util.normalize(v_start_world_frame)

                g = np.array([0., 0., -1.])
                
                if not transformation_util.is_colinear(v_start_world_frame, g, error=0.01): # TODO:tune error
                    x = v_start_world_frame
                    z = g
                    y = transformation_util.normalize(np.cross(z, x))
                    z = transformation_util.normalize(np.cross(x, y))
                    R = np.array([x, y, z]).T
                else:
                    x = v_start_world_frame
                    y = np.array([0., 1., 0.])
                    z = transformation_util.normalize(np.cross(x, y))
                    y = transformation_util.normalize(np.cross(z, x))
                    R = np.array([x, y, z]).T
                
                p = Tworld_goalstart[:3, 3]
                updated_Tworld_goalstart = transformation_util.get_transformation_matrix_with_rotation_matrix(R, p)
                T_goal_world = transformation_util.get_fixed_frame_transformation(Tworld_goalstart, Tworld_goalend)
                updated_Tworld_goalend = np.matmul(T_goal_world, updated_Tworld_goalstart)
        
        Ttool_goals = self._get_Ttool_goals_from_group(updated_Tworld_goalstart, is_sub=is_sub, sub_tool_name=sub_tool_name, sub_goal_name=sub_goal_name, Tsourcetool_subtool=Tsourcetool_subtool, Tworld_subgoal=Tworld_subgoal)
        Tworld_toolstarts = [np.matmul(updated_Tworld_goalstart, transformation_util.get_transformation_matrix_inverse(Ttool_goal)) for Ttool_goal in Ttool_goals]
        Tworld_toolends = []
        if self.task_type == constants.TASK_TYPE_POSE_CHANGE:
            Tworld_toolends = [np.matmul(updated_Tworld_goalend, transformation_util.get_transformation_matrix_inverse(Ttool_goal)) for Ttool_goal in Ttool_goals]           
        elif self.task_type == constants.TASK_TYPE_OTHER:
            for Tworld_toolstart in Tworld_toolstarts:
                Tworld_toolend = deepcopy(Tworld_toolstart)
                for i in range(len(self.sample_averaged_trajectory_axis)):
                    S = self.sample_averaged_trajectory_axis[i]
                    theta = self.sample_averaged_trajectory_theta[i]
                    T_change = transformation_util.get_transformation_matrix_from_exponential(S, theta)
                    Tworld_toolend = np.matmul(Tworld_toolend, T_change)
                Tworld_toolends.append(Tworld_toolend)
        return Tworld_toolstarts, Tworld_toolends    
    
    def step_4_get_trajectory(self, Tworld_toolstarts, Tworld_toolends, num_circles, functional_component_rescale):
        trajectories = []
        
        if len(self.trajectory_axis) == 0: # no trajectory (e.g., knock) or trajectory same as interpolation (e.g., push)
            print "[tool_usage][step_4_get_trajectory] no trajectory or the trajectory is the same as the interpolation"
            for i in range(len(Tworld_toolstarts)):
                Tworld_toolstart = Tworld_toolstarts[i]
                Tworld_toolend = Tworld_toolends[i]
                
                # for testing purpose to have only limited points on the trajectory
                if constants.is_testing():
                    trajectory = [Tworld_toolstart, Tworld_toolend]
                else:
                    # This is for the real system, get the interpolated points
                    trajectory = transformation_util.get_trajectory_interpolation_with_transformations(Tworld_toolstart, Tworld_toolend)
                
                trajectories.append(trajectory)
        
        elif len(self.trajectory_axis) == 1: # just 1 fragment and needs to follow certain trajectory (e.g., screw driver)
            print "[tool_usage][step_4_get_trajectory] just 1 fragment and needs to follow it"
            S = self.trajectory_axis[0]
            theta = self.trajectory_theta[0]
            screw_trajectory = [(S, theta),]
            screw_trajectory = transformation_util.rescale_trajectory(functional_component_rescale, screw_trajectory)
            S = screw_trajectory[0][0]
            theta = screw_trajectory[0][1]
            for i in range(len(Tworld_toolstarts)):
                Tworld_toolstart = Tworld_toolstarts[i]
                Tworld_toolend = Tworld_toolends[i]       
                T = transformation_util.get_body_frame_transformation(Tworld_toolstart, Tworld_toolend)
                theta = 0.
                if num_circles != -1:
                    theta = num_circles * np.pi * 2.
                else:
                    theta = transformation_util.get_delta_with_chosen_screw_axis_from_transformation(T, S)

                if constants.is_testing():
                    interval = np.pi / 2.
                    if theta < 0:
                        interval = -np.pi / 2.
                    num_interval = abs(int(theta / interval))
                    trajectory = [np.matmul(Tworld_toolstart, transformation_util.get_transformation_matrix_from_exponential(S, interval * i)) for i in range(num_interval)]
                    trajectory.append(np.matmul(Tworld_toolstart, transformation_util.get_transformation_matrix_from_exponential(S, theta)))
                else:
                    trajectory = transformation_util.get_trajectory_interpolation_with_body_frame_screw_axis(S, theta, Tworld_toolstart)
                
                trajectories.append(trajectory)
        else:
            print "[tool_usage][step_4_get_trajectory] multiple fragments"
            for i in range(len(Tworld_toolstarts)):
                Tworld_toolstart = Tworld_toolstarts[i]
                Tworld_toolend = Tworld_toolends[i]
                trajectory = []
                screw_trajectory = [(self.trajectory_axis[i], self.trajectory_theta[i]) for i in range(len(self.trajectory_axis))]
                screw_trajectory = transformation_util.rescale_trajectory(functional_component_rescale, screw_trajectory)
                if self.task == "draw":
                    invalid_input = False
                    while not invalid_input:
                        try:
                            scale = float(raw_input("scale: "))
                            rotate = np.deg2rad(float(raw_input("rotation(degree): ")))
                            invalid_input = True
                        except ValueError:
                            print "Invalid input! Need a float!"
                    screw_trajectory = transformation_util.rescale_trajectory(scale, screw_trajectory)
                
                if self.task_type == constants.TASK_TYPE_POSE_CHANGE:
                    screw_trajectory = transformation_util.goal_trajectory(Tworld_toolstart, screw_trajectory, np.array([0., 0., 0.]), np.array([Tworld_toolend[0, 3], Tworld_toolend[1, 3], Tworld_toolend[2, 3]]))
                T_start = Tworld_toolstart
                
                if self.task == "draw":
                    # x axis:
                    T = np.array([[ 1., 0.,              0.,             0.],
                                  [ 0., np.cos(rotate), -np.sin(rotate), 0.],
                                  [ 0., np.sin(rotate),  np.cos(rotate), 0.],
                                  [ 0., 0.,              0.,             1.]])
                    T_start = np.matmul(T_start, T)
                
                # for simulators and for testing purpose to have only limited points on the trajectory
                if constants.is_testing():
                    trajectory.append(T_start)
                    for (S, theta) in screw_trajectory:
                        trajectory.append(np.matmul(T_start, transformation_util.get_transformation_matrix_from_exponential(S, theta)))
                        T_start = trajectory[-1]
                else:
                   # Provide interpolated points for real systems
                    for (S, theta) in screw_trajectory:
                        trajectory += transformation_util.get_trajectory_interpolation_with_body_frame_screw_axis(S, theta, T_start)
                        T_start = trajectory[-1]                
                
                trajectories.append(trajectory)

        return trajectories
    
    def step_5_add_hovering(self, tool_trajectory_world_frame):
        Tworld_effectivestart = tool_trajectory_world_frame[0]
        Tworld_effectiveend = tool_trajectory_world_frame[-1]
        
        Tworld_hoveringstart = np.matmul(Tworld_effectivestart, self.Teffectivestart_hoveringstart)
        Tworld_hoveringend = np.matmul(Tworld_effectivestart, self.Teffectivestart_hoveringend)
        Tworld_unhoveringend = np.matmul(Tworld_effectiveend, self.Teffectiveend_hoveringend)
        
        trajectory = []      

        if constants.is_testing():
            trajectory.append(Tworld_hoveringstart)
            trajectory.append(Tworld_hoveringend)
            trajectory.append(Tworld_effectivestart)
            trajectory += tool_trajectory_world_frame
            trajectory.append(Tworld_unhoveringend)
        else:
            # provide the interpolated trajectory to robot to make sure it will follow it exactly
            trajectory += transformation_util.get_trajectory_interpolation_with_transformations(Tworld_hoveringstart, Tworld_hoveringend)
            trajectory += transformation_util.get_trajectory_interpolation_with_transformations(Tworld_hoveringend, Tworld_effectivestart)
            trajectory += tool_trajectory_world_frame
            trajectory += transformation_util.get_trajectory_interpolation_with_transformations(Tworld_effectiveend, Tworld_unhoveringend)
        
        print "[tool usage][step_5_add_hovering]trajectory constains {} points".format(len(trajectory))
        
        return trajectory  

    def get_star_1_control_condition_usage(self, Tworld_goalstart):
        trajectories_goal_frame = []
        
        selected_sample = random.choice(self.samples)
        tool_trajectory_goal_frame = selected_sample.get_tool_trajectory_goal_frame()
        if self.task_type == constants.TASK_TYPE_POSE_CHANGE:
            Ttool_goal = selected_sample.get_reference_point()
            Tgoal_tool = transformation_util.get_transformation_matrix_inverse(Ttool_goal)
            tool_trajectory_goal_frame = deepcopy(tool_trajectory_goal_frame)
            tool_trajectory_goal_frame.insert(2, Tgoal_tool)
        training_Tee_tool = constants.STAR_1_TEE_TOOL[self.tool]
        ee_trajectory_goal_frame = [np.matmul(Tgoal_tool, transformation_util.get_transformation_matrix_inverse(training_Tee_tool)) for Tgoal_tool in tool_trajectory_goal_frame]
        ee_trajectory_world_frame = [np.matmul(Tworld_goalstart, Tgoal_ee) for Tgoal_ee in ee_trajectory_goal_frame]
        
        interpolated_ee_trajectory_world_frame = []
        for i in range(len(ee_trajectory_world_frame) - 1):
            Tstart = deepcopy(ee_trajectory_world_frame[i])
            Tend = deepcopy(ee_trajectory_world_frame[i + 1])
            interpolated_ee_trajectory_world_frame += transformation_util.get_trajectory_interpolation_with_transformations(Tstart, Tend)

        return [interpolated_ee_trajectory_world_frame]

    def get_usage(self, Tworld_goalstart, Tworld_goalend=None, circle_num=-1, is_sub=False, sub_tool_name="", sub_goal_name="", Tsourcetool_subtool=np.identity(4), Tworld_subgoal=np.identity(4), functional_component_rescale=1.0):
        # step 2 (corresponding to learning): get Tworld_toolstart, Tworld_toolend, Ttool_goal
        Tworld_toolstarts, Tworld_toolends = self.step_2_Ttool_goals(Tworld_goalstart, Tworld_goalend, is_sub=is_sub, sub_tool_name=sub_tool_name, sub_goal_name=sub_goal_name, Tsourcetool_subtool=Tsourcetool_subtool, Tworld_subgoal=Tworld_subgoal)
        
        # step 4: get trajectory
        tool_trajectories_world_frame = self.step_4_get_trajectory(Tworld_toolstarts, Tworld_toolends, circle_num, functional_component_rescale)
        
        # step 5: add hovering
        complete_tool_trajectories_world_frame = []
        for trajectory in tool_trajectories_world_frame:
            complete_tool_trajectories_world_frame.append(self.step_5_add_hovering(trajectory))
        
        return complete_tool_trajectories_world_frame

    """
    getters
    """
    def get_task(self):
        return self.task
    
    def get_tool_name(self):
        return self.tool
    
    def get_goal_name(self):
        return self.goal

    def get_task_type(self):
        return self.task_type

    def get_subtask_type(self):
        return self.subtask_type
    
    def get_data_dir_path(self):
        return self.base_data_dir    
    
    def is_learned(self):
        return self.is_learned
    
    def get_Ttool_goal_group(self):
        return self.Ttool_goal_group

    def get_tool_contact_area_index(self):
        return self.tool_contact_area_index
    
    def get_goal_contact_area_index(self):
        return self.goal_contact_area_index

    def get_trajectory_axis(self):
        return self.trajectory_axis
    
    def get_trajectory_theta(self):
        return self.trajectory_theta
    
    def get_sample_averaged_trajectory_axis(self):
        return self.sample_averaged_trajectory_axis
    
    def get_sample_averaged_trajectory_theta(self):
        return self.sample_averaged_trajectory_theta
    
    def get_Tend(self):
        return self.Tend
    
    def get_Tend_index(self):
        return self.Tend_index
    
    def circular(self):
        return self.is_circular

    def get_Teffectivestart_hoveringstart(self):
        return self.Teffectivestart_hoveringstart

    def get_Teffectivestart_hoveringend(self):
        return self.Teffectivestart_hoveringend

    def Teffectiveend_unhoveringend(self):
        return self.Teffectiveend_unhoveringend

    def get_file_name(self):
        return os.path.join(self.base_data_dir, FILENAME_RAW_TOOL_USAGE)

    """
    I/O save this tool usage, or read the learned usages
    """
    def save(self):
        base_data_dir = os.path.join(constants.get_package_dir(), self.base_data_dir)
        file_util.create_dir(base_data_dir)

        tool_usage_file_path = os.path.join(base_data_dir, FILENAME_TOOL_USAGE)
        raw_tool_usage_file_path = os.path.join(base_data_dir, FILENAME_RAW_TOOL_USAGE)

        relative_tool_usage_file_path = os.path.join(self.base_data_dir, FILENAME_TOOL_USAGE)
        relative_raw_tool_usage_file_path = os.path.join(self.base_data_dir, FILENAME_RAW_TOOL_USAGE)

        relative_sample_file_path = []
        for sample in self.samples:
            sample.save()
            relative_sample_file_path.append(sample.get_file_name())

        tool_usage_file_content = self.get_json(is_elaborate=True)
        tool_usage_file_content[STR_TOOL_USAGE_PATH]     = relative_tool_usage_file_path     # str
        tool_usage_file_content[STR_RAW_TOOL_USAGE_PATH] = relative_raw_tool_usage_file_path # str
        tool_usage_file_content[LIST_STR_SAMPLE_PATH]    = relative_sample_file_path         # [str]
        print "tool_usage_file_content"
        for key in tool_usage_file_content.keys():
            print key, ":"
            print "\t", tool_usage_file_content[key]

        with open(tool_usage_file_path, "w") as write_file:
            json.dump(tool_usage_file_content, write_file, indent = 4, cls=file_util.ToolUseEncoder)        

        raw_tool_usage_file_content = self.get_json(is_elaborate=False)
        raw_tool_usage_file_content[STR_TOOL_USAGE_PATH]     = relative_tool_usage_file_path     # str
        raw_tool_usage_file_content[STR_RAW_TOOL_USAGE_PATH] = relative_raw_tool_usage_file_path # str
        raw_tool_usage_file_content[LIST_STR_SAMPLE_PATH]    = relative_sample_file_path         # [str]
        print "raw_tool_usage_file_content"
        print raw_tool_usage_file_content
        with open(raw_tool_usage_file_path, "w") as write_file:
            json.dump(raw_tool_usage_file_content, write_file, indent = 4, cls=file_util.ToolUseEncoder)                

    def read(self, recalculate=False):
        tool_usage_file_path = os.path.join(constants.get_package_dir(), self.base_data_dir, FILENAME_TOOL_USAGE)

        if not os.path.exists(tool_usage_file_path):
            return False

        # basic info
        self.task = file_util.read_variable(tool_usage_file_path, STR_TASK)
        self.tool = file_util.read_variable(tool_usage_file_path, STR_TOOL)
        self.goal = file_util.read_variable(tool_usage_file_path, STR_GOAL)
        
        # samples
        relative_sample_file_path = file_util.read_variable(tool_usage_file_path, LIST_STR_SAMPLE_PATH)
        for file_path in relative_sample_file_path:
            sample = learning_sample.Sample(self.task, self.tool, self.goal, self.base_data_dir)
            sample_path = os.path.join(constants.get_package_dir(), file_path)
            sample.read(sample_path)
            self.samples.append(sample)        
        
        if recalculate:
            self.process_learned_samples()
        else:
            self.task_type    = file_util.read_variable(tool_usage_file_path, STR_TASK_TYPE)
            self.subtask_type = file_util.read_variable(tool_usage_file_path, STR_SUBTASK_TYPE)
            
            Ttool_goal_group = file_util.read_variable(tool_usage_file_path, DICT_TTOOL_GOAL)
            for key, value in Ttool_goal_group.items():               
                transformation_util.AngleGroup.from_json(value[0])
                self.Ttool_goal_group[tuple(map(float, key.split('|')))] = [transformation_util.AngleGroup.from_json(value[0]), np.array(value[1])]
            
            self.tool_contact_area_index = file_util.read_variable(tool_usage_file_path, LIST_INT_TOOL_CONTACT_AREA_INDEX)
            self.goal_contact_area_index = file_util.read_variable(tool_usage_file_path, LIST_INT_GOAL_CONTACT_AREA_INDEX)
            
            self.trajectory_axis                  = file_util.read_variable(tool_usage_file_path, LIST_ARRAY_TRAJECTORY_AXIS, variable_type = file_util.TYPE_NUMPY, variable_collection_type=file_util.TYPE_LIST)
            self.trajectory_theta                 = file_util.read_variable(tool_usage_file_path, LIST_FLOAT_TRAJECTORY_THETA)
            self.sample_averaged_trajectory_axis  = file_util.read_variable(tool_usage_file_path, LIST_ARRAY_SAMPLE_TRAJECTORY_AXIS, variable_type = file_util.TYPE_NUMPY, variable_collection_type=file_util.TYPE_LIST)
            self.sample_averaged_trajectory_theta = file_util.read_variable(tool_usage_file_path, LIST_FLOAT_SAMPLE_TRAJECTORY_THETA)
            self.Tend                             = file_util.read_variable(tool_usage_file_path, MATRIX_TEND, variable_type = file_util.TYPE_NUMPY)
            self.Tend_index                       = file_util.read_variable(tool_usage_file_path, INT_TEND_INDEX)
            self.is_circular                      = file_util.read_variable(tool_usage_file_path, BOOL_CIRCULAR)
            
            self.Teffectivestart_hoveringstart = file_util.read_variable(tool_usage_file_path, MATRIX_TEFFECTIVESTART_HOVERINGSTART, variable_type = file_util.TYPE_NUMPY)
            self.Teffectivestart_hoveringend   = file_util.read_variable(tool_usage_file_path, MATRIX_TEFFECTIVESTART_HOVERINGEND, variable_type = file_util.TYPE_NUMPY)
            self.Teffectiveend_hoveringend     = file_util.read_variable(tool_usage_file_path, MATRIX_TEFFECTIVEEND_HOVERINGEND, variable_type = file_util.TYPE_NUMPY)
            
        self.is_learned = True

        return True

    def archive(self, to_reset=False):
        file_util.archive(os.path.join(constants.get_package_dir(), self.base_data_dir))
        if to_reset:
            self.reset()

    def get_json(self, is_elaborate=False):
        json_result = {}
        
        # perception data (samples will retrieved in the save function)
        json_result[STR_TASK] = self.task # str
        json_result[STR_TOOL] = self.tool # str
        json_result[STR_GOAL] = self.goal # str
        
        # calculated data
        if is_elaborate:
            json_result[STR_TASK_TYPE]    = self.task_type    # str
            json_result[STR_SUBTASK_TYPE] = self.subtask_type # str
            
            formatted_Ttool_goal_group = {}
            for key in self.Ttool_goal_group.keys():
                formatted_Ttool_goal_group['|'.join([str(num) for num in key])] = self.Ttool_goal_group[key]
            json_result[DICT_TTOOL_GOAL] = formatted_Ttool_goal_group # {str(tuple): [AngleGroup, np.array(4, 4)]}
            
            json_result[LIST_INT_TOOL_CONTACT_AREA_INDEX] = self.tool_contact_area_index # [int, int, int, ...]
            json_result[LIST_INT_GOAL_CONTACT_AREA_INDEX] = self.goal_contact_area_index # [int, int, int, ...]
            
            json_result[LIST_ARRAY_TRAJECTORY_AXIS]         = self.trajectory_axis                  # [np.array(1, 6)]
            json_result[LIST_FLOAT_TRAJECTORY_THETA]        = self.trajectory_theta                 # [float]
            json_result[LIST_ARRAY_SAMPLE_TRAJECTORY_AXIS]  = self.sample_averaged_trajectory_axis  # [np.array(1, 6)]
            json_result[LIST_FLOAT_SAMPLE_TRAJECTORY_THETA] = self.sample_averaged_trajectory_theta # [float]
            json_result[MATRIX_TEND]                        = self.Tend                             # np.array(4, 4)
            json_result[INT_TEND_INDEX]                     = self.Tend_index                       # int
            json_result[BOOL_CIRCULAR]                      = self.is_circular                      # bool
            
            json_result[MATRIX_TEFFECTIVESTART_HOVERINGSTART] = self.Teffectivestart_hoveringstart # np.array(4, 4)
            json_result[MATRIX_TEFFECTIVESTART_HOVERINGEND]   = self.Teffectivestart_hoveringend   # np.array(4, 4)
            json_result[MATRIX_TEFFECTIVEEND_HOVERINGEND]     = self.Teffectiveend_hoveringend     # np.array(4, 4)
            
            
        return json_result

class SourceToolUsage(ToolUsage):
    def __init__(self, task, tool, goal, base_data_dir):
        super(SourceToolUsage, self).__init__(task, tool, goal, base_data_dir)
        self.base_data_dir = os.path.join(base_data_dir, tool, goal)
    
    """
    learn the usage
    """
    def step_2_update_Ttool_goal(self, sample):
        # get updated R
        v_start_world_frame = transformation_util.normalize(sample.get_v_world())
        v_start_goal_frame = transformation_util.normalize(sample.get_v_goal())
        v_start_goal_frame = np.array([[v_start_goal_frame[0], v_start_goal_frame[1], v_start_goal_frame[2], 1.]]).T
        v_start_world_frame_end = np.matmul(sample.get_Tworld_goalstart(), v_start_goal_frame)
        v_start_world_frame_end = v_start_world_frame_end.T[0][0:3]
        v_start_world_frame_start = np.matmul(sample.get_Tworld_goalstart(), np.array([[0., 0., 0., 1.]]).T)
        v_start_world_frame_start = v_start_world_frame_start.T[0][0:3]
        v_start_world_frame = v_start_world_frame_end - v_start_world_frame_start
        v_start_world_frame = transformation_util.normalize(v_start_world_frame)
        g = np.array([0., 0., -1.])
        R = np.identity(3)
        
        if not transformation_util.is_colinear(v_start_world_frame, g, error=0.01):
            x = v_start_world_frame
            z = g
            y = transformation_util.normalize(np.cross(z, x))
            z = transformation_util.normalize(np.cross(x, y))
            R = np.array([x, y, z]).T
        else:
            x = v_start_world_frame
            y = np.array([0., 1., 0.])
            z = transformation_util.normalize(np.cross(x, y))
            y = transformation_util.normalize(np.cross(z, x))
            R = np.array([x, y, z]).T
        
        # get updated Ttool_goal
        original_Tworld_goal = sample.get_Tworld_goalstart()
        original_p, original_R = transformation_util.decompose_homogeneous_transformation_matrix_to_rotation_matrix(original_Tworld_goal)
        
        updated_Tworld_goal = transformation_util.get_transformation_matrix_with_rotation_matrix(R, original_p)
        Tworld_tool = sample.get_Tworld_toolstart()
        
        Ttool_goal = np.matmul(transformation_util.get_transformation_matrix_inverse(Tworld_tool), updated_Tworld_goal)
        
        return Ttool_goal, v_start_world_frame
    
    def _get_default_screw_axis(self, Ttool_goal, axis):
        direction = axis
        point = Ttool_goal[:3, 3]
        angle = np.pi / 2.0
        T = tfs.rotation_matrix(angle, direction, point)
        
        S, theta = transformation_util.get_exponential_from_transformation_matrix(T)
        
        return S
    
    def step_2_categorize_Tool_goal(self, Ttool_goals, axes):
        removed_none_axes = []
        for each_axis in axes:
            if each_axis is not None:
                removed_none_axes.append(each_axis)
        
        axis = transformation_util.normalize(transformation_util.average_vector(removed_none_axes))
        
        print "[tool_usage][step_2_categorize_Tool_goal] axis: ", axis
        
        print "[tool_usage][step_2_categorize_Tool_goal] Ttool_goals"
        for Ttool_goal in Ttool_goals:
            print Ttool_goal
        
        origin = Ttool_goals[0]
        Ts = [transformation_util.get_fixed_frame_transformation(origin, i) for i in Ttool_goals]
        print "[tool_usage][step_2_categorize_Tool_goal]Ts:"
        for T in Ts:
            print T
        
        exponential_group = [transformation_util.get_exponential_from_transformation_matrix(i) for i in Ts]
        print "[tool_usage][step_2_categorize_Tool_goal]exponential group: "
        for group in exponential_group:
            S, theta = group
            print "S: ", S, "; theta: ", np.rad2deg(theta)
        
        angles = []
        S_screw_axis = []

        S_start, theta_start = exponential_group[0]
        angles.append(theta_start)
        exponential_group.pop(0) # the first 1 is the identity
        
        if axis is None:
            axis = exponential_group[0][0][:3]
        
        i = 0
        for (S, theta) in exponential_group:
            w = S[:3]
            if abs(theta) < np.deg2rad(5.):
                angles.append(theta)
                default_axis = self._get_default_screw_axis(origin, axis)
                S_screw_axis.append(default_axis)
            else:
                if transformation_util.is_similar_vector(w, axis, error = 5.0):
                    if not S is None:
                        angles.append(theta)
                        S_screw_axis.append(S)
                elif transformation_util.is_opposite_direction([w], [axis], error = 5.0):
                    if not S is None:
                        angles.append(np.pi*2 - theta)
                        S_screw_axis.append(-S)
                else:
                    print "{}'s sample axis {} too different from the average {}".format(i, w, axis)
            i += 1
        
        print "[tool_usage][step_2_categorize_Tool_goal]S_screw_axis"
        for S in S_screw_axis:
            print S
        
        averaged_S = transformation_util.average_screw_axis(S_screw_axis)
        
        print "[tool_usage][step_2_categorize_Tool_goal S: ", averaged_S
        print "[tool_usage][step_2_categorize_Tool_goal] angles: ", np.rad2deg(angles)
        
        angle_group = transformation_util.AngleGroup(angles)
        
        print "[tool_usage][step_2_categorize_Tool_goal]angle_group: "
        json_result = angle_group.to_json()
        print "angles: "
        print json_result["angles"]
        print "clustered_angle"
        print json_result["clustered_angle"]
        print "clustered_angle_range"
        print json_result["clustered_angle_range"]
        print "is_uniform_distribution"
        print json_result["is_uniform_distribution"]     
        
        self.Ttool_goal_group[tuple(averaged_S)] = [angle_group, origin]
    
    def step_3_process_contact_area(self, indices):
        all_indices = []
        for index in indices:
            all_indices += list(index)
        all_indices = list(set(all_indices))
        
        return all_indices
    
    def _reset_goal_trajectory(self, T1, trajectory, final_position, start_position=None):
        if start_position is None:
            start_position = np.array([T1[0, 3], T1[1, 3], T1[2, 3]])
        # transformation in the previous frame
        Ts = [transformation_util.get_transformation_matrix_from_exponential(i[0], i[1]) for i in trajectory]
        
        final_pose = T1
        pose_trajectory = []
        for T in Ts:
            final_pose = np.matmul(final_pose, T)
            pose_trajectory.append(deepcopy(final_pose))
        final_pose = pose_trajectory[self.Tend_index]       
            
        p, R = transformation_util.decompose_homogeneous_transformation_matrix_to_rotation_matrix(final_pose)
        start_p, R = transformation_util.decompose_homogeneous_transformation_matrix_to_rotation_matrix(T1)
        
        current_direction = p - start_p
        desired_direction = final_position - start_position
        scale = transformation_util.get_vector_length(final_position - start_position) / transformation_util.get_vector_length(p - start_p)
        
        rotation_matrix = transformation_util.get_rotation_matrix_from_directions([p - start_p], [final_position - start_p], axis = np.array([[0., 0., 1.]]))
        
        T = transformation_util.get_transformation_matrix_with_rotation_matrix(rotation_matrix, np.array([0., 0., 0.]))
        T_start = np.matmul(T, T1)
        T_start[0, 3] = start_position[0]
        T_start[1, 3] = start_position[1]
        T_start[2, 3] = start_position[2]

        new_trajectory = []
        for S, theta in trajectory:
            if transformation_util.is_screw_axis_translational(S):
                scaled_S = S
                theta *= scale
            else:
                scaled_S = transformation_util.rescale_S(S, scale, scale, scale)
            new_trajectory.append((scaled_S, theta))
        
        return new_trajectory
    
    def _find_Tend_trajectory(self, axes, angles, Tend):
        if not transformation_util.is_zero_vector(Tend[:3, 3], error=0.05):
            return Tend, -1
        
        Ts = [transformation_util.get_transformation_matrix_from_exponential(axes[i], angles[i]) for i in range(len(angles))]
        
        final_pose = np.identity(4)
        i = 0
        max_length = transformation_util.get_vector_length(Tend[:3, 3])
        max_length_T = np.identity(4)
        index = 0
        for T in Ts:
            final_pose = deepcopy(np.matmul(final_pose, T))
            p = final_pose[:3, 3]
            if transformation_util.get_vector_length(p) > max_length:
                max_length = transformation_util.get_vector_length(p)
                max_length_T = deepcopy(final_pose)
                index = i
            i += 1
        
        return max_length_T, index
        
    # trajectories: [[axes, angles, Tend], ...]
    def step_4_process_trajectory(self, trajectories):
        # exclude trajectories
        segment_count = {}
        for trajectory in trajectories:
            num_segments = len(trajectory[0])
            if num_segments in segment_count.keys():
                segment_count[num_segments].append(trajectory)
            else:
                segment_count[num_segments] = [trajectory]
        majority_trajectory_count = max(segment_count.keys())
        trajectories = segment_count[majority_trajectory_count]
        
        # no trajectory case, e.g., knocking
        if len(trajectories[0][0]) == 0:
            return
        
        is_interpolate = False
        # pose change general 
        if self.subtask_type == constants.TASK_SUBTYPE_POSE_CHANGE_GENERAL: # trajectory dependent on the goal end and can be obtained by interpolation
            is_interpolate = True
        print "[tool_usage][step_4_process_trajectory]is_interpolate: ", is_interpolate
        
        # just follow 1 path/screw axis/fragment. most tool use are in this category.
        if len(trajectories[0][0]) == 1:
            initial_axis, initial_angle, T = trajectories[0][0][0], trajectories[0][1][0], trajectories[0][2]
            T_axis, T_angle = transformation_util.get_exponential_from_transformation_matrix(T)
            
            axes = []
            thetas = []             
            #is_interpolate = False
            for trajectory in trajectories:
                axis = trajectory[0][0]
                angle = trajectory[1][0]
                T = trajectory[2]
                T_axis, T_angle = transformation_util.get_exponential_from_transformation_matrix(T)
                if transformation_util.is_opposite_sign(angle, initial_angle):
                    axis *= -1.
                    angle *= -1.
                if self.task_type == constants.TASK_TYPE_OTHER and not transformation_util.is_screw_axis_translational(axis) and abs(angle) > np.pi * 2:
                    self.is_circular = True
                #if (not self.is_circular) and (not is_interpolate) and (transformation_util.same_screw_axis(axis, T_axis, angle_error = 10.0, v_error = 10.0) or transformation_util.same_screw_axis(axis, -T_axis, angle_error = 10.0, v_error = 10.0)): # TODO: test this threshold
                    #is_interpolate = True
                axes.append(axis)
                thetas.append(angle)
            average_axis = transformation_util.average_screw_axis(axes)
            average_angle = np.mean(thetas)
            
            #print "[tool_usage][step_4_process_trajectory]is_interpolate: ", is_interpolate
            
            if not is_interpolate: # need to follow certain trajectory, not just fill in the trajectory by interpolation
                self.trajectory_axis.append(average_axis)
                self.trajectory_theta.append(average_angle)
            self.sample_averaged_trajectory_axis.append(average_axis)
            self.sample_averaged_trajectory_theta.append(average_angle)
            
            print "[tool_usage][step_4_process_trajectory]self.trajectory_axis"
            for i in range(len(self.trajectory_axis)):
                print "S: {}; theta: {}".format(self.trajectory_axis[i], self.trajectory_theta[i])
            
            return
        
        # more than 1
        self.Tend, self.Tend_index = self._find_Tend_trajectory(trajectories[0][0], trajectories[0][1], trajectories[0][2])
        print "[tool_usage][step_4_process_trajectory]self.Tend_index = {}".format(self.Tend_index)
        standard_pend = np.array([self.Tend[0, 3], self.Tend[1, 3], self.Tend[2, 3]])
        axes = [[i] for i in trajectories[0][0]]
        angles = [[i] for i in trajectories[0][1]]
        for index in range(1, len(trajectories)):
            current_axes, current_angles, current_Tend = trajectories[index]
            current_trajectory = [(current_axes[i], current_angles[i]) for i in range(len(current_axes))]
            updated_trajectory = self._reset_goal_trajectory(np.identity(4), current_trajectory, standard_pend)          
            for i in range(len(updated_trajectory)):
                axes[i].append(updated_trajectory[i][0])
                angles[i].append(updated_trajectory[i][1])

        for axis in axes:
            self.trajectory_axis.append(transformation_util.average_screw_axis(axis))
            self.sample_averaged_trajectory_axis.append(transformation_util.average_screw_axis(axis))
        for angle in angles:
            self.trajectory_theta.append(np.mean(angle))
            self.sample_averaged_trajectory_theta.append(np.mean(angle))
        
        for i in range(len(axes)):
            print "[tool_usage][step_4_process_trajectory]each sample in axis {}".format(i)
            for j in range(len(axes[i])):
                print "S = {}, theta = {}".format(axes[i][j], angles[i][j])
                
        print "[tool_usage][step_4_process_trajectory]self.trajectory_axis"
        for i in range(len(self.trajectory_axis)):
            print "S: {}; theta: {}".format(self.trajectory_axis[i], self.trajectory_theta[i])        
        
    def step_5_process_hovering(self, Teffectivestart_hoveringstarts, Teffectivestart_hoveringends, Teffectiveend_hoveringends):
        self.Teffectivestart_hoveringstart = transformation_util.average_transformation_matrix(Teffectivestart_hoveringstarts)
        self.Teffectivestart_hoveringend = transformation_util.average_transformation_matrix(Teffectivestart_hoveringends)
        self.Teffectiveend_hoveringend = transformation_util.average_transformation_matrix(Teffectiveend_hoveringends)
    
    def process_learned_samples(self):
        self.clear_learned_data()
        
        # step 1: get the task type
        # a. get the general task type
        self.task_type = self.step_1_find_task_type_from_samples()
        # b. get the sub task
        if self.task_type == constants.TASK_TYPE_POSE_CHANGE:
            self.subtask_type = self.step_1_find_subtask_type_from_samples()
        
        print "task type: ", self.task_type
        print "subtask type: ", self.subtask_type
        
        # step 2: learn Ttool_goal
        Ts = []
        axes = []
        for sample in self.samples:
            Ttool_goal = sample.get_reference_point()
            if self.subtask_type == constants.TASK_SUBTYPE_POSE_CHANGE_GENERAL:
                Ttool_goal, axis = self.step_2_update_Ttool_goal(sample)
            Ts.append(Ttool_goal)
            axes.append(sample.get_v_tool())  
        self.step_2_categorize_Tool_goal(Ts, axes)
        
        # step 3: learn the contact area
        tool_contact_area_indices = []
        goal_contact_area_indices = []
        for sample in self.samples:
            tool_contact_area_indices.append(sample.get_tool_contact_area_index())
            goal_contact_area_indices.append(sample.get_goal_contact_area_index())
        self.tool_contact_area_index = self.step_3_process_contact_area(tool_contact_area_indices)
        self.goal_contact_area_index = self.step_3_process_contact_area(goal_contact_area_indices)

        # step 4: learn the trajectory
        trajectories = []
        for sample in self.samples:
            trajectory_axis = sample.get_trajectory_screw_axes()
            trajectory_angle = sample.get_trajectory_thetas()
            Teffectivesstart_unhoveringstart = sample.get_Teffectivestart_unhoveringstart()
            trajectories.append([trajectory_axis, trajectory_angle, Teffectivesstart_unhoveringstart])
        self.step_4_process_trajectory(trajectories)
        
        # step 5: learn the hovering/unhovering
        Teffectivestart_hoveringstarts = []
        Teffectivestart_hoveringends = []
        Teffectiveend_hoveringends = []
        for sample in self.samples:
            Teffectivestart_hoveringstarts.append(sample.get_Teffectivestart_hoveringstart())
            Teffectivestart_hoveringends.append(sample.get_Teffectivestart_hoveringend())
            Teffectiveend_hoveringends.append(sample.get_Teffectiveend_unhoveringend())
        self.step_5_process_hovering(Teffectivestart_hoveringstarts, Teffectivestart_hoveringends, Teffectiveend_hoveringends)

        self.is_learned = True

class SubstituteToolUsage(ToolUsage):
    def __init__(self, task, tool, goal, base_data_dir, source_usage, is_control=False):
        super(SubstituteToolUsage, self).__init__(task, tool, goal, base_data_dir)
        self.source_usage = source_usage
        self.task_type = self.source_usage.get_task_type()
        self.subtask_type = self.source_usage.get_subtask_type()
        self.source_tool = source_usage.get_tool_name()
        self.source_goal = source_usage.get_goal_name()
        self.trajectory_rescale = self.get_trajectory_rescale()
        self.calculated_Tsourcetool_substitutetool = None
        self.calculated_Tsourcegoal_substitutegoal = None
        
        self.initialize_usage_with_source_tool()
        
        self.get_calculated_Tsourcetool_substitutetool(is_control=is_control)
        self.get_calculated_Tsourcegoal_substitutegoal(is_control=is_control)
    
    # assuming using the source tool on the sub goal
    def initialize_usage_with_source_tool(self):
        # copy
        self.Ttool_goal_group = self.source_usage.Ttool_goal_group
        
        self.trajectory_axis = self.source_usage.trajectory_axis
        self.trajectory_theta = self.source_usage.trajectory_theta
        self.sample_averaged_trajectory_axis = self.source_usage.sample_averaged_trajectory_axis
        self.sample_averaged_trajectory_theta = self.source_usage.sample_averaged_trajectory_theta
        self.Tend = self.source_usage.Tend
        self.Tend_index = self.source_usage.Tend_index
        self.is_circular = self.source_usage.is_circular
        
        self.Teffectivestart_hoveringstart = self.source_usage.Teffectivestart_hoveringstart
        self.Teffectivestart_hoveringend = self.source_usage.Teffectivestart_hoveringend
        self.Teffectiveend_hoveringend = self.source_usage.Teffectiveend_hoveringend
        
        # update trajectory when necessary
        if self.task_type == constants.TASK_TYPE_OTHER:
            source_goal_dimension = 1. * max(pointcloud_util.get_goal_mesh(self.source_goal).get_oriented_bounding_box().extent)
            sub_goal_dimension = 1. * max(pointcloud_util.get_goal_mesh(self.goal).get_oriented_bounding_box().extent)
            scale = sub_goal_dimension / source_goal_dimension
            for i in range(len(self.trajectory_axis)):
                S = self.trajectory_axis[i]
                if transformation_util.is_screw_axis_translational(S):
                    self.trajectory_theta[i] *= scale
                else:
                    S = transformation_util.rescale_S(S, scale, scale, scale)
                    self.trajectory_axis[i] = S
            for i in range(len(self.sample_averaged_trajectory_axis)):
                S = self.sample_averaged_trajectory_axis[i]
                if transformation_util.is_screw_axis_translational(S):
                    self.sample_averaged_trajectory_theta[i] *= scale
                else:
                    S = transformation_util.rescale_S(S, scale, scale, scale)
                    self.sample_averaged_trajectory_axis[i] = S
    
    def _get_Tworld_sourcegoal(Tworld_subgoal):
        if Tworld_subgoal is None:
            return None
        
        Tworld_sourcegoal = deepcopy(Tworld_subgoal)
        if self.subtask_type == constants.TASK_SUBTYPE_POSE_CHANGE_GENERAL:
            return Tworld_sourcegoal

        return self._get_Tworld_sourcegoal_helper(Tworld_subgoal)  
    
    def get_usage(self, subgoal_Tworld_goalstart, subgoal_Tworld_goalend=None, num_circles=-1):       
        sourcegoal_Tworld_goalstart = np.matmul(subgoal_Tworld_goalstart,
                                                transformation_util.get_transformation_matrix_inverse(self.calculated_Tsourcegoal_substitutegoal))
        
        sourcegoal_Tworld_goalend = None
        if not subgoal_Tworld_goalend is None:
            sourcegoal_Tworld_goalend = np.matmul(subgoal_Tworld_goalend, 
                                                  transformation_util.get_transformation_matrix_inverse(self.calculated_Tsourcegoal_substitutegoal))
        
        source_tool_trajectories_world_frame = self.source_usage.get_usage(sourcegoal_Tworld_goalstart, 
                                                                           sourcegoal_Tworld_goalend, 
                                                                           num_circles, is_sub=True, 
                                                                           sub_tool_name=self.tool, 
                                                                           sub_goal_name=self.goal, 
                                                                           Tsourcetool_subtool=deepcopy(self.calculated_Tsourcetool_substitutetool), 
                                                                           Tworld_subgoal=deepcopy(subgoal_Tworld_goalstart),
                                                                           functional_component_rescale=self.trajectory_rescale)
        
        sub_tool_trajectories_world_frame = []
        
        print "[tool_usage][get_usage] self.calculated_Tsourcetool_substitutetool: "
        print self.calculated_Tsourcetool_substitutetool
        
        print "[tool_usage][get_usage] calculate Tsub_tool START"
        for source_tool_trajectory_world_frame in source_tool_trajectories_world_frame:
            sub_tool_trajectory_world_frame = []
            for Tsourcetool in source_tool_trajectory_world_frame:               
                Tsubtool = np.matmul(Tsourcetool, self.calculated_Tsourcetool_substitutetool)
                sub_tool_trajectory_world_frame.append(Tsubtool)                
            sub_tool_trajectories_world_frame.append(deepcopy(sub_tool_trajectory_world_frame))
        
        print "[tool_usage][SubstituteToolUsage][get_usage] visualize trajectory"
        pointcloud_util.visualize_trajectory(self.tool, self.goal, sub_tool_trajectories_world_frame, subgoal_Tworld_goalstart, subgoal_Tworld_goalend)
            
        source_tool_trajectory_goal_frame = [np.matmul(transformation_util.get_transformation_matrix_inverse(sourcegoal_Tworld_goalstart), Tworld_sourcetool) for Tworld_sourcetool in source_tool_trajectories_world_frame[0]]
        visualize(self.tool, self.goal, 0, sourcegoal_Tworld_goalstart, sourcegoal_Tworld_goalstart, source_tool_trajectory_goal_frame, self.calculated_Tsourcetool_substitutetool, self.calculated_Tsourcegoal_substitutegoal)
        
        return sub_tool_trajectories_world_frame

    def get_trajectory_rescale(self):
        rescale = 1.0
        
        if self.task_type == constants.TASK_TYPE_OTHER:
            source_dimension = pointcloud_util.get_object_bb_dimension(self.source_goal, constants.OBJECT_TYPE_GOAL)
            sub_dimension = pointcloud_util.get_object_bb_dimension(self.goal, constants.OBJECT_TYPE_GOAL)
            rescale = sub_dimension / source_dimension
        
        return rescale

    def get_calculated_Tsourcegoal_substitutegoal(self, is_control=False):
        if self.goal == self.source_goal:
            self.calculated_Tsourcegoal_substitutegoal = np.identity(4)
            return
        
        if self.subtask_type == constants.TASK_SUBTYPE_POSE_CHANGE_GENERAL:
            self.calculated_Tsourcegoal_substitutegoal = np.identity(4)
            return
        
        contact_pnt_idx = self.source_usage.get_goal_contact_area_index()
        if is_control:
            self.calculated_Tsourcegoal_substitutegoal, _ = pointcloud_util.calculate_Tsource_substitute(self.source_goal, self.goal, contact_pnt_idx, constants.OBJECT_TYPE_GOAL, task=self.task, is_control=True)
        else:
            self.calculated_Tsourcegoal_substitutegoal, _ = pointcloud_util.calculate_Tsource_substitute(self.source_goal, self.goal, contact_pnt_idx, constants.OBJECT_TYPE_GOAL, task=self.task)

    # Get the pose of the aruco of the substitute tool in the frame of the aruco of the source tool
    def get_calculated_Tsourcetool_substitutetool(self, is_control=False):
        if self.tool == self.source_tool:
            self.calculated_Tsourcetool_substitutetool = np.identity(4)
            return

        contact_pnt_idx = self.source_usage.get_tool_contact_area_index()
        if is_control:
            self.calculated_Tsourcetool_substitutetool, _ = pointcloud_util.calculate_Tsource_substitute(self.source_tool, self.tool, contact_pnt_idx, constants.OBJECT_TYPE_TOOL, task=self.task, is_control=True)
        else:
            self.calculated_Tsourcetool_substitutetool, _ = pointcloud_util.calculate_Tsource_substitute(self.source_tool, self.tool, contact_pnt_idx, constants.OBJECT_TYPE_TOOL, task=self.task)

def visualize(sub_tool, sub_goal, sample_index, Tworld_sourcegoalstart, Tworld_sourcegoalend, Tworld_sourcetooltrajectory_goalframe, Tsourcetool_subtool, Tsourcegoal_subgoal):
    Tworld_substitutegoalstart = np.matmul(Tworld_sourcegoalstart, Tsourcegoal_subgoal)
    Tworld_substitutegoalend   = np.matmul(Tworld_sourcegoalend,   Tsourcegoal_subgoal)
    
    Tworld_sourcetooltrajectory_worldframe = []
    for Tgoal_tool in Tworld_sourcetooltrajectory_goalframe:
        Tworld_tool = np.matmul(Tworld_sourcegoalstart, Tgoal_tool)
        Tworld_sourcetooltrajectory_worldframe.append(Tworld_tool)
    
    Tworld_subtooltrajectory_worldframe = []
    for Tworld_sourcetool in Tworld_sourcetooltrajectory_worldframe:
        Tworld_subtool = np.matmul(Tworld_sourcetool, Tsourcetool_subtool)
        Tworld_subtooltrajectory_worldframe.append(Tworld_subtool)
    
    pcs = []
    goalstart = get_pc(sub_goal, constants.get_goal_mesh_path, np.array([0., 1., 0.]), Tworld_substitutegoalstart)
    goalend   = get_pc(sub_goal, constants.get_goal_mesh_path, np.array([0., 0., 1.]), Tworld_substitutegoalend)
    pcs.append(goalstart)
    pcs.append(goalend)
    for Tworld_subtool in Tworld_subtooltrajectory_worldframe:
        tool = get_pc(sub_tool, constants.get_tool_mesh_path, np.array([1., 0., 0.]), Tworld_subtool)
        pcs.append(tool)
    
    o3d.visualization.draw_geometries(pcs, "{}-{} index: {}".format(sub_tool, sub_goal, sample_index))

def get_pc(name, func, color, T):
    mesh_path = func(name)
    
    pc = o3d.io.read_point_cloud(mesh_path)
    
    pc.paint_uniform_color(color)
    pc.transform(T)

    return pc