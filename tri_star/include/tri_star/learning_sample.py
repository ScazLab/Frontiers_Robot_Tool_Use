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
import json

import numpy as np

from tri_star import transformation_util
from tri_star import file_util
from tri_star import pointcloud_util
from tri_star.constants import PERCEPTION_METHOD_ARUCO, PERCEPTION_METHOD_POINTCLOUD
from tri_star.constants import TASK_TYPE_POSE_CHANGE, TASK_TYPE_OTHER
from tri_star.constants import get_package_dir, get_perception_method

# strings to use to save files, in the format of <variable type>_<name>, :p
# task related
STR_TASK        = "task"
STR_TOOL        = "tool"
STR_GOAL        = "goal"
STR_TASK_TYPE   = "task_type"
STR_SAMPLE_PATH = "sample file path"
STR_RAW_SAMPLE_PATH = "raw sample file path"

# perceived variables
MATRIX_GOALSTART_WORLD_FRAME       = "Tworld_goalstart"
MATRIX_GOALEND_WORLD_FRAME         = "Tworld_goalend"
LIST_MATRIX_TRAJECTORY_WORLD_FRAME = "tool_trajectory_world_frame"

# variables for learning for all tasks
MATRIX_TOOL_GOAL                            = "Ttool_goal"
LIST_MATRIX_TRAJECTORY_EFFECTIVESTART_FRAME = "trajectory"
LIST_ARRAY_TRAJECTORY_AXES                  = "trajectory_screw_axes"
LIST_FLOAT_TRAJECTORY_THETAS                = "trajectory_thetas"
LIST_INT_TOOL_CONTACT_INDEX                 = "tool_contact_area_index"
LIST_INT_GOAL_CONTACT_INDEX                 = "goal_contact_area_index"
MATRIX_EFFECTIVESTART_HOVERINGSTART         = "Teffectivestart_hoveringstart"
MATRIX_EFFECTIVESTART_HOVERINGEND           = "Teffectivestart_hoveringend"
MATRIX_EFFECTIVESTART_UNHOVERINGSTART       = "Teffectivestart_unhoveringstart"
MATRIX_EFFECTIVEEND_UNHOVERINGEND           = "Teffectiveend_unhoveringend"
# variables for learning for tasks with pose changes
ARRAY_EFFECT_V_WORLD_FRAME                  = "v_start_world_frame"
ARRAY_EFFECT_V_GOAL_FRAME                   = "v_start_goal_frame"
ARRAY_EFFECT_V_TOOL_FRAME                   = "v_start_tool_frame"

# other calculated variables
MATRIX_TOOLEND_WORLD_FRAME         = "Tworld_toolend"
MATRIX_TOOLSTART_WORLD_FRAME       = "Tworld_toolstart"
LIST_MATRIX_TRAJECTORY_GOAL_FRAME  = "tool_trajectory_goal_frame"
LIST_INT_CHANGING_POINT            = "changing_point"
MATRIX_TRAJECTORY_HOVERING_START   = "hovering_start"
MATRIX_TRAJECTORY_HOVERING_END     = "hovering_end"
MATRIX_TRAJECTORY_UNHOVERING_START = "unhovering_start"
MATRIX_TRAJECTORY_UNHOVERING_END   = "unhovering_end"

# File template
FILENAME_SAMPLE = "sample_{}.txt"
FILENAME_RAW_SAMPLE = "raw_sample_{}.txt"
DIR_RAW_SAMPLE = "raw"
DIR_SAMPLE = "sample"

"""
Get a learning sample
"""
class Sample:
    def __init__(self, task, tool, goal, base_data_dir):
        # get task related information
        self.task = task
        self.tool = tool
        self.goal = goal
        self.task_type = None # whether it is with pose changes or not
        self.base_data_dir = base_data_dir
        
        # Perceived variables
        self.Tworld_goalstart = None
        self.Tworld_goalend = None
        self.tool_trajectory_world_frame = []
        
        # Calculated variables. The following doesn't need to be saved, needed for calculation purposes only
        self.Tworld_toolend = None
        self.Tworld_toolstart = None
        self.tool_trajectory_goal_frame = [] # tool trajectory in the goal frame
        self.changing_point = [1, -2] # index on the trajectory
        #self.effective_start = None # Tgoal_tool: the tool's pose that just start to touch the object            
        self.hovering_start = None # in the goal frame
        self.hovering_end = None # in the goal frame
        self.unhovering_start = None # in the goal frame
        self.unhovering_end = None # in the goal frame      
        
        # Calculated variables needed for learning
        # The following are the change from effect_start to certain location in the goal frame.
        self.Ttool_goal = None # the effect_start frame: reference point
        self.trajectory = [] # Teffectivestart_tool in the self.effect_start frame, starting from effective start, and ends with effective end
        self.trajectory_screw_axes = [] # a list of screw axis, in the self.effect_start frame
        self.trajectory_thetas = [] # a list of float
        self.tool_contact_area_index = [] # a list of int
        self.goal_contact_area_index = [] # a list of int
        self.Teffectivestart_hoveringstart = None # the tool hoveringstart position in the effective start tool frame
        self.Teffectivestart_hoveringend = None
        self.Teffectivestart_unhoveringstart = None
        self.Teffectiveend_unhoveringend = None        
        # effect for TASK - pose change, v_start is the goal pose changes
        self.v_start_world_frame = None # v_start in the world frame
        self.v_start_goal_frame = None # v_start in the goal frame. The initial speed change. The old self.v_start_goal.
        self.v_start_tool_frame = None # v_start in the tool frame. The old self.v_start_s1
        
        # file location:
        self.file_path = "" # raw file path
    
    # The main function to call to analyze the current sample
    def process_sample(self, Tworld_goalstart, Tworld_goalend, tool_trajectory_world_frame):
        self.Tworld_goalstart = Tworld_goalstart.copy()
        self.Tworld_goalend = Tworld_goalend.copy()
        self.Tworld_toolend = tool_trajectory_world_frame[self.changing_point[-1]].copy()
      
        self.tool_trajectory_world_frame = [i.copy() for i in tool_trajectory_world_frame]
        self.tool_trajectory_world_frame = transformation_util.merged_trajectory(self.tool_trajectory_world_frame)
        self.tool_trajectory_goal_frame = [np.matmul(transformation_util.get_transformation_matrix_inverse(self.Tworld_goalstart), i.copy()) for i in tool_trajectory_world_frame]
        
        # check what type of task it is. Whether it is pose change or not
        if transformation_util.same_pose(self.Tworld_goalstart, self.Tworld_goalend, translation_threshold=0.03) or self.Tworld_goalend is None:
            self.task_type = TASK_TYPE_OTHER
        else:
            self.task_type = TASK_TYPE_POSE_CHANGE
        
        # initial process of the trajectory. The trajectory has already been converted to goal frame
        self.process_trajectory(self.tool_trajectory_goal_frame)
        
        self.get_goalend()

    # if the perception method is aruco, Tgoal_tool is Tgoalaruco_toolaruco
    # if the perception method is pointcloud, Tgoal_tool is Tgoalpointcloud_toolpointcloud
    def calc_contact_area(self, Tgoal_tool, r=1.1):
        """
        Calculate contact area on Src tool with goal object.
        """
        if get_perception_method() == PERCEPTION_METHOD_ARUCO:
            Tgoal_goalaruco = perception_util.get_Tgoal_aruco(self.get_goal_name())
            Ttool_toolaruco = perception_util.get_Ttool_aruco(self.get_tool_name())
            Tgoal_tool = np.matmul(Tgoal_goalaruco,
                               np.matmul(Tgoalaruco_toolaruco,
                                         transformation_util.get_transformation_matrix_inverse(Ttool_toolaruco)))
        elif get_perception_method() == PERCEPTION_METHOD_POINTCLOUD:
            Tgoal_tool = Tgoal_tool        

        tool_contact_area_index, goal_contact_area_index = pointcloud_util.contact_surface(self.get_tool_name(), self.get_goal_name(), Tgoal_tool)
        return tool_contact_area_index, goal_contact_area_index

    """
    functions used to process the trajectory
    """
    def process_trajectory(self, trajectory): # initially provided list of tool transformation matrices in the goal start frame
        self.trajectory = trajectory[self.changing_point[0]:(self.changing_point[-1] + 1)]
        
        # Recognize the start and end phase
        self.hovering_start = trajectory[0]
        self.hovering_end = trajectory[self.changing_point[0]]
        self.unhovering_start = trajectory[self.changing_point[-1]]
        self.unhovering_end = trajectory[-1]
    
    def get_goalend(self):
        self.Ttool_goal = np.identity(4)
        self.Tworld_toolstart = np.identity(4)
        if self.task_type == TASK_TYPE_POSE_CHANGE:
            self.Ttool_goal = np.matmul(transformation_util.get_transformation_matrix_inverse(self.Tworld_toolend), self.Tworld_goalend)
            self.Tworld_toolstart = np.matmul(self.Tworld_goalstart, transformation_util.get_transformation_matrix_inverse(self.Ttool_goal))
        elif self.task_type == TASK_TYPE_OTHER:
            self.Tworld_toolstart = self.hovering_end.copy() 
            Tgoal_tool = self.hovering_end.copy() # self.hovering_end is in the goalstart frame
            self.Ttool_goal = transformation_util.get_transformation_matrix_inverse(Tgoal_tool)
            self.Tworld_toolstart = np.matmul(self.Tworld_goalstart, Tgoal_tool)
        
        # find the precise effective stage start phase
        # convert the trajectory to the effective_start/tool frame
        self.effective_start = transformation_util.get_transformation_matrix_inverse(self.Ttool_goal) # Tgoal_tool
        self.trajectory[0] = self.effective_start
        self.trajectory = [np.matmul(transformation_util.get_transformation_matrix_inverse(self.effective_start), point) for point in self.trajectory] # convert to tool frame, the effective start as the reference point
        
        # analyze the trajectory
        self.trajectory_screw_axes, self.trajectory_thetas = transformation_util.category_trajectory(self.trajectory)
        
        # Get the contact area
        self.tool_contact_area_index, self.goal_contact_area_index = self.calc_contact_area(self.effective_start, r=1.1)
        
        # in effect_start/tool frame
        self.Teffectivestart_hoveringstart = np.matmul(self.Ttool_goal, self.hovering_start)
        self.Teffectivestart_hoveringend = np.matmul(self.Ttool_goal, self.hovering_end)
        self.Teffectivestart_unhoveringstart = np.matmul(self.Ttool_goal, self.unhovering_start)
    
        # in unhovering_end frame
        self.Teffectiveend_unhoveringend = np.matmul(transformation_util.get_transformation_matrix_inverse(self.unhovering_start), self.unhovering_end)              
        
        if len(self.trajectory_screw_axes) != 0:
            S_tool_frame = self.trajectory_screw_axes[0]
    
            Ttoolframe = self.Tworld_toolstart.copy()
            Tgoalframe = self.Tworld_goalstart.copy()
            Tworldframe = np.identity(4)
            
            S_world_frame = transformation_util.change_twist_frame(S_tool_frame, Ttoolframe, Tworldframe)
            S_goal_frame = transformation_util.change_twist_frame(S_tool_frame, Ttoolframe, Tgoalframe)
            
            self.v_start_world_frame = np.array([S_world_frame[3], S_world_frame[4], S_world_frame[5]])
            self.v_start_goal_frame = np.array([S_goal_frame[3], S_goal_frame[4], S_goal_frame[5]])
            self.v_start_tool_frame = np.array([S_tool_frame[3], S_tool_frame[4], S_tool_frame[5]])
        
    """
    getters
    """
    def get_task_type(self):
        return self.task_type
    
    def get_goal_name(self):
        return self.goal
    
    def get_tool_name(self):
        return self.tool    
    
    def get_Tworld_goalstart(self):
        return self.Tworld_goalstart

    def get_Tworld_goalend(self):
        return self.Tworld_goalend

    def get_Tworld_toolstart(self):
        return self.Tworld_toolstart

    def get_Tworld_toolend(self):
        return self.Tworld_toolend

    def get_reference_point(self):
        return self.Ttool_goal
    
    def get_trajectory(self):
        return self.trajectory # in the effective start frame, the trajectory of the tool while actually make effect on the target object
    
    def get_tool_trajectory_goal_frame(self):
        return self.tool_trajectory_goal_frame
        
    def get_trajectory_screw_axes(self):
        return self.trajectory_screw_axes
    
    def get_trajectory_thetas(self):
        return self.trajectory_thetas    
    
    def get_tool_contact_area_index(self):
        return self.tool_contact_area_index
    
    def get_goal_contact_area_index(self):
        return self.goal_contact_area_index    

    def get_Teffectivestart_hoveringstart(self):
        return self.Teffectivestart_hoveringstart
    
    def get_Teffectivestart_hoveringend(self):
        return self.Teffectivestart_hoveringend
    
    def get_Teffectivestart_unhoveringstart(self):
        return self.Teffectivestart_unhoveringstart
    
    def get_Teffectiveend_unhoveringend(self):
        return self.Teffectiveend_unhoveringend    

    def get_v_world(self):
        return self.v_start_world_frame

    def get_v_goal(self):
        return self.v_start_goal_frame
    
    def get_v_tool(self):
        return self.v_start_tool_frame
    
    def get_file_name(self):
        return self.file_path
    
    """
    I/O, save this sample, or read a learned sample
    """
    def save(self):
        sample_dir = os.path.join(get_package_dir(), self.base_data_dir, DIR_SAMPLE)
        raw_sample_dir = os.path.join(get_package_dir(), self.base_data_dir, DIR_RAW_SAMPLE)
        file_util.create_dir(sample_dir)
        file_util.create_dir(raw_sample_dir)            
        
        relative_sample_dir = os.path.join(self.base_data_dir, DIR_SAMPLE)
        relative_raw_sample_dir = os.path.join(self.base_data_dir, DIR_RAW_SAMPLE)
        
        sample_file_index = file_util.get_index_in_dir(sample_dir, file_util.get_re_from_file_name(FILENAME_SAMPLE))
        sample_file_path = os.path.join(sample_dir, FILENAME_SAMPLE.format(sample_file_index))
        relative_sample_file_path = os.path.join(relative_sample_dir, FILENAME_SAMPLE.format(sample_file_index))

        raw_sample_file_index = file_util.get_index_in_dir(raw_sample_dir, file_util.get_re_from_file_name(FILENAME_RAW_SAMPLE))
        raw_sample_file_path = os.path.join(raw_sample_dir, FILENAME_RAW_SAMPLE.format(raw_sample_file_index))
        relative_raw_sample_file_path = os.path.join(relative_raw_sample_dir, FILENAME_RAW_SAMPLE.format(raw_sample_file_index))
    
        sample_file_content = self.get_json(is_elaborate=True)
        sample_file_content[STR_SAMPLE_PATH]     = relative_sample_file_path     # str
        sample_file_content[STR_RAW_SAMPLE_PATH] = relative_raw_sample_file_path # str
        with open(sample_file_path, "w") as write_file:
            json.dump(sample_file_content, write_file, indent = 4, cls=file_util.ToolUseEncoder)
        
        raw_sample_file_content = self.get_json(is_elaborate=False)
        raw_sample_file_content[STR_SAMPLE_PATH]     = relative_sample_file_path     # str
        raw_sample_file_content[STR_RAW_SAMPLE_PATH] = relative_raw_sample_file_path # str
        with open(raw_sample_file_path, "w") as write_file:
            json.dump(sample_file_content, write_file, indent = 4, cls=file_util.ToolUseEncoder)
        
        self.file_path = relative_sample_file_path
    
    def read(self, sample_file_path, recalculate=False):
        relative_sample_file_path = file_util.read_variable(sample_file_path, STR_SAMPLE_PATH)
        self.file_path = relative_sample_file_path
        
        # perceived data
        self.task      = file_util.read_variable(sample_file_path, STR_TASK)
        self.tool      = file_util.read_variable(sample_file_path, STR_TOOL)
        self.goal      = file_util.read_variable(sample_file_path, STR_GOAL)
        
        self.Tworld_goalstart            = file_util.read_variable(sample_file_path, MATRIX_GOALSTART_WORLD_FRAME, variable_type = file_util.TYPE_NUMPY)
        self.Tworld_goalend              = file_util.read_variable(sample_file_path, MATRIX_GOALEND_WORLD_FRAME, variable_type = file_util.TYPE_NUMPY)
        self.tool_trajectory_world_frame = file_util.read_variable(sample_file_path, LIST_MATRIX_TRAJECTORY_WORLD_FRAME, variable_type = file_util.TYPE_NUMPY, variable_collection_type=file_util.TYPE_LIST)
        
        if recalculate:
            self.process_sample(self.Tworld_goalstart, self.Tworld_goalend, self.tool_trajectory_world_frame)
        else:
            # calculated data
            self.Tworld_toolend             = file_util.read_variable(sample_file_path, MATRIX_TOOLEND_WORLD_FRAME, variable_type = file_util.TYPE_NUMPY)
            self.Tworld_toolstart           = file_util.read_variable(sample_file_path, MATRIX_TOOLSTART_WORLD_FRAME, variable_type = file_util.TYPE_NUMPY)
            self.tool_trajectory_goal_frame = file_util.read_variable(sample_file_path, LIST_MATRIX_TRAJECTORY_GOAL_FRAME, variable_type = file_util.TYPE_NUMPY, variable_collection_type=file_util.TYPE_LIST)
            self.changing_point             = file_util.read_variable(sample_file_path, LIST_INT_CHANGING_POINT)   
            self.hovering_start             = file_util.read_variable(sample_file_path, MATRIX_TRAJECTORY_HOVERING_START, variable_type = file_util.TYPE_NUMPY)
            self.hovering_end               = file_util.read_variable(sample_file_path, MATRIX_TRAJECTORY_HOVERING_END, variable_type = file_util.TYPE_NUMPY)
            self.unhovering_start           = file_util.read_variable(sample_file_path, MATRIX_TRAJECTORY_UNHOVERING_START, variable_type = file_util.TYPE_NUMPY)
            self.unhovering_end             = file_util.read_variable(sample_file_path, MATRIX_TRAJECTORY_UNHOVERING_END, variable_type = file_util.TYPE_NUMPY)        
            
            # variables needed for tool use
            self.task_type = file_util.read_variable(sample_file_path, STR_TASK_TYPE)
    
            self.Ttool_goal                      = file_util.read_variable(sample_file_path, MATRIX_TOOL_GOAL, variable_type = file_util.TYPE_NUMPY)
            self.trajectory                      = file_util.read_variable(sample_file_path, LIST_MATRIX_TRAJECTORY_EFFECTIVESTART_FRAME, variable_type = file_util.TYPE_NUMPY, variable_collection_type=file_util.TYPE_LIST)
            self.trajectory_screw_axes           = file_util.read_variable(sample_file_path, LIST_ARRAY_TRAJECTORY_AXES, variable_type = file_util.TYPE_NUMPY, variable_collection_type=file_util.TYPE_LIST)
            self.trajectory_thetas               = file_util.read_variable(sample_file_path, LIST_FLOAT_TRAJECTORY_THETAS)
            self.tool_contact_area_index         = file_util.read_variable(sample_file_path, LIST_INT_TOOL_CONTACT_INDEX)
            self.goal_contact_area_index         = file_util.read_variable(sample_file_path, LIST_INT_GOAL_CONTACT_INDEX)
            self.Teffectivestart_hoveringstart   = file_util.read_variable(sample_file_path, MATRIX_EFFECTIVESTART_HOVERINGSTART, variable_type = file_util.TYPE_NUMPY)
            self.Teffectivestart_hoveringend     = file_util.read_variable(sample_file_path, MATRIX_EFFECTIVESTART_HOVERINGEND, variable_type = file_util.TYPE_NUMPY)
            self.Teffectivestart_unhoveringstart = file_util.read_variable(sample_file_path, MATRIX_EFFECTIVESTART_UNHOVERINGSTART, variable_type = file_util.TYPE_NUMPY)
            self.Teffectiveend_unhoveringend     = file_util.read_variable(sample_file_path, MATRIX_EFFECTIVEEND_UNHOVERINGEND, variable_type = file_util.TYPE_NUMPY)
            ## for pose changes task
            #if self.task_type == TASK_TYPE_POSE_CHANGE:
            self.v_start_world_frame = file_util.read_variable(sample_file_path, ARRAY_EFFECT_V_WORLD_FRAME, variable_type = file_util.TYPE_NUMPY)
            self.v_start_goal_frame  = file_util.read_variable(sample_file_path, ARRAY_EFFECT_V_GOAL_FRAME, variable_type = file_util.TYPE_NUMPY)
            self.v_start_tool_frame  = file_util.read_variable(sample_file_path, ARRAY_EFFECT_V_TOOL_FRAME, variable_type = file_util.TYPE_NUMPY)

    def get_json(self, is_elaborate = False):
        json_result = {}
        
        # perception data
        json_result[STR_TASK] = self.task # str
        json_result[STR_TOOL] = self.tool # str
        json_result[STR_GOAL] = self.goal # str
        
        json_result[MATRIX_GOALSTART_WORLD_FRAME]       = self.Tworld_goalstart            # np.array(4, 4)
        json_result[MATRIX_GOALEND_WORLD_FRAME]         = self.Tworld_goalend              # np.array(4, 4)
        json_result[LIST_MATRIX_TRAJECTORY_WORLD_FRAME] = self.tool_trajectory_world_frame # [np.array(4, 4)]
        
        if is_elaborate:
            # calculated variables
            json_result[MATRIX_TOOLEND_WORLD_FRAME]         = self.Tworld_toolend             # np.array(4, 4)
            json_result[MATRIX_TOOLSTART_WORLD_FRAME]       = self.Tworld_toolstart           # np.array(4, 4)
            json_result[LIST_MATRIX_TRAJECTORY_GOAL_FRAME]  = self.tool_trajectory_goal_frame # [np.array(4, 4)]
            json_result[LIST_INT_CHANGING_POINT]            = self.changing_point             # [int]
            json_result[MATRIX_TRAJECTORY_HOVERING_START]   = self.hovering_start             # np.array(4, 4)
            json_result[MATRIX_TRAJECTORY_HOVERING_END]     = self.hovering_end               # np.array(4, 4)
            json_result[MATRIX_TRAJECTORY_UNHOVERING_START] = self.unhovering_start           # np.array(4, 4)
            json_result[MATRIX_TRAJECTORY_UNHOVERING_END]   = self.unhovering_end             # np.array(4, 4)
            
            # variables needed for tool usage
            json_result[STR_TASK_TYPE] = self.task_type # str
            
            json_result[MATRIX_TOOL_GOAL]                            = self.Ttool_goal                      # np.array(4, 4)
            json_result[LIST_MATRIX_TRAJECTORY_EFFECTIVESTART_FRAME] = self.trajectory                      # [np.array(4, 4)]
            json_result[LIST_ARRAY_TRAJECTORY_AXES]                  = self.trajectory_screw_axes           # [np.array(1, 6)]
            json_result[LIST_FLOAT_TRAJECTORY_THETAS]                = self.trajectory_thetas               # [float]
            json_result[LIST_INT_TOOL_CONTACT_INDEX]                 = self.tool_contact_area_index         # [int]
            json_result[LIST_INT_GOAL_CONTACT_INDEX]                 = self.goal_contact_area_index         # [int]
            json_result[MATRIX_EFFECTIVESTART_HOVERINGSTART]         = self.Teffectivestart_hoveringstart   # np.array(4, 4)
            json_result[MATRIX_EFFECTIVESTART_HOVERINGEND]           = self.Teffectivestart_hoveringend     # np.array(4, 4)
            json_result[MATRIX_EFFECTIVESTART_UNHOVERINGSTART]       = self.Teffectivestart_unhoveringstart # np.array(4, 4)
            json_result[MATRIX_EFFECTIVEEND_UNHOVERINGEND]           = self.Teffectiveend_unhoveringend   # np.array(4, 4)
            #if self.task_type == TASK_TYPE_POSE_CHANGE:
            json_result[ARRAY_EFFECT_V_WORLD_FRAME] = self.v_start_world_frame # np.array(1, 3)
            json_result[ARRAY_EFFECT_V_GOAL_FRAME]  = self.v_start_goal_frame  # np.array(1, 3)
            json_result[ARRAY_EFFECT_V_TOOL_FRAME]  = self.v_start_tool_frame  # np.array(1, 3)
        
        return json_result