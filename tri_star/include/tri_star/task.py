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

from tri_star import tool_usage
from tri_star import file_util
from tri_star.constants import TASK_TYPE_POSE_CHANGE, TASK_TYPE_OTHER
from tri_star.constants import get_candidate_tools, get_candidate_goals
from tri_star.constants import get_package_dir
from tri_star.tool_usage import FILENAME_TOOL_USAGE
from tri_star.tool_usage import SourceToolUsage, SubstituteToolUsage

class Task(object):
    def __init__(self, task, data_dir):
        self.task = task
        self.task_type = None
        self.base_data_dir = os.path.join(data_dir, task)
        
        self.reset()
        
    def reset(self):
        self.usages = {}
    
    """
    getters
    """
    def get_source_tool(self, source_tool_name):
        if not source_tool_name in self.usages.keys():
            return None
        return self.usages[source_tool_name]
    
    def get_source_tool_goal(self, source_tool_name, source_goal_name):
        source_tool = self.get_source_tool(source_tool_name)
        if source_tool is None or not source_goal_name in source_tool.keys():
            return None
        return source_tool[source_goal_name]

    def get_task_name(self):
        return self.task

    def get_task_type(self):
        return self.task_type

    def get_learned_source_tools(self):
        return self.usages.keys()
    
    def get_unlearned_source_tools(self):
        return [tool for tool in get_candidate_tools() if tool not in self.get_learned_source_tools()]
    
    def get_learned_source_goals(self, source_tool):
        source = self.get_source_tool(source_tool)
        if source is None:
            return []
        return source.keys()
    
    def get_unlearned_source_goals(self, source_tool):
        return [goal for goal in get_candidate_goals() if goal not in self.get_learned_source_goals(source_tool)]
    
    """
    i/o
    """
    def reload_task(self):
        self.reset()
        return self.load_task()
    
    def load_task(self):
        base_data_dir = os.path.join(get_package_dir(), self.base_data_dir)
        if not os.path.exists(base_data_dir):
            return False

        source_tool_names = file_util.get_dirnames_in_dir(base_data_dir)
        source_tool_names = [i for i in source_tool_names if i in get_candidate_tools()]
        if len(source_tool_names) == 0:
            return False
        
        for source_tool_name in source_tool_names:
            source_tool_dir = os.path.join(base_data_dir, source_tool_name)
            source_goal_names = file_util.get_dirnames_in_dir(source_tool_dir)
            source_goal_names = [i for i in source_goal_names if i in get_candidate_goals()]
            if len(source_goal_names) == 0:
                return False
            get_result = False
            for source_goal_name in source_goal_names:
                get_result = get_result or self.load_source(source_tool_name, source_goal_name)
            if not get_result:
                return False
        
        return True
            
    def load_source(self, source_tool_name, source_goal_name):
        source = SourceToolUsage(self.task, source_tool_name, source_goal_name, self.base_data_dir)
        
        if source.read(): # has been learned
            if source_tool_name not in self.usages.keys():
                self.usages[source_tool_name] = {}
            self.usages[source_tool_name][source_goal_name] = source
            if self.task_type is None:
                self.task_type = source.get_task_type()
            return True
        
        return False
    
    def save_task(self):
        self.archive_task(to_reload=False) # just archive everything just in case there are updates. no need to bother what the changes are
        
        for source_tool_name in self.source_tools.keys():
            self.save_source_tool(source_tool_name)
    
    def save_source_tool(self, source_tool_name):
        if source_tool_name not in self.source_tools.keys():
            return
        
        source_tool = self.get_source_tool(source_tool_name)
        
        if not source_tool is None:
            self.archive_source_tool(source_tool_name, to_reload=False) # in case there are changes in source tool
            
            source_tool.save()
            for source_goal_name in self.usages[source_tool_name].keys():
                self.save_source_goal(source_tool_name, source_goal_name)
    
    def save_source_goal(self, source_tool_name, source_goal_name, force_save=False):
        if force_save or not self.get_source_tool_goal(source_tool_name, source_goal_name):
            self.archive_source_goal(source_tool_name, source_goal_name, to_reload=False) # in case there are changes. 
            self.usages[source_tool_name][source_goal_name].save()
    
    def archive_task(self, to_reload=False):
        file_util.archive(self.base_data_dir)
        if to_reload:
            self.reload_task()
    
    def archive_source_tool(self, source_tool_name, to_reload=False):
        source_tool_dir = os.path.join(self.base_data_dir, source_tool_name)
        file_util.archive(source_tool_dir)
        if to_reload:
            self.reload_task()
    
    def archive_source_goal(self, source_tool_name, source_goal_name, to_reload=False):
        usage = self.get_source_tool_goal(source_tool_name, source_goal_name)
        if not usage is None:
            usage.archive()
            if to_reload:
                self.reload_task()

    """
    learn samples
    """
    def process_source_sample(self, source_tool_name, source_goal_name, Tworld_goalstart, Tworld_goalend, tool_trajectory_world_frame):
        usage = self.get_source_tool_goal(source_tool_name, source_goal_name)
        
        if usage is None:
            if self.get_source_tool(source_tool_name) is None:
                self.usages[source_tool_name] = {}
            usage = SourceToolUsage(self.task, source_tool_name, source_goal_name, self.base_data_dir)
            self.usages[source_tool_name][source_goal_name] = usage
        
        usage.process_new_sample(Tworld_goalstart, Tworld_goalend, tool_trajectory_world_frame)
    
    def train(self, source_tool_name, source_goal_name):
        usage = self.get_source_tool_goal(source_tool_name, source_goal_name)
        if not usage is None:
            usage.process_learned_samples()
        if self.task_type is None:
            self.task_type = usage.get_task_type()
    
    """
    calculate the usage to achieve the goal
    """
    # Tworld_goalend is allowed to be None for task type: other
    def source_usage(self, source_tool_name, source_goal_name, Tworld_goalstart, Tworld_goalend=None):
        usage = self.get_source_tool_goal(source_tool_name, source_goal_name)
        
        if not usage is None:
            num_circles = -1
            if usage.circular() and usage.task_type == TASK_TYPE_OTHER:
                num_circles = int(raw_input("how many circles do you want?"))
            return usage.get_usage(Tworld_goalstart, Tworld_goalend, num_circles) # TODO: optional. get the smallest angle changes
        
        return None
    
    # Tworld_goalend is allowed to be None for task type: other
    def substitute_usage(self, source_tool_name, source_goal_name, sub_tool_name, sub_goal_name, Tworld_goalstart, Tworld_goalend=None):
        # usage = self.source_usage(sub_tool_name, sub_goal_name, Tworld_goalstart, Tworld_goalend)
        source = self.get_source_tool_goal(source_tool_name, source_goal_name)
        print "SOURCE: ", source
        if not source is None:
            sub = SubstituteToolUsage(self.task, sub_tool_name, sub_goal_name, self.base_data_dir, source)
            num_circles = -1
            if source.circular() and source.task_type == TASK_TYPE_OTHER:
                num_circles = int(raw_input("how many circles do you want?"))            
            usage = sub.get_usage(Tworld_goalstart, Tworld_goalend, num_circles)
    
        return usage
    
    def source_control_usage(self, Tworld_goalstart, source_tool_name, source_goal_name):
        usage = self.get_source_tool_goal(source_tool_name, source_goal_name)
        
        if not usage is None:
            return usage.get_star_1_control_condition_usage(Tworld_goalstart)
        
        return None 
    
    def substitute_control_usage(self, source_tool_name, source_goal_name, substitute_tool_name, substitute_goal_name, Tworld_goalstart, Tworld_goalend):
        source = self.get_source_tool_goal(source_tool_name, source_goal_name)
        
        if not source is None:
            sub = SubstituteToolUsage(self.task, substitute_tool_name, substitute_goal_name, self.base_data_dir, source, is_control=True)
            num_circles = -1
            if source.circular() and source.task_type == TASK_TYPE_OTHER:
                num_circles = int(raw_input("how many circles do you want?"))            
            usage = sub.get_usage(Tworld_goalstart, Tworld_goalend, num_circles)
            
            return usage
        
        return None