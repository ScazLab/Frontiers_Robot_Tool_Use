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

from tri_star import transformation_util
from tri_star.constants import DIRNAME_ARCHIVE
from tri_star.constants import get_learned_data_dir, get_candidate_tools, get_candidate_tasks, get_robot_platform
from tri_star.task import Task

class TaskManager(object):
    def __init__(self):
        self.tasks = {}
        self.demo_tasks = {}
        self.demo_robot_platform = get_robot_platform()
        self.current_task_name = None # str
        self.current_source_tool_name = None # str
        self.current_substitute_tool_name = None # str
        self.current_source_goal_name = None # str
        self.current_substitute_goal_name = None # str

        self.tasks = self.load_tasks()
        self.demo_tasks = self.load_tasks(self.demo_robot_platform) # where the data is trained, if the training platform is different from the testing platform

    def load_tasks(self, robot_platform=get_robot_platform()):
        tasks = {}
        
        all_task_name = get_candidate_tasks()

        for task_name in all_task_name:
            current_task = Task(task_name, get_learned_data_dir(robot_platform))
            if current_task.load_task():
                tasks[task_name] = current_task
        
        return tasks

    def reset(self):
        self.current_task_name = None
        self.current_source_tool_name = None
        self.current_substitute_tool_name = None
        self.current_source_goal_name = None
        self.current_substitute_goal_name = None        

    """
    setters
    """
    def set_demo_platform(self, demo_platform):
        self.demo_robot_platform = demo_platform
        self.demo_tasks = self.load_tasks(demo_platform)
    
    def set_current_task(self, task_name):
        if not task_name in self.tasks.keys():
            current_task = Task(task_name, get_learned_data_dir())
            current_task.load_task()
            self.tasks[task_name] = current_task

        self.current_task_name = task_name

    def set_current_source_tool(self, source_tool_name):
        self.current_source_tool_name = source_tool_name

    def set_current_substitute_tool(self, substitute_tool_name):
        self.current_substitute_tool_name = substitute_tool_name
        
    def set_current_source_goal(self, souce_goal_name):
        self.current_source_goal_name = souce_goal_name
        
    def set_current_substitute_goal(self, substitute_goal_name):
        self.current_substitute_goal_name = substitute_goal_name

    """
    getters
    """
    def get_task(self, task_name=None, is_demo=False):
        if task_name is None:
            task_name = self.current_task_name
        
        tasks = self.tasks
        if is_demo:
            tasks = self.demo_tasks       

        if task_name not in tasks.keys():
            return None
        else:
            return tasks[task_name]

    def get_current_task_name(self):
        return self.current_task_name

    def get_task_type(self, task_name=None):
        if task_name is None:
            task_name = self.current_task_name
        
        task_type = self.tasks[task_name].get_task_type()
        if task_type is None:
            task_type = self.demo_tasks[task_name].get_task_type()
            
        return task_type

    def get_current_source_tool(self):
        return self.current_source_tool_name
    
    def get_current_source_goal(self):
        return self.current_source_goal_name

    """
    manage source tool
    """
    def process_source_sample(self, Tworld_goalstart, Tworld_goalend, tool_trajectory_world_frame, source_tool_name=None, source_goal_name=None, task=None):
        if source_tool_name is None:
            source_tool_name = self.current_source_tool_name
        if source_goal_name is None:
            source_goal_name = self.current_source_goal_name
        self.get_task(task).process_source_sample(source_tool_name, source_goal_name, Tworld_goalstart, Tworld_goalend, tool_trajectory_world_frame)

    def train(self, source_tool_name=None, source_goal_name=None, task=None):
        if source_tool_name is None:
            source_tool_name = self.current_source_tool_name
        if source_goal_name is None:
            source_goal_name = self.current_source_goal_name
        self.get_task(task).train(source_tool_name, source_goal_name)
        
    def learned_source_tool(self, task_name=None, is_demo=False):
        return self.get_task(task_name, is_demo).get_learned_source_tools()

    def reset_source_tool(self, source_tool_name=None, task_name=None):
        current_task = self.get_task(task_name)
        if source_tool_name is None:
            source_tool_name = self.current_source_tool_name
        if task_name is None:
            task_name = self.current_task_name         

        print "archiving learned tool usages of {} - {}...".format(task_name, source_tool_name)
        current_task.archive_source_tool(source_tool_name, to_reload=True)

    def save_source_tool(self, source_tool_name=None, task_name=None):
        if source_tool_name is None:
            source_tool_name = self.current_source_tool_name
        if task_name is None:
            task_name = self.current_task_name
        current_task = self.get_task(task_name)

        print "saving {} - {}...".format(task_name, source_tool_name)
        current_task.save_source_tool(source_tool_name)

    # Tworld_goalend is allowed to be None for task type: other
    def test_source(self, Tworld_goalstart, Tworld_goalend=None, source_tool_name=None, source_goal_name=None, task=None, is_demo=False):
        current_task = self.get_task(task, is_demo)
        if source_tool_name is None:
            source_tool_name = self.current_source_tool_name
        if source_goal_name is None:
            source_goal_name = self.current_source_goal_name

        return current_task.source_usage(source_tool_name, source_goal_name, Tworld_goalstart, Tworld_goalend)

    def test_source_control(self, Tworld_goalstart, source_tool_name=None, source_goal_name=None, task=None, is_demo=False):
        current_task = self.get_task(task, is_demo)
        if source_tool_name is None:
            source_tool_name = self.current_source_tool_name
        if source_goal_name is None:
            source_goal_name = self.current_source_goal_name        
        
        return current_task.source_control_usage(Tworld_goalstart, source_tool_name, source_goal_name)       

    """
    manage source goals
    """
    def learned_source_goal(self, source_tool_name=None, task=None, is_demo=False):
        if source_tool_name is None:
            source_tool_name = self.current_source_tool_name

        return self.get_task(task, is_demo).get_learned_source_goals(source_tool_name)

    def reset_tool_goal(self, source_tool_name=None, source_goal_name=None, task_name=None):
        current_task = self.get_task(task_name)
        if source_tool_name is None:
            source_tool_name = self.current_source_tool_name
        if source_goal_name is None:
            source_goal_name = self.current_source_goal_name
        if task_name is None:
            task_name = self.current_task_name         

        print "archiving learned tool usages of {} - tool: {} - goal: {}...".format(task_name, source_tool_name, source_goal_name)
        current_task.archive_source_goal(source_tool_name, source_goal_name, to_reload=True)

    def save_tool_goal(self, source_tool_name=None, source_goal_name=None, task_name=None):
        current_task = self.get_task(task_name)
        if source_tool_name is None:
            source_tool_name = self.current_source_tool_name
        if source_goal_name is None:
            source_goal_name = self.current_source_goal_name
        if task_name is None:
            task_name = self.current_task_name

        print "saving {} - tool: {} - goal: {}...".format(task_name, source_tool_name, source_goal_name)
        current_task.save_source_goal(source_tool_name, source_goal_name, force_save=True)

    """
    manage subs
    """
    def test_substitute(self, Tworld_goalstart, Tworld_goalend=None, source_tool_name=None, source_goal_name=None, substitute_tool_name=None, substitute_goal_name=None, is_demo=False):
        current_task = self.get_task(is_demo=is_demo, task_name=None)
        if source_tool_name is None:
            source_tool_name = self.current_source_tool_name
        if substitute_tool_name is None:
            substitute_tool_name = self.current_substitute_tool_name
        if source_goal_name is None:
            source_goal_name = self.current_source_goal_name
        if substitute_goal_name is None:
            substitute_goal_name = self.current_substitute_goal_name
        
        return current_task.substitute_usage(source_tool_name, source_goal_name, substitute_tool_name, substitute_goal_name, Tworld_goalstart, Tworld_goalend)
 
    def test_substitute_control(self, Tworld_goalstart, Tworld_goalend, source_tool_name, source_goal_name, substitute_tool_name, substitute_goal_name):
        current_task = self.get_task(is_demo=False, task_name=None)
        if source_tool_name is None:
            source_tool_name = self.current_source_tool_name
        if substitute_tool_name is None:
            substitute_tool_name = self.current_substitute_tool_name
        if source_goal_name is None:
            source_goal_name = self.current_source_goal_name
        if substitute_goal_name is None:
            substitute_goal_name = self.current_substitute_goal_name
        
        return current_task.substitute_control_usage(source_tool_name, source_goal_name, substitute_tool_name, substitute_goal_name, Tworld_goalstart, Tworld_goalend) 
 
    """
    task related
    """
    def save_task(self, task_name=None):
        current_task = self.get_task(task_name)
        current_task.save_task()
