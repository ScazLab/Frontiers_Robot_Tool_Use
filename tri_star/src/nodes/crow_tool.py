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

import rospy
import numpy as np

from std_msgs.msg import String

from tri_star import color_print
from tri_star import file_util
from tri_star import perception_util
from tri_star import pointcloud_util
from tri_star import robot_util
from tri_star import constants
from tri_star.task_manager import TaskManager
from tri_star.interaction_manager import InteractionManager
from tri_star.constants import TOOL_TYPE_SOURCE, TOOL_TYPE_SUBSTITUTE
from tri_star.constants import GOAL_TYPE_SOURCE, GOAL_TYPE_SUBSTITUTE
from tri_star.constants import get_candidate_tasks, get_candidate_tools, get_candidate_goals, get_candidate_tasks

DEFAULT_COLOR = "default_color"
WARNING_COLOR = "warning_color"

COLOR = {TOOL_TYPE_SOURCE     : color_print.YELLOW,
         TOOL_TYPE_SUBSTITUTE : color_print.GREEN,
         GOAL_TYPE_SOURCE     : color_print.MAGENTA,
         GOAL_TYPE_SUBSTITUTE : color_print.CYAN,
         DEFAULT_COLOR        : color_print.BLUE,
         WARNING_COLOR        : color_print.RED}

class CrowTool(object):
    def __init__(self):
        self.task_manager = TaskManager()
        self.interaction_manager = InteractionManager()
        self.robot = robot_util.Robot()
        
        # for simulator interaction and perception
        self.task_pub = rospy.Publisher("tri_star/task", String, queue_size=10)
        self.tool_pub = rospy.Publisher("tri_star/tool", String, queue_size=10)
        self.goal_pub = rospy.Publisher("tri_star/goal", String, queue_size=10)
    
    """
    print related functions
    """
    def print_minor_deliminator(self):
        print "-" * 30
    
    def print_major_deliminator(self):
        print "=" * 30 
        
    """
    get user input
    """
    def get_options(self, option_type, option_candidates, color=None):
        option_str = ""
        found = False
        
        if color is None:
            color = COLOR[DEFAULT_COLOR]
            
        option_type = color_print.colored_text(option_type, text_color=color, bold=True)
            
        while not found:
            question = "what is the {}? The options are:".format(option_type)
            print question
            
            for i in range(len(option_candidates)):
                print "\t", str(i), ": ", option_candidates[i]
                
            choice = raw_input("{} index: ".format(option_type))
            
            choice = file_util.str_to_int(choice)
            
            if choice is not None:
                if choice >= 0 and choice < len(option_candidates):
                    found = True
                    option_str = option_candidates[choice]
    
            if not found:
                print color_print.colored_text("invalid", background_color=COLOR[WARNING_COLOR], bold=True) + " input!!!!!!!!!!!!!"
        
        print "You choose: " + color_print.colored_text(option_str, text_color=color, bold=True, italic=True)
        
        return option_str
    
    def get_boolean_option(self, question):
        option_str = ""
        found = False
        option = False
        
        while not found:
            option_str = raw_input(question + " (n/y): ")
            if option_str.lower() == "n":
                found = True
                option = False
            elif option_str.lower() == "y":
                found = True
                option = True
        
        print "You choose: " + color_print.colored_text(option_str.lower(), text_color=COLOR[DEFAULT_COLOR], bold=True, italic=True)
        
        return option

    def get_task_name(self):
        tasks = get_candidate_tasks()
        
        return self.get_options("task", tasks)

    def get_object_str(self, object_type, learned_objects, object_color, all_objects, excluded_objects=[]):
        learned_object_names = [i for i in learned_objects if i not in excluded_objects]
        learned_object_adjusted_names = ["{} (learned)".format(i) for i in learned_object_names]
        objects = [i for i in learned_object_adjusted_names]
        
        for each_object in all_objects:
            if each_object not in learned_objects and each_object not in excluded_objects:
                objects.append(each_object)
        
        object_choice = self.get_options(object_type, objects, color=object_color)
        
        if object_choice in learned_object_adjusted_names:
            object_choice_index = learned_object_adjusted_names.index(object_choice)
            object_choice = learned_object_names[object_choice_index]
        
        return object_choice        

    def get_tool_str(self, tool_type, learned_tools, tool_color, excluded_tools=[]):
        all_tools = get_candidate_tools()
        
        return self.get_object_str(tool_type, learned_tools, tool_color, all_tools, excluded_tools)

    def get_goal_str(self, goal_type, learned_goals, goal_color, excluded_goals=[]):
        all_goals = get_candidate_goals()
        
        return self.get_object_str(goal_type, learned_goals, goal_color, all_goals, excluded_goals)

    def task_tool_combo_str(self, task, tool, color=None):
        text = "TASK {} - TOOL {}".format(task, tool)
        if color:
            text = color_print.colored_text(text, text_color=color, bold=True, italic=True)
        return text  
       
    """
    source
    """
    def get_source_tool(self, is_another_platform=False):
        if not self.task_manager.get_current_source_tool() is None:
            change_source_tool = self.get_boolean_option("Change the source tool {} of the task {}?".format(self.task_manager.get_current_task().get_source_tool_name(), task_str))
            if change_source_tool:
                self.task_manager.reset()

        learned_source_tool = self.task_manager.learned_source_tool(is_demo=is_another_platform)
        source_tool_name = self.get_tool_str("source tool", learned_source_tool, COLOR[TOOL_TYPE_SOURCE])
        self.task_manager.set_current_source_tool(source_tool_name)        

        return source_tool_name

    def check_learn_source(self, source_tool_name, is_another_platform):
        to_learn = True
        
        learned_source_goals = self.task_manager.learned_source_goal(source_tool_name, is_demo=is_another_platform)
        source_goal_name = self.get_goal_str("source goal", learned_source_goals,  COLOR[GOAL_TYPE_SOURCE])
        self.task_manager.set_current_source_goal(source_goal_name)
        
        if not is_another_platform:
            if source_goal_name in learned_source_goals:
                # to relearn or update the exsiting tool
                to_change = self.get_boolean_option("Relearn or Update the learned usage?")
                # to relearn or update the exsiting tool
                if not to_change:
                    to_learn = False
                else:
                    relearn_or_update = self.get_options("operation", ["Relearn", "Update"])
                    if relearn_or_update == "Relearn":
                        self.task_manager.reset_tool_goal()
            else:
                to_learn = self.get_boolean_option("Do you want to learn the usage (get a sample) for this tool on the current goal?")
        else:
            to_learn = False
        
        return source_goal_name, to_learn
    
    def learn(self, source_tool_name, source_goal_name):
        finish = False
        while not finish:
            Tworld_goalstart, Tworld_goalend, tool_trajectory_world_frame = self.get_sample(source_tool_name, source_goal_name)
            self.task_manager.process_source_sample(Tworld_goalstart, Tworld_goalend, tool_trajectory_world_frame)
            finish = not self.get_boolean_option("Get another sample?")
        self.task_manager.train()
        print "save the usage..."
        self.task_manager.save_tool_goal(source_tool_name, source_goal_name)
    
    def get_source_goal(self, source_tool_name, is_another_platform=False):
        source_goal_name, to_learn = self.check_learn_source(source_tool_name, is_another_platform)
        self.goal_pub.publish(source_goal_name)
        if to_learn:
            self.learn(source_tool_name, source_goal_name) 
        return source_goal_name
    
    """
    substitutes
    """
    def get_substitute(self, type_str, source_name, color, candidates_func, manager_set_current_func):
        substitute_name = source_name
        
        if self.get_boolean_option("use a different {}?".format(type_str)):
            all_objects = candidates_func()
            candidate_objects = []
            for each_object in all_objects:
                if each_object == source_name:
                    each_object = each_object + " (source {})".format(type_str)
                candidate_objects.append(each_object)
            substitute_name = self.get_options("substitute {}".format(type_str), candidate_objects, color)
        
        manager_set_current_func(substitute_name)        

        return substitute_name       
    
    def get_substitute_tool(self, source_tool_name):
        return self.get_substitute("tool", source_tool_name, COLOR[TOOL_TYPE_SUBSTITUTE], get_candidate_tools, self.task_manager.set_current_substitute_tool)
    
    def get_substitute_goal(self, source_goal_name):
        return self.get_substitute("goal", source_goal_name, COLOR[GOAL_TYPE_SUBSTITUTE], get_candidate_goals, self.task_manager.set_current_substitute_goal)
    
    """
    get samples
    """
    def get_sample(self, tool, goal):
        return self.interaction_manager.perceive_training_sample(tool, goal)

    def get_test_sample(self, tool, goal, task):
        return self.interaction_manager.perceive_testing_sample(self.task_manager.get_task_type(), tool, goal, task)  

    def run_test(self, trajectories, tool, goal, control_condition=False):
        self.interaction_manager.run_testing_sample(goal, trajectories, self.task_manager.get_task_type(), tool, control_condition=control_condition)

    """
    test
    """
    # Tworld_goalend is allowed to be None for task type: other
    def test_source(self, Tworld_goalstart, Tworld_goalend, source_tool_name, source_goal_name, is_another_robot_platform=False):
        return self.task_manager.test_source(Tworld_goalstart, Tworld_goalend, source_tool_name, source_goal_name, is_demo=is_another_robot_platform)
    
    # Tworld_goalend is allowed to be None for task type: other
    def test_substitute(self, Tworld_goalstart, Tworld_goalend, source_tool_name, source_goal_name, substitute_tool_name, substitute_goal_name, is_another_robot_platform=False):
        return self.task_manager.test_substitute(Tworld_goalstart, Tworld_goalend, source_tool_name, source_goal_name, substitute_tool_name, substitute_goal_name, is_demo=is_another_robot_platform)    
    
    # star 1 control
    # Tworld_goalend is allowed to be None for task type: other
    def test_control(self, Tworld_goalstart, Tworld_goalend, source_tool_name, source_goal_name):
        return self.task_manager.test_source_control(Tworld_goalstart, source_tool_name, source_goal_name)
    
    # star 2 control
    def test_substitute_control(self, Tworld_goalstart, Tworld_goalend, source_tool_name, source_goal_name, substitute_tool_name, substitute_goal_name):
        return self.task_manager.test_substitute_control(Tworld_goalstart, Tworld_goalend, source_tool_name, source_goal_name, substitute_tool_name, substitute_goal_name)
    
    # Tworld_goalend is allowed to be None for task type: other
    def test_tool(self, task, source_tool_name, source_goal_name, substitute_tool_name=None, substitute_goal_name=None, is_source=True, is_another_robot_platform=False, star_1_control_condition=False, star_2_control_condition=False):
        tool = source_tool_name
        if not is_source:
            tool = substitute_tool_name
        goal = source_goal_name
        if not is_source:
            goal = substitute_goal_name
        
        finish = False
        while not finish:
            Tworld_goalstart, Tworld_goalend = self.get_test_sample(tool, goal, task)
            trajectories = []
            if star_1_control_condition:
                trajectories = self.test_control(Tworld_goalstart, Tworld_goalend, source_tool_name, source_goal_name)            
                ee_trajectories = trajectories
                training_Tee_tool = constants.STAR_1_TEE_TOOL[tool]
                training_tool_trajectories = []
                for ee_trajectory in ee_trajectories:
                    tool_trajectory = [np.matmul(Tworld_ee, training_Tee_tool) for Tworld_ee in ee_trajectories]
                    training_tool_trajectories.append(tool_trajectory)
                pointcloud_util.visualize_trajectory(tool, goal, training_tool_trajectories[0], Tworld_goalstart, Tworld_goalend, index=0)                
            elif is_source:
                trajectories = self.test_source(Tworld_goalstart, Tworld_goalend, source_tool_name, source_goal_name, is_another_robot_platform)
            else:
                if star_2_control_condition:
                    trajectories = self.test_substitute_control(Tworld_goalstart, Tworld_goalend, source_tool_name, source_goal_name, substitute_tool_name, substitute_goal_name)
                else:
                    trajectories = self.test_substitute(Tworld_goalstart, Tworld_goalend, source_tool_name, source_goal_name, substitute_tool_name, substitute_goal_name, is_another_robot_platform)
            Tactual_world_goalend = self.run_test(trajectories, tool, goal, control_condition=star_1_control_condition)

            self.print_minor_deliminator()
            finish = not self.get_boolean_option("continue to test the tool {} with the goal {}?".format(tool, goal))

    """
    switch tool question
    """
    def switch_tool(self):
        if self.get_boolean_option("need to add, switch or repose the tool chosen?"):
            self.interaction_manager.switch_tool()        
    
    """
    main UI loop
    """
    def run(self):
        rospy.sleep(1.0)
        self.robot.connect_robot(True)
        self.robot.reset_robot()

        # TODO
        #perception_util.add_desk()
        
        is_finish = False

        while not rospy.is_shutdown() and not is_finish:
            self.print_major_deliminator()

            task_name = self.get_task_name()
            self.task_pub.publish(task_name)
            self.task_manager.set_current_task(task_name)

            source_tool_name = None
            substitute_tool_name = None
            
            source_goal_name = None
            substitute_goal_name = None

            source_tool_name = self.get_source_tool()
            source_goal_name = self.get_source_goal(source_tool_name)
            self.tool_pub.publish(source_tool_name)
            self.goal_pub.publish(source_goal_name)

            self.switch_tool()
            
            self.task_manager.set_current_source_tool(source_tool_name)
            self.task_manager.set_current_source_goal(source_goal_name)
            
            print self.print_major_deliminator()
            
            transfer_platform = None
            is_transfer_from_another_platform = False
            
            move_on_to_next_task = False
            while not move_on_to_next_task:
                options = ["relearn the source tool {} - source goal {}".format(source_tool_name, source_goal_name),
                           "test the source tool {} - source goal {}".format(source_tool_name, source_goal_name), 
                           "choose and test the substitutes",
                           "set whether transfer from another robot platform",
                           "move on to the next task",
                           "star 1 control testing",
                           "quit"
                           ]
                choice = self.get_options("next thing", options)
                
                if choice == options[0]: # relearn the source
                    self.task_manager.reset_tool_goal()
                    #self.switch_tool()
                    self.learn(source_tool_name, source_goal_name)
                elif choice == options[1]: # test the source
                    if source_tool_name is None:
                        print "the tool was unknown!"
                    else:
                        #self.switch_tool()
                        self.test_tool(task_name, source_tool_name, source_goal_name, is_source=True, is_another_robot_platform=is_transfer_from_another_platform)
                    print self.print_major_deliminator()
                elif choice == options[2]: # choose and test the substitutes
                    # self.switch_tool()
                    substitute_tool_name = self.get_substitute_tool(source_tool_name)
                    substitute_goal_name = self.get_substitute_goal(source_goal_name)
                    self.tool_pub.publish(substitute_tool_name)
                    self.goal_pub.publish(substitute_goal_name)
                    # if source_tool_name != substitute_tool_name:
                    #     self.interaction_manager.reset_grasping_gesture()
                    self.test_tool(task_name, source_tool_name, source_goal_name, substitute_tool_name, substitute_goal_name, is_source=False, is_another_robot_platform=is_transfer_from_another_platform)
                    print self.print_major_deliminator()
                elif choice == options[3]: # transfer from another robot platform
                    if self.get_boolean_option("learn from another platform?"):
                        is_transfer_from_another_platform = True
                        platform = self.get_options("robot platform", ["ur", "baxter", "kuka"])
                        self.task_manager.set_demo_platform(platform)
                    else:
                        is_transfer_from_another_platform = False 
                elif choice == options[4]: # move on to next task
                    self.task_manager.save_task()
                    self.tool_pub.publish("")
                    self.goal_pub.publish("")
                    move_on_to_next_task = True
                    self.interaction_manager.reset_grasping_gesture()
                    print self.print_major_deliminator()
                elif choice == options[5]: # star 1 control testing
                    self.test_tool(task_name, source_tool_name, source_goal_name, is_source=True, is_another_robot_platform=False, star_1_control_condition=True)
                elif choice == options[6]: # quit
                    move_on_to_next_task = True
                    is_finish = True

if __name__ == '__main__':  
    try:
        rospy.init_node('crow_tool', anonymous=True)

        main_function = CrowTool()
        main_function.run()

    except rospy.ROSInterruptException:
        pass
