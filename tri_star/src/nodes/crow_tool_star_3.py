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

from std_msgs.msg import String

from tri_star import constants
from tri_star import perception_util

from crow_tool import CrowTool

class CrowToolStarThree(CrowTool):
    def __init__(self):
        super(CrowToolStarThree, self).__init__()
        self.task_manager.set_demo_platform(constants.get_demo_robot_platform())
    
    """
    main UI loop
    """
    def run(self):
        rospy.sleep(1.0)
        #self.robot.connect_robot(True)
        #self.robot.reset_robot()

        # TODO
        #perception_util.add_desk()
        
        is_finish = False

        while not rospy.is_shutdown() and not is_finish:
            self.print_major_deliminator()

            #task_name = "push"
            #source_tool_name = "plunger"
            #source_goal_name = "blue_puck"
            #substitute_tool_name = "butcher_knife"
            #substitute_goal_name = "octopus"
            
            #task_name = "knock"
            #source_tool_name = "gavel"
            #source_goal_name = "gavel_target"
            #substitute_tool_name = "toy_hammer"
            #substitute_goal_name = "wood_block"
            
            #task_name = "stir"
            #source_tool_name = "small_blue_spatula"
            #source_goal_name = "large_bowl"
            #substitute_tool_name = "small_plastic_spoon"
            #substitute_goal_name = "ramen_bowl"
            
            task_name = "scoop"
            source_tool_name = "blue_scooper"
            source_goal_name = "duck"
            substitute_tool_name = "frying_pan"
            substitute_goal_name = "penguin"
            
            #task_name = "cut"
            #source_tool_name = "butcher_knife"
            #source_goal_name = "square_playdough"
            #substitute_tool_name = "long_blue_spatula"
            #substitute_goal_name = "circle_playdough"
            
            #task_name = "draw"
            #source_tool_name = "writing_brush"
            #source_goal_name = "buddha_board"
            #substitute_tool_name = "writing_brush"
            #substitute_goal_name = "buddha_board"
            
            #task_name = "screw"
            #source_tool_name = "paint_scraper"
            #source_goal_name = "screw"
            #substitute_tool_name = "paint_scraper"
            #substitute_goal_name = "screw"            
            
            self.task_pub.publish(task_name)
            self.task_manager.set_current_task(task_name)

            self.tool_pub.publish(source_tool_name)
            self.goal_pub.publish(source_goal_name)

            #self.switch_tool()
            
            self.task_manager.set_current_source_tool(source_tool_name)
            self.task_manager.set_current_source_goal(source_goal_name)
            
            print self.print_major_deliminator()
            
            #transfer_platform = None
            is_transfer_from_another_platform = True
            
            move_on_to_next_task = False
            
            self.tool_pub.publish(substitute_tool_name)
            self.goal_pub.publish(substitute_goal_name)
            self.test_tool(task_name, source_tool_name, source_goal_name, substitute_tool_name, substitute_goal_name, is_source=False, is_another_robot_platform=is_transfer_from_another_platform)
            is_finish = True
            print self.print_major_deliminator()

if __name__ == '__main__':  
    try:
        rospy.init_node('crow_tool_star_3', anonymous=True)

        main_function = CrowToolStarThree()
        main_function.run()

    except rospy.ROSInterruptException:
        pass
