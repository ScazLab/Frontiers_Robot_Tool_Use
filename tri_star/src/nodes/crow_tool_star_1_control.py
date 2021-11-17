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

from crow_tool import CrowTool 

from tri_star import perception_util

class CrowToolStarOneControl(CrowTool):
    def __init__(self):
        super(CrowToolStarOneControl, self).__init__()  
    
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

            #task_name = "push"
            #source_tool_name = "plunger"
            #source_goal_name = "blue_puck"
            
            task_name = "knock"
            source_tool_name = "gavel"
            source_goal_name = "gavel_target"              
            
            self.task_pub.publish(task_name)
            self.task_manager.set_current_task(task_name)

            self.tool_pub.publish(source_tool_name)
            self.goal_pub.publish(source_goal_name)

            #self.switch_tool()
            
            self.task_manager.set_current_source_tool(source_tool_name)
            self.task_manager.set_current_source_goal(source_goal_name)
            
            print self.print_major_deliminator()
            
            transfer_platform = None
            is_transfer_from_another_platform = False
            self.test_tool(task_name, source_tool_name, source_goal_name, substitute_tool_name=None, substitute_goal_name=None, is_source=True, is_another_robot_platform=False, star_1_control_condition=True)
            print self.print_major_deliminator()

if __name__ == '__main__':  
    try:
        rospy.init_node('crow_tool_star_1_control', anonymous=True)

        main_function = CrowToolStarOneControl()
        main_function.run()

    except rospy.ROSInterruptException:
        pass
