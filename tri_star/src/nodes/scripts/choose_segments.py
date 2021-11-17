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
import glob
import random
import shutil

pc_path = "/home/meiying/ros_devel_ws/src/meiying_crow_tool/pointcloud/tools_segmented_numbered"

for tool_name in [os.path.basename(i) for i in glob.glob(pc_path + "/*")]:
    if tool_name.startswith("plunger"):
        num_cluster_counter = {}
        for num_cluster in [os.path.basename(i) for i in glob.glob(pc_path + "/" + tool_name + "/*")]:
            num_cluster_counter[num_cluster] = len([i for i in glob.glob(pc_path + "/" + tool_name + "/" + str(num_cluster) + "/*")])
        
        # choose the one with maximum number of pcs
        max_number = 0
        max_cluster = None
        for key, value in num_cluster_counter.items():
            if value > max_number:
                max_number = value
                max_cluster = key
        if max_cluster == "1":
            second_max_number = 0
            second_max_cluster = None
            for key, value in num_cluster_counter.items():
                if value > second_max_number and key != "1":
                    second_max_number = value
                    second_max_cluster = key
            #print "second_max_cluster: ", second_max_cluster
            #print "second_max_number: ", second_max_number
            if max_number < 3 * second_max_number:
                max_number = second_max_number
                max_cluster = second_max_cluster
        
        print tool_name
        print num_cluster_counter
        print "choose: ", max_cluster
        
        #if max_cluster is not None:
            #chosen_pc_path = os.path.join(pc_path, tool_name, max_cluster)
            #pc_names = [i for i in glob.glob(chosen_pc_path + "/*")]
            #chosen_pc = random.choice(pc_names)
            #chosen_pc_name = os.path.basename(chosen_pc)
            #print "chosen pc is: ", chosen_pc_name
            #save_path = "/home/meiying/ros_devel_ws/src/meiying_crow_tool/pointcloud/chosen_segmented_tools"
            ## copy the file
            #shutil.copy(chosen_pc, save_path)
            ## rename the file
            #os.rename(save_path + "/" + chosen_pc_name, save_path + "/" + tool_name + ".ply")
