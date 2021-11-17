#!/usr/bin/env python

# Software License Agreement (MIT License)
#
# Copyright (c) 2020, batch_segmentation
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

import glob
import os

def get_file_names_in_directory(directory, file_type):
    file_names = []
    for _file in glob.glob(directory + "/*." + file_type):
        _file = _file.strip().split("/")[-1]
        file_names.append(_file[:-(len(file_type) + 1)])
    
    return file_names

def organize_tools(file_names):
    tools = {}
    
    for tool_combination in file_names:
        tool_combination = tool_combination.strip().split("_")
        lam = int(tool_combination[-2])
        k = int(tool_combination[-3])
        tool_name = "_".join(tool_combination[:-4])
        if tools.has_key(tool_name):
            if tools[tool_name].has_key(k):
                tools[tool_name][k].append(lam)
                tools[tool_name][k].sort()
            else:
                tools[tool_name][k] = [lam]
        else:
            tools[tool_name] = {}
            tools[tool_name][k] = [lam]
    
    return tools

def get_num_segments(directory, file_name):
    ply_file = open(directory + "/" + file_name, "r")
    n_segs = -1
    for line in ply_file:
        line = line.strip().split()
        if len(line) == 3:
            if line[0] == "comment" and line[1] == "n_segms":
                n_segs = int(line[2])
    ply_file.close()
    return n_segs

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_plys(tool_nums, input_folder, output_folder):
    for tool_name in tool_nums.keys():
        create_directory(output_folder + "/" + tool_name)
        for n_seg in tool_nums[tool_name]:
            output_directory = output_folder + "/" + tool_name + "/" + str(n_seg)
            create_directory(output_directory)
            for file_name in tool_nums[tool_name][n_seg]:
                os.system('cp ' + input_folder + '/' + file_name + ' ' + output_directory + '/' + file_name)

if __name__ == '__main__':
    input_folder = "data_demo_segmented_unique"
    output_folder = "data_demo_segmented_numbered"
    
    file_names = get_file_names_in_directory(input_folder, "ply")
    #print file_names
    
    tools = organize_tools(file_names)
    
    tool_names = tools.keys()
    
    tool_nums = {}
    
    for tool_name in tool_names:
        index = 0
        for k in tools[tool_name]:
            for lam in tools[tool_name][k]:
                file_name = "_".join([tool_name, "out", str(k), str(lam), "fused"]) + ".ply"
                num_segments = get_num_segments(input_folder, file_name)
                if tool_nums.has_key(tool_name):
                    if tool_nums[tool_name].has_key(num_segments):
                        tool_nums[tool_name][num_segments].append(file_name)
                    else:
                        tool_nums[tool_name][num_segments] = [file_name]
                else:
                    tool_nums[tool_name] = {}
                    tool_nums[tool_name][num_segments] = [file_name]
        
    create_directory(output_folder)
    save_plys(tool_nums, input_folder, output_folder)