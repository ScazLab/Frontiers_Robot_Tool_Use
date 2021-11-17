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
import open3d as o3d
import numpy as np

from tool_substitution.tool_pointcloud import ToolPointCloud

from tri_star import constants
from tri_star import transformation_util
from tri_star import file_util

def normalize_pc(original_path_name):
    pc = o3d.io.read_point_cloud(original_path_name)
    
    o3d_bb = pc.get_oriented_bounding_box()
    o3d_bb_center = o3d_bb.get_center()
    
    print "o3d center: ", o3d_bb_center

    pc_np = np.asarray(pc.points)
    pc_tpc = ToolPointCloud(pc_np, normalize=False)
    
    R = pc_tpc.get_axis()
    p = pc_tpc.get_bb_centroid()
    
    print "path: ", original_path_name
    print "\tp: ", p
    print "\tR: "
    print R
    
    p = p * -1.
    
    TR = transformation_util.get_transformation_matrix_with_rotation_matrix(R, np.array([0., 0., 0.]))
    TR = transformation_util.get_transformation_matrix_inverse(TR)
    
    return p, TR

def normalize_object_group(group_names, write_group_path, read_group_path):
    for object_name in group_names:
        original_path_name = os.path.join(read_group_path, "{}.ply".format(object_name))
        saved_path_name = os.path.join(write_group_path, "{}.ply".format(object_name))
        p, TR = normalize_pc(original_path_name)
        pc = o3d.io.read_triangle_mesh(original_path_name)
        pc.transform(TR)
        pc.translate(p)
        print "pc center after normalization: ", pc.get_center()
        o3d.io.write_triangle_mesh(saved_path_name, pc, write_ascii=True)

def get_candidate_tools():
    tools = file_util.get_filenames_in_dir(os.path.join(constants.pointcloud_dir(), "raw_tools"), ext="ply")
    
    return [os.path.splitext(i)[0] for i in tools]

def get_candidate_goals():
    goals = file_util.get_filenames_in_dir(os.path.join(constants.pointcloud_dir(), "raw_goals"), ext="ply")
    
    return [os.path.splitext(i)[0] for i in goals]

#all_goals = get_candidate_goals()
#all_goals = ["buddha_board", "ramen_bowl", "litter_box", "seahorse"]
#all_goals = ["duck", "seahorse", "octopus", "G_letter", "penguin", "five_number"]
#all_goals = ["blue_puck"]
all_goals = ["circle_playdough", "heart_playdough", "star_playdough"]
#all_tools = get_candidate_tools()

normalize_object_group(all_goals, constants.goal_mesh_dir(), os.path.join(constants.pointcloud_dir(), "raw_goals"))
#normalize_object_group(all_tools, constants.tool_mesh_dir(), os.path.join(constants.pointcloud_dir(), "raw_tools"))
