#!/usr/bin/env python

# Software License Agreement (MIT License)
#
# Copyright (c) 2020, tool_substitution
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

import argparse
import cv2
import numpy as np
from compare_tools import bbs_to_img, CompareTools, FIGS_PATH
import matplotlib.pyplot as plt
import open3d as o3d

from sample_pointcloud import GeneratePointcloud

from tool_segmentation_example import compare_two_tools
from tool_pointcloud import ToolPointCloud

from util import min_point_distance

parser = argparse.ArgumentParser()
parser.add_argument("-t", type=str, default='hammer',
                    help="Tool to create bounding box for. \
                    Options: 'hammer', 'guitar', 'saw', 'rake', 'L', 'knife' ")
parser.add_argument("-H", type=int, default=3,
                    help="Calc Hamming dist of two segmented tools. Takes num segments as arg. ")

args = parser.parse_args()

gp = GeneratePointcloud()

def guitar_pc():
    return gp.get_guitar_points(7000)

def hammer_pc():
    return gp.get_hammer_points(7000)

def saw_pc():
    return gp.get_saw_points(7000)

def rake_pc():
    return gp.get_rake_points(7000)

def l_pc():
    return gp.get_l_points(7000)

def knife_pc():
    return gp.get_knife_points(7000)

def plot_l_PC():
    pnts = gp.get_l_points(7000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pca = PCA(n_components=3)
    new_pnts = pca.fit_transform(pnts)
    ax.scatter(xs=new_pnts[:,0], ys=new_pnts[:,1], zs=new_pnts[:,2], c='b')
    plt.show()


def hamming_3d_test(pnts):
    for a  in np.arange(0, 420, 60 ):
        a1 = a
        a2 = 0
        a3 = 0
        
        pnts_rot = r_z(a3).dot(r_y(a2).dot(r_x(a1).dot(pnts.T))).T
        print("Rot {} score : {}".format(a, min_point_distance(pnts, pnts_rot)))
        # pc = ToolPointCloud(pnts_rot)
        # pc.visualize()

def hamming_compare_two_pnts(pnts1, pnts2):
    pc1 = ToolPointCloud(pnts1)
    pc2 = ToolPointCloud(pnts2)

    # visualize_two_tools(pc1.get_pc_bb_axis_frame_centered(),
    #                     pc2.get_pc_bb_axis_frame_centered())

    pnts1 = pc1.get_pc_bb_axis_frame_centered()
    pnts2 = pc2.get_pc_bb_axis_frame_centered()

    pc1 = ToolPointCloud(pnts1, normalize=False)
    pc2 = ToolPointCloud(pnts2, normalize=False)

    pc1.scale_pnts_to_target(pc2, keep_proportional=True)


    pnts1, _ = pc1.bb_2d_projection([0,1], 2, visualize=True)
    pnts2, _ = pc2.bb_2d_projection([0,1], 2, visualize=True)

    print(pnts1.shape)
    target_norm = pc2.bb.dim_lens
    target_vol  = target_norm[0] * target_norm[1] * target_norm[2]
    print("TARGET DIMS: {}".format(target_norm))
    print("TARGET VOL: {}".format(target_vol))


    print("RAKE TOOL AFTER SCALING")
    new_norm = pc1.bb.dim_lens
    new_vol  = new_norm[0] * new_norm[1] * new_norm[2]
    print("SRC DIMS: {}".format(new_norm))
    print("SRC VOL: {}".format(new_vol))

    score = []
    degs = np.linspace(0.0, 2.0  * np.pi , 10 )

    for a  in degs:

        # pnts_rot = r_x(a).dot(pnts2.T).T
        score = min_point_distance(pnts1, pnts2)

        # visualize_two_tools(pnts1, pnts_rot)
        print("Rot {} score : {}".format(a, score))
    
    i = np.argmin(score)
    a = degs[i]

    visualize_two_tools(pnts1, pnts_rot)


def r_x(a):
    return np.array([
            [1., 0., 0.],
            [0., np.cos(a), -np.sin(a)],
            [0., np.sin(a), np.cos(a)]
        ])


def r_y(a):
        return np.array([
            [np.cos(a), 0, np.sin(a)],
            [0, 1, 0],
            [-np.sin(a), 0,  np.cos(a)]
        ])

def r_z(a):
        return  np.array([
            [np.cos(a), -np.sin(a), 0],
            [np.sin(a), np.cos(a), 0],
            [0, 0, 1]
        ])


def scaling_pnts_test():

    pnts1 = gp.get_random_ply(1500)
    pnts2 = gp.get_random_ply(1500)

    old_norm = pc1.bb.dim_lens
    old_vol  = old_norm[0] * old_norm[1] * old_norm[2]
    print("SRC TOOL BEFORE SCALING")
    print("SRC DIMS: {}".format(old_norm))
    print("SRC VOL: {}".format(old_vol))
    print("bb: {}".format(pc1.bb))

    target_norm = pc2.bb.dim_lens
    target_vol  = target_norm[0] * target_norm[1] * target_norm[2]
    print("TARGET DIMS: {}".format(target_norm))
    print("TARGET VOL: {}".format(target_vol))

    pc1.visualize_bb()
    pc1.scale_pnts_to_target(pc2, False)

    print("SRC TOOL AFTER SCALING")
    new_norm = pc1.bb.dim_lens
    new_vol  = new_norm[0] * new_norm[1] * new_norm[2]
    print("SRC DIMS: {}".format(new_norm))
    print("SRC VOL: {}".format(new_vol))

    pc1.visualize_bb()

def visualize_two_tools(tool1, tool2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    t1_min = tool1.min(axis=0)
    t2_min = tool2.min(axis=0)
    t1_max = tool1.max(axis=0)
    t2_max = tool2.max(axis=0)

    ax.set_xlim3d(min(t1_min[0], t2_min[0]),
                  max(t1_max[0], t2_max[0]))
    ax.set_ylim3d(min(t1_min[1], t2_min[1]),
                  max(t1_max[1], t2_max[1]))
    ax.set_zlim3d(min(t1_min[2], t2_min[2]),
                  max(t1_max[2], t2_max[2]))

    # ax.axis('equal')

    ax.scatter(xs=tool1[:, 0], ys=tool1[:, 1], zs=tool1[:, 2], c='b')
    ax.scatter(xs=tool2[:, 0], ys=tool2[:, 1], zs=tool2[:, 2], c='r')
    plt.show()
