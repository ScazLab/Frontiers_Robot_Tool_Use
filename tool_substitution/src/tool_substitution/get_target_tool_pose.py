#!/usr/bin/env python

import numpy as np

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

# Note: All the model frame should already by normalized/centered!

# Naming convention: T_<object>_<frame>, 4 * 4 transformation homogenous matrix
#                    Ps, many points, n * 3

# Given the perceived aruco pose, find the position of each point on the model in the world frame
def get_pnts_world_frame(T_aruco_world, T_aruco_model, Ps_pnts_model):
    # the aruco_world is perceived
    # others are saved value
    # Ps_pnts_model is a n by 3 matrix, each row is a point

    # There should be no stretching, so the unit between the world frame and the model frame must be the consistent!
    T_model_rotation_in_world = np.matmul(T_aruco_world, np.linalg.inv(T_aruco_model))

    #print "T_model_rotation_in_world"
    #print T_model_rotation_in_world

    # should be in the format of [x, y, z, 1].T
    Ps_pnts_model = np.hstack([Ps_pnts_model, np.ones((Ps_pnts_model.shape[0], 1))]).T

    Ps_pnts_world = np.matmul(T_model_rotation_in_world, Ps_pnts_model).T

    Ps_pnts_world = Ps_pnts_world[:, :-1] # get rid of the 1s in the end

    return Ps_pnts_world

# Given the pose of a point on the tool in the world point, and the point in the model frame, the aruco pose in the model frame, the T that the model that has been rotated
# Find the pose of the aruco in the world frame
def get_aruco_world_frame(T_aruco_model, Ps_pnts_model, Ps_pnts_world, R_pnts):
    # Ps_pnts_model, Ps_pnts_world: 1 by 3 matrix
    # T_aruco_model is known, get when scanning in the 3d model
    # Ps_pnts_model is from the geomatric matching result
    # Ps_pnts_world is the desired position in the world frame, which is obtained when learning the task
    # R_pnts_world is a 3 by 3 rotation matrix, which is the result of the geomatric matching
    # T_rotation = np.hstack([model_rotation, (Ps_pnts_world - Ps_pnts_model).T])
    # T_rotation = np.vstack([T_rotation, np.array([0, 0, 0, 1])])

    # T_pnts_rotation * T_pnts_initial = T_pnts_final
    # T_aruco_rotation * T_aruco_initial = T_aruco_final
    # T_pnts_rotation = T_aruco_rotation (in the world frame, not body frame)

    T_pnts_initial = get_T_from_R_p(Ps_pnts_model)
    T_pnts_final = get_T_from_R_p(Ps_pnts_world, R_pnts)

    T_pnts_rotation = np.matmul(T_pnts_final, np.linalg.inv(T_pnts_initial))

    #print "T_pnts_rotation"
    #print T_pnts_rotation

    T_aruco_rotation = T_pnts_rotation

    T_aruco_world = np.matmul(T_aruco_rotation, T_aruco_model)

    return T_aruco_world

def get_T_from_R_p(p=np.zeros((1,3)), R = np.identity(3)):
    # p is 1 by 3

    if not p.shape[0] == 1:
        p = p.reshape(1,-1)

    T = np.hstack([R, p.T])
    T = np.vstack([T, np.array([0, 0, 0, 1])])

    return T

def get_scaling_T(scale=np.ones((1,3)), center=np.zeros((1,3))):
    """
    @scale (1x3 ndarray) scale factor for scaling a point cloud.
    @center (1x3 ndarray) center of scaling (this point will not be scaled.)

    returns corresponding T matrix.

    https://math.stackexchange.com/questions/3245481/rotate-and-scale-a-point-around-different-origins

    """
    trans_init  = get_T_from_R_p(p=center)
    scale       = get_T_from_R_p(R=np.identity(3)*scale)
    trans_final = get_T_from_R_p(p=center*-1.0 )

    return np.linalg.multi_dot([trans_init, scale, trans_final])


def T_inv(T):
    """
    Returns inverse of T.
    """
    R = T[0:3, 0:3]
    p = T[0:3, -1].reshape(1,-1)
    inv_R = np.linalg.inv(R)

    T = np.hstack([inv_R, -inv_R.dot(p.T)])
    T = np.vstack([T, np.array([0, 0, 0, 1])])

    return T
