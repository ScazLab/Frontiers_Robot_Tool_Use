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

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.spatial import ConvexHull

from bounding_box import BoundingBox3D
from util import close_to

class ToolPointCloud(object):
    """Creates bounding box around pointcloud data"""
    def __init__(self, pnts, eps=0.001, normalize=True, contact_pnt_idx=None):
        "Point cloud of the tool"
        if pnts.shape[1] == 4:
            # First 3 columns are coords, last is segment
            self.pnts = pnts[:,[0,1,2]]
            self.segments = pnts[:,3].astype(np.int)
            self.segment_list = np.unique(self.segments)
        else:
            self.pnts = pnts
            self.segments = np.repeat(1, pnts.shape[0])
            self.segment_list = [1]
        self.segment_offsets = self.gen_segment_offsets()
        self.eps = eps # Error term for deciding best bounding box
        self.mean = None
        self.bb = None # 10 by 3, the 5th and 10th is the reptead point
        self.aruco_frame = {} # there are multiple arucos on an object. in unnormalized_pc frame
        self.scale = 0 # The scale used to scale to unit in meters.
        self.scales = []
        self.is_scaled = False

        self.mean = self.pnts.mean(axis = 0)
        self.normalize = normalize
        if normalize:
            self._normalize_pointcloud()

        self.contact_pnt_idx = contact_pnt_idx
        
        print "[tool_pointcloud] self.contact_pnt_idx"
        print self.contact_pnt_idx
        
        self.bounding_box()

    def scale_pnts_to_target(self, target_tpc, keep_proportional=False):
        """
        Scale points to match the dims of a target tool pointcloud.
        If keep_proportional == True, volumns of bbs will be scaled while
        keeping side proportions constant.
        """

        # Get the len of the sides of each bounding box.
        src_dim_lens   = self.bb.norms
        target_dim_lens = target_tpc.bb.norms


        if keep_proportional:
            # Get volumns of both bbs
            targ_vol =  target_dim_lens[0] * target_dim_lens[1] * target_dim_lens[2]
            src_vol = src_dim_lens[0] * src_dim_lens[1] * src_dim_lens[2]

            # Get ratio of columns
            scale_val = targ_vol / src_vol
            # Proportionally scale src pc based on this ratio
            scale_factor = math.pow(scale_val, 1.0/3.0)
        else:
            scale_factor = target_dim_lens / src_dim_lens

        # Scale points along bb axis for proper scaling.
        scaled_pnts = self.get_pc_bb_axis_frame_centered() * scale_factor
        # Rotate scaled points back to original alignment.
        scaled_pnts = np.linalg.inv(self.get_axis()).dot(scaled_pnts.T).T

        return scaled_pnts, scale_factor

    def gen_segment_offsets(self):
        offset = 0
        offsets = {}

        segs, seg_counts = np.unique(self.segments, return_counts=True)

        for i in range(segs.shape[0]):
            offsets[segs[i]] = offset
            offset += seg_counts[i]

        return offsets

    def get_segment_from_point(self, idx):
        """
        Returns the segment to which idx of point belongs.
        @idx the index of the point.
        """

        segs, counts = np.unique(self.segments[idx], return_counts=True)
        return segs[np.argmax(counts)]

    def get_pnts_in_segment(self, segment):
        """
        Returns all the points in segment.
        @segment, the number of the segment.
        """
        idx = np.where(self.segments == segment)[0]
        return self.pnts[idx, :], idx

    def get_number_of_segments(self):
        return len(set(self.segments.tolist()))

    def get_action_segment(self):
        """
        Get the segment of the tool containing the most contact points.
        """
        cp_segments = self.get_segment_from_point(self.contact_pnt_idx)
        segs, seg_counts = np.unique(cp_segments, return_counts=True)

        return segs[np.argmax(seg_counts)]

    def get_pnt(self, i):
        return self.pnts[i, :]

    def get_bb(self):
        return self.bb

    def idx_to_segment_idx(self, idx):
        """
        Given a point idx for the entire pointcloud, get the idx of the same
        point in the context of just the segment with which it belongs.
        """
        seg = self.get_segment_from_point(idx)
        pnt = self.get_pnt(idx)
        seg_pnts, _ = self.get_pnts_in_segment(seg)

        idx = (seg_pnts[:,None] == pnt).all(2).any(1)

        return idx

    def segment_idx_to_idx(self, seg, seg_idx):
        """
        Given a point idx within a particular segment, get the idx
        of the same point in the context of entire pointcloud.
        """
        seg_pnts, _ = self.get_pnts_in_segment(seg)
        seg_pnt = seg_pnts[seg_idx, :]

        return (self.pnts[:, None] == seg_pnt).all(2).any(1)

    """
    The aruco_frame related functions are yet to be integrated with the rest of this class.
    Code needs to be refactored
    """
    def set_scale(self, model_length, actual_length):
        if model_length == 0 or actual_length == 0:
            raise Exception("either the model length of the actual length is 0!! Cannot determin the scale")

        scale = actual_length / model_length

        if self.scale == 0: # not set yet
            self.scale = scale
        else:
            if close_to(self.scale, scale):
                self.scales.append(scale)
            else:
                raise Exception("Current scale {} is very different from saved scale {}".format(scale, self.scale))

        self.scale = sum(self.scales) / len(self.scales)

    def scale(self):
        self.pnts *= self.scale
        self.mean *= self.scale
        self.bounding_box()

        for frame in self.aruco_frame.values():
            frame[:-1, 3] *= self.scale

        self.is_scaled = True

    def set_aruco_frame_with_four_corners(self, corners, aruco_size, aruco_id):
        # corners are in the scanned object frame, not the world frame when run the experiment
        # https://docs.opencv.org/trunk/d5/dae/tutorial_aruco_detection.html
        # https://github.com/pal-robotics/aruco_ros
        # The corners are in the order of:
        #      - 1. the first corner is the top left corner;
        #      - 2. followed by the top right;
        #      - 3. bottom right;
        #      - 4. and bottom left
        # X direction: 1 -> 4; 2 -> 3
        # Y direction: 1 -> 2; 4 -> 3
        # Z direction: cross produce
        # center: the centroid of the shape
        # aruco size is in meters

        if self.aruco_frame.has_key(aruco_id):
            print "aruco_id: ", aruco_id, " has already set!"
            return

        x_direction_14 = corners[3, :]  - corners[0, :]
        x_direction_23 = corners[2, :]  - corners[1, :]

        x_direction_14_length = np.linalg.norm(x_direction_14)
        x_direction_23_length = np.linalg.norm(x_direction_23)

        x_direction_14 = normalize(x_direction_14)
        x_direction_23 = normalize(x_direction_23)

        if not close_to(np.dot(x_direction_14, x_direction_23), 1):
            raise Exception("The corners of the markers are not right. \
            The line of 1(top left)-4(bottom left) and 2(top right)-3(bottom right) \
            are not parallel.")
        if not close_to(x_direction_14_length, x_direction_23_length):
            raise Exception("The corners of the markers are not right. \
            The length of 1(top left)-4(bottom left) and 2(top right)-3(bottom right) \
            are not the same.")

        x_length = (x_direction_14_length, x_direction_23_length) / 2
        x_direction = (x_direction_14 + x_direction_23) / 2
        x_direction = normalize(x_direction)

        y_direction_12 = corners[1, :]  - corners[0, :]
        y_direction_43 = corners[2, :]  - corners[3, :]

        y_direction_12_length = np.linalg.norm(y_direction_12)
        y_direction_43_length = np.linalg.norm(y_direction_43)

        y_direction_12 = normalize(y_direction_12)
        y_direction_43 = normalize(y_direction_43)

        if not close_to(np.dot(y_direction_12, y_direction_43), 1):
            raise Exception("The corners of the markers are not right. \
            The line of 1(top left)-2(top right) and 4(bottom left)-3(bottom right) \
            are not parallel.")
        if not close_to(y_direction_12_length, y_direction_43_length):
            raise Exception("The corners of the markers are not right. The length of 1(top left)-2(top right) and 4(bottom left)-3(bottom right) are not the same.")

        y_length = (y_direction_12_length, y_direction_43_length) / 2
        y_direction = (y_direction_12 + y_direction_43) / 2
        y_direction = normalize(y_direction)

        if not close_to(np.dot(x_direction, y_direction), 0):
            raise Exception("The corners of the markers are not right. x and y are not perpendicular.")
        if not close_to(x_length, y_length):
            raise Exception("The corners of the markers are not right. The length of x and y are not the same.")

        z_direction = np.cross(x_direction, y_direction)
        z_direction = normalize(z_direction)

        centroid = np.mean(corners, axis=0)

        aruco_frame = np.vstack([x_direction, y_direction, z_direction, centroid]).T
        aruco_frame = np.vstack([aruco_frame], np.array([0, 0, 0, 1]))
        self.aruco_frame[aruco_id] = aruco_frame

    """
    aruco related functions finished
    """

    def get_axis(self, axis_order=[0,1,2]):
        axes = self.bb.get_normalized_axis()
        R = axes[:, axis_order]
        x = R.T[0]
        y = R.T[1]
        z = np.cross(x, y)
        new_R = np.array([x, y, z]).T
        return new_R
        #return axes[:, axis_order]

    def transform(self, pnts):
        axes = self.get_axis()
        return np.dot(axes, pnts)

    def get_unnormalized_pc(self):
        if self.normalize:
            return self.pnts + self.mean
        else:
            return self.pnts

    def get_normalized_pc(self):
        if self.normalize:
            return self.pnts
        else:
            return self.pnts - self.mean

    def get_pc_aruco_frame(self):
        unnomalized_pc = self.get_unnormalized_pc()
        return np.matmul(np.linalg.inv(self.aruco_frame), unnomalized_pc)

    def get_pc_bb_axis_frame(self, axis_order=[0,1,2]):
        return np.matmul(np.linalg.inv(self.get_axis(axis_order)), self.pnts.T).T

    def get_bb_centroid(self):
        """
        Return centroid of (1x3) np array, of bounding box.
        """
        bb_trimed = self.bb.bb.copy()
        bb_trimed = np.delete(bb_trimed, np.s_[4], axis=0)
        bb_trimed = np.delete(bb_trimed, np.s_[-1], axis=0)

        bb_trimed_axis_frame = np.matmul(np.linalg.inv(self.get_axis()),
                                         bb_trimed.T).T
        return np.mean(bb_trimed_axis_frame, axis=0)


    def get_pc_bb_axis_frame_centered(self, axis_order=[0,1,2]):
        pc_bb_axis_frame = self.get_pc_bb_axis_frame(axis_order)
        bb_trimed = self.bb.bb.copy()
        bb_trimed = np.delete(bb_trimed, np.s_[4], axis=0)
        bb_trimed = np.delete(bb_trimed, np.s_[-1], axis=0)
        bb_trimed_axis_frame = np.matmul(np.linalg.inv(self.get_axis(axis_order)),
                                         bb_trimed.T).T
        bb_centroid = np.mean(bb_trimed_axis_frame, axis=0)

        return pc_bb_axis_frame - bb_centroid

    def bb_2d_projection(self, projection_index, norm_index, visualize=True):
        # returns both the pc and the 2D bb
        return self.bb.bb_2d_projection(projection_index, norm_index, visualize)

    def _normalize_pointcloud(self):
        self.pnts -= self.mean

    def bounding_box(self):
        """
        get the bounding box of the point cloud
        TODO: add correction
        """
        found_box = False
        current_axis = None
        result_box = None
        max_loop = 10
        i = 0
        box = None
        bbs = []
        self.bb = None

        while not found_box and i <= max_loop:
            vols = []
            bbs = []
            for [projection_axis_index, norm_axis_index] in [[[0, 1], 2], [[0, 2], 1], [[1, 2], 0]]:
                bb = self._get_bb_helper(current_axis, projection_axis_index, norm_axis_index)
                vols.append(bb.volumn())
                bbs.append(bb)
            max_vol, min_vol = max(vols), min(vols)
            if close_to(max_vol / min_vol, 1, self.eps):
                found_box = True

            self.bb = bbs[vols.index(min(vols))]
            current_axis = self.bb.get_normalized_axis()

            i += 1

        print "final round: ", i

    def _get_bb_helper(self, axis, projection_axis_index, norm_axis_index):
        box = BoundingBox3D(self.pnts)
        box.set_axis(axis)
        box.set_projection_axis(projection_axis_index, norm_axis_index)
        return box

    def visualize(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if not self.segments is None:
            c = self.segments
        else:
            c = 'b'

        ax.axis('equal')
        ax.scatter(xs=self.pnts[:, 0], ys=self.pnts[:, 1], zs=self.pnts[:, 2], c=c)
        plt.show()

    def visualize_bb(self):
        self.bb.visualize("3D")
