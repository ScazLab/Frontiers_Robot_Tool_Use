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
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.spatial import ConvexHull

from util import normalize_vector, close_to, get_sorted_index, is_2d_point_cloud_overlap

class BoundingBox(object):
    def __init__(self, pnts, eps=0.05):
        self.pnts = pnts # pnts is a n by 2 or 3 matrix.
        self.bb = None # bb reprsentes a bounding box. It is a a 5 or 10 by 2 or 3 matrix. Each row is a point. For a 2D bounding box, the 5th is the same as the 1st point. For a 3D bounding box, 5th=1st, and 10th=6th.
        self.normalized_axis = None # axis is a homogeneous  matrix. Each COLUMN is a basis
        self.unnormalized_axis = None
        self.norms = None # The norm of each axis
        self.eps = eps

    def get_normalized_axis(self):
        """
        Returns the bounding box axes normalized and in descending size order.
        """
        return self.normalized_axis

    def get_unnormalized_axis(self):
        """
        Returns the bounding box axes unnormalized and in descending size order.
        """
        return self.unnormalized_axis

    def get_unnormalized_unordered_axis(self):
        """
        Returns the bounding box axes unnormalized and unordered.
        """
        axis_1 = np.array([bb[1] - bb[0]]).T
        axis_2 = np.array([bb[3] - bb[0]]).T
        axis_3 = np.array([bb[5] - bb[0]]).T
        return np.hstack([axis_1, axis_2, axis_3])

    def get_bb(self):
        return self.bb

    def get_pc(self):
        return self.pnts

    @property
    def dim_lens(self):
        ranges = self.get_ranges()
        return np.abs(ranges[1,: ] - ranges[0, :])

    def visualize(self):
        pass

    def scale_bb(self, scales):
        # assert scales.shape[] == self.bb.shape[1]
        scales = np.array(scales)

        self.bb *= scales
        self.norms *= scales


    def _calculate_axis(self):
        pass

    def _get_bb_from_axis(self, axis):
        pass

    def _order_unnormalized_axis_from_axis(self, unordered_axis):
        normalized_axis, norms = normalize(unordered_axis, norm='l2', axis=0, return_norm=True)
        sorted_index = get_sorted_index(norms, reverse_order=True)

        return unordered_axis[:, sorted_index], norms[sorted_index]

    def _get_unnormalized_axis_from_bounding_box(self, bb):
        """
        2 * 2 or 3 * 3, each column is an axis.
        It is ordered by length, so that the first column is the primary axis.
        return: unnormalized_axis, norm
        """
        return None, None

    def _get_unnormalized_axis_from_normalized_axis(self, axis):
        bb = self._get_bb_from_axis(axis)
        return self._get_unnormalized_axis_from_bounding_box(bb)

    def _get_normalized_axis_from_unnormalized_axis(self, axis):
        bb = self._get_bb_from_axis(axis)
        return self._get_normalized_axis_from_bounding_box(bb)

    def _get_normalized_axis_from_bounding_box(self, bb):
        unnormalized_axis, norm = self._get_unnormalized_axis_from_bounding_box(bb)

        return normalize(unnormalized_axis, norm='l2', axis=0), norm

    def get_ranges(self):
        # Returns 2xdim matrix where each column has extrema along each dim.
        return np.vstack([self.bb.min(axis=0), self.bb.max(axis=0)])

class BoundingBox2D(BoundingBox):
    def __init__(self, pnts, eps=0.05):
        #super().__init__(pnts) # python 3 syntax
        super(BoundingBox2D, self).__init__(pnts, eps)
        self._calculate_axis()

    def area(self):
        if self.norms is None:
            return
        return self.norms[0] * self.norms[1]

    def perimeter(self):
        if self.norms is None:
            return
        return (self.norms[0] + self.norms[1]) * 2

    def visualize(self):
        """Visualize points and bounding box"""
        if self.bb is None:
            return

        fig = plt.figure()

        ax = fig.add_subplot(111)
        ax.axis('equal')

        ax.scatter(x=self.pnts[:, 0], y=self.pnts[:, 1], c='b')

        ax.scatter(x=self.bb[0, 0], y=self.bb[0, 1], c='r')
        ax.scatter(x=self.bb[1, 0], y=self.bb[1, 1], c='r')
        ax.scatter(x=self.bb[2, 0], y=self.bb[2, 1], c='r')
        ax.scatter(x=self.bb[3, 0], y=self.bb[3, 1], c='r')

        ax.plot(self.bb.T[0], self.bb.T[1], c='r')

        plt.show()

    def _calculate_axis(self):
        pca_axis = self._pca_axis()
        min_area_axis = self._min_area_axis()

        pca_unnormalized, pca_norm = self._get_unnormalized_axis_from_normalized_axis(pca_axis)
        min_area_unnormalized, min_area_norm = self._get_unnormalized_axis_from_normalized_axis(min_area_axis)
        pca_area = pca_norm[0] * pca_norm[1]
        min_area_area = min_area_norm[0] * min_area_norm[1]

        ratio =  pca_area / min_area_area
        #print "ratio: ", ratio

        if close_to(ratio, 1, error=self.eps):
            #print("take original pca result")
            self.normalized_axis = pca_axis.copy()
        else:
            #print("choose new result")
            self.normalized_axis = min_area_axis.copy()

        self.bb = self._get_bb_from_axis(self.normalized_axis)
        self.unnormalized_axis, self.norms = self._get_unnormalized_axis_from_bounding_box(self.bb)

    def _pca_axis(self):
        pca_2D = PCA(n_components=2)
        pca_2D.fit(self.pnts)
        pca_axis = pca_2D.components_
        return pca_axis.T # each column is an axis

    def _min_area_axis(self):
        cv_rect = cv2.minAreaRect(np.array(self.pnts, dtype='f'))
        cv_box = cv2.boxPoints(cv_rect)
        x = cv_box[0] - cv_box[1]
        y = cv_box[0] - cv_box[3]

        length_x = np.linalg.norm(x)
        length_y = np.linalg.norm(y)

        primary_axis = None
        second_axis = None

        if length_x > length_y:
            primary_axis, second_axis = x / length_x, y / length_y
        else:
            primary_axis, second_axis = y / length_y, x / length_x

        axis = np.array([primary_axis, second_axis]).T
        return axis # each column is an axis

    def _get_bb_from_axis(self, axis): # the axis could be normalized or not, could be ordered or not
        normalized_axis = normalize(axis, norm='l2', axis=0)
        projection = np.matmul(np.linalg.inv(normalized_axis), self.pnts.T)

        projection_x = projection[0, :]
        projection_y = projection[1, :]

        project_x_min, project_x_max = np.min(projection_x), np.max(projection_x)
        project_y_min, project_y_max = np.min(projection_y), np.max(projection_y)
        projected_boundary = np.array([[project_x_min, project_y_min],
                                       [project_x_min, project_y_max],
                                       [project_x_max, project_y_max],
                                       [project_x_max, project_y_min],
                                       [project_x_min, project_y_min]])

        bb = np.matmul(normalized_axis, projected_boundary.T).T # 5 by 2 matrix

        return bb

    def _get_unnormalized_axis_from_bounding_box(self, bb):
        """ 2 * 2, each column is an axis. It is ordered by length, so that the first column is the primary axis."""
        axis_1 = np.array([bb[1] - bb[0]]).T
        axis_2 = np.array([bb[3] - bb[0]]).T
        unnormalized_unordered_axis = np.hstack([axis_1, axis_2])
        return self._order_unnormalized_axis_from_axis(unnormalized_unordered_axis)

class BoundingBox3D(BoundingBox):
    """Creates bounding box around pointcloud data"""
    def __init__(self, pnts, eps=0.05):
        "Generates 3D bounding box around points"
        super(BoundingBox3D, self).__init__(pnts, eps)
        self.projection_frame = None
        self.bb2D = None

    def volumn(self):
        if self.norms is None:
            return
        return self.norms[0] * self.norms[1] * self.norms[2]

    def surface_area(self):
        if self.norms is None:
            return
        return (self.norms[0] * self.norms[1] + self.norms[0] * self.norms[2] + self.norms[1] * self.norms[2]) * 2

    def bb_2d_projection(self, projection_index, norm_index, visualize=True):
        print "in function"
        projection_frame = self.normalized_axis[:, [projection_index[0], projection_index[1], norm_index]]
        pnts_projection_frame = np.matmul(np.linalg.inv(projection_frame), self.pnts.T)
        bb = BoundingBox2D(pnts_projection_frame[:-1, :].T)
        if visualize:
            print "visualize"
            bb.visualize()

        return pnts_projection_frame.T[:, :-1], bb

    def set_axis(self, axis = None):
        """
        The axis should be n by 3 (each COLUMN is an axis)
        """
        if axis is None:
            pca = PCA(n_components=3)
            self.normalized_axis = pca.fit(self.pnts).components_.T # each column is an axis
        else:
            self.normalized_axis, norm = self._get_normalized_axis_from_unnormalized_axis(axis)
        self.projection_frame = self.normalized_axis.copy()

    def set_projection_axis(self, projection_index, norm_index):
        if len(projection_index) < 2:
            raise Exception("The number of projection indices should be >= 2")
        if self.projection_frame is None:
            raise Exception("Did not set the projection frame yet")

        self.projection_frame = self.projection_frame[:, [projection_index[0], projection_index[1], norm_index]]

        self._calculate_axis()

    def visualize(self, n="3D"):
        """Visualize points and bounding box"""

        if n is "3D":
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.axis('equal')

            ax.scatter(xs=self.pnts[:,0], ys=self.pnts[:,1], zs=self.pnts[:,2], c='b')
            ax.scatter(xs=self.bb[:,0], ys=self.bb[:,1], zs=self.bb[:,2], c='r', s=100)

            ax.plot(self.bb.T[0], self.bb.T[1], self.bb.T[2], c='r')
            ax.plot(self.bb[(1, 6), :].T[0], self.bb[(1, 6), :].T[1], self.bb[(1, 6), :].T[2], c='r')
            ax.plot(self.bb[(2, 7), :].T[0], self.bb[(2, 7), :].T[1], self.bb[(2, 7), :].T[2], c='r')
            ax.plot(self.bb[(3, 8), :].T[0], self.bb[(3, 8), :].T[1], self.bb[(3, 8), :].T[2], c='r')

            plt.show()
        elif n is "2D":
            self.bb2D.visualize()

    def _calculate_axis(self):
        # in axis, each column is an axis
        pnts_projection_frame = np.matmul(np.linalg.inv(self.projection_frame), self.pnts.T)

        self.bb2D = BoundingBox2D(pnts_projection_frame[:-1, :].T)
        bb2D_axis_projection_frame = self.bb2D.get_normalized_axis()
        bb2D_axis_projection_frame = np.vstack([bb2D_axis_projection_frame, np.array([0, 0])])
        bb2D_axis_world_frame = np.matmul(self.projection_frame, bb2D_axis_projection_frame)

        axis = np.hstack([bb2D_axis_world_frame, np.array([self.projection_frame[:, 2]]).T])

        self.bb = self._get_bb_from_axis(axis)
        self.unnormalized_axis, self.norms = self._get_unnormalized_axis_from_bounding_box(self.bb)
        self.normalized_axis, _ = self._get_normalized_axis_from_bounding_box(self.bb)

    def _get_bb_from_axis(self, axis):
        normalized_axis = normalize(axis, norm='l2', axis=0)
        projection = np.matmul(np.linalg.inv(normalized_axis), self.pnts.T)

        projection_x = projection[0, :]
        projection_y = projection[1, :]
        projection_z = projection[2, :]

        project_x_min, project_x_max = np.min(projection_x), np.max(projection_x)
        project_y_min, project_y_max = np.min(projection_y), np.max(projection_y)
        project_z_min, project_z_max = np.min(projection_z), np.max(projection_z)

        projected_boundary = np.array([[project_x_min, project_y_min, project_z_max],
                                       [project_x_min, project_y_max, project_z_max],
                                       [project_x_max, project_y_max, project_z_max],
                                       [project_x_max, project_y_min, project_z_max],
                                       [project_x_min, project_y_min, project_z_max],
                                       [project_x_min, project_y_min, project_z_min],
                                       [project_x_min, project_y_max, project_z_min],
                                       [project_x_max, project_y_max, project_z_min],
                                       [project_x_max, project_y_min, project_z_min],
                                       [project_x_min, project_y_min, project_z_min]])

        bb = np.matmul(normalized_axis, projected_boundary.T).T # 10 by 2 matrix

        return bb

    def _get_unnormalized_axis_from_bounding_box(self, bb):
        """
        2 * 2, each column is an axis.
        It is ordered by length, so that the first column is the primary axis.
        return: unnormalized_axis, norm
        """
        axis_1 = np.array([bb[1] - bb[0]]).T
        axis_2 = np.array([bb[3] - bb[0]]).T
        axis_3 = np.array([bb[5] - bb[0]]).T
        unnormalized_unordered_axis = np.hstack([axis_1, axis_2, axis_3])

        return self._order_unnormalized_axis_from_axis(unnormalized_unordered_axis)
