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

import os
import copy
import util
import math
import numpy as np
import random

import open3d as o3d
import stl

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering
from sklearn.neighbors import kneighbors_graph

from get_target_tool_pose import get_T_from_R_p

TOOL_DIR  = "../../../meiying_crow_tool/pointcloud/tools"

def color_tool(pcd, cp_idx=None, segment=None):

    p = copy.deepcopy(pcd)
    p.paint_uniform_color([0, 0, 1]) # Blue result

    colors = np.asarray(p.colors)
    if not segment is None:
        segment = [int(s) for s in segment]
    seg_id, count = np.unique(segment, return_counts=True)

    print "SEGMENT COUNTS: ", count

    if not segment is None:
        for i in seg_id:
            colors[segment==i, :] = np.random.uniform(size=(3))


    if not cp_idx is None:
        colors[cp_idx, : ] = np.array([1,0,0])
    print colors
    p.colors = o3d.utility.Vector3dVector(colors)

    return p

class Mesh(object):

    STL = 'stl'
    PLY = 'ply'
    PCD = 'pcd'

    def __init__(self, fn):
        "docstring"
        self.is_pcd = False
        if ".ply" in fn:
            self._mesh = o3d.io.read_point_cloud(fn)


        elif ".stl" in fn:
            self._mesh = mesh.Mesh.from_file(fn)
            self._f_type = self.STL
            self.from_mesh = True

    @property
    def v0(self):
        if self._f_type == self.STL:
            return self._mesh.v0
        elif self._f_type == self.PLY:
            verts = self._mesh.elements[0].data[self.vert_idx[:,0]][['x','y', 'z']]
            # Transform into numpy array.
            return np.array([list(v) for v in verts])

    @property
    def v1(self):
        if self._f_type == self.STL:
            return self._mesh.v1
        elif self._f_type == self.PLY:
            verts = self._mesh.elements[0].data[self.vert_idx[:,1]][['x','y', 'z']]
            # Transform into numpy array.
            return np.array([list(v) for v in verts])

    @property
    def v2(self):
        if self._f_type == self.STL:
            return self._mesh.v2
        elif self._f_type == self.PLY:
            verts = self._mesh.elements[0].data[self.vert_idx[:,2]][['x','y', 'z']]
            # Transform into numpy array.
            return np.array([list(v) for v in verts])

    @property
    def colors(self):
        if self._f_type == self.PLY:
           # print self._mesh.elements[0].data[self.vert_idx[:,[0,1,2]]][['red',
           #                                                                 'green',
           #                                                                 'blue']]
           colors = self._mesh.elements[0].data[self.vert_idx[:, 0]][['red',
                                                                 'green',
                                                                 'blue']]
           colors = np.array([self.segment_dict[(c[0], c[1], c[2])] for c in colors])

           return colors
    def _gen_segment_dict(self):

        if self._mesh.has_color:
            segments = np.unique(np.asarray(self._mesh.colors))

            self.segment_dict = {}
            for s in range(segments.shape[0]):
                self.segment_dict[tuple(segments[s])] = s


class Mesh2Pointcloud(object):

    def __init__(self, n,  mesh):
        "docstring"
        self.mesh = mesh
        self.n    = n

    def _triangle_area_multi(self, v1,v2,v3):
        """Compute area of multiple triangles given in vertices """
        return 0.4 * np.linalg.norm(np.cross(v2 - v1,
                                            v3 - v1), axis=1)
    def _weighted_rand_indices(self):

        areas = self._triangle_area_multi(self.mesh.v0, self.mesh.v1, self.mesh.v2)
        probs = areas / areas.sum()

        return np.random.choice(range(len(areas)), size=self.n, p=probs)


    def get_pointcloud(self, get_color=False):

        if not self.mesh.is_pcd:
            indx  = self._weighted_rand_indices()
            v1_xyz, v2_xyz, v3_xyz = self.mesh.v0[indx], self.mesh.v1[indx], self.mesh.v2[indx]

            # Get samples via barrycentric coords

            u      = np.random.rand(self.n, 1)
            v      = np.random.rand(self.n, 1)
            is_oob = u + v > 1

            u[is_oob] = 1 - u[is_oob]
            v[is_oob] = 1 - v[is_oob]
            w = 1 - (u + v)

            results = (v1_xyz * u) + (v2_xyz * v) + (w * v3_xyz)

            if self.mesh._f_type == self.mesh.PLY and get_color:
                colors = self.mesh.colors[indx]
                # Add colors to pointcloud matrix and return.
                return np.vstack([results.astype(np.float32).T, colors]).T
            else:
                return results.astype(np.float32)
        else:
            # pnts = self.mesh._mesh.sample_points_uniformly(self.n)
            pnts = self.mesh._mesh
            return np.asarray(pnts.points)


class GeneratePointcloud(object):
    def __init__(self):
        "docstring"
        self.m2p = Mesh2Pointcloud

    def _get_pc_with_segments(self, pcd):
        """
        @pcd: o3d pointcliud.
        returns: (4x4) ndarray representation of segmented points where cols 0-3 contain the
        points and col 4 contains segmentation info.

        """
        pnts = np.asarray(pcd.points)
        # Determine all segments (each is (1x3) RGB ndarray)
        segments = np.unique(np.rint(np.asarray(pcd.colors)), axis=0).tolist()

        # Convert RGB segments to ints
        segment_array = np.apply_along_axis(lambda c: segments.index(c.tolist()),
                                            axis=1,
                                            arr=np.asarray(np.rint(pcd.colors)))

        return np.vstack([pnts.T, segment_array]).T

    def _gen_segmented_pc(self, pcd):
        """
        If pointcloud is not segmented, generate 2 segments naively using kmeans.

        @pcd: o3d pointcloud.
        returns: (4x4) ndarray representation of segmented points where cols 0-3 contain the
        points and col 4 contains segmentation info.


        """
        print "Generating PC segments..."
        pnts = np.asarray(pcd.points)
        # kmeans = KMeans(n_clusters=2).fit(pnts)
        # connectivity = kneighbors_graph(pnts, n_neighbors=3, include_self=False, p=1)
        # kmeans  = AgglomerativeClustering(n_clusters=2, linkage="ward", connectivity=connectivity ).fit(pnts)
        kmeans = SpectralClustering(2, eigen_solver='arpack', assign_labels='kmeans', affinity='nearest_neighbors', n_init=20, degree=10).fit(pnts)

        # kmeans  = AgglomerativeClustering(n_clusters=2, linkage="average", affinity='manhattan', ).fit(pnts)
        #kmeans  = AgglomerativeClustering(n_clusters=2, linkage="average", affinity='l2', ).fit(pnts)

        
        labels = [int(l) for l in kmeans.labels_]

        return np.vstack([pnts.T, labels]).T


    def load_pointcloud(self, fn, n=10000, gen_segments=False):
        """
        Load pointcloud from file.

        @fn: str, path to pointcloud file.
        @get_segments: bool, whether to return segmentation info of points

        returns: (4x4) ndarray representation of segmented points where cols 0-3 contain the
        points and col 4 contains segmentation info.

        """
        print "Loading {}...".format(fn)
        mesh = o3d.io.read_triangle_mesh(fn)
        pcd = mesh.sample_points_uniformly(n)
        pnts = np.asarray(pcd.points)
        print "tool shape: ", pnts.shape

        if gen_segments:
            pnts = self._gen_segmented_pc(pcd)
        else:
            if pcd.has_colors():
                pnts = self._get_pc_with_segments(pcd)

        return pnts


    def mesh_to_pointcloud(self, fn, n, get_color=False):
        mesh = Mesh(fn)

        return self.m2p(n, mesh).get_pointcloud(get_color)

    def get_random_pointcloud(self, n):
        paths = []
        for path, subdirs, files in os.walk(TOOL_DIR):
            name = random.choice(files)
            print "NAME: ", name
            if ".ply" in name or ".pcd" in name or '.stl' in name:
                path = os.path.join(path, name)
                paths.append(path)

        path = random.choice(paths)
        print("LOADING {}\n".format(path))
        return self.mesh_to_pointcloud(path, n, get_color=False)



    def get_rake(self, get_segments=True):
        return self.load_pointcloud("../../tool_files/rake.ply", get_segments=get_segments)

    def get_a_bowl(self, get_segments=True):
        return self.load_pointcloud("../../tool_files/point_clouds/a_bowl.ply",
                                  get_segments=get_segments)

    def get_a_knifekitchen2(self, get_segments=True):
        return self.load_pointcloud("../../tool_files/point_clouds/a_knifekitchen2.ply",
                                    get_segments=get_segments)

    def get_a_bowl(self, get_segments=True):
        return self.load_pointcloud("../../tool_files/point_clouds/a_bowl.ply",
                                    get_segments=get_segments)


    def gen_rectangle(self, n):
        return np.random.uniform(size=(n, 3))

    def test_tool_segmentation(self, obj=None):
        if not obj is None:
            tool_path = os.path.join(TOOL_DIR, "{}.ply".format(obj))
        else:
            fs = os.listdir(TOOL_DIR)
            tool_path = os.path.join(TOOL_DIR, "{}".format(np.random.choice(fs)))

        pnts = self.load_pointcloud(fn=tool_path, n=10000, gen_segments=True)
        pcd = util.np_to_o3d(pnts[:, :3])

        #return pcd
        util.visualize_tool(pcd, segment=pnts[:, 3])
        return color_tool(pcd, segment=pnts[:,3])


    def test_all_tool_segmentations(self):
        tools = os.listdir(TOOL_DIR)
        pcds = []
        trans = np.array([.0, .3, 0.])
        for i, t in enumerate(tools):
            print "t"
            pcd = self.test_tool_segmentation(t)
            pcd.translate(trans * i)
            pcds.append(pcd)

        o3d.visualization.draw_geometries(pcds, "Segmentations")



        # eps = .02
        # for e in range(1000):
        #     eps += .000051
        #     labels = np.array( pcd.cluster_dbscan(eps=eps, min_points=800))
        #     max_label = labels.max()
        #     colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        #     print "eps: ", eps
        #     print("point cloud has {} clusters".format(max_label + 1))
        #     colors[labels < 0] = 0
        #     pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        #     if max_label +1 > 1:
        #         o3d.visualization.draw_geometries([pcd])





if __name__ == '__main__':
    gp = GeneratePointcloud()
    # gp.test_all_tool_segmentations()
    gp.test_tool_segmentation("xylo_stick")
    gp.test_tool_segmentation("purple_spatula")

    gp.test_tool_segmentation("butcher_knife")

    gp.test_tool_segmentation("long_blue_spatula")
    # gp.test_tool_segmentation("frying_pan")

    # gp.test_tool_segmentation("red_mallet")
    # gp.test_tool_segmentation("gavel")
    # guitar_mesh = mesh.Mesh.from_file('./tool_files/guitar.stl'
    # tools_mesh = mesh.Mesh.from_file('./tool_files/tools.stl')
    #tools_mesh = Mesh('./tool_files/tools.stl')
    # print(tools_mesh.v1)
    # mesh = gen_mesh_cube()
    # plot_mesh(mesh)
    # pnts = get_hammer_points(50000)
    # pnts = get_l_points(500)
    # plot_pnts(pnts)
    # test_sampling(5000, tools_mesh)
    # fn = "hammer/3/hammer_out_4_10_fused.ply"
    # mesh_to_pointcloud(100, fn)


    # rake1 = GeneratePointcloud().mesh_to_pointcloud(1000, './tool_files/rake.stl')
    # rake2 = GeneratePointcloud().get_rake_points(1000)
    
