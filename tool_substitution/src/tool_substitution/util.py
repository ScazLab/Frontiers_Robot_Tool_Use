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

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from copy import deepcopy

from scipy.spatial.transform import Rotation as Rot
from mpl_toolkits.mplot3d import Axes3D, art3d
from sklearn.metrics.pairwise import cosine_similarity

def np_to_o3d(pnts):
    """
    Get o3d pc from ToolPointcloud object.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pnts)

    return pcd

def close_to(m, n, error=1e-6):
    return m >= n - error and m <= n + error

def normalize_vector(vector):
    if close_to(np.linalg.norm(vector), 0):
        return vector
    return vector / np.linalg.norm(vector)

def is_same_axis_matrix(mat_1, mat_2):
    """
    both mat_1 and both_2 are ordered, with the top row being the primary axis
    """
    compare_1 = mat_1.copy()
    compare_2 = mat_2.copy()

    if compare_2[0][0] * compare_1[0][0] < 0:
        compare_2[0] *= -1.0
    if compare_2[1][0] * compare_1[1][0] < 0:
        compare_2[1] *= -1.0
    if compare_2[2][0] * compare_1[2][0] < 0:
        compare_2[2] *= -1.0

    difference = compare_1 - compare_2

    length = np.matmul(difference, difference.T)
    return close_to(length[0][0], 0) and close_to(length[1][1], 0) and close_to(length[2][2], 0)

def get_sorted_index(array, reverse_order=False):
    """
    get the order of the index of a one dimensions array
    @reverse_order: True, large->small
                    False, small->large
    """

    dtype = [('index', 'i4'), ('value', 'f8')]
    data = np.array([(i, array[i]) for i in range(len(array))], dtype = dtype)
    data.sort(order='value')
    order_index = data['index']
    if reverse_order:
        order_index = np.flip(order_index)
    return order_index

def min_point_distance(pc1, pc2):
    """
    pc1: 3 * n
    pc2: 3 * m
    threshold: int
    """
    length_pc1 = pc1.shape[1]
    length_pc2 = pc2.shape[1]

    # repeat each column length_pc1 times, for example, let length_pc1 = 3
    # then the array
    # x = np.array([[1,2],
    #               [3,4]])
    # becomes
    # array([[1, 1, 1, 2, 2, 2],
    #        [3, 3, 3, 4, 4, 4]])
    repeated_pc2 = np.repeat(pc2, length_pc1, axis=1)
    # repeat the entire marix and repeat itself length_pc2 times, for example, let length_pc2 = 3
    # then the array x becomes
    # array([[1, 2, 1, 2, 1, 2],
    #        [3, 4, 3, 4, 3, 4]])
    repeated_pc1 = np.hstack([pc1] * length_pc2)
    difference = repeated_pc2 - repeated_pc1
    all_distance = np.multiply(difference, difference)
    all_distance = np.sqrt(np.sum(all_distance, axis=0))
    all_distance = np.reshape(all_distance, (length_pc2, length_pc1))
    min_distance = np.amin(all_distance, axis=1)
    avg_min_distance = np.mean(min_distance)

    print "avg_min_distance = ", avg_min_distance

    return avg_min_distance

def weighted_min_point_distance(src_pc, sub_pc, src_cntct_pnt, sigma=1):
    """
    Same as min_point_distance but weights sub_pc pnts by how far they are
    from src_cntct_pnt. This is because we really only care about the local
    properties of the tool near the contact pnt.
    """
    length_pc1 = src_pc.shape[1]
    length_pc2 = sub_pc.shape[1]

    weights = np.apply_along_axis(lambda col: 1.0 - rbf(src_cntct_pnt, col, sigma) ,
                                  arr=sub_pc, axis=0)

    print("weights")

    # repeat each column length_pc1 times, for example, let length_pc1 = 3
    # then the array
    # x = np.array([[1,2],
    #               [3,4]])
    # becomes
    # array([[1, 1, 1, 2, 2, 2],
    #        [3, 3, 3, 4, 4, 4]])
    repeated_pc2 = np.repeat(sub_pc, length_pc1, axis=1)
    # repeat the entire marix and repeat itself length_pc2 times, for example, let length_pc2 = 3
    # then the array x becomes
    # array([[1, 2, 1, 2, 1, 2],
    #        [3, 4, 3, 4, 3, 4]])
    repeated_pc1 = np.hstack([src_pc] * length_pc2)
    difference = repeated_pc2 - repeated_pc1
    all_distance = np.multiply(difference, difference)
    all_distance = np.sqrt(np.sum(all_distance, axis=0))
    all_distance = np.reshape(all_distance, (length_pc2, length_pc1))
    min_distance = np.amin(all_distance, axis=1)
    min_distance *= weights
    avg_min_distance = np.mean(min_distance)

    print "avg_min_distance = ", avg_min_distance

    return avg_min_distance

def is_2d_point_cloud_overlap(pc1, pc2, threshold):
    """
    pc1: 2 * n
    pc2: 2 * m
    threshold: int
    """

    test_1 = min_point_distance(pc1, pc2) < threshold
    test_2 = min_point_distance(pc2, pc1) < threshold
    return test_1 and test_2

def rbf(x1, x2, sigma = 1):
    """
    Radial basis function. Returns vals btwn [0-1].
    1 is returned if x1 == x2.
    """
    diff = x1 - x2
    return np.exp( -(np.dot(diff, diff)) / (2. * (sigma ** 2)))

def rotation_matrix_from_box_rots(sub_dir,src_dir):
    sub_dir  = sub_dir / np.linalg.norm(sub_dir)
    src_dir = src_dir / np.linalg.norm(src_dir)

    print sub_dir
    print src_dir
    v = np.cross(sub_dir, src_dir)
    c = np.dot(sub_dir, src_dir)
    theta = np.arccos(c)

    eps = .15
    is_parallel =  np.isclose(np.linalg.norm(v), np.zeros(1), atol=eps).item()
    is_same_dir  = np.isclose(theta,  np.zeros(1), atol=eps).item()
    print "SAME DIR: ", is_same_dir
    print "IS PARALLEL: ", is_parallel

    rots = np.arange(np.pi /2, 2*np.pi, step=.10)
    scores = []

    if is_parallel and is_same_dir:
        print "PC already pointing in right dir. No rotation required."
        return np.identity(3)
    elif not is_parallel:
        rot_vec = v
    else:
        a = src_dir
        rot_vec = np.array([a[1], -a[0], 0.]) if a[2] < a[0] else np.array([0, -a[2], a[1]])

    for rot in rots:
        rot_vec = rot_vec * rot
        R = Rot.from_rotvec(rot_vec)
        R = R.as_dcm()
        rot_sub_dir = R.dot(sub_dir.T)
        dir_score = np.arccos(np.dot(src_dir, rot_sub_dir))
        scores.append((R, dir_score)) 

    best_R, fitness = min(scores, key=lambda score: score[1])
    print "BEST ROTATION COSINE SIM ", fitness

    return best_R

def rotation_matrix_from_vectors(a, b):
    """ Find the rotation matrix that aligns a to b
    :param a: A 3d "source" vector
    :param b: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to a, aligns it with b.
    """
    a, b = (a / np.linalg.norm(a)).reshape(3), (b / np.linalg.norm(b)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    theta = np.arccos(c)

    eps = .15
    is_parallel =  np.isclose(np.linalg.norm(v), np.zeros(1), atol=eps).item()
    is_opp_dir  = np.isclose(theta, np.pi, atol=eps).item()
    is_same_dir  = np.isclose(theta,  np.zeros(1), atol=eps).item()

    print "V ", v
    print "IS PARALLEL {}: {} ".format(is_parallel, np.linalg.norm(v))
    print "THETA ", theta
    print "SAME DIR: ", is_same_dir
    print "OPP DIR: ", is_opp_dir

    if is_parallel:
        if is_same_dir:
            return np.identity(3)
        elif is_opp_dir:
            v = np.array([a[1], -a[0], 0.]) if a[2] < a[0] else np.array([0, -a[2], a[1]])
            v *= np.pi

    s = np.linalg.norm(v)
    v = v /s
    print "NEW V ", v

    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    return R

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

def visualize_contact_area(pnts, cntct_idx):
    """Visualize a tool and it's contact area."""
    tool_idx = [i for i  in range(pnts.shape[0]) if not i in cntct_idx]
    tool_pnts = pnts[tool_idx]
    cntct_pnts = pnts[cntct_idx]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs=tool_pnts[:, 0], ys=tool_pnts[:, 1], zs=tool_pnts[:, 2], c='r')
    ax.scatter(xs=cntct_pnts[:, 0], ys=cntct_pnts[:, 1], zs=cntct_pnts[:, 2], c='b', s=200)

    plt.show()

def visualize_two_pcs(pnts1, pnts2, s1=None, s2=None, c1=None, c2=None):
    """
    Just plots two pointclouds for easy comparison.
    """
    faces = {'a':[1,2,6,7],
             'b':[2,7,3,8],
             'c':[5,8,6,7],
             'd':[1,6,5,0],
             'e':[5,8,3,2],
             'f':[0,1,2,3]}

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print "Pnts 1 mean: {} Pnts 2 mean: {}".format(pnts1.mean(axis=0),
                                                   pnts2.mean(axis=0))

    if not s1 is None:
        bb, s = s1
        face = [bb[faces[s], :]]
        side = art3d.Poly3DCollection(face)
        side.set_color('r')
        ax.add_collection3d(side)

    if not s2 is None:
        bb, s = s2
        face = [bb[faces[s], :]]
        side = art3d.Poly3DCollection(face)
        side.set_color('b')
        ax.add_collection3d(side)

    if not c1 is None:
        ax.scatter(xs=c1[ 0], ys=c1[ 1], zs=c1[ 2], c='g', s=350)

    if not c2 is None:
        ax.scatter(xs=c2[ 0], ys=c2[ 1], zs=c2[ 2], c='y', s=350)

    ax.axis('equal')
    ax.scatter(xs=pnts1[:, 0], ys=pnts1[:, 1], zs=pnts1[:, 2], c='r', s=5)
    ax.scatter(xs=pnts2[:, 0], ys=pnts2[:, 1], zs=pnts2[:, 2], c='b', s=10)

    plt.show()

def visualize_vectors(vecs):
    vecs = (vecs.T / np.linalg.norm(vecs, axis=1)).T
    U, V, W = zip(*vecs)
    n = vecs.shape[0]
    X = np.zeros(n)
    Y = np.zeros(n)
    Z = np.zeros(n)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([-1, 1.])
    ax.set_ylim([-1, 1.])
    ax.set_zlim([-1, 1.])

    ax.quiver(X, Y, Z, U, V, W)

    plt.show()

def visualize_tool(pcd, cp_idx=None, segment=None, name="Tool"):
    p = deepcopy(pcd)
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
    o3d.visualization.draw_geometries([p], name)

def visualize_multiple_cps(pcd, orig_cp_idx, cp_idx_list, name="Multiple cps"):
    p = deepcopy(pcd)
    p.paint_uniform_color([0, 0, 1]) # Blue result

    colors = np.asarray(p.colors)
    colors[orig_cp_idx, :] = np.array([1, 0, 0])
 
    for idx in cp_idx_list:
        colors[idx, :] = np.random.uniform(size=(1,3))

    p.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([p], name)

def visualize_reg(src, target, result, result_cp_idx=None, target_cp_idx=None, name="Result"):
    s = deepcopy(src)
    t = deepcopy(target)
    r = deepcopy(result)

    s.paint_uniform_color([1, 0, 0]) # Red src
    t.paint_uniform_color([0, 1, 0]) # Green target
    r.paint_uniform_color([0, 0, 1]) # Blue result

    if not result_cp_idx is None:
        colors = np.asarray(r.colors)
        colors[result_cp_idx, :] = np.array([.5,.5,0])
        r.colors = o3d.utility.Vector3dVector(colors)

    if not target_cp_idx is None:
        colors = np.asarray(t.colors)
        colors[target_cp_idx, :] = np.array([.9,.1,0])
        t.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([s, t, r], name)

def align_pcd_select_size(pcds):
    size = 0.0025
    min_threshold = 10.0
    pcd_aligned = None
    aligned_set = None
    min_size = 0.0
    min_transformations = None

    for i in range(5):
        size += 0.0025
        pcd_aligned = [deepcopy(pc) for pc in pcds]
        for pc in pcd_aligned:
            add_color_normal(pc, paint=False)
        _, transformations = align_pcds_helper(pcd_aligned, size=size)
        
        total_distance = 0.0
        for i in range(len(pcd_aligned)):
            if i > 0:
                total_distance += get_average_distance(pcd_aligned[0], pcd_aligned[i])
        
        distance = 0.0
        if len(pcd_aligned) > 1.0:
            distance = total_distance / (len(pcd_aligned) - 1.0)
            
        print "size: ", size, "; distance: ", distance, "weighted: ", size * distance
        if distance < min_threshold:
            min_threshold = distance
            min_size = size
            min_transformations = transformations
            aligned_set = pcd_aligned

    print "Chosen size thresh: ", min_size
    print 
    
    return aligned_set, min_transformations, min_threshold

# function from: https://qiita.com/tttamaki/items/648422860869bbccc72d
def add_color_normal(pcd, paint=False): # in-place coloring and adding normal
    if paint:
        pcd.paint_uniform_color(np.random.rand(3))
    kdt_n = o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=5)
    pcd.estimate_normals(search_param=kdt_n, fast_normal_computation=False)

# function from: https://qiita.com/tttamaki/items/648422860869bbccc72d
# this is a multiway registration, or full registration
# more information can be found: http://www.open3d.org/docs/release/tutorial/Advanced/multiway_registration.html
# and https://blog.csdn.net/weixin_36219957/article/details/106432869
def align_pcds_helper(pcds, size):
    pose_graph = o3d.registration.PoseGraph()
    accum_pose = np.identity(4)
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(accum_pose))

    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            source = pcds[source_id]
            target = pcds[target_id]

            trans = register(source, target, size)
            
            GTG_mat = o3d.registration.get_information_matrix_from_point_clouds(source, target, size, trans)

            if target_id == source_id + 1:
                accum_pose = np.matmul(trans, accum_pose)
                pose_graph.nodes.append(o3d.registration.PoseGraphNode(np.linalg.inv(accum_pose)))

            pose_graph.edges.append(o3d.registration.PoseGraphEdge(source_id,
                                                               target_id,
                                                               trans,
                                                               GTG_mat,
                                                               uncertain=True))

    solver = o3d.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.registration.GlobalOptimizationConvergenceCriteria()
    option = o3d.registration.GlobalOptimizationOption(max_correspondence_distance=size / 10,
                                                       edge_prune_threshold=size / 10,
                                                       reference_node=0)

    o3d.registration.global_optimization(pose_graph,
                                         method=solver,
                                         criteria=criteria,
                                         option=option)

    transformations = []
    for pcd_id in range(n_pcds):
        trans = pose_graph.nodes[pcd_id].pose
        transformations.append(trans)
        pcds[pcd_id].transform(trans)

    return pcds, transformations

# function from: https://qiita.com/tttamaki/items/648422860869bbccc72d
# it did a pairwise registration
def register(pcd1, pcd2, size, n_iter=4):

    kdt_n = o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=5)
    kdt_f = o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=5)

    pcd1_d = pcd1.voxel_down_sample(size)
    pcd2_d = pcd2.voxel_down_sample(size)
    pcd1_d.estimate_normals(search_param=kdt_n, fast_normal_computation=False)
    pcd2_d.estimate_normals(search_param=kdt_n, fast_normal_computation=False)

    pcd1_f = o3d.registration.compute_fpfh_feature(pcd1_d, kdt_f)
    pcd2_f = o3d.registration.compute_fpfh_feature(pcd2_d, kdt_f)

    checker = [o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
               o3d.registration.CorrespondenceCheckerBasedOnDistance(size * 2)]

    est_ptp = o3d.registration.TransformationEstimationPointToPoint()
    est_ptpln = o3d.registration.TransformationEstimationPointToPlane()

    criteria = o3d.registration.RANSACConvergenceCriteria(max_iteration=400000, max_validation=500)
    icp_criteria = o3d.registration.ICPConvergenceCriteria(max_iteration=1000)
    
    # Perform ICP n_iter times and choose best result.
    res = []
    min_distance = 10.
    chosen_transformation = np.identity(4)
    for i in range(n_iter):
        result1 = o3d.registration.registration_ransac_based_on_feature_matching(pcd1_d, pcd2_d,
                                                                             pcd1_f, pcd2_f,
                                                                             #max_correspondence_distance=size * 2,
                                                                             max_correspondence_distance=0.01,
                                                                             estimation_method=est_ptp,
                                                                             ransac_n=4,
                                                                             checkers=checker,
                                                                             criteria=criteria)

        result2 = o3d.registration.registration_icp(pcd1_d, pcd2_d, size, result1.transformation, est_ptpln, criteria=icp_criteria)
        
        # using distance
        distance = get_average_distance(pcd1_d, pcd2_d)
        if distance < min_distance:
            chosen_transformation = result2.transformation

    return chosen_transformation

def get_average_distance(pc1, pc2):
    distance = pc1.compute_point_cloud_distance(pc2)
    return np.average(distance)

def get_homogenous_transformation_matrix_inverse(T):
    R, p = get_R_p_from_matrix(T)
    return get_homogenous_transformation_matrix(R.T, -np.matmul(R.T, p.T).T)

def get_R_p_from_matrix(T):
    return T[0:-1, 0:-1], np.array([T[0:-1, -1]])

def get_homogenous_transformation_matrix(R, p):
    assert(R.shape[0] == R.shape[1])
    assert(R.shape[0] == p.shape[1])
    return np.c_[np.r_[R, np.zeros((1, R.shape[0]))], np.r_[p.T, [[1]]]]
