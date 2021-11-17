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

from itertools import combinations
import copy
import cv2
import threading
import Queue
import json
import random

import rospy
import numpy as np
import open3d as o3d
import ros_numpy

import meshlabxml as mlx

from copy import deepcopy

from scipy.spatial import cKDTree
import transformations as tfs

from tool_substitution.tool_substitution_controller import ToolSubstitution
from tool_substitution.goal_substitution import GoalSubstitution 

from tool_substitution.tool_pointcloud import ToolPointCloud
from tool_substitution.sample_pointcloud import GeneratePointcloud
from tool_substitution.util import visualize_two_pcs

from tri_star import constants
from tri_star import transformation_util
from tri_star.srv import CurrentPointCloud

def np_to_o3d(pnts):
    print "pnts"
    print pnts
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pnts)

    return pcd

def tpc_to_o3d(tpc):
    """
    Get o3d pc from ToolPointcloud object.
    """
    return np_to_o3d(tpc.pnts)

def visualize_tool(pcd, cp_idx=None, segment=None):

    p = deepcopy(pcd)
    p.paint_uniform_color([0, 0, 1]) # Blue result

    colors = np.asarray(p.colors)

    if not segment is None:
        colors[segment==0, :] = np.array([0,1,0])
        colors[segment==1, :] = np.array([0,.5,.5])
    if not cp_idx is None:
        colors[cp_idx, : ] = np.array([1,0,0])

    p.colors = o3d.utility.Vector3dVector(colors)

def calc_contact_surface_from_camera_view(src_pnts, goal_pnts, r=.15):
    """
    @src_pnts: (n x 3) ndarray
    @goal_pnts: (m x 3) ndarray
    @r: float, Search radius multiplier for points in contact surface.

    return list of ints caontaining indicies closest points
    in src_pnts to goal_pnts

    Calculates contact surface of src tool by calculating points visible from
    the vantage point of the contacted goal object.
    
    Refer to: http://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html#Hidden-point-removal
    """
    
    # First, get the closest r% of points in src tool to goal obj.
    initial_cntct_idx = calc_contact_surface(src_pnts, goal_pnts, r=r)

    src_pcd = np_to_o3d(src_pnts)
    goal_pcd = np_to_o3d(goal_pnts)

    src_kd = cKDTree(src_pnts)
    goal_kd = cKDTree(goal_pnts)

    # Find the closest point on the goal obj to the src obj and use
    # that as the position of the 'camera.'
    dists, i = goal_kd.query(src_pnts)
    min_i, min_d = sorted(zip(i,dists), key=lambda d: d[1])[0]

    camera = goal_pnts[min_i, :]
    # Use this heuristic to get the radius of the spherical projection
    diameter = np.linalg.norm(
    np.asarray(src_pcd.get_max_bound()) - np.asarray(src_pcd.get_min_bound()))
    radius = diameter * 100
    # Get idx of points in src_tool from the vantaage of the closest point in
    # goal obj.
    _, camera_cntct_idx = src_pcd.hidden_point_removal(camera, radius)
    # Get intersection of points from both contact surface calculating methods.
    
    if camera_cntct_idx is None and not initial_cntct_idx is None:
        print "Failed getting contact surface using camera method, using just proximity instead"
        final_cntct_idx = initial_cntct_idx
    elif initial_cntct_idx is None and not camera_cntct_idx is None:
        print "Failed getting contact surface using proximity, using just camera method instead"
        final_cntct_idx = camera_cntct_idx
    elif not initial_cntct_idx is None and not camera_cntct_idx is None:
        print "Using both camera method and proximity to get contact surface"
        final_cntct_idx = list(set(camera_cntct_idx).intersection(set(initial_cntct_idx)))
    
    # NOTE: newer versions of o3d have renamed this function 'select_by_index'
    visualize_tool(src_pcd, final_cntct_idx)

    return final_cntct_idx    
    
def load_mesh(mesh_path, paint=True):
    mesh = o3d.io.read_point_cloud(mesh_path)
    add_color_normal(mesh, paint)
    
    return mesh

def get_goal_mesh(goal_name, paint=True):
    mesh_path = constants.get_goal_mesh_path(goal_name)
    
    return load_mesh(mesh_path, paint)

def get_tool_mesh(tool_name, paint=True):
    mesh_path = constants.get_tool_mesh_path(tool_name)
    
    return load_mesh(mesh_path, paint)

def write_mesh(fn, obj, write_ascii=False):
    if isinstance(obj, o3d.geometry.PointCloud):
        o3d.io.write_point_cloud(fn, obj, write_ascii=write_ascii)
    elif isinstance(obj, o3d.geometry.TriangleMesh):
        o3d.io.write_triangle_mesh(fn, obj)
        
def get_line(point, direction, length):
    line_space_right = [0.001 * j for j in range(1, int(length / 0.001) + 1)]
    line_space_left = [-0.001 * j for j in range(1, int(length / 0.001) + 1)]
    line_space_left.reverse()
    line_space = line_space_left + [0.] + line_space_right
    
    line_points = np.array([point + line * direction for line in line_space])
    
    line_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(line_points))
    
    return line_pc

def get_tool_contact_area(tool_name, contact_indices):
    tool_pc = get_tool_mesh(tool_name, paint=False)
    contact_area_np = copy.deepcopy(np.asarray(tool_pc.points)[contact_indices])
    contact_area_pc = o3d.geometry.PointCloud()
    contact_area_pc.points = o3d.utility.Vector3dVector(contact_area_np)
    
    return contact_area_pc

def get_contact_surface(tool_pnts, goal_pnts, get_goal=True):
    """
    @tool_cps: (nx3) ndarray points in tool pc.
    @goal_pnts: (mx3) ndarray points in goal pc.

    Returns idx of pnts in tool_pnts and in goal_pnts estimated to be its contact surface.
    """
    tool_pcs = o3d.geometry.PointCloud()
    tool_pcs.points = o3d.utility.Vector3dVector(tool_pnts)
    goal_pcs = o3d.geometry.PointCloud()
    goal_pcs.points = o3d.utility.Vector3dVector(goal_pnts)
    tool_pcs.paint_uniform_color(np.array([0., 0., 1.]))
    goal_pcs.paint_uniform_color(np.array([0., 1., 0.]))
    
    tool_distance = np.asarray(tool_pcs.compute_point_cloud_distance(goal_pcs))
    tool_min_distance = max(np.min(tool_distance) * 1.2, 0.005)
    tool_contact_area_index = np.where(tool_distance < tool_min_distance)[0]
    print "[pointcloud_util][get_contact_surface] tool_distance: ", np.min(tool_distance)
    print "[pointcloud_util][get_contact_surface] tool_min_distance: ", np.min(tool_min_distance)
    
    if not get_goal:
        return tool_contact_area_index
    
    tool_pcs_color = np.asarray(tool_pcs.colors)
    tool_pcs_color[tool_contact_area_index, :] = np.array([1., 1., 0.])
    tool_pcs.colors = o3d.utility.Vector3dVector(tool_pcs_color)
    
    tool_contact_area_pcs = o3d.geometry.PointCloud()
    tool_contact_area_pcs.points = o3d.utility.Vector3dVector(tool_pnts[tool_contact_area_index])
    goal_distance = np.asarray(goal_pcs.compute_point_cloud_distance(tool_contact_area_pcs))
    
    goal_min_distance = max(np.min(goal_distance) * 1.2, 0.003)
    goal_contact_area_index = np.where(goal_distance < goal_min_distance)[0]
    
    goal_pcs.paint_uniform_color(np.array([0., 1., 0.]))
    goal_pcs_color = np.asarray(goal_pcs.colors)
    goal_pcs_color[goal_contact_area_index, :] = np.array([1., 0., 0.])
    goal_pcs.colors = o3d.utility.Vector3dVector(goal_pcs_color)
    
    return np.array(tool_contact_area_index), np.array(goal_contact_area_index)

def get_closest_pnt(pnt, pntcloud):
    """
    returns the point in pntcloud closest to pnt.
    """
    tree = cKDTree(pntcloud)
    _, i = tree.query(pnt)

    return pntcloud[i,:], i

def calc_contact_surface(src_pnts, goal_pnts, r=.15):
    """
    @src_pnts: (n x 3) ndarray
    @goal_pnts: (m x 3) ndarray
    @r: float, Search radius multiplier for points in contact surface.

    return list of ints caontaining indicies closest points
    in src_pnts to goal_pnts
    """

    rospy.loginfo("Finding contact surface of source tool...")
    # Create ckdtree objs for faster point distance computations.
    src_kd = cKDTree(src_pnts)
    goal_kd = cKDTree(goal_pnts)

    # For each of goal_pnts find pnt in src_pnts with shortest distance and idx
    dists, i = src_kd.query(goal_pnts)
    sorted_pnts_idx = [j[0] for j in sorted(zip(i,dists), key=lambda d: d[1])]
    # Get search radius by finding the distance of the top rth point
    top_r_idx = int(r * dists.shape[0])

    # Get top r
    search_radius = sorted(dists)[top_r_idx]
    print "SEARCH RADIUS: {}".format(search_radius)

    # src_pnts that are within search_radius from goal_pnts are considered part
    # of the contact surface
    cntct_idx = src_kd.query_ball_tree(goal_kd, search_radius)
    # cntct_idx is a list of lists containing idx of goal_pnts within search_radius.
    cntct_idx = [i for i, l in enumerate(cntct_idx) if not len(l) == 0]

    print "SHAPE OF ESITMATED SRC CONTACT SURFACE: ", src_pnts[cntct_idx].shape
    return cntct_idx

def calc_tool_contact_surface_container(goal_pnts, tool_pnts):
    """
    @goal_pnts: (m x 3) ndarray
    @tool_pnts: (n x 3) ndarray

    return list of indices of tool_pnts that are inside the container (goal_pnts)
    """

    goal_pcd = np_to_o3d(goal_pnts)
    tool_pcd = np_to_o3d(tool_pnts)

    goal_bb = goal_pcd.get_axis_aligned_bounding_box()

    return goal_bb.get_point_indices_within_bounding_box(tool_pcd.points)

def calc_contact_surface_mesh_wrapper(src_mesh_path, goal_mesh_path, Tgoal_tool, r=.15, get_goal_surface=True):
    """
    Wrapper around calc_contact_surface method that takes in paths to meshes rather than pointclouds.

    @goal_name: str.
    @tool_name: str.
    @Tgoal_tool: (4x4) ndarray. Inital transformation of src tool.
    @r: float, Search radius multiplier for points in contact surface.

    """
    gp = GeneratePointcloud()
    goal_pnts = gp.load_pointcloud(goal_mesh_path)
    src_pnts = gp.load_pointcloud(src_mesh_path)
    src_pcd = tpc_to_o3d(ToolPointCloud(src_pnts, normalize=False))

    # Transform src points for proper alignment.
    src_pnts = np.asarray(src_pcd.transform(Tgoal_tool).points)
    src_surfaces, goal_surfaces = get_contact_surface(src_pnts, goal_pnts)

    if get_goal_surface:
        ret = (src_surfaces, goal_surfaces)
    else:
        ret = src_surface

    return ret

# convert a PointCloud2 ros message to a open3d PointCloud
# include the rgb will make it slower, so do not set include_rgb=True unless necessary
# code adapted from https://github.com/eric-wieser/ros_numpy/blob/master/src/ros_numpy/point_cloud2.py
# and https://github.com/felixchenfy/open3d_ros_pointcloud_conversion/blob/master/lib_cloud_conversion_between_Open3D_and_ROS.py
def pointcloud2msg_to_open3dpointcloud(msg, include_rgb=True):
    field_names=[field.name for field in msg.fields]
    pc_nparray = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
    point_cloud = o3d.geometry.PointCloud()
    
    if include_rgb and "rgb" in field_names:
        mask = np.isfinite(pc_nparray['x']) & np.isfinite(pc_nparray['y']) & np.isfinite(pc_nparray['z'])
        pc_nparray = pc_nparray[mask]
        xyz = np.zeros(pc_nparray.shape + (3,), dtype=np.float)
        xyz[...,0] = pc_nparray['x']
        xyz[...,1] = pc_nparray['y']
        xyz[...,2] = pc_nparray['z']
        pc_nparray_rgb = ros_numpy.point_cloud2.split_rgb_field(pc_nparray)
        rgb = np.zeros(pc_nparray.shape + (3,), dtype=np.float)
        rgb[...,0] = pc_nparray_rgb['r'] / 255.0
        rgb[...,1] = pc_nparray_rgb['g'] / 255.0
        rgb[...,2] = pc_nparray_rgb['b'] / 255.0
        point_cloud.points = o3d.utility.Vector3dVector(xyz)
        point_cloud.colors = o3d.utility.Vector3dVector(rgb)
    else:
        xyz = ros_numpy.point_cloud2.get_xyz_points(pc_nparray, remove_nans=True)
        point_cloud.points = o3d.utility.Vector3dVector(xyz)
    
    return point_cloud

# function from: https://qiita.com/tttamaki/items/648422860869bbccc72d
def merge(pcds, paint=True):
    all_points = []
    all_color = []
    for pcd in pcds:
        all_points.append(copy.deepcopy(np.asarray(pcd.points)))
        all_color.append(copy.deepcopy(np.asarray(pcd.colors)))

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    merged_pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_color))

    add_color_normal(merged_pcd, paint)

    return merged_pcd

def center_pc(pcd):
    centered_pc = o3d.geometry.PointCloud(pcd)
    center = centered_pc.get_oriented_bounding_box().get_center()
    
    centered_pc.translate(center * -1.0, relative=True)
    
    return centered_pc, center * -1.0

# function adapted from: https://github.com/intel-isl/Open3D/blob/master/examples/python/Basic/pointcloud.ipynb
# under Point Cloud Distance
def remove_background(background_pcd, pcd, threshold=0.1):
    difference = np.asarray(pcd.compute_point_cloud_distance(background_pcd))
    index = np.where(difference > threshold)[0]
    object_pcd = pcd.select_down_sample(index)

    return object_pcd

def transform_to_robot_frame(pcd, frame_id):
    Trobot_camera = constants.get_Trobot_camera(frame_id)
    pcd.transform(Trobot_camera)

    return pcd
    
def get_workspace_bb():
    min_boundary, max_boundary = constants.get_work_space_boundary()
    return o3d.geometry.AxisAlignedBoundingBox(min_boundary, max_boundary)

def get_tool_perception_workspace_bb(mode="test", pos=None):
    """
    Attempts to crop scaned object from background.
    """
    # Uses pre-specified boundaries in ur_config.xml
    if mode == "test":
        min_boundary, max_boundary = constants.get_tool_perception_work_space_boundary()
        bb = o3d.geometry.AxisAlignedBoundingBox(min_boundary, max_boundary)
    # Uses 
    elif mode == "scan":
        min_boundary = [-.2, -.2, 0.0072]
        max_boundary = [.2, .2, .3]

        bb = o3d.geometry.AxisAlignedBoundingBox(min_boundary, max_boundary)
        pnts = bb.get_box_points()
        bb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.geometry.PointCloud(pnts).transform(pos).points)

    elif mode == "platform":
        min_boundary = [-.2, -.2, 0.015]
        max_boundary = [.2, .2, .3]
        
        bb = o3d.geometry.AxisAlignedBoundingBox(min_boundary, max_boundary)
        pnts = bb.get_box_points()
        bb = bb.create_from_points(o3d.geometry.PointCloud(pnts).transform(pos).points)

    return  bb 

# function from: https://qiita.com/tttamaki/items/648422860869bbccc72d
def add_color_normal(pcd, paint=True): # in-place coloring and adding normal
    if paint:
        pcd.paint_uniform_color(np.random.rand(3))
    size = np.abs((pcd.get_max_bound() - pcd.get_min_bound())).max() / 30
    kdt_n = o3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
    pcd.estimate_normals(search_param=kdt_n, fast_normal_computation=False)

# this is just a rough estimation. It only considers axis-aligned bbs.
# oriented bounding box is harder to handle.
def bb_overlap_percentage(pc1, pc2):
    pc1_min_bound = tuple(pc1.get_min_bound())
    pc1_max_bound = tuple(pc1.get_max_bound())

    pc2_min_bound = tuple(pc2.get_min_bound())
    pc2_max_bound = tuple(pc2.get_max_bound())

    pc1_x_min, pc1_y_min, pc1_z_min = pc1_min_bound
    pc1_x_max, pc1_y_max, pc1_z_max = pc1_max_bound

    pc2_x_min, pc2_y_min, pc2_z_min = pc2_min_bound
    pc2_x_max, pc2_y_max, pc2_z_max = pc2_max_bound

    pc_x_min = max(pc1_x_min, pc2_x_min)
    pc_y_min = max(pc1_y_min, pc2_y_min)
    pc_z_min = max(pc1_z_min, pc2_z_min)

    pc_x_max = min(pc1_x_max, pc2_x_max)
    pc_y_max = min(pc1_y_max, pc2_y_max)
    pc_z_max = min(pc1_z_max, pc2_z_max)

    overlap_vol = 0
    if pc_x_min <= pc_x_max and pc_y_min <= pc_y_max and pc_z_min <= pc_z_max:
        bb = o3d.geometry.AxisAlignedBoundingBox([pc_x_min, pc_y_min, pc_z_min], [pc_x_max, pc_y_max, pc_z_max])
        overlap_vol = bb.volume()

    base = pc2.get_axis_aligned_bounding_box().volume()

    return overlap_vol / base

def remove_noise(object_pcd, eps=0.005, min_points=10, paint=True): # based on cluster
    labels = np.array(object_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    max_label = labels.max()
    print "point cloud has {} clusters".format(max_label + 1)
    print set(labels)
    clusters = []
    max_cluster = None
    max_volume = 0
    max_index = 0
    current_index = 0
    for i in set(labels):
        if i == -1: # indicate noise
            continue
        index = np.where(labels == i)[0]
        cluster = object_pcd.select_down_sample(index)
        volume = 0
        try:
            volume = cluster.get_oriented_bounding_box().volume()
        except RuntimeError:
            pass
        if volume > max_volume:
            max_volume = volume
            max_cluster = cluster
            max_index = current_index
        clusters.append(cluster)
        current_index += 1

    current_index = 0
    max_min_bound = tuple(max_cluster.get_min_bound())
    max_max_bound = tuple(max_cluster.get_max_bound())
    final_clusters = [max_cluster]
    for cluster in clusters:
        if current_index == max_index:
            continue
        if bb_overlap_percentage(max_cluster, cluster) > 0.5:
            final_clusters.append(cluster)

    result = merge(final_clusters, paint)
    add_color_normal(result, paint)

    return result

# function from: https://qiita.com/tttamaki/items/648422860869bbccc72d
# it did a pairwise registration
def register(pcd1, pcd2, size, n_iter=5):
    kdt_n = o3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
    kdt_f = o3d.geometry.KDTreeSearchParamHybrid(radius=size * 10, max_nn=50)

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
    icp_criteria = o3d.registration.ICPConvergenceCriteria(max_iteration=400)
    
    # Perform ICP n_iter times and choose best result.
    res = []
    min_distance = 10.
    chosen_transformation = np.identity(4)
    for i in range(n_iter):
        result1 = o3d.registration.registration_ransac_based_on_feature_matching(pcd1_d, pcd2_d,
                                                                             pcd1_f, pcd2_f,
                                                                             max_correspondence_distance=size * 2,
                                                                             estimation_method=est_ptp,
                                                                             ransac_n=4,
                                                                             checkers=checker,
                                                                             criteria=criteria)

        result2 = o3d.registration.registration_icp(pcd1_d, pcd2_d, size, result1.transformation, est_ptpln, criteria=icp_criteria)
        
        res.append(result2)

    # using fitness score
    result2 = max(res, key=lambda r: r.fitness)
    return result2.transformation

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

    # trans: new point = trans * old point
    return pcds, transformations

# using distance
def align_pcd_select_size(pcds):
    size = 0.0025
    min_threshold = 0.0
    pcd_aligned = None
    aligned_set = None
    min_size = 0.0
    min_transformations = None

    for i in range(5):
        size += 0.0025
        pcd_aligned = [copy.deepcopy(pc) for pc in pcds]
        for pc in pcd_aligned:
            add_color_normal(pc, paint=False)
        _, transformations = align_pcds_helper(pcd_aligned, size=size)
        
        total_distance = 0.0
        for i in range(len(pcd_aligned)):
            if i > 0:
                fitness = get_fitness(pcd_aligned[0], pcd_aligned[i], 0.001)
                total_distance += fitness
        
        distance = 0.0
        if len(pcd_aligned) > 1.0:
            distance = total_distance / (len(pcd_aligned) - 1.0)
            
        print "size: ", size, "; distance: ", distance
        if distance > min_threshold:
            min_threshold = distance
            min_size = size
            min_transformations = transformations
            aligned_set = pcd_aligned

    print "Chosen size thresh: ", min_size
    print 
    
    return aligned_set, min_transformations, min_threshold

def get_bb_volume(pc):
    try:
        v2 = pc.get_oriented_bounding_box().volume()

        return v2
    except:
        return 0.0

def transform_oriented_bounding_box(bb, T):
    """
    Work around for the fact that o3d BoundingBox obj has not implemented transform method.
    """
    # 
    print "BB: ", bb
    print "T: ", T.shape
    pnts = bb.get_box_points()
    pnts_transformed = o3d.geometry.PointCloud(pnts).transform(T).points
    bb = o3d.geometry.OrientedBoundingBox.create_from_points(pnts_transformed)

    return bb

def trim_pcd_workspace(pcd):
    workspace_bb = get_workspace_bb()

    pcd = pcd.crop(workspace_bb)
    return pcd

def trim_pcd_tool_perception_workspace(pcd, pos=None, mode="test"):
    workspace_bb = get_tool_perception_workspace_bb(mode, pos)
    pnts = workspace_bb.get_point_indices_within_bounding_box(pcd.points)
    print "TOOL PNTS: ", len(pnts)
    
    pcd = pcd.crop(workspace_bb)

    return pcd

# convert to robot frame
# trim to leave only the workspace
def goal_basic_processing(pcd, paint=True):
    pcd = trim_pcd_workspace(pcd)
    # about segment_plant (under Plane segmentation): http://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.002, ransac_n=50, num_iterations=1000) # TODO: tune parameters
    goal_pc = pcd.select_down_sample(inliers, invert=True)
    add_color_normal(pcd, True)
    
    return goal_pc

def tool_basic_processing(pcd, mode="test", pos=None, paint=True):
    pcd = trim_pcd_tool_perception_workspace(pcd, pos, mode)
    add_color_normal(pcd, paint)
    
    return pcd

def remove_platform(pcd, pos=None, paint=False):
    return tool_basic_processing(pcd, mode="platform", pos=pos, paint=paint)

def centralize_pc(pc):
    centered_pc = o3d.geometry.PointCloud(pc)
    center = centered_pc.get_oriented_bounding_box().get_center()
    
    centered_pc.translate(center * -1.0, relative=True)
    
    return centered_pc, center * -1.0
    
def remove_robot(pcd, rot, mode="test",):
    robot = constants.get_robot_platform()
    if  robot == constants.ROBOT_PLATFORM_KUKA or robot == constants.ROBOT_PLATFORM_UR5E:
        z_axis = np.asarray(pcd.points)[:, 2]
        robot_boundary = np.dot(rot, np.array([0,0,.015,1]))
        print "ROBOT BOUNDARY: ", robot_boundary
        robot_boundary = robot_boundary[2]
        index = np.where(z_axis >= robot_boundary)[0]
        pcd = pcd.select_down_sample(index)
    
    return pcd

def combine_tool_pcs(pcs, robot_poses):
    for pose in robot_poses:
        print pose

    standard_pose = robot_poses[0]
    
    transformations = [np.matmul(standard_pose, transformation_util.get_transformation_matrix_inverse(pose)) for pose in robot_poses]
    
    copy_pcs = []
    i = 0
    for pc in pcs:
        pc.transform(transformations[i])
        copy_pcs.append(pc)
        i += 1
        
    return copy_pcs, standard_pose

def get_tool_pointcloud_combinations(pcs, robot_poses, num_member=1):
    assert len(pcs) == len(robot_poses)
    
    index_list = [i for i in range(len(pcs))]
    combs = list(combinations(index_list, num_member))
    
    combined_pcs = []
    combined_robot_poses = []
    
    for comb in combs:
        pc, robot_pose = combine_tool_pcs([pcs[i] for i in comb], [robot_poses[i] for i in comb])
        combined_pcs.append(pc)
        combined_robot_poses.append(robot_pose)
    
    return combined_pcs, combined_robot_poses

def get_average_distance(pc1, pc2):
    distance = pc1.compute_point_cloud_distance(pc2)
    return np.average(distance)

def get_fitness(pc1, pc2, max_correspondence_distance=0.005):
    return o3d.registration.evaluate_registration(pc1, pc2, max_correspondence_distance).fitness

# adapted from https://github.com/intel-isl/Open3D/issues/1437
# a singleton design: https://www.tutorialspoint.com/python_design_patterns/python_design_patterns_singleton.htm
class PointCloudPerception(object):
    __instance = None
    @staticmethod
    def getInstance():
        if PointCloudPerception.__instance == None:
            PointCloudPerception()
        return PointCloudPerception.__instance 
    
    def __init__(self):
        if PointCloudPerception.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            PointCloudPerception.__instance = self
            
            self.master_device_id = 0
            self.master_device_config_file_name = constants.get_azure_config_file_name("master")
            self.master_device = self.get_device(self.master_device_config_file_name, self.master_device_id)
            
            self.sub_device_id = 1
            self.sub_device_config_file_name = constants.get_azure_config_file_name("sub")
            self.sub_device = self.get_device(self.sub_device_config_file_name, self.sub_device_id)            
            
            self.intrinsic = {}
            self.intrinsic["master"] = o3d.camera.PinholeCameraIntrinsic(2048, 1536, 981.9638061523438, 981.630126953125, 1020.3690795898438, 790.924560546875)
            self.intrinsic["sub"]    = o3d.camera.PinholeCameraIntrinsic(2048, 1536, 980.5792846679688, 980.0770263671875, 1020.9423828125, 780.787353515625)

            self.extrinsic = {}
            translation = np.array([-0.032, -0.002, 0.004])
            quaternion = np.array([-0.048, -0.001, -0.001, 0.999])

            rot_master = np.array([[1, -0.000406638, -0.000661461],
                                   [0.000472794, 0.99465, 0.103305],
                                    [0.000615914, -0.103306, 0.994649]])

            rot_sub = np.array([[0.999998, 0.00135182, -0.00174016],
                                   [-0.00117766, 0.995336, 0.0964568],
                                   [0.00186243, -0.0964545, 0.995336]])

            p_master = np.array([-32.0644, -2.02311, 4.00924]) / 1000.
            p_sub    = np.array([-32.0452, -2.11213, 4.07797]) / 1000.

            T_master = transformation_util.get_transformation_matrix_with_rotation_matrix(rot_master, p_master)
            T_master =  transformation_util.get_transformation_matrix_inverse(T_master)
            T_sub = transformation_util.get_transformation_matrix_with_rotation_matrix(rot_sub, p_sub)
            T_sub =  transformation_util.get_transformation_matrix_inverse(T_sub)
            
            self.extrinsic["master"] = T_master 
            self.extrinsic["sub"]    = T_sub 

    def get_intrinsic(self, frame):
        return self.intrinsic[frame].intrinsic_matrix

    def get_extrinsic(self, frame):
        return self.extrinsic[frame]
    
    def camera_info(self, frame):
        return self.get_intrinsic(frame), self.get_extrinsic(frame)

    def get_device(self, config_file_name, device_index):
        config_dict = {}
        with open(config_file_name, "r") as read_file:
            config_dict = json.load(read_file)
        config = o3d.io.AzureKinectSensorConfig(config_dict)
        device = o3d.io.AzureKinectSensor(config)
        if not device.connect(device_index):
            raise RuntimeError('Failed to connect to sensor')        
        return device
    
    def retrieve_point_cloud(self, frame):
        device = None
        
        if frame == "master":
            device = self.master_device
        else:
            device = self.sub_device

        rgbd = None
        while rgbd is None:
            rgbd = device.capture_frame(enable_align_depth_to_color = True)
        
        color = np.asarray(rgbd.color).astype(np.uint8)
        depth = np.asarray(rgbd.depth).astype(np.float32) / 1000.0

        return color, depth
    
    def get_current_point_cloud(self, frame=""):
        color, depth = self.retrieve_point_cloud(frame)
        # Make sure depth and color imgs are same size.
        depth = o3d.geometry.Image(depth)
        color = o3d.geometry.Image(color)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=1.0, convert_rgb_to_intensity=False)
        intrinsic = self.intrinsic[frame]
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        return pcd    

def merge_adjacent_pcs(pcs):
    reordered_pcs = []
    
    for i in range(int(round(len(pcs) / 2.0))):
        pair = []
        pair.append(copy.deepcopy(pcs[i * 2]))
        if i * 2 + 1 < len(pcs):
            pair.append(copy.deepcopy(pcs[i * 2 + 1]))
        reordered_pcs.append(pair)
    
    result = []
    num_total_keep = 0
    for pair in reordered_pcs:
        if len(pair) > 1:
            pc1 = copy.deepcopy(pair[0])
            pc2 = copy.deepcopy(pair[1])
            
            before_fitness = get_fitness(pc1, pc2)
            print "=========================================================="
            print "[before] average fitness 0-1 is ", get_fitness(pc1, pc2)
            print "[before] average fitness 1-0 is ", get_fitness(pc2, pc1)
            
            aligned_set, _, _ = align_pcd_select_size([pc1, pc2])
            pc1, pc2 = aligned_set[0], aligned_set[1]
            
            object_size = max(pc1.get_oriented_bounding_box().extent)
            print "object size is ", object_size
            print "[after] average fitness 0-1 is ", get_fitness(pc1, pc2)
            print "[after] average fitness 1-0 is ", get_fitness(pc2, pc1)
            after_fitness = get_fitness(pc1, pc2)
            if before_fitness <= after_fitness and after_fitness > 0.9:
                print "keep this!"
                merged_pc = merge([pc1, pc2], paint=False)
                merged_pc = merged_pc.voxel_down_sample(0.001)
                result.append(merged_pc)
            else:
                print "do not keep!"
                num_total_keep += 2
                print "pc1 num points: ", np.asarray(pc1.points).shape[0]
                print "pc2 num points: ", np.asarray(pc1.points).shape[0]
                if np.asarray(pc1.points).shape[0] >= np.asarray(pc2.points).shape[0]:
                    print "keep pc1"
                    result.append(pc1)
                else:
                    print "keep pc2"
                    result.append(pc2)
        else:
            result.append(pair[0])
    
    return result

# return the point cloud merged from different cams in the robot frame
def get_current_point_cloud(frames=["master", "sub"], to_transform=True, paint=True):
    pc_perception = PointCloudPerception.getInstance()
    pcs = []
    for frame in frames:
        count = 20
        got_pc = False
        while not got_pc or count > 0:
            pc = pc_perception.get_current_point_cloud(frame)
            got_pc = pc.has_points()
            count -= 1
        if to_transform:
            pc = transform_to_robot_frame(pc, frame)
        pcs.append(pc)
        rospy.sleep(0.5)

    return pcs

def get_camera_info():
    pc_perception = PointCloudPerception.getInstance()
    pc_perception.get_current_point_cloud('master')
    
    return pc_perception.camera_info()

def is_container(tool_name, goal_name, Tgoal_tool):
    tool_pc = get_tool_mesh(tool_name)
    goal_pc = get_goal_mesh(goal_name)
    
    tool_pc.transform(Tgoal_tool)
    
    print "[pointcloud_util][is_container] bb_overlap_percentage: ", bb_overlap_percentage(goal_pc, tool_pc)
    
    if bb_overlap_percentage(goal_pc, tool_pc) > 0.3: # TODO: need to tune this value
        return True
    
    return False

def contact_surface(tool_name, goal_name, Tgoal_tool, r=.05):
    tool_mesh_path = constants.get_tool_mesh_path(tool_name)
    goal_mesh_path = constants.get_goal_mesh_path(goal_name)
    
    if is_container(tool_name, goal_name, Tgoal_tool):
        return calc_contact_surface_container_wrapper(tool_mesh_path, goal_mesh_path, Tgoal_tool)
    else:
        return calc_contact_surface_mesh_wrapper(tool_mesh_path, goal_mesh_path, Tgoal_tool, r=r)

def calc_contact_surface_from_camera_view_mesh_wrapper(src_mesh_path, goal_mesh_path, Tgoal_tool, r=.15, get_goal_surface=False):
    """
    Wrapper around calc_contact_surface method that takes in paths to meshes rather than pointclouds.

    @src_mesh_path: str.
    @goal_mesh_path: str.
    @Tgoal_tool: (4x4) ndarray. Inital transformation of src tool.
    @r: float, Search radius multiplier for points in contact surface.

    """

    gp = GeneratePointcloud()
    goal_pnts = gp.load_pointcloud(goal_mesh_path)
    src_pnts = gp.load_pointcloud(src_mesh_path)
    src_pcd = tpc_to_o3d(ToolPointCloud(src_pnts, normalize=False))
    # Transform src points for proper alignment.
    src_pnts = np.asarray(src_pcd.transform(Tgoal_tool).points)
    src_surface = calc_contact_surface_from_camera_view(src_pnts, goal_pnts, r)

    if get_goal_surface:
        ret = (src_surface, calc_contact_surface_from_camera_view(goal_pnts, src_pnts, r))
    else:
        ret = src_surface

    return ret

def calc_contact_surface_container_wrapper(src_mesh_path, goal_mesh_path, Tgoal_tool, get_goal_surface=True):
    """
    Wrapper around calc_contact_surface_container method that takes in paths to meshes
    rather than pointclouds.

    @src_mesh_path: str.
    @goal_mesh_path: str.
    @Tgoal_tool: (4x4) ndarray. Inital transformation of src tool.

    """
    gp = GeneratePointcloud()
    goal_pnts = gp.load_pointcloud(goal_mesh_path)
    src_pnts = gp.load_pointcloud(src_mesh_path)
    src_pcd = tpc_to_o3d(ToolPointCloud(src_pnts, normalize=False))
    
    # Transform src points for proper alignment.
    src_pnts = np.asarray(src_pcd.transform(Tgoal_tool).points)

    src_surface = calc_tool_contact_surface_container(goal_pnts, src_pnts)

    if get_goal_surface:
        _, goal_surface = get_contact_surface(src_pnts, goal_pnts)
        ret = (src_surface, goal_surface)
    else:
        ret = src_surface

    return ret

# the line_pc and pc has 2 intersect points
def get_line_pc_intersect(line_pc, goal_pc):
    distance = line_pc.compute_point_cloud_distance(goal_pc)
    distance_close_points = np.where(np.array(distance) < 0.005)
    
    point_1_index = np.min(distance_close_points)
    point_2_index = np.max(distance_close_points)
    
    print "point_1_index: ", point_1_index
    print "point_2_index: ", point_2_index
    line_pc.paint_uniform_color(np.array([1., 1., 0.]))
    colors = np.asarray(line_pc.colors)
    print "colors"
    print colors
    colors[point_1_index] = np.array([1., 0., 0])
    colors[point_2_index] = np.array([0., 0., 1])
    line_pc.colors = o3d.utility.Vector3dVector(colors)
    
    point_1 = np.array(line_pc.points)[point_1_index]
    point_2 = np.array(line_pc.points)[point_2_index]
    
    return point_1, point_2, line_pc

def pc_to_mesh(pc):
    """
    Creates a mesh from the point cloud and then samples n points from it.
    Returns o3d point cloud.
    """
    pc = deepcopy(pc)
    pc.compute_vertex_normals()

    radii = [0.005, 0.01, 0.02, 0.04, .1]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector(radii))

    return mesh

def sample_mesh(mesh, n=10000):
    return mesh.sample_points_poisson_disk(n)

def transform_pc(pc, T):
    origin_pc = copy.deepcopy(pc)
    origin_pc.transform(T)
    
    return origin_pc

def is_translational_changes(pc_start, pc_end):
    pc_start = copy.deepcopy(pc_start)
    pc_end = copy.deepcopy(pc_end)
    
    start_center = pc.start.get_oriented_bounding_box().get_center()
    end_center = pc.start.get_oriented_bounding_box().get_center()

    pc_start.translate(end_center, relative = False)
    
    if get_average_distance(pc_start, pc_end) < 0.001:
        return True, end_center - start_center
    
    return False, None

def get_gripper_bb(T_ee):
    bb = None
    R = T_ee[:3, :3]
    p = T_ee[:3, 3]
    p[0] += .005
    min_boundary = [-.2, -.2, 0.0072]
    max_boundary = [.2, .2, .3]
    if constants.get_robot_platform() == constants.ROBOT_PLATFORM_BAXTER:
        bb = o3d.geometry.OrientedBoundingBox(center=np.array([0., 0., 0.]), R=np.identity(3), extent=np.array([0.05, 0.05, 0.05]))
    elif constants.get_robot_platform() == constants.ROBOT_PLATFORM_UR5E:
        bb = o3d.geometry.OrientedBoundingBox(center=p, R=R, extent=np.array([0.1, 0.27, 0.035]))

    elif constants.get_robot_platform() == constants.ROBOT_PLATFORM_KUKA:
        bb = o3d.geometry.OrientedBoundingBox(center=np.array([0., 0., 0.]), R=np.identity(3), extent=np.array([0.05, 0.05, 0.1]))
    
    return bb

def remove_gripper(pc, T_ee):
    bb = get_gripper_bb(T_ee)
    gripper_pc = pc.crop(bb)

    return remove_background(gripper_pc, pc, threshold=0.0005)

def pc_to_mesh(pc, mesh_type="ball pivot"):
    """
    Creates a mesh from the point cloud and then samples n points from it.
    Returns o3d triangle mesh.
    """
    pc = deepcopy(pc)
    pc.estimate_normals()

    if mesh_type == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=10)
        vertices_to_remove = densities < np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(vertices_to_remove)

    elif mesh_type == "ball pivot":

        dist = pc.compute_nearest_neighbor_distance()
        avg_dist = np.mean(dist)
        radius = 1.5 * avg_dist
        print "Radius ", radius
        radii = [radius, radius * 2]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc,
            o3d.utility.DoubleVector(radii))

    return mesh

def sample_mesh(mesh, n=10000):
    return mesh.sample_points_poisson_disk(n)

def meshlab_postprocess(fin, fout, n=5000, ball_radius=0.0149183):
    """
    Use meshlabserver to remesh pointcloud/mesh
    """
    script = mlx.FilterScript(file_in=fin, file_out=fout, ml_version='1.3.2')

    mlx.sampling.poisson_disk(script, n)
    mlx.remesh.ball_pivoting(script, ball_radius=ball_radius)
    mlx.smooth.taubin(script)
    mlx.clean.close_holes(script)

    script.run_script()

"""
the calculations need for object_monitor (the main logic of that): object pose in work space related function
"""
# pc order: master, sub
def _remove_backgrounds(background_pcs, foreground_pcs, threshold):
    isolated_pcs = []

    isolated_pcs.append(remove_background(background_pcs[constants.PC_MASTER_INDEX], foreground_pcs[constants.PC_MASTER_INDEX], threshold=threshold))
    isolated_pcs.append(remove_background(background_pcs[constants.PC_SUB_INDEX],    foreground_pcs[constants.PC_SUB_INDEX],    threshold=threshold))

    isolated_pcs = [pc.remove_statistical_outlier(10, 0.001)[0] for pc in isolated_pcs] # remove noise.

    return isolated_pcs

def _isolate_tool(foreground_pcs, Tee):
    R = Tee[:3, :3]
    p = Tee[:3, 3]
    bb = o3d.geometry.OrientedBoundingBox(center=p, R=R, extent=np.array([0.35, 0.65, 0.1])) # TODO: meiying, need to tune this value for each robot
    tool_pcs = []
    tool_pcs.append(foreground_pcs[0].crop(bb))
    tool_pcs.append(foreground_pcs[1].crop(bb))
    
    return tool_pcs

def _align_pc(object_pcs, standard_pc, visualize=False):
    Tworld_objects = []
    icp_evaluation = []
    
    i = 0

    for object_pc in object_pcs:
        add_color_normal(object_pc, paint=False)
        add_color_normal(standard_pc, paint=False)
        if object_pc.has_points():
            transformed_pcs, Ts, _ = align_pcd_select_size([object_pc, standard_pc]) # it is in this order, because it Tworld_goal is how the standard pc needs to do to get the scanne position
            if visualize:
                camera = ""
                if i == 0:
                    camera = "master"
                else:
                    camera = "sub"
                o3d.visualization.draw_geometries(transformed_pcs, "Aligned pcs from {}".format(camera))

            Tworld_objects.append(Ts[1])
            icp_evaluation.append(get_average_distance(transformed_pcs[0], transformed_pcs[1])) # we want the one with more points covered on the standard one

        i += 1
    # Select which pointcloud produces the best fit between the master and sub versions
    if len(Tworld_objects) == 0:
        return None
        
    if len(Tworld_objects) == 1:
        return Tworld_objects[0]
    
    Tworld_object = np.identity(4)
    max_icp_evaluation = max(icp_evaluation)
    if abs(icp_evaluation[0] - icp_evaluation[1]) / max_icp_evaluation < 0.05:
        print "Tworld_object: use the average"
        Tworld_object = transformation_util.average_transformation_matrix(Tworld_objects)
    elif icp_evaluation[1] ==  max_icp_evaluation:
        print "Tworld_object: choose master"
        Tworld_object = Tworld_objects[0]
    else:
        print "Tworld_object: choose sub"
        Tworld_object = Tworld_objects[1]
    
    return Tworld_object

def _get_tool_pose(Tee, tool_bbs, current_Tee, current_pcs, tool_name, visualize=False):
    T = transformation_util.get_fixed_frame_transformation(Tee, current_Tee)
    
    bbs = []
    bbs.append(transform_oriented_bounding_box(tool_bbs[0], T))
    bbs.append(transform_oriented_bounding_box(tool_bbs[1], T))

    tool_pcs = []
    tool_pcs.append(current_pcs[0].crop(bbs[0]))
    tool_pcs.append(current_pcs[1].crop(bbs[1]))
    
    tool_standard = get_tool_mesh(tool_name)
    Tworld_tool = _align_pc(tool_pcs, tool_standard, visualize)
    
    return Tworld_tool

def get_tool_bbs(Tee, background_pcs, foreground_pcs, threshold=.0099):
    # Process pc to remove uneeded stuff like gripper.
    background_pcs = [trim_pcd_workspace(pc) for pc in background_pcs]
    background_pcs = [remove_gripper(pc, Tee) for pc in background_pcs]

    foreground_pcs = [trim_pcd_workspace(pc) for pc in foreground_pcs]
    foreground_pcs = [remove_gripper(pc, Tee) for pc in foreground_pcs]

    tool_pcs = _remove_backgrounds(background_pcs, foreground_pcs, threshold=threshold)

    tool_bbs = []
    if tool_pcs[0].has_points():
        tool_bbs.append(tool_pcs[0].get_oriented_bounding_box())
    else:
        tool_bbs.append(o3d.geometry.OrientedBoundingBox(np.array([0., 0., 0.]), np.identity(3), np.array([0.1, 0.1, 0.1])))
    if tool_pcs[1].has_points():
        tool_bbs.append(tool_pcs[1].get_oriented_bounding_box())
    else:
        tool_bbs.append(o3d.geometry.OrientedBoundingBox(np.array([0., 0., 0.]), np.identity(3), np.array([0.1, 0.1, 0.1])))

    return Tee, tool_bbs, tool_pcs    

def get_Tee_tools(tool_name, Tee, tool_bbs, tool_pcs, sample_pcs, sample_Tees):
    tool_standard = get_tool_mesh(tool_name)
    Tworld_tool = _align_pc(tool_pcs, tool_standard)

    Tee_tools = []
    
    if not Tworld_tool is None:
        Tee_tools.append(np.matmul(transformation_util.get_transformation_matrix_inverse(Tee), Tworld_tool))
    
    for i in range(len(sample_pcs)):
        current_pc = sample_pcs[i]
        current_Tee = sample_Tees[i]
        Tworld_tool = _get_tool_pose(Tee, tool_bbs, current_Tee, current_pc, tool_name)
        if not Tworld_tool is None:
            Tee_tools.append(np.matmul(transformation_util.get_transformation_matrix_inverse(current_Tee), Tworld_tool))    

    return get_main_Tee_tool(Tee_tools)

def get_main_Tee_tool(Tee_tools):
    major_Tee_tools = transformation_util.find_major_transformations(Tee_tools) # find the Tee_tool that most samples agree with
    
    Tee_tool = np.identity(4)
    if major_Tee_tools is not None:
        Tee_tool = transformation_util.average_transformation_matrix(major_Tee_tools) # Average val of tool in end-effector frame
    
    return Tee_tool    

def get_goal_pose(goal_name, background_pcs, foreground_pcs, threshold=.004, Tstart=np.identity(4), visualize=False):
    """
    @Tstart identity matrix for tool use. For goal start Tstart is identity. for goalend its goalstart
    """
    background_pcs = [trim_pcd_workspace(pc) for pc in background_pcs]

    foreground_pcs = [trim_pcd_workspace(pc) for pc in foreground_pcs]
    foreground_pcs = [goal_basic_processing(pc) for pc in foreground_pcs]

    goal_pcs_final = _remove_backgrounds(background_pcs, foreground_pcs, threshold=threshold)

    goal_standard = get_goal_mesh(goal_name)

    Tworld_goal = _align_pc(goal_pcs_final, goal_standard, visualize)

    return Tworld_goal

"""
tool sub wrapper
"""
# this is for tool, the goal has a different one
def calculate_Tsource_substitute(source_name, sub_name, source_contact_index, object_type, task="", is_control=False):
    get_mesh_path_func = None
    if object_type == constants.OBJECT_TYPE_GOAL:
        get_mesh_path_func = constants.get_goal_mesh_path
    elif object_type == constants.OBJECT_TYPE_TOOL:
        get_mesh_path_func = constants.get_tool_mesh_path
        
    substitute_mesh_path = get_mesh_path_func(sub_name)
    source_mesh_path = get_mesh_path_func(source_name)
    
    print "[pointcloud_util][calculate_Tsource_substitute] source_mesh_path: ", source_mesh_path
    print "[pointcloud_util][calculate_Tsource_substitute] substitute_mesh_path: ", substitute_mesh_path
    
    n_iter = 3
    # Get numpy pointclouds from meshes
    gp = GeneratePointcloud()
    sub_pnts = gp.load_pointcloud(substitute_mesh_path)
    src_pnts = gp.load_pointcloud(source_mesh_path)
    # Turn pointclouds into TPC objects inorder to align tools.
    sub_tpc = ToolPointCloud(sub_pnts, normalize=False)
    src_tpc = ToolPointCloud(src_pnts, normalize=False)
    src_tpc.contact_pnt_idx = source_contact_index
    ts = None
    if object_type == constants.OBJECT_TYPE_GOAL:
        print "[pointcloud_util][calculate_Tsource_substitute] GOAL substitution"
        ts = GoalSubstitution(src_tpc, sub_tpc, voxel_size=0.02, visualize=False)
    elif object_type == constants.OBJECT_TYPE_TOOL:
        print "[pointcloud_util][calculate_Tsource_substitute] TOOL substitution"
        ts = ToolSubstitution(src_tpc, sub_tpc, voxel_size=0.02, visualize=False)
    # Calculate sub tool alignment and contact surface.
    if is_control:
        calculated_Tsource_substitute, cp = ts.get_random_contact_pnt()
    else:
        calculated_Tsource_substitute, cp = ts.get_T_cp(n_iter=n_iter)
    
    return calculated_Tsource_substitute, cp

def get_line_axis_pc(axis, length, color):
    axis_pc = o3d.geometry.LineSet()
    axis_pc.points = o3d.utility.Vector3dVector([np.array([0., 0., 0.]), axis * length])
    axis_pc.lines = o3d.utility.Vector2iVector([[0, 1],])
    axis_pc.paint_uniform_color(color)
    
    return axis_pc

def get_object_bb_dimension(object_name, object_type):
    get_pc_func = None
    if object_type == constants.OBJECT_TYPE_GOAL:
        get_pc_func = constants.get_goal_mesh
    elif object_type == constants.OBJECT_TYPE_TOOL:
        get_pc_func = constants.get_tool_mesh
    
    object_pc = get_pc_func(object_name)
    bb = object_pc.get_oriented_bounding_box()
    
    return bb.extent

def visualize_trajectory(tool, goal, tool_trajectories_world_frame, Tworld_goalstart, Tworld_goalend, index=0):
    pcs = []
    
    goal_pc = get_goal_mesh(goal)
    goal_start_pc = copy.deepcopy(goal_pc)
    goal_start_pc.transform(Tworld_goalstart)
    goal_end_pc = copy.deepcopy(goal_pc)
    if not Tworld_goalend is None:
        goal_end_pc.transform(Tworld_goalend)
    
    print "[pointcloud_util][visualize_trajectory] Tworld_goalstart"
    print Tworld_goalstart
    print "[pointcloud_util][visualize_trajectory] Tworld_goalend"
    print Tworld_goalend
    
    goal_start_pc.paint_uniform_color(np.array([0., 1., 0.]))
    goal_end_pc.paint_uniform_color(np.array([0., 0., 1.]))
    pcs.append(goal_start_pc)
    if not Tworld_goalend is None:
        pcs.append(goal_end_pc)
    
    tool_pc = get_tool_mesh(tool)
    tool_pc.paint_uniform_color(np.array([1., 0., 0.]))
    
    tool_trajectory = []
    tool_trajectory_world_frame = copy.deepcopy(tool_trajectories_world_frame[index])
    if len(tool_trajectory_world_frame) > 30:
        step = int(round(len(tool_trajectory_world_frame) * 1. / 30.))
        for i in range(0, len(tool_trajectory_world_frame), step):
            tool_trajectory.append(copy.deepcopy(tool_trajectory_world_frame[i]))
    else:
        for T in tool_trajectory_world_frame:
            tool_trajectory.append(copy.deepcopy(T))
    
    print "[pointcloud_util][visualize_trajectory] tool_trajectory"   
    for T in tool_trajectory:
        copy_tool_pc = copy.deepcopy(tool_pc)
        copy_tool_pc.transform(T)
        print T
        pcs.append(copy_tool_pc)
    
    x_axis = get_line_axis_pc(np.array([1., 0., 0.]), 0.5, np.array([1., 0., 0.]))
    y_axis = get_line_axis_pc(np.array([0., 1., 0.]), 0.5, np.array([0., 1., 0.]))
    z_axis = get_line_axis_pc(np.array([0., 0., 1.]), 0.5, np.array([0., 0., 1.]))
    
    pcs.append(x_axis)
    pcs.append(y_axis)
    pcs.append(z_axis)
    
    goal_x_axis = get_line_axis_pc(np.array([1., 0., 0.]), 0.1, np.array([1., 0., 0.]))
    goal_x_axis.transform(Tworld_goalstart)
    goal_y_axis = get_line_axis_pc(np.array([0., 1., 0.]), 0.1, np.array([0., 1., 0.]))
    goal_y_axis.transform(Tworld_goalstart)
    goal_z_axis = get_line_axis_pc(np.array([0., 0., 1.]), 0.1, np.array([0., 0., 1.]))
    goal_z_axis.transform(Tworld_goalstart) 
    
    pcs.append(goal_x_axis)
    pcs.append(goal_y_axis)
    pcs.append(goal_z_axis)
    
    tool_x_axis = get_line_axis_pc(np.array([1., 0., 0.]), 0.1, np.array([1., 0., 0.]))
    tool_x_axis.transform(tool_trajectory[0])
    tool_y_axis = get_line_axis_pc(np.array([0., 1., 0.]), 0.1, np.array([0., 1., 0.]))
    tool_y_axis.transform(tool_trajectory[0])
    tool_z_axis = get_line_axis_pc(np.array([0., 0., 1.]), 0.1, np.array([0., 0., 1.]))
    tool_z_axis.transform(tool_trajectory[0])
    
    pcs.append(tool_x_axis)
    pcs.append(tool_y_axis)
    pcs.append(tool_z_axis)    
    
    o3d.visualization.draw_geometries(pcs, "visualize trajectories")
