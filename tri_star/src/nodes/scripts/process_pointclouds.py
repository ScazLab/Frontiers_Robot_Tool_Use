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
import copy
import sys
import open3d as o3d
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d

import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2

from tri_star import pointcloud_util
from tri_star import robot_util
from tri_star import file_util
from tri_star import transformation_util
from tri_star import constants
from tri_star import transformations as tfs

class Optimizer(object):
    def __init__(self, pcs, robot_poses, gripper_pos):
        self.pcs = [copy.deepcopy(pc) for pc in pcs]
        self.robot_poses = robot_poses
        self.initial_axis = None
        self.initial_point = None  
        self.gripper_pos = gripper_pos
        self.angles = self.get_angle()

    def get_angle(self):
        standard_pose = self.robot_poses[0]
        angles = []
        transformations = [np.matmul(standard_pose, transformation_util.get_transformation_matrix_inverse(pose)) for pose in self.robot_poses]
        for i in range(1, len(transformations)):
            S, theta = transformation_util.get_exponential_from_transformation_matrix(transformations[i])
            axis = S[:3]
            if self.initial_axis is None:
                self.initial_axis = axis
                self.inital_vel   = S[3:] / np.linalg.norm(axis)
            if transformation_util.is_opposite_direction([axis], [self.initial_axis], error=np.deg2rad(15.0)):
                theta = -theta
            angles.append(theta)
        
        return angles

    def objective_function(self, x):
        error_pcs = [copy.deepcopy(pc) for pc in self.pcs]
        
        S = np.zeros(6)
        S[:3] = transformation_util.normalize(self.initial_axis + x[:3])
        S[3:] = x[3:] / 100  + self.inital_vel 

        bbs = np.zeros((len(self.angles), 24))
        for i in range(len(self.angles)):
            theta = self.angles[i]
            T = transformation_util.get_transformation_matrix_from_exponential(S, theta)
            error_pcs[i].transform(T)
            bb_pnts = np.asarray(error_pcs[i].get_axis_aligned_bounding_box().get_box_points())

            bbs[i, : ] = bb_pnts.reshape(1,-1)
            
        merged_pc = pointcloud_util.merge(error_pcs, paint=False)
        merged_bb = np.asarray(merged_pc.get_axis_aligned_bounding_box().get_box_points())

        bb_dist = np.apply_along_axis(lambda bb: np.linalg.norm(bb - merged_bb.reshape(1,-1)),
            axis=1, arr=bbs)
        
        return bb_dist.mean()* 10     # cm3


    def constraint_1(self, x):
        axis = x[:3] + self.initial_axis
        return transformation_util.get_vector_length(axis) - 1.0 # = 0
    
    def constraint_2(self, x):
        """
        Angle between new axis and original one has to be within 15 degree
        """
        
        axis = transformation_util.normalize(x[:3] + self.initial_axis)
        angle = abs(transformation_util.get_angle([axis], [self.initial_axis]))
        if angle > np.pi / 2:
            angle = np.pi - angle
        angle = np.rad2deg(angle)
        return 3. - angle # >= 0
    
    def constraint_3(self, x):
        point_diff = x[3:]
        return 0.5 - transformation_util.get_vector_length(point_diff) # >= 0, now in cm 
    
    def optimize(self):
        x0 = np.array([-0.00199436,  0.00190794, -0.62361108,  0.48919917,  1.6175556,   0.02860613])
        con1 = {'type': 'eq', 'fun': self.constraint_1}
        con2 = {'type': 'ineq', 'fun': self.constraint_2}
        con3 = {'type': 'ineq', 'fun': self.constraint_3}
        cons = [con1, con2, con3]
        
        b_axis = (-0.05, 0.05)
        b_point = (-0.5, 0.5) # in cm
        bnds = (b_axis, b_axis, b_axis, b_point, b_point, b_point)

        sol = minimize(self.objective_function, x0, method='SLSQP')
        print "Sol: ", sol.x
        print "object value: ", self.objective_function(sol.x)
        print "initial objective value: ", self.objective_function(x0)
        print "---------------------------------------"
        pcs = [copy.deepcopy(pc) for pc in self.pcs]
        i = 0
        S = np.zeros(6)
        S[:3] = transformation_util.normalize(self.initial_axis +  sol.x[:3])
        S[3:] = self.inital_vel  + sol.x[3:] / 100
         
        for theta in self.angles:
            T = transformation_util.get_transformation_matrix_from_exponential(S, theta)
            pcs[i].transform(T)
            i += 1

        merged_pc = pointcloud_util.merge(pcs, paint=False)
        
        return merged_pc

class ScanObject:
    def __init__(self):
        self.robot = robot_util.Robot()
        
        self.bridge = CvBridge()
        self.is_get_image = False
        
        self.color_image = None
        self.depth_image = None
        self.pointcloud = None
    
    def color_image_callback(self, data):
        if self.is_get_image:
            if self.color_image is None:
                self.color_image = data
    
    def depth_image_callback(self, data):
        if self.is_get_image:
            if self.depth_image is None:
                self.depth_image = data
    
        
    def generate_object_point_cloud_single_angle(self, raw_dir):

        i = 0
        pcs = []
        robot_poses = []
        tool = "white_scooper"

        T_file = "{}/robot_poses.npz".format(raw_dir)
        robot_pose_zip = np.load(T_file)
        robot_f_names = ["arr_{}".format(j) for j in range(len(robot_pose_zip))] 
        for f in robot_f_names:
            robot_poses.append(robot_pose_zip[f])
            i += 1
            print "i:", i
            pc_fn = "/{}/{}.ply".format(raw_dir, i-1)
            raw_pc = o3d.io.read_point_cloud(pc_fn)

            raw_pc, _ = raw_pc.remove_statistical_outlier(90, 0.001)

            pcs.append(raw_pc)

        print "finish getting the data needed, processing..."

        self.gripper_pos = robot_poses[0][:3, 3]
        print self.gripper_pos
        optimizer = Optimizer(pcs, robot_poses, self.gripper_pos)
        object_pc = optimizer.optimize()        

        object_pc, _ = object_pc.remove_statistical_outlier(80, 0.001)
        mesh = pointcloud_util.pc_to_mesh(object_pc)
        
        return mesh
            

    def mesh_to_pc(self, obj, n=3000):

        tool_path = os.path.join(constants.tool_mesh_dir(), "{}.ply".format(obj))
        save_path = os.path.join(constants.pointcloud_dir(), "new_tool_meshes", "{}.ply".format(obj))
        print tool_path
        try:
            mesh = o3d.io.read_triangle_mesh(tool_path)
        
            pc = mesh.sample_points_poisson_disk(n)
            pc, _ = pointcloud_util.center_pc(pc)
            o3d.visualization.draw_geometries([pc], "combined")

            mesh =  pointcloud_util.pc_to_mesh(pc)
            o3d.io.write_triangle_mesh(save_path, mesh)
            print "FINISHED"
        except RuntimeError:
            pass

    def generate_object_point_cloud(self, obj_name):

        obj_path = os.path.join(constants.pointcloud_raw_dir(), obj_name)
        i = 1
        for raw in os.listdir(obj_path):
            if not ".ply" in raw:
                path = os.path.join(obj_path, raw)
                print obj_path
                mesh = self.generate_object_point_cloud_single_angle(path)
                o3d.io.write_triangle_mesh(os.path.join(obj_path, "{}_{}.ply".format(obj_name, i)), mesh)
                i += 1

    def merge_scans(self, obj_name):
        
        save_dir = constants.pointcloud_tool_dir()
        tool_dir = os.path.join(constants.pointcloud_raw_dir(), obj_name)

        fs = [f for f in os.listdir(tool_dir) if os.path.isfile(os.path.join(tool_dir, f))]
        meshes = [o3d.io.read_triangle_mesh(os.path.join(tool_dir,f)) for f in fs]

        if len(fs) > 1:
            print fs
            pcs  = [pointcloud_util.sample_mesh(m, 20000) for m in meshes]
            print pcs
            res,_,_ = pointcloud_util.align_pcd_select_size(pcs)
            final_pc = pointcloud_util.merge(res, paint=False)
            final_mesh = pointcloud_util.pc_to_mesh(final_pc)
        else: 
            final_mesh = meshes[0]

        o3d.visualization.draw_geometries([final_mesh], "combined")

        save = raw_input("Save (y/n): ")
        if save == "y":
            fn = os.path.join(save_dir, "{}.off").format(obj_name)
            o3d.io.write_triangle_mesh(fn, final_mesh)

    def center_pcs(self):
        tool_dir = constants.pointcloud_tool_dir()
        for f in os.listdir(tool_dir):
            path = os.path.join(tool_dir,f)
            pcd = o3d.io.read_point_cloud(path)
            cent = pcd.get_center()
            print "prev cent: ", cent
            pcd.translate(cent * -1.)
            print "after cent: ", pcd.get_center()
            o3d.io.write_point_cloud(path, pcd)

    def pcs_to_mesh(self):
        tool_dir = constants.pointcloud_tool_dir()
        mesh_dir = os.path.join(constants.pointcloud_dir(), "tool_meshes")
        for f in os.listdir(tool_dir):
            path = os.path.join(tool_dir,f)
            pcd = o3d.io.read_point_cloud(path)
            mesh = self.upsample_pc(pcd)

            save_dir = os.path.join(mesh_dir, f)
            o3d.io.write_triangle_mesh(save_dir, mesh)

    def meshlab_remesh(self, obj):
        fin = os.path.join(constants.pointcloud_dir(), "tools", "{}.ply".format(obj))
        fout = os.path.join(constants.pointcloud_dir(), "meshlab_tools", "{}.ply".format(obj))
        pointcloud_util.meshlab_postprocess(fin, fout, n=10000)

    def smooth_mesh(self, obj_name):
        tool_path = os.path.join(constants.pointcloud_dir(),'new_tool_meshes', "{}.ply".format(obj_name))
        save_dir = os.path.join(constants.tool_mesh_dir(), "smoothed_tools")
        print tool_path
        mesh_in = o3d.io.read_triangle_mesh(tool_path)
        mesh_out = mesh_in.filter_smooth_laplacian(5)
        mesh_out.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh_out])

    def run(self):

        is_continue = True
        object_pcs = {}
        raw_dir = constants.pointcloud_raw_dir()
        
        funcs = {"1": self.generate_object_point_cloud, "2": self.merge_scans,
                 "3": self.meshlab_remesh, "4": self.mesh_to_pc,
                 "5": self.smooth_mesh}
        func_key = raw_input("Press\n\t 1 to recalculate scans from raws\n\t 2 to merge scans\n\t 3 to use meshlab\n\t 4 to mesh-->pc\n\t 5 to smooth mesh...")
        func = funcs[func_key]

        obj_name = raw_input("Obj name (or 'all' for all of them): ")
        if obj_name == "all":
            objs = os.listdir(raw_dir)
        else:
            objs = [obj_name]
        for o in objs:
            func(o)

# assume the robot arm is kuka  
if __name__ == '__main__':
    try:
        rospy.init_node('scan_object', anonymous=True)

        scan_object = ScanObject()

        scan_object.run()

    except rospy.ROSInterruptException:
        pass
