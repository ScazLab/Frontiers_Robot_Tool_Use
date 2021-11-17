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
    def __init__(self, pcs, robot_poses):
        self.pcs = [copy.deepcopy(pc) for pc in pcs]
        self.robot_poses = robot_poses
        self.initial_axis = None

        self.angles = self.get_angle()
    
    def get_angle(self):
        standard_pose = self.robot_poses[0]
        angles = []
        transformations = [np.matmul(standard_pose, transformation_util.get_transformation_matrix_inverse(pose)) for pose in self.robot_poses]
        for i in range(1, len(transformations)):
            # Get screwaxis and angle
            S, theta = transformation_util.get_exponential_from_transformation_matrix(transformations[i])
            axis = S[:3] # First 3 values in S are the screw axis
            if self.initial_axis is None:
                self.initial_axis = axis
                self.initial_vel   = S[3:] / np.linalg.norm(axis) # last three are velocity
            if transformation_util.is_opposite_direction([axis], [self.initial_axis], error=np.deg2rad(15.0)):
                theta = -theta
            angles.append(theta)
        
        return angles
    
    def objective_function(self, x):
        """
        For a given set of tool scans, measures the mean distance between the bounding boxs
        of each scan after being rotated and the bounding box of the merged scan (ideally should be 0)
        """
        error_pcs = [copy.deepcopy(pc) for pc in self.pcs]
        
        # Update the screw axis around which to rotate the scans
        S = np.zeros(6)
        S[:3] = transformation_util.normalize(self.initial_axis + x[:3])
        S[3:] = x[3:] / 100  + self.initial_vel 

        bbs = np.zeros((len(self.angles), 24)) # Stores the bb of each scan as flattened array
        for i in range(len(self.angles)):
            theta = self.angles[i]
            T = transformation_util.get_transformation_matrix_from_exponential(S, theta)
            error_pcs[i].transform(T)
            
            bb_pnts = np.asarray(error_pcs[i].get_axis_aligned_bounding_box().get_box_points())
            bbs[i, : ] = bb_pnts.reshape(1,-1) # Flatten the bb
            
        merged_pc = pointcloud_util.merge(error_pcs, paint=False)
        merged_bb = np.asarray(merged_pc.get_axis_aligned_bounding_box().get_box_points())

        bb_dist = np.apply_along_axis(lambda bb: np.linalg.norm(bb - merged_bb.reshape(1,-1)),
            axis=1, arr=bbs)
        
        return bb_dist.mean()* 10.     # cm3
    
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
        return 15. - angle # >= 0
    
    def constraint_3(self, x):
        point_diff = x[3:]
        return 0.5 - transformation_util.get_vector_length(point_diff) # >= 0, now in cm 
    
    def optimize(self, x0=np.zeros(6)):
        
        con1 = {'type': 'eq', 'fun': self.constraint_1}
        con2 = {'type': 'ineq', 'fun': self.constraint_2}
        con3 = {'type': 'ineq', 'fun': self.constraint_3}
        cons = [con1, con3]

        sol = minimize(self.objective_function, x0, method='SLSQP')

        S = np.zeros(6)
        axis = transformation_util.normalize(self.initial_axis +  sol.x[:3])
        vel = self.initial_vel  + sol.x[3:] / 100
        
        S[:3] = axis
        S[3:] = vel
        x0 = x0[0]
        print "angles: ", self.angles
        print "axis: ", axis
        print "initial axis: ", self.initial_axis
        print "---------------------------------------"
        print "vel: ", vel
        print "initial vel: ", self.initial_vel       
        
        pcs = [copy.deepcopy(pc) for pc in self.pcs]
        for i, theta in enumerate(self.angles):
            T = transformation_util.get_transformation_matrix_from_exponential(S, theta)
            pcs[i].transform(T)
        
        o3d.visualization.draw_geometries(pcs, "corrected with minimize transform")
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
    
    def pointcloud_callback(self, data):
        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1get point cloud"
    
    def pointcloud2_callback(self, data):
        print "get point cloud"    
    
    def get_point_cloud(self):
        self.is_get_image = True
        
        while self.depth_image is None:
            rospy.sleep(0.1)
        
        self.is_get_image = False

        depth_image = np.asarray(self.bridge.imgmsg_to_cv2(self.depth_image, "32FC1"))     

        o3d_image = o3d.geometry.Image(depth_image)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(1024, 1024, 504.1826477050781, 504.3000183105469, 512.6762084960938, 509.60888671875)
        
        extrinsic = np.array([[1., -0.000406638, -0.000661461, -0.0320644],
                              [0.000472794, 0.99465, 0.103305, -0.00202311],
                              [0.000615914, -0.103306, 0.994649, 0.00400924],
                              [0., 0., 0., 1.]])
        extrinsic = transformation_util.get_transformation_matrix_inverse(extrinsic)
        pointcloud = o3d.geometry.PointCloud.create_from_depth_image(depth=o3d_image, intrinsic=intrinsic, extrinsic=extrinsic, depth_scale=1.0)
        
        self.depth_image = None
        
        pointcloud = pointcloud_util.transform_to_robot_frame(pointcloud, "master")
        
        return pointcloud

    def write_camera_info_to_file(self, fn):
        print "Writing camera info to file..."
        intrinsic, extrinsic = pointcloud_util.get_camera_info()

        original_stdout = sys.stdout
        with open(fn, 'w') as f:
            sys.stdout = f # Change the standard output to the file we created.
            print "intrinsic: {}".format(intrinsic)
            print "extrinsic: {}".format(extrinsic)
            sys.stdout = original_stdout  # Reset the standard output to its original value
        
    def generate_object_point_cloud_single_angle(self):
        robot_joint_angles = []
        robot_joint_names = []
        opposite_index = -1
        
        if constants.get_robot_platform() == constants.ROBOT_PLATFORM_KUKA:
            robot_joint_names = constants.get_robot_joint_names()
            robot_joint_initial_angles = [5.0, 8.06884615385, -21.3624, 34.6233802817, 316.841830986]
            raw_input("press any key to reset the robot...")
            self.robot.set_robot_angle(robot_joint_names, robot_joint_initial_angles)
            # 5.0, 305.0, 6
            min_value = 5.0
            max_value = 320.0
            num_samples = 2
            interval = (max_value - min_value) / (num_samples - 1.0)
            print "interval is ", interval
            for joint_1 in [min_value + i * interval for i in range(num_samples)]:
            #for joint_1 in [5 + i * 41.25 for i in range(9)]:
                print "angle: ", joint_1
                angles = [j for j in robot_joint_initial_angles]
                angles[0] = joint_1
                robot_joint_angles.append(angles)
        elif constants.get_robot_platform() == constants.ROBOT_PLATFORM_UR5E:
            # TODO - JAKE: change the values here
            robot_joint_names = constants.get_robot_joint_names()

            upright_joint_initial_angles = [173.72, -0.01, -127.51, 37.38, 89.95, -270.0] # upright pos
            platform_joint_initial_angles = [173.72, -0.01, -149.45, 58.65, 89.95, -270.0] # upright pos
            tilted_joint_initial_angles = [173.72, -0.01, -143.56, 85.10, 89.95, -270.0] # towards camera pos
            side_joint_initial_angles = [173.72, -19.78, -125.56, -40.05, 0.0, -270.0] # towards camera pos
            
            UPRIGHT, TILTED, SIDE, PLATFORM = 1,2, 3, 4

            good_input = False
            while not good_input:
                choice = raw_input("Press: \n\t'1' for upright position, \n\t'2' for tilted position towards camera, \n\t'3' for side view \n\t'4' for platform...")
                good_input = choice in ["1", "2", "3", "4"]

            if int(choice) == UPRIGHT:
                robot_joint_initial_angles = upright_joint_initial_angles
                scan_pos = UPRIGHT

            elif int(choice) == TILTED:
                robot_joint_initial_angles = tilted_joint_initial_angles
                scan_pos = TILTED
            elif int(choice) == SIDE:
                robot_joint_initial_angles = side_joint_initial_angles
                scan_pos = SIDE
            elif int(choice) == PLATFORM:
                robot_joint_initial_angles = platform_joint_initial_angles
                scan_pos = PLATFORM

            self.robot.set_robot_angle(robot_joint_names, robot_joint_initial_angles)
            # 5.0, 305.0, 6
            min_value = -270.
            max_value = 90.
            num_samples = 24 # make sure this is an even number
            rot_joint = 5 # This is the wrist joint closest to endeffector.
            interval = (max_value - min_value) / num_samples * 1.0
            print "interval is ", interval
            index = 0
            for joint_1 in [min_value + i * interval for i in range(num_samples)]:
                print "angle: ", joint_1
                angles = [j for j in robot_joint_initial_angles]
                angles[rot_joint] = joint_1
                robot_joint_angles.append(angles)
                if joint_1 == min_value + 180.:
                    opposite_index = index
                index += 1
        
        raw_input("make sure the robot is holding the tool, press any key to get the point cloud...")
        
        is_manual = raw_input("would you like to take pcs manually?")
        
        self.robot_poses = []
        self.raw_tool_pcs = []
        pcs = []
        
        if is_manual == "y":
            is_finish = False
            while not is_finish:
                raw_input("press any key to get the current result")
                pose = self.robot.get_robot_pose()
                raw_pc = pointcloud_util.get_current_point_cloud(["master"], paint=False)
                raw_pc = pointcloud_util.tool_basic_processing(raw_pc, mode="scan", paint=False)
                raw_pc = pointcloud_util.remove_robot(raw_pc, mode="scan")
                raw_pc = pointcloud_util.remove_noise(raw_pc, eps=0.005, min_points=7, paint=False)
                o3d.visualization.draw_geometries([raw_pc], "pc")
                is_keep = raw_input("keep the point?")
                if is_keep == "y":
                    self.robot_poses.append(pose)
                    pcs.append(raw_pc) 
                rospy.sleep(1.0)               
                is_finish = raw_input("finish?") == "y"
        else:
            i = 0
            init_pos = self.robot.get_robot_pose()
            self.gripper_pos = np.around(init_pos[:3, 3], decimals=3)
            for angle in robot_joint_angles:
                i += 1
                print "i:", i
                self.robot.set_robot_angle(robot_joint_names, angle)
                print "reach pose"
                rospy.sleep(2.0)
                pos = self.robot.get_robot_pose()
                print "POS: \n", pos
                
                print "start getting point cloud"
                raw_pc = pointcloud_util.get_current_point_cloud(["master"], paint=False)
                print raw_pc
                #Crop out tool
                raw_pc = pointcloud_util.tool_basic_processing(raw_pc, mode="scan", pos=init_pos,
                                                               paint=False)
                print "PC processed!"
                # Remove robot arm points.
                if scan_pos == PLATFORM: 
                    print "Removing Platform"
                    raw_pc = pointcloud_util.remove_platform(raw_pc, init_pos)
                self.raw_tool_pcs.append(copy.deepcopy(raw_pc))
                # Remove noise.
                raw_pc, _ = raw_pc.remove_statistical_outlier(90, 0.001)

                print "finish getting point cloud"

                pos[:3, 3] = np.around(pos[:3, 3], decimals=3)
                if i == 1:
                    self.gripper_pos = pos[:3, 3]
                
                self.robot_poses.append(pos)
                rospy.sleep(1.0)
                pcs.append(raw_pc)
        
        print "finish getting the data needed, processing..."

        o3d.visualization.draw_geometries(pcs, "scanned pcs")
        
        if scan_pos == UPRIGHT or scan_pos == PLATFORM:
            x0 = np.array([np.array([-0.00199436,  0.00190794, -0.62361108,  0.48919917,  1.6175556,   0.02860613])])
        elif scan_pos == TILTED:
            x0 = np.array([  0.68309339, -0.08303256, -1.08556536,  0.17357975,  1.47610081,  0.53896121])
        elif scan_pos == SIDE:
            x0 = np.array([  0.68309339, -0.08303256, -1.08556536,  0.17357975,  1.47610081,  0.53896121])
        # fine tune the axis
        optimizer = Optimizer(pcs, self.robot_poses)
        object_pc = optimizer.optimize(x0)        
        
        object_pc, _ = object_pc.remove_statistical_outlier(80, 0.001)
        final_pc = self.sample_mesh(object_pc, n=10000)
        final_pc = pointcloud_util.center_pc(final_pc)
        mesh = pointcloud_util.pc_to_mesh(final_pc)
        
        o3d.visualization.draw_geometries([mesh], "combined")
        
        keep_result_str = raw_input("keep this result(y/n)?")
        keep_result = False
        if keep_result_str == "y":
            keep_result = True
             
        return mesh, keep_result

    def proc_robot_poses(self, robot_poses):
        trans = np.array([p[:3, 3] for p in robot_poses])

        return pc

    def generate_object_point_cloud(self, object_name):
        is_continue = True
        saved_pc = []
        pc = None

        raw_save_path = constants.pointcloud_raw_dir()
        sample_path = os.path.join(raw_save_path, object_name)

        
        file_util.create_dir(sample_path)

        sample_index = file_util.get_index_in_dir(sample_path, file_name_template="^"+object_name+"_[0-9]+\.ply")
        
        print "sample_index: ", sample_index
        
        if sample_index > 1:
            for i in range(1, sample_index):
                saved_pc.append(o3d.io.read_point_cloud(os.path.join(sample_path, "{}_{}.ply".format(object_name, i))))
            for each_pc in saved_pc:
                if pc is None:
                    pc = each_pc
                else:
                    result, _,_ = pointcloud_util.align_pcd_select_size([pc, each_pc])

                    pc = pointcloud_util.merge(result, paint=False)
                    pc = pc.voxel_down_sample(0.001)


                    o3d.visualization.draw_geometries([pc], "merge current")
            to_save = raw_input("save current result?")
            if to_save == "y":
                return pc
        
        while is_continue:
            object_pc, keep_result = self.generate_object_point_cloud_single_angle()
            if keep_result:
                pointcloud_util.write_mesh(os.path.join(sample_path, "{}_{}.ply".format(object_name, sample_index)), object_pc, write_ascii=True)
                
                sample_raw_path = os.path.join(sample_path, "raw_{}".format(sample_index))
                file_util.create_dir(sample_raw_path)

                for i, pcd in enumerate(self.raw_tool_pcs):
                    fn = os.path.join(sample_raw_path, "{}.ply".format(i))
                    print "FN: ", fn
                    pointcloud_util.write_mesh(pcd, fn)

                # Save robot poses to file.
                np.savez(os.path.join(sample_raw_path,"robot_poses"), *self.robot_poses)
                self.write_camera_info_to_file(os.path.join(sample_raw_path,"camera_info.txt"))

                sample_index += 1
                saved_pc.append(object_pc)
                
                print "pc: ", pc
                if pc is None:
                    pc = object_pc
                else:
                    transformation, _, _ = pointcloud_util.align_pcd_select_size([pc, object_pc])
                    merged_pc = pointcloud_util.merge(transformation, paint=False)
                    o3d.visualization.draw_geometries([merged_pc], "merge current")
                   
                    keep_result_str = raw_input("keep this result(y/n)?")
                    if keep_result_str == "y":
                        pc = merged_pc
                        
                finish_str = raw_input("finish(y/n)?")
                if finish_str == "y":
                    is_continue = False
        return pc

    def run(self):
        is_continue = True
        object_pcs = {}
        constants.pointcloud_dir()
        while is_continue:
            object_name = raw_input("object name (do not include space): ")
            pc = self.generate_object_point_cloud( object_name)
            pc_path = os.path.join(constants.pointcloud_tool_dir(), "{}.ply".format(object_name))
            pointcloud_util.write_mesh(pc_path, pc, write_ascii=True)
            is_continue_str = raw_input("scan another object?(y/n)")
            if is_continue_str == "n":
                is_continue = False

# assume the robot arm is kuka  
if __name__ == '__main__':
    try:
        rospy.init_node('scan_object', anonymous=True)

        scan_object = ScanObject()

        scan_object.run()

    except rospy.ROSInterruptException:
        pass
