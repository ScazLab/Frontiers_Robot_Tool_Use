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

import threading
import random

import numpy as np

import rospy

from std_msgs.msg import Bool
from geometry_msgs.msg import Vector3, Quaternion

from control_wrapper.srv import GetPose

from tri_star.msg import TargetPositions

from tri_star import transformation_util
from tri_star.robot_util import Robot

class Calibration:
    def __init__(self):
        self.check_data_lock = threading.Lock()    
        
        self.reference_points = []
        self.second_points = []
        
        self.transformation_matrices = []
        
        self.reference_point = None
        self.second_point = None

        self.second_frame_name = rospy.get_param("second_frame_name")
        
        self.sample_repeatition = rospy.get_param("perception_repeatition")
        
        self.robot = Robot()
        
        rospy.Subscriber('/tri_star/aruco_3d_perceptions', TargetPositions, self.get_points)
    
    def get_end_effector(self):
        return self.robot.get_robot_pose()
    
    def get_transformation_matrix_from_msg(self, position_msg, quaternion_msg):
        position = np.array([position_msg.x, position_msg.y, position_msg.z])
        quaternion = np.array([quaternion_msg.x, quaternion_msg.y, quaternion_msg.z, quaternion_msg.w])
        return transformation_util.get_homogeneous_transformation_matrix_from_quaternion(quaternion, position)
        
    def get_points(self, data):
        set_reference = False
        set_second = False
        
        for info in data.positions:
            if info.frame == self.second_frame_name:
                self.second_point = self.get_transformation_matrix_from_msg(info.position, info.quaternion)
                set_second = True

    def get_transformation_matrix(self, reference_point, second_point):
        return np.matmul(reference_point, transformation_util.get_transformation_matrix_inverse(second_point))
        
    def hand_eye_calibration_shay(self):
        # Using Shah's method: https://www.semanticscholar.org/paper/Solving-the-Robot-World%2FHand-Eye-Calibration-Using-Shah/b3874687ff07e2a959e47651830f3c109fb47e8f
        # solve AX = YB given A is the robot, B is the camera
        # X is end_tool, Y is base_camera
        
        # reference: https://github.com/xumzsy/dVRK-Hand-Eye-Calibration/blob/master/MatlabCode/shah.m
        # https://github.com/NUbots/NUtilities/blob/master/autocal/CalibrationTools.cpp
        
        assert(len(self.reference_points) == len(self.second_points))
       
        n = len(self.reference_points) # number of samples
        
        #A = np.zeros((9 * n, 18))
        T = np.zeros((9, 9))
        #b = np.zeros((9 * n, 1))
        
        # find rotation
        for i in range(n):
            Ra = self.reference_points[i][:3, :3]
            Rb = self.second_points[i][:3, :3]
            #A[9 * i:9 * (i + 1), :] = np.concatenate([np.kron(Rb, Ra), -np.identity(9)], axis=1)
            T += np.kron(Rb, Ra)
        
        u, s, vh = np.linalg.svd(T)
        v = vh.T
        
        print "u: ", u
        print "s: ", s
        print "v: ", v
        #index = 0
        
        #for i in range(len(s)):
            #print("distance: ", abs(s[i] - n))
            #if abs(s[i] - n) < 0.001:
                #index = i
                #print("eigenvalue = ", s[i])
                #break
        un = u[:, 0]
        vn = v[:, 0]
        
        print "un: ", un
        print "vn: ", vn
        
        Vx = vn.reshape((3, 3)).T
        Vy = un.reshape((3, 3)).T
        
        print "Vx: ", Vx
        print "Vy: ", Vy
        
        #alpha = np.sign(Vx) * 1.0 / np.cbrt(np.linalg.det(Vx))
        #beta = np.sign(Vy) * 1.0 / np.cbrt(np.linalg.det(Vy))
        
        alpha = 1.0 / np.cbrt(np.linalg.det(Vx))
        beta = 1.0 / np.cbrt(np.linalg.det(Vy))
        
        print "alpha: ", alpha
        print "beta: ", beta
        
        Rx_vectorize = alpha * Vx
        Ry_vectorize = beta * Vy
        
        print "Vec(Rx): ", Rx_vectorize
        print "Vec(Ry): ", Ry_vectorize
        
        Rx = Rx_vectorize.reshape((3, 3))
        Ry = Ry_vectorize.reshape((3, 3))
        
        print "Rx: ", Rx
        print "Ry: ", Ry    
        
        # orthogonalize Rx, Ry
        ux, sx, vhx = np.linalg.svd(Rx)
        Rx = np.matmul(ux, vhx)
        uy, sy, vhy = np.linalg.svd(Ry)
        Ry = np.matmul(uy, vhy)
        
        print "Rx (after orthogonalize): ", Rx
        print "Ry (after orthogonalize): ", Ry        
        
        # find translation
        # left [tY; tX] = right
        left = np.zeros((3 * n, 6))
        right = np.zeros((3 * n, 1))
        
        for i in range(n):
            left[3 * i:3 * (i + 1), :] = np.concatenate([-self.reference_points[i][:3, :3], np.identity(3)], axis=1)
            right[3 * i:3 * (i + 1), :] = np.array([self.reference_points[i][:3, 3]]).T - np.matmul(np.kron(np.array([self.second_points[i][:3, 3]]), np.identity(3)), Ry.reshape((9, 1), order='F'))
        
        print "left"
        print left
        print "right"
        print right
        t = np.linalg.lstsq(left, right)[0]
        
        print "t: ", t
        
        print "tx: ", t[:3, :]
        print "ty: ", t[3:, :]
        
        Tx = transformation_util.get_transformation_matrix_with_rotation_matrix(Rx, np.array([t[0][0], t[1][0], t[2][0]]))
        Ty = transformation_util.get_transformation_matrix_with_rotation_matrix(Ry, np.array([t[3][0], t[4][0], t[5][0]]))
        
        return Tx, Ty

    def run(self):
        clear_data = rospy.get_param("clear_data")
        get_data = rospy.get_param("get_data")
        file_name = rospy.get_param("file_name")
        
        if clear_data:
            self.robot_points = []
            self.second_points = []
        else:
            data = np.load(file_name)
            self.reference_points = data['name1']
            self.second_points = data['name2']
            
            self.robot_points = [np.array(i) for i in self.reference_points.tolist()]
            self.second_points = [np.array(i) for i in self.second_points.tolist()]
        
        print "reference(robot) points", self.reference_points
        print "second(camera) points", self.second_points
        
        if get_data:
            i = 0
            stop = False
            while not stop:
                stop_str = raw_input("press any key to continue to unlock the robot, or s to check result")
                if stop_str.lower() != "s":
                    self.robot.free_drive_robot(True)
                    print "robot unclocked!"
                    raw_input("press any key to lock")
                    self.robot.free_drive_robot(False)
                    print "locked!"
                    raw_input("press any key to get the current location")
                    j = 0
                    # automatically take multiple points and take average
                    reference_points = []
                    second_points = []
                    average_reference_point = None
                    average_camera_point = None
                    print "getting a few samples..."
                    while j < self.sample_repeatition:
                        if self.second_point is not None:
                            self.reference_point = self.get_end_effector()
                            print "REF POINT 1 ", self.reference_point
                            print "REF POINT 2 ", self.second_point
                            print "got end effector"
                            reference_points.append(self.reference_point.copy())
                            second_points.append(self.second_point.copy())
                            self.reference_point = None
                            self.second_point = None
                            j += 1
                    average_reference_point = transformation_util.average_transformation_matrix(reference_points)
                    average_second_point = transformation_util.average_transformation_matrix(second_points)
                    if average_reference_point is not None and average_second_point is not None:
                        print "get point ", i
                        print "reference(robot) point"
                        print average_reference_point
                        print "secondary(camera) point"
                        print average_second_point
                        reject_str = raw_input("press s to ignore this point, otherwise accept it: ")
                        if reject_str != "s":
                            i += 1
                            self.reference_points.append(average_reference_point.copy())
                            self.second_points.append(average_second_point.copy())
                        print("================================================================")                        
                else:
                    print "reference points"
                    print self.reference_points
                    print "camera points"
                    print self.second_points
                    np.savez(file_name, name1=np.array(self.reference_points), name2=np.array(self.second_points))
            
                    end_tool_matrix, base_camera_matrix = self.hand_eye_calibration_shay()
                    
                    for i in range(len(self.reference_points)):
                        Ra = self.reference_points[i][:3, :3]
                        ta = self.reference_points[i][3, :3]
                        Rb = self.second_points[i][:3, :3]
                        tb = self.second_points[i][3, :3]
                        rotation_error = 1 - transformation_util.get_vector_length(np.matmul(Ra, end_tool_matrix[:3, :3]) - np.matmul(base_camera_matrix[:3, :3], Rb)) / 8
                        print "sample ", i, "rotation error", rotation_error
                        
                        translation_error = transformation_util.get_vector_length(np.matmul(Ra, ta) - np.matmul(Rb, tb))
                        print "sample ", i, "translation error", translation_error
                   
                    print "-----------------------------" 
                    print "end_tool_matrix: "
                    print end_tool_matrix
                    print "robot_camera_matrix: " 
                    print base_camera_matrix
                    print "please fill it in the <robot>_config.xml file: " 
                    print base_camera_matrix.flatten().tolist()
        else:
            print "reference points"
            print self.reference_points
            print "camera points"
            print self.second_points
    
            end_tool_matrix, base_camera_matrix = self.hand_eye_calibration_shay()
            
            for i in range(len(self.reference_points)):
                Ra = self.reference_points[i][:3, :3]
                ta = self.reference_points[i][3, :3]
                Rb = self.second_points[i][:3, :3]
                tb = self.second_points[i][3, :3]
                rotation_error = 1 - transformation_util.get_vector_length(np.matmul(Ra, end_tool_matrix[:3, :3]) - np.matmul(base_camera_matrix[:3, :3], Rb)) / 8
                print "sample ", i, "rotation error", rotation_error
                
                translation_error = transformation_util.get_vector_length(np.matmul(Ra, ta) - np.matmul(Rb, tb))
                print "sample ", i, "translation error", translation_error
           
            print "-----------------------------"
            print "end_tool_matrix: "
            print end_tool_matrix
            print "robot_camera_matrix (please fill it in the perception.xml file): "
            print base_camera_matrix
            print base_camera_matrix.flatten().tolist()
    
if __name__ == '__main__':
    try:
        rospy.init_node('calibrate_camera_robot', anonymous=True)

        calibration = Calibration()

        calibration.run()

    except rospy.ROSInterruptException:
        pass
