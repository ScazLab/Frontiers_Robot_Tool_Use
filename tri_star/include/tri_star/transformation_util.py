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

import random
import numpy as np
import copy
from itertools import combinations

from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.stats import kstest
import open3d as o3d

from tri_star import transformations as tfs
from tri_star import kinematics as kt
from tri_star import averageQuaternions as aq
from tri_star import datastructure_util
from tri_star import constants

# cluster angles
MAX_NUM_ANGLE_GROUP = 15

def similar_translation(vector_1, vector_2, threshold=np.radians(5.0)):
    if is_zero_vector(vector_1, error=0.001) and is_zero_vector(vector_2, error=0.001):
        #print "both are zero vector"
        return True
    
    if is_zero_vector(vector_1, error=0.001) or is_zero_vector(vector_2, error=0.001):
        #print "only one is zero vector"
        return False

    return abs(get_angle([vector_1], [vector_2])) < threshold

def similar_rotation(rotation_1, rotation_2, axis_threshold=0.005, point_threshold=0.005):
    angle_1, axis_1, point_1 = get_axis_angle_from_quaternion(rotation_1)
    angle_2, axis_2, point_2 = get_axis_angle_from_quaternion(rotation_2)
    
    if same_rotation_helper(rotation_1, rotation_2):
        #print "same rotation"
        return True
    
    if angle_1 < np.radians(1.0) and angle_2 < np.radians(1.0):
        #print "small angle"
        return True
    
    if not same_translation(axis_1, axis_2, axis_threshold):
        #print "axis is not similar"
        return False
    
    if not same_translation(point_1, point_2, point_threshold):
        #print "point is not similar"
        return False

    return True

def similar_pose(pose_1, pose_2, rotation_error = 5.0, translation_error = 0.01):
    translation_1, rotation_1 = decompose_homogeneous_transformation_matrix_to_rotation_matrix(pose_1)
    translation_2, rotation_2 = decompose_homogeneous_transformation_matrix_to_rotation_matrix(pose_2)

    if get_vector_length(translation_1 - translation_2) > translation_error:
        return False
    
    rotation_1_x = rotation_1.T[0]
    rotation_1_y = rotation_1.T[1]
    rotation_1_z = rotation_1.T[2]
    rotation_2_x = rotation_2.T[0]
    rotation_2_y = rotation_2.T[1]
    rotation_2_z = rotation_2.T[2]
    
    if not is_similar_vector(rotation_1_x, rotation_2_x, error = rotation_error):
        #print "x very different"
        return False

    if not is_similar_vector(rotation_1_y, rotation_2_y, error = rotation_error):
        #print "y very different"
        return False
    
    if not is_similar_vector(rotation_1_z, rotation_2_z, error = rotation_error):
        #print "z very different"
        return False
    
    return True

def similar_pose_test(pose_1, pose_2, rotation_error = 5.0, translation_error = 0.01, compare_translation = [True, True, True]):
    translation_1, rotation_1 = decompose_homogeneous_transformation_matrix_to_rotation_matrix(pose_1)
    translation_2, rotation_2 = decompose_homogeneous_transformation_matrix_to_rotation_matrix(pose_2)
  
    if get_vector_length(translation_1 - translation_2) > translation_error:
        #print "translation length ", get_vector_length(translation_1 - translation_2), " > ", translation_error
        return False
    else:
        similar_sign = translation_1 * translation_2 > 0
        large_enough_difference = abs(translation_1 - translation_2) < 0.015
        both_sign_difference = np.logical_or(similar_sign, large_enough_difference)
        all_condition = np.logical_or(both_sign_difference, np.array([not i for i in compare_translation]))
        #print "compare translation"
        print compare_translation
        if not np.alltrue(all_condition):
            #print "translation_1: ", translation_1
            #print "translation_2: ", translation_2
            #print "similar sign: "
            #print translation_1 * translation_2 > 0
            #print "large enough difference:"
            #print "sign not consistent"
            return False
    
    rotation_1_x = rotation_1.T[0]
    rotation_1_y = rotation_1.T[1]
    rotation_1_z = rotation_1.T[2]
    rotation_2_x = rotation_2.T[0]
    rotation_2_y = rotation_2.T[1]
    rotation_2_z = rotation_2.T[2]
    
    if not is_similar_vector(rotation_1_x, rotation_2_x, error = rotation_error):
        #print "x very different"
        return False

    if not is_similar_vector(rotation_1_y, rotation_2_y, error = rotation_error):
        #print "y very different"
        return False
    
    if not is_similar_vector(rotation_1_z, rotation_2_z, error = rotation_error):
        #print "z very different"
        return False
    
    return True

def same_translation(vector_1, vector_2, threshold=0.005):
    return kt.get_distance(vector_1, vector_2) < threshold

def same_rotation_helper(rotation_1, rotation_2):
    alpha_1, beta_1, gamma_1 = tfs.euler_from_quaternion(rotation_1)
    alpha_2, beta_2, gamma_2 = tfs.euler_from_quaternion(rotation_2)
    
    if abs(alpha_1 - alpha_2) < np.radians(5.0) and abs(beta_1 - beta_2) < np.radians(5.0) and abs(gamma_1 - gamma_2) < np.radians(5.0):
        return True
    
    return False    

def same_rotation(rotation_1, rotation_2, axis_threshold=0.005, point_threshold=0.005, angle_threshold=np.radians(5.0)):
    angle_1, axis_1, point_1 = get_axis_angle_from_quaternion(rotation_1)
    angle_2, axis_2, point_2 = get_axis_angle_from_quaternion(rotation_2)
    
    if same_rotation_helper(rotation_1, rotation_2):
        #print "same rotation"
        return True
    
    if angle_1 < np.radians(1.0) and angle_2 < np.radians(1.0):
        #print "small angle"
        return True
    
    if not same_translation(axis_1, axis_2, axis_threshold):
        #print "axis is not same"
        return False
    
    if not same_translation(point_1, point_2, point_threshold):
        #print "point is not same"
        return False
    
    #print "angle difference", abs(angle_1 - angle_2)
    if abs(angle_1 - angle_2) > angle_threshold:
        #print "angle is not same"
        return False
    
    return True

# pose_1 and pose_2 are matrices
def same_pose(pose_1, pose_2, translation_threshold=0.05, axis_threshold=0.005, point_threshold=0.005, angle_threshold=np.radians(5.0)):
    position_1, quaternion_1 = decompose_homogeneous_transformation_matrix(pose_1)
    position_2, quaternion_2 = decompose_homogeneous_transformation_matrix(pose_2)
    
    position = same_translation(position_1, position_2, translation_threshold)
    rotation = same_rotation(quaternion_1, quaternion_2, axis_threshold, point_threshold, angle_threshold)
    
    #print "position is same: ", position
    #print "quaternion is same: ", rotation
    
    return position and rotation

def opposite_screw_axis(screw_axis_1, screw_axis_2, angle_error = 5.0, v_error = 5.0, w_zero_vector_error=0.01, v_zero_vector_error=0.02):
    #print "opposite screw axis"
    return same_screw_axis(-screw_axis_1, screw_axis_2, angle_error, v_error, w_zero_vector_error, v_zero_vector_error)

def same_screw_axis(screw_axis_1, screw_axis_2, angle_error = 5.0, v_error = 5.0, w_zero_vector_error=0.01, v_zero_vector_error=0.02):
    w1 = np.array([screw_axis_1[0], screw_axis_1[1], screw_axis_1[2]])
    v1 = np.array([screw_axis_1[3], screw_axis_1[4], screw_axis_1[5]])
    
    w2 = np.array([screw_axis_2[0], screw_axis_2[1], screw_axis_2[2]])
    v2 = np.array([screw_axis_2[3], screw_axis_2[4], screw_axis_2[5]])
    
    #print "w1: ", w1
    #print "v1: ", v1
    #print "w2: ", w2
    #print "v2: ", v2
    
    if kt.is_zero_vector(w1, error=w_zero_vector_error) and kt.is_zero_vector(w2, error=w_zero_vector_error):
        #print "both are zero"
        #print "v1 and v2 angle", np.degrees(abs(get_angle(np.array([v1]), np.array([v2]))))
        if abs(get_angle(np.array([v1]), np.array([v2]))) > np.radians(angle_error):
            #print "v1 and v2 angle", abs(get_angle(np.array([v1]), np.array([v2]))), " > ", np.radians(angle_error)
            return False
        return True
    
    if kt.is_zero_vector(w1, error=w_zero_vector_error) or kt.is_zero_vector(w2, error=w_zero_vector_error):
        #print "only 1 is zero"
        #print "w1: ", kt.is_zero_vector(w1, error=0.01)
        #print "w2: ", kt.is_zero_vector(w2, error=0.01)
        return False

    if abs(get_angle(np.array([w1]), np.array([w2]))) > np.radians(angle_error):
        #print "w1 and w2 angle", abs(get_angle(np.array([w1]), np.array([w2]))), " > ", np.radians(angle_error)
        return False
    
    if kt.is_zero_vector(v1, error=v_zero_vector_error) and kt.is_zero_vector(v2, error=v_zero_vector_error):
        #print "both v's are zero"
        return True
    
    if kt.is_zero_vector(v1, error=v_zero_vector_error) or kt.is_zero_vector(v2, error=v_zero_vector_error):
        #print "only 1 of v is zero"
        #print "v1: ", v1
        #print "v1: ", kt.is_zero_vector(v1, error=v_zero_vector_error)
        #print "v2: ", v2
        #print "v2: ", kt.is_zero_vector(v2, error=v_zero_vector_error)
        return False    
    
    if not is_similar_vector(v1, v2, v_error): # may needs to be the same vector?
        #print "v1: ", v1
        #print "v2: ", v2       
        #print "v_error: ", v_error
        #print "v1 and v2 are not similar"
        return False
    
    #print "the screw axis is the same"
    
    return True

# return whether the rotation direction is the same, and whether the screw axis itself is the same
def same_screw_axis_with_same_direction(screw_axis_1, screw_axis_2, angle_1, angle_2, angle_error = 5.0, v_error = 5.0):
    #print "[transformation_util][same_screw_axis_with_same_direction] v_error: ", v_error
    if same_screw_axis(screw_axis_1, screw_axis_2, angle_error, v_error, v_zero_vector_error=0.01):
        if kt.close_to(angle_1, 0, error=np.deg2rad(5)) or kt.close_to(angle_2, 0, error=np.deg2rad(5)):
            return True, True
        elif angle_1 / abs(angle_1) * angle_2 / abs(angle_2) > 0:
            return True, True
    elif opposite_screw_axis(screw_axis_1, screw_axis_2, angle_error, v_error, v_zero_vector_error=0.01):
        if kt.close_to(angle_1, 0, error=np.deg2rad(5)) or kt.close_to(angle_2, 0, error=np.deg2rad(5)):
            return True, False
        elif angle_1 / abs(angle_1) * angle_2 / abs(angle_2) < 0:
            return True, False
    
    return False, False

def get_axis_angle_from_quaternion(quaternion):
    matrix = tfs.quaternion_matrix(quaternion)
    angle, axis, point = get_axis_angle_from_matrix(matrix)
    return angle, axis, point

def get_axis_angle_from_matrix(matrix):
    angle, axis, point = tfs.rotation_from_matrix(matrix)
    return angle, axis, point

# input is list of matrices
def average_transformation_matrix(matrices):
    quaternions = []
    translations = []
    
    for matrix in matrices:
        translation, quaternion = decompose_homogeneous_transformation_matrix(matrix)
        quaternions.append([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
        translations.append([translation[0], translation[1], translation[2]])
    
    quaternion = aq.averageQuaternions(np.array(quaternions))
    quaternion_average = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    translation_average = np.mean(np.array(translations), axis=0)
    
    return get_homogeneous_transformation_matrix_from_quaternion(quaternion_average, translation_average)

def is_same_vector(v1, v2, rtol = 0.01, atol = 0.001):
    return np.allclose(v1, v2, rtol, atol) or np.allclose(v2, v1, rtol, atol)

def is_similar_vector(v1, v2, error = 5.0): # error in degrees
    angle = abs(get_angle(np.array([v1]), np.array([v2])))
    #print "angle: ", np.degrees(angle)
    #print "error: ", error
    return is_same_vector(v1, v2) or angle < np.radians(error)

def is_zero_vector(v, error):
    return kt.is_zero_vector(v, error)

# v1: np.array([...])
# v2: np.array([...])
def is_colinear(v1, v2, error): # error in radians
    #print "v1: ", v1
    #print "v2: ", v2
    if is_same_vector(v1, v2):
        return True
    angle = get_angle([v1], [v2])
    #print "angle: ", np.degrees(angle)
    #print "error: ", np.degrees(error)
    #print "abs(angle) < error: ", abs(angle) < error
    #print "abs(angle - np.pi): ", abs(angle - np.pi) < error
    #print "abs(angle) < error or abs(angle - np.pi) < error: ", abs(angle) < error or abs(angle - np.pi) < error
    return abs(angle) < error or abs(angle - np.pi) < error

def is_opposite_direction(v1, v2, error):
    if is_same_vector(v1, v2):
        return False
    angle = get_angle(v1, v2)
    print "angle: ", np.degrees(angle)
    #print "error: ", np.degrees(error)
    #print "abs(angle) < error: ", abs(angle) < error
    #print "abs(angle - np.pi): ", abs(angle - np.pi) < error
    #print "abs(angle) < error or abs(angle - np.pi) < error: ", abs(angle) < error or abs(angle - np.pi) < error
    return abs(angle - np.pi) < error    

def get_quaternion_from_direction(initial_direction, direction):
    return None

def get_default_quaternion():
    #return quaternion_about_axis(0.0, (1.0, 0.0, 0.0))
    return get_quaternion_from_rotation_matrix(np.idendity(3))

def get_quaternion_from_rotation_matrix(matrix):
    result = np.identity(4)
    result[:3, :3] = matrix
    return tfs.quaternion_from_matrix(result)

def get_rotation_matrix_from_quaternion(quaternion):
    alpha, beta, gamma = tfs.euler_from_quaternion(quaternion)
    rotation_matrix = tfs.euler_matrix(alpha, beta, gamma)
    
    return rotation_matrix[:3, :3]

def get_transformation_matrix_with_rotation_matrix(rotation_matrix, translation_vector):
    # translation_vector: np.array([1, 2, 3])
    result = np.identity(4)
    result[:3, 3] = translation_vector
    result[:3, :3] = rotation_matrix
    return result

def get_homogeneous_transformation_matrix_from_quaternion(rotation_quaternion, translation_vector):
    # translation_vector: np.array([1, 2, 3])
    alpha, beta, gamma = tfs.euler_from_quaternion(rotation_quaternion)
    rotation_matrix = tfs.euler_matrix(alpha, beta, gamma)

    result = rotation_matrix
    result[:3, 3] = translation_vector

    return result

def decompose_homogeneous_transformation_matrix_to_rotation_matrix(matrix):
    translation = np.array([matrix[0, 3], matrix[1, 3], matrix[2, 3]])
    rotation_matrix = matrix[:3, :3]
    return translation, rotation_matrix

def decompose_homogeneous_transformation_matrix(matrix):
    translation = np.array([matrix[0, 3], matrix[1, 3], matrix[2, 3]])
    # translation.x = matrix[0, 3]
    # translation.y = matrix[1, 3]
    # translation.z = matrix[2, 3]

    quaternion_matrix = matrix.copy()
    quaternion_matrix[:3, 3] = 0
    quaternion = tfs.quaternion_from_matrix(quaternion_matrix)

    rotation = np.array([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
    # rotation.x = quaternion[0]
    # rotation.y = quaternion[1]
    # rotation.z = quaternion[2]
    # rotation.w = quaternion[3]

    return translation, rotation

def normalize(vector):
    if vector is None:
        return None
    if get_vector_length(vector) == 0:
        return vector
    else:
        return vector / get_vector_length(vector)

def get_vector_length(vector):
    return np.linalg.norm(vector)

# input: list of list, like [1, 0., 0.]
def get_angle(start_vector, end_vector):
    result = np.dot(start_vector[0], end_vector[0]) / (get_vector_length(start_vector) * get_vector_length(end_vector))
    result = min(1.0, result)
    result = max(-1.0, result)
    return np.arccos(result)

def get_rotation_matrix(axis, angle):
    # formula: https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/
    return np.cos(angle) * np.identity(3) + (1 - np.cos(angle)) * np.matmul(np.array([[axis[0][0]], [axis[0][1]], [axis[0][2]]]), np.array([[axis[0][0], axis[0][1], axis[0][2]]])) + np.sin(angle) * kt.get_skew_symmetric_matrix(axis)

# input: np.array([[....]])
# axis needs to be provided if the two directions are colinear
def get_rotation_matrix_from_directions(start_direction, end_direction, axis = None):
    if not is_colinear(start_direction[0], end_direction[0], error=np.deg2rad(1.)):
    #if axis is None:
        axis = normalize(np.cross(start_direction, end_direction))
    else:
        axis = normalize(axis)
    angle = get_angle(start_direction, end_direction)
    
    rotation_matrix = None

    if kt.is_zero_vector(axis):
        rotation_matrix = np.identity(3)
    else:
        #print "axis", axis
        # print "angle", np.degrees(get_angle(start_direction, end_direction))
        rotation_matrix = get_rotation_matrix(axis, angle)
    
    return rotation_matrix

def radian_to_degrees(radians):
    return np.degrees(radians)

def degrees_to_radian(degrees):
    return np.radians(degrees)

def get_transformation_matrix(start_direction, end_direction, final_location, axis = None):
    rotation_matrix = get_rotation_matrix_from_directions(start_direction, end_direction, axis)
    transformation_matrix = kt.get_homogenous_transformation_matrix(rotation_matrix, final_location)
    return transformation_matrix

def get_transformation_matrix_inverse(matrix):
    return kt.get_homogenous_transformation_matrix_inverse(matrix)

def average_vector(vector_list):
    if len(vector_list) == 0:
        return None
    return np.mean(np.array(vector_list), axis = 0)

def average_screw_axis(screw_axis_list):
    averaged = average_vector(screw_axis_list)
    
    return normalize_screw_axis(averaged)

def get_exponential_from_transformation_matrix(T, threshold=0.01, default=None):
    S, theta = kt.get_exponential_representation_from_homogenous_transformation_matrix(T, threshold, default=default)
    S = S[0]
    
    return S, theta

def get_transformation_matrix_from_exponential(S, theta):
    #print "[transformation_util][get_transformation_matrix_from_exponential]S:", S
    #print "[transformation_util][get_transformation_matrix_from_exponential]theta:", theta
    S = np.array([S])
    return kt.get_homogenous_transformation_matrix_from_exponential_representation(S, theta)

def get_adjoint_representation_matrix(T):
    return kt.get_adjoint_representation_matrix(T)

def normalize_screw_axis(V):
    V = V.reshape((1, 6))
    S = kt.get_screw_axis(V)
    S = V.reshape((6, ))
    return S

# D = PAP-1
# current to original
def convert_A_to_D(A, P):
    return np.matmul(np.matmul(P, A), get_transformation_matrix_inverse(P))

# A = P-1DP
# original to current
def convert_D_to_A(D, P):
    return np.matmul(get_transformation_matrix_inverse(P), np.matmul(D, P))

def get_eular_angle_from_quaternion(quaternion):
    angles = tfs.euler_from_quaternion(quaternion)
    return angles[0], angles[1], angles[2]

def get_rotation_matrix_from_euler_angle(ax, ay, az):
    return tfs.euler_matrix(ax, ay, az)[:3, :3]

def get_euler_angle_from_transformation_matrix(transformation_matrix):
    position, quaternion = decompose_homogeneous_transformation_matrix(transformation_matrix)
    ax, ay, az = get_eular_angle_from_quaternion(quaternion)
    x, y, z = position[0], position[1], position[2]
    return x, y, z, ax, ay, az

def get_transformation_matrix_from_euler_angle(x, y, z, ax, ay, az):
    rotation_matrix = get_rotation_matrix_from_euler_angle(ax, ay, az)
    position = np.array([x, y, z])
    return get_transformation_matrix_with_rotation_matrix(rotation_matrix, position)

def get_Tobject_aruco(corners): # order bottom right, top right, top left, bottom left
    x_direction_14 = corners[3, :]  - corners[0, :]
    x_direction_23 = corners[2, :]  - corners[1, :]
    
    x_direction_14_length = np.linalg.norm(x_direction_14)
    x_direction_23_length = np.linalg.norm(x_direction_23)
    
    x_direction_14 = normalize(x_direction_14)
    x_direction_23 = normalize(x_direction_23)
    
    if not kt.close_to(np.dot(x_direction_14, x_direction_23), 1, error=1e-4):
        raise Exception("The corners of the markers are not right. The line of 1(top left)-4(bottom left) and 2(top right)-3(bottom right) are not parallel.")
    if not kt.close_to(x_direction_14_length, x_direction_23_length, error=1e-3):
        raise Exception("The corners of the markers are not right. The length of 1(top left)-4(bottom left) and 2(top right)-3(bottom right) are not the same.")
        
    x_length = (x_direction_14_length + x_direction_23_length) / 2
    x_direction = (x_direction_14 + x_direction_23) / 2
    x_direction = normalize(x_direction)
    
    y_direction_12 = corners[1, :]  - corners[0, :]
    y_direction_43 = corners[2, :]  - corners[3, :]
    
    y_direction_12_length = np.linalg.norm(y_direction_12)
    y_direction_43_length = np.linalg.norm(y_direction_43)
    
    y_direction_12 = normalize(y_direction_12)
    y_direction_43 = normalize(y_direction_43)
    
    if not kt.close_to(np.dot(y_direction_12, y_direction_43), 1, error=1e-4):
        raise Exception("The corners of the markers are not right. The line of 1(top left)-2(top right) and 4(bottom left)-3(bottom right) are not parallel.")
    if not kt.close_to(y_direction_12_length, y_direction_43_length, error=1e-3):
        raise Exception("The corners of the markers are not right. The length of 1(top left)-2(top right) and 4(bottom left)-3(bottom right) are not the same.")
    
    y_length = (y_direction_12_length + y_direction_43_length) / 2
    y_direction = (y_direction_12 + y_direction_43) / 2
    y_direction = normalize(y_direction)

    if not kt.close_to(np.dot(x_direction, y_direction), 0, error=1e-2):
        raise Exception("The corners of the markers are not right. x and y are not perpendicular.")
    if not kt.close_to(x_length, y_length, error=1e-3):
        raise Exception("The corners of the markers are not right. The length of x and y are not the same.")
        
    z_direction = np.cross(y_direction, x_direction)
    z_direction = normalize(z_direction)
    
    # make sure y is exactly perpendicular
    y_direction = np.cross(x_direction, z_direction)
    y_direction = normalize(y_direction)
    
    centroid = np.mean(corners, axis=0)
    
    Tobject_aruco = np.vstack([y_direction, x_direction, z_direction, centroid]).T
    Tobject_aruco = np.vstack([Tobject_aruco, np.array([[0, 0, 0, 1]])])
    
    return Tobject_aruco

def get_cluster_min_samples(num_samples):
    return int(num_samples * 1.0 / 3.0)

def find_major_transformations(matrices):
    """
    Cluster transformations based on the rotation and translation
    """
    if len(matrices) == 1 or len(matrices) == 2:
        return matrices
    
    translations = []
    rotations = []
    
    for matrix in matrices:
        # rotation represented by euler angles.
        x, y, z, ax, ay, az = get_euler_angle_from_transformation_matrix(matrix)
        translations.append(np.array([x, y, z]))
        rotations.append(np.rad2deg(np.array([ax, ay, az])))
    
    translations = np.array(translations)
    rotations = np.array(rotations)
    # Cluser based on translations
    translation_cluster = DBSCAN(eps=0.05, min_samples=get_cluster_min_samples(len(translations))).fit(translations)
    translation_major_labels = get_major_group_index(translation_cluster)
    
    if translation_major_labels is None:
        return None    
    
    print translation_major_labels
    
    print "total: ", len(matrices)
    print "correct translation: ", len(translation_major_labels)
    print "rate: ", len(translation_major_labels) * 1.0 / len(matrices)
    
    # Recluster major group translations based on rotation.
    rotations_regroup = rotations[translation_major_labels]
    rotation_cluster = DBSCAN(eps=10.0, min_samples=get_cluster_min_samples(len(rotations_regroup))).fit(rotations_regroup)
    rotation_major_labels = get_major_group_index(rotation_cluster)
    print "rotation_major_labels"
    print rotation_major_labels
    
    if rotation_major_labels is None:
        return None
    
    major_translation = translations[translation_major_labels][rotation_major_labels]
    major_rotation = rotations[translation_major_labels][rotation_major_labels]
    
    print "selected index: "
    print translation_major_labels[rotation_major_labels]
    print "total: ", len(matrices)
    print "selected: ", len(translation_major_labels[rotation_major_labels])
    print "correct_percentage: ", len(translation_major_labels[rotation_major_labels]) * 1.0 / len(matrices)
   
   # Return list of matrix in major group. 
    major_matrix = []
    for i in range(len(major_translation)):
        translation = major_translation[i]
        rotation = major_rotation[i]
        matrix = get_transformation_matrix_from_euler_angle(translation[0], translation[1], translation[2], np.deg2rad(rotation[0]), np.deg2rad(rotation[1]), np.deg2rad(rotation[2]))
        major_matrix.append(matrix)
    
    return major_matrix

# this is really just for a new_screw_axis with that is different by 2n * pi
def get_delta_with_different_screw_axis(old_screw_axis, old_theta, new_screw_axis):
    T_old = get_transformation_matrix_from_exponential(old_screw_axis, old_theta)
    T_new = get_transformation_matrix_from_exponential(new_screw_axis, old_theta)
    
    old_translation, old_rotation = decompose_homogeneous_transformation_matrix(T_old)
    new_translation, new_rotation = decompose_homogeneous_transformation_matrix(T_new)
    
    translation_difference = old_translation - new_translation
    T_new_unit = get_transformation_matrix_from_exponential(new_screw_axis, 2 * np.pi)
    new_unit_translation, new_unit_quaternion = decompose_homogeneous_transformation_matrix(T_new_unit)
    n = 0
    if not is_zero_vector(new_unit_translation, error=1e-4):
        n = translation_difference / new_unit_translation
    
    new_theta = old_theta + 2 * n * np.pi
    T_new = get_transformation_matrix_from_exponential(new_screw_axis, new_theta)
    
    if not same_pose(T_old, T_new):
        print "[transformation_util][get_delta_with_different_screw_axis]: not the same pose"
    
    return new_theta

def _get_delta_with_chosen_screw_axis_from_transformation(direction, interval, transformation, S):
    print "S: ", S
    
    distance = get_vector_length(transformation[0:3, 3])
    final_p = transformation[0:3, 3]
    T = np.identity(4)
    num_interation = 10000
    i = 0
    delta = 0.
    
    while distance > 0.005 and i < num_interation:
        T = get_transformation_matrix_from_exponential(S, delta)
        current_p = T[0:3, 3]
        distance = get_vector_length(final_p - current_p)
        delta += interval * direction
        i += 1
    
    return delta, distance < 0.005

def get_delta_with_chosen_screw_axis_from_transformation(transformation, new_screw_axis):
    direction = -1.
    interval = np.deg2rad(1.)
    
    delta, get_result = _get_delta_with_chosen_screw_axis_from_transformation(direction, interval, transformation, new_screw_axis)
    
    if not get_result:
        direction = -1.
        delta, get_result = _get_delta_with_chosen_screw_axis_from_transformation(direction, interval, transformation, new_screw_axis)

    return delta

def get_fixed_frame_transformation(T_start, T_end):
    return np.matmul(T_end, get_transformation_matrix_inverse(T_start))

def get_fixed_frame_exponential_transformation(T_start, T_end):
    T = get_fixed_frame_transformation(T_start, T_end)
    return get_exponential_from_transformation_matrix(T)

def get_body_frame_transformation(T_start, T_end):
    return np.matmul(get_transformation_matrix_inverse(T_start), T_end)

def get_body_frame_exponential_transformation(T_start, T_end):
    T = get_body_frame_transformation(T_start, T_end)
    return get_exponential_from_transformation_matrix(T)

# change the frame of S from {a} to {b}, Sa -> Sb
def change_twist_frame(Sa, Tworld_b, Tworld_a):
    Tab = np.matmul(get_transformation_matrix_inverse(Tworld_a), Tworld_b)
    AD_Tab = get_adjoint_representation_matrix(Tab)
    
    Sb = np.matmul(AD_Tab, np.array([Sa]).T)
    Sb = Sb.T[0]
    
    return Sb

def category_trajectory(trajectory, merge_opposite_direction=True): # trajectory is a list of transformations
    categorized_screw_axis = []
    categorized_theta = []    
    
    screw_axes = []
    thetas = []    
    
    if len(trajectory) == 0 or len(trajectory) == 1:
        return categorized_screw_axis, categorized_theta
    
    screw_axes = []
    thetas = []
    
    previous_transformation = trajectory[0]
    for i in range(1, len(trajectory)):
        current_transformation = trajectory[i]
        screw_axis, theta = get_body_frame_exponential_transformation(previous_transformation, current_transformation)
        screw_axes.append(screw_axis)
        thetas.append(theta)
        previous_transformation = current_transformation
    
    print "[transformation_util][category_trajectory] screw_axes: "
    print screw_axes
    print "[transformation_util][category_trajectory] thetas: "
    print thetas
    
    if len(screw_axes) == 1:
        return screw_axes, thetas
    
    current_screw_axes_group = [screw_axes[0]]
    current_thetas = [thetas[0]]
    current_averaged_screw_axis = screw_axes[0]
    
    for i in range(1, len(screw_axes)):
        current_screw_axis = screw_axes[i]
        current_theta = thetas[i]
        
        is_same_shape, is_same_screw_axis = same_screw_axis_with_same_direction(current_averaged_screw_axis, current_screw_axis, sum(current_thetas), current_theta, angle_error = 5.0, v_error = 20.0)
        
        is_added = False
        if is_same_shape:
            to_add = True
            if not merge_opposite_direction:
                if is_same_screw_axis:
                    if current_theta * sum(current_thetas) < 0:
                        to_add = False
                else:
                    if current_theta * sum(current_thetas) >= 0:
                        to_add = False
            if to_add:
                if not is_same_screw_axis:
                    current_screw_axis *= -1.0
                    current_theta *= -1.0 # TODO: check whether this is necessary
                current_screw_axes_group.append(current_screw_axis)
                current_thetas.append(current_theta)
                current_averaged_screw_axis = average_screw_axis(current_screw_axes_group)
                is_added = True
                
        if not is_added:
            categorized_screw_axis.append(current_averaged_screw_axis)
            categorized_theta.append(np.sum(current_thetas))
            
            current_screw_axes_group = [current_screw_axis]
            current_thetas = [current_theta]
            current_averaged_screw_axis = current_screw_axis
            
    categorized_screw_axis.append(current_averaged_screw_axis)
    categorized_theta.append(np.sum(current_thetas))

    print "[transformation_util][category_trajectory] categorized_screw_axis"
    for screw in categorized_screw_axis:
        print screw
    
    print "[transformation_util][category_trajectory] categorized_theta"
    print categorized_theta
    
    return categorized_screw_axis, categorized_theta

def merge_keyframes(trajectory):
    merged_trajectory = [trajectory[0]]
    
    screw_axis, theta = category_trajectory(trajectory)
    
    for i in range(len(screw_axis)):
        T_body = get_transformation_matrix_from_exponential(screw_axis[i], theta[i])
        merged_trajectory.append(np.matmul(merged_trajectory[-1], T_body))
    
    return merged_trajectory
    
def get_r_z(a):
    return  np.array([[np.cos(a), -np.sin(a), 0],
                      [np.sin(a), np.cos(a), 0],
                      [0, 0, 1]])

########## cluster related functions ##############

# input: [[float], [float], ...]
def cluster_array_GMM(array, n_components):
    cluster = GaussianMixture(n_components=n_components, tol=0.01).fit(array)
    num_cluster = len(list(cluster.means_))
    
    # return label for each data point, number of clusters, the mean and the diagonal covariances of each cluster
    return cluster.predict(np.array(array)), num_cluster, cluster.means_, cluster.covariances_

# some random stuffs
# http://cs229.stanford.edu/section/gaussians.pdf
# http://cs229.stanford.edu/notes2020fall/notes2020fall/more_on_gaussians.pdf

# Use the Kolmogorov-Smirnov normality test
# https://stackoverflow.com/questions/26307202/kolmogorov-smirnov-test-in-scipy-with-non-normalized-data
# https://stackoverflow.com/questions/22392562/how-can-check-the-distribution-of-a-variable-in-python
# array: np.array([float, float, ...])
def is_scale_uniform_distribution(array):
    labels, num_cluster, means, covariances = cluster_array_GMM(array, n_components=1)
    array_for_kstest = [i[0] for i in array]
    is_uniform = False
    
    if num_cluster == 1:
        print "num_cluster == 1"
        min_value = np.min(array_for_kstest)
        max_value = np.max(array_for_kstest)
        print "min_value: ", min_value
        print "max_value: ", max_value
        print "array: "
        print array_for_kstest
        print "kstest uniform result: ", kstest(array_for_kstest, 'uniform', args=(min_value, max_value)).pvalue
        is_uniform = kstest(array_for_kstest, 'uniform', args=(min_value, max_value)).pvalue > 0.9 # TODO: tune this value, null hypothesis means the 2 distribution are the same

    return is_uniform

# 0 - 360
# to
# -180 - 180
def convert_angle_range(angles):
    converted_angle = []
    for angle in angles:
        if angle > np.pi:
            converted_angle.append(angle - np.pi * 2)
        else:
            converted_angle.append(angle)
    return converted_angle

# -180 - 180
# to
# 0 - 360
def convert_angle_range_backward(angles):
    converted_angle = []
    for angle in angles:
        if angle < 0:
            converted_angle.append(angle + np.pi * 2)
        else:
            converted_angle.append(angle)
    return converted_angle

# angles: 0 - 360
# at the boundary, the value of the range will be in -180 - 180
# the mean is in 0 - 360
def get_angle_range_mean(angles, eps):
    is_across_zero = False
    clustered_angle, _ = cluster_array_DBSCAN([[i] for i in angles], eps=eps, min_samples=1)
    if len(clustered_angle.keys()) > 1:
        is_across_zero = True
    # get the mean, range
    mean_angle = 0.
    range_angle = [0., 0.]
    if is_across_zero: # convert to [-180, 180]
        converted_angles = convert_angle_range(angles)
        mean_angle = sum(converted_angles) / len(converted_angles)
        range_angle = [min(converted_angles), max(converted_angles)]
        if mean_angle < 0:
            mean_angle += np.pi * 2
    else:
        mean_angle = sum(angles) / len(angles)
        range_angle = [min(angles), max(angles)]
    
    return mean_angle, range_angle   

class AngleGroup(object):
    def __init__(self, angles):
        if len(angles) == 0:
            return
        
        self.angles = angles
        
        self.clustered_angle, self.clustered_angle_range = self.cluster_angle_DBSCAN()
        
        self.is_uniform_distribution = False
        if len(self.clustered_angle.keys()) == 1:
            print "check whether it is uniform distribution!!!!!"
            min_range, max_range = self.clustered_angle_range.values()[0]
            clustered_angles = self.clustered_angle.values()[0]
            if min_range < 0.:
                print "in min_range < 0."
                converted_angle = convert_angle_range(clustered_angles)
                self.is_uniform_distribution = is_scale_uniform_distribution([[i] for i in converted_angle])
            else:
                print "in min_range >= 0."
                self.is_uniform_distribution = is_scale_uniform_distribution([[i] for i in clustered_angles])
    
    def is_uniform(self):
        return self.is_uniform_distribution
    
    def cluster(self): # {mean:[min, max]}
        return self.clustered_angle_range
    
    def group_range(self):
        if self.is_uniform():
            return self.clustered_angle_range.values()[0]
        else:
            return None
    
    def group_mean(self):
        return self.clustered_angle_range.keys()
    
    # https://datascience.stackexchange.com/questions/5990/what-is-a-good-way-to-transform-cyclic-ordinal-attributes/6335#6335
    def cluster_angle_DBSCAN(self):
        # convert angle to circle (x, y) information
        converted_angle = [[np.cos(angle), np.sin(angle)] for angle in self.angles]
        
        threshold_angle = np.pi * 2. / MAX_NUM_ANGLE_GROUP  # consider at most 15 groups
        eps = get_vector_length(np.array([np.cos(threshold_angle), np.sin(threshold_angle)]))
        clustered_converted_angles, clustered_indices = cluster_array_DBSCAN(np.array(converted_angle), eps=eps, min_samples=1)
        
        clusters = {}
        clusters_range = {}
        for label in clustered_indices.keys():
            angles = np.array(self.angles)[clustered_indices[label]]
            # check whether the current group go across 0
            # cluster the angles with the current label
            mean_angle, range_angle = get_angle_range_mean(angles, threshold_angle)
            clusters[mean_angle] = list(angles)
            clusters_range[mean_angle] = range_angle
        
        return clusters, clusters_range
    
    def to_json(self):
        json_result = {}
        json_result["angles"] = self.angles
        json_result["clustered_angle"] = self.clustered_angle
        json_result["clustered_angle_range"] = self.clustered_angle_range
        json_result["is_uniform_distribution"] = self.is_uniform_distribution
        
        return json_result
    
    @staticmethod
    def from_json(json_result):
        angle_group = AngleGroup([])
        
        angle_group.angles = json_result["angles"]
        angle_group.is_uniform_distribution = json_result["is_uniform_distribution"]
        
        angle_group.clustered_angle = {}
        for key in json_result["clustered_angle"].keys():
            angle_group.clustered_angle[float(key)] = json_result["clustered_angle"][key]
            
        angle_group.clustered_angle_range = {}
        for key in json_result["clustered_angle_range"].keys():
            angle_group.clustered_angle_range[float(key)] = json_result["clustered_angle_range"][key]

        return angle_group

class ArrayGroup(object):
    def __init__(self, arrays, axis):
        self.arrays = arrays
        self.axis = axis
        
        threshold_angle = np.pi * 2 / len(arrays)
        eps = get_vector_length(np.array([np.cos(threshold_angle), np.sin(threshold_angle)]))
        clustered_arrays, clustered_indices = cluster_array_DBSCAN(np.array(list(arrays)), eps=eps, min_samples=1)        

        self.clustered = datastructure_util.ArrayDict()
        for label, cluster in clustered_arrays.items():
            mean_cluster = np.mean(cluster, axis=0)
            self.clustered[mean_cluster] = cluster

        self.is_uniform_distribution = False
        if len(self.clustered_arrays) == 1:
            angles = self.get_angles(self.clustered_arrays.items()[0])
            mean_angle, range_angle = get_angle_range_mean(angles)
            min_angle = range_angle[0]
            if min_range < 0.:
                converted_angle = convert_angle_range(angles)
                self.is_uniform_distribution = is_scale_uniform_distribution(np.array(converted_angle))
            else:
                self.is_uniform_distribution = is_scale_uniform_distribution(np.array(angles))
    
    def is_uniform(self):
        return self.is_uniform_distribution
    
    def get_angles(self, arrays):
        angles = []
        origin = arrays[0]
        for array in arrays:
            if origin == array:
                angles.append(0.)
            else:
                axis = np.cross(origin, array)
                angle_sign = 1.
                if get_angle([axis], [self.axis]) > np.pi / 2.: # counter-clockwise
                    angle_sign = -1.
                angle = get_angle([origin], [array]) * angle_sign # range[-180, 180]
                if angle_sign == -1.: # convert to range[0, 360]
                    angle += np.pi * 2
                angles.append(angle)
        return angles
    
    # https://datascience.stackexchange.com/questions/5990/what-is-a-good-way-to-transform-cyclic-ordinal-attributes/6335#6335
    def cluster_angle_DBSCAN(self):
        # convert angle to circle (x, y) information
        converted_angle = [[np.cos(angle), np.sin(angle)] for angle in self.angles]
        
        threshold_angle = np.pi * 2 / len(angles)
        eps = get_vector_length(np.array([np.cos(threshold_angle), np.sin(threshold_angle)]))
        clustered_converted_angles, clustered_indices = cluster_array_DBSCAN(np.array(converted_angle), eps=eps, min_samples=1)
        
        clusters = {}
        clusters_range = {}
        for label in clustered_indices.keys():
            angles = np.array(self.angles)[clustered_indices[label]]
            # check whether the current group go across 0
            # cluster the angles with the current label
            is_across_zero = False
            clustered_angle, _ = cluster_array_DBSCAN(np.array(angles), eps=eps, min_samples=1)
            if len(clustered_angle.keys()) > 1:
                is_across_zero = True
            # get the mean, range
            mean_angle = 0.
            range_angle = [0., 0.]
            if is_across_zero: # convert to [-180, 180]
                converted_angles = convert_angle_range(angles)
                mean_angle = sum(converted_angles) / len(converted_angles)
                range_angle = [min(converted_angles), max(converted_angles)]
                if mean_angle < 0:
                    mean_angle += np.pi * 2
            else:
                mean_angle = sum(angles) / len(angles)
                range_angle = [min(converted_angles), max(converted_angles)]
            # add the result:
            clusters[mean_angle] = angles
            clusters_range[mean_angle] = range_angle
        
        return clusters, clusters_range

def cluster_array_with_same_axis(arrays, axis):
    # project all arrays to the plane that is perpendicular to axis
    projected_arrays = []
    for array in arrays:
        axis_projection = np.dot(array, axis) / get_vector_length(axis)
        projected_array = normalize(array - axis_projection)
        projected_arrays.append(projected_array)

    return ArrayGroup(projected_arrays, axis)

# arrays: [np.array([]), np.array([]), ...]
# return: keys are the axis, and values are the members
# return: {axis: ArrayGroup}
def cluster_array_based_on_axis(arrays):
    array_groups = merge_linear_arrays(arrays)

    axis_group = datastructure_util.ArrayDict()
    axis = np.array([0., 0., 0.])
    unclustered_array = [np.array(i) for i in array_groups.keys()]
    while axis is not None:
        axis, clustered_array, unclustered_array = major_cluster_nonlinear_array_groups(unclustered_array)
        if axis is not None:
            axis_group[set(axis)] = clustered_array
        else:
            if len(unclustered_array) != 0:
                axis_group["other"] = unclustered_array
    
    # add all linear vectors
    clusters = datastructure_util.ArrayDict(unique=False)
    for axis in axis_group.keys():
        clusters[axis] = datastructure_util.ArrayList(unique=False)
        for linear_group in axis_group[axis]:
            linear_arrays = array_groups[linear_group]
            for linear_array in linear_arrays:
                clusters[axis].append(linear_array)
    
    array_features = {}
    for axis in clusters.keys():
        array_features[set(axis)] = cluster_array_with_same_axis(list(clusters[axis]), axis)
    
    return array_features
    
def merge_linear_arrays(arrays):
    array_pairs = combinations(arrays, 2)
    
    # grouping the ones with the same or opposite directions
    array_groups = datastructuril.ArrayDict(unique=False)
    ungrouped_arrays = datastructure_util.ArrayList(copy.deepcopy(arrays), unique=False)
    for pair in array_pairs:
        if is_colinear(pair[0], pair[1], error=np.deg2rad(1.)): # TODO: tune this
            found_group = False
            for array_representative in array_groups.keys():
                if is_colinear(array_representative, pair[0], error=np.deg2rad(1.)): # TODO: tune this
                    found_group = True
                    if pair[0] in ungrouped_arrays:
                        array_groups[array_representative].append(pair[0])
                        ungrouped_arrays.remove(pair[0])
                    if pair[1] in ungrouped_arrays:
                        array_groups[array_representative].append(pair[1])
                        ungrouped_arrays.remove(pair[1])
            if not found_group:
                array_groups[pair[0]] = [pair[0], pair[1]]
                ungrouped_arrays.remove(pair[0])
                ungrouped_arrays.remove(pair[1])
    
    for array in ungrouped_arrays:
        array_groups[array] = [array]
    
    return array_groups

def div(x, y):
    if y == 0:
        return 1
    else:
        return x / y

def sign(x):
    return div(x, abs(x))

def major_cluster_nonlinear_array_groups(arrays):
    if len(arrays) == 0 or len(arrays) == 1:
        axis = None
        clustered_array = datastructure_util.ArrayList()
        unclustered_array = arrays
        return axis, clustered_array, unclustered_array
        
    # update the arrays to contain non colinear arrays
    updated_arrays = list(arrays)
    updated_pairs = combinations(updated_arrays, 2)
    
    # axis
    axes = datastructuril.ArrayDict(unique=True)
    for pair in updated_pairs:
        axis = np.cross(pair[0], pair[1])
        if sign(axis[0]) * sign(axis[1]) * sign(axis[2]) < 0.:
            axis *= -1.
        axis = normalize(axis)
        if axis not in axes.keys():
            axes[axis] = [pair[0], pair[1]]
        else:
            axes[axis].append(pair[0])
            axes[axis].append(pair[1])
    
    # cluster axis
    labels, num_cluster, means, covariances = cluster_array_GMM(axes.keys(), 3)
    unique_labels = list(np.unique(labels))
    label_count_dict = {}
    for label in unique_labels:
        if label == -1:
            continue
        label_index = np.where(labels == label)
        current_array = np.array(axes.keys())[label_index]
        count = 0
        for array in current_array:
            count += len(axes[array])
        label_count_dict[label] = count
    
    # find the major group
    major_count = 0
    major_label = -1
    for key, value in label_count_dict.items():
        if value > major_count:
            major_count = value
            major_label = key
    
    # get the major group
    label_index = np.where(labels == major_label)
    current_axes_array = np.array(axes.keys())[label_index]
    clustered_group = datastructure_util.ArrayList()
    unclustered_group = datastructure_util.ArrayList(content=list(arrays))
    for axis in current_axes_array:
        arrays = axes[axis]
        for array in arrays:
            clustered_group.append(array)
            unclustered_group.remove(array)
    
    axis = normalize(np.mean(current_axes_array, axis=0))
    
    return axis, clustered_group, unclustered_group
    
def cluster_transformations(transformations):
    pass

# array: [np.array, np.array, ...]
def cluster_array_DBSCAN(array, eps=0.05, min_samples=2):
    cluster = DBSCAN(eps=eps, min_samples=min_samples).fit(array)
    labels = cluster.labels_
    unique_labels = list(np.unique(cluster.labels_))
    
    results = {}
    result_indices = {}
    for label in list(unique_labels):
        label_index = np.where(labels == label)
        results[label] = [np.array(i) for i in list(np.array(array)[label_index])]
        result_indices[label] = label_index
    
    return results, result_indices

def get_major_group_index(cluster):
    label = cluster.labels_
    noise_labels = np.where(label == -1)[0]
    label_without_noises = np.delete(label, noise_labels)
    
    if np.bincount(label_without_noises).size == 0:
        return None
    
    major_label = np.bincount(label_without_noises).argmax()
    
    return np.where(label == major_label)[0]

def is_opposite_sign(a, b):
    if a == 0 and b == 0:
        return True
    
    if a * b == 0:
        return False
    
    if sign(a) * sign(b) == 1:
        return True
    
    return False

"""
trajectory generation
"""
def get_trajectory_interpolation_with_transformations(Tstart, Tend): 
    T = get_fixed_frame_transformation(Tstart, Tend)
    S, theta = get_exponential_from_transformation_matrix(T)
    
    return get_trajectory_interpolation_with_fixed_frame_screw_axis(S, theta, Tstart)

def get_trajectory_interpolation_with_fixed_frame_screw_axis(S, theta, Tstart):
    interval = theta
    num_interval = 1
    direction = theta / abs(theta)
    if is_screw_axis_translational(S):
        interval = 0.001
        num_interval = abs(int(theta / interval))
    else:
        interval = np.deg2rad(1.)
        num_interval = abs(int(theta / interval))

    interval *= direction
    Ss = [get_transformation_matrix_from_exponential(S, interval * i) for i in range(num_interval)]

    trajectory = [np.matmul(get_transformation_matrix_from_exponential(S, interval * i), Tstart) for i in range(num_interval)]
    trajectory.append(np.matmul(get_transformation_matrix_from_exponential(S, theta), Tstart))
 
    return trajectory

def get_trajectory_interpolation_with_body_frame_screw_axis(S, theta, Tstart):
    interval = theta
    num_interval = 1
    direction = theta / abs(theta)
    if is_screw_axis_translational(S):
        interval = 0.001
        num_interval = abs(int(theta / interval))
    else:
        interval = np.deg2rad(1.)
        num_interval = abs(int(theta / interval))
    
    interval *= direction
    
    print "[transformation_util][get_trajectory_interpolation_with_body_frame_screw_axis] interval: ", interval
    print "[transformation_util][get_trajectory_interpolation_with_body_frame_screw_axis] num_interval: ", num_interval

    trajectory = [np.matmul(Tstart, get_transformation_matrix_from_exponential(S, interval * i)) for i in range(num_interval)]
    trajectory.append(np.matmul(Tstart, get_transformation_matrix_from_exponential(S, theta)))
    
    print "[transformation_util][get_trajectory_interpolation_with_body_frame_screw_axis] len(trajectory) = ", len(trajectory)

    return trajectory

def rescale_S(S, rx, ry, rz):
    w = np.array(S[:3])
    v = np.array(S[3:])
    
    v[0] *= rx * 1.0
    v[1] *= ry * 1.0
    v[2] *= rz * 1.0
    
    rescaled_S = np.array([w[0], w[1], w[2], v[0], v[1], v[2]])
    
    return rescaled_S

def rescale_trajectory(scale, screw_trajectory):
    updated_trajectory = []
    
    for (S, theta) in screw_trajectory:
        new_S, new_theta = S, theta
        if is_screw_axis_translational(S):
            new_theta *= scale
        else:
            new_S = rescale_S(S, scale, scale, scale)
        updated_trajectory.append((new_S, new_theta))
    
    return updated_trajectory

def distort_trajectory(distort, screw_trajectory):
    updated_trajectory = []
    
    for (S, theta) in screw_trajectory:
        new_S, new_theta = S, theta
        if not is_screw_axis_translational(S):
            new_S = rescale_S(S, distort, 1., distort)
            #new_theta /= distort
        updated_trajectory.append((new_S, new_theta))
    
    return updated_trajectory    

# screw_trajectory: [(axis, theta), (axis, theta), ...]
def reform_trajectory(scale, distort, screw_trajectory):
    screw_trajectory = rescale_trajectory(scale, screw_trajectory)
    screw_trajectory = distort_trajectory(distort, screw_trajectory)
    
    return screw_trajectory

# T1: original Tstart, transformatoin matrix
# trajectory: [(S, theta), (S, theta), ...], S is in the previous frame
# new start translation: np.array([x, y, z])
# new final translation: np.array([x, y, z])
def goal_trajectory(T1, trajectory, start_position, final_position):
    Ts = [get_transformation_matrix_from_exponential(i[0], i[1]) for i in trajectory]
    
    final_pose = T1
    for T in Ts:
        final_pose = np.matmul(final_pose, T)
        
    p, R = decompose_homogeneous_transformation_matrix_to_rotation_matrix(final_pose)
    start_p, R = decompose_homogeneous_transformation_matrix_to_rotation_matrix(T1)
    scale = get_vector_length(final_position - start_position) / get_vector_length(p - start_p)
    
    rotation_matrix = get_rotation_matrix_from_directions([p - start_p], [final_position - start_p], axis = np.array([[0., 0., 1.]]))
    
    T = get_transformation_matrix_with_rotation_matrix(rotation_matrix, np.array([0., 0., 0.]))
    T_start = np.matmul(T, T1)
    T_start[0, 3] = start_position[0]
    T_start[1, 3] = start_position[1]
    T_start[2, 3] = start_position[2]

    new_trajectory = []
    for S, theta in trajectory:
        scaled_S = rescale_S(S, scale, scale, scale)
        new_trajectory.append((scaled_S, theta))
    
    return new_trajectory, T_start

def is_screw_axis_translational(screw_axis):
    S = screw_axis[:3]
    return is_zero_vector(S, error=0.001)

def transform_axis(transformation, axis):
    reshaped_axis = np.array([[axis[0], axis[1], axis[2], 1.]]).T
    transformed_axis = np.matmul(transformation, reshaped_axis).T[0]
    
    return np.array([transformed_axis[0], transformed_axis[1], transformed_axis[2]])

# remove the points that are exactly the same
def clear_trajectory(trajectory, tool, Tee_tool=np.identity(4)):
    trajectory = copy.deepcopy(trajectory)
    cleared_trajectory = []
    lowest_ee = np.inf
    lowest_Tworld_ee = np.identity(4)
    
    for index in range(len(trajectory) - 1):
        current_T = trajectory[index]
        next_T = trajectory[index + 1]

        current_z = current_T[2, 3]
        if current_z < lowest_ee:
            lowest_ee = current_z
            lowest_Tworld_ee = copy.deepcopy(current_T)
        
        if not np.array_equal(np.around(current_T, decimals=7), np.around(next_T, decimals=7)):
            cleared_trajectory.append(current_T)

    cleared_trajectory.append(trajectory[-1])
    
    lowest_Tworld_tool = np.matmul(lowest_Tworld_ee, Tee_tool)
    tool_pc_path = constants.get_tool_mesh_path(tool)
    tool_pc = o3d.io.read_point_cloud(tool_pc_path)
    tool_pc.transform(lowest_Tworld_tool)
    tool_pnts = np.asarray(tool_pc.points)
    tool_pnts_z = tool_pnts[:, 2]
    lowest_tool_pnts = np.min(tool_pnts[:, 2])
    
    print "[transformation_util][clear_trajectory] lowest_tool_pnts: ", lowest_tool_pnts
    
    lowest_pnt_intended = 0.
    if lowest_tool_pnts - lowest_pnt_intended < 0.: # below the point
        padding = abs(lowest_pnt_intended - lowest_tool_pnts)
        for T in cleared_trajectory:
            T[2, 3] += padding
    
    return cleared_trajectory

def clear_ee_trajectory(trajectory):
    trajectory = copy.deepcopy(trajectory)
    cleared_trajectory = []
    
    for index in range(len(trajectory) - 1):
        current_T = trajectory[index]
        next_T = trajectory[index + 1]
        
        if not np.array_equal(np.around(current_T, decimals=7), np.around(next_T, decimals=7)):
            cleared_trajectory.append(current_T)

    cleared_trajectory.append(trajectory[-1])
    
    return cleared_trajectory
