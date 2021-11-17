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

def close_to(m, n, error=1e-6):
    return m >= n - error and m <= n + error

def is_unit_vector(v, error=1e-6):
    """
    given a vector, return whether it is a unit vector

    Parameters:
    v: vector

    returns:
    boolean
    """
    return close_to(np.linalg.norm(v), 1, error)

def is_zero_vector(v, error=1e-6):
    """
    given a vector, return whether it is a zero vector

    Parameters:
    v: vector

    returns:
    boolean
    """
    return close_to(np.linalg.norm(v), 0, error)

def is_identity_matrix(R, error=1e-6):
    assert(R.shape[0] == R.shape[1])

    return R[0][0] >= 1 - error and R[1][1] >= 1 - error and R[2][2] >= 1 - error

def get_distance(a, b):
    return np.linalg.norm(a - b)

def get_skew_symmetric_matrix(w):
    assert(w.shape[0] == 1)
    return np.cross(w, -np.identity(w.shape[1]))

def get_vector_from_skew_symmetric_matrix(R):
    """
    currently only works for 3*3 matrix
    """
    return np.array([[R[2, 1], R[0, 2], R[1, 0]]])

def get_homogenous_transformation_matrix(R, p):
    assert(R.shape[0] == R.shape[1])
    assert(R.shape[0] == p.shape[1])
    return np.c_[np.r_[R, np.zeros((1, R.shape[0]))], np.r_[p.T, [[1]]]]

def get_homogenous_transformation_matrix_inverse(T):
    R, p = get_R_p_from_matrix(T)
    return get_homogenous_transformation_matrix(R.T, -np.matmul(R.T, p.T).T)

def get_R_p_from_matrix(T):
    return T[0:-1, 0:-1], np.array([T[0:-1, -1]])

def get_rotation_matrix_from_exponential_representation(w, theta):
    return np.identity(w.shape[1]) + np.sin(theta) * get_skew_symmetric_matrix(w) + (1 - np.cos(theta)) * np.matmul(get_skew_symmetric_matrix(w), get_skew_symmetric_matrix(w))

def get_exponential_representation_from_rotation_matrix(R, default=None):
    """
    current only work with 3 * 3 matrix
    """
    theta = 0
    w = np.zeros((1, R.shape[0]))

    if is_identity_matrix(R):
        pass
    elif close_to(np.trace(R), -1, error=1e-3):
        theta = np.pi
        option = random.randint(0, 2)
        sign = (-1) ** random.randint(0, 1)
        if default is None:
            w = sign * 1 / np.sqrt(2 * (1 + R[option, option])) * np.array([[R[0, option], R[1, option], 1 + R[2, option]]])
        else:
            w = np.array([default])
    else:
        theta = np.arccos((np.trace(R) - 1) / 2)
        w = get_vector_from_skew_symmetric_matrix(0.5 / np.sin(theta) * (R - R.T))

    return w, theta

def get_twist(w, v):
    return np.c_[w, v]

def get_w_v_from_twist(V):
    return np.array(V[:, :V.shape[1]/2]), np.array(V[:, V.shape[1]/2:])

def _get_twist_matrix(w, v):
    return np.r_[np.c_[get_skew_symmetric_matrix(w), v.T], np.zeros((1, w.shape[1] + 1))]

def get_twist_matrix(V):
    w, v = get_w_v_from_twist(V)
    return _get_twist_matrix(w, v)

def get_radian_from_degree(theta):
    return theta / 180.0 * np.pi

def _get_adjoint_representation_matrix(R, p):
    return np.r_[np.c_[R, np.zeros(R.shape)], np.c_[np.matmul(get_skew_symmetric_matrix(p), R), R]]

def get_adjoint_representation_matrix(T):
    return _get_adjoint_representation_matrix(np.array(T[:-1, :-1]), np.array([T[:-1, -1]]))

def get_v(w, r):
    return np.cross(w, -r)

def _get_screw_axis(w, v):
    if is_zero_vector(w):
        return get_twist(w, v) / np.linalg.norm(v)
    else:
        return get_twist(w, v) / np.linalg.norm(w)

def get_screw_axis(V):
    w, v = get_w_v_from_twist(V)
    return _get_screw_axis(w, v)

def get_homogenous_transformation_matrix_from_exponential_representation(S, theta):
    S = get_screw_axis(S)
    
    w = np.array(S[:, :S.shape[1]/2])
    v = np.array(S[:, S.shape[1]/2:])

    R = np.identity(w.shape[1])
    p = np.zeros(w.shape)

    if is_unit_vector(w, error=1e-4):
        R = get_rotation_matrix_from_exponential_representation(w, theta)
        p = np.matmul(np.identity(w.shape[1]) * theta + (1 - np.cos(theta)) * get_skew_symmetric_matrix(w) + (theta - np.sin(theta)) * np.matmul(get_skew_symmetric_matrix(w), get_skew_symmetric_matrix(w)), v.T).T
    elif is_zero_vector(w, error=1e-4) and is_unit_vector(v, error=1e-4):
        p = v * theta

    return get_homogenous_transformation_matrix(R, p)

def get_exponential_representation_from_homogenous_transformation_matrix(T, threshold=1e-6, default=None):
    R, p = get_R_p_from_matrix(T)

    w = np.zeros((1, R.shape[0]))
    v = np.zeros((1, R.shape[0]))
    theta = 0.

    if is_identity_matrix(R, threshold):
        if is_zero_vector(p):
            v = np.zeros((1, R.shape[0]))
        else:
            v = p / np.linalg.norm(p)
        theta = np.linalg.norm(p)
    else:
        w, theta = get_exponential_representation_from_rotation_matrix(R, default=default)
        v = np.matmul(np.identity(R.shape[0]) / theta - get_skew_symmetric_matrix(w) / 2 + (1 / theta - 0.5 / np.tan(theta / 2)) * np.matmul(get_skew_symmetric_matrix(w), get_skew_symmetric_matrix(w)), p.T).T
    
    return get_screw_axis(get_twist(w, v)), theta

def forward_kinematics_s_frame(S_list, theta_list, M):
    result_matrix = np.identity(M.shape[0])
    for i in range(S_list.shape[0]):
        result_matrix = np.matmul(result_matrix, get_homogenous_transformation_matrix_from_exponential_representation(np.array([S_list[i, :]]), theta_list[:, i]))
    result_matrix = np.matmul(result_matrix, M)
    return result_matrix

def forward_kinematics_s_frame_all_joints(S_list, theta_list, M_list):
    all_joints_result_matrix = []
    result_matrix = np.identity(M_list[-1].shape[0])
    for i in range(S_list.shape[0]):
        result_matrix = np.matmul(result_matrix, get_homogenous_transformation_matrix_from_exponential_representation(np.array([S_list[i, :]]), theta_list[:, i]))
        all_joints_result_matrix.append(np.matmul(result_matrix, M_list[i]))
    return all_joints_result_matrix

def forward_kinematics_s_frame_position(S_list, theta_list, M):
    result_matrix = forward_kinematics_s_frame(S_list, theta_list, M)
    return np.array([result_matrix[:-1, -1]])

def forward_kinematics_b_frame(B_list, theta_list, M):
    result_matrix = M
    for i in range(B_list.shape[0]):
        result_matrix = np.matmul(result_matrix, get_homogenous_transformation_matrix_from_exponential_representation(np.array([B_list[i, :]]), theta_list[:, i]))
    return result_matrix

def space_jacobian(S_list, theta_list):
    S_list_comp = np.r_[[[1, 0, 0, 0, 0, 0]], S_list]
    theta_list_comp = np.c_[[[0]], theta_list]

    result_matrix = np.zeros(S_list.T.shape)
    T = np.identity(S_list.shape[1] / 2 + 1)

    for i in range(1, S_list_comp.shape[0]):
        T = np.matmul(T, get_homogenous_transformation_matrix_from_exponential_representation(np.array([S_list_comp[i - 1, :]]), theta_list_comp[:, i - 1]))
        result_matrix[:, i - 1] = np.matmul(get_adjoint_representation_matrix(T), np.array([S_list_comp[i, :]]).T)[:, 0]

    return result_matrix

def body_jacobian(B_list, theta_list):
    B_list_comp = np.r_[B_list, [[1, 0, 0, 0, 0, 0]]]
    theta_list_comp = np.c_[theta_list, [[0]]]

    result_matrix = np.zeros(B_list.T.shape)
    T = np.identity(B_list.shape[1] / 2 + 1)

    for i in range(B_list.shape[0], 0, -1):
        T = np.matmul(T, get_homogenous_transformation_matrix_from_exponential_representation(-np.array([B_list_comp[i, :]]), theta_list_comp[:, i]))
        result_matrix[:, i - 1] = np.matmul(get_adjoint_representation_matrix(T), np.array([B_list_comp[i - 1, :]]).T)[:, 0]

    return result_matrix

def _matrix_ellipspod_singularity_analysis(A):
    eigenvalues = np.linalg.eigvals(A).tolist()
    criteria_2 = max(eigenvalues) * 1.0 / min(eigenvalues)
    criteria_1 = np.sqrt(criteria_1)
    criteria_3 = np.linalg.det(A)
    return criteria_1, criteria_2, criteria_3

def manipulability_linear_analysis(body_jacobian):
    return _matrix_ellipspod_singularity_analysis(np.matmul(body_jacobian[S.shape[1]/2:, :], body_jacobian[S.shape[1]/2:, :].T))

def manipulability_angular_analysis(body_jacobian):
    return _matrix_ellipspod_singularity_analysis(np.matmul(body_jacobian[0:S.shape[1]/2, :], body_jacobian[0:S.shape[1]/2, :].T))

def ik_newton_raphson_b_frame(B_list, theta_list, M, target, num_iteration=20):
    to_stop = False
    i = 0

    while not to_stop and i < num_iteration:
        Tbs = get_homogenous_transformation_matrix_inverse(forward_kinematics_b_frame(B_list, theta_list, M))
        Sb, theta = get_exponential_representation_from_homogenous_transformation_matrix(np.matmul(Tbs, target))
        Vb = Sb * theta
        w, v = get_w_v_from_twist(Vb)
        if not is_zero_vector(w, 0.001) or not is_zero_vector(v, 0.001):
            theta_list += np.matmul(np.linalg.pinv(body_jacobian(B_list, theta_list)), Vb.T).T
        else:
            to_stop = True
        i += 1
    return theta_list, to_stop

def ik_newton_raphson_s_frame(S_list, theta_list, M, target, num_iteration=20):
    to_stop = False
    i = 0

    while not to_stop and i < num_iteration:
        Tsb = forward_kinematics_s_frame(S_list, theta_list, M)
        Tbs = get_homogenous_transformation_matrix_inverse(Tsb)
        Sb, theta = get_exponential_representation_from_homogenous_transformation_matrix(np.matmul(Tbs, target))
        Vb = Sb * theta
        Vs = np.matmul(get_adjoint_representation_matrix(Tsb), Vb.T).T
        w, v = get_w_v_from_twist(Vs)
        if not is_zero_vector(w, 0.001) or not is_zero_vector(v, 0.001):
            theta_list += np.matmul(np.linalg.pinv(space_jacobian(S_list, theta_list)), Vs.T).T
        else:
            to_stop = True
        i += 1
    return theta_list, to_stop

def ik_newton_raphson_s_frame_get_direction(S_list, theta_list, M, target):
    Tsb = forward_kinematics_s_frame(S_list, theta_list, M)
    Tbs = get_homogenous_transformation_matrix_inverse(Tsb)
    Sb, theta = get_exponential_representation_from_homogenous_transformation_matrix(np.matmul(Tbs, target))
    Vb = Sb * theta
    Vs = np.matmul(get_adjoint_representation_matrix(Tsb), Vb.T).T
    return np.matmul(np.linalg.pinv(space_jacobian(S_list, theta_list)), Vs.T).T

def ik_newton_raphton_s_frame_follow_trajectory(S_list, theta_list, M, target, step=0.001, num_iteration=20):
    results = []
    temp_angles = np.array(theta_list)

    current_position = forward_kinematics_s_frame_position(S_list, theta_list, M)
    target_position = np.array([target[:-1, -1]])
    direction = target_position - current_position
    distance = np.linalg.norm(direction)

    num_sub_targets = int(round(distance / step))

    subtargets = [current_position + direction / num_sub_targets * (i + 1) for i in range(num_sub_targets)]

    for i in range(num_sub_targets):
        subtarget = np.array(target)
        subtarget[:-1, -1] = subtargets[i]
        temp_angles, temp_is_success = ik_newton_raphson_s_frame(S_list, temp_angles, M, subtarget, num_iteration)
        results.append((np.array(temp_angles), temp_is_success))
    return results
