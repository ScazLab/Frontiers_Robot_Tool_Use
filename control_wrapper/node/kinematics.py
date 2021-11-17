#!/usr/bin/env python

# Software License Agreement (MIT License)
#
# Copyright (c) 2020, control_warpper
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

import sys
import copy
from math import pi

import numpy as np

from tf import transformations as tfs

import rospy
import rospkg
import geometry_msgs.msg
import sensor_msgs.msg
from std_msgs.msg import String, Header, Bool
from geometry_msgs.msg import Pose, Point, Quaternion
from shape_msgs.msg import SolidPrimitive

import moveit_commander
import moveit_msgs.msg
from moveit_msgs.srv import GetPlanningScene
from moveit_msgs.msg import PlanningScene, PlanningSceneComponents, AllowedCollisionMatrix, AllowedCollisionEntry, CollisionObject

from control_wrapper.msg import SceneObjects, SceneObject, SceneObjectAllowCollision
from control_wrapper.srv import SetPose, SetPoseResponse
from control_wrapper.srv import GetPose, GetPoseResponse
from control_wrapper.srv import CheckPose, CheckPoseResponse
from control_wrapper.srv import SetJoints, SetJointsResponse
from control_wrapper.srv import GetJoints, GetJointsResponse
from control_wrapper.srv import SetTrajectory, SetTrajectoryResponse
from control_wrapper.srv import AddObject, AddObjectResponse, AddObjectRequest
from control_wrapper.srv import AttachObject, AttachObjectResponse, AttachObjectRequest
from control_wrapper.srv import DetachObject, DetachObjectResponse
from control_wrapper.srv import RemoveObject, RemoveObjectResponse
from control_wrapper.srv import GetInterpolatePoints, GetInterpolatePointsResponse
from control_wrapper.srv import Reset, ResetResponse

# adapted from moveit online tutorial
def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    all_equal = True
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        goal_list = pose_to_list(goal)
        actual_list = pose_to_list(actual)
        
        print "goal list: ", pose_to_list(goal)
        print "actual list: ", pose_to_list(actual)
        
        return all_close(goal_list[:3], actual_list[:3], tolerance) and all_close(goal_list[3:], actual_list[3:], tolerance * 10)

    return True

def pose_to_list(pose_msg):
    pose_list = []
    pose_list.append(pose_msg.position.x)
    pose_list.append(pose_msg.position.y)
    pose_list.append(pose_msg.position.z)
    pose_list += tfs.euler_from_quaternion(np.array([pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w]))
    return pose_list

def decompose_homogeneous_transformation_matrix(matrix):
    translation = np.array([matrix[0, 3], matrix[1, 3], matrix[2, 3]])

    quaternion_matrix = matrix.copy()
    quaternion_matrix[:3, 3] = 0
    quaternion = tfs.quaternion_from_matrix(quaternion_matrix)

    rotation = np.array([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])

    return translation, rotation

def get_homogeneous_transformation_matrix_from_quaternion(rotation_quaternion, translation_vector):
    # translation_vector: np.array([1, 2, 3])
    alpha, beta, gamma = tfs.euler_from_quaternion(rotation_quaternion)
    rotation_matrix = tfs.euler_matrix(alpha, beta, gamma)

    result = rotation_matrix
    result[:3, 3] = translation_vector

    return result

def pose_matrix_to_msg(matrix):
    translation, rotation = decompose_homogeneous_transformation_matrix(matrix)

    position_msg = Point(translation[0], translation[1], translation[2])
    quaternion_msg = Quaternion(rotation[0], rotation[1], rotation[2], rotation[3])
    
    return Pose(position_msg, quaternion_msg)

def pose_msg_to_matrix(pose_msg):
    position = np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
    quaternion = np.array([pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w])
    return get_homogeneous_transformation_matrix_from_quaternion(quaternion, position)

# adapted from moveit online tutorial with major modifications
class Kinematics(object):
    def __init__(self, robot_name, side, group_name, joint_names, grapsing_group, base_frame, joint_state_topic="", goal_tolarance=None):
        super(Kinematics, self).__init__()

        if joint_state_topic:
            moveit_commander.roscpp_initialize(['joint_states:=' + joint_state_topic])
        else:
            moveit_commander.roscpp_initialize(sys.argv)

        robot = moveit_commander.RobotCommander()

        scene = moveit_commander.PlanningSceneInterface()
        
        found_move_group = False
        while not found_move_group:
            try:
                move_group = moveit_commander.MoveGroupCommander(group_name)
                found_move_group = True
            except RuntimeError:
                found_move_group = False

        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)

        # We can get the name of the reference frame for this robot:
        planning_frame = move_group.get_planning_frame()
        print "============ Planning frame: %s" % planning_frame

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        print "============ End effector link: %s" % eef_link

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print "============ Available Planning Groups:", robot.get_group_names()
        
        if not goal_tolarance is None:
            move_group.set_goal_joint_tolerance(goal_tolarance[0])
            move_group.set_goal_position_tolerance(goal_tolarance[1])
            move_group.set_goal_orientation_tolerance(goal_tolarance[2])
        print "============ Get Goal Tolarance: ", move_group.get_goal_tolerance()

        # Misc variables
        #self.box_name = ''
        self.robot = robot
        self.grasping_group = grapsing_group
        self.base_frame = base_frame
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
        
        self.joint_names = joint_names
        
        self.topic_name = '/' + robot_name + '/control_wrapper/' + side
        
        self.scene_obj_publisher = rospy.Publisher(self.topic_name + '/scene_objects', SceneObjects, queue_size=20)
        self.planning_scene_publisher = rospy.Publisher('planning_scene', PlanningScene, queue_size=10)
        
        rospy.Subscriber(self.topic_name + '/scene_allow_collision', SceneObjectAllowCollision, self.scene_object_allow_collision)

        rospy.Service(self.topic_name + '/set_pose', SetPose, self.set_pose)
        rospy.Service(self.topic_name + '/check_pose', CheckPose, self.check_pose)
        rospy.Service(self.topic_name + '/get_pose', GetPose, self.get_pose)
        rospy.Service(self.topic_name + '/set_joints', SetJoints, self.set_joints)
        rospy.Service(self.topic_name + '/get_joints', GetJoints, self.get_joints)
        rospy.Service(self.topic_name + '/follow_trajectory', SetTrajectory, self.set_trajectory)
        rospy.Service(self.topic_name + '/add_object', AddObject, self.add_object)
        rospy.Service(self.topic_name + '/attach_object', AttachObject, self.attach_object)
        rospy.Service(self.topic_name + '/detach_object', DetachObject, self.detach_object)
        rospy.Service(self.topic_name + '/remove_object', RemoveObject, self.remove_object)
        rospy.Service(self.topic_name + '/get_interpolate_points', GetInterpolatePoints, self.get_interpolate_points)
        rospy.Service(self.topic_name + '/reset', Reset, self.reset)
    
    def reset(self, data):
        # this function needs to be implemented
        return ResetResponse(True)
    
    def convert_joint_msg_to_list(self, joint_msg):
        joint_names = joint_msg.name
        joints = joint_msg.position
        return [joints[joint_names.index(name)] for name in self.joint_names]
        
    def convert_joint_list_to_message(self, joints):
        msg = sensor_msgs.msg.JointState()
        msg.name = self.joint_names
        msg.position = joints
        return msg
    
    def get_joints(self, data):
        print "get values", self.move_group.get_current_joint_values()
        print "convert to message", self.convert_joint_list_to_message(self.move_group.get_current_joint_values())
        return GetJointsResponse(self.convert_joint_list_to_message(self.move_group.get_current_joint_values()))

    def set_joints(self, data):
        move_group = self.move_group

        joints = self.convert_joint_msg_to_list(data.request_joints)

        move_group.go(joints, wait=True)

        move_group.stop()

        current_joints = move_group.get_current_joint_values()
        
        is_reached = all_close(joints, current_joints, 0.01)
        
        return SetJointsResponse(is_reached, self.convert_joint_list_to_message(current_joints))

    def get_pose(self, data):
        return GetPoseResponse(self.move_group.get_current_pose().pose)

    def check_pose(self, data):
        move_group = self.move_group
        pose_goal = data.request_pose

        (plan, fraction) = move_group.compute_cartesian_path([pose_goal], 10.0, 0.0)
        plan = plan.joint_trajectory
        
        move_group.clear_pose_targets()
        
        joint_names = plan.joint_names
        joints = plan.points[-1].positions
        
        msg = sensor_msgs.msg.JointState()
        msg.name = plan.joint_names
        msg.position = joints
        
        current_angle = self.convert_joint_msg_to_list(self.convert_joint_list_to_message(self.move_group.get_current_joint_values())) # to ensure the name list sequence is the same
        planned_angle = self.convert_joint_msg_to_list(msg) # to ensure the name list sequence is the same
        changes = abs(np.array(planned_angle) - np.array(current_angle))
        
        could_reach = len(plan.points) > 1
        joint_changes = sum(changes)
        
        move_group.clear_pose_targets()
        
        return CheckPoseResponse(could_reach, joint_changes, msg)

    def set_pose(self, data):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## We can plan a motion for this group to a desired pose for the
        ## end-effector:
        pose_goal = data.request_pose

        move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.
        plan = move_group.go(wait=True)
        
        print "plan is:"
        print plan
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()

        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()

        ## END_SUB_TUTORIAL

        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_pose = self.move_group.get_current_pose().pose
        is_reached = all_close(pose_goal, current_pose, 0.01)
        return SetPoseResponse(is_reached, current_pose)
    
    # DO NOT ADD THE THe INITIAL POINT if it is the same as the current one!! (this is a moveit bug)
    # https://answers.ros.org/question/253004/moveit-problem-error-trajectory-message-contains-waypoints-that-are-not-strictly-increasing-in-time/
    def set_trajectory(self, data):
        move_group = self.move_group
        (plan, fraction) = move_group.compute_cartesian_path(data.trajectory, 0.01, 0.0)
        
        print "trajectory planned by moveit"
        print plan
        
        move_group.execute(plan, wait=True)
        
        move_group.stop()

        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets()
        move_group.clear_pose_targets()        
        
        current_pose = self.move_group.get_current_pose().pose
        is_reached = all_close(data.trajectory[-1], current_pose, 0.01)
        
        return SetTrajectoryResponse(is_reached, current_pose)
    
    #def add_plane(self, data):
        #box_name = data.name
        #pose = data.pose
        #normal = (data.normal.x, data.normal.y, data.normal.z)
        #offset = data.offset
        
        #scene = self.scene

        #scene.add_plane(name, pose, normal, offset)

    def forward_kinematics(self, joint_state_msg):
        # https://groups.google.com/forum/#!topic/moveit-users/Wb7TqHuf-ig
        pose = None
        rospy.wait_for_service('compute_fk')
        try:
            fk = rospy.ServiceProxy('compute_fk', moveit_msgs.msg.GetPositionFK)
            header = Header(0, rospy.Time.now(), self.base_frame)
            fk_link_names = [self.move_group.get_end_effector_link()]
            robot_state = self.robot.get_current_state()
            robot_state.joint_state = joint_state_msg
            pose = fk(header, fk_link_names, robot_state).pose_stamped[0].pose
        except rospy.ServiceException, e:
            rospy.logerror("Service call failed: %s"%e)
        
        return pose

    def get_interpolate_points(self, data):
        move_group = self.move_group
        (plan, fraction) = move_group.compute_cartesian_path([data.start, data.end], data.step, 0.0)
        plan = plan.joint_trajectory
        
        move_group.clear_pose_targets()
        
        joint_names = plan.joint_names
        
        poses = []
        
        for joints in plan.points:
            msg = sensor_msgs.msg.JointState()
            msg.name = plan.joint_names
            msg.position = joints.positions
            pose_msg = forward_kinematics(msg)
            poses.pose_msg
            move_group.clear_pose_targets()
        
        return GetInterpolatePointsResponse(poses)

    def display_trajectory(self, plan):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        robot = self.robot
        display_trajectory_publisher = self.display_trajectory_publisher

        ## BEGIN_SUB_TUTORIAL display_trajectory
        ##
        ## Displaying a Trajectory
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ## You can ask RViz to visualize a plan (aka trajectory) for you. But the
        ## group.plan() method does this automatically so this is not that useful
        ## here (it just displays the same trajectory again):
        ##
        ## A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
        ## We populate the trajectory_start with our current robot state to copy over
        ## any AttachedCollisionObjects and add our plan to the trajectory.
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        # Publish
        display_trajectory_publisher.publish(display_trajectory);

        ## END_SUB_TUTORIAL

    def wait_for_state_update(self, object_name, object_is_known=False, object_is_attached=False, timeout=4):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        #box_name = self.box_name
        scene = self.scene

        ## BEGIN_SUB_TUTORIAL wait_for_scene_update
        ##
        ## Ensuring Collision Updates Are Receieved
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        ## If the Python node dies before publishing a collision object update message, the message
        ## could get lost and the box will not appear. To ensure that the updates are
        ## made, we wait until we see the changes reflected in the
        ## ``get_attached_objects()`` and ``get_known_object_names()`` lists.
        ## For the purpose of this tutorial, we call this function after adding,
        ## removing, attaching or detaching an object in the planning scene. We then wait
        ## until the updates have been made or ``timeout`` seconds have passed
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = scene.get_attached_objects([object_name])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the box is in the scene.
            # Note that attaching the box will remove it from known_objects
            is_known = object_name in scene.get_known_object_names()

            # Test if we are in the expected state
            if (object_is_attached == is_attached) and (object_is_known == is_known):               
                return True

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return False
        ## END_SUB_TUTORIAL

    def add_object(self, data):
        timeout = 4
        scene = self.scene

        object_pose = geometry_msgs.msg.PoseStamped()
        object_pose.header.frame_id = self.base_frame
        object_pose.pose = data.pose
        object_name = data.object_name
        object_type = data.object_type
        
        if object_type == AddObjectRequest.TYPE_BOX:
            box_size = (data.size.x, data.size.y, data.size.z)
            scene.add_box(object_name, object_pose, size=box_size)
        elif object_type == AddObjectRequest.TYPE_CYLINDER:
            height = data.size.x
            radius = data.size.y
            try:
                scene.add_cylinder(object_name, object_pose, height, radius)
            except AttributeError as e: # indigo version doesn't have this functions
                # adapted from http://docs.ros.org/melodic/api/moveit_commander/html/planning__scene__interface_8py_source.html#l00296
                co = CollisionObject()
                co.operation = CollisionObject.ADD
                co.id = object_name
                co.header.frame_id = self.base_frame
                cylinder = SolidPrimitive()
                cylinder.type = SolidPrimitive.CYLINDER
                cylinder.dimensions = [height, radius]
                co.primitives = [cylinder]
                co.primitive_poses = [data.pose]
                scene._pub_co.publish(co)
        elif object_type == AddObjectRequest.TYPE_MESH:
            object_filename = data.mesh_filename
            if "/" not in object_filename:
                rospack = rospkg.RosPack()
                current_package = rospack.get_path('control_wrapper')            
                object_filename = current_package + "/pointcloud/" +  object_filename
            scene.add_mesh(object_name, object_pose, object_filename)
        else:
            return AddObjectResponse(False)

        return AddObjectResponse(self.wait_for_state_update(object_name, object_is_known=True, timeout=timeout))

    def attach_object(self, data):
        print "in attach object"
        
        timeout = 4
        scene = self.scene
        
        object_name = data.object_name
        # already attached
        if self.wait_for_state_update(object_name, object_is_known=False, object_is_attached=True, timeout=1):
            print "already attached"
            return AttachObjectResponse(True)
        # not in the workspace
        elif not self.wait_for_state_update(object_name, object_is_known=True, object_is_attached=False, timeout=1):
            print "not in the workspace add it first!"
            # add object
            print "wait for service"
            rospy.wait_for_service(self.topic_name + '/add_object')
            print "service found!"
            add_object = rospy.ServiceProxy(self.topic_name + '/add_object', AddObject)
            try:
                name = object_name
                pose = data.pose
                size = data.size
                mesh_filename = data.mesh_filename
                if "/" not in mesh_filename:
                    rospack = rospkg.RosPack()
                    current_package = rospack.get_path('control_wrapper')            
                    mesh_filename = current_package + "/pointcloud/" +  mesh_filename
                object_type = data.object_type
                response = add_object(name, pose, size, mesh_filename, object_type).is_success
            except rospy.ServiceException as exc:
                print "Service did not process request: " + str(exc)
            if not response:
                print "object not added to world successfully"
                return AttachObjectResponse(False)
            
        print "object added"
        
        # now in the workspace but not attached
        # attach object
        robot = self.robot
        eef_link = self.eef_link
        group_names = self.group_names

        touch_links = robot.get_link_names(group=self.grasping_group)
        if data.object_type == AttachObjectRequest.TYPE_MESH:
            scene.attach_mesh(eef_link, object_name, touch_links=touch_links)
        else:
            scene.attach_box(eef_link, object_name, touch_links=touch_links)

        print "object attched, wait to be updated!"
        
        return AttachObjectResponse(self.wait_for_state_update(object_name, object_is_attached=True, object_is_known=False, timeout=timeout))

    def detach_object(self, data):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        timeout = 4
        
        # detach box
        object_name = data.object_name
        scene = self.scene
        eef_link = self.eef_link

        scene.remove_attached_object(eef_link, name=object_name)
        
        if not self.wait_for_state_update(object_name, object_is_known=True, object_is_attached=False, timeout=timeout):
            return DetachObjectResponse(False)

        # remove box
        if data.to_remove:
            scene.remove_world_object(object_name)
            return DetachObjectResponse(self.wait_for_state_update(object_name, object_is_attached=False, object_is_known=False, timeout=timeout))

        return DetachObjectResponse(True)

    def remove_object(self, data):
        timeout = 4
        object_name = data.object_name
        scene = self.scene

        scene.remove_world_object(object_name)

        # We wait for the planning scene to update.
        return RemoveObjectResponse(self.wait_for_state_update(object_name, object_is_attached=False, object_is_known=False, timeout=timeout))
    
    def scene_object_allow_collision(self, data):
        object_id = data.name
        allow_collision = data.allow_collision
        
        # the object is not in the scene
        if object_id not in self.scene.get_objects().keys() and object_id not in self.scene.get_attached_objects().keys():
            return
        
        rospy.wait_for_service('/get_planning_scene', 10.0)
        get_planning_scene = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)
        
        request = PlanningSceneComponents(components=PlanningSceneComponents.ALLOWED_COLLISION_MATRIX)
        response = get_planning_scene(request)
        
        original_acm = response.scene.allowed_collision_matrix

        original_acm.default_entry_names.append(object_id)
        original_acm.default_entry_values.append(allow_collision)
        planning_scene_diff = PlanningScene(is_diff=True, allowed_collision_matrix = original_acm)
        self.planning_scene_publisher.publish(planning_scene_diff)
    
    def run(self):
        ros_rate = rospy.Rate(30)
        
        while not rospy.is_shutdown():
            free_objects = self.scene.get_objects()
            
            objects_msg = SceneObjects()
            
            for object_id in free_objects.keys():
                object_msg = SceneObject()
                object_msg.name = object_id
                if len(free_objects[object_id].primitive_poses) != 0:
                    object_msg.pose = free_objects[object_id].primitive_poses[0]
                elif len(free_objects[object_id].mesh_poses) != 0:
                    object_msg.pose = free_objects[object_id].mesh_poses[0]
                else:
                    object_msg.pose = free_objects[object_id].plane_poses[0]
                object_msg.state = SceneObject.STATE_FREE
                objects_msg.objects.append(object_msg)
            
            attached_objects = self.scene.get_attached_objects()
            
            for attached_object_id in attached_objects.keys():
                object_msg = SceneObject()
                object_msg.name = attached_object_id
                if len(attached_objects[attached_object_id].object.primitive_poses) != 0:
                    object_msg.pose = attached_objects[attached_object_id].object.primitive_poses[0]
                elif len(attached_objects[attached_object_id].object.mesh_poses) != 0:
                    object_msg.pose = attached_objects[attached_object_id].object.mesh_poses[0]
                else:
                    object_msg.pose = attached_objects[attached_object_id].object.plane_poses[0]
                
                Tee_object = pose_msg_to_matrix(object_msg.pose)
                Tworld_ee = pose_msg_to_matrix(self.move_group.get_current_pose().pose)
                Tworld_object = np.matmul(Tworld_ee, Tee_object)
                object_msg.pose = pose_matrix_to_msg(Tworld_object)
                
                object_msg.state = SceneObject.STATE_ATTACHED
                objects_msg.objects.append(object_msg)
            
            self.scene_obj_publisher.publish(objects_msg)
            
            ros_rate.sleep()
            
