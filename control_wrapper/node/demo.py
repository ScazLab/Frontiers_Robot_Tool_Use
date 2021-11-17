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

import numpy as np

import rospy
import rospkg
from std_msgs.msg import String, Bool
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3

from moveit_msgs.srv import GetPlanningScene
from moveit_msgs.msg import PlanningScene, PlanningSceneComponents, AllowedCollisionMatrix, AllowedCollisionEntry

from control_wrapper.msg import SceneObjectAllowCollision

from control_wrapper.srv import SetPose
from control_wrapper.srv import GetPose
from control_wrapper.srv import GetJoints
from control_wrapper.srv import SetJoints

from control_wrapper.srv import AddObject, AddObjectRequest
from control_wrapper.srv import AttachObject, AttachObjectRequest
from control_wrapper.srv import DetachObject, DetachObjectRequest
from control_wrapper.srv import RemoveObject, RemoveObjectRequest

from control_wrapper.srv import Reset, ResetRequest

class Demo(object):
    def __init__(self, robot, side):
        self.topic = "/" + robot + "/control_wrapper/" + side + "/"
        
        self.free_drive_pub = rospy.Publisher(self.topic + "enable_freedrive", Bool, queue_size=10)
        self.gripper_pub = rospy.Publisher(self.topic + "gripper", Bool, queue_size=10)
        self.connect_pub = rospy.Publisher(self.topic + "connect", Bool, queue_size=10)
        self.collision_pub = rospy.Publisher(self.topic + 'scene_allow_collision', SceneObjectAllowCollision, queue_size=10)
    
    def get_pose(self):
        rospy.wait_for_service(self.topic + "get_pose")
        get_current_pose = rospy.ServiceProxy(self.topic + "get_pose", GetPose)
        current_pose = None
        try:
            current_pose = get_current_pose().pose
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc) 
        return current_pose
    
    def get_angle(self):
        rospy.wait_for_service(self.topic + "get_joints")
        get_current_joints = rospy.ServiceProxy(self.topic + "get_joints", GetJoints)
        current_joints = None
        try:
            current_joints = get_current_joints().joints
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc) 
        return current_joints
    
    def set_angle(self):
        rospy.wait_for_service(self.topic + "get_joints")
        get_current_joints = rospy.ServiceProxy(self.topic + "get_joints", GetJoints)

        try:
            response = get_current_joints().joints
            joint_names = response.name
            current_joints = response.position
            new_joint_angles = [i for i in current_joints]
            for i in range(len(joint_names)):
                angle_change_degree = float(raw_input("{}({}): ".format(joint_names[i], np.degrees(current_joints[i]))))
                angle_change_radian = np.radians(angle_change_degree)
                if angle_change_radian != 0.0:
                    new_joint_angles[i] += angle_change_radian
                
                    rospy.wait_for_service(self.topic + "set_joints")
                    set_joints = rospy.ServiceProxy(self.topic + "set_joints", SetJoints) 
                    
                    joints = JointState()
                    joints.name = joint_names
                    joints.position = new_joint_angles
                    is_reached = set_joints(joints).is_reached
           
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc) 
      
    def set_default_angles(self):
        rospy.wait_for_service(self.topic + "reset")
        reset = rospy.ServiceProxy(self.topic + "reset", Reset)
        try:
            response = reset().is_reached
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc)      
    
    def set_goal_default_postion(self):
        self.set_default_angles()
    
    def add_box(self):
        rospy.wait_for_service(self.topic + "add_object")
        add_box = rospy.ServiceProxy(self.topic + "add_object", AddObject)
        try:
            name = "box_object"
            #pose = Pose(Point(-0.34, -0.0075, -0.023), Quaternion(0.0, 0.0, 0.0, 1.0))
            pose = Pose(Point(0.5, 0.5, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0))
            size = Vector3(0.1, 0.1, 0.1)
            object_type = AddObjectRequest.TYPE_BOX
            mesh_filename = ""
            response = add_box(name, pose, size, mesh_filename, object_type).is_success
            if response:
                print "successfully added!"
            else:
                print "did not add successfully"
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc)
    
    def add_desk(self):
        rospy.wait_for_service(self.topic + "add_object")
        add_box = rospy.ServiceProxy(self.topic + "add_object", AddObject)
        try:
            name = "desk"
            #pose = Pose(Point(-0.34, -0.0075, -0.023), Quaternion(0.0, 0.0, 0.0, 1.0))
            pose = Pose(Point(0.5, 0.5, -0.1), Quaternion(0.0, 0.0, 0.0, 1.0))
            size = Vector3(0.7, 0.4, 0.1)
            object_type = AddObjectRequest.TYPE_BOX
            mesh_filename = ""
            response = add_box(name, pose, size, mesh_filename, object_type).is_success
            if response:
                print "successfully added!"
            else:
                print "did not add successfully"
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc)    
    
    def allow_collision_added_box(self, is_allow_collision):
        msg = SceneObjectAllowCollision("box_object", is_allow_collision)
        self.collision_pub.publish(msg)
    
    def get_scene_info(self):
        rospy.wait_for_service('/get_planning_scene', 10.0)
        get_planning_scene = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)
        
        request = PlanningSceneComponents(components=sum([PlanningSceneComponents.SCENE_SETTINGS,
                                                          PlanningSceneComponents.ROBOT_STATE,
                                                          PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS,
                                                          PlanningSceneComponents.WORLD_OBJECT_NAMES,
                                                          PlanningSceneComponents.WORLD_OBJECT_GEOMETRY,
                                                          PlanningSceneComponents.OCTOMAP,
                                                          PlanningSceneComponents.TRANSFORMS,
                                                          PlanningSceneComponents.ALLOWED_COLLISION_MATRIX,
                                                          PlanningSceneComponents.LINK_PADDING_AND_SCALING,
                                                          PlanningSceneComponents.OBJECT_COLORS]))
        response = get_planning_scene(request)
        print "attached object from scene:"
        print response.scene.robot_state.attached_collision_objects
    
    def attach_box(self):
        print "wait for service"
        rospy.wait_for_service(self.topic + "attach_object")
        print "service found"
        attach_box = rospy.ServiceProxy(self.topic + "attach_object", AttachObject)
        try:
            name = "box_tool"
            pose = self.get_pose()
            #pose = Pose(Point(-0.34, -0.0075, -0.023), Quaternion(0.0, 0.0, 0.0, 1.0))
            size = Vector3(0.1, 0.1, 0.1)
            mesh_filename = ""
            object_type = AttachObjectRequest.TYPE_BOX
            response = attach_box(name, pose, size, mesh_filename, object_type).is_success
            if response:
                print "successfully attached!"
            else:
                print "did not attach successfully"
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc)
    
    def detach_box(self):
        rospy.wait_for_service(self.topic + "detach_object")
        detach_box = rospy.ServiceProxy(self.topic + "detach_object", DetachObject)
        try:
            name = "box_tool"
            response = detach_box(name, True).is_success
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc)
    
    def remove_box(self):
        rospy.wait_for_service(self.topic + "remove_object")
        remove_box = rospy.ServiceProxy(self.topic + "remove_object", RemoveObject)
        try:
            name = "box_object"
            response = remove_box(name).is_success
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc)
    
    def add_mesh(self):
        rospy.wait_for_service(self.topic + "add_object")
        add_box = rospy.ServiceProxy(self.topic + "add_object", AddObject)
        try:
            name = "mesh_object_1"
            #pose = self.get_pose()
            #pose = Pose(Point(-0.34, -0.0075, -0.023), Quaternion(0.0, 0.0, 0.0, 1.0))
            pose = Pose(Point(0.3, 0.3, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0))
            size = Vector3(1.0, 1.0, 1.0)
            rospack = rospkg.RosPack()
            current_package = rospack.get_path('control_wrapper')            
            mesh_filename = current_package + "/mesh/chineseknife_1_3dwh_out_2_50_fused.ply"
            object_type = AddObjectRequest.TYPE_MESH
            response = add_box(name, pose, size, mesh_filename, object_type).is_success
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc)
    
    def attach_mesh(self):
        print "wait for service"
        rospy.wait_for_service(self.topic + "attach_object")
        print "service found"
        attach_box = rospy.ServiceProxy(self.topic + "attach_object", AttachObject)
        try:
            name = "mesh_object_2"
            #pose = Pose(Point(-0.34, -0.0075, -0.023), Quaternion(0.0, 0.0, 0.0, 1.0))
            pose = self.get_pose()
            size = Vector3(1.0, 1.0, 1.0)
            rospack = rospkg.RosPack()
            current_package = rospack.get_path('control_wrapper')            
            mesh_filename = current_package + "/mesh/chineseknife_1_3dwh_out_2_50_fused.ply"
            object_type = AddObjectRequest.TYPE_MESH
            response = attach_box(name, pose, size, mesh_filename, object_type).is_success
            if response:
                print "successfully attached!"
            else:
                print "did not attach successfully"
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc)
    
    def detach_mesh(self):
        rospy.wait_for_service(self.topic + "detach_object")
        detach_box = rospy.ServiceProxy(self.topic + "detach_object", DetachObject)
        try:
            name = "mesh_object_2"
            response = detach_box(name, True).is_success
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc)
    
    def remove_mesh(self):
        rospy.wait_for_service(self.topic + "remove_object")
        remove_box = rospy.ServiceProxy(self.topic + "remove_object", RemoveObject)
        try:
            name = "mesh_object_1"
            response = remove_box(name).is_success
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc)    
    
    def move_arm(self, direction=None):
        current_pose = self.get_pose()
        
        rospy.wait_for_service(self.topic + "set_pose")
        set_current_pose = rospy.ServiceProxy(self.topic + "set_pose", SetPose)
        try:
            if direction is None:
                current_pose.position.x = float(raw_input("position x: "))
                current_pose.position.y = float(raw_input("position y: "))
                current_pose.position.z = float(raw_input("position z: "))
                current_pose.orientation.x = float(raw_input("orientation x: "))
                current_pose.orientation.y = float(raw_input("orientation y: "))
                current_pose.orientation.z = float(raw_input("orientation z: "))
                current_pose.orientation.w = float(raw_input("orientation w: "))
                
                ## downloaded testing position
                ##current_pose.position.x = 0.0409343985978
                ##current_pose.position.y = 0.232418167122
                ##current_pose.position.z = 0.362970810641
                ##current_pose.orientation.x = -0.572493767534
                ##current_pose.orientation.y = 0.393922042086
                ##current_pose.orientation.z = 0.363487039587
                ##current_pose.orientation.w = 0.620446196656
                
                ## downloaded left down position
                ##current_pose.position.x = 0.0396019622729
                ##current_pose.position.y = 0.226827045945
                ##current_pose.position.z = 0.138745943708
                ##current_pose.orientation.x = -0.98840445644
                ##current_pose.orientation.y = -0.0272096944741
                ##current_pose.orientation.z = -0.0143484046645
                ##current_pose.orientation.w = 0.148695616275
                
                ## downloaded desk down position
                #current_pose.position.x = 0.0258132977872
                #current_pose.position.y = -0.234580830708
                #current_pose.position.z = 0.180995232298
                #current_pose.orientation.x = 0.990171113745
                #current_pose.orientation.y = -0.000169798305907
                #current_pose.orientation.z =  0.00110507027509
                #current_pose.orientation.w =  0.139856767777
                
                # end
                #current_pose.position.x = 0.0937673666222
                #current_pose.position.y = -0.224195986097
                #current_pose.position.z = 0.112892926565
                #current_pose.orientation.x = 0.998105326863
                #current_pose.orientation.y = 0.0599306980812
                #current_pose.orientation.z = 0.00333463315477
                #current_pose.orientation.w = 0.0135258322126
                
                # start
                #current_pose.position.x = 0.0166845576822
                #current_pose.position.y = -0.23441005703
                #current_pose.position.z = 0.112965311791
                #current_pose.orientation.x = 0.998643701429
                #current_pose.orientation.y = 0.0503722187734
                #current_pose.orientation.z = -0.00107326297116
                #current_pose.orientation.w = 0.0131242248511
                
                #current_pose_transformation = np.array([[ 9.99303014e-01, -2.34514575e-03, 3.72558228e-02, 1.12575944e-02],
                                                        #[-2.34915824e-03, -9.99997242e-01,  6.40959561e-05, -2.30165499e-01],
                                                        #[ 3.72555693e-02, -1.51574659e-04, -9.99305755e-01,  1.12458831e-01],
                                                        #[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
                
                #from tf import transformations as tfs
                #quaternion = tfs.quaternion_from_matrix(current_pose_transformation)
                #print "quaternion"
                #print quaternion
                #current_pose.orientation.x = quaternion[0]
                #current_pose.orientation.y = quaternion[1]
                #current_pose.orientation.z = quaternion[2]
                #current_pose.orientation.w = quaternion[3]             
                
                
            elif direction == "x+":
                current_pose.position.x += 0.05
            elif direction == "x-":
                current_pose.position.x -= 0.05
            elif direction == "y+":
                current_pose.position.y += 0.05
            elif direction == "y-":
                current_pose.position.y -= 0.05
            elif direction == "z+":
                current_pose.position.z += 0.05
            elif direction == "z-":
                current_pose.position.z -= 0.05
            response = set_current_pose(current_pose)
            pose = response.response_pose
            is_reached = response.is_reached
        except rospy.ServiceException as exc:
            print "Service did not process request: " + str(exc)
    
    def run(self):
        while not rospy.is_shutdown():
            print "====================================================="
            command_input = raw_input("Freedrive: fs(start);fe-end: \nGripper: go(open); gc(close);\nConnect: c(connect);\nGet End Effector Pose: ep; \nGet joint angles: ja; \nSet joint angles: sa\nGo to Default Position: d; \ngo to the goal default position: gd;\nBoxes: ad(add desk); ab(add box); eab(enable added box collision); dab(disable added box collision); atb(attach box); ;db(detach box); rb(remove box) \nScene info: si\nMeshs: am(add mesh); atm(attach mesh); dm(detach mesh); rm(remove mesh) \nMove arm: ma(specify pose); x+(x direction move up 5 cm); x-; y+; y-; z+; z-: \n")
            if command_input == "fs":
                self.free_drive_pub.publish(True)
            elif command_input == "fe":
                self.free_drive_pub.publish(False)
            elif command_input == "go":
                self.gripper_pub.publish(True)
            elif command_input == "gc":
                self.gripper_pub.publish(False)
            elif command_input == "c":
                self.connect_pub.publish(True)
            elif command_input == "ep":
                print self.get_pose()
            elif command_input == "ja":
                angle_msg = self.get_angle()
                angle_names = angle_msg.name
                angle_positions = angle_msg.position
                for i in range(len(angle_names)):
                    print "{}: {}".format(angle_names[i], np.rad2deg(angle_positions[i]))
            elif command_input == "sa":
                self.set_angle()
            elif command_input == "d":
                self.set_default_angles()
            elif command_input == "ad":
                self.add_desk()
            elif command_input == "ab":
                self.add_box()
            elif command_input == "eab":
                self.allow_collision_added_box(True)
            elif command_input == "dab":
                self.allow_collision_added_box(False)
            elif command_input == "atb":
                self.attach_box()
            elif command_input == "db":
                self.detach_box()
            elif command_input == "rb":
                self.remove_box()
            elif command_input == "si":
                self.get_scene_info()
            elif command_input == "am":
                self.add_mesh()
            elif command_input == "atm":
                self.attach_mesh()
            elif command_input == "dm":
                self.detach_mesh()
            elif command_input == "rm":
                self.remove_mesh() 
            elif command_input == "ma":
                self.move_arm()
            elif command_input == "gd":
                self.set_goal_default_postion()
            else: # move arm
                direction = command_input
                self.move_arm(direction)