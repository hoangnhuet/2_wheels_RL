import math
import threading
import os
import sys
import random
import subprocess
import time
from os import path
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
from std_msgs.msg import Float64
from tf.transformations import euler_from_quaternion
import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)
# logging.getLogger('supervision').setLevel(logging.ERROR)

class GazeboEnv:
    def __init__(self, launchfile, environment_dim, package):
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0
        self.odom_r = 0
        self.odom_p = 0
        self.odom_y = 0
        self.actor_x = 0
        self.actor_y = 0

        self.velodyne_data = np.ones(self.environment_dim) * 10
        self.last_odom = None
        self.set_self_state = ModelState()
        self.set_self_state.model_name = "final"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0
        self.roll_threshold = 0.5
        self.pitch_threshold = 0.5
        self.time_step = 0
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.collapse_callback)
        self.person_data = [-1, -1, -1, -1]  # YOLO xyxy
        self.gaps = np.linspace(-math.pi, math.pi, self.environment_dim + 1)
        self.collapse = False
        self.bridge = CvBridge()
        # original_stdout = sys.stdout
        # sys.stdout = open(os.devnull, 'w')
        self.yolo_model = YOLO('yolov8n.pt')
        # sys.stdout = original_stdout
        rospy.init_node("gym", anonymous=True)
        self.package = package
        self.launch_file = launchfile
        self.command = ['roslaunch', self.package, self.launch_file]
        self.process = threading.Thread(target=self.launching)
        self.process.start()
        self.rw_controller = rospy.Publisher("/right_wheel_effort_controller/command", Float64, queue_size=1)
        self.lw_controller = rospy.Publisher("/left_wheel_effort_controller/command", Float64, queue_size=1)
        self.h_controller = rospy.Publisher("/hj_position_controller/command", Float64, queue_size=1)
        self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.velodyne = rospy.Subscriber("/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1)
        self.odom = rospy.Subscriber("/odom", Odometry, self.odom_callback, queue_size=1)
        self.human_pos = rospy.Subscriber("/gazebo/model_states", ModelStates, self.actor_pos_callback)
        self.image_sub = rospy.Subscriber('/final/camera/camera/image_raw', Image, self.image_callback)
        self.euclid_distance = 10
        self.test_dist = [5,5]
    
    def launching(self):
        try:
            subprocess.check_call(self.command)
        except subprocess.CalledProcessError as e:
            print(f"Error launching Gazebo: {e}")
        except KeyboardInterrupt:
            print("\nInterrupted")

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return
        results = self.yolo_model(cv_image)[0]

        person_class_id = 0
        if len(results) == 0 or len(results[0].boxes.xyxy) == 0:
            self.person_data = [[-1, -1, -1, -1]]
        else:
            self.person_data = []
            for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
                if int(cls) == person_class_id:
                    self.person_data.append(box.cpu().numpy())

    def velodyne_callback(self, data):
        points = list(pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10
        for point in points:
            x, y, z = point[:3]
            beta = math.atan2(y, x)
            dist = math.sqrt(x ** 2 + y ** 2)
            for j in range(len(self.gaps) - 1):
                if self.gaps[j] <= beta < self.gaps[j + 1]:
                    self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                    break
    def collapse_callback(self, odom_msg):
        orientation_q = odom_msg.pose.pose.orientation
        (roll, pitch, yaw) = euler_from_quaternion(
            [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])

        if abs(roll) > self.roll_threshold or abs(pitch) > self.pitch_threshold:
            self.collapse = True
        else:
            self.collapse = False

    def odom_callback(self, od_data):
        self.last_odom = od_data

    def actor_pos_callback(self, msg):
        try:
            actor_index = msg.name.index("actor_walking")
            actor_pose = msg.pose[actor_index]
            self.actor_x = actor_pose.position.x
            self.actor_y = actor_pose.position.y
        except ValueError:
            pass

    def step(self, action):
        target = False
        self.rw_controller.publish(action[0])
        self.lw_controller.publish(action[1])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")
        time.sleep(0.1)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        if self.last_odom:
            orientation_q = self.last_odom.pose.pose.orientation
            (roll, pitch, yaw) = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        else:
            roll, pitch, yaw = 0.0, 0.0, 0.0

        local_state = np.array([self.odom_x, self.odom_y, roll, pitch, yaw])
        # print("PREVIOUS: ",self.velodyne_data)
        # print("AFTER: ", (self.velodyne_data - np.mean(self.velodyne_data))/np.cov(self.velodyne_data))
        # print("MEANS: ", np.mean(self.velodyne_data), "COV: ", np.cov(self.velodyne_data))
        # self.velodyne_data = (self.velodyne_data - np.mean(self.velodyne_data))/np.cov(self.velodyne_data)
        state = np.concatenate((self.velodyne_data, self.person_data), axis=None)
        print("PERSON: ",self.person_data)
        # print("RPY: ",roll, pitch,  yaw)
        # print("LASER: ",self.velodyne_data)

        done = self.collapse or self.observe_collision(self.velodyne_data)
        reward = self.get_reward(target, done, action, min(self.velodyne_data), roll, pitch)
        # print("ODOM: ", self.odom_x, self.odom_y)
        # self.euclid_distance = math.sqrt((self.odom_x - self.actor_x)**2 + (self.odom_y - self.actor_y)**2)
        self.euclid_distance = math.sqrt((self.odom_x - self.test_dist[0])**2 + (self.odom_y - self.test_dist[1])**2)
        # print("ACTOR: ", self.actor_x, self.actor_y)
        # return state, reward, done, target
        return local_state, state, reward, done, target

    def reset(self):
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_world service call failed")
        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, 0.0)
        object_state = self.set_self_state
        object_state.pose.position.x = 0
        object_state.pose.position.y = 0
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y
        if self.last_odom:
            orientation_q = self.last_odom.pose.pose.orientation
            (roll, pitch, yaw) = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        else:
            roll, pitch, yaw = 0.0, 0.0, 0.0

        local_state = np.array([self.odom_x, self.odom_y, roll, pitch, yaw])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print("/gazebo/unpause_physics service call failed")
        time.sleep(0.1)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as e:
            print("/gazebo/pause_physics service call failed")
        # print("BEFORE: ",self.velodyne_data)
        # self.velodyne_data = (self.velodyne_data - np.mean(self.velodyne_data))/np.cov(self.velodyne_data)
        # print("AFTER: ",self.velodyne_data)


        state = np.concatenate((self.velodyne_data, self.person_data), axis=None)
        print("RESET STATE: ", state.shape)
        return local_state, state
        # return state

    @staticmethod
    def observe_collision(laser_data):
        return np.min(laser_data) < 0.1

    @staticmethod
    def cleanup_processes():
        subprocess.call(["pkill", "-f", "gzserver"])
        subprocess.call(["pkill", "-f", "gzclient"])

    @staticmethod
    def cleanup_ros_nodes():
        subprocess.call(["rosnode", "kill", "-a"])

    def get_reward(self, target, done, action, min_laser, roll, pitch):
        reward = 0
        if self.time_step < 150000:
            reward = reward - 4*abs(roll) * 2 - abs(pitch) * 2 - abs(action[0]) *2 - abs(action[1])*2
        else:
            reward  = reward - abs(roll) * 2 - abs(pitch) * 2  - abs(action[0]) *2 - abs(action[1])*2 - self.euclid_distance - min_laser

        # print("DISTANCE: ",self.euclid_distance)
        # print("min_laser: ", min_laser)
        # print(self.time_step)
        self.time_step +=1
        return reward

    def cleanup(self):
        self.cleanup_processes()
        self.cleanup_ros_nodes()

# the reward of mujoco often consist 3 parts: Healthy rew for surviving each step, forward reward 
# with each motion that lead to the right orientation, and penalty rew for take too large action 


# in very first step of training, the vehicle couldn't self balancing so that when it fall, the min laser will
# be impacted and not work properly, so min laser should be consider after the car can balanced by itself. 
