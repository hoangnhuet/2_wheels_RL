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
from std_msgs.msg import Float64
from tf.transformations import euler_from_quaternion

class GazeboEnv_B:
    def __init__(self, launchfile, package):
        self.odom_x = 0
        self.odom_y = 0
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
        
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        rospy.init_node("gym", anonymous=True)
        self.package = package
        self.launch_file = launchfile
        self.command = ['roslaunch', self.package, self.launch_file]

        # Start Gazebo in a separate thread
        self.process = threading.Thread(target=self.launching)
        self.process.start()

        self.rw_controller = rospy.Publisher("/right_wheel_effort_controller/command", Float64, queue_size=1)
        self.lw_controller = rospy.Publisher("/left_wheel_effort_controller/command", Float64, queue_size=1)
        self.h_controller = rospy.Publisher("/hj_position_controller/command", Float64, queue_size=1)
        self.set_state = rospy.Publisher("gazebo/set_model_state", ModelState, queue_size=10)
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
    
    def launching(self):
        try:
            subprocess.check_call(self.command)
        except subprocess.CalledProcessError as e:
            print(f"Error launching Gazebo: {e}")
        except KeyboardInterrupt:
            print("\nInterrupted")

    def odom_callback(self, od_data):
        self.last_odom = od_data

    def step(self, action):
        # Publish wheel efforts based on action
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

        # Get roll, pitch, yaw from the last odom message
        if self.last_odom:
            orientation_q = self.last_odom.pose.pose.orientation
            (roll, pitch, yaw) = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        else:
            roll, pitch, yaw = 0.0, 0.0, 0.0

        done = self.check_collapse(roll, pitch)
        reward = self.get_reward(done, roll, pitch)

        state = np.array([roll, pitch, yaw])
        return state, reward, done

    def reset(self):
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_world service call failed")

        quaternion = Quaternion.from_euler(0.0, 0.0, 0.0)
        object_state = self.set_self_state
        object_state.pose.position.x = 0
        object_state.pose.position.y = 0
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)
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

        # Get initial roll, pitch, yaw
        if self.last_odom:
            orientation_q = self.last_odom.pose.pose.orientation
            (roll, pitch, yaw) = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        else:
            roll, pitch, yaw = 0.0, 0.0, 0.0

        state = np.array([roll, pitch, yaw])
        return state

    def get_reward(self, done, roll, pitch):
        reward = 0
        if done:
            reward -= 20  # Penalty for collapse
        else:
            # Penalize large roll or pitch angles
            reward -= abs(roll) * 2 + abs(pitch) * 2
        return reward
    def check_collapse(self, roll, pitch):
        if abs(roll) > self.roll_threshold or abs(pitch) > self.pitch_threshold:
            return True
        return False
    @staticmethod
    def cleanup_processes():
        subprocess.call(["pkill", "-f", "gzserver"])
        subprocess.call(["pkill", "-f", "gzclient"])

    @staticmethod
    def cleanup_ros_nodes():
        subprocess.call(["rosnode", "kill", "-a"])

    def cleanup(self):
        self.cleanup_processes()
        self.cleanup_ros_nodes()

