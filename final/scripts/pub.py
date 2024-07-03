#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

class IMUVelocityPublisher:
    def __init__(self):
        rospy.init_node('imu_velocity_publisher')
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/imu', Imu, self.imu_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.vel_cmd = Twist()

        rospy.spin()

    def imu_callback(self, imu_msg):
        pass

    def odom_callback(self, odom_msg):

        self.vel_cmd.linear.x = 0.2 

        self.vel_pub.publish(self.vel_cmd)

if __name__ == '__main__':
    try:
        imu_velocity_publisher = IMUVelocityPublisher()
    except rospy.ROSInterruptException:
        pass

