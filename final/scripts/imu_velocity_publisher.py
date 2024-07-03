#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class OdomVelocityPublisher:
    def __init__(self):
        rospy.init_node('odom_velocity_publisher')

        self.vel_pub = rospy.Publisher('/final/cmd_vel', Twist, queue_size=10)

        rospy.Subscriber('/final/odom', Odometry, self.odom_callback)

        self.vel_cmd = Twist()

        rospy.spin()

    def odom_callback(self, odom_msg):
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y

        orientation_q = odom_msg.pose.pose.orientation
        (roll, pitch, yaw) = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])

        print("x: ", x, "y: ", y, "yaw: ", yaw)

        # Calculate the necessary angular velocity to correct the yaw
        # Here, kp is a proportional gain for the controller
        kp = 1.0
        self.vel_cmd.angular.z = -kp * yaw

        self.vel_cmd.linear.x = 0.1
        self.vel_cmd.linear.y = 0.1
        self.vel_pub.publish(self.vel_cmd)

if __name__ == '__main__':
    try:
        odom_velocity_publisher = OdomVelocityPublisher()
    except rospy.ROSInterruptException:
        pass

