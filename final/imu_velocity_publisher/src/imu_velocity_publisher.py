#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class OdomVelocityPublisher:
    def __init__(self):
        rospy.init_node('odom_velocity_publisher')


        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)


        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # Initialize twist message
        self.vel_cmd = Twist()

        rospy.spin()

    def odom_callback(self, odom_msg):
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y
        z = odom_msg.pose.pose.position.z

        orientation_q = odom_msg.pose.pose.orientation
        (roll, pitch, yaw) = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        print("x: ",x,"y: ",y)


        self.vel_cmd.linear.x = 0.1  
        self.vel_cmd.linear.y = 0.1
        self.vel_pub.publish(self.vel_cmd)

if __name__ == '__main__':
    try:
        odom_velocity_publisher = OdomVelocityPublisher()
    except rospy.ROSInterruptException:
        pass

