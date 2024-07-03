#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

class CollapseChecker:
    def __init__(self):
        rospy.init_node('collapse_checker', anonymous=True)


        self.roll_threshold = 0.5  
        self.pitch_threshold = 0.5  

        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        rospy.spin()

    def odom_callback(self, odom_msg):

        orientation_q = odom_msg.pose.pose.orientation
        (roll, pitch, yaw) = euler_from_quaternion(
            [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])


        if abs(roll) > self.roll_threshold or abs(pitch) > self.pitch_threshold:
            rospy.logwarn("The car has collapsed!")
        else:
            rospy.loginfo("The car is stable.")

if __name__ == '__main__':
    try:
        CollapseChecker()
    except rospy.ROSInterruptException:
        pass
