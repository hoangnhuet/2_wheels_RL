#!/usr/bin/env python

import rospy
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry

def odom_callback(data):
    rospy.loginfo("Odometry: %s", data)

def main():
    rospy.init_node('controller', anonymous=True)

    # Publishers for wheel effort controllers
    right_wheel_pub = rospy.Publisher('/right_wheel_effort_controller/command', Float64, queue_size=10)
    left_wheel_pub = rospy.Publisher('/left_wheel_effort_controller/command', Float64, queue_size=10)

    # Subscriber for odometry
    rospy.Subscriber('/odom', Odometry, odom_callback)

    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        # Publish command = 1 to both wheel effort controllers
        command = Float64()
        command1 = Float64()
        command.data = 1.0
        command1.data = -1.0
        right_wheel_pub.publish(command)
        left_wheel_pub.publish(command1)

        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

