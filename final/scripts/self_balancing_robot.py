#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

class SelfBalancingRobot:
    def __init__(self):
        rospy.init_node('self_balancing_robot')

        self.vel_pub = rospy.Publisher('/final/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/final/odom', Odometry, self.odom_callback)

        self.pitch = 0.0
        self.pitch_rate = 0.0
        self.vel_cmd = Twist()

        self.kp = 10.0
        self.ki = 0.5
        self.kd = 1.0

        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = rospy.Time.now()

        self.disturbance_duration = 1.0  # Duration to apply the disturbance
        self.disturbance_applied = False

        rospy.spin()

    def odom_callback(self, odom_msg):
        orientation_q = odom_msg.pose.pose.orientation
        (roll, pitch, yaw) = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])

        self.pitch = pitch
        self.pitch_rate = odom_msg.twist.twist.angular.y

        if not self.disturbance_applied:
            self.apply_disturbance()
        else:
            self.update_control()

    def apply_disturbance(self):
        self.vel_cmd.linear.x = 0.1 
        self.vel_pub.publish(self.vel_cmd)
        rospy.sleep(self.disturbance_duration)  
        self.vel_cmd.linear.x = 0.0  
        self.vel_pub.publish(self.vel_cmd)
        self.disturbance_applied = True  

    def update_control(self):
        current_time = rospy.Time.now()
        dt = (current_time - self.previous_time).to_sec()

        if dt == 0:  # Prevent division by zero
            return

        error = 0.0 - self.pitch
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt

        control_effort = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Ensure control effort is within reasonable bounds
        control_effort = max(min(control_effort, 1.0), -1.0)

        self.vel_cmd.angular.z = 0.0
        self.vel_cmd.linear.x = control_effort
        self.vel_pub.publish(self.vel_cmd)

        # Print the command for debugging
        rospy.loginfo(f"Control Effort: {control_effort}, Error: {error}, Integral: {self.integral}, Derivative: {derivative}")

        self.previous_error = error
        self.previous_time = current_time

    def stop_robot(self):
        # Send zero velocity to stop the robot
        self.vel_cmd.linear.x = 0.0
        self.vel_cmd.angular.z = 0.0
        self.vel_pub.publish(self.vel_cmd)

if __name__ == '__main__':
    try:
        self_balancing_robot = SelfBalancingRobot()
    except rospy.ROSInterruptException:
        pass

