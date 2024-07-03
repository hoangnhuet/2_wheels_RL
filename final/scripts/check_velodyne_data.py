#!/usr/bin/env python

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointCloud
from geometry_msgs.msg import Point32
import numpy as np
import math

class VelodyneProcessor:
    def __init__(self):
        # Initialize node
        rospy.init_node('velodyne_processor', anonymous=True)
        
        # Parameters
        self.environment_dim = 36  # Number of samples
        self.gaps = np.linspace(-math.pi, math.pi, self.environment_dim + 1)  # Create 36 angular gaps
        self.velodyne_data = np.ones(self.environment_dim) * 10  # Initialize with large default values

        # Subscriber
        rospy.Subscriber("/velodyne_points", PointCloud2, self.velodyne_callback)
        
        # Publisher
        self.pc_pub = rospy.Publisher("/velodyne_2d", PointCloud, queue_size=10)
        
        rospy.loginfo("Node started and listening to /velodyne_points")
        rospy.spin()

    def velodyne_callback(self, v):
        # Read x, y, z fields from PointCloud2
        data = list(pc2.read_points(v, skip_nans=True, field_names=("x", "y", "z")))
        
        # Initialize velodyne_data with large values
        self.velodyne_data = np.ones(self.environment_dim) * 10
        
        # Process each point in the point cloud
        for point in data:
            # Extract x, y, z coordinates
            x, y, z = point[:3]
            
            # Calculate angle beta
            beta = math.atan2(y, x)
            
            # Calculate distance
            dist = math.sqrt(x ** 2 + y ** 2)
            
            # Update velodyne_data based on gaps
            for j in range(len(self.gaps) - 1):
                if self.gaps[j] <= beta < self.gaps[j + 1]:
                    self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                    break
        
        # Publish 2D representation to a PointCloud message
        self.publish_2d_representation()

    def publish_2d_representation(self):
        # Create a PointCloud message
        pc_msg = PointCloud()
        pc_msg.header.stamp = rospy.Time.now()
        pc_msg.header.frame_id = "velodyne"  # Set appropriate frame
        
        # Create Point32 points from velodyne_data
        for i, dist in enumerate(self.velodyne_data):
            angle = (self.gaps[i] + self.gaps[i + 1]) / 2  # Midpoint of the angle gap
            x = dist * math.cos(angle)
            y = dist * math.sin(angle)
            point = Point32()
            point.x = x
            point.y = y
            point.z = 0  # 2D representation, so z = 0
            pc_msg.points.append(point)
        
        # Ensure there are exactly 36 points
        assert len(pc_msg.points) == 36, "Number of points is not equal to 36!"
        
        # Log the number of points
        rospy.loginfo("Number of points published: %d", len(pc_msg.points))
        
        # Publish the PointCloud message
        self.pc_pub.publish(pc_msg)
        rospy.loginfo("Published 2D representation to /velodyne_2d")

if __name__ == '__main__':
    try:
        processor = VelodyneProcessor()
    except rospy.ROSInterruptException:
        pass

