#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from ultralytics import YOLO

class YOLOv8Node:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('yolov8_node', anonymous=True)
        
        # Load YOLOv8 model
        self.model = YOLO('yolov8n.pt')
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Subscriber to the camera image topic
        self.image_sub = rospy.Subscriber('/final/camera/camera/image_raw', Image, self.image_callback)
        rospy.spin()
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return
        
        # Perform YOLOv8 inference
        results = self.model(cv_image)
        
        # Convert the detections into a format we can use to draw on the image
        inferred_image = self.draw_results(cv_image, results)
        
        # Display the image with detections
        cv2.imshow("YOLOv8 Detection", inferred_image)
        cv2.waitKey(1)  # Refresh the window
    
    def draw_results(self, image, results):
        # Iterate over detections and draw bounding boxes and labels
        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0]
            conf = result.conf[0]
            cls = result.cls[0]
            label = f"{self.model.names[int(cls)]}: {conf:.2f}"
            color = (0, 255, 0)  # Green color for bounding boxes
            
            # Draw rectangle
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Put label
            cv2.putText(image, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image

if __name__ == '__main__':
    try:
        YOLOv8Node()
    except rospy.ROSInterruptException:
        pass

