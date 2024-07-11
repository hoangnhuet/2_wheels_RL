#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import supervision as sv

class YOLOVisualizer:
    def __init__(self):
        rospy.init_node('yolo_visualizer', anonymous=True)
        self.bridge = CvBridge()
        self.yolo_model = YOLO('yolov8n.pt')
        self.image_sub = rospy.Subscriber('/r1/front_camera/image_raw', Image, self.callback)
        self.image_pub = rospy.Publisher('/yolo_visualizer/annotated_image', Image, queue_size=10)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return
        
        # Perform inference using YOLO model
        results = self.yolo_model(cv_image)[0]
        print(results[0].boxes.xyxy)
        # Convert results to detections
        detections = sv.Detections.from_ultralytics(results)

        # Annotate bounding boxes and labels
        bounding_box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        annotated_image = bounding_box_annotator.annotate(scene=cv_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        # Convert the annotated image to a ROS message
        annotated_image_msg = self.bridge.cv2_to_imgmsg(annotated_image, 'bgr8')

        # Publish the annotated image
        self.image_pub.publish(annotated_image_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    visualizer = YOLOVisualizer()
    visualizer.run()

