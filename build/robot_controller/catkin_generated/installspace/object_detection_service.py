#!/usr/bin/env python2

import rospy
from sensor_msgs.msg import Image
from robot_controller.msg import DetectedImage
from cv_bridge import CvBridge
import torch
from ultralytics import YOLO

# Define the image callback function
def detect_objects(msg):
    # Convert the ROS image message to an OpenCV image
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # Perform inference
    results = model(cv_image, stream=True)

    # Process the results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Your existing code for processing the results goes here
            if class_name == "chair":
               # box coordinates
               x1, y1, x2, y2 = box.xyxy[0]
               # class name
               cls = int(box.cls[0])
               class_name = classNames[cls]
               return DetectedImage(x1, y1, x2, y2, class_name)
    return None


if __name__ == '__main__':
    classNames = ["turtlebot", "rosbot", "3D printer", "chair", "table", "person"]

    # Initialize the ROS node
    rospy.init_node('object_detection_service')

    # Create a CvBridge object for converting between ROS and OpenCV images
    bridge = CvBridge()

    # Initialize the YOLO model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = YOLO("./yolo8s.pt")
    model.to(device)

    image_sub = rospy.Service('/detect_objects', Image, detect_objects)

    # Spin until the node is stopped
    rospy.spin()
