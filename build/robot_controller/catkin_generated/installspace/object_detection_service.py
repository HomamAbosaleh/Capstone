#!/usr/bin/env python2

from robot_controller.srv import DetectObjects
from robot_controller.msg import DetectedObject

from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
import rospy
import math

# Define the image callback function
def detect_objects(req):
    # Convert the ROS image message to a PIL Image
    np_arr = np.fromstring(req.image.data, np.uint8)
    pil_image = Image.fromarray(np_arr.reshape((req.image.height, req.image.width, 3)), 'RGB')

    # Perform inference
    results = model(pil_image)

    # Process the results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # class name
            cls = int(box.cls[0])
            class_name = classNames[cls]
            # Your existing code for processing the results goes here
            if class_name != "small chair":
               return DetectedObject(-1, -1, -1, -1, "NULL")
            print(box.xyxy[0])
            # box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            return DetectedObject(int(x1), int(y1), int(x2), int(y2), class_name)
    return DetectedObject(-1, -1, -1, -1, "NULL")

if __name__ == '__main__':
    classNames = ["turtlebot", "rosbot", "3D printer", "small chair", "big chair", "small table", "big table 1", "big table 2", "big table 3", "person", "big bin", "medium bin", "small bin"]

    # Initialize the ROS node
    rospy.init_node('object_detection_service')

    # Initialize the YOLO model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = YOLO("./yolo8s.pt")
    model.to(device)
    
    print("Model has been loaded")
    image_sub = rospy.Service('/detect_objects', DetectObjects, detect_objects)

    # Spin until the node is stopped
    rospy.spin()

