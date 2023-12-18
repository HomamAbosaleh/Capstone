#!/usr/bin/env python

from sensor_msgs.msg import Image
from robot_controller.msg import DetectedObject

from ultralytics import YOLO
import rospy
import cv2
import torch
import numpy as np
import math

classNames = ['turtlebot', 'rosbot', '3D printer', 'small chair', 'big chair', 'small table', 'big table 1', 'big table 2', 'big table 3', 'person', 'big bin', 'medium bin', 'small bin']

# Define the gstreamer pipeline
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=416,
    capture_height=416,
    display_width=416,
    display_height=416,
    framerate=30,
    flip_method=2,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def draw_frames(frame, x1, y1, x2, y2, class_name, confidence):
# put box in cam
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

    # draw the center of the object
    cv2.circle(frame, (int((x1+x2)/2), int((y1+y2)/2)), radius=5, color=(0, 0, 255), thickness=-1)
            
    # draw the center of the image
    cv2.circle(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)), radius=5, color=(0, 255, 0), thickness=-1)

            
    # Concatenate class name and confidence
    text = class_name + ' (' + confidence + ')'

    # object details
    org = [x1, y1-5]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255, 0, 0)
    thickness = 1

    cv2.putText(frame, text, org, font, fontScale, color, thickness)

def camera_publisher():
    # Initialize the node with rospy
    rospy.init_node('camera_publisher')

    # Create publisher. We publish to the /camera topic
    #publisher = rospy.Publisher('/camera', Image, queue_size=10)
    publisher = rospy.Publisher('/camera', DetectedObject, queue_size=10)     

    # Open the camera
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    while not rospy.is_shutdown():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Perform inference
            results = model(frame)
            # Convert the image to ROS Image message manually
            #ros_image = Image()
            #ros_image.height = frame.shape[0]
            #ros_image.width = frame.shape[1]
            #ros_image.encoding = "bgr8"
            #ros_image.is_bigendian = False
            #ros_image.step = ros_image.width * 3
            #ros_image.data = np.array(frame).tobytes()
            
            # Publish the image
            #publisher.publish(ros_image)
            # Process the results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # class name
                    cls = int(box.cls[0])
                    class_name = classNames[cls]
                    # Your existing code for processing the results goes here
                    # if class_name != "chair":
                    #     publisher.publish(DetectedObject(-1, -1, -1, -1, "NULL"))
                    #     continue

                    confidence = math.ceil((box.conf[0]*100))/100
                    if confidence < 0.70:
                        continue
                    # box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    publisher.publish(DetectedObject(int(x1), int(y1), int(x2), int(y2), class_name))
                    
                    draw_frames(frame, int(x1), int(y1), int(x2), int(y2), class_name, str(confidence))
            # Streaming the images
            publisher.publish(DetectedObject(-1, -1, -1, -1, "NULL"))
            cv2.imshow('Robot View', frame)

        k = cv2.waitKey(1) & 0xFF

        # If 'q' is pressed, break from the loop
        if k == ord('q'):
            break


    # After the loop, close the camera
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    # Initialize the YOLO model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = YOLO("./yolo8s.pt")
    model.to(device)

    camera_publisher()

