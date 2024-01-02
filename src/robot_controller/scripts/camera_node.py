#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from robot_controller.msg import DetectedObject
import cv2
from cv_bridge import CvBridge

class CameraNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/camera/image_raw", Image, queue_size=1)
        self.cap = cv2.VideoCapture((
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                0,
                416,
                416,
                30,
                2,
                416,
                416,
            )
        ), cv2.CAP_GSTREAMER)
        self.frame = None

    def main(self):
        ret, frame = self.cap.read()
        if ret:
            image_message = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.image_pub.publish(image_message)
            cv2.imshow('Robot View', frame)
        else:
            rospy.logerr("Unable to capture video frame")
        
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('camera_node')
    camera_node = CameraNode()
    camera_node.main()
