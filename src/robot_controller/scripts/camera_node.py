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
        self.camera_sub = rospy.Subscriber("/detection", DetectedObject, self.detection_callback)
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

    def detection_callback(self, msg):
        if self.frame is not None:
            cv2.rectangle(self.frame, (msg.x1, msg.y1), (msg.x2, msg.y2), (255, 0, 255), 3)

            # draw the center of the object
            cv2.circle(self.frame, (int((msg.x1+msg.x2)/2), int((msg.y1+msg.y2)/2)), radius=5, color=(0, 0, 255), thickness=-1)
                    
            # draw the center of the image
            cv2.circle(self.frame, (int(self.frame.shape[1]/2), int(self.frame.shape[0]/2)), radius=5, color=(0, 255, 0), thickness=-1)

            # object details
            org = (msg.x1, msg.y1 - 5)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 1

            cv2.putText(self.frame, msg.class_name, org, font, fontScale, color, thickness)
            cv2.rectangle(self.frame, (msg.x1, msg.y1), (msg.x2, msg.y2), (0, 255, 0), 2)
            cv2.imshow('Robot View', self.frame)

        k = cv2.waitKey(1) & 0xFF

        # If 'q' is pressed, break from the loop
        if k == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()

    def main(self):
        ret, self.frame = self.cap.read()
        if ret:
            image_message = self.bridge.cv2_to_imgmsg(self.frame, encoding="bgr8")
            self.image_pub.publish(image_message)
        else:
            rospy.logerr("Unable to capture video frame")
        
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('camera_node', anonymous=True)
    camera_node = CameraNode()
    camera_node.main()
