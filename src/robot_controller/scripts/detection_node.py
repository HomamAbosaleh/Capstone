#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from robot_controller.msg import DetectedObject
import cv2
from cv_bridge import CvBridge

from models import TRTModule
from config import CLASSES
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox

class DetectionNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.detection_pub = rospy.Publisher("/detection", DetectedObject, queue_size=10)
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.image_callback)
        # Initialize your object detection model here
        self.device = torch.device('cuda:0')
        self.Engine = TRTModule('engine_file', self.device)
        self.H, self.W = self.Engine.inp_info[0].shape[-2:]
        self.Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    def image_callback(self, msg):
        detected = False
        # Convert the image message to an OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        draw = cv_image.copy()
        frame, ratio, dwdh = letterbox(cv_image, (self.W, self.H))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.tensor(dwdh * 2, dtype=torch.float32, device=self.device)
        tensor = torch.tensor(tensor, device=self.device)
        # inference
        data = self.Engine(tensor)
        bboxes, scores, labels = det_postprocess(data)
        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, label) in zip(bboxes, scores, labels):
            detected = True
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            # Publish the detected objects
            # You need to replace this with your actual detection code
            detected_object = DetectedObject()
            detected_object.x1 = bbox[0]
            detected_object.y1 = bbox[1]
            detected_object.x2 = bbox[2]
            detected_object.y2 = bbox[3]
            detected_object.class_name = cls
            self.detection_pub.publish(detected_object)

        if not detected:
            detected_object = DetectedObject()
            detected_object.x1 = -1
            detected_object.y1 = -1
            detected_object.x2 = -1
            detected_object.y2 = -1
            detected_object.class_name = "NULL"
            self.detection_pub.publish(detected_object)
        

if __name__ == '__main__':
    rospy.init_node('detection_node')
    detection_node = DetectionNode()
    rospy.spin()
