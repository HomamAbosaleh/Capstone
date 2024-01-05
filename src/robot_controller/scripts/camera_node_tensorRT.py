#!/usr/bin/env python

from robot_controller.msg import DetectedObject

from models import TRTModule  # isort:skip
from ultralytics import YOLO
import rospy
import cv2
import torch

from config import CLASSES
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox

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

    # object details
    org = [x1, y1-5]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255, 0, 0)
    thickness = 1

    cv2.putText(frame, f'{class_name} ({confidence:.3f})', org, font, fontScale, color, thickness)

def camera_publisher():
    # Initialize the node with rospy
    rospy.init_node('camera_publisher')

    # Initializing the engine
    device = torch.device("cuda:0")
    Engine = TRTModule("yolo8m.engine", device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    # Create publisher. We publish to the /camera topic
    publisher = rospy.Publisher('/camera', DetectedObject, queue_size=10)     

    # Open the camera
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            print("VideoCapture read return false.")
            break

        draw = frame.copy()
        frame, ratio, dwdh = letterbox(frame, (W, H))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.tensor(dwdh * 2, dtype=torch.float32, device=device)
        tensor = torch.tensor(tensor, device=device)
        # inference
        data = Engine(tensor)

        bboxes, scores, labels = det_postprocess(data)
        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            publisher.publish(DetectedObject(bbox[0], bbox[1], bbox[2], bbox[3], cls))
            # draw_frames(frame, bbox[0], bbox[1], bbox[2], bbox[3], cls, score)

        publisher.publish(DetectedObject(-1, -1, -1, -1, "NULL"))
        # cv2.imshow('result', draw)
        # k = cv2.waitKey(1) & 0xFF

        # # If 'q' is pressed, break from the loop
        # if k == ord('q'):
        #     break


    # After the loop, close the camera
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    camera_publisher()

