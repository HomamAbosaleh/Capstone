from models import TRTModule  # isort:skip
import argparse
from pathlib import Path

import cv2
import numpy as np
import rospy
import torch
from sensor_msgs.msg import Image as ROSImage
from robot_controller.msg import DetectedObject

from config import CLASSES
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox

class ImageSubscriber(object):
    def __init__(self, topic, publisher, args):
        self.device = torch.device(args.device)
        self.Engine = TRTModule(args.engine, self.device)
        # set desired output names order
        self.Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
        self.height, self.width = self.Engine.inp_info[0].shape[-2:]
        self.image_sub = rospy.Subscriber(topic, ROSImage, self.callback)
        self.publisher = publisher

    def callback(self, data):
        image_data = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        if image_data is not None:
            frame = image_data
            frame, ratio, dwdh = letterbox(frame, (self.width, self.height))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = blob(rgb, return_seg=False)
            dwdh = torch.tensor(dwdh * 2, dtype=torch.float32, device=self.device)
            tensor = torch.tensor(tensor, device=self.device)
            # inference
            data = self.Engine(tensor)

            bboxes, scores, labels = det_postprocess(data)
            if bboxes.numel() != 0:
                bboxes.sub_(dwdh)
                bboxes.div_(ratio)
                print("Entering the loop")
                for (bbox, score, label) in zip(bboxes, scores, labels):
                    bbox = bbox.round().int().tolist()
                    cls_id = int(label)
                    cls = CLASSES[cls_id]
                    self.publisher.publish(DetectedObject(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), cls))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, required=True, help='Engine file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    args = parser.parse_args()

    rospy.init_node('detection_publisher')

    
    publisher = rospy.Publisher('/detection', DetectedObject, queue_size=10)

    # subscribing to the image publisher
    image_subscriber = ImageSubscriber("/camera/image_raw", publisher, args)


    rospy.spin()


if __name__ == '__main__':
    main()

