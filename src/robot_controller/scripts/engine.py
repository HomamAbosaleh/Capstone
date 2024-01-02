from models import TRTModule  # isort:skip
import argparse
from pathlib import Path

import cv2
import torch

from config import CLASSES, COLORS
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox


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

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    while cap.isOpened():
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
        #if bboxes.numel() == 0:
            # if no bounding box
        #    cv2.imshow('result', frame)
        #    continue
        bboxes -= dwdh
        bboxes /= ratio

        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().int().tolist()
            cls_id = int(label)
            cls = CLASSES[cls_id]
            color = COLORS[cls]
            cv2.rectangle(draw, bbox[:2], bbox[2:], color, 2)
            cv2.putText(draw,
                        f'{cls}:{score:.3f}', (bbox[0], bbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, [225, 255, 255],
                        thickness=2)

        cv2.imshow('result', draw)
        k = cv2.waitKey(1) & 0xFF

        # If 'q' is pressed, break from the loop
        if k == ord('q'):
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, required=True, help='Engine file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
