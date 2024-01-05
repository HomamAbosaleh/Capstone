import argparse
import jetson_utils
import torch
import cv2

from config import CLASSES, COLORS
from models import TRTModule
from models.torch_utils import det_postprocess
from models.utils import blob, letterbox

def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    Engine = TRTModule(args.engine, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    # Create a VideoSource object
    cap = jetson_utils.videoSource("csi://0", argv=['--input-flip=rotate-180', '--input-width=416', '--input-height=416'])

    while True:
        # Capture the frame
        frame = cap.Capture()

        # If no frame is captured, break the loop
        if not frame:
            print("VideoSource Capture return false.")
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
        if bboxes.numel() == 0:
            # if no bounding box
            cv2.imshow('result', frame)
            continue
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
