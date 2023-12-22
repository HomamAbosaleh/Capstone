import common
import colorsys
import random
import time
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import os

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

def normalize(img):
    img = np.asarray(img, dtype="float32")
    img = img / 127.5 - 1.0
    return img


def random_colors(N):
    N = N + 1
    hsv = [(i / N, 1.0, 1.0) for i in range(N)]
    colors = list(
        map(lambda c: tuple(int(i * 255) for i in colorsys.hsv_to_rgb(*c)), hsv)
    )
    random.shuffle(colors)
    return colors


def draw_rectangle(image, box, color, thickness=3):
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness)


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
    )
    cv2.putText(
        image, caption, (b[0], b[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1
    )

classNames = ["turtlebot", "rosbot", "3D printer", "chair", "table", "person"]

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

model_name = "yolo8s.engine"
score_threshold = 0.4

# Load the serialized engine
with open(model_name, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# Create an execution context for the engine
context = engine.create_execution_context()
inputs, outputs, bindings, stream = common.allocate_buffers(engine)

elapsed_list = []

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("VideoCapture read return false.")
        break

    random.seed(42)
    colors = random_colors(len(classNames))

    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_im = cv2.resize(im, (416, 416))
    normalized_im = normalize(resized_im)
    normalized_im = np.expand_dims(normalized_im, axis=0)

    # inference.
    start = time.perf_counter()
    inputs[0].host = normalized_im
    trt_outputs = common.do_inference_v2(
        context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
    )
    inference_time = (time.perf_counter() - start) * 10000

    boxs = trt_outputs[1].reshape([int(trt_outputs[0]), 4])
    for index, box in enumerate(boxs):
        if trt_outputs[2][index] < score_threshold:
            continue

        # Draw bounding box.
        class_id = int(trt_outputs[3][index])
        score = trt_outputs[2][index]
        caption = "{0}({1:.2f})".format(classNames[class_id - 1], score)

        xmin = int(box[0] * w)
        xmax = int(box[2] * w)
        ymin = int(box[1] * h)
        ymax = int(box[3] * h)
        draw_rectangle(frame, (xmin, ymin, xmax, ymax), colors[class_id])
        draw_caption(frame, (xmin, ymin - 10), caption)

    # Calc fps.
    elapsed_list.append(inference_time)
    avg_text = ""
    if len(elapsed_list) > 100:
        elapsed_list.pop(0)
        avg_elapsed_ms = np.mean(elapsed_list)
        avg_text = " AGV: {0:.2f}ms".format(avg_elapsed_ms)

    # Display fps
    fps_text = "Inference: {0:.2f}ms".format(inference_time)
    display_text = model_name + " " + fps_text + avg_text
    draw_caption(frame, (10, 30), display_text)

    # # Output video file
    # if video_writer is not None:
    #     video_writer.write(frame)

    # Display
    cv2.imshow("TensorRT detection example.", frame)

    k = cv2.waitKey(1) & 0xFF

    # If 'q' is pressed, break from the loop
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
