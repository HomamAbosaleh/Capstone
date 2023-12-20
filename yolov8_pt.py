from ultralytics import YOLO
#import torch
import numpy as np
import math
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

def calculate_distance(w, h):
    # Coefficients
    a = 394.1
    b = -0.000153
    c = 242.1
    d = -1.367e-05
    
    return a * np.exp(b * w * h) + c * np.exp(d * w * h)

def calculate_bearning_angle(img_w, x1, x2):
   middle_point_x = (x2 + x1) / 2.0
   print("middle point ", middle_point_x)

   difference_object_image = (img_w/2.0) - middle_point_x
   px_in_meter = 1.12e-6 * (3280/img_w)  # Convert from micrometers to meters
   focal_in_meter = 2.96e-3  # Convert from millimeters to meters
   print("difference between object and image ", difference_object_image)

   atan_x = math.atan((difference_object_image * px_in_meter) / focal_in_meter)
   print("Arctan: ", atan_x)
   return math.degrees(atan_x)

   

classNames = ['turtlebot', 'rosbot', '3D printer', 'small chair', 'big chair', 'small table', 'big table 1', 'big table 2', 'big table 3', 'person', 'big bin', 'medium bin', 'small bin']

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

model = YOLO("./test.engine")

#model.to(device)

while(True):
    ret, org_frame = cap.read()

    # replicating the frame
    if not ret or org_frame is None:
       continue
    frame = org_frame.copy()
    
    # Perform inference
    results = model(frame, stream=True)

    for result in results:
        boxes = result.boxes

        for box in boxes:

            # class name
            cls = int(box.cls[0])
            class_name = classNames[cls]
            if class_name != 'small bin':
            	continue
            print("Class name -->", class_name)
            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            #if confidence < 0.50:
            #	continue
            
            confidence_str = str(confidence)  # Convert confidence to string
            #print("Confidence --->",confidence)
        
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(org_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # draw the center of the object
            cv2.circle(org_frame, (int((x1+x2)/2), int((y1+y2)/2)), radius=5, color=(0, 0, 255), thickness=-1)
            
            # draw the center of the image
            cv2.circle(org_frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)), radius=5, color=(0, 255, 0), thickness=-1)

            # corner coordinations
            #print(f"Coordinates top left ---> ({x1}, {y1})")
            #print(f"Coordinates bottom right ---> ({x2}, {y2})")
            # width and height
            print("Width --->",x2-x1)
            print("Height --->",y2-y1)
            #print("Distance --->", calculate_distance(x2-x1, y2-y1))
            #print("Bearing Angle --->", calculate_bearning_angle(frame.shape[1], x1, x2))

            
            # Concatenate class name and confidence
            text = class_name + ' (' + confidence_str + ')'

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(org_frame, text, org, font, fontScale, color, thickness)

    # Show the image with bounding boxes
    cv2.imshow('frame', org_frame)

    k = cv2.waitKey(1) & 0xFF

    # If 'q' is pressed, break from the loop
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

