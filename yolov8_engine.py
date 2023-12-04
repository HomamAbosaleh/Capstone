# import numpy as np
# import rospy
# from sensor_msgs.msg import Imu
# from nav_msgs.msg import Odometry
# from geometry_msgs.msg import Twist
# from gazebo_msgs.msg import ModelStates

# class RobotController:
#     def __init__(self):
#         # Initialize the measurement subscribers
#         self.states_sub = rospy.Subscriber("/model_states", ModelStates, self.states_callback)
#         self.imu_sub = rospy.Subscriber("/imu", Imu, self.imu_callback)
#         self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)

#         # Initialize the control publisher
#         self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

#         # Initialize the motion model publisher
#         self.motion_model_pub = rospy.Publisher("/motion_model_output", Odometry, queue_size=10)

#         # Initialize the EKF publisher
#         self.ekf_pub = rospy.Publisher("/ekf_estimate", Odometry, queue_size=10)

#         # Initialize the state of the robot
#         """
#         x: x position
#         y: y position
#         theta: orientation
#         v: linear velocity
#         w: angular velocity
#         """
#         self.x = 0.0
#         self.y = -4.0
#         self.theta = 0.0
#         self.v = 0.0
#         self.w = 0.0

#         # Initialize the motion model noise
#         self.R = np.array([[0.01, 0, 0],
#                         [0, 0.01, 0],
#                         [0, 0, 0.01]])

#         self.Q = np.array([[0.01, 0],
#                         [0, 0.01]])
        
#         self.mu = np.array([self.x, self.y, self.theta])
#         self.Sigma = np.array([[0.01, 0, 0],
#                             [0, 0.01, 0],
#                             [0, 0, 0.01]])
    
#     def states_callback(self, msg):
#         # Extract the pose and twist from the message
#         pose = msg.pose[1]
#         twist = msg.twist[1]

#         # Update the state of the robot
#         self.x = pose.position.x
#         self.y = pose.position.y
#         self.theta = 2 * np.arctan2(pose.orientation.z, pose.orientation.w)  # Convert quaternion to euler
#         self.v = twist.linear.x
#         self.w = twist.angular.z
    
#     def imu_callback(self, msg):
#         # Extract the orientation from the message
#         orientation = msg.orientation

#         # Update the orientation of the robot
#         self.theta = 2 * np.arctan2(orientation.z, orientation.w)  # Convert quaternion to euler

#     def odom_callback(self, msg):
#         # Extract the pose and twist from the message
#         pose = msg.pose.pose
#         twist = msg.twist.twist

#         # Update the state of the robot
#         self.x = pose.position.x
#         self.y = pose.position.y
#         self.theta = 2 * np.arctan2(pose.orientation.z, pose.orientation.w)  # Convert quaternion to euler
#         self.v = twist.linear.x
#         self.w = twist.angular.z

#     def motion_model(self, v, w, dt):
#         """
#         Simulate the motion of the robot.

#         Parameters:
#         v: Linear velocity
#         w: Angular velocity
#         dt: Time step

#         Returns:
#         The new state of the robot.
#         """
#         # Update the orientation
#         self.theta += w * dt
#         self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))  # Normalize to [-pi, pi]

#         # Update the position
#         self.x += v * np.cos(self.theta) * dt
#         self.y += v * np.sin(self.theta) * dt

#         # Create an Odometry message
#         odom_msg = Odometry()
#         odom_msg.pose.pose.position.x = self.x
#         odom_msg.pose.pose.position.y = self.y
#         odom_msg.pose.pose.orientation.z = np.sin(self.theta / 2.0)
#         odom_msg.pose.pose.orientation.w = np.cos(self.theta / 2.0)

#         # Set the timestamp to the current time
#         odom_msg.header.stamp = rospy.Time.now()

#         # Publish the Odometry message
#         self.motion_model_pub.publish(odom_msg)

#         return self.x, self.y, self.theta
    
#     def ekf(self, mu, Sigma, u, z, x_obj, y_obj):
#         # Time step
#         dt = 0.1

#         # Motion model
#         v = u[0]
#         w = u[1]
#         theta = mu[2]
#         g = np.array([mu[0] + v * dt * np.cos(theta),
#                     mu[1] + v * dt * np.sin(theta),
#                     mu[2] + w * dt])

#         # Jacobian of the motion model
#         G = np.array([[1, 0, -v * dt * np.sin(theta)],
#                     [0, 1, v * dt * np.cos(theta)],
#                     [0, 0, 1]])

#         # Predicted state and covariance
#         mu_bar = g
#         Sigma_bar = np.dot(np.dot(G, Sigma), G.T)  + self.R # Add motion noise here

#         # Measurement model
#         dx = x_obj - mu_bar[0]
#         dy = y_obj - mu_bar[1]
#         h = np.array([np.sqrt(dx**2 + dy**2),
#                     np.arctan2(dy, dx) - mu_bar[2]])

#         # Jacobian of the measurement model
#         H = np.array([[-dx / np.sqrt(dx**2 + dy**2), -dy / np.sqrt(dx**2 + dy**2), 0],
#                     [dy / (dx**2 + dy**2), -dx / (dx**2 + dy**2), -1]])

#         # Kalman gain
#         S = np.dot(H, np.dot(Sigma_bar, H.T)) + self.Q # the innovation (or residual) covariance
#         K = np.dot(Sigma_bar, np.dot(H.T, np.linalg.inv(S)))

#         # Updated state and covariance
#         mu = mu_bar + np.dot(K, (z - h))
#         Sigma = np.dot((np.eye(3) - np.dot(K, H)), Sigma_bar)

#         # Create an Odometry message for the EKF estimate
#         ekf_msg = Odometry()
#         ekf_msg.pose.pose.position.x = mu[0]
#         ekf_msg.pose.pose.position.y = mu[1]
#         ekf_msg.pose.pose.orientation.z = np.sin(mu[2] / 2.0)
#         ekf_msg.pose.pose.orientation.w = np.cos(mu[2] / 2.0)

#         # Set the timestamp to the current time
#         ekf_msg.header.stamp = rospy.Time.now()

#         # Publish the EKF estimate
#         self.ekf_pub.publish(ekf_msg)

#         return mu, Sigma
    
#     def slam(self, u, z):
#         # Time step
#         dt = 0.1

#         # Motion model
#         v = u[0]
#         w = u[1]
#         theta = self.mu[2]
#         g = np.array([self.mu[0] + v * dt * np.cos(theta),
#                     self.mu[1] + v * dt * np.sin(theta),
#                     self.mu[2] + w * dt])

#         # Jacobian of the motion model
#         G = np.array([[1, 0, -v * dt * np.sin(theta)],
#                     [0, 1, v * dt * np.cos(theta)],
#                     [0, 0, 1]])

#         # Predicted state and covariance
#         mu_bar = g
#         Sigma_bar = np.dot(np.dot(G, self.Sigma), G.T)  + self.R # Add motion noise here

#         # Measurement model
#         for i in range(len(z)):
#             dx = z[i][0] - mu_bar[0]
#             dy = z[i][1] - mu_bar[1]
#             h = np.array([np.sqrt(dx**2 + dy**2),
#                         np.arctan2(dy, dx) - mu_bar[2]])

#             # Jacobian of the measurement model
#             H = np.array([[-dx / np.sqrt(dx**2 + dy**2), -dy / np.sqrt(dx**2 + dy**2), 0],
#                         [dy / (dx**2 + dy**2), -dx / (dx**2 + dy**2), -1]])

#             # Kalman gain
#             S = np.dot(H, np.dot(Sigma_bar, H.T)) + self.Q # the innovation (or residual) covariance
#             K = np.dot(Sigma_bar, np.dot(H.T, np.linalg.inv(S)))

#             # Updated state and covariance
#             self.mu = mu_bar + np.dot(K, (z[i] - h))
#             self.Sigma = np.dot((np.eye(3) - np.dot(K, H)), Sigma_bar)

#             # Update landmark estimates
#             m = np.array([self.mu[0] + z[i][0] * np.cos(z[i][1] + self.mu[2]),
#                         self.mu[1] + z[i][0] * np.sin(z[i][1] + self.mu[2])])

#             # Add the landmark to the state vector
#             self.mu = np.hstack((self.mu, m))

#             # Add the landmark to the covariance matrix
#             self.Sigma = np.vstack((self.Sigma, np.zeros((2, self.Sigma.shape[1]))))
#             self.Sigma = np.hstack((self.Sigma, np.zeros((self.Sigma.shape[0], 2))))
#             self.Sigma[-2:, -2:] = np.eye(2) * self.Q
    
#     def findDistanceBearing(self):
#         # Check if LIDAR data is available
#         if self.ranges is None:
#             return None, None

#         # Find the minimum range (closest point)
#         min_range = min(self.ranges)

#         # Find the index of the minimum range
#         min_index = self.ranges.index(min_range)

#         # Calculate the distance and bearing to the closest point
#         distance = min_range
#         bearing = min_index * 2 * np.pi / len(self.ranges)  # Assuming the LIDAR has a 360 degree field of view

#         return distance, bearing
    
#     def draw_a_circle(self):
#         # Create a new Twist message
#         vel_msg = Twist()

#         # Set the linear velocity (forward speed) to 1 m/s
#         vel_msg.linear.x = 0.6

#         # Set the angular velocity (turn speed) based on the desired radius of the circle
#         # Angular velocity is linear velocity divided by the radius
#         # For a circle of radius 5 meters, the angular velocity is 0.6 / 3.0 = 0.2 rad/s
#         vel_msg.angular.z = 0.2

#         # Publish the velocity message
#         self.cmd_pub.publish(vel_msg)

#     def run(self):
#         # Set the rate of the loop
#         rate = rospy.Rate(10)
        
#         while not rospy.is_shutdown():
#             # Move the robot in a circle
#             self.draw_a_circle()

#             # Call the motion model function
#             v = 0.6 # linear velocity
#             w = 0.2  # angular velocity
#             dt = 0.1  # time step (corresponding to the rate of 10Hz)
#             self.motion_model(v, w, dt)

#             # Call the measurement model function
#             distance, bearing = self.findDistanceBearing()

#             # Only call the EKF function if distance and bearing are not None
#             if distance is not None and bearing is not None:
#                 # Call the EKF function
#                 u = np.array([v, w])
#                 z = np.array([distance, bearing])
#                 x_obj = 0.0
#                 y_obj = 0.0
#                 self.mu, self.Sigma = self.ekf(self.mu, self.Sigma, u, z, x_obj, y_obj)

#                 # Call the SLAM function
#                 u = np.array([v, w])
#                 z = np.array([distance, bearing])
#                 self.slam(u, z)

#             # Sleep for the remainder of the loop
#             rate.sleep()


# if __name__ == "__main__":



#     # Initialize the node
#     rospy.init_node("robot_controller")

#     # Create an instance of the RobotController class
#     robot_controller = RobotController()

#     # Run the robot controller
#     robot_controller.run()


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

model_name = "best1.engine"
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
    inference_time = (time.perf_counter() - start) * 1000

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
