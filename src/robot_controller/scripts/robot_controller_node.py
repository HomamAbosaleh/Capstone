#! /usr/bin/env python

from sensor_msgs.msg import Imu, Image
from nav_msgs.msg import Odometry
from robot_controller.msg import DetectedObject
from geometry_msgs.msg import Twist

from robot_controller.srv import DetectObjects

import numpy as np
import rospy
import math


# Defining Constants
IMG_WIDTH = 416
IMG_HEIGHT = 416

CLASS_EQUATIONS = {
    'small chair': lambda x1, x2, y1, y2: 549 * np.exp(-0.0001644 * (x2 - x1) * (y2 - y1)) + 238.6 * np.exp(-1.342e-05 * (x2 - x1) * (y2 - y1)),
    'big bin': lambda x1, x2, y1, y2: 510.4 * np.exp(-6.066e-05 * (x2 - x1) * (y2 - y1)) + 75.81 * np.exp(3.734e-06 * (x2 - x1) * (y2 - y1)),
    'medium bin': lambda x1, x2, y1, y2: 438.8 * np.exp(-0.0002002 * (x2 - x1) * (y2 - y1)) + 239.7 * np.exp(-1.652e-05 * (x2 - x1) * (y2 - y1)),
    'small bin': lambda x1, x2, y1, y2: 478.2 * np.exp(-0.000597 * (x2 - x1) * (y2 - y1)) + 254 * np.exp(-4.499e-05 * (x2 - x1) * (y2 - y1)),
}


class RobotController:
    def __init__(self):

        # Initialize the camera subscriber
        self.camera_sub = rospy.Subscriber("/camera", DetectedObject, self.camera_callback)    

        # Initialize the measurement subscribers
        self.imu_sub = rospy.Subscriber("/imu", Imu, self.imu_callback)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)

        # Initialize the control publisher
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Initialize the motion model publisher
        self.motion_model_pub = rospy.Publisher("/motion_model_output", Odometry, queue_size=10)

        # Initialize the EKF publisher
        self.ekf_pub = rospy.Publisher("/ekf_estimate", Odometry, queue_size=10)

        # Initializing distance and bearing
        self.distance = None
        self.bearing = None

        # Initialize the state of the robot
        """
        x: x position
        y: y position
        theta: orientation
        v: linear velocity
        w: angular velocity
        """
        self.x = 0.0
        self.y = -4.0
        self.theta = 0.0
        self.v = 0.0
        self.w = 0.0

        # Initialize the motion model noise
        self.R = np.array([[0.01, 0, 0],
                        [0, 0.01, 0],
                        [0, 0, 0.01]])

        self.Q = np.array([[0.01, 0],
                        [0, 0.01]])
        
        self.mu = np.array([self.x, self.y, self.theta])
        self.Sigma = np.array([[0.01, 0, 0],
                            [0, 0.01, 0],
                            [0, 0, 0.01]])
    
    def imu_callback(self, msg):
        # Extract the orientation from the message
        orientation = msg.orientation

        # Update the orientation of the robot
        self.theta = 2 * np.arctan2(orientation.z, orientation.w)  # Convert quaternion to euler

    def odom_callback(self, msg):
        # Extract the pose and twist from the message
        pose = msg.pose.pose
        twist = msg.twist.twist

        # Update the state of the robot
        self.x = pose.position.x
        self.y = pose.position.y
        self.theta = 2 * np.arctan2(pose.orientation.z, pose.orientation.w)  # Convert quaternion to euler
        self.v = twist.linear.x
        self.w = twist.angular.z

    def camera_callback(self, msg):
         try:
            # Create a service proxy
            #object_detection_service = rospy.ServiceProxy('/detect_objects', DetectObjects)
            #self.detected_object = object_detection_service(msg).object
            if msg.class_name != "NULL":
                # Call the measurement model function
                self.distance, self.bearing = self.calculate_distance_and_angle(msg.x1, msg.x2, msg.y1, msg.y2, msg.class_name)
         except rospy.ServiceException as e:
            print("Service call failed: %s"%e)
       
    def calculate_distance_and_angle(self, x1, x2, y1, y2, class_name):
        # Calculate the distance to the object
        distance_in_meters = CLASS_EQUATIONS[class_name](x1, x2, y1, y2)
        
        # Calculate the bearing angle to the object
        middle_point_x = (x2 + x1) / 2.0

        difference_object_image = (IMG_WIDTH/2.0) - middle_point_x
        px_in_meter = 1.12e-6 * (3280/IMG_WIDTH)  # Convert from micrometers to meters
        focal_in_meter = 2.96e-3  # Convert from millimeters to meters

        angle_in_radians = math.atan((difference_object_image * px_in_meter) / focal_in_meter)
        print("Distance in centimeters: ", distance_in_meters)
        print("Bearning angle in radians: ", angle_in_radians)

        return distance_in_meters, angle_in_radians

    def motion_model(self, v, w, dt):
        """
        Simulate the motion of the robot.

        Parameters:
        v: Linear velocity
        w: Angular velocity
        dt: Time step

        Returns:
        The new state of the robot.
        """
        # Update the orientation
        self.theta += w * dt
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))  # Normalize to [-pi, pi]

        # Update the position
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt

        # Create an Odometry message
        odom_msg = Odometry()
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.orientation.z = np.sin(self.theta / 2.0)
        odom_msg.pose.pose.orientation.w = np.cos(self.theta / 2.0)

        # Set the timestamp to the current time
        odom_msg.header.stamp = rospy.Time.now()

        # Publish the Odometry message
        self.motion_model_pub.publish(odom_msg)

        return self.x, self.y, self.theta
    
    def ekf(self, mu, Sigma, u, z, x_obj, y_obj):
        # Time step
        dt = 0.1

        # Motion model
        v = u[0]
        w = u[1]
        theta = mu[2]
        g = np.array([mu[0] + v * dt * np.cos(theta),
                    mu[1] + v * dt * np.sin(theta),
                    mu[2] + w * dt])

        # Jacobian of the motion model
        G = np.array([[1, 0, -v * dt * np.sin(theta)],
                    [0, 1, v * dt * np.cos(theta)],
                    [0, 0, 1]])

        # Predicted state and covariance
        mu_bar = g
        Sigma_bar = np.dot(np.dot(G, Sigma), G.T)  + self.R # Add motion noise here

        # Measurement model
        dx = x_obj - mu_bar[0]
        dy = y_obj - mu_bar[1]
        h = np.array([np.sqrt(dx**2 + dy**2),
                    np.arctan2(dy, dx) - mu_bar[2]])

        # Jacobian of the measurement model
        H = np.array([[-dx / np.sqrt(dx**2 + dy**2), -dy / np.sqrt(dx**2 + dy**2), 0],
                    [dy / (dx**2 + dy**2), -dx / (dx**2 + dy**2), -1]])

        # Kalman gain
        S = np.dot(H, np.dot(Sigma_bar, H.T)) + self.Q # the innovation (or residual) covariance
        K = np.dot(Sigma_bar, np.dot(H.T, np.linalg.inv(S)))

        # Updated state and covariance
        mu = mu_bar + np.dot(K, (z - h))
        Sigma = np.dot((np.eye(3) - np.dot(K, H)), Sigma_bar)

        # Create an Odometry message for the EKF estimate
        ekf_msg = Odometry()
        ekf_msg.pose.pose.position.x = mu[0]
        ekf_msg.pose.pose.position.y = mu[1]
        ekf_msg.pose.pose.orientation.z = np.sin(mu[2] / 2.0)
        ekf_msg.pose.pose.orientation.w = np.cos(mu[2] / 2.0)

        # Set the timestamp to the current time
        ekf_msg.header.stamp = rospy.Time.now()

        # Publish the EKF estimate
        self.ekf_pub.publish(ekf_msg)

        return mu, Sigma
       
    def slam(self, u, z):
        # Time step
        dt = 0.1

        # Motion model
        v = u[0]
        w = u[1]
        theta = self.mu[2]
        g = np.array([self.mu[0] + v * dt * np.cos(theta),
                    self.mu[1] + v * dt * np.sin(theta),
                    self.mu[2] + w * dt])

        # Jacobian of the motion model
        G = np.array([[1, 0, -v * dt * np.sin(theta)],
                    [0, 1, v * dt * np.cos(theta)],
                    [0, 0, 1]])

        # Predicted state and covariance
        mu_bar = g
        Sigma_bar = np.dot(np.dot(G, self.Sigma), G.T)  + self.R # Add motion noise here

        # Measurement model
        for i in range(len(z)):
            dx = z[i][0] - mu_bar[0]
            dy = z[i][1] - mu_bar[1]
            h = np.array([np.sqrt(dx**2 + dy**2),
                        np.arctan2(dy, dx) - mu_bar[2]])

            # Jacobian of the measurement model
            H = np.array([[-dx / np.sqrt(dx**2 + dy**2), -dy / np.sqrt(dx**2 + dy**2), 0],
                        [dy / (dx**2 + dy**2), -dx / (dx**2 + dy**2), -1]])

            # Kalman gain
            S = np.dot(H, np.dot(Sigma_bar, H.T)) + self.Q # the innovation (or residual) covariance
            K = np.dot(Sigma_bar, np.dot(H.T, np.linalg.inv(S)))

            # Updated state and covariance
            self.mu = mu_bar + np.dot(K, (z[i] - h))
            self.Sigma = np.dot((np.eye(3) - np.dot(K, H)), Sigma_bar)

            # Update landmark estimates
            m = np.array([self.mu[0] + z[i][0] * np.cos(z[i][1] + self.mu[2]),
                        self.mu[1] + z[i][0] * np.sin(z[i][1] + self.mu[2])])

            # Add the landmark to the state vector
            self.mu = np.hstack((self.mu, m))

            # Add the landmark to the covariance matrix
            self.Sigma = np.vstack((self.Sigma, np.zeros((2, self.Sigma.shape[1]))))
            self.Sigma = np.hstack((self.Sigma, np.zeros((self.Sigma.shape[0], 2))))
            self.Sigma[-2:, -2:] = np.eye(2) * self.Q
    
    def draw_a_circle(self):
        # Create a new Twist message
        vel_msg = Twist()

        # Set the linear velocity (forward speed) to 1 m/s
        vel_msg.linear.x = 0.1

        # Set the angular velocity (turn speed) based on the desired radius of the circle
        # Angular velocity is linear velocity divided by the radius
        # For a circle of radius 1 meters, the angular velocity is 0.1 / 1 = 0.1 rad/s
        vel_msg.angular.z = 0.1

        # Publish the velocity message
        self.cmd_pub.publish(vel_msg)

    def run(self):
        # Set the rate of the loop
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            # Move the robot in a circle
            self.draw_a_circle()

            # Call the motion model function
            v = 0.1 # linear velocity
            w = 0.1  # angular velocity
            dt = 0.1  # time step (corresponding to the rate of 10Hz)
            self.motion_model(v, w, dt)

            # Only call the EKF function if distance and bearing are not None
            if self.distance is not None and self.bearing is not None:
                # Call the EKF function
                u = np.array([v, w])
                z = np.array([self.distance, self.bearing])
                x_obj = 0.0
                y_obj = 0.0
                self.mu, self.Sigma = self.ekf(self.mu, self.Sigma, u, z, x_obj, y_obj)
                

            # Sleep for the remainder of the loop
            rate.sleep()


if __name__ == "__main__":
    # Initialize the node
    rospy.init_node("robot_controller_node")

    # Ensuring that the service is running
    #rospy.wait_for_service('/detect_objects')

    # Create an instance of the RobotController class
    robot_controller = RobotController()

    # Run the robot controller
    robot_controller.run()








   

