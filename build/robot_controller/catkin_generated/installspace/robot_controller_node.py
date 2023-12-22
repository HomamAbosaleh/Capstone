#!/usr/bin/env python2

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from robot_controller.msg import DetectedObject
from geometry_msgs.msg import Twist

from robot_controller.srv import DetectObjects

import numpy as np
import rospy
import math

import pandas as pd
import os

# Defining Constants
IMG_WIDTH = 416
IMG_HEIGHT = 416

CLASS_EQUATIONS = {
    'small chair': lambda x1, x2, y1, y2: 549 * np.exp(-0.0001644 * (x2 - x1) * (y2 - y1)) + 238.6 * np.exp(-1.342e-05 * (x2 - x1) * (y2 - y1)),
    'big bin': lambda x1, x2, y1, y2: 510.4 * np.exp(-6.066e-05 * (x2 - x1) * (y2 - y1)) + 75.81 * np.exp(3.734e-06 * (x2 - x1) * (y2 - y1)),
    'medium bin': lambda x1, x2, y1, y2: 438.8 * np.exp(-0.0002002 * (x2 - x1) * (y2 - y1)) + 239.7 * np.exp(-1.652e-05 * (x2 - x1) * (y2 - y1)),
    'small bin': lambda x1, x2, y1, y2: 478.2 * np.exp(-0.000597 * (x2 - x1) * (y2 - y1)) + 254 * np.exp(-4.499e-05 * (x2 - x1) * (y2 - y1)),
}

class Landmark:
    def __init__(self, r, phi, signature, mu, sigma):
        self.r = r
        self.phi = phi
        self.signature = signature
        self.mu = mu
        self.sigma = sigma


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
        # self.ekf_pub = rospy.Publisher("/ekf_estimate", Odometry, queue_size=10)

        # Initialize the state of the robot
        """
        x: x position
        y: y position
        theta: orientation
        v: linear velocity
        w: angular velocity
        """
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        # self.v = 0.0
        # self.w = 0.0

        self.imu_x = 0.0
        self.imu_y = 0.0
        self.imu_theta = 0.0

        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_theta = 0.0

        # Initialize the motion model noise
        self.R = np.array([[0.01, 0, 0],
                        [0, 0.01, 0],
                        [0, 0, 0.01]])
        
        self.mu = np.array([self.x, self.y, self.theta])
        self.Sigma = np.array([[0.01, 0, 0],
                            [0, 0.01, 0],
                            [0, 0, 0.01]])
        
        # Initialize the landmark estimates
        self.landmarks = []

        # Initialize the extended mu and Sigma
        self.mu_extended = np.array([self.x, self.y, self.theta])
        self.Sigma_extended = np.array([[0.01, 0, 0],
                            [0, 0.01, 0],
                            [0, 0, 0.01]])

    
    def imu_callback(self, msg):
        # Extract the orientation from the message
        orientation = msg.orientation

        # Update the orientation of the robot
        self.imu_x = orientation.x
        self.imu_y = orientation.y
        self.imu_theta = 2 * np.arctan2(orientation.z, orientation.w)  # Convert quaternion to euler

    def odom_callback(self, msg):
        # Extract the pose and twist from the message
        pose = msg.pose.pose
        twist = msg.twist.twist

        # Update the state of the robot
        self.odom_x = pose.position.x
        self.odom_y = pose.position.y
        self.odom_theta = 2 * np.arctan2(pose.orientation.z, pose.orientation.w)  # Convert quaternion to euler
        # self.v = twist.linear.x
        # self.w = twist.angular.z

    def camera_callback(self, msg):
        if msg.class_name != "NULL":
            # Call the measurement function based on the class name
            if msg.class_name == "big bin" or msg.class_name == "medium bin" or msg.class_name == "small bin":
                distance, bearing = self.calculate_distance_and_angle(msg.x1, msg.x2, msg.y1, msg.y2, msg.class_name)
                landmark = next((landmark for landmark in self.landmarks if landmark.signature == msg.class_name), None)
                if(landmark is None):
                    self.landmarks.append(Landmark(distance, bearing, msg.class_name, np.array([self.x + distance * np.cos(bearing + self.theta),
                                        self.y + distance * np.sin(bearing + self.theta)]), np.array([[0.01, 0],
                                            [0, 0.01]])))
                else:
                    landmark.r = distance
                    landmark.phi = bearing
       
    def calculate_distance_and_angle(self, x1, x2, y1, y2, class_name):
        """
        calculate the distance and angle to the object.

        Parameters:
            x1: x coordinate of the top left corner of the bounding box
            x2: x coordinate of the bottom right corner of the bounding box
            y1: y coordinate of the top left corner of the bounding box
            y2: y coordinate of the bottom right corner of the bounding box
            class_name: name of the class of the object
        
        Returns:
            distance_in_meters: distance to the object in meters
            angle_in_radians: bearing angle to the object in radians
        """
        # Calculate the distance to the object
        distance_in_meters = CLASS_EQUATIONS[class_name](x1, x2, y1, y2)
        
        # Calculate the bearing angle to the object
        middle_point_x = (x2 + x1) / 2.0

        difference_object_image = (IMG_WIDTH/2.0) - middle_point_x
        px_in_meter = 1.12e-6 * (3280/IMG_WIDTH)  # Convert from micrometers to meters
        focal_in_meter = 2.96e-3  # Convert from millimeters to meters

        angle_in_radians = math.atan((difference_object_image * px_in_meter) / focal_in_meter)
        #print("Class name: ", class_name)
        #print("Distance in centimeters: ", distance_in_meters)
        #print("Bearning angle in radians: ", angle_in_radians)
        #print("==================================================================================")

        return distance_in_meters, angle_in_radians

    def motion_model(self, v, w, dt):
        """
        Simulate the motion of the robot.

        Parameters:
            v: Linear velocity
            w: Angular velocity
            dt: Time step

        Updates:
            x: x position
            y: y position
            theta: orientation
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
    
    # def ekf_predict(self, u, dt, mu, Sigma, R):
    #     # Motion model
    #     v = u[0]
    #     w = u[1]
    #     theta = mu[2]
    #     g = np.array([mu[0] + v * dt * np.cos(theta),
    #                 mu[1] + v * dt * np.sin(theta),
    #                 mu[2] + w * dt])

    #     # Jacobian of the motion model
    #     G = np.array([[1, 0, -v * dt * np.sin(theta)],
    #                 [0, 1, v * dt * np.cos(theta)],
    #                 [0, 0, 1]])

    #     # Predicted state and covariance
    #     mu_bar = g
    #     Sigma_bar = np.dot(np.dot(G, Sigma), G.T)  + R # Add motion noise here

    #     return mu_bar, Sigma_bar
    
    # def ekf_update(self, mu_bar, Sigma_bar, z):
    #     # Measurement model
    #     dx = z[0] - mu_bar[0]
    #     dy = z[1] - mu_bar[1]
    #     h = np.array([np.sqrt(dx**2 + dy**2),
    #                 np.arctan2(dy, dx) - mu_bar[2]])

    #     # Jacobian of the measurement model
    #     H = np.array([[-dx / np.sqrt(dx**2 + dy**2), -dy / np.sqrt(dx**2 + dy**2), 0],
    #                 [dy / (dx**2 + dy**2), -dx / (dx**2 + dy**2), -1]])

    #     # Kalman gain
    #     S = np.dot(H, np.dot(Sigma_bar, H.T)) + self.Q
    #     K = np.dot(Sigma_bar, np.dot(H.T, np.linalg.inv(S)))

    #     # Updated state and covariance
    #     mu = mu_bar + np.dot(K, (z - h))
    #     Sigma = np.dot((np.eye(3) - np.dot(K, H)), Sigma_bar)

    #     return mu, Sigma
    
    # def ekf(self, mu, Sigma, u, z, dt, R, Q, x_obj, y_obj) -> tuple[float, float]:
    #     """
    #     ekf: Extended Kalman Filter

    #     Parameters:
    #         mu: mean of the state
    #         Sigma: covariance of the state
    #         u: control input
    #         z: measurement
    #         dt: time step
    #         R: motion noise
    #         Q: measurement noise
    #         x_obj: x position of the object
    #         y_obj: y position of the object
        
    #     Returns:
    #         mu: new mean of the state
    #         Sigma: new covariance of the state
    #     """
    #     # Motion model
    #     v = u[0]
    #     w = u[1]
    #     theta = mu[2]
    #     g = np.array([mu[0] + v * dt * np.cos(theta),
    #                 mu[1] + v * dt * np.sin(theta),
    #                 mu[2] + w * dt])

    #     # Jacobian of the motion model
    #     G = np.array([[1, 0, -v * dt * np.sin(theta)],
    #                 [0, 1, v * dt * np.cos(theta)],
    #                 [0, 0, 1]])

    #     # Predicted state and covariance
    #     mu_bar = g
    #     Sigma_bar = np.dot(np.dot(G, Sigma), G.T)  + R # Add motion noise here

    #     # Measurement model
    #     dx = x_obj - mu_bar[0]
    #     dy = y_obj - mu_bar[1]
    #     h = np.array([np.sqrt(dx**2 + dy**2),
    #                 np.arctan2(dy, dx) - mu_bar[2]])

    #     # Jacobian of the measurement model
    #     H = np.array([[-dx / np.sqrt(dx**2 + dy**2), -dy / np.sqrt(dx**2 + dy**2), 0],
    #                 [dy / (dx**2 + dy**2), -dx / (dx**2 + dy**2), -1]])

    #     # Kalman gain
    #     S = np.dot(H, np.dot(Sigma_bar, H.T)) + Q # the innovation (or residual) covariance
    #     K = np.dot(Sigma_bar, np.dot(H.T, np.linalg.inv(S)))

    #     # Updated state and covariance
    #     mu = mu_bar + np.dot(K, (z - h))
    #     Sigma = np.dot((np.eye(3) - np.dot(K, H)), Sigma_bar)

    #     # # Create an Odometry message for the EKF estimate
    #     # ekf_msg = Odometry()
    #     # ekf_msg.pose.pose.position.x = mu[0]
    #     # ekf_msg.pose.pose.position.y = mu[1]
    #     # ekf_msg.pose.pose.orientation.z = np.sin(mu[2] / 2.0)
    #     # ekf_msg.pose.pose.orientation.w = np.cos(mu[2] / 2.0)

    #     # # Set the timestamp to the current time
    #     # ekf_msg.header.stamp = rospy.Time.now()

    #     # # Publish the EKF estimate
    #     # self.ekf_pub.publish(ekf_msg)

    #     return mu, Sigma
    
    def EKF_SLAM(self, mu, Sigma, u, z, R, dt): # http://ais.informatik.uni-freiburg.de/teaching/ws15/mapping/pdf/slam05-ekf-slam.pdf
        """
        Parameters:
            mu: mean of the state
            Sigma: covariance of the state
            u: control input
            z: measurements of all the landmarks
            R: motion noise
        """

        F = np.eye(3, len(mu))

        # Calculate motion model
        v = u[0]
        w = u[1]
        theta = mu[2]
        g = mu + np.dot(F.T, np.array([v * dt * np.cos(theta),
                v * dt * np.sin(theta),
                w * dt]))
        
        # Jacobian of the motion model
        G = np.eye(len(mu), len(mu)) + np.dot(np.dot(F.T, np.array([[0, 0, -v * dt * np.sin(theta)],
                [0, 0, v * dt * np.cos(theta)],
                [0, 0, 0]])), F)
        
        # Predicted state and covariance
        mu_bar = g
        Sigma_bar = np.dot(np.dot(G, Sigma), G.T) + np.dot(np.dot(F.T, R), F) # Add motion noise

        # Measurement model
        Q = np.dot(np.eye(2), 0.01) # measurement noise

        for j in range(len(z)):

            # if c[j] == -1: # if the landmark is not in the state vector
            #     # Add the landmark to the state vector
            #     self.landmarks.append(Landmark(z[j][0], z[j][1], z[j][2], mu_bar[0] + z[j][0] * np.cos(z[j][1] + mu_bar[2]), mu_bar[1] + z[j][0] * np.sin(z[j][1] + mu_bar[2])))

            #     # Add the landmark to the covariance matrix
            #     Sigma_bar = np.vstack((Sigma_bar, np.zeros((2, Sigma_bar.shape[1]))))
            #     Sigma_bar = np.hstack((Sigma_bar, np.zeros((Sigma_bar.shape[0], 2))))
            #     Sigma_bar[-2:, -2:] = np.eye(2) * Q

            #     # Update the landmark index
            #     c[j] = len(self.landmarks) - 1

            delta = np.array([mu_bar[3 + 2*j] - mu_bar[0], mu_bar[3 + 2*j + 1] - mu_bar[1]])
            q = np.dot(delta.T, delta)
            z_hat = np.array([np.sqrt(q),
                            np.arctan2(delta[1], delta[0]) - mu_bar[2]]) # predicted measurement h

            # construct Fxj
            #! Create the blocks
            # identity_3x3 = np.eye(3)
            # zeros_2x3 = np.zeros((2, 3))
            # zeros_3x2j_2 = np.zeros((3, 2*(j + 1) - 2))
            # zeros_2x2j_2 = np.zeros((2, 2*(j + 1) - 2))
            # zeros_3x3 = np.zeros((3, 3))
            # identity_2x2 = np.eye(2)
            # zeros_3x2N_2j = np.zeros((3, 2*len(z) - 2*(j + 1)))
            # zeros_2x2N_2j = np.zeros((2, 2*len(z) - 2*(j + 1)))

            # #! Create the matrix
            # Fxj = np.block([[identity_3x3, zeros_3x2j_2, zeros_3x3, zeros_3x2N_2j],
            #                 [zeros_2x3, zeros_2x2j_2, identity_2x2, zeros_2x2N_2j]])
            #! Create the top part
            identity_3xN = np.eye(3, 3 + 2*(j + 1) - 2 + 2 + 2*len(z) - 2*(j + 1))

            #! Create the bottom part
            zeros_2xN = np.zeros((2, 3 + 2*(j + 1) - 2)) # it is 3 + N
            identity_2x2 = np.eye(2)
            zeros_2x2N = np.zeros((2, 2*len(z) - 2*(j + 1)))
            bottom_part = np.hstack([zeros_2xN, identity_2x2, zeros_2x2N])

            #! Create the matrix
            Fxj = np.vstack([identity_3xN, bottom_part])

            # Jacobian of the measurement model
            H = (1 / q) * np.dot(np.array([[- np.sqrt(q) * delta[0], - np.sqrt(q) * delta[1], 0, np.sqrt(q) * delta[0], np.sqrt(q) * delta[1]],
                    [delta[1], - delta[0], - q, - delta[1], delta[0]]]), Fxj)

            # Kalman gain
            K = np.dot(np.dot(Sigma_bar, H.T), np.linalg.inv(np.dot(np.dot(H, Sigma_bar), H.T) + Q))
            mu_bar = mu_bar + np.dot(K, (z[j] - z_hat))
            Sigma_bar = np.dot((np.eye(len(mu_bar)) - np.dot(K, H)), Sigma_bar)

            self.landmarks[j].mu = mu_bar[3 + 2*j:3 + 2*j + 2]
            self.landmarks[j].sigma = Sigma_bar[3 + 2*j:3 + 2*j + 2, 3 + 2*j:3 + 2*j + 2]

        self.mu = mu_bar[0:3]
        self.Sigma = Sigma_bar[0:3, 0:3]
        return mu_bar, Sigma_bar
        



    # def slam(self, u, z, dt):
    #     # Motion model
    #     v = u[0]
    #     w = u[1]
    #     theta = self.mu[2]
    #     g = np.array([self.mu[0] + v * dt * np.cos(theta),
    #                 self.mu[1] + v * dt * np.sin(theta),
    #                 self.mu[2] + w * dt])

    #     # Jacobian of the motion model
    #     G = np.array([[1, 0, -v * dt * np.sin(theta)],
    #                 [0, 1, v * dt * np.cos(theta)],
    #                 [0, 0, 1]])

    #     # Predicted state and covariance
    #     mu_bar = g
    #     Sigma_bar = np.dot(np.dot(G, self.Sigma), G.T)  + self.R # Add motion noise here

    #     # Measurement model
    #     for i in range(len(z)):
    #         dx = z[i][0] - mu_bar[0]
    #         dy = z[i][1] - mu_bar[1]
    #         h = np.array([np.sqrt(dx**2 + dy**2),
    #                     np.arctan2(dy, dx) - mu_bar[2]])

    #         # Jacobian of the measurement model
    #         H = np.array([[-dx / np.sqrt(dx**2 + dy**2), -dy / np.sqrt(dx**2 + dy**2), 0],
    #                     [dy / (dx**2 + dy**2), -dx / (dx**2 + dy**2), -1]])

    #         # Kalman gain
    #         S = np.dot(H, np.dot(Sigma_bar, H.T)) + self.Q # the innovation (or residual) covariance
    #         K = np.dot(Sigma_bar, np.dot(H.T, np.linalg.inv(S)))

    #         # Updated state and covariance
    #         self.mu = mu_bar + np.dot(K, (z[i] - h))
    #         self.Sigma = np.dot((np.eye(3) - np.dot(K, H)), Sigma_bar)

    #         # Update landmark estimates
    #         m = np.array([self.mu[0] + z[i][0] * np.cos(z[i][1] + self.mu[2]),
    #                     self.mu[1] + z[i][0] * np.sin(z[i][1] + self.mu[2])])

    #         # Add the landmark to the state vector
    #         self.mu = np.hstack((self.mu, m))

    #         # Add the landmark to the covariance matrix
    #         self.Sigma = np.vstack((self.Sigma, np.zeros((2, self.Sigma.shape[1]))))
    #         self.Sigma = np.hstack((self.Sigma, np.zeros((self.Sigma.shape[0], 2))))
    #         self.Sigma[-2:, -2:] = np.eye(2) * self.Q

    # Update function
    def extend_sigma_mu(self, previous_landmarks, landmarks):
        """
        Parameters:
            previous_landmarks: list of previously seen landmarks
            landmarks: list of all landmarks

        Returns:
            previous_landmarks: updated list of previously seen landmarks
            mu_extended: extended mean of the state
            Sigma_extended: extended covariance of the state
        """

        new_landmarks = [landmark for landmark in landmarks if landmark not in previous_landmarks]

        if new_landmarks == []:
            return previous_landmarks, self.mu_extended, self.Sigma_extended
        
        # Update the list of previously seen landmarks
        previous_landmarks += new_landmarks

        # Update mu_extended with new landmarks only
        mu_extended = np.hstack([self.mu_extended] + [landmark.mu.flatten() for landmark in new_landmarks])

        # Create a block diagonal matrix for new landmarks
        block_matrix = np.block([[np.eye(2) if i != j else landmark.sigma 
                                for j, landmark in enumerate(new_landmarks)] 
                                for i, land in enumerate(new_landmarks)])

        # Update Sigma_extended with new landmarks only
        Sigma_extended = np.block([[self.Sigma_extended, np.zeros((self.Sigma_extended.shape[0], 2*len(new_landmarks)))], 
                                        [np.zeros((2*len(new_landmarks), self.Sigma_extended.shape[1])), 
                                        block_matrix]])
            
        return previous_landmarks, mu_extended, Sigma_extended

    
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

    def export_to_csv(self):
        # Define the CSV file path
        odom_file = 'odom.csv'    
        imu_file = 'imu.csv'
        ekf_file = 'ekf.csv'
        # Create a DataFrame from mu_extended and Sigma_extended

        ekf_dict = {}
        ekf_dict['mu'] = [self.mu.flatten()]
        ekf_dict['sigma'] = [self.Sigma.flatten()]
        
        for i in range(3):
            if i < len(self.landmarks) and self.landmarks[i] is not None:
                ekf_dict['mu_{}'.format(i)] = [self.landmarks[i].mu.flatten()]
                ekf_dict['sigma_{}'.format(i)] = [self.landmarks[i].sigma.flatten()]
            else:
                ekf_dict['mu_{}'.format(i)] = [np.array([-1000.0, -1000.0]).flatten()]
                ekf_dict['sigma_{}'.format(i)] = [np.array([[-1000.0, 0.0], [-1000.0, 0.0]]).flatten()]

        ekf_data = pd.DataFrame(ekf_dict, index=[0])

        # Check if the file exists
        if os.path.isfile(ekf_file):
            # If the file exists, append without writing the header
            ekf_data.to_csv(ekf_file, mode='a', header=False, index=False)
        else:
            # If the file does not exist, write the DataFrame with the header
            ekf_data.to_csv(ekf_file, index=False)
            
        #! odom
        odom_data = pd.DataFrame({
            'x': [self.odom_x],
            'y': [self.odom_y],
            'theta': [self.odom_theta]
        })

        # Check if the file exists
        if os.path.isfile(odom_file):
            # If the file exists, append without writing the header
            odom_data.to_csv(odom_file, mode='a', header=False, index=False)
        else:
            # If the file does not exist, write the DataFrame with the header
            odom_data.to_csv(odom_file, index=False)

        #! imu
        imu_data = pd.DataFrame({
            'x': [self.imu_x],
            'y': [self.imu_y],
            'theta': [self.imu_theta]
        })

        # Check if the file exists
        if os.path.isfile(imu_file):
            # If the file exists, append without writing the header
            imu_data.to_csv(imu_file, mode='a', header=False, index=False)
        else:
            # If the file does not exist, write the DataFrame with the header
            imu_data.to_csv(imu_file, index=False)
        

    def run(self):
        # Set the rate of the loop
        rate = rospy.Rate(10)

        # initiate previously seen landmarks
        previous_landmarks = []
        
        while not rospy.is_shutdown():
            # Move the robot in a circle
            self.draw_a_circle()

            # Call the motion model function
            v = 0.1 # linear velocity
            w = 0.1  # angular velocity
            dt = 0.1  # time step (corresponding to the rate of 10Hz)
            self.motion_model(v, w, dt)

            

            if self.landmarks != []:
                # Call the EKF function
                u = np.array([v, w])
                z = np.vstack([np.array([landmark.r, landmark.phi]) for landmark in self.landmarks])
                previously_landmarks, self.mu_extended, self.Sigma_extended = self.extend_sigma_mu(previous_landmarks, self.landmarks)
                self.mu_extended, self.Sigma_extended = self.EKF_SLAM(mu=self.mu_extended, Sigma=self.Sigma_extended, u=u, z=z, R=self.R, dt=dt)
                print("mu_extended: ", self.mu_extended)
                print("Sigma: ", self.Sigma_extended)
                print("==================================================================================")
            self.export_to_csv()

            # Only call the EKF function if distance and bearing are not None
            # if self.distance is not None and self.bearing is not None:
                # # Call the EKF function
                # u = np.array([v, w])
                # z = np.array([self.distance, self.bearing])
                # x_obj = 0.0
                # y_obj = 0.0
                # self.mu, self.Sigma = self.ekf(mu=self.mu, Sigma=self.Sigma, u=u, z=z, dt=dt, R=self.R, Q=self.Q, x_obj=x_obj, y_obj=y_obj)
                # mu_extended = np.hstack((self.mu, self.landmarks))
                

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
