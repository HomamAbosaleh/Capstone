import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, atan2
import rospy
import pandas as pd
import os

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from robot_controller.msg import DetectedObject
from geometry_msgs.msg import Twist


# Defining Constants
IMG_WIDTH = 416
IMG_HEIGHT = 416

CLASS_EQUATIONS = {
    'small chair': lambda x1, x2, y1, y2: 549 * np.exp(-0.0001644 * (x2 - x1) * (y2 - y1)) + 238.6 * np.exp(-1.342e-05 * (x2 - x1) * (y2 - y1)),
    'big bin': lambda x1, x2, y1, y2: 510.4 * np.exp(-6.066e-05 * (x2 - x1) * (y2 - y1)) + 75.81 * np.exp(3.734e-06 * (x2 - x1) * (y2 - y1)),
    'medium bin': lambda x1, x2, y1, y2: 438.8 * np.exp(-0.0002002 * (x2 - x1) * (y2 - y1)) + 239.7 * np.exp(-1.652e-05 * (x2 - x1) * (y2 - y1)),
    'small bin': lambda x1, x2, y1, y2: 478.2 * np.exp(-0.000597 * (x2 - x1) * (y2 - y1)) + 254 * np.exp(-4.499e-05 * (x2 - x1) * (y2 - y1)),
}

class Plotting:
    def __init__(self):
        self.true_x, self.true_y, self.true_theta = [], [], []
        self.pred_x, self.pred_y, self.pred_theta = [], [], []
        self.pred_lm_x, self.pred_lm_y = [], []
        self.time = []

    def update(self, true_states, pred_states, time):
        self.true_x.append(true_states[0])
        self.true_y.append(true_states[1])
        self.true_theta.append(true_states[2])

        self.pred_x.append(pred_states[0])
        self.pred_y.append(pred_states[1])
        self.pred_theta.append(pred_states[2])

        self.pred_lm_x.append(pred_states[3])
        self.pred_lm_y.append(pred_states[4])

        self.time.append(time)

    def show(self, mu, landmarks, N):
        plt.plot(self.true_x, self.true_y, label='True')
        plt.plot(self.pred_x, self.pred_y, label='Predicted')
        plt.plot([mark.x for mark in landmarks], [mark.y for mark in landmarks], 'gX', label='True Landmarks')
        plt.plot([mu[3 + 2 * idx, 0] for idx in range(N)],
                 [mu[4 + 2 * idx, 0] for idx in range(N)], 'rX', label='Predicted Landmarks')
        plt.legend()
        plt.grid()
        plt.show()


class Landmark:
    def __init__(self, x, y, sig, r, phi, classname):
        """
        x: x position of landmark
        y: y position of landmark
        sig: unique signature of landmark
        r: distance to the landmark
        phi: bearing angle
        classname: name of the class
        """
        self.r = r
        self.phi = phi
        self.class_name = classname
        self.mu = np.zeros((2, 1))
        self.sigma = 1000.0 * np.eye(2)

        self.x = x
        self.y = y
        self.s = sig
        self.seen = False

        self.x_hat = 0.0
        self.y_hat = 0.0
        # self.s_hat = 0.


class Measurement:
    def __init__(self, rng, ang, j, landmark):
        """
        Parameters:
            rng: distance to the landmark
            ang: bearing angle
            j: signature
            landmarks: list of landmark
        """
        self.rng = rng
        self.ang = ang
        self.id = j
        self.landmark = landmark
        # for lmrk in landmarks:
        #     if lmrk.s == j:
        #         self.landmark = lmrk
        


class EKFSLAM:

    @staticmethod
    def motion(v, w, theta, dt):
        """
        Parameters:
            v: linear velocity of the robot
            w: angular velocity of the robot
            theta: heading angle of the robot
            dt: time step

        Return:
            a: motion model
            b: jacobian of motion model
        """
        # Avoid divide by zero
        if w == 0.:
            w += 1e-5

        # Motion without noise/errors
        theta_dot = w * dt

        x_dot = (-v/w) * sin(theta) + (v/w) * sin(theta + w*dt)
        y_dot = (v/w) * cos(theta) - (v/w) * cos(theta + w*dt)
        a = np.array([x_dot, y_dot, theta_dot]).reshape(-1, 1)

        # Derivative of above motion model
        b = np.zeros((3, 3))
        b[0, 2] = (-v/w) * cos(theta) + (v/w) * cos(theta + w*dt)
        b[1, 2] = (-v/w) * sin(theta) + (v/w) * sin(theta + w*dt)

        return a, b

    def predict(self, prev_mu=None, prev_sigma=None, u=None, z=None, dt=None, N=None, R=None, Q=None):
        """
        Parameters:
            prev_mu: previous state of the robot and landmarks
            prev_sigma: previous covariance of the robot and landmarks
            u: input to the motion model
            z: measurements of the landmarks
            dt: time step
            N: number of landmarks
            R: motion noise
            Q: measurement noise
        
        Return:
            mu: new state of the robot and landmarks
            sigma: new covariance of the robot and landmarks
        """
        Fx = np.eye(3, 2*N+3)

        f, g = self.motion(u[0], u[1], prev_mu[2, 0], dt)
        mu_bar = prev_mu + (Fx.T @ f)

        G = (Fx.T @ g @ Fx) + np.eye(2*N+3)
        sigma_bar = (G @ prev_sigma @ G.T) + (Fx.T @ R @ Fx)

        for obs in z:
            j = obs.landmark.s
            z_i = np.array([obs.rng, obs.ang]).reshape(-1, 1)
            if not obs.landmark.seen:
                mu_bar[3+2*j, 0] = mu_bar[0, 0] + obs.rng * cos(obs.ang + mu_bar[2, 0])  # x
                mu_bar[4+2*j, 0] = mu_bar[1, 0] + obs.rng * sin(obs.ang + mu_bar[2, 0])  # y
                # mu_bar[5+3*j, 0] = obs.landmark.s  # s
                obs.landmark.seen = True

            delt_x = mu_bar[3+2*j, 0] - mu_bar[0, 0]
            delt_y = mu_bar[4+2*j, 0] - mu_bar[1, 0]
            delt = np.array([delt_x, delt_y]).reshape(-1, 1)
            q = delt.T @ delt

            z_i_hat = np.zeros((2, 1))
            z_i_hat[0, 0] = np.sqrt(q)
            z_i_hat[1, 0] = atan2(delt_y, delt_x) - mu_bar[2, 0]
            # z_i_hat[2, 0] = obs.landmark.s

            Fxj_a = np.eye(5, 3)
            Fxj_b = np.zeros((5, 2*N))
            Fxj_b[3:, 2*j:2+2*j] = np.eye(2)
            Fxj = np.hstack((Fxj_a, Fxj_b))

            h = np.zeros((2, 5))
            h[0, 0] = -np.sqrt(q) * delt_x
            h[0, 1] = -np.sqrt(q) * delt_y
            h[0, 3] = np.sqrt(q) * delt_x
            h[0, 4] = np.sqrt(q) * delt_y
            h[1, 0] = delt_y
            h[1, 1] = -delt_x
            h[1, 2] = -q
            h[1, 3] = -delt_y
            h[1, 4] = delt_x
            # h[2, 5] = q

            H_i = (1/q) * (h @ Fxj)
            K_i = sigma_bar @ H_i.T @ np.linalg.inv((H_i @ sigma_bar @ H_i.T + Q))

            mu_bar = mu_bar + (K_i @ (z_i-z_i_hat))
            sigma_bar = (np.eye(sigma_bar.shape[0]) - (K_i @ H_i)) @ sigma_bar

        return mu_bar, sigma_bar


class RobotController:
    def __init__(self):
        # Initialize the camera subscriber
        self.camera_sub = rospy.Subscriber("/camera", DetectedObject, self.camera_callback)    

        # Initialize the measurement subscribers
        # self.imu_sub = rospy.Subscriber("/imu", Imu, self.imu_callback)
        # self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)

        # Initialize the control publisher
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

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
        self.R = 0.1*np.eye(3)
        
        self.Q = 0.05*np.eye(2)
        
        # Initialize the landmark estimates
        self.landmarks = []
        
    def camera_callback(self, msg):
        if msg.class_name != "NULL":
            # Call the measurement function based on the class name
            if msg.class_name == "big bin" or msg.class_name == "medium bin" or msg.class_name == "small bin":
                distance, bearing = self.calculate_distance_and_angle(msg.x1, msg.x2, msg.y1, msg.y2, msg.class_name)
                landmark = next((landmark for landmark in self.landmarks if landmark.class_name == msg.class_name), None)
                if(landmark is None):
                    self.landmarks.append(Landmark(0.0, 0.0, 
                        sig=len(self.landmarks), r=distance, phi=bearing, classname=msg.class_name))
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
        
        Return:
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

        angle_in_radians = atan2(difference_object_image * px_in_meter, focal_in_meter)
        #print("Class name: ", class_name)
        #print("Distance in centimeters: ", distance_in_meters)
        #print("Bearning angle in radians: ", angle_in_radians)
        #print("==================================================================================")

        return distance_in_meters, angle_in_radians

    def performance(self, pred, N):
        """
        Parameters:
            pred: 
        """
        pred_dict = dict()
        pred_dict['X'] = pred[0, 0]
        pred_dict['Y'] = pred[1, 0]
        pred_dict['THETA'] = pred[2, 0]
        for n in range(N):
            pred_dict['LM_' + str(n) + ' X'] = pred[3 + 2 * n, 0]
            pred_dict['LM_' + str(n) + ' Y'] = pred[4 + 2 * n, 0]
            # pred_dict['LM_' + str(n) + ' ID'] = pred[5 + 3 * n, 0]

        print('PREDICTED STATES')
        print(pred_dict)


    # def sensor(self, landmark, states, Q):
    #     """
    #     Parameters:
    #         landmark: landmark object
    #         states: robot states
    #         Q: measurement noise
        
    #     Return:
    #         z: measurment of the landmark
    #     """
    #     rng_noise = np.random.normal(0., Q[0, 0])
    #     ang_noise = np.random.normal(0., Q[1, 1])
    #     z_rng = np.sqrt((states[0] - landmark.x) ** 2 + (states[1] - landmark.y) ** 2) + rng_noise
    #     z_ang = atan2(landmark.y - states[1], landmark.x - states[0]) - states[2] + ang_noise
    #     z_j = landmark.s
    #     z = Measurement(z_rng, z_ang, z_j)
    #     return z

    def state_update(self, states, input, R, dt):
        """
        Parameters:
            states: robot states
            input: velocity and angular velocity 
            R: motion model noise
            dt: time step
        Return:
            new states of the robot
        """
        x, y, theta = states
        v, w = input

        theta_dot = w
        x_dot = v*cos(theta)
        y_dot = v*sin(theta)

        theta += (theta_dot + np.random.normal(0., R[2, 2])) * dt
        x += (x_dot + np.random.normal(0., R[0, 0])) * dt
        y += (y_dot + np.random.normal(0., R[1, 1])) * dt


        # updating the robot x, y, theta
        self.x = x
        self.y = y
        self.theta = theta
        return np.array([x, y, theta])
    
    # Update function
    def extend_sigma_mu(self, previous_landmarks, landmarks):
        """
        Parameters:
            previous_landmarks: list of previously seen landmarks
            landmarks: list of all landmarks

        Return:
            previous_landmarks: updated list of previously seen landmarks
            mu_extended: extended mean of the state
            sigma_extended: extended covariance of the state
        """

        new_landmarks = [landmark for landmark in landmarks if landmark not in previous_landmarks]

        if len(new_landmarks) == 0:
            return previous_landmarks, self.mu_extended, self.sigma_extended
        
        # Update the list of previously seen landmarks
        previous_landmarks += new_landmarks

        # Update mu_extended with new landmarks only
        mu_extended = np.vstack((self.mu_extended, np.array([landmark.mu.flatten() for landmark in new_landmarks]).reshape(-1, 1)))

        # Create a block diagonal matrix for new landmarks
        block_matrix = np.block([[np.eye(2) if i != j else landmark.sigma 
                                for j, landmark in enumerate(new_landmarks)] 
                                for i, land in enumerate(new_landmarks)])

        # Update sigma_extended with new landmarks only
        sigma_extended = np.block([[self.sigma_extended, np.zeros((self.sigma_extended.shape[0], 2*len(new_landmarks)))], 
                                        [np.zeros((2*len(new_landmarks), self.sigma_extended.shape[1])), 
                                        block_matrix]])
            
        return previous_landmarks, mu_extended, sigma_extended
    
    def draw_a_circle(self):
        # Create a new Twist message
        vel_msg = Twist()

        # Set the linear velocity (forward speed) to 0.03 m/s
        vel_msg.linear.x = 0.03

        # Set the angular velocity (turn speed) based on the desired radius of the circle
        # Angular velocity is linear velocity divided by the radius
        # For a circle of radius 0.5 meters, the angular velocity is 0.03 / 0.5 = 0.06 rad/s
        vel_msg.angular.z = 0.06

        # Publish the velocity message
        self.cmd_pub.publish(vel_msg)
    
    def run(self):
        # Set the rate of the loop
        rate = rospy.Rate(10)

        # initiate previously seen landmarks
        previous_landmarks = []

        v = 0.03 # linear velocity
        w = 0.06  # angular velocity
        dt = 0.1  # time step (corresponding to the rate of 10Hz)
        t = 0.0

        ekf = EKFSLAM()
        plot = Plotting()

        self.mu_extended = np.array([self.x, self.y, self.theta]).reshape(-1, 1)
        self.sigma_extended = np.zeros((3, 3))

        states = np.array([self.x, self.y, self.theta])
        u = np.array([v, w])

        while not rospy.is_shutdown():
            # Move the robot in a circle
            self.draw_a_circle()

            N = len(self.landmarks)
            previously_landmarks, self.mu_extended, self.sigma_extended = self.extend_sigma_mu(previous_landmarks, self.landmarks)
            if N != 0:
                plot.update(states.flatten().copy(), self.mu_extended.flatten().copy(), t)

            measurements = [Measurement(rng=landmark.r, ang=landmark.phi, j=landmark.s, landmark=landmark) for landmark in self.landmarks]
            states = self.state_update(states, u, self.R, dt)

            self.mu_extended, self.sigma_extended = ekf.predict(prev_mu=self.mu_extended, prev_sigma=self.sigma_extended, u=u, z=measurements, dt=dt, N=N, Q=self.Q, R=self.R)
            t += dt

            # Sleep for the remainder of the loop
            rate.sleep()
        self.performance(np.round(self.mu_extended, 3), N)
        plot.show(self.mu_extended, self.landmarks, N)


if __name__ == "__main__":
    # Initialize the node
    rospy.init_node("robot_controller_node")

    # Create an instance of the RobotController class
    robot_controller = RobotController()

    # Run the robot controller
    robot_controller.run()