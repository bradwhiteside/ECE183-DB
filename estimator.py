import numpy as np
import math
import time
import threading

class EKF():
    def __init__(self, get_time, get_position, get_linear_rate, get_linear_accelerations, get_IMU_accelertaions, get_orientation, get_Gyro, get_Magnetometer, get_state, get_motor_speeds, get_covariances, get_GPS, params, quads, quad_identifier):
                # get_time, get_position, get_linear_rate, get_orientation, get_angular_rate, quad.get_state, params=CONTROLLER_PARAMETERS,quad_identifier='q1'
       
        self.get_states = get_state
        self.get_time = get_time  
        self.get_positions = get_position  #x,y,z inertial frame
        self.get_linear_rates = get_linear_rate # x_dot, y_dot, z_dot in the inertial frame
        self.get_linear_accelerations = get_linear_accelerations
        self.get_IMU_accelertaions = get_IMU_accelertaions
        self.get_Magnetometer = get_Magnetometer
        self.get_orientations = get_orientation
        # self.get_angular_rates = get_angular_rate
        self.get_GPS = get_GPS
        self.get_Gyro = get_Gyro
        self.get_motor_speeds = get_motor_speeds #m1,m2,m3,m4
        
        self.quad_id = quad_identifier
        self.quad = quads[quad_identifier]
        self.weight = quads[quad_identifier]['weight']
        self.g = 9.81
        # self.b = 0.0245

        #Estimate varibles
        self.mean = np.zeros(12)
        self.cov = np.eye(12,12) * 0.1

        self.Q_accel, self.R, self.Q_gyro = get_covariances(self.quad_id) #3*3 for now

        self.Q =np.sqrt(self.Q_gyro**2 +  self.Q_accel**2)
  


        self.mean[0:3] = self.get_positions(self.quad_id) #x,y,z
        self.mean[3:6] = self.get_linear_rates(self.quad_id) #x_dot, y_dot, z_dot
        self.mean[6:9] = self.get_orientations(self.quad_id) #theta, phi, gamma

        self.thread_object = None
        self.target = [0,0,0]
        self.time_update_rate = 0.005   # the start thread function changes it to whatever set in main
        self.run = True


    def rotation_matrix(self,angles):
        ct = math.cos(angles[0])
        cp = math.cos(angles[1])
        cg = math.cos(angles[2])
        st = math.sin(angles[0])
        sp = math.sin(angles[1])
        sg = math.sin(angles[2])
        R_x = np.array([[1,0,0],[0,ct,-st],[0,st,ct]])
        R_y = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        R_z = np.array([[cg,-sg,0],[sg,cg,0],[0,0,1]])
        R = np.dot(R_z, np.dot( R_y, R_x))
        return R

    def wrap_angle(self,val):
        return((val + np.pi) % (2 * np.pi ) - np.pi)

    def convert_accel_to_angle(self,accel):
        a_x = accel[0]
        a_y = accel[1]
        a_z = accel[2]
        
        phi = -np.arctan2(a_y, (a_x**2+a_z**2)**0.5)
        theta = np.arctan2(a_x, (a_y**2+a_z**2)**0.5)
        gamma = np.arctan2((a_x**2+a_y**2)**0.5, a_z) - np.pi
        
        return np.array([phi, theta,  gamma])

    
    def time_update(self):
        #[x,y,z,x_dot,y_dot,z_dot,theta,phi,gamma,theta_dot,phi_dot,gamma_dot]
        #Complementary Filter for orientation estimation
        
        #Gyro
        self.mean[9:12] = self.get_Gyro(self.quad_id)   #theta_dot, phi_dot, gamma_dot
        self.mean[6:9] += self.mean[9:12] * self.time_update_rate #calc angles using the gyro
        
        accel_mean = self.get_IMU_accelertaions(self.quad_id)
        complementary_angles = self.convert_accel_to_angle(accel_mean) #self.time_update_rate **0 #calc angles using the accelerometer
        complementary_angles[2] = self.get_Magnetometer(self.quad_id)[2] 
        
        correction_gain = 0.5
        complementary_angles *= correction_gain
        #Complementary Filter
        G1 = 0.80
        G2 = 0.2
        self.mean[6:8] = G1 * self.mean[6:8] + G2 * complementary_angles[0:2]  #only for roll and pitch
        G3 = 0.98
        G4 = 0.02
        self.mean[8] = G3 * complementary_angles[2] + G4 * self.mean[8]

        #Kalman filter for x,y,z
        linear_accelerations = (self.rotation_matrix(self.mean[6:9]) @ (accel_mean)) + [0,0, self.g] # we add the gravity since when the drone is not accelrating

        self.mean[3:6] += linear_accelerations * self.time_update_rate
        F = self.mean[3:6]
        self.mean[0:3] += F * self.time_update_rate
        # print(self.mean[2] - self.get_positions(self.quad_id)[2])
        # print(self.mean[0:3])
        A = np.eye(3,3)
        G = np.eye(3,3) * [0.01,0.01,20]#* self.time_update_rate
        self.cov[0:3,0:3] = self.cov[0:3,0:3] + self.time_update_rate * (A @ self.cov[0:3,0:3] + self.cov[0:3,0:3] @ A.T + G @ self.Q @ G.T)
        # print(np.degrees(self.mean[6:9] - self.get_states(self.quad_id)[6:9]))
        
    def observation_update(self):
        #Start here
        _y = self.mean[0:3] 
        
        C = np.eye(3,3)
        
        y = self.get_GPS(self.quad_id) #measured y

        #Kalman Gain
        P = self.cov[0:3,0:3]
        H = C @ P @ C.T
        L = P @ C.T @ np.linalg.inv(self.R + H)
        
        LC = L @ C
        I = np.eye(LC.shape[0],LC.shape[1])
        P  = (I - LC) @ P

        #State and Cov update
        self.mean[0:3] +=  L @ (y - _y)
        self.cov[0:3,0:3] = P

    def get_estimated_state(self, quad_name):
        return self.mean

    def time_update_thread_run(self,time_update_rate,time_scaling):
        time_update_rate = time_update_rate*time_scaling
        last_update = self.get_time()
        while(self.run==True):
            time.sleep(0)
            self.time = self.get_time()
            if (self.time - last_update).total_seconds() > time_update_rate:
                self.time_update()
                # print("time_update")
                # print(self.mean[0:3] - self.get_positions(self.quad_id)) 
                last_update = self.time

    def observ_update_thread_run(self,observation_update_rate,time_scaling):
        observation_update_rate = observation_update_rate*time_scaling
        last_update = self.get_time()
        while(self.run==True):
            time.sleep(0)
            self.time = self.get_time()
            if (self.time - last_update).total_seconds() > observation_update_rate:
                self.observation_update()
                # print("observation_update")
                # print(((self.mean[0:3] - self.get_positions(self.quad_id)))) 
                last_update = self.time
            

    def start_thread(self,time_update_rate=0.005, observation_update_rate = 0.01, time_scaling=1):
        # self.update_rate = update_rate
        self.time_update_rate = time_update_rate #* time_scaling
        self.thread_object_time_update = threading.Thread(target=self.time_update_thread_run,args=(time_update_rate,time_scaling))
        self.thread_object_observ_update = threading.Thread(target=self.observ_update_thread_run,args=(observation_update_rate,time_scaling))
        self.thread_object_time_update.start()
        self.thread_object_observ_update.start()

    def stop_thread(self):
        self.run = False