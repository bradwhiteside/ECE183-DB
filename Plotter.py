import numpy as np
import matplotlib.pyplot as plt
import math
import time
import threading

class plotter():
    def __init__(self, get_time):
        self.times = np.empty(0)
        self.true_states = np.empty((0, 12), float)
        self.est_states = np.empty((0, 12), float)
        self.torques = np.empty((0,4), float)
        self.speeds = np.empty((0,6), float)
        self.accels = np.empty((0,3),float)
        self.input_goal = np.empty((0, 3), float)
        self.yaw_goal =np.empty(0) 
        self.get_time = get_time 

        self.run = True   


    def update(self):
        i_1 = -20
        i_2 = -1 
        # plt.close()
        # fig1, ax1 = plt.subplots(3,2,figsize=(10,  7))
        # fig1.suptitle('x, y, z, roll, pitch, yaw', fontsize=16)
        # # ax1[0,0].plot(times, self.est_states[:,0], label = "x dir_est", color = "green")
        # ax1[0,0].plot(self.times[i_1:i_2], self.input_goal[i_1:i_2,0], label = "x goal")
        # ax1[0,0].plot(self.times[i_1:i_2], self.true_states[i_1:i_2,0], label = "x dir")
        # ax1[0,0].set_xlabel('time (s)')
        # ax1[0,0].set_ylabel('x (m)')
        # ax1[0,0].legend()

        # # ax1[1,0].plot(self.times, self.est_states[:,1], label = "y dir_est" , color = "green")
        # ax1[1,0].plot(self.times[i_1:i_2], self.input_goal[i_1:i_2,1], label = "y goal")
        # ax1[1,0].plot(self.times[i_1:i_2], self.true_states[i_1:i_2,1], label = "y dir")
        # ax1[1,0].set_xlabel('time (s)')
        # ax1[1,0].set_ylabel('y (m)')
        # ax1[1,0].legend()

        # # ax1[2,0].plot(self.times, self.est_states[:,2], label = "z dir_est", color = "green")
        # ax1[2,0].plot(self.times[i_1:i_2], self.input_goal[i_1:i_2,2], label = "z goal")
        # ax1[2,0].plot(self.times[i_1:i_2], self.true_states[i_1:i_2,2], label = "z (altitude)")
        # ax1[2,0].set_xlabel('time (s)')
        # ax1[2,0].set_ylabel('z (m)')
        # # ax1[2,0].set_ylim([0,11])
        # ax1[2,0].legend()

        # # ax1[0,1].plot(self.times, np.degrees(self.est_states[:,6]), label = "roll_est", color = "green")
        # ax1[0,1].plot(self.times[i_1:i_2], np.degrees(self.true_states[i_1:i_2,6]), label = "roll")
        # ax1[0,1].set_xlabel('time (s)')
        # ax1[0,1].set_ylabel('roll (deg)')
        # ax1[0,1].legend()

        # # ax1[1,1].plot(self.times, np.degrees(self.est_states[:,7]), label = "pitch_est", color = "green")
        # ax1[1,1].plot(self.times[i_1:i_2], np.degrees(self.true_states[i_1:i_2,7]), label = "pitch")
        # ax1[1,1].set_xlabel('time (s)')
        # ax1[1,1].set_ylabel('pitch (deg)')
        # ax1[1,1].legend()

        # # ax1[2,1].plot(self.times, np.degrees(self.est_states[:,8]), label = "yaw_est", color = "green")
        # ax1[2,1].plot(self.times[i_1:i_2], np.degrees(self.yaw_goal[i_1:i_2]), label = "yaw goal")
        # ax1[2,1].plot(self.times[i_1:i_2], np.degrees(self.true_states[i_1:i_2,8]), label = "yaw")
        # ax1[2,1].set_xlabel('time (s)')
        # ax1[2,1].set_ylabel('yaw (deg)')
        # ax1[2,1].legend()


        # fig2, ax2 = plt.subplots(3,2,figsize=(10,  7))
        # fig2.suptitle('v_x, v_y, v_z, roll_rate, pitch_rate, yaw_rate', fontsize=16)
        # ax2[0,0].plot(self.times, self.true_states[:,3], label = "x_vel")
        # ax2[0,0].plot(self.times, self.est_states[:,3], label = "x_vel_est", color = "green")
        # ax2[0,0].set_xlabel('time (s)')
        # ax2[0,0].set_ylabel('v_x (m/s)')
        # ax2[0,0].legend()

        # ax2[1,0].plot(self.times, self.true_states[:,4], label = "y_vel")
        # ax2[1,0].plot(self.times, self.est_states[:,4], label = "y_vel_est", color = "green")
        # ax2[1,0].set_xlabel('time (s)')
        # ax2[1,0].set_ylabel('v_y (m/s)')
        # ax2[1,0].legend()

        # ax2[2,0].plot(self.times, self.true_states[:,5], label = "z_vel")
        # ax2[2,0].plot(self.times, self.est_states[:,5], label = "z_vel_est", color = "green")
        # ax2[2,0].set_xlabel('time (s)')
        # ax2[2,0].set_ylabel('v_z (m/s)')
        # ax2[2,0].legend()

        # ax2[0,1].plot(self.times, self.true_states[:,9], label = "roll_rate")
        # ax2[0,1].plot(self.times, self.est_states[:,9], label = "roll_rate_est", color = "green")
        # ax2[0,1].set_xlabel('time (s)')
        # ax2[0,1].set_ylabel('phi_rate (rad/s)')
        # ax2[0,1].legend()

        # ax2[1,1].plot(self.times, self.true_states[:,10], label = "pitch_rate")
        # ax2[1,1].plot(self.times, self.est_states[:,10], label = "pitch_rate_est", color = "green")
        # ax2[1,1].set_xlabel('time (s)')
        # ax2[1,1].set_ylabel('theta_rate (rad/s)')
        # ax2[1,1].legend()

        # ax2[2,1].plot(self.times, self.true_states[:,11], label = "yaw_rate_vel")
        # ax2[2,1].plot(self.times, self.est_states[:,11], label = "yaw_rate_est", color = "green")
        # ax2[2,1].set_xlabel('time (s)')
        # ax2[2,1].set_ylabel('gamma_rate (rad/s)')
        # ax2[2,1].legend()



        # fig3, ax3 = plt.subplots(4,1, figsize=(10,  7))
        # fig3.suptitle('Torques, roll, pitch, yaw', fontsize=16)
        # ax3[0].plot(self.times, torques[:,0], label = "roll torque")
        # ax3[0].set_xlabel('time (s)')
        # ax3[0].set_ylabel('roll (N.m)')
        # ax3[0].legend()
        # ax3[1].plot(self.times, torques[:,1], label = "pitch torque")
        # ax3[1].set_xlabel('time (s)')
        # ax3[1].set_ylabel('pitch (N.m)')
        # ax3[1].legend()
        # ax3[2].plot(self.times, torques[:,2], label = "yaw torque")
        # ax3[2].set_xlabel('time (s)')
        # ax3[2].set_ylabel('yaw (N.m)')
        # ax3[2].legend()
        # ax3[3].plot(self.times, torques[:,3], label = "Vertical Thrust")
        # ax3[3].set_xlabel('time (s)')
        # ax3[3].set_ylabel('T (N)')
        # ax3[3].legend()

        # fig4, ax4 = plt.subplots(6,1, figsize=(10,  7))
        # fig4.suptitle('motor speeds', fontsize=16)
        # for idx in range(0,6):
        #     ax4[idx].plot(self.times, speeds[:,idx], label = "m{idx}")
        #     ax4[idx].set_xlabel('time (s)')
        #     ax4[idx].set_ylabel('m{idx} (rad/s)')
        #     ax4[idx].legend()

        # fig5, ax5 = plt.subplots(3,1, figsize=(10,  7))
        # fig5.suptitle('Accelerations, a_x, a_y, a_z', fontsize=16)
        # ax5[0].plot(self.times, accels[:,0], label = "a_x")
        # ax5[0].set_xlabel('time (s)')
        # ax5[0].set_ylabel('a_x (m/s^2)')
        # ax5[0].legend()
        # ax5[1].plot(self.times, accels[:,1], label = "a_y")
        # ax5[1].set_xlabel('time (s)')
        # ax5[1].set_ylabel('a_y (m/s^)')
        # ax5[1].legend()
        # ax5[2].plot(self.times, accels[:,2], label = "a_z")
        # ax5[2].set_xlabel('time (s)')
        # ax5[2].set_ylabel('a_z (m/s^2)')
        # ax5[2].legend()
        
    def update_variables(self,times, true_states, est_states, torques, speeds, accels, input_goal, yaw_goal):
        self.times = times
        self.true_states = true_states
        self.est_states = est_states
        self.torques = torques
        self.speeds = speeds
        self.accels = accels
        self.input_goal = input_goal
        self.yaw_goal = yaw_goal


    def thread_run(self,update_rate,time_scaling):
        update_rate = update_rate*time_scaling
        last_update = self.get_time()
        while(self.run==True):
            time.sleep(0)
            self.time = self.get_time()
            if (self.time - last_update).total_seconds() > update_rate:
                self.update()
                last_update = self.time

    def start_thread(self,update_rate=0.005,time_scaling=1):
        self.thread_object = threading.Thread(target=self.thread_run,args=(update_rate,time_scaling))
        self.thread_object.start()

    def stop_thread(self):
        self.run = False

