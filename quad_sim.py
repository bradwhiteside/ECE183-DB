import quadcopter,gui, pycontroller, estimator
import signal
import sys
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import norm

# Constants
TIME_SCALING = 1.0 # Any positive number(Smaller is faster). 1.0->Real Time, 0.0->Run as fast as possible
QUAD_DYNAMICS_UPDATE = 0.001 # seconds
CONTROLLER_DYNAMICS_UPDATE = 0.005 # seconds
ESTIMATION_TIME_UPDATE = 0.005
ESTIMATION_OBSERVATION_UPDATE = 0.1
run = True

def Single_Point2Point():
    #Ramp input
    x_ramp = np.linspace(0,10,10)
    # x_ramp = np.hstack((x_ramp,np.linspace(10,0,10)))
    y_ramp = np.linspace(0,20,10)
    # y_ramp = np.hstack((y_ramp,np.linspace(20,0,10)))
    z_ramp = np.linspace(5,20,10)
    GOALS = np.vstack((x_ramp,y_ramp,z_ramp)).T 
    YAWS = np.hstack((np.linspace(0, np.pi/4,10)))#, np.linspace(0, 0, 10)))
    # YAWS = [0,0,0, np.pi/4,np.pi/4, np.pi/2,np.pi/2, 0.7 * np.pi, 0.7 *np.pi, 0.7 * np.pi, 0,0,0,0,0]
    start = GOALS[0]

    # Define the quadcopters
    QUADCOPTER={'q1':{'position': start,'orientation':[0,0,0],'L':0.5,'r':0.2,'prop_size':[21,9.5],'weight':7}} #w in kg, L and r in mm, prop_size in in

    #CONTROLLER_PARAMETERS _if using estimator
    CONTROLLER_PARAMETERS = {'Motor_limits':[1000, 45000],
                        'Tilt_limits':[-5, 5],   #degrees
                        'Z_XY_offset':500,
                        'Linear_PID':{'P':[1160000,1100000,30000],'I':[20,20,30],'D':[940000,940000,50000]},
                        'Linear_To_Angular_Scaler':[1,1,0],
                        'Yaw_Control_Limits':[-900,900],
                        'Yaw_Rate_Scaler':1.1,
                        'Angular_PID':{'P':[8000,8000,3000],'I':[0,0,0],'D':[2000,2000,1500]},
                        }

    # Catch Ctrl+C to stop threads
    signal.signal(signal.SIGINT, signal_handler)
    # Make objects for quadcopter, gui and controller
    quad = quadcopter.Quadcopter(QUADCOPTER)
    est = estimator.EKF(quad.get_time, quad.get_position,quad.get_linear_rate, quad.get_linear_accelertaions, quad.get_IMU_accelertaions, quad.get_orientation, quad.get_Gyro, quad.get_state, quad.get_motor_speeds,quad.get_covariances, quad.get_GPS, params=CONTROLLER_PARAMETERS, quads=QUADCOPTER, quad_identifier='q1')
    ctrl = pycontroller.Controller_PID_Point2Point(quad.get_state,quad.get_time,quad.set_motor_speeds,est.get_estimated_state, quad.get_L, params=CONTROLLER_PARAMETERS,quad_identifier='q1')
    # gui_object = gui.GUI(quads=QUADCOPTER)

    # Start the threads
    quad.start_thread(dt=QUAD_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
    ctrl.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
    est.start_thread(time_update_rate=ESTIMATION_TIME_UPDATE, observation_update_rate = ESTIMATION_OBSERVATION_UPDATE ,time_scaling=TIME_SCALING)
    
    # Update the GUI while switching between destination poitions
    # output_file_name = "outputs/" + map.split('/')[2] + "_path_data.csv"
    # output_file = open(output_file_name, 'w', buffering=65536)
    times = np.empty(0)
    input_goal = np.empty((0, 3), float)
    yaw_goal = np.empty(0)

    true_states = np.empty((0, 12), float)
    est_states = np.empty((0,12), float)
    torques = np.empty((0,4), float)
    speeds = np.empty((0,6), float)
    accels = np.empty((0,3),float)

    simulation_start_time = quad.get_time()
    #Simulation
    time_limit = 3 * TIME_SCALING  #Amount of time limit to spend on a Goal            
    tolorance = 0.8                    #Steady state error

    for goal,yaw in zip(GOALS,YAWS):
        print(goal)
        ctrl.update_target(goal)
        ctrl.update_yaw_target(yaw)
        goal_start_time = quad.get_time()
        time_laps = 0
        true_pos = np.array(quad.get_state('q1')[0:3])
        est_state =  np.array(est.get_estimated_state('q1')[0:3])
        dist = np.linalg.norm(est_state - goal)
        

        while dist > tolorance: #time_laps < time_limit:# 
            # print("dist",dist)
            # print(t)
            # gui_object.quads['q1']['position'] = quad.get_position('q1')
            # gui_object.quads['q1']['orientation'] = quad.get_orientation('q1')
            # gui_object.update()

            true_state = np.array(quad.get_state('q1'))
            est_state =  np.array(est.get_estimated_state('q1'))

            true_states = np.append(true_states, np.array([true_state]), axis=0)
            est_states = np.append(est_states, np.array([est_state]), axis=0)
            torque = quad.get_tau()
            torques = np.append(torques, np.array([torque]), axis = 0)
            speeds = np.append(speeds, np.array([quad.get_motor_speeds('q1')]), axis = 0)
            accels = np.append(accels, np.array([quad.get_linear_accelertaions('q1')]), axis = 0)


            time = quad.get_time()
            times = np.append(times, np.array([(time-simulation_start_time).total_seconds()]), axis=0)
            time_laps = (datetime.datetime.now()-goal_start_time).total_seconds() 
            
            # dist = np.linalg.norm(true_state[0:3] - goal)
            dist = np.linalg.norm(est_state[0:3] - goal)

            input_goal = np.append(input_goal, np.array([goal]), axis=0)
            yaw_goal = np.append(yaw_goal, np.array([yaw]), axis=0)

    quad.stop_thread()
    ctrl.stop_thread()
    est.stop_thread()

    error = true_states - est_states

    # np.savetxt('error_analysis/true_data.txt', true_states)
    # np.savetxt('error_analysis/est_data.txt', est_states)


    #---------------------------------
    #Plot the path
    fig1, ax1 = plt.subplots(3,2,figsize=(10,  7))
    fig1.suptitle('x, y, z, roll, pitch, yaw', fontsize=16)
    ax1[0,0].plot(times, est_states[:,0], label = "x dir_est", color = "green")
    ax1[0,0].plot(times, input_goal[:,0], label = "x goal")
    ax1[0,0].plot(times, true_states[:,0], label = "x dir")
    ax1[0,0].set_xlabel('time (s)')
    ax1[0,0].set_ylabel('x (m)')
    ax1[0,0].legend()
    
    ax1[1,0].plot(times, est_states[:,1], label = "y dir_est" , color = "green")
    ax1[1,0].plot(times, input_goal[:,1], label = "y goal")
    ax1[1,0].plot(times, true_states[:,1], label = "y dir")
    ax1[1,0].set_xlabel('time (s)')
    ax1[1,0].set_ylabel('y (m)')
    ax1[1,0].legend()
    
    ax1[2,0].plot(times, est_states[:,2], label = "z dir_est", color = "green")
    ax1[2,0].plot(times, input_goal[:,2], label = "z goal")
    ax1[2,0].plot(times, true_states[:,2], label = "z (altitude)")
    ax1[2,0].set_xlabel('time (s)')
    ax1[2,0].set_ylabel('z (m)')
    # ax1[2,0].set_ylim([0,11])
    ax1[2,0].legend()

    ax1[0,1].plot(times, np.degrees(est_states[:,6]), label = "roll_est", color = "green")
    ax1[0,1].plot(times, np.degrees(true_states[:,6]), label = "roll")
    ax1[0,1].set_xlabel('time (s)')
    ax1[0,1].set_ylabel('roll (deg)')
    ax1[0,1].legend()
    
    ax1[1,1].plot(times, np.degrees(est_states[:,7]), label = "pitch_est", color = "green")
    ax1[1,1].plot(times, np.degrees(true_states[:,7]), label = "pitch")
    ax1[1,1].set_xlabel('time (s)')
    ax1[1,1].set_ylabel('pitch (deg)')
    ax1[1,1].legend()
    
    ax1[2,1].plot(times, np.degrees(est_states[:,8]), label = "yaw_est", color = "green")
    ax1[2,1].plot(times, np.degrees(yaw_goal), label = "yaw goal")
    ax1[2,1].plot(times, np.degrees(true_states[:,8]), label = "yaw")
    ax1[2,1].set_xlabel('time (s)')
    ax1[2,1].set_ylabel('yaw (deg)')
    ax1[2,1].legend()

    
    fig2, ax2 = plt.subplots(3,2,figsize=(10,  7))
    fig2.suptitle('v_x, v_y, v_z, roll_rate, pitch_rate, yaw_rate', fontsize=16)
    ax2[0,0].plot(times, true_states[:,3], label = "x_vel")
    ax2[0,0].plot(times, est_states[:,3], label = "x_vel_est", color = "green")
    ax2[0,0].set_xlabel('time (s)')
    ax2[0,0].set_ylabel('v_x (m/s)')
    ax2[0,0].legend()
    
    ax2[1,0].plot(times, true_states[:,4], label = "y_vel")
    ax2[1,0].plot(times, est_states[:,4], label = "y_vel_est", color = "green")
    ax2[1,0].set_xlabel('time (s)')
    ax2[1,0].set_ylabel('v_y (m/s)')
    ax2[1,0].legend()

    ax2[2,0].plot(times, true_states[:,5], label = "z_vel")
    ax2[2,0].plot(times, est_states[:,5], label = "z_vel_est", color = "green")
    ax2[2,0].set_xlabel('time (s)')
    ax2[2,0].set_ylabel('v_z (m/s)')
    ax2[2,0].legend()

    ax2[0,1].plot(times, true_states[:,9], label = "roll_rate")
    ax2[0,1].plot(times, est_states[:,9], label = "roll_rate_est", color = "green")
    ax2[0,1].set_xlabel('time (s)')
    ax2[0,1].set_ylabel('phi_rate (rad/s)')
    ax2[0,1].legend()
    
    ax2[1,1].plot(times, true_states[:,10], label = "pitch_rate")
    ax2[1,1].plot(times, est_states[:,10], label = "pitch_rate_est", color = "green")
    ax2[1,1].set_xlabel('time (s)')
    ax2[1,1].set_ylabel('theta_rate (rad/s)')
    ax2[1,1].legend()

    ax2[2,1].plot(times, true_states[:,11], label = "yaw_rate_vel")
    ax2[2,1].plot(times, est_states[:,11], label = "yaw_rate_est", color = "green")
    ax2[2,1].set_xlabel('time (s)')
    ax2[2,1].set_ylabel('gamma_rate (rad/s)')
    ax2[2,1].legend()



    fig3, ax3 = plt.subplots(4,1, figsize=(10,  7))
    fig3.suptitle('Torques, roll, pitch, yaw', fontsize=16)
    ax3[0].plot(times, torques[:,0], label = "roll torque")
    ax3[0].set_xlabel('time (s)')
    ax3[0].set_ylabel('roll (N.m)')
    ax3[0].legend()
    ax3[1].plot(times, torques[:,1], label = "pitch torque")
    ax3[1].set_xlabel('time (s)')
    ax3[1].set_ylabel('pitch (N.m)')
    ax3[1].legend()
    ax3[2].plot(times, torques[:,2], label = "yaw torque")
    ax3[2].set_xlabel('time (s)')
    ax3[2].set_ylabel('yaw (N.m)')
    ax3[2].legend()
    ax3[3].plot(times, torques[:,3], label = "Vertical Thrust")
    ax3[3].set_xlabel('time (s)')
    ax3[3].set_ylabel('T (N)')
    ax3[3].legend()

    fig4, ax4 = plt.subplots(6,1, figsize=(10,  7))
    fig4.suptitle('motor speeds', fontsize=16)
    for idx in range(0,6):
        ax4[idx].plot(times, speeds[:,idx], label = "m{idx}")
        ax4[idx].set_xlabel('time (s)')
        ax4[idx].set_ylabel('m{idx} (rad/s)')
        ax4[idx].legend()

    fig5, ax5 = plt.subplots(3,1, figsize=(10,  7))
    fig5.suptitle('Accelerations, a_x, a_y, a_z', fontsize=16)
    ax5[0].plot(times, accels[:,0], label = "a_x")
    ax5[0].set_xlabel('time (s)')
    ax5[0].set_ylabel('a_x (m/s^2)')
    ax5[0].legend()
    ax5[1].plot(times, accels[:,1], label = "a_y")
    ax5[1].set_xlabel('time (s)')
    ax5[1].set_ylabel('a_y (m/s^)')
    ax5[1].legend()
    ax5[2].plot(times, accels[:,2], label = "a_z")
    ax5[2].set_xlabel('time (s)')
    ax5[2].set_ylabel('a_z (m/s^2)')
    ax5[2].legend()
    plt.show()

    return error


def Multi_Point2Point():
    # Set goals to go to
    GOALS_1 = [(-1,-1,4),(1,1,2)]
    GOALS_2 = [(1,-1,2),(-1,1,4)]
    # Define the quadcopters
    QUADCOPTERS={'q1':{'position':[1,0,4],'orientation':[0,0,0],'L':0.3,'r':0.1,'prop_size':[10,4.5],'weight':1.2},
        'q2':{'position':[-1,0,4],'orientation':[0,0,0],'L':0.15,'r':0.05,'prop_size':[6,4.5],'weight':0.7}}
    # Controller parameters
    CONTROLLER_1_PARAMETERS = {'Motor_limits':[4000,9000],
                        'Tilt_limits':[-10,10],
                        'Yaw_Control_Limits':[-900,900],
                        'Z_XY_offset':500,
                        'Linear_PID':{'P':[300,300,7000],'I':[0.04,0.04,4.5],'D':[450,450,5000]},
                        'Linear_To_Angular_Scaler':[1,1,0],
                        'Yaw_Rate_Scaler':0.18,
                        'Angular_PID':{'P':[22000,22000,1500],'I':[0,0,1.2],'D':[12000,12000,0]},
                        }
    CONTROLLER_2_PARAMETERS = {'Motor_limits':[4000,9000],
                        'Tilt_limits':[-10,10],
                        'Yaw_Control_Limits':[-900,900],
                        'Z_XY_offset':500,
                        'Linear_PID':{'P':[300,300,7000],'I':[0.04,0.04,4.5],'D':[450,450,5000]},
                        'Linear_To_Angular_Scaler':[1,1,0],
                        'Yaw_Rate_Scaler':0.18,
                        'Angular_PID':{'P':[22000,22000,1500],'I':[0,0,1.2],'D':[12000,12000,0]},
                        }

    # Catch Ctrl+C to stop threads
    signal.signal(signal.SIGINT, signal_handler)
    # Make objects for quadcopter, gui and controllers
    gui_object = gui.GUI(quads=QUADCOPTERS)
    quad = quadcopter.Quadcopter(quads=QUADCOPTERS)
    ctrl1 = pycontroller.Controller_PID_Point2Point(quad.get_state,quad.get_time,quad.set_motor_speeds,params=CONTROLLER_1_PARAMETERS,quad_identifier='q1')
    ctrl2 = pycontroller.Controller_PID_Point2Point(quad.get_state,quad.get_time,quad.set_motor_speeds,params=CONTROLLER_2_PARAMETERS,quad_identifier='q2')
    # Start the threads
    quad.start_thread(dt=QUAD_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
    ctrl1.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
    ctrl2.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
    # Update the GUI while switching between destination poitions
    while(run==True):
        for goal1,goal2 in zip(GOALS_1,GOALS_2):
            ctrl1.update_target(goal1)
            ctrl2.update_target(goal2)
            for i in range(150):
                for key in QUADCOPTERS:
                    gui_object.quads[key]['position'] = quad.get_position(key)
                    gui_object.quads[key]['orientation'] = quad.get_orientation(key)
                gui_object.update()
    quad.stop_thread()
    ctrl1.stop_thread()
    ctrl2.stop_thread()

def Single_Velocity():
    # Set goals to go to
    GOALS = [(0.5,0,2),(0,0.5,2),(-0.5,0,2),(0,-0.5,2)]
    # Define the quadcopters
    QUADCOPTER={'q1':{'position':[0,0,0],'orientation':[0,0,0],'L':0.3,'r':0.1,'prop_size':[10,4.5],'weight':1.2}}
    # Controller parameters
    CONTROLLER_PARAMETERS = {'Motor_limits':[4000,9000],
                        'Tilt_limits':[-10,10],
                        'Yaw_Control_Limits':[-900,900],
                        'Z_XY_offset':500,
                        'Linear_PID':{'P':[2000,2000,7000],'I':[0.25,0.25,4.5],'D':[50,50,5000]},
                        'Linear_To_Angular_Scaler':[1,1,0],
                        'Yaw_Rate_Scaler':0.18,
                        'Angular_PID':{'P':[22000,22000,1500],'I':[0,0,1.2],'D':[12000,12000,0]},
                        }

    # Catch Ctrl+C to stop threads
    signal.signal(signal.SIGINT, signal_handler)
    # Make objects for quadcopter, gui and controller
    quad = quadcopter.Quadcopter(QUADCOPTER)
    gui_object = gui.GUI(quads=QUADCOPTER)
    ctrl = pycontroller.Controller_PID_Velocity(quad.get_state,quad.get_time,quad.set_motor_speeds,params=CONTROLLER_PARAMETERS,quad_identifier='q1')
    # Start the threads
    quad.start_thread(dt=QUAD_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
    ctrl.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
    # Update the GUI while switching between destination poitions
    while(run==True):
        for goal in GOALS:
            ctrl.update_target(goal)
            for i in range(150):
                gui_object.quads['q1']['position'] = quad.get_position('q1')
                gui_object.quads['q1']['orientation'] = quad.get_orientation('q1')
                gui_object.update()
    quad.stop_thread()
    ctrl.stop_thread()

def parse_args():
    parser = argparse.ArgumentParser(description="Quadcopter Simulator")
    parser.add_argument("--sim", help='single_p2p, multi_p2p or single_velocity', default='single_p2p')
    parser.add_argument("--time_scale", type=float, default=-1.0, help='Time scaling factor. 0.0:fastest,1.0:realtime,>1:slow, ex: --time_scale 0.1')
    parser.add_argument("--quad_update_time", type=float, default=0.0, help='delta time for quadcopter dynamics update(seconds), ex: --quad_update_time 0.002')
    parser.add_argument("--controller_update_time", type=float, default=0.0, help='delta time for controller update(seconds), ex: --controller_update_time 0.005')
    return parser.parse_args()

def signal_handler(signal, frame):
    global run
    run = False
    print('Stopping')
    sys.exit(0)

if __name__ == "__main__":

    number_of_trials = 1
    error = Single_Point2Point()
    print("error shape is:", error.shape[0])

    for i in range(0,number_of_trials-1):
        print("Trial", i)
        new_error = Single_Point2Point()
        error_len = error.shape[0]
        new_error_leng = new_error.shape[0]
        diff = np.abs(new_error_leng - error_len)
        
        if new_error_leng > error_len :
            # new_error = new_error[:-diff]
            # error += new_error #np.concatenate((error, new_error))
            np.concatenate((error, new_error))
            print('new_error shranked by {} elements'.format(diff))

        elif error_len > new_error_leng :
            # error = error[:-diff]
            # error += new_error
            np.concatenate((error, new_error))
            print('error shranked by {} elements'.format(diff))


        # print("Error length is:",new_error_leng)
        # error = np.concatenate((error, Single_Point2Point()))

    #error = error/number_of_trials
    print(error.shape)
    np.savetxt('error_analysis/errors.txt', error)


    # args = parse_args()
    # if args.time_scale>=0: TIME_SCALING = args.time_scale
    # if args.quad_update_time>0: QUAD_DYNAMICS_UPDATE = args.quad_update_time
    # if args.controller_update_time>0: CONTROLLER_DYNAMICS_UPDATE = args.controller_update_time
    # if args.sim == 'single_p2p':
    #     Single_Point2Point()
    # elif args.sim == 'multi_p2p':
    #     Multi_Point2Point()
    # elif args.sim == 'single_velocity':
    #     Single_Velocity()
