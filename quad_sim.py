import quadcopter,gui,controller
import signal
import sys
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random

# Constants
TIME_SCALING = 0.1 # Any positive number(Smaller is faster). 1.0->Real Time, 0.0->Run as fast as possible
QUAD_DYNAMICS_UPDATE = 0.001 # seconds
CONTROLLER_DYNAMICS_UPDATE = 0.005 # seconds
run = True

def Single_Point2Point():
    # Set goals to go to
    # Ramp input
    x_ramp = np.linspace(0,20,10)
    y_ramp = np.linspace(0,20,10)
    z_ramp = np.linspace(10,5,10)
    GOALS = np.vstack((x_ramp,y_ramp,z_ramp)).T 
    # YAWS = [0] * len(GOALS)
    YAWS = [0,0,0, np.pi/4,np.pi/4, np.pi/2,np.pi/2, 0.7 * np.pi, 0.7 *np.pi, 0.7 * np.pi]
    start = GOALS[0]

    #-------------------------------
    #Regular input
    # GOALS = np.array([[0,0,5], [1,0,5], [0,1,5], [0,0,5], [1,1,5]])#, [0,0,5], [0,0,5]])
    # YAWS = [np.pi/4, np.pi/2, 0.6*np.pi,0.6*np.pi, 0]
    # start = [0,0,5]
    
    # GOALS = [(0,0,5), (0,0,6), (0,0,7), (0,0,4), (1,0,4), (2,0,4), (0,0,4), (-1,0,4), (-2,0,4), (0,0,4), (0,1,4), (0,2,4), (0,0,4), (0,-1,4), (0,-2,4),(0,0,4),(0,0,4),(0,0,4),(0,0,4),(0,0,4),(0,0,4),(1,1,4),(2,2,4),(0,0,4),(-1,-1,4), (-2,-2,4),(1,-1,4),(-1,1,4), (0,0,4)]
    # YAWS = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,np.pi/4,np.pi/2, 0.70 * np.pi,-np.pi/4,-np.pi/2,-0.70 * np.pi, 0, ]
    #----------------------------------
    #Randomized input
    # random.seed(1)
    # GOALS = list()
    # for hh in range(4,7):
    #     for mm in range(0,3):
    #         for nn in range(0,3):
    #             GOALS.append([mm,nn,hh])
    
    # random.shuffle(GOALS)
    # YAWS = np.linspace(0,np.pi/2,len(GOALS))
    # random.shuffle(YAWS)
    # YAWS = [0] * len(GOALS)
    # start = GOALS[0]
    #-------------------------------------
    # Define the quadcopters
    QUADCOPTER={'q1':{'position': start,'orientation':[0,0,0],'L':0.5,'r':0.2,'prop_size':[21,9.5],'weight':7}} #w in kg, L and r in mm, prop_size in in
    # Controller parameters
    CONTROLLER_PARAMETERS = {'Motor_limits':[1000, 45000],
                        'Tilt_limits':[-2, 2],   #degrees
                        'Yaw_Control_Limits':[-900,900],
                        'Z_XY_offset':500,
                        'Linear_PID':{'P':[600000,600000,72000],'I':[30,30,60],'D':[800000,800000,80000]},
                        'Linear_To_Angular_Scaler':[1,1,0],
                        'Yaw_Rate_Scaler':1.1,
                        'Angular_PID':{'P':[7500,6500,3000],'I':[0,0,0],'D':[2000,2000,1200]},
                        }

    # Catch Ctrl+C to stop threads
    signal.signal(signal.SIGINT, signal_handler)
    # Make objects for quadcopter, gui and controller
    quad = quadcopter.Quadcopter(QUADCOPTER)
    # gui_object = gui.GUI(quads=QUADCOPTER)
    ctrl = controller.Controller_PID_Point2Point(quad.get_state,quad.get_time,quad.set_motor_speeds, quad.get_L, params=CONTROLLER_PARAMETERS,quad_identifier='q1')
    
    # Start the threads
    quad.start_thread(dt=QUAD_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
    ctrl.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE,time_scaling=TIME_SCALING)
    
    # Update the GUI while switching between destination poitions
    # output_file_name = "outputs/" + map.split('/')[2] + "_path_data.csv"
    # output_file = open(output_file_name, 'w', buffering=65536)
    times = np.empty(0)
    input_goal = np.empty((0, 3), float)
    yaw_goal = np.empty(0)

    true_states = np.empty((0, 12), float)
    # est_states = np.empty((0,12), float)
    torques = np.empty((0,4), float)
    speeds = np.empty((0,6), float)

    simulation_start_time = quad.get_time()
    #Simulation
    time_limit = 20 *  TIME_SCALING#20  *  TIME_SCALING  #Amount of time limit to spend on a Goal            
    tolorance = 0.2                     #Steady state error

    for goal,yaw in zip(GOALS,YAWS):
        print(goal)
        ctrl.update_target(goal)
        ctrl.update_yaw_target(yaw)
        goal_start_time = quad.get_time()
        time_laps = 0
        true_pos = np.array(quad.get_state('q1')[0:3])
        dist = np.linalg.norm(true_pos - goal)
        

        while time_laps < time_limit:# and dist > tolorance: #
        
            # print("dist",dist)
            # print(t)
            
            # gui_object.quads['q1']['position'] = quad.get_position('q1')
            # gui_object.quads['q1']['orientation'] = quad.get_orientation('q1')
            # gui_object.update()

            true_state = np.array(quad.get_state('q1'))
            # est_state =  np.array(est.get_estimated_state('q1'))

            true_states = np.append(true_states, np.array([true_state]), axis=0)
            # est_states = np.append(est_states, np.array([est_state]), axis=0)
            torque = quad.get_tau()
            torques = np.append(torques, np.array([torque]), axis = 0)
            speeds = np.append(speeds, np.array([quad.get_motor_speeds('q1')]), axis = 0)
         
        

            time = quad.get_time()
            times = np.append(times, np.array([(time-simulation_start_time).total_seconds()]), axis=0)
            time_laps = (datetime.datetime.now()-goal_start_time).total_seconds() 
            

            dist = np.linalg.norm(true_state[0:3] - goal)
           

            input_goal = np.append(input_goal, np.array([goal]), axis=0)
            yaw_goal = np.append(yaw_goal, np.array([yaw]), axis=0)
            
    quad.stop_thread()
    ctrl.stop_thread()
    # est.stop_thread()

    # times = times / TIME_SCALING
    #Plot the path
    fig1, ax1 = plt.subplots(3,2,figsize=(10,  7))
    fig1.suptitle('x, y, z, roll, pitch, yaw', fontsize=16)
    # ax1[0,0].plot(times, est_states[:,0], label = "x dir_est", color = "green")
    ax1[0,0].plot(times, input_goal[:,0], label = "x goal")
    ax1[0,0].plot(times, true_states[:,0], label = "x dir")
    ax1[0,0].set_xlabel('time (s)')
    ax1[0,0].set_ylabel('x (m)')
    ax1[0,0].legend()
    
    # ax1[1,0].plot(times, est_states[:,1], label = "y dir_est" , color = "green")
    ax1[1,0].plot(times, input_goal[:,1], label = "y goal")
    ax1[1,0].plot(times, true_states[:,1], label = "y dir")
    ax1[1,0].set_xlabel('time (s)')
    ax1[1,0].set_ylabel('y (m)')
    ax1[1,0].legend()
    
    # ax1[2,0].plot(times, est_states[:,2], label = "z dir_est", color = "green")
    ax1[2,0].plot(times, input_goal[:,2], label = "z goal")
    ax1[2,0].plot(times, true_states[:,2], label = "z (altitude)")
    ax1[2,0].set_xlabel('time (s)')
    ax1[2,0].set_ylabel('z (m)')
    ax1[2,0].legend()

    # ax1[0,1].plot(times, np.degrees(est_states[:,6]), label = "roll_est", color = "green")
    ax1[0,1].plot(times, np.degrees(true_states[:,6]), label = "roll")
    ax1[0,1].set_xlabel('time (s)')
    ax1[0,1].set_ylabel('roll (deg)')
    ax1[0,1].legend()
    
    # ax1[1,1].plot(times, np.degrees(est_states[:,7]), label = "pitch_est", color = "green")
    ax1[1,1].plot(times, np.degrees(true_states[:,7]), label = "pitch")
    ax1[1,1].set_xlabel('time (s)')
    ax1[1,1].set_ylabel('pitch (deg)')
    ax1[1,1].legend()
    
    # ax1[2,1].plot(times, np.degrees(est_states[:,8]), label = "yaw_est", color = "green")
    ax1[2,1].plot(times, np.degrees(yaw_goal), label = "yaw goal")
    ax1[2,1].plot(times, np.degrees(true_states[:,8]), label = "yaw")
    ax1[2,1].set_xlabel('time (s)')
    ax1[2,1].set_ylabel('yaw (deg)')
    ax1[2,1].legend()
    plt.show()
    
    
    fig2, ax2 = plt.subplots(3,2,figsize=(10,  7))
    fig2.suptitle('v_x, v_y, v_z, roll_rate, pitch_rate, yaw_rate', fontsize=16)
    ax2[0,0].plot(times, true_states[:,3], label = "x_vel")
    # ax2[0,0].plot(times, est_states[:,3], label = "x_vel_est", color = "green")
    ax2[0,0].set_xlabel('time (s)')
    ax2[0,0].set_ylabel('v_x (m/s)')
    ax2[0,0].legend()
    
    ax2[1,0].plot(times, true_states[:,4], label = "y_vel")
    # ax2[1,0].plot(times, est_states[:,4], label = "y_vel_est", color = "green")
    ax2[1,0].set_xlabel('time (s)')
    ax2[1,0].set_ylabel('v_y (m/s)')
    ax2[1,0].legend()

    ax2[2,0].plot(times, true_states[:,5], label = "z_vel")
    # ax2[2,0].plot(times, est_states[:,5], label = "z_vel_est", color = "green")
    ax2[2,0].set_xlabel('time (s)')
    ax2[2,0].set_ylabel('v_z (m/s)')
    ax2[2,0].legend()

    ax2[0,1].plot(times, true_states[:,9], label = "roll_rate")
    # ax2[0,1].plot(times, est_states[:,9], label = "roll_rate_est", color = "green")
    ax2[0,1].set_xlabel('time (s)')
    ax2[0,1].set_ylabel('phi_rate (rad/s)')
    ax2[0,1].legend()
    
    ax2[1,1].plot(times, true_states[:,10], label = "pitch_rate")
    # ax2[1,1].plot(times, est_states[:,10], label = "pitch_rate_est", color = "green")
    ax2[1,1].set_xlabel('time (s)')
    ax2[1,1].set_ylabel('theta_rate (rad/s)')
    ax2[1,1].legend()

    ax2[2,1].plot(times, true_states[:,11], label = "yaw_rate_vel")
    # ax2[2,1].plot(times, est_states[:,11], label = "yaw_rate_est", color = "green")
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
        ax4[idx].plot(times, speeds[:,idx], label = "m{}".format(idx+1))
        ax4[idx].set_xlabel('time (s)')
        ax4[idx].set_ylabel('m {} (RPM)'.format(idx+1))
        ax4[idx].legend()




    # plt.show()

    


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
    ctrl1 = controller.Controller_PID_Point2Point(quad.get_state,quad.get_time,quad.set_motor_speeds,params=CONTROLLER_1_PARAMETERS,quad_identifier='q1')
    ctrl2 = controller.Controller_PID_Point2Point(quad.get_state,quad.get_time,quad.set_motor_speeds,params=CONTROLLER_2_PARAMETERS,quad_identifier='q2')
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
    ctrl = controller.Controller_PID_Velocity(quad.get_state,quad.get_time,quad.set_motor_speeds,params=CONTROLLER_PARAMETERS,quad_identifier='q1')
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
    Single_Point2Point()
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
