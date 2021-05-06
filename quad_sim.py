import quadcopter, gui, controller, estimator
import signal
import sys
import datetime
import argparse
import path_planner.path_planner as pp
import numpy as np
import matplotlib.pyplot as plt

# Constants
TIME_SCALING = 1.0 # Any positive number(Smaller is faster). 1.0->Real Time, 0.0->Run as fast as possible
QUAD_DYNAMICS_UPDATE = 0.002  # seconds
CONTROLLER_DYNAMICS_UPDATE = 0.005  # seconds
ESTIMATION_TIME_UPDATE = 0.005
ESTIMATION_OBSERVATION_UPDATE = 0.1
run = True


def Single_Point2Point(start, target, alt, map, DEBUG):
    # Set goals to go to
    a = pp.A_star(map, DEBUG)
    GOALS, path_cost = a.find_path(start, target, alt)

    # Define the quadcopters
    QUADCOPTER = {'q1': {'position': start, 'orientation': [0, 0, 0], 'L': 0.3, 'r': 0.1, 'prop_size': [10, 4.5],
                         'weight': 1.2}}
    # Controller parameters
    CONTROLLER_PARAMETERS = {'Motor_limits': [4000, 9000],
                             'Tilt_limits': [-10, 10],
                             'Yaw_Control_Limits': [-900, 900],
                             'Z_XY_offset': 500,
                             'Linear_PID': {'P': [800, 800, 7000], 'I': [0, 0, 4.5], 'D': [7000, 7000, 5000]},
                             'Linear_To_Angular_Scaler': [1, 1, 0],
                             'Yaw_Rate_Scaler': 0.18,
                             'Angular_PID': {'P': [22000, 22000, 1500], 'I': [0, 0, 1.2], 'D': [32000, 32000, 0]},
                             }

    # Catch Ctrl+C to stop threads
    signal.signal(signal.SIGINT, signal_handler)
    # Make objects for quadcopter, gui and controller
    quad = quadcopter.Quadcopter(QUADCOPTER)
    # gui_object = gui.GUI(quads=QUADCOPTER)

    #Initiate the Estimator and the Controller
    est = estimator.EKF(quad.get_time, quad.get_position,quad.get_linear_rate, quad.get_linear_accelertaions, quad.get_IMU_accelertaions, quad.get_orientation, quad.get_Gyro, quad.get_state, quad.get_motor_speeds,quad.get_covariances, quad.get_GPS, params=CONTROLLER_PARAMETERS, quads=QUADCOPTER, quad_identifier='q1')
    ctrl = controller.Controller_PID_Point2Point(quad.get_state, quad.get_time, quad.set_motor_speeds, est.get_estimated_state, params=CONTROLLER_PARAMETERS, quad_identifier='q1')
    
    # Start the threads
    quad.start_thread(dt=QUAD_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)
    start_time = datetime.datetime.now()
    ctrl.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)
    est.start_thread(time_update_rate=ESTIMATION_TIME_UPDATE, observation_update_rate = ESTIMATION_OBSERVATION_UPDATE ,time_scaling=TIME_SCALING)
    
    # Update the GUI while switching between destination poitions
    output_file_name = "outputs/" + map.split('/')[2] + "_path_data.csv"
    output_file = open(output_file_name, 'w', buffering=65536)
    path_taken = np.empty((0, 3), float)
    estimated_path = np.empty((0, 3), float)
    path_error = np.empty((0, 3), float)
    orientation_errors = np.empty((0, 3), float)
    true_orientations = np.empty((0, 3), float) #Roll, pith, yaw
    true_angular_rates = np.empty((0, 3), float) #Roll, pith, yaw rates

    #Simulation
    time_limit = 50  *  TIME_SCALING  #Amount of time limit to spend on a Goal            
    tolorance = 2                     #Steady state error

    for goal in GOALS:
        print(goal)
        ctrl.update_target(goal)
        goal_start_time = datetime.datetime.now()
        dt = 0
        est_pos = np.array(est.get_estimated_state('q1')[0:3])
        dist = np.linalg.norm(est_pos - goal)
        while dt < time_limit and dist > tolorance:
        
            # print("dist",dist)
            t = (quad.get_time() - start_time).total_seconds()
            # print(t)
            dt = (datetime.datetime.now()-goal_start_time).total_seconds()
            
            true_state = np.array(quad.get_state('q1'))
            est_state =  np.array(est.get_estimated_state('q1'))
            true_pos, true_velocity, true_orientation, true_angular_rate = true_state[0:3], true_state[3:6],true_state[6:9],true_state[9:12]
            est_pos, est_velocity, est_orientation, est_angular_rate = est_state[0:3], est_state[3:6], est_state[6:9], est_state[9:12]  #np.array(est.get_estimated_state('q1')[0:3])
            
           
            dist = np.linalg.norm(est_pos - goal)
            pos_error = true_pos - est_pos
            orientation_error = true_orientation - est_orientation

            path_taken = np.append(path_taken, np.array([true_pos]), axis=0)
            estimated_path = np.append(estimated_path, np.array([est_pos]), axis=0)
            path_error = np.append(path_error, np.array([pos_error]), axis=0)
            orientation_errors = np.append(orientation_errors, np.array([orientation_error]), axis=0)

            
            true_orientations = np.append(true_orientations, np.array([true_orientation]), axis=0)
            true_angular_rates = np.append(true_angular_rates, np.array([true_angular_rate]), axis=0)

            output_str = '{:.3f}'.format(t)
            for e in est_pos: output_str += ' {:.3f}'.format(e)
            for e in pos_error: output_str += ' {:.3f}'.format(e)
            output_str += '\n'
            output_file.write(output_str)

            # gui_object.quads['q1']['position'] = pos
            # gui_object.quads['q1']['orientation'] = quad.get_orientation('q1')
            # gui_object.update()
            # print("Time: {}\tGoal is {}\tCur pos is {}\tEst pos is {}\t Dist = {}".format(t, goal, pos, est_pos, dist))

            # if dist < 2:
            #     break

        output_file.flush()

    quad.stop_thread()
    ctrl.stop_thread()
    est.stop_thread()

    #Plot the path
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.invert_yaxis()
    ax1.set_xticks(range(256)[::32])
    ax1.set_yticks(range(256)[::-32])
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    img1 = pp.draw_path(map, GOALS, (255, 255, 0, 255))  # yellow
    ax1.imshow(img1, extent=[0, 256, 0, 256])
    plt.savefig("outputs/" + map.split('/')[2] + "_computed_path.jpg")
    img2 = pp.draw_path(img1, estimated_path, (255, 0, 255, 255))  # pink
    ax1.imshow(img2, extent=[0, 256, 0, 256])
    plt.savefig("outputs/" + map.split('/')[2] + "_path_taken.jpg")
    
    #Pos Error plot
    fig2, axs2 = plt.subplots(3, 1, figsize=(12, 8))
    fig2.suptitle('XYZ Estimation Errors')
    time = np.linspace(0, t, len(path_error))
    axs2[0].plot(time, path_error[:, 0])
    axs2[0].title.set_text("X error")
    axs2[0].set_xlabel("t (s)")
    axs2[0].set_ylabel("x (m)")
    axs2[1].plot(time, path_error[:, 1])
    axs2[1].title.set_text("Y error")
    axs2[1].set_xlabel("t (s)")
    axs2[1].set_ylabel("y (m)")
    axs2[2].plot(time, path_error[:, 2])
    axs2[2].title.set_text("Z error")
    axs2[2].set_xlabel("t (s)")
    axs2[2].set_ylabel("z (m)")
    plt.subplots_adjust(hspace=0.4, bottom=0.07, left=0.095, right=0.95)
    plt.savefig("outputs/" + map.split('/')[2] + "_error.jpg")
    # plt.show()

    #Estimation plots
    fig3, axs3 = plt.subplots(3, 1, figsize=(12, 8))
    fig3.suptitle('Absolute XYZ State Estimations')
    time = np.linspace(0, t, len(estimated_path))
    axs3[0].plot(time, estimated_path[:, 0])
    axs3[0].title.set_text("X")
    axs3[0].set_xlabel("t (s)")
    axs3[0].set_ylabel("x (m)")
    axs3[1].plot(time, estimated_path[:, 1])
    axs3[1].title.set_text("Y")
    axs3[1].set_xlabel("t (s)")
    axs3[1].set_ylabel("y (m)")
    axs3[2].plot(time, estimated_path[:, 2])
    axs3[2].title.set_text("Z")
    axs3[2].set_xlabel("t (s)")
    axs3[2].set_ylabel("z (m)")
    axs3[2].set_ylim([0, 4])
    plt.subplots_adjust(hspace=0.4, bottom=0.07, left=0.095, right=0.95)
    plt.savefig("outputs/" + map.split('/')[2] + "_est.jpg")
    # plt.show()




    #True Trajectory Plots
    fig4, axs4 = plt.subplots(3, 1, figsize=(12, 8))
    fig4.suptitle('Absolute XYZ True States')
    time = np.linspace(0, t, len(path_taken))
    axs4[0].plot(time, path_taken[:, 0])
    axs4[0].title.set_text("X")
    axs4[0].set_xlabel("t (s)")
    axs4[0].set_ylabel("x (m)")
    axs4[1].plot(time, path_taken[:, 1])
    axs4[1].title.set_text("Y")
    axs4[1].set_xlabel("t (s)")
    axs4[1].set_ylabel("y (m)")
    axs4[2].plot(time, path_taken[:, 2])
    axs4[2].title.set_text("Z")
    axs4[2].set_xlabel("t (s)")
    axs4[2].set_ylabel("z (m)")
    axs4[2].set_ylim([0, 4])
    plt.subplots_adjust(hspace=0.4, bottom=0.07, left=0.095, right=0.95)
    plt.savefig("outputs/" + map.split('/')[2] + "_true.jpg")
    # plt.show()

    #True Angular Rate Plots
    fig5, axs5 = plt.subplots(3, 1, figsize=(12, 8))
    fig5.suptitle('True Roll, Pitch, Yaw Rates')
    time = np.linspace(0, t, len(true_angular_rates))
    axs5[0].plot(time, true_angular_rates[:, 0])
    axs5[0].title.set_text("theta_dot")
    axs5[0].set_xlabel("t (s)")
    axs5[0].set_ylabel("roll_rate (rad/s)")
    axs5[1].plot(time, true_angular_rates[:, 1])
    axs5[1].title.set_text("phi_dot")
    axs5[1].set_xlabel("t (s)")
    axs5[1].set_ylabel("pitch_rate (rad/s)")
    axs5[2].plot(time, true_angular_rates[:, 2])
    axs5[2].title.set_text("gamma_dot")
    axs5[2].set_xlabel("t (s)")
    axs5[2].set_ylabel("yaw_rate(rad/s)")
    # axs5[2].set_ylim([0, 4])
    plt.subplots_adjust(hspace=0.4, bottom=0.07, left=0.095, right=0.95)
    plt.savefig("outputs/" + map.split('/')[2] + "_true_angular_rates.jpg")
    # plt.show()

    #True Angular Rate Plots
    fig6, axs6 = plt.subplots(3, 1, figsize=(12, 8))
    fig6.suptitle('True Roll, Pitch, Yaw States')
    time = np.linspace(0, t, len(true_orientations))
    axs6[0].plot(time, true_orientations[:, 0])
    axs6[0].title.set_text("Roll Angle")
    axs6[0].set_xlabel("t (s)")
    axs6[0].set_ylabel("roll (rad)")
    axs6[1].plot(time, true_orientations[:, 1])
    axs6[1].title.set_text("Pitch Angle")
    axs6[1].set_xlabel("t (s)")
    axs6[1].set_ylabel("pitch (rad)")
    axs6[2].plot(time, true_orientations[:, 2])
    axs6[2].title.set_text("Yaw Angle")
    axs6[2].set_xlabel("t (s)")
    axs6[2].set_ylabel("yaw(rad)")
    # axs5[2].set_ylim([0, 4])
    plt.subplots_adjust(hspace=0.4, bottom=0.07, left=0.095, right=0.95)
    plt.savefig("outputs/" + map.split('/')[2] + "_true_orientations.jpg")
    # plt.show()


    #Orientation Error plot
    fig7, axs7 = plt.subplots(3, 1, figsize=(12, 8))
    fig7.suptitle('Orientation Errors')
    time = np.linspace(0, t, len(path_error))
    axs7[0].plot(time, np.degrees(orientation_errors[:, 0]))
    axs7[0].title.set_text("Roll Error")
    axs7[0].set_xlabel("t (s)")
    axs7[0].set_ylabel("Roll (rad)")
    axs7[1].plot(time, np.degrees(orientation_errors[:, 1]))
    axs7[1].title.set_text("Pitch Error")
    axs7[1].set_xlabel("t (s)")
    axs7[1].set_ylabel("Pitch (rad)")
    axs7[2].plot(time, np.degrees(orientation_errors[:, 2]))
    axs7[2].title.set_text("Yaw Error")
    axs7[2].set_xlabel("t (s)")
    axs7[2].set_ylabel("Yaw (rad)")
    plt.subplots_adjust(hspace=0.4, bottom=0.07, left=0.095, right=0.95)
    plt.savefig("outputs/" + map.split('/')[2] + "_orinetation_error.jpg")
    # plt.show()

    output_file.close()
    


def Multi_Point2Point():
    # Set goals to go to
    GOALS_1 = [(-1, -1, 4), (1, 1, 2)]
    GOALS_2 = [(1, -1, 2), (-1, 1, 4)]
    # Define the quadcopters
    QUADCOPTERS = {'q1': {'position': [1, 0, 4], 'orientation': [0, 0, 0], 'L': 0.3, 'r': 0.1, 'prop_size': [10, 4.5],
                          'weight': 1.2},
                   'q2': {'position': [-1, 0, 4], 'orientation': [0, 0, 0], 'L': 0.15, 'r': 0.05, 'prop_size': [6, 4.5],
                          'weight': 0.7}}
    # Controller parameters
    CONTROLLER_1_PARAMETERS = {'Motor_limits': [4000, 9000],
                               'Tilt_limits': [-10, 10],
                               'Yaw_Control_Limits': [-900, 900],
                               'Z_XY_offset': 500,
                               'Linear_PID': {'P': [300, 300, 7000], 'I': [0.04, 0.04, 4.5], 'D': [450, 450, 5000]},
                               'Linear_To_Angular_Scaler': [1, 1, 0],
                               'Yaw_Rate_Scaler': 0.18,
                               'Angular_PID': {'P': [22000, 22000, 1500], 'I': [0, 0, 1.2], 'D': [12000, 12000, 0]},
                               }
    CONTROLLER_2_PARAMETERS = {'Motor_limits': [4000, 9000],
                               'Tilt_limits': [-10, 10],
                               'Yaw_Control_Limits': [-900, 900],
                               'Z_XY_offset': 500,
                               'Linear_PID': {'P': [300, 300, 7000], 'I': [0.04, 0.04, 4.5], 'D': [450, 450, 5000]},
                               'Linear_To_Angular_Scaler': [1, 1, 0],
                               'Yaw_Rate_Scaler': 0.18,
                               'Angular_PID': {'P': [22000, 22000, 1500], 'I': [0, 0, 1.2], 'D': [12000, 12000, 0]},
                               }

    # Catch Ctrl+C to stop threads
    signal.signal(signal.SIGINT, signal_handler)
    # Make objects for quadcopter, gui and controllers
    gui_object = gui.GUI(quads=QUADCOPTERS)
    quad = quadcopter.Quadcopter(quads=QUADCOPTERS)
    ctrl1 = controller.Controller_PID_Point2Point(quad.get_state, quad.get_time, quad.set_motor_speeds,
                                                  params=CONTROLLER_1_PARAMETERS, quad_identifier='q1')
    ctrl2 = controller.Controller_PID_Point2Point(quad.get_state, quad.get_time, quad.set_motor_speeds,
                                                  params=CONTROLLER_2_PARAMETERS, quad_identifier='q2')
    # Start the threads
    quad.start_thread(dt=QUAD_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)
    ctrl1.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)
    ctrl2.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)
    # Update the GUI while switching between destination poitions
    while (run == True):
        for goal1, goal2 in zip(GOALS_1, GOALS_2):
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
    GOALS = [(0.5, 0, 2), (0, 0.5, 2), (-0.5, 0, 2), (0, -0.5, 2)]
    # Define the quadcopters
    QUADCOPTER = {'q1': {'position': [0, 0, 0], 'orientation': [0, 0, 0], 'L': 0.3, 'r': 0.1, 'prop_size': [10, 4.5],
                         'weight': 1.2}}
    # Controller parameters
    CONTROLLER_PARAMETERS = {'Motor_limits': [4000, 9000],
                             'Tilt_limits': [-10, 10],
                             'Yaw_Control_Limits': [-900, 900],
                             'Z_XY_offset': 500,
                             'Linear_PID': {'P': [2000, 2000, 7000], 'I': [0.25, 0.25, 4.5], 'D': [50, 50, 5000]},
                             'Linear_To_Angular_Scaler': [1, 1, 0],
                             'Yaw_Rate_Scaler': 0.18,
                             'Angular_PID': {'P': [22000, 22000, 1500], 'I': [0, 0, 1.2], 'D': [12000, 12000, 0]},
                             }

    # Catch Ctrl+C to stop threads
    signal.signal(signal.SIGINT, signal_handler)
    # Make objects for quadcopter, gui and controller
    quad = quadcopter.Quadcopter(QUADCOPTER)
    gui_object = gui.GUI(quads=QUADCOPTER)
    ctrl = controller.Controller_PID_Velocity(quad.get_state, quad.get_time, quad.set_motor_speeds,
                                              params=CONTROLLER_PARAMETERS, quad_identifier='q1')
    # Start the threads
    quad.start_thread(dt=QUAD_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)
    ctrl.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)
    # Update the GUI while switching between destination poitions
    while (run == True):
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
    parser.add_argument("--time_scale", type=float, default=-1.0,
                        help='Time scaling factor. 0.0:fastest,1.0:realtime,>1:slow, ex: --time_scale 0.1')
    parser.add_argument("--quad_update_time", type=float, default=0.0,
                        help='delta time for quadcopter dynamics update(seconds), ex: --quad_update_time 0.002')
    parser.add_argument("--controller_update_time", type=float, default=0.0,
                        help='delta time for controller update(seconds), ex: --controller_update_time 0.005')
    parser.add_argument("--start", help='Starting point for simulation\n usafe=%(prog)s --sim [sim_arg] --start X1 Y1 Z1 --target X2 Y2 Z2',
                        required='--sim' in sys.argv, nargs=3, type=int)
    parser.add_argument("--target", help='Target point for simulation\n usage=%(prog)s --sim [sim_arg] --start X1 Y1 Z1 --target X2 Y2 Z2',
                        required='--sim' in sys.argv, nargs=3, type=int)
    parser.add_argument("--map", help='Name of map image for the path planner\n usage=%(prog)s [options] --map venues/Coachella/CoachellaMap.png',
                        required='--sim' in sys.argv, type=str)
    parser.add_argument("--alt", help='Relative altitude for simulation', type=int, default=10)
    parser.add_argument("--DEBUG", action='store_true')

    return parser.parse_args()


def signal_handler(signal, frame):
    global run
    run = False
    print('Stopping')
    sys.exit(0)


if __name__ == "__main__":
    args = parse_args()
    if args.time_scale >= 0: TIME_SCALING = args.time_scale
    if args.quad_update_time > 0: QUAD_DYNAMICS_UPDATE = args.quad_update_time
    if args.controller_update_time > 0: CONTROLLER_DYNAMICS_UPDATE = args.controller_update_time
    if args.sim == 'single_p2p':
        if args.map[0] == 'R':
            map = "path_planner/venues/RoseBowl/RoseBowlMap.png"
        elif args.map[0] == 'C':
            map = "path_planner/venues/Coachella/CoachellaMap.png"
        else:
            map = "path_planner/venues/Test/Test.png"
        Single_Point2Point(args.start, args.target, args.alt, map, args.DEBUG)
    elif args.sim == 'multi_p2p':
        Multi_Point2Point()
    elif args.sim == 'single_velocity':
        Single_Velocity()
