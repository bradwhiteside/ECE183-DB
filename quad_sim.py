import argparse
import datetime
import signal
import sys
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import controller
import estimator
import gui
import pycontroller
import quadcopter
from Plot_results import plot_results, init_plot
from path_planner.path_planner import get_test_paths

# Constants
TIME_SCALING = 1.0  # Any positive number(Smaller is faster). 1.0->Real Time, 0.0->Run as fast as possible
QUAD_DYNAMICS_UPDATE = 0.002  # seconds
CONTROLLER_DYNAMICS_UPDATE = 0.005  # seconds
ESTIMATION_TIME_UPDATE = 0.005
ESTIMATION_OBSERVATION_UPDATE = 0.1
PLOTTER_UPDATE = 1.0
run = True


def Single_Point2Point(GOALS, goal_time_limit, tolerance):
    start = GOALS[0]
    YAWS = [0] * len(GOALS)

    plt.ion()
    fig, axes, lines = init_plot(plt_show=True)

    # Define the quadcopters
    QUADCOPTER = {'q1': {'position': start, 'orientation': [0, 0, 0], 'L': 0.5, 'r': 0.2, 'prop_size': [21, 9.5],
                         'weight': 7}}  # w in kg, L and r in mm, prop_size in in

    # Controller parameters Without estimator
    CONTROLLER_PARAMETERS = {'Motor_limits': [1000, 45000],
                             'Tilt_limits': [-2, 2],  # degrees
                             'Yaw_Control_Limits': [-900, 900],
                             'Z_XY_offset': 500,
                             'Linear_PID': {'P': [500000, 550000, 72000], 'I': [30, 30, 60],
                                            'D': [1500000, 1200000, 80000]},
                             'Linear_To_Angular_Scaler': [1, 1, 0],
                             'Yaw_Rate_Scaler': 1.1,
                             'Angular_PID': {'P': [7000, 6500, 3000], 'I': [0, 0, 0], 'D': [3000, 3000, 1200]},
                             }
    # CONTROLLER_PARAMETERS _if using estimator
    #  'Linear_PID':{'P':[1160000,1100000,30000],'I':[30,30,30],'D':[1500000,1500000,50000]},
    #  'Angular_PID':{'P':[8000,8000,3000],'I':[0,0,0],'D':[2100,2100,1500]},

    # Catch Ctrl+C to stop threads
    signal.signal(signal.SIGINT, signal_handler)

    # Make objects for quadcopter, gui and controller
    quad = quadcopter.Quadcopter(QUADCOPTER)
    est = estimator.EKF(quad.get_time, quad.get_position, quad.get_linear_rate, quad.get_linear_accelertaions,
                        quad.get_IMU_accelertaions, quad.get_orientation, quad.get_Gyro, quad.get_state,
                        quad.get_motor_speeds, quad.get_covariances, quad.get_GPS, params=CONTROLLER_PARAMETERS,
                        quads=QUADCOPTER, quad_identifier='q1')
    ctrl = controller.Controller_PID_Point2Point(quad.get_state, quad.get_time, quad.set_motor_speeds,
                                                 est.get_estimated_state, quad.get_L, params=CONTROLLER_PARAMETERS,
                                                 quad_identifier='q1')
    # plotter_obj = Plotter.plotter(quad.get_time)
    # gui_object = gui.GUI(quads=QUADCOPTER)

    # Start the threads
    quad.start_thread(dt=QUAD_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)
    ctrl.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)
    est.start_thread(time_update_rate=ESTIMATION_TIME_UPDATE, observation_update_rate=ESTIMATION_OBSERVATION_UPDATE,
                     time_scaling=TIME_SCALING)
    # plotter_obj.start_thread(update_rate=PLOTTER_UPDATE, time_scaling=TIME_SCALING)

    # Update the GUI while switching between destination poitions
    # output_file_name = "outputs/" + map.split('/')[2] + "_path_data.csv"
    # output_file = open(output_file_name, 'w', buffering=65536)
    times = np.empty(0)
    input_goal = np.empty((0, 3), float)
    yaw_goal = np.empty(0)

    true_states = np.empty((0, 12), float)
    est_states = np.empty((0, 12), float)
    torques = np.empty((0, 4), float)
    speeds = np.empty((0, 6), float)
    accels = np.empty((0, 3), float)

    simulation_start_time = quad.get_time()
    # Simulation
    for goal, yaw in zip(GOALS, YAWS):
        print(goal)
        ctrl.update_target(goal)
        ctrl.update_yaw_target(yaw)
        goal_start_time = quad.get_time()
        time_laps = 0
        true_pos = np.array(quad.get_state('q1')[0:3])
        est_state = np.array(est.get_estimated_state('q1')[0:3])
        dist = np.linalg.norm(est_state - goal)

        while dist > tolerance or time_laps < goal_time_limit:
            # print("dist",dist)
            # print(t)
            # gui_object.quads['q1']['position'] = quad.get_position('q1')
            # gui_object.quads['q1']['orientation'] = quad.get_orientation('q1')
            # gui_object.update()

            true_state = np.array(quad.get_state('q1'))
            est_state = np.array(est.get_estimated_state('q1'))

            true_states = np.append(true_states, np.array([true_state]), axis=0)
            est_states = np.append(est_states, np.array([est_state]), axis=0)
            torque = quad.get_tau()
            torques = np.append(torques, np.array([torque]), axis=0)
            speeds = np.append(speeds, np.array([quad.get_motor_speeds('q1')]), axis=0)
            accels = np.append(accels, np.array([quad.get_linear_accelertaions('q1')]), axis=0)

            time = quad.get_time()
            times = np.append(times, np.array([(time - simulation_start_time).total_seconds()]), axis=0)
            time_laps = (datetime.datetime.now() - goal_start_time).total_seconds()

            # dist = np.linalg.norm(true_state[0:3] - goal)
            dist = np.linalg.norm(est_state[0:3] - goal)

            input_goal = np.append(input_goal, np.array([goal]), axis=0)
            yaw_goal = np.append(yaw_goal, np.array([yaw]), axis=0)
        plot_results(fig, axes, lines, times, true_states, est_states, torques, speeds, accels, input_goal, yaw_goal, plt_pause=True)

    quad.stop_thread()
    ctrl.stop_thread()
    est.stop_thread()
    # plotter_obj.stop_thread()

    error = true_states - est_states

    np.savetxt('error_analysis/true_data.txt', true_states)
    np.savetxt('error_analysis/est_data.txt', est_states)
    # plot_results(times, true_states, est_states, torques, speeds, accels, input_goal, yaw_goal, plt_pause=False)

    return error


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
    ctrl = pycontroller.Controller_PID_Velocity(quad.get_state, quad.get_time, quad.set_motor_speeds,
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
    return parser.parse_args()


def signal_handler(signal, frame):
    global run
    run = False
    print('Stopping')
    sys.exit(0)


if __name__ == "__main__":
    paths = get_test_paths(venue="Test")

    for path_index, GOALS in paths.items():
        x = GOALS[:, 0]
        y = GOALS[:, 1]
        z = GOALS[:, 2]
        input_range = np.arange(0, len(x))
        f_x = interpolate.interp1d(input_range, x)
        f_y = interpolate.interp1d(input_range, y)
        f_z = interpolate.interp1d(input_range, z)

        data_points = 400
        new_range = np.linspace(0, len(input_range) - 1, data_points)
        x_new = f_x(new_range)
        y_new = f_y(new_range)
        z_new = f_z(new_range)
        INTERPOLATED_GOALS = np.vstack((x_new, y_new, z_new)).T
        print("New Goal shape is:", INTERPOLATED_GOALS.shape)

        number_of_trials = 1
        goal_time_limit = 2  # Amount of time limit to spend on a Goal
        tolerance = 1.5  # Steady state error

        error = Single_Point2Point(GOALS=GOALS, goal_time_limit=goal_time_limit, tolerance=tolerance)
        # print("error shape is:", error.shape[0])

    # for i in range(0,number_of_trials-1):
    #     print("Trial", i)
    #     new_error = Single_Point2Point()
    #     error_len = error.shape[0]
    #     new_error_leng = new_error.shape[0]
    #     diff = np.abs(new_error_leng - error_len)

    #     if new_error_leng > error_len :
    #         # new_error = new_error[:-diff]
    #         # error += new_error #np.concatenate((error, new_error))
    #         np.concatenate((error, new_error))
    #         print('new_error shranked by {} elements'.format(diff))

    #     elif error_len > new_error_leng :
    #         # error = error[:-diff]
    #         # error += new_error
    #         np.concatenate((error, new_error))
    #         print('error shranked by {} elements'.format(diff))

    #     # print("Error length is:",new_error_leng)
    #     # error = np.concatenate((error, Single_Point2Point()))

    # #error = error/number_of_trials
    # print(error.shape)
    # np.savetxt('error_analysis/errors.txt', error)

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
