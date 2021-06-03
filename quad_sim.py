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
import cv2
import quadcopter
from Plot_results import plot_results, init_plot, plot_all_results
from path_planner.path_planner import get_test_paths

# Constants
TIME_SCALING = 1.0  # Any positive number(Smaller is faster). 1.0->Real Time, 0.0->Run as fast as possible
QUAD_DYNAMICS_UPDATE = 0.002  # seconds
CONTROLLER_DYNAMICS_UPDATE = 0.005  # seconds
ESTIMATION_TIME_UPDATE = 0.005
ESTIMATION_OBSERVATION_UPDATE = 0.1
PLOTTER_UPDATE = 1.0
run = True


def distance(x, y):
    return np.linalg.norm(x - y)

def calc_overshoot(line_start_pt, line_end_pt, est_pos):
    return np.linalg.norm(np.cross(line_start_pt-line_end_pt, est_pos-line_start_pt)/np.linalg.norm(line_start_pt-line_end_pt))


def Single_Point2Point(GOALS, goal_time_limit, tolerance, plt_show=False, venue_path=None):
    start = GOALS[0]
    YAWS = [0] * len(GOALS)

    # Define the quadcopters
    QUADCOPTER = {'q1': {'position': start, 'orientation': [0, 0, 0], 'L': 0.5, 'r': 0.2, 'prop_size': [21, 9.5],
                         'weight': 7}}  # w in kg, L and r in mm, prop_size in in

    # Controller parameters Without estimator
    # CONTROLLER_PARAMETERS = {'Motor_limits': [1000, 45000],
    #                          'Tilt_limits': [-20, 20],  # degrees
    #                          'Yaw_Control_Limits': [-900, 900],
    #                          'Z_XY_offset': 500,
    #                          'Linear_PID': {'P': [120000, 120000, 10000],
    #                                         'I': [0, 0, 0],
    #                                         'D': [100000, 100000, 2000]},
    #                          'Linear_To_Angular_Scaler': [1, 1, 0],
    #                          'Yaw_Rate_Scaler': 1.1,
    #                          'Angular_PID': {'P': [7000, 6500, 3000],
    #                                          'I': [0, 0, 0],
    #                                          'D': [3000, 3000, 1200]},
    #                          }

    # Controller parameters with the estimator
    CONTROLLER_PARAMETERS = {'Motor_limits': [1000, 45000],
                             'Tilt_limits': [-20, 20],  # degrees
                             'Yaw_Control_Limits': [-900, 900],
                             'Z_XY_offset': 500,
                             'Linear_PID': {'P': [120000, 120000, 15000],
                                            'I': [70, 80, 7],
                                            'D': [150000, 155000, 5500]},
                             'Linear_To_Angular_Scaler': [1, 1, 0],
                             'Yaw_Rate_Scaler': 1.1,
                             'Angular_PID': {'P': [7000, 6500, 3000],
                                             'I': [0, 0, 5],
                                             'D': [3000, 3000, 1200]},
                             }
    # CONTROLLER_PARAMETERS _if using estimator
    #  'Linear_PID':{'P':[1160000,1100000,30000],'I':[30,30,30],'D':[1500000,1500000,50000]},
    #  'Angular_PID':{'P':[8000,8000,3000],'I':[0,0,0],'D':[2100,2100,1500]},

    # Catch Ctrl+C to stop threads
    signal.signal(signal.SIGINT, signal_handler)

    # Make objects for quadcopter, gui and controller
    quad = quadcopter.Quadcopter(QUADCOPTER)
    est = estimator.EKF(quad.get_time, quad.get_position, quad.get_linear_rate, quad.get_linear_accelertaions,
                        quad.get_IMU_accelertaions, quad.get_orientation, quad.get_Gyro, quad.get_Magnetometer, quad.get_state,
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
    overshoots = np.empty((0, 1), dtype=float)

    simulation_start_time = quad.get_time()
    plt.ion()
    fig, axes, lines = init_plot(plt_show=plt_show)

    if venue_path is not None:
        map_image = cv2.imread(venue_path, cv2.IMREAD_UNCHANGED)

    # Simulation
    total_distance_travelled = 0
    PATH_LENGTH = len(GOALS)
    sim_start_time = datetime.datetime.now()
    for i in range(PATH_LENGTH):
        if i < PATH_LENGTH - 1:
            distance_to_go = distance(GOALS[i], GOALS[i + 1])
            total_distance_travelled += distance_to_go

        goal = GOALS[i]
        last_goal = GOALS[i] if i == 0 else GOALS[i - 1]
        next_goal = GOALS[i] if i == PATH_LENGTH - 1 else GOALS[i + 1]
        nextnext_goal = GOALS[i] if i == PATH_LENGTH - 1 else GOALS[i + 1]
        print("Goal:{0}, idx:{1}".format(goal, i))
        ctrl.update_target(goal)
        ctrl.update_yaw_target(0)
        goal_start_time = quad.get_time()
        time_lapse = 0
        true_pos = np.array(quad.get_state('q1')[0:3])
        est_pos = np.array(est.get_estimated_state('q1')[0:3])

        dist = distance(est_pos, goal)
        next_dist = distance(est_pos, next_goal)
        nextnext_dist = distance(est_pos, nextnext_goal)

        while not (dist < tolerance or
                   next_dist < tolerance or
                   nextnext_dist < 0.75 * tolerance):  # and not \
            # time_lapse > goal_time_limit:
             # print("dist",dist)
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
            input_goal = np.append(input_goal, np.array([goal]), axis=0)
            yaw_goal = np.append(yaw_goal, np.array([0]), axis=0)
            overshoots = np.append(overshoots, np.array([[calc_overshoot(last_goal, goal, est_state[0:3])]]), axis=0)

            time = quad.get_time()
            times = np.append(times, np.array([(time - simulation_start_time).total_seconds()]), axis=0)
            time_lapse = (datetime.datetime.now() - goal_start_time).total_seconds()

            dist = distance(est_state[0:3], goal)
            next_dist = distance(est_state[0:3], next_goal)
            nextnext_dist = distance(est_state[0:3], nextnext_goal)
            avg_velocity = total_distance_travelled / (datetime.datetime.now() - simulation_start_time).total_seconds()

            if venue_path is not None:
                new_image = cv2.circle(map_image.copy(), (int(true_state[0]), int(true_state[1])),
                                       3, (255, 0, 255, 255), -1)
                cv2.imshow('Real time path', new_image)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    print("Closing window but continuing with simulation...")
                    venue_path = None
                elif key == 27:  # Esc key
                    cv2.destroyAllWindows()
                    print('Stopping simulation.')
                    quad.stop_thread()
                    ctrl.stop_thread()
                    est.stop_thread()
                    exit(1)


            plot_results(fig, axes, lines, times, true_states, est_states, torques, speeds, accels, input_goal,
                         overshoots, avg_velocity, plt_pause=True)

        

    sim_end_time = datetime.datetime.now()
    sim_total_time = (sim_end_time - sim_start_time).total_seconds()
    print("Simulation took {}".format(sim_total_time))
    print("Travelled a total of {} meters".format(total_distance_travelled))

    quad.stop_thread()
    ctrl.stop_thread()
    est.stop_thread()
    # plotter_obj.stop_thread()

    error = true_states - est_states

    np.savetxt('error_analysis/true_data.txt', true_states)
    np.savetxt('error_analysis/est_data.txt', est_states)
    plt.ioff()
    plot_all_results(times, true_states, est_states, torques, speeds, accels, input_goal, overshoots, plt_show=True)


    return error


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
    venue_name = "Test"
    paths = get_test_paths(venue=venue_name)
    map_image_path = './path_planner/venues/' + venue_name + '/' + venue_name + 'DiffusedPath.png'

    for path_index, GOALS in paths.items():

        number_of_trials = 1
        goal_time_limit = 2  # Amount of time limit to spend on a Goal
        tolerance = 2  # Steady state error

        error = Single_Point2Point(GOALS=GOALS, goal_time_limit=goal_time_limit, tolerance=tolerance, plt_show=True, venue_path=map_image_path)
        np.savetxt('error_analysis/errors.txt', error)

    # for path_index, GOALS in paths.items():
    #     x = GOALS[0:-1, 0]
    #     y = GOALS[0:-1, 1]
    #     z = GOALS[0:-1, 2]
    #     input_range = np.arange(0, len(x))
    #     f_x = interpolate.interp1d(input_range, x)
    #     f_y = interpolate.interp1d(input_range, y)
    #     f_z = interpolate.interp1d(input_range, z)

    #     print("Goal shape is:", GOALS.shape )
    #     data_points = 160
    #     new_range = np.linspace(0, len(input_range) - 1, data_points)
    #     x_new = f_x(new_range)
    #     y_new = f_y(new_range)
    #     z_new = f_z(new_range)
    #     INTERPOLATED_GOALS = np.vstack((x_new, y_new, z_new)).T
    #     print("New Goal shape is:", INTERPOLATED_GOALS.shape)

    # for path_index, GOALS in paths.items():
    #     x = GOALS[0:-1, 0]
    #     y = GOALS[0:-1, 1]
    #     z = GOALS[0:-1, 2]
    #     input_range = np.arange(0, len(x))
    #     f_x = interpolate.interp1d(input_range, x)
    #     f_y = interpolate.interp1d(input_range, y)
    #     f_z = interpolate.interp1d(input_range, z)

    #     print("Goal shape is:", GOALS.shape )
    #     data_points = 160
    #     new_range = np.linspace(0, len(input_range) - 1, data_points)
    #     x_new = f_x(new_range)
    #     y_new = f_y(new_range)
    #     z_new = f_z(new_range)
    #     INTERPOLATED_GOALS = np.vstack((x_new, y_new, z_new)).T
    #     print("New Goal shape is:", INTERPOLATED_GOALS.shape)
