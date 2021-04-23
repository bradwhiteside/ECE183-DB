import quadcopter, gui, controller, estimator
import signal
import sys
import datetime
import argparse
import path_planner.path_planner as pp
import numpy as np
import matplotlib.pyplot as plt

# Constants
TIME_SCALING = 1.0  # Any positive number(Smaller is faster). 1.0->Real Time, 0.0->Run as fast as possible
QUAD_DYNAMICS_UPDATE = 0.002  # seconds
CONTROLLER_DYNAMICS_UPDATE = 0.005  # seconds
ESTIMATION_TIME_UPDATE = 0.005
ESTIMATION_OBSERVATION_UPDATE = 0.01
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
                             'Angular_PID': {'P': [22000, 22000, 1500], 'I': [0, 0, 1.2], 'D': [12000, 12000, 0]},
                             }

    # Catch Ctrl+C to stop threads
    signal.signal(signal.SIGINT, signal_handler)
    # Make objects for quadcopter, gui and controller
    quad = quadcopter.Quadcopter(QUADCOPTER)
    # gui_object = gui.GUI(quads=QUADCOPTER)
    est = estimator.EKF(quad.get_time, quad.get_position,quad.get_linear_rate, quad.get_linear_accelertaions, quad.get_IMU_accelertaions, quad.get_orientation, quad.get_Gyro, quad.get_state, quad.get_motor_speeds,quad.get_covariances, quad.get_GPS, params=CONTROLLER_PARAMETERS, quads=QUADCOPTER, quad_identifier='q1')
    ctrl = controller.Controller_PID_Point2Point(quad.get_state, quad.get_time, quad.set_motor_speeds,
                                                 params=CONTROLLER_PARAMETERS, quad_identifier='q1')
    # Start the threads
    quad.start_thread(dt=QUAD_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)
    start_time = datetime.datetime.now()
    ctrl.start_thread(update_rate=CONTROLLER_DYNAMICS_UPDATE, time_scaling=TIME_SCALING)
    est.start_thread(time_update_rate=ESTIMATION_TIME_UPDATE, observation_update_rate = ESTIMATION_OBSERVATION_UPDATE ,time_scaling=TIME_SCALING)
    # Update the GUI while switching between destination poitions
    output_file_name = "outputs/" + map.split('/')[2] + "_path_data.csv"
    output_file = open(output_file_name, 'w', buffering=65536)
    path_taken = []
    estimated_path =[]
    t = 0
    limit = 40
    while (run == True and t < limit):
        for goal in GOALS:
            ctrl.update_target(goal)
            while run == True and t < limit:
                pos = quad.get_position('q1')
                est_pos = est.get_estimated_state()[0:3]
                print(pos-est_pos)
                # gui_object.quads['q1']['position'] = pos
                # gui_object.quads['q1']['orientation'] = quad.get_orientation('q1')
                # gui_object.update()
                dist = np.linalg.norm(np.array(pos) - goal)
                t = (quad.get_time()-start_time).total_seconds()
                path_taken.append(pos)
                estimated_path.append(est_pos)
                output_str = '{} {} {} {}\n'.format(t, pos[0], pos[1], pos[2])
                output_file.write(output_str)
                # print("Time: {}\tGoal is {}\tCurrent position is {}\t Dist = {}".format(t, goal, pos, dist))
                if dist < 2:
                    break
            output_file.flush()

    # print("true path len",len(path_taken))
    # print("est pth len",len(estimated_path))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.invert_yaxis()
    ax.set_xticks(range(256)[::16])
    ax.set_yticks(range(256)[::16])
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    img1 = pp.draw_path(map, GOALS, (255, 255, 0, 255))  # yellow
    img2 = pp.draw_path(img1, estimated_path, (255, 0, 255, 255), True)  # pink
    # img3 = pp.draw_path(img1, estimated_path, (250, 0, 250, 250), True)
    ax.imshow(img2, extent=[0, 256, 0, 256])

    plt.show()

    output_file.close()
    quad.stop_thread()
    ctrl.stop_thread()
    est.stop_thread()


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
