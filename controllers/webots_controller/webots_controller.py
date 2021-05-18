"""flight_controller controller."""

from controller import Robot, Supervisor
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pycontroller as ctrl
import numpy as np

MAX_SPEED = 310.4
mult = 1
odd = -mult * MAX_SPEED
even = mult * MAX_SPEED

super = Supervisor()
drone = super.getFromDef('drone')

prop1 = super.getDevice("prop1")
prop2 = super.getDevice("prop2")
prop3 = super.getDevice("prop3")
prop4 = super.getDevice("prop4")
prop5 = super.getDevice("prop5")
prop6 = super.getDevice("prop6")
gyro = super.getDevice('GYRO')
gps = super.getDevice('GPS')
compass = super.getDevice('COMPASS')
prop1.setPosition(float('+inf'))
prop2.setPosition(float('+inf'))
prop3.setPosition(float('+inf'))
prop4.setPosition(float('+inf'))
prop5.setPosition(float('+inf'))
prop6.setPosition(float('+inf'))

timestep = int(super.getBasicTimeStep())
# drone.enable(timestep)

CONTROLLER_PARAMETERS = {'Motor_limits': [0, MAX_SPEED],
                         'Tilt_limits': [-2, 2],  # degrees
                         'Yaw_Control_Limits': [-900, 900],
                         'Z_XY_offset': 500,
                         'Linear_PID': {'P': [100000, 100000, 20000], 'I': [20, 20, 20], 'D': [200000, 200000, 12000]},
                         'Linear_To_Angular_Scaler': [1, 1, 0],
                         'Yaw_Rate_Scaler': 1,
                         'Angular_PID': {'P': [7000, 6000, 0.1], 'I': [0, 0, 0], 'D': [2000, 2000, 0.01]},
                         }


def actuate_motors(str, M):
    prop1.setVelocity(-M[0])
    prop2.setVelocity(M[1])
    prop3.setVelocity(-M[2])
    prop4.setVelocity(M[3])
    prop5.setVelocity(-M[4])
    prop6.setVelocity(M[5])


def get_time():
    return super.getTime()


def convert_orientation_matrix(m):
    sy = m[1, 0];
    cy = m[0, 0]
    yaw = np.arctan2(sy, cy)

    sp = -m[2, 0];
    cp = np.sqrt(m[2, 1] ** 2 + m[2, 2] ** 2)
    pitch = np.arctan2(sp, cp)

    sr = m[2, 1];
    cr = m[2, 2]
    roll = np.arctan2(sr, cr)

    return np.array([roll, pitch, yaw])


def get_L():
    return 0.5


def get_state(str):
    pos = np.array(drone.getPosition())
    lin_v = np.array(drone.getVelocity()[:3])
    o = np.array(drone.getOrientation()).reshape((3, 3))
    ang = convert_orientation_matrix(o.T)
    ang_v = np.array(drone.getVelocity()[3:])
    return np.array([pos, lin_v, ang, ang_v]).flatten()


#fc = ctrl.Controller_PID_Point2Point(get_state, get_time, actuate_motors, get_L, CONTROLLER_PARAMETERS, 'q1')
#fc.start_thread(update_rate=timestep, time_scaling=1)

def reached_goal(cur, goal):
    return np.linalg.norm(cur - goal) < 1


path = [(0, 0, 0), (0, 3, 0), (3, 3, 0), (0, 3, 3)]
path = np.array(path).reshape((4, 3))

actuate_motors('', np.full(6, even))

goal_index = 0
num_goals = path.shape[0]
while super.step(timestep) != -1 and goal_index < num_goals:
    pass

# Enter here exit cleanup code.
