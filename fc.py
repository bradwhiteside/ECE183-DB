#!/usr/bin/python
import quadcopter
import controller_sim as controller
from controller import Robot, Supervisor

import signal
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random 
from quad_sim import Single_Point2Point
# Constants
TIME_SCALING = 1 # Any positive number(Smaller is faster). 1.0->Real Time, 0.0->Run as fast as possible
QUAD_DYNAMICS_UPDATE = 0.002 # seconds
CONTROLLER_DYNAMICS_UPDATE = 0.005 # seconds


if __name__ == "__main__":
    MAX_SPEED = 310.4
    # robot = Robot()
    robot = 1
    super = Supervisor()
    print("Hello")
    Single_Point2Point(robot, super)
   