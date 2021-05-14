"""flight_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot

MAX_SPEED = 310.4
mult = 1
odd = -mult * MAX_SPEED
even = mult * MAX_SPEED

# create the Robot instance.
robot = Robot()
prop1 = robot.getDevice("prop1")
prop2 = robot.getDevice("prop2")
prop3 = robot.getDevice("prop3")
prop4 = robot.getDevice("prop4")
prop5 = robot.getDevice("prop5")
prop6 = robot.getDevice("prop6")
prop1.setPosition(float('+inf'))
prop2.setPosition(float('+inf'))
prop3.setPosition(float('+inf'))
prop4.setPosition(float('+inf'))
prop5.setPosition(float('+inf'))
prop6.setPosition(float('+inf'))
prop1.setVelocity(odd)
prop2.setVelocity(even)
prop3.setVelocity(odd)
prop4.setVelocity(even)
prop5.setVelocity(odd)
prop6.setVelocity(even)


# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.