import numpy as np
import RobotDART as rd
import dartpy
import copy
from utils import angle_wrap_multi

# Class for PI controller in joint-space (works for 1-D or N-D vectors)
class PIJoint:
    def __init__(self, target, dt, Kp = 10., Ki = 0.1):
        self._target = target
        self._dt = dt
        self._Kp = Kp
        self._Ki = Ki
        self._sum_error = 0.

    def set_target(self, target):
        self._target = target
    def update(self, current):
        #print(self._target)
        #print(current)
        error = angle_wrap_multi(self._target - current)
        self._sum_error = self._sum_error + error * self._dt


        return self._Kp * error + self._Ki * self._sum_error


# Load simple arm
packages = [("franka_description", "franka/franka_description")]
robot = rd.Robot("franka/franka.urdf", packages)
robot.set_color_mode("material")
# fix to world and minor things
robot.fix_to_world()
robot.set_position_enforced(True)
# select servo actuators --> velocity commands
robot.set_actuator_types("servo")

# Create simulator object
dt = 0.001
simu = rd.RobotDARTSimu(dt)
simu.set_collision_detector("fcl")

# Create Graphics
gconfig = rd.gui.GraphicsConfiguration(1024, 768)
graphics = rd.gui.Graphics(gconfig) 
simu.set_graphics(graphics)
graphics.look_at([3., 1., 2.], [0., 0., 0.])

# Add robot and nice floor
simu.add_robot(robot)
simu.add_checkerboard_floor()

Kp = 10
Ki = 0.1

# Multi-joint controller
target_positions = copy.copy(robot.positions())
target_positions[5] = np.pi / 2.0
target_positions[7] = 0.04
print("Target:", target_positions)
#print(robot.positions())

robot.reset()
robot.set_positions(robot.positions()+np.random.rand(robot.num_dofs())*np.pi/1.5-np.pi/3.)

controller = PIJoint(target_positions, dt, Kp, Ki)

# Run simulation
while True:
    if simu.step_world():
        break
    cmd = controller.update(robot.positions())
    robot.set_commands(cmd)

    print("PI Result:", robot.positions())

