import numpy as np
import RobotDART as rd
import dartpy
import copy
from utils import damped_pseudoinverse, AdT

class PITask:
    def __init__(self, target, dt, Kp = 10., Ki = 0.1, Kd = 0.1):
        self._target = target
        self._dt = dt
        self._Kp = Kp
        self._Ki = Ki
        self._Kd = Kd
        self._sum_error = 0
        self._present_t = 0
        self._previous_error = 1000 # Sto arxiko arxiko sfalma ebala mia megalh timh gia na sumbolisw to sigouro megalo sfalma.

    def set_target(self, target):
        self._target = target
    
    # function to compute error
    def error(self, tf):
        return rd.math.logMap(tf.inverse().multiply(self._target))
    
    def update(self, current):
        Ad_tf = AdT(current)
        error_in_body_frame = self.error(current)
        error_in_world_frame = Ad_tf @ error_in_body_frame

        self._sum_error = self._sum_error + error_in_world_frame * self._dt

        self._present_t = self._present_t + self._dt #To Orisa gia na upologizw ton sunoliko xrono.

        paragwgos = (self._previous_error - error_in_world_frame) / self._dt

        self._previous_error=error_in_world_frame

        return self._Kp * error_in_world_frame + self._Ki * self._sum_error + self._Kd * paragwgos

# Load a robot
packages = [("franka_description", "franka/franka_description")]
robot = rd.Robot("franka/franka.urdf", packages)
robot.set_color_mode("material")

# fix to world and minor things
robot.fix_to_world()
robot.set_position_enforced(True)
robot.set_actuator_types("servo")


#robot.set_actuator_types(float std::string& locked, float std::string& panda_finger_joint2, bool override_mimic = True, bool override_base = False)

# set initial joint positions
positions = robot.positions()
positions[5] = np.pi / 2.0
positions[7] = 0.04
robot.set_positions(positions)


# get end-effector pose
eef_link_name = "panda_link8"
tf_desired = robot.body_pose(eef_link_name)

# set robot back to zero positions
robot.reset()
robot.set_positions(robot.positions()+np.random.rand(robot.num_dofs())*np.pi/1.5-np.pi/3.)

# Create simulator object
dt = 0.001
simu = rd.RobotDARTSimu(dt)
# we need the FCL collision detector for mesh collisions
simu.set_collision_detector("fcl")

# Create Graphics
gconfig = rd.gui.GraphicsConfiguration(1024, 768) # Create a window of 1024x768 resolution/size
graphics = rd.gui.Graphics(gconfig) # create graphics object with configuration
simu.set_graphics(graphics)
graphics.look_at([3., 1., 2.], [0., 0., 0.])

# Add robot and nice floor
simu.add_robot(robot)
simu.add_checkerboard_floor()

Kp = 10
Ki = 0.1
Kd = 0.1
controller = PITask(tf_desired, dt, Kp, Ki, Kd)


# Run simulation
while True:
    if simu.step_world():
        break

    #print('Mass matrix:\n', robot.mass_matrix())
    #print('Inverse mass matrix:\n', robot.inv_mass_matrix())
    #print('Coriolis:', robot.coriolis_forces())
    #print('Gravity:', robot.gravity_forces())
    #print('Coriolis/gravity:', robot.coriolis_gravity_forces())

    tf = robot.body_pose(eef_link_name)
    torq = controller.update(tf)
    jac = robot.jacobian(eef_link_name)
    jac_pinv = damped_pseudoinverse(jac)
    cmd = jac_pinv @ torq

    robot.set_commands(cmd)
    print(robot.positions())