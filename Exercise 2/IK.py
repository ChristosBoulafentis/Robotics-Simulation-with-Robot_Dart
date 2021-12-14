import numpy as np
import RobotDART as rd
import dartpy  # OSX breaks if this is imported before RobotDART
import copy
from utils import damped_pseudoinverse, AdT, angle_wrap_multi

# Load a robot
packages = [("franka_description", "franka/franka_description")]
robot = rd.Robot("franka/franka.urdf", packages)
robot.set_color_mode("material")
robot.set_actuator_types("servo")
# fix to world and minor things
robot.fix_to_world()

# Load ghost robot for visualization
#robot_ghost = robot.clone_ghost()

# set initial joint positions
target_positions = copy.copy(robot.positions())
target_positions[5] = np.pi / 2.0
target_positions[7] = 0.3
# target_positions[1] = -np.pi / 2.0
robot.set_positions(target_positions)

# get end-effector pose
eef_link_name = "panda_link8"
tf_desired = robot.body_pose(eef_link_name)

# set robot back to zero positions
robot.reset()
robot.set_positions(robot.positions()+np.random.rand(robot.num_dofs())*np.pi/1.5-np.pi/3.)

# function to compute error in Transformation space
def error(tf, tf_desired):
    return rd.math.logMap(tf.inverse().multiply(tf_desired))

# optimization Newton-Raphson
def ik_jac(init_positions, tf_desired, step = np.pi/4., max_iter = 100, min_error = 1e-6):
    pos = init_positions
    for _ in range(max_iter):
        robot.set_positions(pos)
        tf = robot.body_pose(eef_link_name)
        Ad_tf = AdT(tf)
        error_in_body_frame = error(tf, tf_desired)
        error_in_world_frame = Ad_tf @ error_in_body_frame

        ferror = np.linalg.norm(error_in_world_frame)
        if ferror < min_error:
            break

        jac = robot.jacobian(eef_link_name)
        jac_pinv = damped_pseudoinverse(jac)

        delta_pos = jac_pinv @ error_in_world_frame
        for i in range(delta_pos.shape[0]):
            if delta_pos[i] > step:
                delta_pos[i] = step
            elif delta_pos[i] < -step:
                delta_pos[i] = -step

        pos = pos + delta_pos

    # We would like to wrap the final joint positions to [-pi,pi)
    pos = angle_wrap_multi(pos)
    print('Final error:', ferror)

    return pos

tt = copy.copy(robot.positions())
res = ik_jac(tt, tf_desired)
print('IK result:', res)
print('Target joint:', target_positions)

# Create simulator object
simu = rd.RobotDARTSimu()
simu.set_collision_detector("fcl")

# Create Graphics
gconfig = rd.gui.GraphicsConfiguration(1024, 768)
graphics = rd.gui.Graphics(gconfig)
simu.set_graphics(graphics)
graphics.look_at([3., 1., 2.], [0., 0., 0.])

# Add robot and nice floor
simu.add_robot(robot)
simu.add_checkerboard_floor()

robot.set_positions(res)

# Run simulation
while True:
    if simu.step_world():
        break
