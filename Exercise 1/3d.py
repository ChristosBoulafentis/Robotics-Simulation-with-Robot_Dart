import numpy as np
import RobotDART as rd
import dartpy

########## Create simulator object ##########
time_step = 0.001
simu = rd.RobotDARTSimu(time_step)

########## Create Graphics ##########
graphics = rd.gui.Graphics()
simu.set_graphics(graphics)
graphics.look_at([1.5, 0, 0], [0., 0., 0.5])

########## Create robot ##########
robot = rd.Robot("1urdf.urdf")
robot.set_position_enforced(True)
robot.fix_to_world()
robot.set_actuator_types("servo")

robot.set_positions([0., np.pi/8.,np.pi/5., np.pi/2., -np.pi/5.,0.,0.])

simu.add_robot(robot)

simu.add_floor()

####################Calculation of locations######################
L0 = 0.02
L1 = 0.1
L2 = 0.4
L3 = 0.3
L4 = 0.135
L5 = 0.015

joints = robot.positions()

tf_01 = dartpy.math.Isometry3()
tf_01.set_rotation(dartpy.math.eulerZYXToMatrix([joints[0], 0., 0.]))
tf_01.set_translation([0., 0., L0])

tf_12 = dartpy.math.Isometry3()
tf_12.set_rotation(dartpy.math.eulerZYXToMatrix([0., joints[1], 0.]))
tf_12.set_translation([0., 0., L1])

tf_23 = dartpy.math.Isometry3()
tf_23.set_rotation(dartpy.math.eulerZYXToMatrix([0., joints[2], 0.]))
tf_23.set_translation([0., 0., L2])

tf_34 = dartpy.math.Isometry3()
tf_34.set_rotation(dartpy.math.eulerZYXToMatrix([0., joints[3], 0.]))
tf_34.set_translation([0., 0., L3])

tf_45 = dartpy.math.Isometry3()
tf_45.set_rotation(dartpy.math.eulerZYXToMatrix([0., joints[4], 0.]))
tf_45.set_translation([0., 0., L4])

tf_56 = dartpy.math.Isometry3()
tf_56.set_rotation(dartpy.math.eulerZYXToMatrix([0., 0., 0.]))
tf_56.set_translation([0., 0., L5])

tf = tf_01.multiply(tf_12).multiply(tf_23).multiply(tf_34).multiply(tf_45).multiply(tf_56)
print(tf)
#########################Calculation of locations with RobotDart#########################################
print(robot.body_pose("end"))
########## Run simulation ##########
while True:
    if simu.step_world():
        break
