import numpy as np
import RobotDART as rd
import dartpy
from numpy.lib import stride_tricks  # OSX breaks if this is imported before RobotDART
from init_tower_disks import init_disks, init_tower
from symbolic import next_move
from utils import damped_pseudoinverse, angle_wrap_multi

#############################     Initialisations     ############################################

# Create simulator object
dt = 0.004 # we want a timestep of 0.004
simu = rd.RobotDARTSimu(dt)
simu.set_collision_detector("fcl") # fcl as the collision detector

# Create graphics
gconfig = rd.gui.GraphicsConfiguration(1024, 768)
graphics = rd.gui.Graphics(gconfig)
simu.set_graphics(graphics)
graphics.look_at([2., 0., 1.])

# index of init positions
index = 0

# load tower
packages = [["tower_of_hanoi", "tower_description"]]
tower = rd.Robot("tower.urdf", packages, "tower_base")
tower.set_positions(init_tower(index))
tower.fix_to_world() # fix to world

tower_poles = ["poleA", "poleB", "poleC"]   #Dimiourgia array pou periexei ta onomata twn poles

# add tower to simulation
simu.add_robot(tower)

# get disk init states
disk_positions = init_disks(index)

# create/load disks
disks = []
for i in range(3):
    disk = rd.Robot("disk" + str(2-i) + ".urdf", packages, "disk" + str(i))
    disk.set_positions(disk_positions[i])
    disks.append(disk)
    # add disk to simulation
    simu.add_robot(disk)

# load/position Franka
packages = [["franka_description", "franka/franka_description"]]
franka = rd.Robot("franka/franka.urdf", packages, "franka")
franka.fix_to_world()
franka.set_color_mode("material")
franka.set_actuator_types("servo")

# set initial joint positions
positions = franka.positions()
positions[5] = np.pi / 2.0
positions[7] = 0.04
positions[8] = 0.04
franka.set_positions(positions)

simu.add_robot(franka)

simu.add_floor()

#########################   Controllers    ############################################
class PITask:   #Xrisi PITask controller gia elegho tou braxiona
    def __init__(self, target, dt, Kp = 10., Ki = 0.1):
        self._target = target
        self._dt = dt
        self._Kp = Kp
        self._Ki = Ki
        self._sum_error = 0
    
    def set_target(self, target):
        self._target = target
    
    # function to compute error
    def error(self, tf):
        #Enallaktikos tropos
            # 2 ways of computing rotation error in world frame
            # # 1st way: compute error in body frame
            # error_in_body_frame = rd.math.logMap(tf.rotation().T @ self._target.rotation())
            # # transform it in world frame
            # error_in_world_frame = error_in_body_frame @ tf.rotation().T
            # 2nd way: compute error directly in world frame

        rot_error = rd.math.logMap(self._target.rotation() @ tf.rotation().T)
        lin_error = self._target.translation() - tf.translation()
        return np.r_[rot_error, lin_error]
    
    def update(self, current):
        error_in_world_frame = self.error(current)

        self._sum_error = self._sum_error + error_in_world_frame * self._dt

        return self._Kp * error_in_world_frame + self._Ki * self._sum_error, error_in_world_frame   #Epistrofh kai tou error gia xrhsh gia metavash sthn epomenh entolh


class PIJoint:  #Xrisi tou controller gia elegxo twn daktulwn ths daganas
    def __init__(self, target, dt, Kp = 10., Ki = 0.1):
        self._target = target
        self._dt = dt
        self._Kp = Kp
        self._Ki = Ki
        self._error = 0.
        self._sum_error = 0.

    def set_target(self, target):
        self._target = target

    def update(self, current):

        last_error = self._error
        self._error = angle_wrap_multi(self._target - current) # since we have angles, it's better to wrap into [-pi,pi)
        rithmos = (last_error - self._error) / self._dt #upologismos tou ruthmou metabolhs tou error
        self._sum_error = self._sum_error + self._error * self._dt

        return self._Kp * self._error + self._Ki * self._sum_error, rithmos #Epistrofh kai tou ruthmou gia xrhsh gia metavash sthn epomenh entolh

############################    Sunartiseis     ############################################

def KleiseDagana():     #Sunartish gia to kleisimo ths daganas

    kleisto_pos = franka.positions()
    kleisto_pos[7] = 0.
    controllerj = PIJoint(kleisto_pos, dt, Kp, Ki) # Kp,Ki could be a numpy array (one for each dim)

    while True:
        if simu.step_world():
            break

        cmd, rithmos = controllerj.update(franka.positions())   #PIJoint controller
        cmd[8] = 0
        franka.set_commands(cmd)

        if np.all(max_rithmall_dak > np.absolute(rithmos)):    #Otan h times tou rithmos perasoun to apodekto megisto orio, metabainei sthn epomenh entolh
            break

def AnoikseDagana():    #Sunartish gia to anoigma ths daganas
    
    anoixto_pos = franka.positions()
    anoixto_pos[7] = 0.07
    controllerj = PIJoint(anoixto_pos, dt, Kp, Ki)      #PIJoint controller

    while True:
        if simu.step_world():
            break

        cmd, error = controllerj.update(franka.positions())     
        cmd[8] = 0
        franka.set_commands(cmd)

        if np.all(max_error_arm > np.absolute(error)):  #Otan h times tou rithmos perasoun to apodekto elaxisto orio, metabainei sthn epomenh entolh
            break

def MoveArm():      #Sunartish gia thn kinish tou braxiona
    controllerT = PITask(stoxos, dt, Kp, Ki)    #PITask Controller

    while True:
        if simu.step_world():
            break
        
        tf = franka.body_pose("panda_hand")
        vel,error = controllerT.update(tf)
        jac = franka.jacobian("panda_hand") # this is in world frame
        
        jac_pinv = damped_pseudoinverse(jac) # np.linalg.pinv(jac) # get pseudo-inverse
        cmd = jac_pinv @ vel

        #Enallaktikos tropos
            #alpha = 2.
            #cmd = alpha * (jac.T @ vel) # using jacobian transpose
        print(error)
        if keep_closed == True:     #An "True", eisagetai h timh gia na askeitai statherh dunami sto tip apo thn dagana
            cmd[7] = -5e-4 #-5e-8 

        franka.set_commands(cmd)
        
        if np.all(max_error_arm > np.absolute(error)):  #Otan h times tou rithmos perasoun to apodekto elaxisto orio, metabainei sthn epomenh entolh
            #time.sleep(1)
            break

def Katastasi():    #Sunartish gia elegxo ths katastashs tou perivallontos kai dhmiourgias tou "state"

    state = [[], [], []]

    if np.all(max_error_diskou > np.absolute(disks[0].body_pose("base_link").translation() - tower.body_pose(tower_poles[0]).translation())[0:2]):
        state[0].append(2)
    if np.all(max_error_diskou > np.absolute(disks[1].body_pose("base_link").translation() - tower.body_pose(tower_poles[0]).translation())[0:2]):
        state[0].append(1)
    if np.all(max_error_diskou > np.absolute(disks[2].body_pose("base_link").translation() - tower.body_pose(tower_poles[0]).translation())[0:2]):
        state[0].append(0)

    if np.all(max_error_diskou > np.absolute(disks[0].body_pose("base_link").translation() - tower.body_pose(tower_poles[1]).translation())[0:2]):
        state[1].append(2)
    if np.all(max_error_diskou > np.absolute(disks[1].body_pose("base_link").translation() - tower.body_pose(tower_poles[1]).translation())[0:2]):
        state[1].append(1)
    if np.all(max_error_diskou > np.absolute(disks[2].body_pose("base_link").translation() - tower.body_pose(tower_poles[1]).translation())[0:2]):
        state[1].append(0)

    if np.all(max_error_diskou > np.absolute(disks[0].body_pose("base_link").translation() - tower.body_pose(tower_poles[2]).translation())[0:2]):
        state[2].append(2)
    if np.all(max_error_diskou > np.absolute(disks[1].body_pose("base_link").translation() - tower.body_pose(tower_poles[2]).translation())[0:2]):
        state[2].append(1)
    if np.all(max_error_diskou > np.absolute(disks[2].body_pose("base_link").translation() - tower.body_pose(tower_poles[2]).translation())[0:2]):
        state[2].append(0)

    return state

def CurPoleofDisk():

    j = 0
    for i in Katastasi():
        if disk in i:
            return j
        
        j = j + 1

def AllagiStoxou(tf_desired, z_axis, anex): #Sunartish gia allagi tou stoxou

    stoxos.set_translation(tf_desired .translation())
    if anex == False:       #H "anex" kathorizei an to "z_axis" exartatai h einai anexarthto tou z tou stoxos.translation()
        stoxos.set_translation([stoxos.translation()[0], stoxos.translation()[1], z_axis])
    else:
        stoxos.set_translation([stoxos.translation()[0], stoxos.translation()[1], stoxos.translation()[2] + z_axis])


#############################   Metavlites   ###########################################

Kp = 1.5 # Kp could be an array of 6 values
Ki = 0.1 # Kp could be an array of 6 values
max_rithmall_dak = 5e-3     #elaxisth apodekth metavoli tou error kata to kleisimo ths daganas se ena disk tip
max_error_arm = 5e-4        #9.5e-3megisto apodekto sfalma sthn thesh ths daganas kata thn kinhsh tou braxiona
max_error_diskou = 1e-1     #Error pou bohtaei sthn sunarthsh Katastasi()
keep_closed = False         #metablhth gia energopoihsh "sfyksimatos" ths daganas kata thn kinhsh tou braxiona


disk, pole = next_move(Katastasi())    #epistrefei ton arithmo tou diskou kai thn kolwna sthn opoia prepei na metakinithei

x_rot = dartpy.math.matrixToEulerXYZ(tower.body_pose("tower_base_link").rotation())     #to rotation tou "tower_base_link" ston z axona

upologismos_stoxou = dartpy.math.Isometry3()               #Dhmiourgia tou pinaka "upologismos_stoxou"
upologismos_stoxou.set_rotation(dartpy.math.eulerZYXToMatrix([0., 0., 0.]))
upologismos_stoxou.set_translation([0., 0., 0.])

stoxos = dartpy.math.Isometry3()               #Dhmiourgia tou pinaka "stoxos"
stoxos.set_rotation(dartpy.math.eulerZYXToMatrix([x_rot[2] + np.pi/2, np.pi, 0.]))
stoxos.set_translation([0., 0., 0.])

########################    MAIN    ###################################

while True: 
    
    AllagiStoxou(disks[2-disk].body_pose("tip"), 0.5, False)    #Orismos neou stoxou
    MoveArm()                                                   #Nea entoli

    AllagiStoxou(disks[2-disk].body_pose("tip"), 0.13, True)
    MoveArm()
    
    KleiseDagana()
    keep_closed = True

    AllagiStoxou(disks[2-disk].body_pose("tip"), 0.5, False)
    MoveArm()
    
    y = CurPoleofDisk()
    oros = tower.body_pose(tower_poles[y]).translation() - disks[2-disk].body_pose("base_link").translation()
    upologismos_stoxou.set_translation(tower.body_pose(tower_poles[pole]).translation() + ((disks[2-disk].body_pose("tip").translation() - disks[2-disk].body_pose("base_link").translation()) + oros))
    AllagiStoxou(upologismos_stoxou, 0.5, False)
    MoveArm()

    AllagiStoxou(upologismos_stoxou, 0.3, False)
    MoveArm()

    keep_closed = False
    AnoikseDagana()

    AllagiStoxou(upologismos_stoxou, 0.5, False)
    MoveArm()

    if Katastasi() == [[2, 1, 0], [], []]:  #Elegxos an luthike to problhma
        while True:
            if simu.step_world():
                break

    disk, pole = next_move(Katastasi())      #An den luthike to problhma, ananewsh metablhtwn gia to epomeno bhma