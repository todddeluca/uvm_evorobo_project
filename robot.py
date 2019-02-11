

import pyrosim
import math
import matplotlib.pyplot as plt
import constants as c
import random


'''
Spreadsheet of object and joint values

obj, x, y, z, shape, shape, shape, r1, r2, r3, r, g, b
o0, 0, 0, L + R, length=L, width=L, height=2R, , , , 0.5, 0.5, 0.5 # grey square body
o1, 0, L, L + R, length=L, radius=R, null, 0, 1, 0, 0.5, 0, 0 # upper dark red forward (y-dir) leg
o2, L, 0, L + R, length=L, radius=R, null, 1, 0, 0, 0, 0.5, 0 # upper dark green right (x-dir) leg
o3, 0, -L, L + R, length=L, radius=R, null, 0, 1, 0, 0, 0, 0.5 # upper dark blue backward (-y-dir) leg
o4, -L, 0, L + R, length=L, radius=R, null, 1, 0, 0, 0.5, 0, 0.5 # upper dark purple left (-x-dir) leg
o5, 0, 1.5 * L, 0.5 * L + R, length=L, radius=R, null, 0, 0, 1, 1, 0, 0 # lower red forward leg
o6, 1.5 * L, 0, 0.5 * L + R, length=L, radius=R, null, 0, 0, 1, 0, 1, 0 # lower green right leg
o7, 0, -1.5 * L, 0.5 * L + R, length=L, radius=R, null, 0, 0, 1, 0, 0, 1 # lower blue backward leg
o8, -1.5 * L, 0, 0.5 * L + R, length=L, radius=R, null, 0, 0, 1, 1, 0, 1 # lower purple right leg


joint, 1st_obj, 2nd_obj, x, y, z, n1, n2, n3
j0, 0, 1, 0, L/2, L+R, -1, 0, 0 # body-red joint
j1, 1, 5, 0, 1.5L, L+R, -1, 0, 0 # red-red joint
j2, 0, 2, L/2, 0, L+R, 0, 1, 0 # body-green joint
j3, 2, 6, 1.5L, 0, L+R, 0, 1, 0 # green-green joint
j4, 0, 3, 0, -L/2, L+R, -1, 0, 0 # body-blue joint
j5, 3, 7, 0, -1.5L, L+R, -1, 0, 0 # blue-blue joint
j6, 0, 4, -L/2, 0, L+R, 0, 1, 0 # body-purple joint
j7, 4, 8, -1.5L, 0, L+R, 0, 1, 0 # purple-purple joint
'''


class Robot:
    def __init__(self, sim, weights):
        objs = self.send_objects(sim)
        joints = self.send_joints(sim, objs)
        touch_sensors, p4 = self.send_sensors(sim, objs)
        self.p4 = p4
        sensor_neurons, motor_neurons = self.send_neurons(sim, touch_sensors, joints)
        self.send_synapses(sim, weights, sensor_neurons, motor_neurons)
        
    def send_joints(self, sim, objs):
        o0, o1, o2, o3, o4, o5, o6, o7, o8 = objs
        j0 = sim.send_hinge_joint(first_body_id=o0, second_body_id=o1, x=0, y=c.L / 2, z=c.L + c.R, 
                                  n1=-1, n2=0, n3=0, lo=-math.pi/2 , hi=math.pi/2)
        j1 = sim.send_hinge_joint(first_body_id=o1, second_body_id=o5, x=0, y=1.5 * c.L, z=c.L + c.R, 
                                  n1=-1, n2=0, n3=0, lo=-math.pi/2 , hi=math.pi/2)
        j2 = sim.send_hinge_joint(first_body_id=o0, second_body_id=o2, x=c.L / 2, y=0, z=c.L + c.R, 
                                  n1=0, n2=1, n3=0, lo=-math.pi/2 , hi=math.pi/2)
        j3 = sim.send_hinge_joint(first_body_id=o2, second_body_id=o6, x=1.5 * c.L, y=0, z=c.L + c.R, 
                                  n1=0, n2=1, n3=0, lo=-math.pi/2 , hi=math.pi/2)
        j4 = sim.send_hinge_joint(first_body_id=o0, second_body_id=o3, x=0, y=-c.L / 2, z=c.L + c.R, 
                                  n1=-1, n2=0, n3=0, lo=-math.pi/2 , hi=math.pi/2)
        j5 = sim.send_hinge_joint(first_body_id=o3, second_body_id=o7, x=0, y=-1.5 * c.L, z=c.L + c.R, 
                                  n1=-1, n2=0, n3=0, lo=-math.pi/2 , hi=math.pi/2)
        j6 = sim.send_hinge_joint(first_body_id=o0, second_body_id=o4, x=-c.L / 2, y=0, z=c.L + c.R, 
                                  n1=0, n2=1, n3=0, lo=-math.pi/2 , hi=math.pi/2)
        j7 = sim.send_hinge_joint(first_body_id=o4, second_body_id=o8, x=-1.5 * c.L, y=0, z=c.L + c.R, 
                                  n1=0, n2=1, n3=0, lo=-math.pi/2 , hi=math.pi/2)
        return [j0, j1, j2, j3, j4, j5, j6, j7]

    def send_objects(self, sim):
        o0 = sim.send_box(x=0, y=0, z=c.L + c.R, length=c.L, width=c.L, height=2 * c.R, r=0.5, g=0.5, b=0.5)
        o1 = sim.send_cylinder(x=0, y=c.L, z=c.L + c.R, length=c.L, radius=c.R, r1=0 , r2=1, r3=0, r=0.5, g=0, b=0)
        o2 = sim.send_cylinder(x=c.L, y=0, z=c.L + c.R, length=c.L, radius=c.R, r1=1 , r2=0, r3=0, r=0, g=0.5, b=0)
        o3 = sim.send_cylinder(x=0, y=-c.L, z=c.L + c.R, length=c.L, radius=c.R, r1=0 , r2=1, r3=0, r=0, g=0, b=0.5)
        o4 = sim.send_cylinder(x=-c.L, y=0, z=c.L + c.R, length=c.L, radius=c.R, r1=1 , r2=0, r3=0, r=0.5, g=0, b=0.5)
        o5 = sim.send_cylinder(x=0, y=1.5 * c.L, z=0.5 * c.L + c.R, length=c.L, radius=c.R, r1=0 , r2=0, r3=1, r=1, g=0, b=0)
        o6 = sim.send_cylinder(x=1.5 * c.L, y=0, z=0.5 * c.L + c.R, length=c.L, radius=c.R, r1=0 , r2=0, r3=1, r=0, g=1, b=0)
        o7 = sim.send_cylinder(x=0, y=-1.5 * c.L, z=0.5 * c.L + c.R, length=c.L, radius=c.R, r1=0 , r2=0, r3=1, r=0, g=0, b=1)
        o8 = sim.send_cylinder(x=-1.5 * c.L, y=0, z=0.5 * c.L + c.R, length=c.L, radius=c.R, r1=0 , r2=0, r3=1, r=1, g=0, b=1)
        return [o0, o1, o2, o3, o4, o5, o6, o7, o8]
            
    def send_sensors(self, sim, objs):
        o0, o1, o2, o3, o4, o5, o6, o7, o8 = objs
        t0 = sim.send_touch_sensor(body_id=o5)
        t1 = sim.send_touch_sensor(body_id=o6)
        t2 = sim.send_touch_sensor(body_id=o7)
        t3 = sim.send_touch_sensor(body_id=o8)
        p4 = sim.send_position_sensor(body_id=o0)
        return [t0, t1, t2, t3], p4

    def send_neurons(self, sim, touch_sensors, joints):
        sensor_neurons = []
        for touch_sensor in touch_sensors:
            sensor_neurons.append(sim.send_sensor_neuron(sensor_id=touch_sensor))
            
        motor_neurons = []
        for joint in joints:
            motor_neurons.append(sim.send_motor_neuron(joint_id=joint, tau=0.3))
            
        return sensor_neurons, motor_neurons
        
    def send_synapses(self, sim, weights, sensor_neurons, motor_neurons):
        for i, s in enumerate(sensor_neurons):
            for j, m in enumerate(motor_neurons):
#                 sim.send_synapse(source_neuron_id=s, target_neuron_id=m, weight=random.random() * 2 - 1)
                sim.send_synapse(source_neuron_id=s, target_neuron_id=m, weight=weights[i, j])
        
        