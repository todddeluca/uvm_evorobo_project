

import pyrosim
import math
import numpy as np
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

'''
Spider:

A spider is a central sphere connected to n legs, where n=2k for some positive integer k. The legs are evenly distributed around the
xy plane bisecting the sphere.

L is leg length.
R is leg radius.
S is sphere radius.

Upper leg angle angle (around circumference of xy plane) of leg i is \theta_i = (2\pi / k) * (i + 1/2).
Upper leg: x=(S+0.5L)*cos(theta), y=(S+0.5L)*sin(theta), z=L+R. r1=cos(theta), r2=sin(theta), r3=0, r, g, b
Lower leg: x=(S+L)*cos(theta), y=(S+L)*sin(theta), z=0.5*L+R, r1=0, r2=0, r3=1, r, g, b

Body to upper leg joint: x=S*cos(theta), y=S*sin(theta), z=L+R, n1=-sin(theta), n2=cos(theta), n3=0, lo=-math.pi/2 , hi=math.pi/2
upper to lower leg joint: x=(S+L)*cos(theta), y=(S+L)*sin(theta), z=L+R, n1=-sin(theta), n2=cos(theta), n3=0, lo=-math.pi/2 , hi=math.pi/2

'''

class Robot:
    def __init__(self, sim, weights, num_legs=4, L=1, R=1, S=1):
        '''
        L: leg length
        R: leg radius
        S: body radius
        '''
        self.group = 'robot'
        body, upper_legs, lower_legs, joints = self.send_objects_and_joints(sim, num_legs, L, R, S)
        sensors, p4, l5 = self.send_sensors(sim, body, upper_legs, lower_legs)
        self.p4 = p4
        self.l5 = l5
        sensor_neurons, motor_neurons = self.send_neurons(sim, sensors, joints)
        self.send_synapses(sim, weights, sensor_neurons, motor_neurons)
        
    def send_objects_and_joints(self, sim, num_legs, L, R, S):
        o0 = sim.send_sphere(x=0, y=0, z=L+R, radius=S, r=0.5, g=0.5, b=0.5,
                             collision_group=self.group)
        upper_legs = []
        lower_legs = []
        joints = []
        for i in range(num_legs):
            theta = (2 * np.pi / num_legs) * i # (i + 0.5) # i=1 central front leg, (i+0.5)=2 symmetric front legs
            upper = sim.send_cylinder(x=(S + 0.5 * L) * np.cos(theta), 
                                    y=(S + 0.5 * L) * np.sin(theta), 
                                    z=(L + R), length=L, radius=R, 
                                    r1=np.cos(theta), r2=np.sin(theta), r3=0,
                                    r=(1+np.cos(theta))/2, g=0, b=(1+np.sin(theta))/2,
                                     collision_group=self.group)
            upper_legs.append(upper)
            lower = sim.send_cylinder(x=(S + L) * np.cos(theta), 
                                    y=(S + L) * np.sin(theta),
                                    z=(0.5 * L + R), length=L, radius=R,
                                    r1=0, r2=0, r3=1,
                                    r=(1+np.cos(theta))/4, g=0, b=(1+np.sin(theta))/4,
                                     collision_group=self.group)
            lower_legs.append(lower)
            # body-to-upper-leg joint
            j0 = sim.send_hinge_joint(first_body_id=o0, second_body_id=upper, 
                                      x=S*np.cos(theta), y=S*np.sin(theta), z=L + R, 
                                      n1=-np.sin(theta), n2=np.cos(theta), n3=0, 
                                      lo=-math.pi/2 , hi=math.pi/2)
            # upper-to-lower-leg joint
            j1 = sim.send_hinge_joint(first_body_id=upper, second_body_id=lower, 
                                      x=(S+L)*np.cos(theta), y=(S+L)*np.sin(theta), z=L + R, 
                                      n1=-np.sin(theta), n2=np.cos(theta), n3=0, 
                                      lo=-math.pi/2 , hi=math.pi/2)
            joints += [j0, j1]

        return o0, upper_legs, lower_legs, joints
            
    def send_sensors(self, sim, body, upper_legs, lower_legs):
        sensors = []
        # lower limb touch sensors
        for lower in lower_legs:
            sensors.append(sim.send_touch_sensor(body_id=lower))
        
        # upper limb touch sensors
#         for upper in upper_legs:
#             sensors.append(sim.send_touch_sensor(body_id=upper))
        
        p4 = sim.send_position_sensor(body_id=body)
        l5 = sim.send_light_sensor(body_id=body)
        sensors.append(l5)
        
        return sensors, p4, l5

    def send_neurons(self, sim, sensors, joints):
        sensor_neurons = []
        for sensor in sensors:
            sensor_neurons.append(sim.send_sensor_neuron(sensor_id=sensor))
            
        motor_neurons = []
        for joint in joints:
            motor_neurons.append(sim.send_motor_neuron(joint_id=joint, tau=0.3))
            
        return sensor_neurons, motor_neurons
        
    def send_synapses(self, sim, weights, sensor_neurons, motor_neurons):
        for i, s in enumerate(sensor_neurons):
            for j, m in enumerate(motor_neurons):
#                 sim.send_synapse(source_neuron_id=s, target_neuron_id=m, weight=random.random() * 2 - 1)
                sim.send_synapse(source_neuron_id=s, target_neuron_id=m, weight=weights[i, j])
        
        