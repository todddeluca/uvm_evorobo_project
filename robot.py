

import pyrosim
import math
import numpy as np
import random


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
    def __init__(self, sim, weights, num_legs=4, L=1, R=1, S=1, num_hidden=4, num_hidden_layers=0,
                 use_proprio=False, use_vestib=False):
        '''
        L: leg length
        R: leg radius
        S: body radius
        '''
        self.group = 'robot'
        body, upper_legs, lower_legs, joints = self.send_objects_and_joints(sim, num_legs, L, R, S)
        sensors, p4, l5, vid = self.send_sensors(sim, body, upper_legs, lower_legs, joints,
                                                 use_proprio=use_proprio, use_vestib=use_vestib)
        self.p4 = p4
        self.l5 = l5
        self.v_id = vid # vestibular sensor
        sensor_neurons, motor_neurons, hidden_layers, bias_neuron = self.send_neurons(
            sim, sensors, joints, num_hidden, num_hidden_layers)
        self.send_synapses(sim, weights, sensor_neurons, motor_neurons, hidden_layers, bias_neuron)
        
    def send_objects_and_joints(self, sim, num_legs, L, R, S):
        o0 = sim.send_sphere(x=0, y=0, z=L+R, radius=S, r=0.5, g=0.5, b=0.5,
                             collision_group=self.group)
        upper_legs = []
        lower_legs = []
        joints = []
        for i in range(num_legs):
            theta = (2 * np.pi / num_legs) * (i + 0.5) # i=1 central front leg, (i+0.5)=2 symmetric front legs
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
            
    def send_sensors(self, sim, body, upper_legs, lower_legs, joints, use_proprio=False, use_vestib=False):
        sensors = []
        
        if use_proprio:
            for joint in joints:
                sensors.append(sim.send_proprioceptive_sensor(joint))
            
        # front leg ray sensors
        # ...todo
        
        # lower limb touch sensors
        for lower in lower_legs:
            sensors.append(sim.send_touch_sensor(body_id=lower))
        
        # upper limb touch sensors
#         for upper in upper_legs:
#             sensors.append(sim.send_touch_sensor(body_id=upper))
        
        p4 = sim.send_position_sensor(body_id=body)
        l5 = sim.send_light_sensor(body_id=body)
#         sensors.append(l5)
        vid = sim.send_vestibular_sensor(body_id=body)
        if use_vestib:
            sensors.append(vid)
        
        return sensors, p4, l5, vid

    def send_neurons(self, sim, sensors, joints, num_hidden, num_hidden_layers):
        bias_neuron = sim.send_bias_neuron()

        sensor_neurons = []
        for sensor in sensors:
            sensor_neurons.append(sim.send_sensor_neuron(sensor_id=sensor))
            
        motor_neurons = []
        for joint in joints:
            motor_neurons.append(sim.send_motor_neuron(joint_id=joint, tau=0.3))
            
        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_neurons = []
            for _ in range(num_hidden):
                hidden_neurons.append(sim.send_hidden_neuron())
                
            hidden_layers.append(hidden_neurons)
            
        return sensor_neurons, motor_neurons, hidden_layers, bias_neuron
        
    def send_synapses(self, sim, weights, sensor_neurons, motor_neurons, hidden_layers, bias_neuron):
        layers = [sensor_neurons + [bias_neuron]] # add bias to input layer
        for layer in hidden_layers:
            layers.append(layer + [bias_neuron]) # add bias to hidden layers
        layers.append(motor_neurons)
        
        pairs = [] # source and target neuron pairs
        for i in range(len(layers) - 1):
            in_layer = layers[i]
            out_layer = layers[i + 1]
            for inp in in_layer:
                for out in out_layer:
                    pairs.append((inp, out))
            
        for i, (s, t) in enumerate(pairs):
            sim.send_synapse(source_neuron_id=s, target_neuron_id=t, weight=weights[i])
        
        