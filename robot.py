

import pyrosim
import math
import matplotlib.pyplot as plt

class Robot:
    def __init__(self, sim, weights):
        white_object = sim.send_cylinder(x=0, y=0, z=0.6, length=1, radius=0.1)
        red_object = sim.send_cylinder(x=0, y=0.5, z=1.1, r=1, g=0, b=0, r1=0 , r2=1, r3=0)
        joint = sim.send_hinge_joint(first_body_id=white_object, second_body_id=red_object, x=0, y=0, z=1.1, n1=-1, n2=0, n3=0, lo=-math.pi/2 , hi=math.pi/2)
        t0 = sim.send_touch_sensor(body_id=white_object)
        t1 = sim.send_touch_sensor(body_id=red_object)
        p2 = sim.send_proprioceptive_sensor(joint_id=joint)
        r3 = sim.send_ray_sensor(body_id=red_object, x=0, y=1.1, z=1.1, r1=0, r2=1, r3=0)
        self.p4 = sim.send_position_sensor(body_id=red_object)
        
        
        sn0 = sim.send_sensor_neuron(sensor_id=t0)
        sn1 = sim.send_sensor_neuron(sensor_id=t1)
        sn2 = sim.send_sensor_neuron(sensor_id=p2)
        sn3 = sim.send_sensor_neuron(sensor_id=r3)
        mn2 = sim.send_motor_neuron(joint_id=joint)

        sensor_neurons = {}
        if True:
            sensor_neurons[0] = sn0
            sensor_neurons[1] = sn1
            sensor_neurons[2] = sn2
            sensor_neurons[3] = sn3
        else:
            # the old robot had only the red object touch sensor connected to the motor neuron
            sensor_neurons[0] = sn1 
        
        motor_neurons = {}
        motor_neurons[0] = mn2

        for s in sensor_neurons:
            for m in motor_neurons:
                sim.send_synapse(source_neuron_id=sensor_neurons[s], 
                                 target_neuron_id=motor_neurons[m], weight=weights[s])

        
