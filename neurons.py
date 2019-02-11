

import pyrosim
import math
import matplotlib.pyplot as plt

sim = pyrosim.Simulator(play_paused=False, eval_time=100)

white_object = sim.send_cylinder(x=0, y=0, z=0.6, length=1, radius=0.1)
red_object = sim.send_cylinder(x=0, y=0.5, z=1.1, r=0, g=0, b=1, r1=0 , r2=1, r3=0)
# redObject = sim.send_cylinder(x=0.5, y=0, z=1.1, r=1, g=0, b=0, r1=1 , r2=0, r3=0)
joint = sim.send_hinge_joint(first_body_id=white_object, second_body_id=red_object, x=0, y=0, z=1.1, n1=-1, n2=0, n3=0, lo=-math.pi/2 , hi=math.pi/2)
# joint = sim.send_hinge_joint(first_body_id=white_object, second_body_id=red_object, x=0, y=0, z=1.1, n1=0, n2=-1, n3=0)
t0 = sim.send_touch_sensor(body_id=white_object)
t1 = sim.send_touch_sensor(body_id=red_object)
p2 = sim.send_proprioceptive_sensor(joint_id=joint)
r3 = sim.send_ray_sensor(body_id=red_object, x=0, y=1.1, z=1.1, r1=0, r2=1, r3=0)
# r3 = sim.send_ray_sensor(body_id=red_object, x=0, y=0.5, z=1, r1=0, r2=0, r3=-1)

sn0 = sim.send_sensor_neuron(sensor_id=t0)
sn1 = sim.send_sensor_neuron(sensor_id=t1)
# mn2 = sim.send_motor_neuron(joint_id=joint)

sim.start()
sim.wait_to_finish()

# sensorData = sim.get_sensor_data(sensor_id=t0)
# sensorData = sim.get_sensor_data(sensor_id=t1)
# sensorData = sim.get_sensor_data(sensor_id=p2)
sensorData = sim.get_sensor_data(sensor_id=r3)
# print(sensorData)

f = plt.figure()
panel = f.add_subplot(111)
plt.plot(sensorData)
# panel.set_ylim(-1,+2)
plt.show()
