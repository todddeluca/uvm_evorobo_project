

import pyrosim
import math
import matplotlib.pyplot as plt
import random

from robot import Robot
from individual import Individual

for i in range(10):
    
    individual = Individual()
    individual.evaluate()
    print('genome:', individual.genome)
    print('fitness:', individual.fitness)
#     sim = pyrosim.Simulator(play_paused=False, eval_time=200)
#     weight = random.random() * 2 - 1
#     robot = Robot(sim, weight=weight)

#     sim.start()
#     sim.wait_to_finish()
    
#     x = sim.get_sensor_data(sensor_id=robot.p4, svi=0) # svi: sensor value index
#     y = sim.get_sensor_data(sensor_id=robot.p4, svi=1)
#     z = sim.get_sensor_data(sensor_id=robot.p4, svi=2)
#     print(y[-1])

#     f = plt.figure()
#     panel = f.add_subplot(111)
#     plt.plot(x, label='x')
#     plt.plot(y, label='y')
#     plt.plot(z, label='z')
#     plt.legend()
#     # panel.set_ylim(-1,+2)
#     plt.show()

# sensorData = sim.get_sensor_data(sensor_id=t0)
# sensorData = sim.get_sensor_data(sensor_id=t1)
# sensorData = sim.get_sensor_data(sensor_id=p2)
# sensorData = sim.get_sensor_data(sensor_id=r3)
# print(sensorData)

# f = plt.figure()
# panel = f.add_subplot(111)
# plt.plot(sensorData)
# # panel.set_ylim(-1,+2)
# plt.show()
