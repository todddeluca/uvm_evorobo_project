

import pyrosim
import math
import matplotlib.pyplot as plt
import random

from robot import Robot

for i in range(10):
    
    sim = pyrosim.Simulator(play_paused=False, eval_time=200)
    weight = random.random() * 2 - 1
    robot = Robot(sim, weight=weight)

    sim.start()
    sim.wait_to_finish()

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
