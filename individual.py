import pyrosim
import math
import matplotlib.pyplot as plt
import random
import numpy as np

import constants as c
from robot import Robot


class Individual:
    
    def __init__(self, id_):
        self.num_legs = c.num_legs
        self.genome = np.random.random((1 + self.num_legs, self.num_legs * 2)) * 2 - 1
        self.fitness = 0
        self.id_ = id_
        
    def start_evaluation(self, env, play_blind=True, play_paused=False):
        self.sim = pyrosim.Simulator(play_paused=play_paused, eval_time=c.eval_time, play_blind=play_blind)
        env.send_to(self.sim)
        robot = Robot(self.sim, weights=self.genome, num_legs=self.num_legs)
        self.position_sensor_id = robot.p4
        self.distance_sensor_id = robot.l5 # distance from light source
        self.sim.start()
    
    def compute_fitness(self):
        self.sim.wait_to_finish()

#         self.fitness = self.sim.get_sensor_data(sensor_id=self.position_sensor_id, svi=1)[-1]
        self.fitness += self.sim.get_sensor_data(sensor_id=self.distance_sensor_id)[-1]
        del self.sim # so deepcopy does not copy the simulator

    def mutate(self):
        i = np.random.randint(self.genome.shape[0])
        j = np.random.randint(self.genome.shape[1])
        new_weight = random.gauss(self.genome[i, j], math.fabs(self.genome[i, j]))
        self.genome[i, j] = np.clip(new_weight, -1, 1)

    def __repr__(self):
        return f'[{self.id_} {self.fitness:.4}]'
    