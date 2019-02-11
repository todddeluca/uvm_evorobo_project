import pyrosim
import math
import matplotlib.pyplot as plt
import random
import numpy as np

from robot import Robot


class Individual:
    
    def __init__(self, id_):
        self.genome = np.random.random(4) * 2 - 1
        self.fitness = 0
        self.id_ = id_
        
    def start_evaluation(self, play_blind=True):
        self.sim = pyrosim.Simulator(play_paused=True, eval_time=2000, play_blind=play_blind)
        robot = Robot(self.sim, weights=self.genome)
        self.position_sensor_id = robot.p4
        self.sim.start()
    
    def compute_fitness(self):
        self.sim.wait_to_finish()

        y = self.sim.get_sensor_data(sensor_id=self.position_sensor_id, svi=1)
        self.fitness = y[-1]
        del self.sim # so deepcopy does not copy the simulator

    def mutate(self):
        gene_to_mutate = random.randrange(len(self.genome))
#         print('gene_to_mutate', gene_to_mutate)
        self.genome[gene_to_mutate] = random.gauss(
            self.genome[gene_to_mutate], math.fabs(self.genome[gene_to_mutate]))

    def __repr__(self):
        return f'[{self.id_} {self.fitness:.4}]'
    