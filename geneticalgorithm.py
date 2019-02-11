

import matplotlib.pyplot as plt
import random
import copy
import pickle

import constants as c
from population import Population
from environments import Environments

envs = Environments()

parents = Population(c.pop_size)
parents.initialize()
parents.evaluate(envs, play_blind=True, play_paused=False)
print(f'0: {parents}')

for i in range(1, c.num_gens + 1):
    children = Population(c.pop_size)
    children.fill_from(parents)
    children.evaluate(envs)
    parents = children
    print(f'{i}: {parents}')
    
parents.p = parents.p[:1]
parents.evaluate(envs, play_blind=False)
