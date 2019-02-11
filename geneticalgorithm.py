

import matplotlib.pyplot as plt
import random
import copy
import pickle

# from individual import Individual

from population import Population

pop_size = 10
parents = Population(pop_size)
parents.initialize()
parents.evaluate(play_blind=True, play_paused=False)
print(f'0: {parents}')

for i in range(1, 200):
    children = Population(pop_size)
    children.fill_from(parents)
    children.evaluate()
    parents = children
    print(f'{i}: {parents}')
    
parents.p = parents.p[:1]
parents.evaluate(play_blind=False)
