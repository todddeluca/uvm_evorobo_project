

import random
import copy
import pickle

import constants as c
from population import Population
from environments import Environments

# environments have a light source in different locations
envs = Environments()

# population contains multiple individuals
parents = Population(c.pop_size)
parents.initialize()
parents.evaluate(envs, play_blind=False, play_paused=True)
print(f'0: {parents}')

for i in range(1, c.num_gens + 1):
    children = Population(c.pop_size)
    children.fill_from(parents)
    children.evaluate(envs)
    parents = children
    print(f'{i}: {parents}')
    
parents.p = parents.p[:1]
parents.evaluate(envs, play_blind=False)

# save best model
with open('robot.pkl', 'wb') as fh:
    pickle.dump(parents, fh)
    
