

import matplotlib.pyplot as plt
import random
import copy
import pickle

# from individual import Individual

from population import Population

parents = Population(10)
parents.evaluate(play_blind=True)
print(f'0: {parents}')

for i in range(1, 200):
    children = copy.deepcopy(parents)
    children.mutate()
    children.evaluate()
    parents.replace_with(children)
    print(f'{i}: {parents}')

parents.evaluate(play_blind=False)
