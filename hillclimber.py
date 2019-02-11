

import matplotlib.pyplot as plt
import random
import copy
import pickle

from individual import Individual

parent = Individual()
parent.evaluate(play_blind=True)
print('genome:', parent.genome)
print('fitness:', parent.fitness)
print(f'[g: -1] [pw: {parent.genome}] [p: {parent.fitness}]')

for i in range(100):
    child = copy.deepcopy(parent)
    child.mutate()
    child.evaluate(play_blind=True)
    print(f'[g: {i}] [pw: {parent.genome}] [p: {parent.fitness}] [c: {child.fitness}]')
    if child.fitness > parent.fitness:
        child.evaluate(play_blind=True)
        parent = child
        # save best model
#         with open('robot.pkl', 'wb') as fh:
#             pickle.dump(parent, fh)
