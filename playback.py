import pickle

# load the best model
with open('robot.pkl', 'rb') as fh:
    best = pickle.load(fh)
    
best.evaluate(play_blind=False)
print(f'[best weight: {best.genome}] [best fitness: {best.fitness}]')

