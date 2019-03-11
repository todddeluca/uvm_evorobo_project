import pickle
from environments import Environments

# environments have a light source in different locations
envs = Environments()

# load the best model
with open('robot.pkl', 'rb') as fh:
    parents = pickle.load(fh)

parents.evaluate(envs, play_blind=False)
print(f'0: {parents}')

