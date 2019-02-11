
import constants as c
from environment import Environment


class Environments:
    
    def __init__(self):
        self.envs = [Environment(i) for i in range(c.num_envs)]
        