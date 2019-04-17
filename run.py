


import argparse
import copy
import math
import numpy as np
import pickle
import random
import pyrosim
import es # from es import SimpleGA, CMAES, PEPG, OpenES

from robot import Robot

class Hyperparams:
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

class BlockEnv:
    
    def __init__(self, id_, L):
        self.group = 'env'
        self.id_ = id_
        self.l = L
        self.w = L
        self.h = L
        self.z = L / 2
        
        if id_ == 0: # front
            self.x = 0
            self.y = 30 * L
        elif id_ == 1: # right
            self.x = 30 * L
            self.y = 0
        elif id_ == 2: # back
            self.x = 0
            self.y = -30 * L
        elif id_ == 3: # left
            self.x = -30 * L
            self.y = 0
                    
    def send_to(self, sim):
        light_source = sim.send_box(x=self.x, y=self.y, z=self.z,
                                    length=self.l, width=self.w, height=self.h,
                                    r=0.9, g=0.9, b=0.9,
                                    collision_group=self.group,
                                   )
        sim.send_light_source(body_id=light_source)
            
class LadderEnv:
    def __init__(self, length, width, thickness, spacing, y_offset):
        self.length = length 
        self.width = width
        self.y_offset = y_offset
        self.thick = thickness # rung thickness
        self.spacing = spacing # rung spacing
        self.group = 'env' # collision group
        
    def send_to(self, sim):

        # Rails of the ladder
        # x, y, z, r1, r2, r3, l, r
        left_rail = (-self.width / 2, self.y_offset, self.length / 2 + self.thick, # account for cylinder end cap
                     0, 0, 1,
                     self.length, self.thick)
        right_rail = (self.width / 2, self.y_offset, self.length / 2 + self.thick,
                     0, 0, 1,
                      self.length, self.thick)
        
        rails = [left_rail, right_rail]
        rail_ids = []
        for x, y, z, r1, r2, r3, l, r in rails:
            id_ = sim.send_cylinder(x=x, y=y, z=z, 
                                  r1=r1, r2=r2, r3=r3, 
                                  length=l, radius=r,
                                  r=0.9, g=0.9, b=0.9,
                                  collision_group=self.group)
            rail_ids.append(id_)
                        
        # rungs: x, y, z, r1, r2, r3, l, w, h
        # make n rungs along the lengthe of the rail, separated by spacing
        rungs = []
        rung_ids = []
        touch_ids = [] # touch sensor ids
        pos = 0 + self.thick + self.thick / 2 # including cylinder cap of rail
        top = self.length
        while pos < top:
            rungs.append((0, self.y_offset, pos,
                          1, 0, 0, # ladder is oriented along x-axis
                          self.width, self.thick,
                         ))
            pos += self.spacing
        for x, y, z, r1, r2, r3, l, r in rungs:
            id_ = sim.send_cylinder(x=x, y=y, z=z, 
                                  r1=r1, r2=r2, r3=r3, 
                                  length=l, radius=r,
                                  r=0.9, g=0.9, b=0.9,
                                  collision_group=self.group)
            rung_ids.append(id_)
            # rung touch sensor
            tid = sim.send_touch_sensor(body_id=id_)
            touch_ids.append(tid)

        # fix the rungs to the rails
        for rid in rung_ids:
            sim.send_fixed_joint(rid, rail_ids[0])
            sim.send_fixed_joint(rid, rail_ids[1])
        
        # fix the rails to the world
        sim.send_fixed_joint(rail_ids[0], -1)
        
        # top rung is the goal / light source
        sim.send_light_source(body_id=rung_ids[-1])

        return touch_ids

class Individual:  
    def __init__(self, id_, genome, num_legs, L, R, S, eval_time):
        self.num_legs = num_legs
        self.L = L
        self.R = R
        self.S = S
        self.eval_time = eval_time
        # rows=num_sensors=(lower leg touch sensors + light sensor) , cols=num_motors=number of joints
        self.genome = genome
        self.id_ = id_
        
    def start_evaluation(self, env, play_blind=True, play_paused=False):
        self.sim = pyrosim.Simulator(play_paused=play_paused, eval_time=self.eval_time,
                                     play_blind=play_blind, dt=0.05) # default dt=0.05
        self.tids = env.send_to(self.sim)
        robot = Robot(self.sim, weights=self.genome, num_legs=self.num_legs, L=self.L, R=self.R, S=self.S)
        self.position_sensor_id = robot.p4
        self.distance_sensor_id = robot.l5 # distance from light source
        self.sim.assign_collision(robot.group, env.group)
        self.sim.start()
    
    def compute_fitness(self):
        self.sim.wait_to_finish()
        
        light_data = self.sim.get_sensor_data(sensor_id=self.distance_sensor_id)
        # max of light sensor leads to robots that get high once and then fall down and twitch
        max_fitness = light_data.max()
        # last reading of sensor leads to robots that get high and cling for dear life.
        last_fitness = light_data[-1]
        mean_max_last_fitness = (max_fitness + last_fitness) / 2
        
        rung_fitness = sum([self.sim.get_sensor_data(sensor_id=tid).max() for tid in self.tids])
#         print(len(self.tids))
#         print(rung_fitness)
        
        del self.sim # so deepcopy does not copy the simulator
        return rung_fitness + last_fitness


class Evaluator:
    '''
    A configurable fitness function that evaluates a population of solutions.
    '''
    def __init__(self, num_legs, L, R, S, eval_time, env, num_hidden):
        self.num_legs = num_legs
        self.L = L
        self.R = R
        self.S = S
        self.eval_time = eval_time
        self.env = env
        self.num_hidden = num_hidden
        
    def __call__(self, solutions, play_blind=True, play_paused=False):
        '''
        solutions: 2d array of params: (pop_size, num_params)
        '''
        fitnesses = np.zeros(len(solutions)) # fitnesses
        
        indivs = []
        for i in range(len(solutions)):
            genome = solutions[i].reshape((self.num_legs + 1, self.num_legs * 2))
            indiv = Individual(i, genome, self.num_legs, self.L, self.R, self.S, self.eval_time)
            indivs.append(indiv)

        for indiv in indivs:
            indiv.start_evaluation(self.env, play_blind=play_blind, play_paused=play_paused)

        for i, indiv in enumerate(indivs):
            fitnesses[i] = indiv.compute_fitness()

        return fitnesses
        

# results: 
# rung 3. pop 20, spacing 1*L, gens 200, fit last.
def train():
    # hyperparameter configuration
    L = 0.1
    hp = Hyperparams(
        L=L, # leg length
        R=L / 5, # leg radius
        S=L / 2, # body radius
        eval_time=500, # number of timesteps
#         pop_size=40, # population size
        pop_size=20, # population size
        num_gens=100, # number of generations
#         num_envs=4, # number of environments each individual will be evaluated in
        num_legs=8,
        num_hidden=3,
        # ladder
        length=L * 10,
        width=L * 5,
        thickness=L / 5,
        spacing=1 * L, 
        y_offset=L * 5,
    )
    hp.num_params = (hp.num_legs + 1) * (hp.num_legs * 2) # (leg sensors + light sensor) * (joints)
    
#     env = BlockEnv(id_=1, L=hp.L)
    env = LadderEnv(length=hp.length, width=hp.width, thickness=hp.thickness, spacing=hp.spacing, y_offset=hp.y_offset)
    evaluator = Evaluator(hp.num_legs, hp.L, hp.R, hp.S, hp.eval_time, env, hp.num_hidden)
    
    # defines genetic algorithm solver
    solver = es.SimpleGA(hp.num_params,                # number of model parameters
               sigma_init=0.5,        # initial standard deviation
               popsize=hp.pop_size,   # population size
               elite_ratio=0.2,       # percentage of the elites
               forget_best=False,     # forget the historical best elites
               weight_decay=0.00,     # weight decay coefficient
              )
    
    # defines CMA-ES algorithm solver
#     solver = es.CMAES(hp.num_params,
#               popsize=hp.pop_size,
#               weight_decay=0.0,
#               sigma_init=0.5
#           )    
    
    history = []
    for i in range(hp.num_gens):
        solutions = solver.ask() # shape: (pop_size, num_params)
        fitnesses = evaluator(solutions, play_blind=True, play_paused=False)
        solver.tell(fitnesses)
        print(f'{i} fitnesses: {np.sort(fitnesses)[::-1]}')
        result = solver.result() # first element is the best solution, second element is the best fitness
        history.append(result[1])
    
    print('history:', history)
    params = solver.result()[0]
    solutions = np.expand_dims(params, axis=0)
    fits = evaluator(solutions, play_blind=False, play_paused=False)    
    save_model('robot.pkl', hp, params, evaluator)
    
    
def play():
    hp, params, evaluator = load_model('robot.pkl')
    solutions = np.expand_dims(params, axis=0)
    evaluator.eval_time = 700
    evaluator(solutions, play_blind=False, play_paused=False)
    
    
def save_model(filename, hp, params, evaluator):
    with open(filename, 'wb') as fh:
        pickle.dump((hp, params, evaluator), fh)
        

def load_model(filename):
    with open(filename, 'rb') as fh:
        hp, params, evaluator = pickle.load(fh)
        
    return hp, params, evaluator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train and evaluate')
    parser.add_argument('action', choices=['train', 'play'], default='train')
    args = parser.parse_args()
    if args.action == 'train':
        train()
    elif args.action == 'play':
        play()
        
