


import argparse
import copy
import datetime
import math
import numpy as np
import pickle
import random
import pyrosim
import es # from es import SimpleGA, CMAES, PEPG, OpenES

from robot import Robot
from environment import LadderEnv, StairsEnv, PhototaxisEnv
from strategy import ParallelHillClimber, AFPO


class Hyperparams:
    '''Dummy class for accessing hyperparams as attributes. Instead of using a dict.'''
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])



class Individual:  
    def __init__(self, id_, genome, num_legs, L, R, S, eval_time, num_hidden, num_hl):
        self.num_legs = num_legs
        self.L = L
        self.R = R
        self.S = S
        self.eval_time = eval_time
        # rows=num_sensors=(lower leg touch sensors + light sensor) , cols=num_motors=number of joints
        self.genome = genome
        self.id_ = id_
        self.num_hidden = num_hidden
        self.num_hl = num_hl
        
    def start_evaluation(self, env, play_blind=True, play_paused=False):
        self.sim = pyrosim.Simulator(play_paused=play_paused, eval_time=self.eval_time,
                                     play_blind=play_blind, dt=0.025) # default dt=0.05
        self.tids = env.send_to(self.sim)
        robot = Robot(self.sim, weights=self.genome, num_legs=self.num_legs,
                      L=self.L, R=self.R, S=self.S, num_hidden=self.num_hidden, 
                      num_hidden_layers=self.num_hl)
        self.position_sensor_id = robot.p4
        self.distance_sensor_id = robot.l5 # distance from light source
        self.v_id = robot.v_id # body vestibular sensor
        self.sim.assign_collision(robot.group, env.group)
        self.sim.start()

    def compute_fitness(self):
        self.sim.wait_to_finish()
            
        # y-position sensor fitness
        y_fitness = self.sim.get_sensor_data(sensor_id=self.position_sensor_id, svi=1)[-1]
        
        # vestibular sensor fitness
        v_data = self.sim.get_sensor_data(sensor_id=self.v_id)
#         print(f'vestibular data: {v_data}')

        # light sensor fitness
        light_data = self.sim.get_sensor_data(sensor_id=self.distance_sensor_id)
        # max of light sensor leads to robots that get high once and then fall down and twitch
        max_fitness = light_data.max()
        # last reading of sensor leads to robots that get high and cling for dear life.
        last_fitness = light_data[-1]
        mean_max_last_fitness = (max_fitness + last_fitness) / 2
        
        # rung touch sensor fitness
        rung_fitness = sum([self.sim.get_sensor_data(sensor_id=tid).max() for tid in self.tids]) / 10
        
        del self.sim # so deepcopy does not copy the simulator
        return 1.0 * last_fitness + 0.0 * rung_fitness + 0.0 * max_fitness + 0.0 * y_fitness


class Evaluator:
    '''
    A configurable fitness function that evaluates a population of solutions.
    '''
    def __init__(self, num_legs, L, R, S, eval_time, env, num_hidden, num_hl, max_parallel=None):
        '''
        max_parallel: run up to max_parallel simulations simultaneously. If none, run all simulations 
        simultaneously. (My puny laptop struggles w/ >40).
        '''
        self.num_legs = num_legs
        self.L = L
        self.R = R
        self.S = S
        self.eval_time = eval_time
        self.env = env
        self.num_hidden = num_hidden
        self.num_hl = num_hl
        self.max_parallel = max_parallel
        
    def __call__(self, solutions, play_blind=True, play_paused=False):
        '''
        solutions: 2d array of params: (pop_size, num_params)
        '''
        fitnesses = np.zeros(len(solutions)) # fitnesses
        
        # process solutions in batches of size batch_size
        batch_size = len(solutions) if self.max_parallel is None else self.max_parallel
        for start_i in range(0, len(solutions), batch_size):
            indivs = []
            for i in range(start_i, min(start_i + batch_size, len(solutions))):
                genome = solutions[i]
                if not hasattr(self, 'num_hl'):
                    self.num_hl = 0
                indiv = Individual(i, genome, self.num_legs, self.L, self.R, self.S, self.eval_time,
                                   self.num_hidden, self.num_hl)
                indivs.append(indiv)

            for indiv in indivs:
                indiv.start_evaluation(self.env, play_blind=play_blind, play_paused=play_paused)

            for i, indiv in enumerate(indivs):
                fitnesses[start_i + i] = indiv.compute_fitness()

        return fitnesses
        


# results: 
# rung 3. pop 20, spacing 1*L, gens 200, fit last.
def train(filename=None):
    
    expid = 'exp_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f'Experiment {expid}')
    # hyperparameter configuration
    L = 0.1
    hp = Hyperparams(
        # robot
        L=L, # leg length
        R=L / 5, # leg radius
        S=L / 2, # body radius
        num_legs=5,
        num_hidden=3, # 6
        num_hl=0, # number of hidden layers
        # ladder
        length=L * 10,
        width=L * 5,
        thickness=L / 5,
        spacing=1 * L, 
        y_offset=L * 5,
        # stairs
        num_stairs=20,
        stair_width=L * 20,
        stair_depth=L, # when angle=pi/2, depth == rung spacing
        stair_thickness=L / 2.5,
        stair_angle=np.pi / 2 / 3.8,
        stair_y_offset=L * 2,
        # Evolutionary Strategy
#         strategy='phc',
#         strategy='ga',
        strategy='afpo',
#         strategy='cmaes',
        decay=0.0000, # weight decay
#         mutation='evorobo', # change one random param, sigma=param
        mutation='noise', # change all params, sigma~=hp.sigma_init*(sigma_decay**generation)
        elite_ratio=0.2,
        eval_time=2000, # number of timesteps
        pop_size=64, # population size
#         pop_size=256, # population size
        max_parallel=32, # max num sims to run simultaneously
#         num_gens=100, # number of generations
        num_gens=1000,
        sigma_init=0.1,
#         num_envs=4, # number of environments each individual will be evaluated in
    )
    
    # calculate number of params
    # make a list of nodes in each layer and calculate the weights between the layers
    # node counts are (num touch sensors + light sensor + bias + vestibular sensor), (num hidden nodes + bias), (num joints) 
#     nodes = np.array([hp.num_legs + 3] + [hp.num_hidden + 1] * hp.num_hl + [hp.num_legs * 2])
    # node counts are (num touch sensors + bias), (num hidden nodes + bias), (num joints) 
    nodes = np.array([hp.num_legs + 1] + [hp.num_hidden + 1] * hp.num_hl + [hp.num_legs * 2])
    hp.num_params = (nodes[:-1] * nodes[1:]).sum()        
    print(f'legs: {hp.num_legs}, hidden units: {hp.num_hidden}, hidden layers: {hp.num_hl}, params: {hp.num_params}')
    
#     env = PhototaxisEnv(id_=1, L=hp.L)
    env = StairsEnv(num_stairs=hp.num_stairs, 
                    depth=hp.stair_depth, width=hp.stair_width, thickness=hp.stair_thickness,
                    angle=hp.stair_angle, y_offset=hp.stair_y_offset)
#     env = LadderEnv(length=hp.length, width=hp.width, thickness=hp.thickness, spacing=hp.spacing, y_offset=hp.y_offset)
    evaluator = Evaluator(hp.num_legs, hp.L, hp.R, hp.S, hp.eval_time, env, hp.num_hidden, hp.num_hl, hp.max_parallel)
    
    if filename is not None:
        hp_unused, params, evaluator_unused = load_model(filename)
        solutions = np.tile(params, (hp.pop_size, 1))
    else:
        solutions = None
    
    
    # defines genetic algorithm solver
    if hp.strategy == 'ga':
        solver = es.SimpleGA(hp.num_params, sigma_init=hp.sigma_init, popsize=hp.pop_size, elite_ratio=hp.elite_ratio, forget_best=False, weight_decay=hp.decay)
    elif hp.strategy == 'afpo':
        solver = AFPO(hp.num_params, hp.pop_size, sigma_init=hp.sigma_init, sols=solutions)    
    elif hp.strategy == 'cmaes':
        solver = es.CMAES(hp.num_params, popsize=hp.pop_size, weight_decay=hp.decay, sigma_init=hp.sigma_init)    
    elif hp.strategy == 'phc':
        solver = ParallelHillClimber(hp.num_params, hp.pop_size, sigma_init=hp.sigma_init, mutation=hp.mutation)

    history = []
    for i in range(hp.num_gens):
        solutions = solver.ask() # shape: (pop_size, num_params)
        fitnesses = evaluator(solutions, play_blind=True, play_paused=False)
        solver.tell(fitnesses)
        print(f'gen: {i} fitnesses: {np.sort(fitnesses)[::-1]}')
        result = solver.result() # first element is the best solution, second element is the best fitness
        history.append(result[1])
    
    print('history:', history)
    params = solver.result()[0]
    solutions = np.expand_dims(params, axis=0)
    fits = evaluator(solutions, play_blind=False, play_paused=False)    
    save_model('robot.pkl', hp, params, evaluator)
    save_model(f'experiments/{expid}_robot.pkl', hp, params, evaluator)
    
    
def play(filename=None):
    if filename is None:
        filename = 'robot.pkl'
        
    hp, params, evaluator = load_model(filename)
    solutions = np.expand_dims(params, axis=0)
#     evaluator.eval_time = 4000
#     evaluator.env.num_stairs = 10
    if not hasattr(evaluator, 'max_parallel'):
        evaluator.max_parallel = None
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
    parser.add_argument('--restore', metavar='FILENAME', help='load and use a saved model')
    args = parser.parse_args()
    
    print('restore filename:', args.restore)
    if args.restore is not None:
        hp, params, evaluator = load_model(args.restore)
        
    if args.action == 'train':
        for i in range(1):
            print(f'experiment # {i}')
            train(args.restore)
    elif args.action == 'play':
        play(args.restore)
        
