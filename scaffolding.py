'''

'''


import argparse
import copy
import datetime
import math
import numpy as np
import pickle
import random
import pprint

import pyrosim

from robot import Robot
from environment import SpatialScaffoldingStairsEnv


class ParallelHillClimber:
    '''
    Evaluate a population in parallel
    '''
    def __init__(self, num_params, pop_size, 
                 sigma_init=0.1, sigma_decay=0.999, sigma_limit=0.01, 
                 seed=None, mutation='noise'):
        '''
        mutation: 'noise' adds gaussian noise to every param. 
          'evorobo' adds noise to one randomly chosen parameter, using sigma=abs(param),
           and clips the parameter.
        '''
        self.seed = seed
        self.num_params = num_params
        self.pop_size = pop_size
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.sigma = self.sigma_init
        self.mutation = mutation
        
        self.sols = None
        self.fits = None
        self.next_sols = None
        self.best_params = None
        self.best_fit = None
        self.best_idx = None

    def ask(self):
        # mutation from class chooses a single weight at random,
        # adds gaussian noise using sigma=weight, and clips to [-1, 1]
        # new_weight = random.gauss(self.genome[i, j], math.fabs(self.genome[i, j]))
        # self.genome[i, j] = np.clip(new_weight, -1, 1)

        if self.sols is None:
            noise = np.random.randn(self.pop_size, self.num_params) * self.sigma
            self.next_sols = noise
        elif self.mutation == 'noise':
            noise = np.random.randn(self.pop_size, self.num_params) * self.sigma
            self.next_sols = self.sols + noise
        elif self.mutation == 'evorobo':
            # select a random param to mutate from each individual
            idx = (np.arange(self.pop_size), np.random.choice(self.num_params, size=self.pop_size))
            # masked noise is based on the magnitude of each parameter
            mask = np.zeros((self.pop_size, self.num_params))
            mask[idx] = 1
            sigma = np.abs(self.sols)
            noise = np.random.randn(self.pop_size, self.num_params) * sigma * mask
            # add noise and clip params to [-1, 1]
            self.next_sols = np.clip(self.sols + noise, -1, 1)
            # np.clip(self.sols[idx] + np.random.randn(self.pop_size) * sigmas, -1, 1)
#             print((self.sols - self.next_sols))
#             print((self.sols - self.next_sols)[np.nonzero(self.sols - self.next_sols)])
        else:
            raise Exception('unrecognized mutation method:', self.mutation)
            
        # decay sigma
        if self.sigma > self.sigma_limit:
            self.sigma = max(self.sigma_limit, self.sigma * self.sigma_decay)
            
        return self.next_sols
                
    def tell(self, fits):
        '''
        compare fitnesses to previous fitnesses. replace if better.
        '''
        if self.sols is None:
            # initial iteration, initialize fitness
            self.sols = self.next_sols
            self.fits = fits
        else:
            # update population with improved indivs
            better_idx = (fits > self.fits)
            print(f'{better_idx.sum()} solutions improved')
            self.sols[better_idx] = self.next_sols[better_idx]
            self.fits[better_idx] = fits[better_idx]
            
        self.best_idx = self.fits.argmax()
        self.best_params = self.sols[self.best_idx]
        self.best_fit = self.fits[self.best_idx]
        print(f'best_fit: {self.best_fit:.4} best_idx: {self.best_idx}')

    def result(self):
        return (self.best_params, self.best_fit, self.best_idx)


class Individual:  
    def __init__(self, id_, genome, num_legs, L, R, S, eval_time, num_hidden, num_hl, 
                 use_proprio=False, use_vestib=False, front_angle=0):
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
        self.use_proprio = use_proprio
        self.use_vestib = use_vestib
        self.front_angle = front_angle
        
    def start_evaluation(self, env, play_blind=True, play_paused=False):
        self.env = env # save for fitness
        self.sim = pyrosim.Simulator(play_paused=play_paused, eval_time=self.eval_time,
                                     play_blind=play_blind, dt=0.025) # default dt=0.05
        self.tids = env.send_to(self.sim)
        robot = Robot(self.sim, weights=self.genome, num_legs=self.num_legs,
                      L=self.L, R=self.R, S=self.S, num_hidden=self.num_hidden, 
                      num_hidden_layers=self.num_hl, 
                      use_proprio=self.use_proprio, use_vestib=self.use_vestib,
                      front_angle=self.front_angle)
        self.position_sensor_id = robot.p4
        self.distance_sensor_id = robot.l5 # distance from light source
        self.v_id = robot.v_id # body vestibular sensor
        self.sim.assign_collision(robot.group, env.group)
        self.sim.start()

    def compute_fitness(self):
        self.sim.wait_to_finish()
            
        # distance to trophy fitness
        x_pos = self.sim.get_sensor_data(sensor_id=self.position_sensor_id, svi=0)[-1]
        y_pos = self.sim.get_sensor_data(sensor_id=self.position_sensor_id, svi=1)[-1]
        z_pos = self.sim.get_sensor_data(sensor_id=self.position_sensor_id, svi=2)[-1]
        robot_pos = np.array([x_pos, y_pos, z_pos])
        goal_pos = np.array(self.env.trophy_pos)
        dist = np.sqrt(np.dot(robot_pos - goal_pos, robot_pos - goal_pos))
        dist_fitness = 1 / (dist + 1)
        del self.sim # so deepcopy does not copy the simulator
        return dist_fitness


class Evaluator:
    '''
    A configurable fitness function that evaluates a population of solutions.
    '''
    def __init__(self, num_legs, L, R, S, eval_time, env, num_hidden, num_hl, max_parallel=None,
                 use_proprio=False, use_vestib=False, front_angle=0, **kwargs):
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
        self.use_proprio = use_proprio
        self.use_vestib = use_vestib
        self.front_angle = front_angle
        
    def __call__(self, solutions, play_blind=True, play_paused=False):
        '''
        solutions: 2d array of params: (pop_size, num_params)
        '''
        # for backwards compatibility with saved evaluators missing these attributes
        if not hasattr(self, 'num_hl'):
            self.num_hl = 0
        if not hasattr(self, 'use_proprio'):
            self.use_proprio = False
        if not hasattr(self, 'use_vestib'):
            self.use_vestib = False
        if not hasattr(self, 'front_angle'):
            self.front_angle = 0
            
        fitnesses = np.zeros(len(solutions)) # fitnesses
        
        # process solutions in batches of size batch_size
        batch_size = len(solutions) if self.max_parallel is None else self.max_parallel
        for start_i in range(0, len(solutions), batch_size):
            indivs = []
            for i in range(start_i, min(start_i + batch_size, len(solutions))):
                genome = solutions[i]
                indiv = Individual(i, genome, self.num_legs, self.L, self.R, self.S, self.eval_time,
                                   self.num_hidden, self.num_hl, 
                                   use_proprio=self.use_proprio, use_vestib=self.use_vestib,
                                   front_angle=self.front_angle)
                indivs.append(indiv)

            for indiv in indivs:
                indiv.start_evaluation(self.env, play_blind=play_blind, play_paused=play_paused)

            for i, indiv in enumerate(indivs):
                fitnesses[start_i + i] = indiv.compute_fitness()

        return fitnesses


def make_hyperparameters():
    '''
    Return a dictionary of the experimental hyperparameters.
    '''
    exp_id = 'exp_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    L = 0.1
    hp = dict(
        # experiment
        exp_id=exp_id, # experiment id
        checkpoint_step=100,
        # robot
        L=L, # leg length
        R=L / 5, # leg radius
        S=L / 2, # body radius
        num_legs=5,
        num_hidden=3, # 6
        num_hl=0, # number of hidden layers
        use_proprio=True,
        use_vestib=True,
        front_angle=np.pi/2, # pi/2 = face the y-direction
        # stairs
        num_stairs=16,
        stair_width=L * 80,
        stair_depth=L, # when angle=pi/2, depth == rung spacing
        stair_thickness=L / 2.5,
        stair_y_offset=L * 2,
        stair_max_rise=L / 2.5, # L/2.5 seems hard but doable. Maybe 2*L/2.5 is stretch goal
        stair_initial_temp=1,
        stair_temp_scale=4,
        # Evolution Strategy
        strategy='phc',
#         mutation='evorobo', # change one random param, sigma=param
        mutation='noise', # change all params, sigma~=hp.sigma_init*(sigma_decay**generation)
        sigma_init=0.1, # mutation noise
#         eval_time=200, # number of timesteps
        eval_time=2000, # number of timesteps
#         pop_size=64, # population size
        pop_size=8, # population size
        max_parallel=8, # max num sims to run simultaneously
#         num_gens=1000, # number of generations
        num_gens=10,
    )
    
    # hyperparameter configuration
    # calculate number of params
    # make a list of nodes in each layer and calculate the weights between the layers
    # input node count: proprioceptive sensors + touch sensors + vestigial sensor + bias
    # hidden node count: hidden nodes + bias
    # output node count: joints/motors
    nodes = np.array([3 * hp['num_legs'] + 2] + [hp['num_hidden'] + 1] * hp['num_hl'] + [hp['num_legs'] * 2])
    hp['num_params'] = (nodes[:-1] * nodes[1:]).sum()        
    return hp

    
def train(filename=None, play_paused=False):

    # case 1: load full model and resume training
    # case 2: create new model and start training
    # case 3: not implemented. load model and use new hp, env, etc.

    # load model
    if filename is not None:
        hp, params, evaluator, solver, history, state = load_model(filename)
    else:
        hp = make_hyperparameters()
        state = {} # training state. Used to restore training and save training history.
    
        env = SpatialScaffoldingStairsEnv(
            num_stairs=hp['num_stairs'], depth=hp['stair_depth'], 
            width=hp['stair_width'], thickness=hp['stair_thickness'],
            y_offset=hp['stair_y_offset'], max_rise=hp['stair_max_rise'],
            temp=hp['stair_initial_temp'], temp_scale=hp['stair_temp_scale'])

        evaluator = Evaluator(env=env, **hp)
        state['env'] = copy.deepcopy(env)
        state['evaluator'] = copy.deepcopy(evaluator)
        state['history'] = []
        state['gen'] = -1
        
        solver = ParallelHillClimber(hp['num_params'], hp['pop_size'], 
                                     sigma_init=hp['sigma_init'], mutation=hp['mutation'])

    print('Experiment', hp["exp_id"])
    print('legs:', hp["num_legs"], 'hidden units:', hp["num_hidden"], 
          'hidden layers:', hp["num_hl"], 'params:', hp["num_params"])
    
    env = copy.deepcopy(state['env'])
        
    # start or restart evolution
    for gen in range(state['gen'] + 1, hp['num_gens']):
        gen_start_time = datetime.datetime.now()
        state['gen'] = gen
        story = {'gen': gen} # track history
        solutions = solver.ask() # shape: (pop_size, num_params)
        fitnesses = evaluator(solutions, play_blind=True, play_paused=False)
        story['fitnesses'] = copy.deepcopy(fitnesses)
        solver.tell(fitnesses)
        print(f'============\ngen: {gen}')
        print(f'fitnesses: {np.sort(fitnesses)[::-1]}')
        result = solver.result() # first element is the best solution, second element is the best fitness
        story['result'] = copy.deepcopy(result) # too heavy b/c of params?
        for attr in ['fits', 'ages', 'lineage', 'best_idx', 'best_age', 'best_fit', 'sigma']:
            if hasattr(solver, attr):
                story[attr] = copy.deepcopy(getattr(solver, attr))
                
        gen_end_time = datetime.datetime.now()
        gen_time = gen_end_time - gen_start_time
        story['gen_time'] = gen_time
        print('gen_time:', gen_time)
        story['result_fitness'] = copy.deepcopy(result[1])
        state['history'].append(story)
        if hp['checkpoint_step'] is not None and (gen + 1) % hp['checkpoint_step'] == 0:
            print('Saving checkpoint at gen', gen)
            params = solver.result()[0]
            save_model('robot.pkl', hp, params, evaluator, solver, state['history'], state)
            save_model('experiments/' + hp['exp_id'] + '_robot.pkl', hp, params, evaluator, solver, state['history'], state)
    
    print('state:')
    pprint.pprint(state)
    params = solver.result()[0]
    solutions = np.expand_dims(params, axis=0)
    fits = evaluator(solutions, play_blind=False, play_paused=play_paused)    
    save_model('robot.pkl', hp, params, evaluator, solver, state['history'], state)
    save_model('experiments/' + hp['exp_id'] + '_robot.pkl', hp, params, evaluator, solver, state['history'], state)
    
    
def play(filename=None, play_paused=False):
    if filename is None:
        filename = 'robot.pkl'
        
    hp, params, evaluator, *etc = load_model(filename)
    solutions = np.expand_dims(params, axis=0)
#     evaluator.eval_time = 4000
#     evaluator.env.num_stairs = 10

    # print hyperparams
    for k, v in hp.items():
        print(k, ':', v)

    evaluator(solutions, play_blind=False, play_paused=play_paused)
    
    
def save_model(filename, *model):
    with open(filename, 'wb') as fh:
        pickle.dump(model, fh)
        

def load_model(filename):
    with open(filename, 'rb') as fh:
        model = pickle.load(fh)
        
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train and evaluate')
    parser.add_argument('action', choices=['train', 'play'], default='train')
    parser.add_argument('--restore', metavar='FILENAME', help='load and use a saved model')
    parser.add_argument('--play-paused', default=False, action='store_true')
    args = parser.parse_args()
    
    print('restore filename:', args.restore)
#     if args.restore is not None:
#         hp, params, evaluator = load_model(args.restore)
        
    if args.action == 'train':
        for i in range(1):
            print(f'experiment #{i}')
            train(args.restore, play_paused=args.play_paused)
    elif args.action == 'play':
        play(args.restore, play_paused=args.play_paused)
        
