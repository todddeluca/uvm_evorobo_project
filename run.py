


import argparse
import copy
import datetime
import math
import numpy as np
import pickle
import random
import pprint

import pyrosim
import es # from es import SimpleGA, CMAES, PEPG, OpenES

from robot import Robot
from environment import (LadderEnv, StairsEnv, PhototaxisEnv, AngledLadderEnv, AngledLatticeEnv,
                         SpatialScaffoldingStairsEnv)
from strategy import ParallelHillClimber, AFPO


class Hyperparam:
    '''Dummy class for accessing hyperparams as attributes. Instead of using a dict.'''
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])


# class Hyperparameters(dict):
#     def translate(self, **kwargs):
#         return Hyperparameters([(new_key, self[old_key]) for new_key, old_key in kwargs.items()])


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
            
        # y-position sensor fitness
        y_fitness = self.sim.get_sensor_data(sensor_id=self.position_sensor_id, svi=1)[-1]
        
        # x position fitness penalizes going around the obstacle.
        x_max_dist = np.abs(self.sim.get_sensor_data(sensor_id=self.position_sensor_id, svi=0)).max()
        x_fitness = -1 * (x_max_dist - max(abs(self.env.x_min), abs(self.env.x_max)))**2 # 
        
        # vestibular sensor fitness. angle between body and vertical.
        v_data = self.sim.get_sensor_data(sensor_id=self.v_id)
#         print(f'vestibular data: {v_data}')
        v_data = np.abs(v_data)
        v_fitness = 1 + v_data[-1] / np.pi # v_data in [0, pi?]

        # light sensor fitness
        light_data = self.sim.get_sensor_data(sensor_id=self.distance_sensor_id)
        # last reading of sensor leads to robots that get high and cling for dear life.
        light_fitness = light_data[-1]
        
        # rung touch sensor fitness
        rung_fitness = sum([self.sim.get_sensor_data(sensor_id=tid).max() for tid in self.tids]) / 10
        
        del self.sim # so deepcopy does not copy the simulator
#         fitness = 1.0 * last_fitness + 0.0 * rung_fitness + 0.0 * max_fitness + 0.0 * y_fitness
#         fitness = light_fitness / v_fitness
        fitness = light_fitness
        return fitness


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
        checkpoint_step=10,
        # robot
        L=L, # leg length
        R=L / 5, # leg radius
        S=L / 2, # body radius
        num_legs=5,
        num_hidden=4, # 6
        num_hl=0, # number of hidden layers
        use_proprio=True,
        use_vestib=True,
        front_angle=np.pi/2, # pi/2 = face the y-direction
        obstacle='scaffolding_stairs',
#         obstacle='stairs',
#         obstacle='angled_lattice',
#         obstacle='angled_ladder',
        # ladder
        length=L * 10,
        width=L * 5,
        thickness=L / 5,
        spacing=1 * L, 
        y_offset=L * 5,
        # stairs
        num_stairs=16,
        stair_width=L * 80,
        stair_depth=L, # when angle=pi/2, depth == rung spacing
        stair_thickness=L / 2.5,
        stair_angle=np.pi / 16,
        stair_y_offset=L * 2,
        stair_max_rise=L / 2.5,
        stair_initial_temp=1,
        stair_temp_scale=4,
        # angled ladder
        ladder_num_rungs=20,
        ladder_angle=np.pi / 8,
        ladder_width=L * 80,
        ladder_thickness=L / 5,
        ladder_y_offset=L * 2,
        ladder_spacing=L,
        # angled lattice
        lat_num_rungs=8, # 20,
        lat_num_rails=6, # 80,
        lat_rung_spacing=L,
        lat_rail_spacing=L * 2, # L*80,
        lat_thickness=L / 10, # L / 5,
        lat_angle=np.pi / 3, # np.pi / 16, # np.pi / 2 / 4,
        lat_y_offset=L * 2,
        # Scaffolding
#         scaffolding_kind = 'linear',
        scaffolding_kind = None,
        scaffolding_initial_angle = 0,
        # Evolution Strategy
        strategy='phc',
#         strategy='ga',
#         strategy='afpo',
#         strategy='cmaes',
#         num_novel=1, # 0, # 1, # 4, # number of new lineages per generation (AFPO)
#         num_novel=0, # use with pop_size=1 for testing.
#         decay=0.0000, # weight decay
#         mutation='evorobo', # change one random param, sigma=param
        mutation='noise', # change all params, sigma~=hp.sigma_init*(sigma_decay**generation)
#         elite_ratio=0.2,
#         eval_time=200, # number of timesteps
        eval_time=2000, # number of timesteps
        pop_size=1, # population size
#         pop_size=256, # ~same # of lineages as 1000gen afpo
#         pop_size=1, # population size
        # 2 = ~28-29 sec
        # 4 = ~24-27 sec
        # 8 = ~24
        # 16 = ~25
        # 32 = ~24-25
        max_parallel=8, # 32, # max num sims to run simultaneously
        num_gens=1000, # number of generations
#         num_gens=10,
        sigma_init=0.1,
#         num_envs=4, # number of environments each individual will be evaluated in
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
    
        if hp['obstacle'] == 'scaffolding_stairs':
            env = SpatialScaffoldingStairsEnv(
                num_stairs=hp['num_stairs'], depth=hp['stair_depth'], 
                width=hp['stair_width'], thickness=hp['stair_thickness'],
                y_offset=hp['stair_y_offset'], max_rise=hp['stair_max_rise'],
                temp=hp['stair_initial_temp'], temp_scale=hp['stair_temp_scale'])
        elif hp['obstacle'] == 'stairs':
            env = StairsEnv(
                num_stairs=hp['num_stairs'], depth=hp['stair_depth'], 
                width=hp['stair_width'], thickness=hp['stair_thickness'],
                angle=hp['stair_angle'], y_offset=hp['stair_y_offset'])
            hp['scaffolding_final_angle'] = hp['stair_angle']
        elif hp['obstacle'] == 'angled_lattice':
            env = AngledLatticeEnv(num_rungs=hp['lat_num_rungs'], num_rails=hp['lat_num_rails'], 
                                   rung_spacing=hp['lat_rung_spacing'], rail_spacing=hp['lat_rail_spacing'], 
                                   thickness=hp['lat_thickness'], angle=hp['lat_angle'], y_offset=hp['lat_y_offset'])
            hp['scaffolding_final_angle'] = hp['lat_angle']
        elif hp['obstacle'] == 'angled_ladder':
            env = AngledLadderEnv(num_rungs=hp['ladder_num_rungs'], spacing=hp['ladder_spacing'], width=hp['ladder_width'],
                                  thickness=hp['ladder_thickness'], angle=hp['ladder_angle'], y_offset=hp['ladder_y_offset'])
            hp['scaffolding_final_angle'] = hp['ladder_angle']
        else:
#             env = LadderEnv(length=hp.length, width=hp.width, thickness=hp.thickness, spacing=hp.spacing, y_offset=hp.y_offset)
            env = PhototaxisEnv(id_=1, L=hp['L'])
        evaluator = Evaluator(env=env, **hp)
        state['env'] = copy.deepcopy(env)
        state['evaluator'] = copy.deepcopy(evaluator)
        state['history'] = []
        state['gen'] = -1
        
        # defines genetic algorithm solver
        if False: # initialize population from best params
            solutions = np.tile(params, (hp['pop_size'], 1))
        else:
            solutions = None
        if hp['strategy'] == 'ga':
            solver = es.SimpleGA(hp['num_params'], sigma_init=hp['sigma_init'], 
                                 popsize=hp['pop_size'], elite_ratio=hp['elite_ratio'], 
                                 forget_best=False, weight_decay=hp['decay'])
        elif hp['strategy'] == 'afpo':
            solver = AFPO(hp['num_params'], hp['pop_size'], sigma_init=hp['sigma_init'], sols=solutions, 
                          num_novel=hp['num_novel'])
        elif hp['strategy'] == 'cmaes':
            solver = es.CMAES(hp['num_params'], popsize=hp['pop_size'], weight_decay=hp['decay'], sigma_init=hp['sigma_init'])    
        elif hp['strategy'] == 'phc':
            solver = ParallelHillClimber(hp['num_params'], hp['pop_size'], sigma_init=hp['sigma_init'],
                                         mutation=hp['mutation'])

    

    print('Experiment', hp["exp_id"])
    print('legs:', hp["num_legs"], 'hidden units:', hp["num_hidden"], 
          'hidden layers:', hp["num_hl"], 'params:', hp["num_params"])
    
    env = copy.deepcopy(state['env'])
#     evaluator = copy.deepcopy(state['evaluator'])
#     solver = copy.deepcopy(state['solver'])
        
    # start or restart evolution
    for gen in range(state['gen'] + 1, hp['num_gens']):
        gen_start_time = datetime.datetime.now()
        state['gen'] = gen
        story = {'gen': gen} # track history
        solutions = solver.ask() # shape: (pop_size, num_params)
        
        if hp.get('scaffolding_kind') == 'linear':
            if hp['num_gens'] == 1:
                angle = hp['scaffolding_final_angle']
            else:
                angle_delta = (hp['scaffolding_final_angle'] - hp['scaffolding_initial_angle']) / (hp['num_gens'] - 1)
                angle = hp['scaffolding_initial_angle'] + angle_delta * gen
            env.angle = angle # this makes my inner functional programmer cry
            evaluator = Evaluator(env=env, **hp)

        fitnesses = evaluator(solutions, play_blind=False, play_paused=True)
        story['fitnesses'] = copy.deepcopy(fitnesses)
        solver.tell(fitnesses)
        print(f'============\ngen: {gen}')
        if hasattr(env, 'angle'):
            story['angle'] = env.angle
            print('angle (radians):', story['angle'], 'angle (degrees):', story['angle'] * 180 / np.pi)
        print(f'fitnesses: {np.sort(fitnesses)[::-1]}')
        result = solver.result() # first element is the best solution, second element is the best fitness
        story['result'] = copy.deepcopy(result) # too heavy b/c of params?
        for attr in ['fits', 'ages', 'lineage', 'front_idx', 'best_idx', 'best_age', 'best_fit', 'sigma']: # AFPO
            if hasattr(solver, attr):
                story[attr] = copy.deepcopy(getattr(solver, attr))
                
        if 'front_idx' in story:
            print('front size:', len(story['front_idx']))
            print('front ages:', story['ages'][story['front_idx']])
            print('front fits:', story['fits'][story['front_idx']])
            print('front lineage:', story['lineage'][story['front_idx']])
            print('front_idx:', story['front_idx'])
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
    if len(etc) > 0:
        solver, *etc = etc
        print('best fitness:', solver.result()[1])
    solutions = np.expand_dims(params, axis=0)
#     evaluator.eval_time = 4000
#     evaluator.env.num_stairs = 10
    if not hasattr(evaluator, 'max_parallel'):
        evaluator.max_parallel = None
        
    # print hyperparams
    if isinstance(hp, Hyperparam):
        for attr in dir(hp):
            if not attr.startswith('_'):
                print(f'{attr}: {getattr(hp, attr)}')  
    else:
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
        
