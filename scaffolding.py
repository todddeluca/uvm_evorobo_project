
import argparse
import copy
import datetime
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import random
import pprint
import seaborn as sns

import pyrosim

from environment import send_trophy


'''
Spider:

A spider is a central sphere connected to n legs, where n=2k for some positive integer k. The legs are evenly distributed around the
xy plane bisecting the sphere.

L is leg length.
R is leg radius.
S is sphere radius.

Upper leg angle angle (around circumference of xy plane) of leg i is \theta_i = (2\pi / k) * (i + 1/2).
Upper leg: x=(S+0.5L)*cos(theta), y=(S+0.5L)*sin(theta), z=L+R. r1=cos(theta), r2=sin(theta), r3=0, r, g, b
Lower leg: x=(S+L)*cos(theta), y=(S+L)*sin(theta), z=0.5*L+R, r1=0, r2=0, r3=1, r, g, b

Body to upper leg joint: x=S*cos(theta), y=S*sin(theta), z=L+R, n1=-sin(theta), n2=cos(theta), n3=0, lo=-math.pi/2 , hi=math.pi/2
upper to lower leg joint: x=(S+L)*cos(theta), y=(S+L)*sin(theta), z=L+R, n1=-sin(theta), n2=cos(theta), n3=0, lo=-math.pi/2 , hi=math.pi/2
'''

class Robot:
    def __init__(self, weights, num_legs=4, L=1, R=1, S=1, hidden_layer_size=4, num_hidden_layers=0,
                 use_proprio=False, use_vestib=False, front_angle=0, one_front_leg=False, **kwargs):
        '''
        L: leg length
        R: leg radius
        S: body radius
        one_front_leg: if True, a single leg faces forward. if False, two legs face forward
        front_angle: in radians. if 0, the front leg/s are in the positive x direction.
          if pi/2, the front leg/s face the positive y direction.
        '''
        self.weights = weights
        self.num_legs = num_legs
        self.L = L
        self.R = R
        self.S = S
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.use_proprio = use_proprio
        self.use_vestib = use_vestib
        self.front_angle = front_angle
        self.one_front_leg = one_front_leg
        self.group = 'robot'

    def send_to(self, sim):
        body, upper_legs, lower_legs, joints = self.send_objects_and_joints(
            sim, self.num_legs, self.L, self.R, self.S)
        sensors, p4, l5, vid = self.send_sensors(
            sim, body, upper_legs, lower_legs, joints, 
            use_proprio=self.use_proprio, use_vestib=self.use_vestib)
        self.p4 = p4
        self.l5 = l5
        self.v_id = vid # vestibular sensor
        sensor_neurons, motor_neurons, hidden_layers, bias_neuron = self.send_neurons(
            sim, sensors, joints, self.hidden_layer_size, self.num_hidden_layers)
        self.send_synapses(sim, self.weights, sensor_neurons, motor_neurons, hidden_layers, bias_neuron)
        
    def send_objects_and_joints(self, sim, num_legs, L, R, S):
        o0 = sim.send_sphere(x=0, y=0, z=L+R, radius=S, r=0.5, g=0.5, b=0.5,
                             collision_group=self.group)
        upper_legs = []
        lower_legs = []
        joints = []
        for i in range(num_legs):
            theta = ((2 * np.pi / num_legs) * (i + (0 if self.one_front_leg else 0.5)) # (i+0.5): 2 symmetric front legs
                     + self.front_angle) # rotate the front to face y direction if pi/2
            upper = sim.send_cylinder(x=(S + 0.5 * L) * np.cos(theta), 
                                    y=(S + 0.5 * L) * np.sin(theta), 
                                    z=(L + R), length=L, radius=R, 
                                    r1=np.cos(theta), r2=np.sin(theta), r3=0,
                                    r=(1+np.cos(theta))/2, g=0, b=(1+np.sin(theta))/2,
                                     collision_group=self.group)
            upper_legs.append(upper)
            lower = sim.send_cylinder(x=(S + L) * np.cos(theta), 
                                    y=(S + L) * np.sin(theta),
                                    z=(0.5 * L + R), length=L, radius=R,
                                    r1=0, r2=0, r3=1,
                                    r=(1+np.cos(theta))/4, g=0, b=(1+np.sin(theta))/4,
                                     collision_group=self.group)
            lower_legs.append(lower)
            # body-to-upper-leg joint
            j0 = sim.send_hinge_joint(first_body_id=o0, second_body_id=upper, 
                                      x=S*np.cos(theta), y=S*np.sin(theta), z=L + R, 
                                      n1=-np.sin(theta), n2=np.cos(theta), n3=0, 
                                      lo=-math.pi/2 , hi=math.pi/2)
            # upper-to-lower-leg joint
            j1 = sim.send_hinge_joint(first_body_id=upper, second_body_id=lower, 
                                      x=(S+L)*np.cos(theta), y=(S+L)*np.sin(theta), z=L + R, 
                                      n1=-np.sin(theta), n2=np.cos(theta), n3=0, 
                                      lo=-math.pi/2 , hi=math.pi/2)
            joints += [j0, j1]

        return o0, upper_legs, lower_legs, joints
            
    def send_sensors(self, sim, body, upper_legs, lower_legs, joints, use_proprio=False, use_vestib=False):
        sensors = []
        
        if use_proprio:
            for joint in joints:
                sensors.append(sim.send_proprioceptive_sensor(joint))
            
        # front leg ray sensors
        # ...todo
        
        # lower limb touch sensors
        for lower in lower_legs:
            sensors.append(sim.send_touch_sensor(body_id=lower))
        
        # upper limb touch sensors
#         for upper in upper_legs:
#             sensors.append(sim.send_touch_sensor(body_id=upper))
        
        p4 = sim.send_position_sensor(body_id=body)
        l5 = sim.send_light_sensor(body_id=body)
#         sensors.append(l5)
        vid = sim.send_vestibular_sensor(body_id=body)
        if use_vestib:
            sensors.append(vid)
        
        return sensors, p4, l5, vid

    def send_neurons(self, sim, sensors, joints, hidden_layer_size, num_hidden_layers):
        bias_neuron = sim.send_bias_neuron()

        sensor_neurons = []
        for sensor in sensors:
            sensor_neurons.append(sim.send_sensor_neuron(sensor_id=sensor))
            
        motor_neurons = []
        for joint in joints:
            motor_neurons.append(sim.send_motor_neuron(joint_id=joint, tau=0.3))
            
        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_neurons = []
            for _ in range(hidden_layer_size):
                hidden_neurons.append(sim.send_hidden_neuron())
                
            hidden_layers.append(hidden_neurons)
            
        return sensor_neurons, motor_neurons, hidden_layers, bias_neuron
        
    def send_synapses(self, sim, weights, sensor_neurons, motor_neurons, hidden_layers, bias_neuron):
        layers = [sensor_neurons + [bias_neuron]] # add bias to input layer
        for layer in hidden_layers:
            layers.append(layer + [bias_neuron]) # add bias to hidden layers
        layers.append(motor_neurons)
        
        pairs = [] # source and target neuron pairs
        for i in range(len(layers) - 1):
            in_layer = layers[i]
            out_layer = layers[i + 1]
            for inp in in_layer:
                for out in out_layer:
                    pairs.append((inp, out))
            
        for i, (s, t) in enumerate(pairs):
            sim.send_synapse(source_neuron_id=s, target_neuron_id=t, weight=weights[i])
        

class SpatialScaffoldingStairsEnv:
    '''
    Floating Stairs with a light source trophy on top.
    
    Each stair is placed vertically relative to the previous stair. This is the
    rise of the stair. The rise of each stair increases from the first stair to
    the last stair according to the scaffolding schedule. This spatial change in rise
    is where the term "Spatial Scaffolding" comes from. The rise of the final stair 
    is always equal to the max_rise.
    '''
    def __init__(self, num_stairs, stair_depth, stair_width, stair_thickness, stair_y_offset, 
                 stair_max_rise, temp=0, stair_temp_scale=4, **kwargs):
        '''
        y_offset: y-axis position of the front of the initial stair.
        max_rise: the maximum rise from one stair to the next.
        temp: scaffolding temperature. Expected to be roughly in the range -1 to 1, but
          can range from -inf to inf. 0 means medium scaffolding, 
          >= 1 means about max_rise for every stair, and <= -1 means about zero
          rise for every stair except the last.
        temp_scale: >= 0. multiplied by temp to generate stairs. larger scale means more extreme
        stairs (either very flat or very steep). At 4, temp=1 corresponds to mostly
        max_rise stairs and temp=-1 corresponds to mostly no rise stairs (except last).
        '''
        self.num_stairs = num_stairs
        self.depth = stair_depth 
        self.width = stair_width
        self.y_offset = stair_y_offset
        self.thick = stair_thickness # stair thickness
        self.max_rise = stair_max_rise
        self.temp = temp
        self.temp_scale = np.abs(stair_temp_scale)
        self.group = 'env' # collision group
        
    def send_to(self, sim):
        # normalized stair position (0 to 1)
        positions = np.arange(self.num_stairs) / (self.num_stairs - 1) 
        # fraction of max_rise for each stair
        fracs = np.where(self.temp >= 0,
                         1 - np.power(1 - positions, np.exp(self.temp_scale * self.temp)),
                         np.power(positions, np.exp(-self.temp_scale * self.temp)))        
        rises = fracs * self.max_rise
        # x, y, z position of each stair
        stair_coords = [(0, 
                         self.y_offset + 0.5 * self.depth + i * self.depth,
                         0.5 * self.thick + rises[:(i+1)].sum()
                        ) for i in range(self.num_stairs)]
        stair_ids = []
        for x, y, z in stair_coords:
            sid = sim.send_box(x=x, y=y, z=z, 
                               length=self.depth, width=self.width, height=self.thick,
                               r=1, g=1, b=1,
                               collision_group=self.group)
            stair_ids.append(sid)
            
        # fix the stairs in place
        for sid in stair_ids:
            sim.send_fixed_joint(sid, -1)
        
        # the trophy is the light source
        self.trophy_pos = (0, stair_coords[-1][1], stair_coords[-1][2] + 0.5 * self.thick)
        send_trophy(sim, x=self.trophy_pos[0], y=self.trophy_pos[1], z=self.trophy_pos[2],
                    size=self.thick, collision_group=self.group)
        
        self.x_min = -self.width / 2
        self.x_max = self.width / 2 

    
class DecayingMutator:    
    def __init__(self, sigma_init=0.1, sigma_decay=0.999, sigma_limit=0.01, seed=None):
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.seed = seed
        self.sigma = self.sigma_init
        
    def mutate(self, genomes):
        noise = np.random.randn(*(genomes.shape)) * self.sigma
        self.sigma = max(self.sigma_limit, self.sigma * self.sigma_decay) # decay sigma           
        return genomes + noise


class EvoRoboMutator:
    def __init__(self, seed=None):
        self.seed = seed
        
    def mutate(self, genomes):
        num_genomes, num_params = genomes.shape
        # select a random param to mutate from each individual
        idx = (np.arange(num_genomes), np.random.choice(num_params, size=num_genomes))
        # masked noise is based on the magnitude of each parameter
        mask = np.zeros(genomes.shape)
        mask[idx] = 1
        sigma = np.abs(genomes)
        noise = np.random.randn(*(genomes.shape)) * sigma * mask
        # add noise and clip params to [-1, 1]
        return np.clip(genomes + noise, -1, 1)


class ScaffoldingPopulation:
    def __init__(self, num_params, pop_size, sigma, mutator, fit_thresh, temp_schedule=None, use_temp_param=False, 
                 seed=None, max_parallel=None, eval_time=None, **kwargs):
        '''
        temp_schedule: scaffolding schedule. list of temperatures. default [1].
        use_temp_param: if False, env.temp = scaffolding temp. 
          if True, env.temp = max(temp param, scaffolding temp)
        '''
        self.num_params = num_params
        self.pop_size = pop_size
        self.sigma = sigma
        self.mutator = mutator
        self.fit_thresh = fit_thresh
        self.temp_schedule = np.array(temp_schedule if temp_schedule is not None else [1])
        self.temp_thresh = temp_schedule[-1]
        self.use_temp_param = use_temp_param
        self.seed = seed
        self.max_parallel = max_parallel
        self.eval_time = eval_time
        self.kwargs = kwargs
        
        self.pop_idx = np.arange(self.pop_size)
        
    def reset(self):
        start = datetime.datetime.now()
        self.history = []
        self.gen = 0 # generation 0
        print('\n==================')
        print('self.gen:', self.gen)
        self.temps_idx = np.zeros(self.pop_size, dtype=int) # index of current scaffolding temp of indivs
        self.ages = np.zeros(self.pop_size, dtype=int) # total gens a lineage has evolved for
        self.dones = np.zeros(self.pop_size, dtype=bool) # if an indiv is done evolving
        self.fits = np.full(self.pop_size, np.nan, dtype=float)
        self.genomes = np.random.randn(self.pop_size, self.num_params) * self.sigma
        
        # update genome temp based on scaffolding temp and use_temp_param
        self.genomes[:, 0] = self.env_temps(self.genomes, self.temps_idx) 
        
        # evaluate fitness, replace as needed, and report replacements
        self.update_fitness()
        # find new done genomes, update dones, and report.
        self.update_dones()
        self.update_best()
        
        gen_time = datetime.datetime.now() - start
        self.history.append({'gen': self.gen, 
                             'gen_time': gen_time,
                             'fits_max': self.fits.max(),
                             'fits_min': self.fits.min(), 
                             'fits_mean': self.fits.mean(),
                             'fits_std': self.fits.std(), 
                             'fits_median': np.median(self.fits)
                            })
        print('gen_time:', gen_time)
        
    def step(self):
        '''
        for not-yet-done genomes: increase age, mutate genome, fix up temp, evaluate fitness. 
        for fitter genomes: replace genome, replace fitness
        for fitter (or all) genomes: update dones, update temps_idx and fitness
        for each individual, if not done, make next genome, evaluate fitness using curr scaffolding temp (and temp param). if fitness passes threshold: if temp passes threshold, done, else, increase scaffolding temp.
        '''
        start = datetime.datetime.now()
        self.gen += 1
        print('\n==================')
        print('self.gen:', self.gen)
        
        self.update_scaffolding_temp()
        self.update_genome_temp()
        
        nan_idx = self.pop_idx[np.isnan(self.fits)] # basically genomes with updated scaffolding temps
        nan_genomes = self.genomes[nan_idx]
        
        mutation_idx = self.pop_idx[(self.dones == False) & (np.isnan(self.fits) == False)]
        mutated_genomes = self.mutator.mutate(self.genomes[mutation_idx])
        mutated_genomes[:, 0] = self.env_temps(mutated_genomes, self.temps_idx[mutation_idx]) # fix temps

        next_idx = np.hstack([nan_idx, mutation_idx])
        next_genomes = np.vstack([nan_genomes, mutated_genomes])
        
        self.update_fitness(next_genomes, next_idx)
        self.update_dones()
        self.update_best()
        gen_time = datetime.datetime.now() - start
        self.history.append({'gen': self.gen, 
                             'gen_time': gen_time,
                             'fits_max': self.fits.max(),
                             'fits_min': self.fits.min(), 
                             'fits_mean': self.fits.mean(),
                             'fits_std': self.fits.std(), 
                             'fits_median': np.median(self.fits)
                            })
        print('gen_time:', gen_time)
                    
    def update_fitness(self, genomes=None, idx=None):
        '''
        genomes: genomes to be evaluated
        idx: index into full population of genomes to be compared
        and potentially replaced.
        '''
        genomes = self.genomes if genomes is None else genomes
        idx = self.pop_idx if idx is None else idx
        fits = self.evaluate(genomes)
        better_fits_idx = np.isnan(self.fits)[idx] | (fits > self.fits[idx])
        better_idx = self.pop_idx[idx][better_fits_idx]
        if len(better_idx) > 0:
            print('better_idx:', better_idx)
            self.genomes[better_idx] = genomes[better_fits_idx]
            self.fits[better_idx] = fits[better_fits_idx]
            self.ages[better_idx] = self.gen
            self.history.append({'gen': self.gen, 
                                 'better_idx': better_idx, 
                                 'better_genomes': self.genomes[better_idx],
                                 'better_fits': self.fits[better_idx],
                                 'better_temps_idx': self.temps_idx[better_idx],
                                 'better_ages': self.ages[better_idx],
                                })
        
    def update_dones(self):
        # done when fitness and temperature are good enough
        next_dones = (self.genomes[:, 0] >= self.temp_thresh) & (self.fits >= self.fit_thresh)
        new_dones_idx = self.pop_idx[(self.dones == False) & next_dones]
        if len(new_dones_idx) > 0:
            print('new_dones_idx:', new_dones_idx)
            self.dones = next_dones
            self.history.append({'gen': self.gen, 
                                 'dones_idx': self.pop_idx[self.dones],
                                 'new_dones_idx': new_dones_idx,
                                })
            
    def update_best(self):
        old_best_idx = self.best_idx if hasattr(self, 'best_idx') else None
        old_best_fit = self.best_fit if hasattr(self, 'best_fit') else None
        self.best_idx = self.fits.argmax()
        self.best_fit = self.fits[self.best_idx]
        self.best_genome = self.genomes[self.best_idx]
        self.best_age = self.ages[self.best_idx]
        if old_best_idx != self.best_idx or old_best_fit != self.best_fit:
            print('best_idx:', self.best_idx, 'best_fit:', self.best_fit,
                  'best_age:', self.best_age, 'best temps_idx:', self.temps_idx[self.best_idx],
                  'best scaffolding temp:', self.temp_schedule[self.temps_idx[self.best_idx]],
                  'best genome temp:', self.genomes[self.best_idx, 0], 
                 )
            self.history.append({'gen': self.gen, 
                                 'best_idx': self.best_idx,
                                 'best_fit': self.best_fit,
                                 'best_genome': self.best_genome,
                                 'best_age': self.best_age,
                                })
            
    def update_scaffolding_temp(self):
        # increment scaffolding when fitness is good enough
        inc_scaf_idx = self.pop_idx[(self.fits > self.fit_thresh) & (self.dones == False)]
        if len(inc_scaf_idx) > 0:
            print('inc_scaf_idx:', inc_scaf_idx)
            self.temps_idx[inc_scaf_idx] += 1
            self.history.append({'gen': self.gen, 
                                 'inc_scaf_idx': inc_scaf_idx,
                                })
            
    def update_genome_temp(self):
        # genomes whose scaffolding temp is greater than their genome temp
        # (because their scaffolding temp was increased)
        idx = self.pop_idx[(self.genomes[:, 0] < self.temp_schedule[self.temps_idx])]
        if len(idx) > 0:
            print('update_genome_temp_idx:', idx)
            self.genomes[idx, 0] = self.temp_schedule[self.temps_idx][idx]
            self.fits[idx] = np.nan
            self.history.append({'gen': self.gen, 
                                 'update_genome_temp_idx': idx,
                                })
                
    def env_temps(self, genomes, temps_idx):
        temps = self.temp_schedule[temps_idx] # scaffolding temperature of environment
        if self.use_temp_param:
            # genome can evolve to use a higher temperature
            return np.where(genomes[:, 0] > temps, genomes[:, 0], temps)
        else:
            return temps

    def play(self, idx=None, play_paused=False):
        # convert None to [int] so genomes[idx] has 2d shape.
        idx = self.best_idx if idx is None else idx
        try:
            # convert int-like things to an array so genomes[idx] has 2d shape.
            idx = [int(idx)]
        except:
            pass
        print('type(idx):', type(idx))
        idx = [idx] if isinstance(idx, int) else idx
        print('idx:', idx)
        print('fits:', self.fits[idx])
        print('ages:', self.ages[idx])
        print('dones:', self.dones[idx])
        print('genome temps:', self.genomes[idx, 0])
        print('temps_idx:', self.temps_idx[idx])
        print('scaffold temps:', self.temp_schedule[self.temps_idx[idx]])
        self.evaluate(self.genomes[idx], play_blind=False, play_paused=play_paused)
        
    def evaluate(self, genomes, play_blind=True, play_paused=False):
        robots = [Robot(weights=genomes[i, 1:], **self.kwargs) for i in range(len(genomes))]
        envs = [SpatialScaffoldingStairsEnv(temp=genomes[i, 0], **self.kwargs) for i in range(len(genomes))]
        fitnesses = np.zeros(len(genomes)) # fitnesses
        
        # process simulations in batches
        batch_size = len(genomes) if self.max_parallel is None else self.max_parallel
        for start_i in range(0, len(genomes), batch_size):
            end_i = min(start_i + batch_size, len(genomes))
            batch_robots = robots[start_i:end_i]
            batch_env = envs[start_i:end_i]
            batch_sims = [pyrosim.Simulator(play_paused=play_paused, eval_time=self.eval_time,
                                            play_blind=play_blind, dt=0.025) # default dt=0.05
                          for i in range(start_i, end_i)]
            
            # start a batch of simulations
            for robot, env, sim in zip(batch_robots, batch_env, batch_sims):
                env.send_to(sim)
                robot.send_to(sim)
                sim.assign_collision(robot.group, env.group)
                sim.start()
                
            for j, (robot, env, sim) in enumerate(zip(batch_robots, batch_env, batch_sims)):
                sim.wait_to_finish()
                # distance to trophy fitness
                x_pos = sim.get_sensor_data(sensor_id=robot.p4, svi=0)[-1]
                y_pos = sim.get_sensor_data(sensor_id=robot.p4, svi=1)[-1]
                z_pos = sim.get_sensor_data(sensor_id=robot.p4, svi=2)[-1]
                robot_pos = np.array([x_pos, y_pos, z_pos])
                goal_pos = np.array(env.trophy_pos)
                dist = np.sqrt(np.dot(robot_pos - goal_pos, robot_pos - goal_pos))
                fitness = 1 / (dist + 1)
                fitnesses[start_i + j] = fitness
                
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
        hidden_layer_size=3,
        num_hidden_layers=0, # number of hidden layers
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
        stair_temp_scale=4,
        # Scaffolding
        # fitness threshold tested with 16 stairs, no rise. 0.6935 gets within 1-2 stairs of trophy
        # 0.8091 touches trophy with leg, 0.8823 touches trophy with body.
        fit_thresh=0.7, # fitness victory condition
#         temp_schedule=[-1, -0.25, 0, 0.25, 1], # scaffolded temp schedule, last entry is temp victory condition
        temp_schedule=[1.], # scaffolded temperature schedule, last entry is temp victory condition
        use_temp_param=False, # True: use evolved temp in combination with scaffolding schedule temperature
        # Evolution Strategy
#         mutation='evorobo', # change one random param, sigma=param
        mutation='noise', # change all params, sigma~=hp.sigma_init*(sigma_decay**generation)
        sigma_init=0.1, # mutation noise
        sigma=0.1,
#         eval_time=200, # number of timesteps
        eval_time=2000, # number of timesteps
#         pop_size=64, # population size
        pop_size=4, # population size
        max_parallel=8, # max num sims to run simultaneously
        num_gens=1000, # number of generations
#         num_gens=4,
    )
    
    # calculate number of params
    # make a list of nodes in each layer and calculate the weights between the layers
    # input node count: proprioceptive sensors + touch sensors + vestigial sensor + bias
    # hidden node count: hidden nodes + bias
    # output node count: joints/motors
    nodes = np.array([3 * hp['num_legs'] + 2] + [hp['hidden_layer_size'] + 1] * hp['num_hidden_layers']
                     + [hp['num_legs'] * 2])
    num_robot_params = (nodes[:-1] * nodes[1:]).sum()
    hp['num_params'] = num_robot_params + 1  # 1 param for scaffolding temp    
    return hp

  
def train(filename=None, play_paused=False):
    '''
for rise in rises:
    make an unscaffolded population (a population with stair min_temp=1)
    train unscaffolded population for 1000 generations or until done (evaluator only simulates fitness of non-done individuals)
    make a scaffolded population (a population with temps schedule)
    train scaffolded population for 1000 generations or until done.
        for each individual, if not done, make next genome, evaluate fitness using curr scaffolding temp (and temp param). if fitness passes threshold: if temp passes threshold, done, else, increase scaffolding temp.
    '''
    # rises and variations can be used with for loops to try a 
    # variety of rises and scaffolding combinations
    L = 0.1
    rises = [0, 0.25 * L / 2.5, 0.5 * L / 2.5, 0.75 * L / 2.5, L / 2.5],
    variations = [ # temp_schedule and use_temp_param
        ([1], False), # no scaffolding
        ([-1, -0.25, 0, 0.25, 1], False), # fixed scaffolding
        ([-1, -0.25, 0, 0.25, 1], True), # evolved scaffolding
    ]

    if filename is not None:
        model = load_model(filename)
        hp = model['hp']
        pop = model['pop']
    else:
        hp = make_hyperparameters()
        if hp['mutation'] == 'noise':
            mutator = DecayingMutator(sigma_init=hp['sigma_init'])
        elif hp['mutation'] == 'evorobo':
            mutator = EvoRoboMutator()

        pop = ScaffoldingPopulation(mutator=mutator, **hp)
        pop.reset()

    for key in ['stair_max_rise', 'temp_schedule', 'use_temp_param', 'num_params', 'num_legs']:
        print(key, hp[key])

    train_population(hp, pop)
                
def train_population(hp, pop):
    if pop.gen < hp['num_gens']:
        while pop.gen < hp['num_gens']:
            pop.step()
            if hp['checkpoint_step'] is not None and pop.gen % hp['checkpoint_step'] == 0:
                model = {'hp': hp, 'pop': pop}
                save_model('population.pkl', model)
                save_model('experiments/' + hp['exp_id'] + '_population.pkl', model)

        pop.play()
        model = {'hp': hp, 'pop': pop}
        save_model('population.pkl', model)
        save_model('experiments/' + hp['exp_id'] + '_population.pkl', model)
        
    print('Done training population. gen is', pop.gen)
            
    
def play(filename=None, play_paused=False):
    if filename is None:
        filename = 'population.pkl'
        
    model = load_model(filename)
    hp = model['hp']
    pop = model['pop']
        
    # print history
#     pprint.pprint(pop.history)
    # print hyperparams
    for k, v in hp.items():
        print(k, ':', v)
        
    print('best_idx:', pop.best_idx, 'best_fit:', pop.best_fit,
          'best_age:', pop.best_age, 'best temps_idx:', pop.temps_idx[pop.best_idx],
          'best scaffolding temp:', pop.temp_schedule[pop.temps_idx[pop.best_idx]],
          'best genome temp:', pop.genomes[pop.best_idx, 0])
    print('sorted fits:', np.sort(pop.fits)[::-1])
    
    print('sorted by fitness within schedule temp:')
    df = (pd.DataFrame({'temp_schedule': pop.temp_schedule[pop.temps_idx], 
                        'fits': pop.fits, 
                        'ages': pop.ages,
                        'pop_idx': pop.pop_idx})
          .sort_values(['temp_schedule', 'fits'], kind='mergesort', ascending=False).reset_index(drop=True))
    print(df)
    
#     pop.eval_time = 4000 # for laughs, watch a robot behave beyond when it was evolved for.
    pop.play(idx=df.loc[0, 'pop_idx']) # play the fittest individual from among those with the highest schedule temp
    pop.play() # play fittest individual
#     pop.play(13) # play arbitrary individual by id
    
    
def save_model(filename, model):
    print('Saving model to', filename)
    with open(filename, 'wb') as fh:
        pickle.dump(model, fh)
        

def load_model(filename):
    print('Loading model from', filename)
    with open(filename, 'rb') as fh:
        model = pickle.load(fh)
        
    return model

def analyze(filename, **kwargs):
    if filename is None:
        filename = 'population.pkl'
        
    model = load_model(filename)
    hp = model['hp']
    pop = model['pop']
    
    exp_id = hp['exp_id']
    if exp_id in ['exp_20190505_024155', 'exp_20190505_144957']:
        exp_kind = 'ns' # not scaffolded
    elif exp_id in ['exp_20190505_024129']:
        exp_kind = 'es' # evolved scaffolding
    elif exp_id in ['']:
        exp_kind = 'ss' # strict scaffolding
                  
    fits = np.zeros((hp['num_gens']+1, pop.pop_size))
    gens = np.arange(hp['num_gens']+1)
    seen_gens = set()
    for story in pop.history:
        gen = story['gen']
        if gen > 0 and gen not in seen_gens:
            seen_gens.add(gen)
            # propagate fitnesses to next generation
            fits[gen, :] = fits[gen - 1, :]
            
        if 'better_idx' in story:
            idx = story['better_idx']
            # update fitnesses
#             print(story['better_fits'])
            fits[gen, idx] = story['better_fits']
#         gens.setdefault(gen, {})

    mean_init_fit = fits[0].mean()
    mean_fit = fits[-1].mean()
    
    # sorted by schedule temp and then fitness
    temp_fit_df = (pd.DataFrame({'temp_schedule': pop.temp_schedule[pop.temps_idx], 
                        'fits': pop.fits, 
                        'ages': pop.ages,
                        'pop_idx': pop.pop_idx})
          .sort_values(['temp_schedule', 'fits'], kind='mergesort', ascending=False).reset_index(drop=True))
    print('top ten', temp_fit_df[:10])
    
    
    # plot fitness history of all genomes
    plot_fitness_history(gens, fits, title='Fitness Evolution of Genomes')
    
    # plot fitness distribution of population
#     sns.kdeplot(fits[-1], shade=True, cut=0)
#     sns.rugplot(fits[-1])
    sns.distplot(fits[-1], rug=True)
    plt.title('Population Fitness Distribution')
    plt.xlabel('Fitness')
    plt.ylabel('Genome Count')
    plt.axvline(0.7, color='r', linestyle='dashed', linewidth=1, label='fitness threshold')
    plt.show()

    # plot fitness history of top-ten genomes (temp + fit)
    top_10_idx = temp_fit_df[:10]['pop_idx']
    print('top_10_idx', top_10_idx)
    plot_fitness_history(gens, fits[:, top_10_idx], 
                         title='Fitness Evolution of Top Ten Genomes', 
                         mean_init_fit=mean_init_fit,
                         mean_fit=mean_fit,
                        )
    
    if exp_kind in ['es', 'ss']:
        # genomes with the max scaffolding temp
        max_temp = pop.temp_schedule[pop.temps_idx.max()]
        max_temp_idx = pop.pop_idx[pop.temps_idx == pop.temps_idx.max()]
        # plot fitness history of max temp genomes
        plot_fitness_history(gens, fits[:, max_temp_idx], 
                             title=f'Fitness Evolution of Best ({max_temp:.2}) Temperature Genomes', 
                             mean_init_fit=mean_init_fit,
                             mean_fit=mean_fit,
                            )
    
        # plot counts of temps_idx, to see how the population as a whole progressed
        temps_counts = np.bincount(pop.temps_idx, minlength=len(pop.temp_schedule))
        temps_props = temps_counts / temps_counts.sum() # proportion of population
        print('temps_counts', temps_counts)
        plt.bar(np.arange(len(temps_counts)), temps_props, tick_label=pop.temp_schedule)
        plt.title('Population Distribution by Scaffolding Temperature')
        plt.ylabel('Population Proportion')
        plt.title('Scaffolding Temperature')
        plt.ylim(0, 1.)
        plt.show()

def plot_fitness_history(gens, fits, title='Evolution of Genome Fitness', 
                         mean_init_fit=None, mean_fit=None):
    for i in range(fits.shape[1]):
        plt.plot(gens, fits[:, i])
        
    plt.axhline(0.7, color='r', linestyle='dashed', linewidth=1, label='fitness threshold')
    if mean_init_fit is not None:
        plt.axhline(mean_init_fit, color='k', linestyle='dashed', linewidth=1, label='mean initial fitness')
    if mean_fit is not None:
        plt.axhline(mean_fit, color='b', linestyle='dashed', linewidth=1, label='mean fitness')
        
    plt.title(title)
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.legend()
    plt.show()
        
#         print('gen', story['gen'], sorted(story.keys()))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train and evaluate')
    parser.add_argument('action', choices=['train', 'play', 'analyze'], default='train')
    parser.add_argument('--restore', metavar='FILENAME', help='load and use a saved model')
    parser.add_argument('--play-paused', default=False, action='store_true')
    args = parser.parse_args()
    
    print('restore filename:', args.restore)
#     if args.restore is not None:
#         hp, params, evaluator = load_model(args.restore)
        
    if args.action == 'train':
        train(args.restore, play_paused=args.play_paused)
    elif args.action == 'play':
        play(args.restore, play_paused=args.play_paused)
    elif args.action == 'analyze':
        analyze(args.restore, play_paused=args.play_paused)
        
