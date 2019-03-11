
import copy
import numpy as np

from individual import Individual


class Population:
    
    def __init__(self, pop_size=1):
        self.pop_size = pop_size
        self.p = []
        
    def initialize(self):
        '''
        Construct the individuals of the population
        '''
        self.p = []
        for i in range(self.pop_size):
            self.p.append(Individual(id_=i))
        
    def __repr__(self):
        return ', '.join([str(indiv) for indiv in self.p])
            
    def evaluate(self, envs, play_blind=True, play_paused=False):
        '''
        Evaluate each individual in each environment. Compute the fitness.
        '''
        for indiv in self.p:
            indiv.fitness = 0
            
        for env in envs.envs:
            for indiv in self.p:
                indiv.start_evaluation(env, play_blind=play_blind, play_paused=play_paused)

            for indiv in self.p:
                indiv.compute_fitness()
                
        for indiv in self.p:
            indiv.fitness /= len(envs.envs) # average fitness per environment
            
    def mutate(self):
        for indiv in self.p:
            indiv.mutate()
            
    def fill_from(self, other):
        self.copy_best_from(other)
        self.collect_children_from(other)
        
    def copy_best_from(self, other):
        '''
        copy best individual from other into the population
        '''
        best_i = np.argmax([indiv.fitness for indiv in other.p])
        self.p.append(copy.deepcopy(other.p[best_i]))
            
    def winner_of_tournament_selection(self):
        '''
        Return the fittest of two randomly selected distinct individuals from the population
        '''
        i, j = np.random.choice(self.pop_size, 2, replace=False)
        idx = i if self.p[i].fitness >= self.p[j].fitness else j
        return self.p[idx]
        
    
    def collect_children_from(self, other):
        '''
        fill remaining population slots from randomly selected and mutated individuals from other
        '''
        for i in range(len(self.p), self.pop_size):
            winner = other.winner_of_tournament_selection()
            child = copy.deepcopy(winner)
            child.mutate()
            self.p.append(child)
        
    def replace_with(self, other):
        for i in self.p:
            if self.p[i].fitness < other.p[i].fitness:
                self.p[i] = other.p[i]
            
            
            
            