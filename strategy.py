
import numpy as np


class ParallelHillClimber:
    '''
    Evaluate a population in parallel
    '''
    def __init__(self, num_params, pop_size, 
                 sigma_init=0.1, sigma_decay=0.999, sigma_limit=0.01, 
                 seed=None, mutation='noise'):
        '''
        mutation: 'noise' adds gaussian noise to every param. 
          'evorobo' adds noise to one param at random (sigma=param) and clips.
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

    def ask(self):
        # mutation from class chooses a single weight at random,
        # adds gaussian noise using sigma=weight, and clips to [-1, 1]
        # new_weight = random.gauss(self.genome[i, j], math.fabs(self.genome[i, j]))
        # self.genome[i, j] = np.clip(new_weight, -1, 1)
        
        noise = np.random.randn(self.pop_size, self.num_params) * self.sigma
        if self.sols is None:
            # initial iteration: initialize solutions
            self.sols = noise
            self.next_sols = self.sols
        else:
            if self.mutation == 'evorobo':
                # select a random element from each solution
                idx = (np.arange(self.pop_size), np.random.choice(self.num_params, size=self.pop_size))
                # sigma is the absolute value of each element (bounded by sigma_limit)
                sigmas = np.abs(self.sols[idx])
                sigmas[sigmas < self.sigma_limit] = self.sigma_limit
                # add noise to selected elements and clip to [-1, 1]
                self.next_sols = self.sols
                self.next_sols[idx] = np.clip(self.sols[idx] + np.random.randn(self.pop_size) * sigmas, -1, 1)
                print((self.sols - self.next_sols))
                print((self.sols - self.next_sols)[np.nonzero(self.sols - self.next_sols)])
            else:
                self.next_sols = self.sols + noise
            
        # decay sigma
        if self.sigma > self.sigma_limit:
            self.sigma = max(self.sigma_limit, self.sigma * self.sigma_decay)

        return self.next_sols
    
    def tell(self, fits):
        '''
        compare fitnesses to previous fitnesses. replace if better.
        '''
        if self.fits is None:
            # initial iteration, initialize fitness
            self.fits = fits
        else:
            better_idx = (fits > self.fits)
            print(f'{better_idx.sum()} solutions improved')
            self.sols[better_idx] = self.next_sols[better_idx]
            self.fits[better_idx] = fits[better_idx]
            
        best_idx = self.fits.argmax()
        self.best_params = self.sols[best_idx]
        self.best_fit = self.fits[best_idx]

    def result(self):
        return (self.best_params, self.best_fit)

    
class AFPO:
    '''
    Age Fitness Pareto Optimization
    https://dl.acm.org/citation.cfm?id=1830584
    
    Maintain population diversity using a pareto front defined by age (in generations)
    and fitness.
    '''
    def __init__(self, num_params, pop_size, 
                 sigma_init=0.1, sigma_decay=0.999, sigma_limit=0.01, 
                 seed=None, sols=None):
        self.seed = seed
        self.num_params = num_params
        self.pop_size = pop_size
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.gens = 0 # current generation
        self.sigma = self.sigma_init
        
        if sols is None:
            self.sols = np.random.randn(self.pop_size, self.num_params) * self.sigma
        else:
            self.sols = sols
            
        self.ages = np.zeros(self.pop_size) # ages
        self.best_params = None
        self.best_age = None
        self.best_fit = None

    def ask(self):
        return self.sols
    
    def tell(self, fits):        
        pop_idx = np.arange(self.pop_size) # numerical index of population
        
        # age the population
        self.ages += 1
        
        # find pareto front
        front_idx = self.pareto_front(self.ages, fits)
        print('front size:', len(front_idx), 'front ages:', self.ages[front_idx])
        
        # track the fittest individual
        best_idx = front_idx[fits[front_idx].argmax()]
        self.best_params = self.sols[best_idx]
        self.best_age = self.ages[best_idx]
        self.best_fit = fits[best_idx]
        
        # discard indivs not on the pareto front
        discard_idx = pop_idx[np.isin(pop_idx, front_idx, invert=True)]
        # discard at least one individual per generation
        if len(discard_idx) == 0:
            # choose an indiv at random to discard (excepting the fittest)
            discard_idx = np.random.choice(pop_idx[pop_idx != best_idx], size=1)

        if self.sigma > self.sigma_limit:
            self.sigma = max(self.sigma_limit, self.sigma * self.sigma_decay)

        # create new indivs
        # create one new individual
        self.sols[discard_idx[0]] = np.random.randn(self.num_params) * self.sigma
        self.ages[discard_idx[0]] = 0
        # create other individuals derived from the front
        for idx in discard_idx[1:]:
            parent_idx = np.random.choice(front_idx)
            self.sols[idx] = self.sols[parent_idx] + np.random.randn(self.num_params) * self.sigma
            self.ages[idx] = self.ages[parent_idx]
            
    def result(self):
        return (self.best_params, self.best_fit, self.best_age)

    def pareto_front(self, obj1, obj2):
        '''
        An individual is on the front if its objective 2 value (e.g. fitness)
        is greater than the objective 2 value of any individual that has a
        lower objective 1 value. 
        This makes sense for age-fitness pareto optimization, where the front
        is min(age) and max(fitness).

        obj1: objective 1, e.g. age
        obj2: objective 2, e.g. fitnesses
        returns: an int index indicating the individuals on the pareto front.
        '''
        num_indivs = len(obj1)
        front_idx_bool = np.zeros(num_indivs, dtype=bool)
        max_obj2 = None # track the best fitness of younger individuals
        for o1 in np.unique(obj1): # sorted unique ages
            # index of fittest individual of an age
            best_idx = np.arange(num_indivs)[obj1==o1][obj2[obj1==o1].argmax()]
            # if the fittest individual of this age is better than the best
            # fitness of younger individuals, add it to the pareto front
            if max_obj2 is None or obj2[best_idx] > max_obj2:
                front_idx_bool[best_idx] = True
                max_obj2 = obj2[best_idx]            
        return np.arange(num_indivs)[front_idx_bool]

