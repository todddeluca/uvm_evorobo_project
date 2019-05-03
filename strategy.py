
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

    def result(self):
        return (self.best_params, self.best_fit, self.best_idx)

    
class AFPO:
    '''
    Age Fitness Pareto Optimization
    https://dl.acm.org/citation.cfm?id=1830584
    
    Maintain population diversity using a pareto front defined by age (in generations)
    and fitness.
    '''
    def __init__(self, num_params, pop_size, 
                 sigma_init=0.1, sigma_decay=0.999, sigma_limit=0.01, 
                 seed=None, sols=None, lineage=None, num_novel=1):
        '''
        num_novel: number of new lineages per generation
        '''
        self.seed = seed
        self.num_params = num_params
        self.pop_size = pop_size
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.gens = 0 # current generation
        self.sigma = self.sigma_init
        self.num_novel = num_novel
        
        if lineage is None:
            self.lineage = np.arange(self.pop_size)
        else:
            self.lineage = lineage
            
        if sols is None:
            self.sols = np.random.randn(self.pop_size, self.num_params) * self.sigma
        else:
            self.sols = sols
            
        self.ages = np.zeros(self.pop_size) # ages
        self.best_params = None
        self.best_age = None
        self.best_fit = None
        
        self.first_iteration = True # There's a first time for everything

    def ask(self):
        if self.first_iteration:
            self.first_iteration = False
            return self.sols

        pop_idx = np.arange(self.pop_size) # numerical index of population        
        # age the population
        self.ages += 1

        # discard indivs not on the pareto front
        self.discard_idx = pop_idx[np.isin(pop_idx, self.front_idx, invert=True)]
        # discard at least num_novel individuals per generation
        # This rarely happens for a sufficiently large population and small num_novel
        if len(self.discard_idx) < self.num_novel:
            # number of extra individuals to discard
            num_more = self.num_novel - len(self.discard_idx)
            # choose indivs at random to discard (excepting the fittest and already discarded)
            more_to_discard_idx = np.random.choice(
                pop_idx[(pop_idx != self.best_idx) & np.isin(pop_idx, self.discard_idx, invert=True)], 
                size=num_more, 
                replace=False)
            self.discard_idx = np.concatenate([self.discard_idx, more_to_discard_idx])
        
        # decay sigma
        if self.sigma > self.sigma_limit:
            self.sigma = max(self.sigma_limit, self.sigma * self.sigma_decay)

        # create num_novel brand new individuals
        self.sols[self.discard_idx[:self.num_novel]] = np.random.randn(self.num_novel, self.num_params) * self.sigma
        self.ages[self.discard_idx[:self.num_novel]] = 0
        max_lineage = np.amax(self.lineage)
        self.lineage[self.discard_idx[:self.num_novel]] = np.arange(
            max_lineage + 1, max_lineage + self.num_novel + 1) # yay new species
        
        # create descendents of the front
        for idx in self.discard_idx[self.num_novel:]:
            parent_idx = np.random.choice(self.front_idx)
            self.sols[idx] = self.sols[parent_idx] + np.random.randn(self.num_params) * self.sigma
            self.ages[idx] = self.ages[parent_idx]
            self.lineage[idx] = self.lineage[parent_idx] # descendant

        return self.sols
    
    def tell(self, fits): 
        
        self.fits = fits
        # find pareto front
        self.front_idx = self.pareto_front(self.ages, self.fits)
        # track the fittest individual
        self.best_idx = self.front_idx[self.fits[self.front_idx].argmax()]
        self.best_params = self.sols[self.best_idx]
        self.best_age = self.ages[self.best_idx]
        self.best_fit = self.fits[self.best_idx]
            
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

