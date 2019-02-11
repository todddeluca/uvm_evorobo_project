from individual import Individual

class Population:
    
    def __init__(self, popSize=1):
        self.p = {}
        for i in range(popSize):
            self.p[i] = Individual(id_=i)
        
    def __repr__(self):
        return ', '.join([str(self.p[i]) for i in self.p])
            
    def evaluate(self, play_blind=True):
        for i in self.p:
            self.p[i].start_evaluation(play_blind=play_blind)
            
        for i in self.p:
            self.p[i].compute_fitness()
            
    def mutate(self):
        for i in self.p:
            self.p[i].mutate()
            
    def replace_with(self, other):
        for i in self.p:
            if self.p[i].fitness < other.p[i].fitness:
                self.p[i] = other.p[i]
            
            
            
            