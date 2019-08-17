import numpy as np
from functools import reduce
import sys
import os

# Plots
import matplotlib.pyplot as plt

# Progress bar
from tqdm import tqdm, trange

# Parallelising fitness
from multiprocessing import Pool

# Used to suppress console output
from contextlib import contextmanager
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


class Genus():

    def __init__(self, genomes: dict):
        self.genomes = genomes

    def create_organisms(self, amount: int = 1):
        ''' Create organisms of this genus. '''
        print("Creating organisms...", end = "\r")
        organisms = np.array([Organism(genus = self, genome =
            {key : np.random.choice(self.genomes[key])
            for key in self.genomes.keys()}) for i in range(amount)])
        return organisms

    def create_genus(self, add = None, remove = None):
        ''' Create copy of this genus with possible modifications. '''
        genomes = np.append(np.array([param for param in self.genomes
                     if param.key not in remove]), add)
        return Genus(genomes)

class Organism():

    def __init__(self, genus: Genus, genome: dict):
        if not genus.genomes.keys() == genome.keys():
            raise TypeError('Genus keys do not match input keys.')

        self.genus = genus
        self.genome = genome
        self.fitness = None

    def breed(self, other):
        ''' Breed with another organism, creating a new organism of
            the same genus. '''

        if not self.genus == other.genus:
            raise TypeError('The two organisms are not of the same genus.')

        # Child will inherit genes from its parents randomly
        child_genome = {
            key : np.random.choice([self.genome[key], other.genome[key]])
                  for key in self.genome.keys()
            }
        return Organism(self.genus, child_genome)

    def mutate(self):
        ''' Mutate the organism, changing on average one gene. '''
        keys = np.array([*self.genus.genomes])
        mut_idx = np.less(np.random.random(keys.size), np.divide(1, keys.size))
        mut_keys = keys[mut_idx]
        mut_vals = np.array([np.random.choice(self.genus.genomes[key])
                             for key in mut_keys])
        self.genome[mut_keys] = mut_vals
        self.fitness = None
        return self

class Population():

    def __init__(self, genus, size, fitness_fn):
        self.genus = genus
        self.size = size
        self.population = genus.create_organisms(size)
        self.fittest = None

        # Fitness function cannot be a lambda expression
        self.fitness_fn = fitness_fn

    def get_fit_organisms(self, amount = 1, multiprocessing = True,
        progress_bar = True):
        ''' Sample a fixed amount of organisms from the population,
            where the fitter an organism is, the more it's likely
            to be chosen. 
    
        INPUT
            (int) amount: number of fit organisms to output
            (bool) multiprocessing: whether fitnesses should be
                   computed in parallel
            (bool) progress_bar: show progress bar

        OUTPUT
            (ndarray) fit subset of population
        '''

        # Get array of organisms in population with no fitness recorded
        pop = self.population
        fn = self.fitness_fn
        with suppress_stdout():
            if multiprocessing:
                # Compute fitness values in parallel
                with Pool() as pool:
                    if progress_bar:
                        fit_iter = tqdm(zip(pop, pool.imap(fn, pop)),
                            total = pop.size)
                        fit_iter.set_description("Computing fitness")
                    else:
                        fit_iter = zip(pop, pool.map(fn, pop))
                    for (org, new_fitness) in fit_iter:
                        org.fitness = new_fitness
            else:
                if progress_bar:
                    fit_iter = tqdm(zip(pop, map(fn, pop)), total = pop.size)
                    fit_iter.set_description("Computing fitness")
                else:
                    fit_iter = zip(pop, map(fn, pop))
                for (org, new_fitness) in fit_iter:
                    org.fitness = new_fitness

        fitnesses = np.asarray([org.fitness for org in pop])
        
        # Convert fitness values into probabilities
        probs = np.divide(fitnesses, sum(fitnesses))

        # Sort the probabilities in descending order and sort the
        # population and fitnesses in the same way
        sorted_idx = np.argsort(probs)[::-1]
        probs = probs[sorted_idx]
        self.population = pop[sorted_idx]

        # Save the fittest genome with its fitness value
        self.fittest = {
            'genome'    : self.population[0].genome,
            'fitness'   : fitnesses[sorted_idx[0]]
            }
       
        # Get random numbers between 0 and 1 
        indices = np.random.rand(amount)

        if progress_bar:
            amount_range = trange(amount)
            amount_range.set_description("Choosing fittest organisms")
        else:
            amount_range = range(amount)

        for i in amount_range:
            # Find the index of the fitness value whose accumulated
            # sum exceeds the value of the i'th random number.
            fn = lambda x, y: (x[0], x[1] + y[1]) \
                              if x[1] + y[1] > indices[i] \
                              else (x[0] + y[0], x[1] + y[1])
            (idx, _) = reduce(fn, map(lambda x: (1, x), probs))
            indices[i] = idx - 1

        # Return the organisms indexed at the indices found above
        return self.population[indices.astype(int)]

    def evolve(self, generations = 1, keep = 0.30, mutate = 0.10,
        multiprocessing = True, verbose = False, show_progress = 2):
        ''' Evolve the population.

        INPUT
            (int) generations: number of generations to evolve
            (float) keep: percentage of population to keep each gen
            (float) mutate: percentage of population to mutate each gen
            (bool) multiprocessing: whether fitnesses should be
                   computed in parallel
            (bool) verbose: verbosity mode
            (int) show_progress: takes values 0-2, where 0 means no
                  progress bars, 1 means only the main one, and 2 means
                  the main one and the fitness one
        '''
    
        history = History()

        if show_progress:
            gen_iter = trange(generations)
            gen_iter.set_description("Evolving population")
        else:
            gen_iter = range(generations)
            print("Evolving population...", end = "\r")

        for generation in gen_iter:
            # Select the portion of the population that will breed
            keep_amount = max(2, np.ceil(self.size * keep).astype(int))
            fit_organisms = self.get_fit_organisms(
                amount = keep_amount,
                multiprocessing = multiprocessing,
                progress_bar = (show_progress == 2)
                )
       
            # Breed until we reach the same size
            parents = np.random.choice(fit_organisms, (self.size, 2))
            children = np.array([parents[i, 0].breed(parents[i, 1])
                for i in range(self.size)])
            mutate_indices = np.less(np.random.random(self.size), mutate)
            map(lambda child: child.mutate(), children[mutate_indices])
            self.population = children
            
            # Store fittest genome for this generation
            history.add_entry(*self.fittest.values())

        # Print another line if there are two progress bars
        if show_progress == 2:
            print("")

        return history

class History():

    def __init__(self):
        self.genomes = []
        self.fitnesses = []
        self.fittest = {'genome' : None, 'fitness' : None}
    
    def add_entry(self, genome, fitness):
        ''' Add genome and fitness to the history. '''
        self.genomes.append(genome)
        self.fitnesses.append(fitness)
        self.fittest = {
            'genome' : self.genomes[np.argmax(self.fitnesses)],
            'fitness' : max(self.fitnesses)
            }
        return self

    def plot(self, save_fig = None, title = 'Fitness by generation'):
        ''' Plot the fitness values. '''
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(self.fitnesses, label = "fitness")
        plt.title("Fitness by generation")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        if save_fig:
            plt.save_fig(save_fig)
        plt.show()
        return self


if __name__ == '__main__':

    Number = Genus({'x' : range(1, 10000), 'y' : range(1, 10000)})
    def fitness_fn(number):
        return number.genome['x'] / number.genome['y']

    numbers = Population(genus = Number, size = 1000, fitness_fn = fitness_fn)
    history = numbers.evolve(generations = 10)

    print(f"Fittest genome across all generations:")
    print(history.fittest)
    history.plot()
