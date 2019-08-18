import numpy as np
from functools import reduce
import sys
import os

# Plots
import matplotlib.pyplot as plt

# Progress bar
from tqdm import tqdm, trange

# Parallelising fitness
from multiprocessing import Pool, cpu_count

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
    ''' Storing information about all the possible gene combinations. '''

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def create_organisms(self, amount = 1):
        ''' Create organisms of this genus. '''
        print("Creating organisms...", end = "\r")
        organisms = np.array([Organism(genus = self, 
            **{key : np.random.choice(self.__dict__[key])
            for key in self.__dict__.keys()}) for i in range(amount)])
        return organisms

class Organism():
    ''' Organism of a particular genus. '''

    def __init__(self, genus, **kwargs):

        # Check that the input parameters match with the genus type,
        # and if any parameters are missing then add random values
        genome = {key:val for (key,val) in kwargs.items() if key in
            genus.__dict__.keys() and val in genus.__dict__[key]}
        for key in genus.__dict__.keys() - kwargs.keys():
            genome[key] = np.random.choice(genus.__dict__[key])

        self.__dict__.update(genome)
        self.genome = genome
        self.genus = genus

    def breed(self, other):
        ''' Breed organism with another organism, returning a new
            organism of the same genus. '''

        if self.genus != other.genus:
            raise Exception("Only organisms of the same genus can breed.")

        # Child will inherit genes from its parents randomly
        child_genome = {
            key : np.random.choice([self.genome[key], other.genome[key]])
                  for key in self.genome.keys()
            }
        return Organism(self.genus, **child_genome)

    def mutate(self):
        ''' Return mutated version of the organism, where the mutated version
            will on average have one gene different from the original. '''
        keys = np.asarray(list(self.genome.keys()))
        mut_idx = np.less(np.random.random(keys.size), np.divide(1, keys.size))
        mut_genes = {key : np.random.choice(self.genus.__dict__[key]) for
            key in keys[mut_idx]}
        return Organism(self.genus, **{**self.genome, **mut_genes})

class Population():
    ''' Population of organisms, all of the same genus. '''

    def __init__(self, genus, size, fitness_fn):
        self.genus = genus
        self.size = size
        self.population = genus.create_organisms(size)

        # Fitness function cannot be a lambda expression
        self.fitness_fn = fitness_fn

    def get_fit_organisms(self, amount = 1, multiprocessing = True,
        workers = cpu_count(), progress_bar = True):
        ''' Sample a fixed amount of organisms from the population,
            where the fitter an organism is, the more it's likely
            to be chosen. 
    
        INPUT
            (int) amount: number of fit organisms to output
            (bool) multiprocessing: whether fitnesses should be
                   computed in parallel
            (int) how many workers to use if multiprocessing is True
            (bool) progress_bar: show progress bar

        OUTPUT
            (ndarray) fit subset of population
        '''

        # Get array of organisms in population with no fitness recorded
        pop = self.population
        fn = self.fitness_fn
        fitnesses = np.zeros(pop.size)
        progress_text = "Computing fitness for the current generation"
        with suppress_stdout():
            if multiprocessing:
                # Compute fitness values in parallel
                with Pool(workers) as pool:
                    if progress_bar:
                        fit_iter = tqdm(enumerate(pool.imap(fn, pop)),
                            total = pop.size)
                        fit_iter.set_description(progress_text)
                    else:
                        fit_iter = enumerate(pool.map(fn, pop))
                    for (i, new_fitness) in fit_iter:
                        fitnesses[i] = new_fitness
            else:
                if progress_bar:
                    fit_iter = tqdm(enumerate(map(fn, pop)), total = pop.size)
                    fit_iter.set_description(progress_text)
                else:
                    fit_iter = enumerate(map(fn, pop))
                for (i, new_fitness) in fit_iter:
                    fitnesses[i] = new_fitness
        
        # Convert fitness values into probabilities
        probs = np.divide(fitnesses, sum(fitnesses))

        # Sort the probabilities in descending order and sort the
        # population and fitnesses in the same way
        sorted_idx = np.argsort(probs)[::-1]
        probs = probs[sorted_idx]
        self.population = pop[sorted_idx]

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
        
        cache = {
            'genomes' : np.array([org.genome for org in pop]),
            'fitnesses' : fitnesses
            }

        # Return the organisms indexed at the indices found above
        return self.population[indices.astype(int)], cache

    def evolve(self, generations = 1, breeding_pool = 0.20,
        mutation_pool = 0.20, multiprocessing = True,
        workers = cpu_count(), verbose = False, show_progress = 2):
        ''' Evolve the population.

        INPUT
            (int) generations: number of generations to evolve
            (float) breeding_pool: percentage of population to breed 
            (float) mutatation_pool: percentage of population to mutate
                    each generation
            (bool) multiprocessing: whether fitnesses should be
                   computed in parallel
            (int) workers: how many workers to use if multiprocessing is True
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
            breeders = max(2, np.ceil(self.size * breeding_pool).astype(int))
            fit_organisms, cache = self.get_fit_organisms(
                amount = breeders,
                multiprocessing = multiprocessing,
                workers = workers,
                progress_bar = (show_progress == 2)
                )

            # Store data for this generation
            history.add_entry(cache)
       
            # Breed until we reach the same size
            parents = np.random.choice(fit_organisms, (self.size, 2))
            children = np.array([parents[i, 0].breed(parents[i, 1])
                for i in range(self.size)])
            mutators = np.less(np.random.random(self.size), mutation_pool)
            children[mutators] = np.array(
                [child.mutate() for child in children[mutators]])
            self.population = children
            
        # Print another line if there are two progress bars
        if show_progress == 2:
            print("")

        return history

class History():
    ''' History of a population's evolution. '''

    def __init__(self):
        self.genome_history = []
        self.fitness_history = []
        self.fittest = {'genome' : None, 'fitness' : 0}
    
    def add_entry(self, cache: dict):
        ''' Add genomes and fitnesses to the history. 

        INPUT
            (dict) cache: dictionary with genomes and fitnesses of
                   the form {'genomes' : [], 'fitnesses' : []}
        '''

        genomes = cache['genomes']
        fitnesses = cache['fitnesses']

        self.genome_history.append(genomes)
        self.fitness_history.append(fitnesses)

        if max(fitnesses) > self.fittest['fitness']:
            self.fittest = {
                'genome' : genomes[np.argmax(fitnesses)],
                'fitness' : max(fitnesses)
                }

        return self

    def plot(self, title = 'Average fitness by generation',
        xlabel = 'Generations', ylabel = 'Average fitness',
        save_to = None, show_plot = True):
        ''' Plot the fitness values.

        INPUT
            (string) title: title on the plot
            (string) xlabel: label on the x-axis
            (string) ylabel: label on the y-axis
            (string) save_to: file name to save the plot to
            (bool) show_plot: whether to show plot as a pop-up
        '''

        gens = len(self.fitness_history)
        mins = np.array([np.min(fit) for fit in self.fitness_history])
        maxs = np.array([np.max(fit) for fit in self.fitness_history])
        means = np.array([np.mean(fit) for fit in self.fitness_history])
        stds = np.array([np.std(fit) for fit in self.fitness_history])

        plt.style.use("ggplot")
        plt.figure()
        plt.errorbar(range(gens), means, stds, fmt = 'ok')
        plt.xlim(-1, gens + 1)
        plt.title("Average fitness by generation")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if save_to:
            plt.savefig(save_to)
        if show_plot:
            plt.show()
        return self


if __name__ == '__main__':

    Number = Genus(x = range(1, 10000), y = range(1, 10000))
    def fitness_fn(number):
        return number.x / number.y

    numbers = Population(genus = Number, size = 50, fitness_fn = fitness_fn)
    history = numbers.evolve(generations = 100, show_progress = 1)

    print(f"Fittest genome across all generations:")
    print(history.fittest)
    history.plot()
