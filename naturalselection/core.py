import numpy as np
import sys
import os
import time
from functools import partial, reduce
from itertools import permutations, chain

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

    def __init__(self, **genomes):
        self.__dict__.update(genomes)

    def create_organisms(self, amount = 1):
        ''' Create organisms of this genus. '''
        organisms = np.array([Organism(genus = self, 
            **{key : np.random.choice(self.__dict__[key])
            for key in self.__dict__.keys()}) for _ in range(amount)])
        return organisms

    def alter_genomes(self, **genomes):
        ''' Add or change genomes to the genus. '''
        self.__dict__.update(genomes)
        return self

    def remove_genomes(self, *keys):
        ''' Remove genomes from the genus. '''
        for key in keys:
            self.__dict__.pop(key, None)
        return self

class Organism():
    ''' Organism of a particular genus. '''

    def __init__(self, genus, **genome):

        # Check that the input parameters match with the genus type,
        # and if any parameters are missing then add random values
        genome = {key : val for (key, val) in genome.items() if key in
            genus.__dict__.keys() and val in genus.__dict__[key]}
        for key in genus.__dict__.keys() - genome.keys():
            genome[key] = np.random.choice(genus.__dict__[key])

        self.__dict__.update(genome)
        self.genus = genus

    def get_genome(self):
        attrs = self.__dict__.items()
        return {key : val for (key, val) in attrs if key != 'genus'}

    def breed(self, other):
        ''' Breed organism with another organism, returning a new
            organism of the same genus. '''

        if self.genus != other.genus:
            raise Exception("Only organisms of the same genus can breed.")

        # Child will inherit genes from its parents randomly
        parents_genomes = {
            key : (self.get_genome()[key], other.get_genome()[key])
                for key in self.get_genome().keys()
            }
        child_genome = {
            key : pair[np.random.choice([0, 1])]
                for (key, pair) in parents_genomes.items()
        }

        return Organism(self.genus, **child_genome)

    def mutate(self):
        ''' Return mutated version of the organism, where the mutated version
            will on average have one gene different from the original. '''
        keys = np.asarray(list(self.get_genome().keys()))
        mut_idx = np.less(np.random.random(keys.size), np.divide(1, keys.size))
        mut_vals = {key : np.random.choice(self.genus.__dict__[key])
                          for key in keys[mut_idx]}
        self.__dict__.update(mut_vals)
        return self

class Population():
    ''' Population of organisms, all of the same genus. '''

    def __init__(self, genus, size, fitness_fn, initial_genome = None):

        self.genus = genus
        self.size = size
        
        # Fitness function cannot be a lambda expression
        self.fitness_fn = fitness_fn

        if initial_genome:
            self.population = np.array(
                [Organism(genus, **initial_genome) for _ in range(size)])
        else:
            self.population = genus.create_organisms(size)

    def get_fit_organisms(self, amount = 1, multiprocessing = True,
        workers = cpu_count(), progress_bar = True, history = None):
        ''' Sample a fixed amount of organisms from the population,
            where the fitter an organism is, the more it's likely
            to be chosen. 
    
        INPUT
            (int) amount: number of fit organisms to output
            (bool) multiprocessing: whether fitnesses should be
                   computed in parallel
            (int) how many workers to use if multiprocessing is True
            (bool) progress_bar: show progress bar
            (History) history: previous genome and fitness history

        OUTPUT
            (ndarray) fit subset of population
        '''

        pop = self.population
        fitnesses = np.zeros(pop.size)

        # Get the unique genomes from the current population
        genomes = np.array([org.get_genome() for org in pop])
        unique_genomes = np.array([dict(dna) for dna
            in set(frozenset(genome.items()) for genome in genomes)])

        # If history is loaded then get the genomes from the current
        # population that are unique across all generations
        if history:
            all_prev_genomes = np.array([past_genome
                for past_genomes in history.genome_history
                for past_genome in past_genomes
                ])
            all_prev_fitnesses = np.array([past_fitness
                for past_fitnesses in history.fitness_history
                for past_fitness in past_fitnesses
                ])
            indices = np.array([(past_idx, idx)
                for (idx, org) in enumerate(pop)
                for (past_idx, past_genome) in enumerate(all_prev_genomes)
                if org.get_genome() == past_genome
                ])
            past_indices = np.array([idx for (_, idx) in indices])

            # Load previous fitnesses of genomes that are occuring now
            for (past_idx, idx) in indices:
                fitnesses[idx] = all_prev_fitnesses[past_idx]

            # Remove genomes that have occured previously
            unique_genomes = np.array([genome for genome in unique_genomes
                if genome not in all_prev_genomes])

        # Pull out the organisms with the unique genomes
        unique_indices = np.array([np.min(np.array([idx
            for (idx, org) in enumerate(pop) if org.get_genome() == genome]))
            for genome in unique_genomes])

        # If there are any organisms whose fitness we didn't already
        # know then compute them
        if unique_indices.size:
            unique_orgs = pop[unique_indices]

            fn = self.fitness_fn
            progress_text = "Computing fitness for the current generation"

            # Compute fitness values without computing the same one twice
            with suppress_stdout():
                if multiprocessing:
                    with Pool(workers) as pool:
                        if progress_bar:
                            fit_iter = tqdm(zip(unique_indices, 
                                pool.imap(fn, unique_orgs)),
                                total = unique_orgs.size)
                            fit_iter.set_description(progress_text)
                        else:
                            fit_iter = zip(unique_indices,
                                pool.map(fn, unique_orgs))
                        for (i, new_fitness) in fit_iter:
                            fitnesses[i] = new_fitness
                else:
                    if progress_bar:
                        fit_iter = tqdm(zip(unique_indices,
                            map(fn, unique_orgs)),
                            total = unique_orgs.size)
                        fit_iter.set_description(progress_text)
                    else:
                        fit_iter = zip(unique_indices, map(fn, unique_orgs))
                    for (i, new_fitness) in fit_iter:
                        fitnesses[i] = new_fitness

        # Copy out the fitness values to the other organisms with same genome
        for (i, org) in enumerate(pop):
            if i not in unique_indices and i not in past_indices:
                prev_unique_idx = np.min(np.array([idx
                    for idx in unique_indices
                    if org.get_genome() == pop[idx].get_genome()]))
                fitnesses[i] = fitnesses[prev_unique_idx]

        # Convert fitness values into probabilities
        probs = np.divide(fitnesses, sum(fitnesses))

        # Sort the probabilities in descending order and sort the
        # population in the same way
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
        
        # Store data for this generation
        history.add_entry(genomes = genomes, fitnesses = fitnesses)

        # Return the organisms indexed at the indices found above
        return self.population[indices.astype(int)], history

    def evolve(self, generations = 1, breeding_pool = 0.20,
        mutation_pool = 0.20, multiprocessing = True, workers = cpu_count(),
        progress_bars = 2, verbose = 0):
        ''' Evolve the population.

        INPUT
            (int) generations: number of generations to evolve
            (float) breeding_pool: percentage of population to breed 
            (float) mutatation_pool: percentage of population to mutate
                    each generation
            (bool) multiprocessing: whether fitnesses should be computed
                   in parallel
            (int) workers: how many workers to use if multiprocessing is True
            (int) progress_bars: number of progress bars to show, where 1
                  only shows the main evolution progress, and 2 shows both
                  the evolution and the fitness computation per generation
            (int) verbose: verbosity mode
        '''
    
        history = History()

        if progress_bars:
            gen_iter = trange(generations)
            gen_iter.set_description("Evolving population")
        else:
            gen_iter = range(generations)

        for generation in gen_iter:

            # Select the portion of the population that will breed
            breeders = max(2, np.ceil(self.size * breeding_pool).astype(int))
            fit_organisms, history = self.get_fit_organisms(
                amount = breeders,
                multiprocessing = multiprocessing,
                workers = workers,
                progress_bar = (progress_bars == 2),
                history = history
                )

            if verbose:
                print("")
                print("Breeding pool:")
                print(np.array([org.get_genome() for org in fit_organisms]))
                print("Breeding...")

            # Breed until we reach the same size
            parents = np.random.choice(fit_organisms, (self.size, 2))
            children = np.array([parents[i, 0].breed(parents[i, 1])
                for i in range(self.size)])

            # Find the mutation pool
            mutators = np.less(np.random.random(self.size), mutation_pool)

            if verbose:
                print("Mutation pool:")
                print(np.array([c.get_genome() for c in children[mutators]]))
                print("Mutating...")

            # Mutate the children
            for mutator in children[mutators]:
                mutator.mutate()

            # The children constitutes our new generation
            self.population = children
            
            if verbose:
                print(f"Mean fitness for previous generation: " \
                      f"{np.mean(history.fitness_history[-1])}")
                print(f"Std fitness for previous generation: " \
                      f"{np.std(history.fitness_history[-1])}")
                print(f"Fittest so far: {history.fittest}")
            
        return history

class History():
    ''' History of a population's evolution. '''

    def __init__(self):
        self.genome_history = []
        self.fitness_history = []
        self.fittest = {'genome' : None, 'fitness' : 0}
    
    def add_entry(self, genomes, fitnesses):
        ''' Add genomes and fitnesses to the history. 

        INPUT
            (ndarray) genomes: array of genomes
            (ndarray) fitnesses: array of fitnesses
        '''

        self.genome_history.append(genomes)
        self.fitness_history.append(fitnesses)

        if max(fitnesses) > self.fittest['fitness']:
            self.fittest['genome'] = genomes[np.argmax(fitnesses)]
            self.fittest['fitness'] = max(fitnesses)

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
        means = np.array([np.mean(fit) for fit in self.fitness_history])
        stds = np.array([np.std(fit) for fit in self.fitness_history])

        plt.style.use("ggplot")
        plt.figure()
        plt.errorbar(range(1, gens + 1), means, stds, fmt = 'ok')
        plt.xlim(0, gens + 1)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if save_to:
            plt.savefig(save_to)

        if show_plot:
            plt.show()

        return self


def __main__():

    # Simple example with numbers
    Number = ns.Genus(x = range(1, 10000), y = range(1, 10000))
    numbers = ns.Population(genus = Number, size = 10, fitness_fn = division)
    history = numbers.evolve(generations = 100, progress_bars = 1)

    print(f"Fittest genome across all generations:")
    print(history.fittest)

    history.plot()


if __name__ == '__main__':

    def division(number):
        return number.x / number.y
    main()
