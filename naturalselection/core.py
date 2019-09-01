import numpy as np
import sys
import os
from functools import reduce

# Suppressing warnings
import warnings

# Plots
import matplotlib.pyplot as plt

# Progress bar
from tqdm import tqdm, trange

# Parallelising fitness
from multiprocessing import Pool, cpu_count

# Logging
import logging

class Genus():
    ''' Storing information about all the possible gene combinations.

    INPUT
        (kwargs) genomes
    '''

    def __init__(self, **genomes):
        self.__dict__.update(
            {key : np.asarray(val) for (key, val) in genomes.items()}
            )

    def create_random_organism(self, _):
        rnd_genes = {key : val[np.random.choice(range(val.shape[0]))]
            for (key, val) in self.__dict__.items()}
        return Organism(genus = self, **rnd_genes)

    def create_organisms(self, amount = 1):
        ''' Create organisms of this genus.
        
        INPUT
            (int) amount = 1
        '''
        genus = self.__dict__.items()
        organisms = np.array([Organism(genus = self, 
            **{key : val[np.random.choice(range(val.shape[0]))]
            for (key, val) in genus}) for _ in range(amount)])
        return organisms

    def alter_genomes(self, **genomes):
        ''' Add or change genomes to the genus.
        
        INPUT
            (kwargs) genomes
        '''
        self.__dict__.update(genomes)
        return self

    def remove_genomes(self, *keys):
        ''' Remove genomes from the genus. '''
        for key in keys:
            self.__dict__.pop(key, None)
        return self

class Organism():
    ''' Organism of a particular genus. 

    INPUT
        (Genus) genus
        (kwargs) genome: genome information
    '''

    def __init__(self, genus, **genome):

        # Check that the input parameters match with the genus type,
        # and if any parameters are missing then add random values
        genome = {key : val for (key, val) in genome.items() if key in
            genus.__dict__.keys() and val in genus.__dict__[key]}
        for key in genus.__dict__.keys() - genome.keys():
            val_idx = np.random.choice(range(genus.__dict__[key].shape[0]))
            genome[key] = genus.__dict__[key][val_idx]

        self.__dict__.update(genome)
        self.genus = genus
        self.fitness = 0

    def get_genome(self):
        return {key : val for (key, val) in self.__dict__.items()
            if key not in {'genus', 'fitness'}}

    def breed(self, other):
        ''' Breed organism with another organism, returning a new
            organism of the same genus.

        INPUT
            (Organism) other
        '''

        if self.genus != other.genus:
            raise Exception("Only organisms of the same genus can breed.")

        self_genome = list(self.get_genome().items())
        other_genome = list(other.get_genome().items())

        rnd = np.random.choice(len(self_genome))
        child_genome = dict(self_genome[:rnd] + other_genome[rnd:])
        child = Organism(self.genus, **child_genome)

        return child

    def mutate(self, mutation_factor = 'default'):
        ''' Return mutated version of the organism.
        
        INPUT
            (float or string) mutation_factor = 'default': given that an
                              organism is being mutated, the probability that
                              a given gene is changed. Defaults to 1/k, where
                              k is the size of the population
        '''
        keys = np.asarray(list(self.get_genome().keys()))
        if mutation_factor == 'default':
            mutation_factor = np.divide(1, keys.size)
        mut_idx = np.less(np.random.random(keys.size), mutation_factor)
        mut_vals = {key : self.genus.__dict__[key]\
            [np.random.choice(range(self.genus.__dict__[key].shape[0]))]
            for key in keys[mut_idx]}
        self.__dict__.update(mut_vals)
        return self

class Population():
    ''' Population of organisms, all of the same genus.

    INPUT
        (Genus) genus
        (int) size: size of the population
        (function) fitness_fn: fitness function which must be pickleable
        (dict) initial_genome = None: start with a population similar to
               the genome, for a warm start
        (float) breeding_rate = 0.8: percentage of population to breed 
        (float) mutation_rate = 0.2: percentage of population to mutate
                each generation
        (float or string) mutation_factor = 'default': given that an
                          organism is being mutated, the probability that
                          a given gene is changed. Defaults to 1/k, where
                          k is the size of the population
        (float) elitism rate = 0.05: percentage of population to keep
                across generations
        (bool) multiprocessing = False: whether fitnesses should be
               computed in parallel
        (int) workers = cpu_count(): how many workers to use if
              multiprocessing is True
        (int) progress_bars = 2: number of progress bars to show, where 1
              only shows the main evolution progress, and 2 shows both
              the evolution and the fitness computation per generation
        (int or string) memory = 'inf': how many generations the
                        population can look back to avoid redundant
                        fitness computations, where 'inf' means unlimited
                        memory.
        (bool) allow_repeats = True: allow computing duplicate fitness vals
        (int) verbose = 0: verbosity mode
        '''

    def __init__(self, genus, size, fitness_fn, initial_genome = None,
        breeding_rate = 0.8, mutation_rate = 0.2, mutation_factor = 'default', 
        elitism_rate = 0.05, multiprocessing = False, workers = cpu_count(), 
        progress_bars = 2, memory = 'inf', allow_repeats = True, verbose = 0):

        self.genus = genus
        self.size = size
        self.fitness_fn = fitness_fn
        self.initial_genome = initial_genome
        self.breeding_rate = breeding_rate
        self.mutation_rate = mutation_rate
        self.mutation_factor = mutation_factor
        self.elitism_rate = elitism_rate
        self.multiprocessing = multiprocessing
        self.workers = workers
        self.progress_bars = progress_bars
        self.memory = memory
        self.allow_repeats = allow_repeats
        self.verbose = verbose

        logging.basicConfig(format = '%(levelname)s: %(message)s')
        self.logger = logging.getLogger()

        if not verbose:
            self.logger.setLevel(logging.WARNING)
        elif verbose == 1:
            self.logger.setLevel(logging.INFO)
        elif verbose == 2:
            self.logger.setLevel(logging.DEBUG)

        self.logger.info("Creating population...")

        if initial_genome:

            # Create a population of identical organisms
            self.population = np.array(
                [Organism(genus, **initial_genome) for _ in range(size)])

            # Mutate 80% of the population
            rnd = np.random.random(self.population.shape)
            for (i, org) in enumerate(self.population):
                if rnd[i] > 0.2:
                    org.mutate()
        else:
            self.population = genus.create_organisms(size)

        self.fittest = np.random.choice(self.population)

    def get_genomes(self):
        return np.asarray([o.get_genome() for o in self.population])

    # Duck typing function to make things immutable
    def make_immutable(self, x):
        try:
            if not isinstance(x, str):
                x = tuple(x)
        except TypeError:
            pass
        return x

    def immute_dict(self, d):
        #return dict(zip(d.keys(), map(self.make_immutable, d.values())))
        return {key : self.make_immutable(val) for (key, val) in d.items()}

    def get_unique_idx(self, genome):
        genome_dict = self.immute_dict(genome)
        genomes = self.get_genomes()
        try:
            idx = np.argmin(genomes != genome_dict)
        except:
            genomes = np.array(list(map(self.immute_dict, genomes)))
            idx = np.argmin(genomes != genome_dict)
        return idx

    def get_fitness(self, history = None):
        ''' Compute fitness values of population.

        INPUT
            (History) history = None: previous genome and fitness history

        OUTPUT
            (ndarray) fitness values
        '''

        unique_genomes = np.array([dict(gene) for gene in
            set(frozenset(self.immute_dict(genome).items())
            for genome in self.get_genomes())
            ])

        # If history is loaded then get the genomes from the current
        # population that are unique across all generations
        past_indices = np.array([])
        if history and not self.allow_repeats:
            g_prev = history.genome_history
            f_prev = history.fitness_history

            indices = np.array([((np.where(g_prev == org.get_genome())[0][0],
                np.where(g_prev == org.get_genome())[1][0]), idx)
                for (idx, org) in enumerate(self.population)
                if org.get_genome() in g_prev
                ])
            past_indices = np.array([idx for (_, idx) in indices])

            # Load previous fitnesses of genomes that are occuring now
            for (past_idx, idx) in indices:
                self.population[idx].fitness = f_prev[past_idx[0], past_idx[1]]

            # Remove genomes that have occured previously
            unique_genomes = np.array([genome for genome in unique_genomes
                if genome not in g_prev])

        # Pull out the organisms with the unique genomes
        imm_genomes = np.array(list(
            map(self.immute_dict, self.get_genomes())))
        imm_unique_genomes = np.array(list(
            map(self.immute_dict, unique_genomes)))
        unique_indices = np.array([np.argmin(imm_genomes != genome) 
            for genome in imm_unique_genomes])

        # If there are any organisms whose fitness we didn't already
        # know then compute them
        if unique_indices.size:

            # Compute fitness values without computing the same one twice
            with warnings.catch_warnings():
                f1_warn = 'F-score is ill-defined and being set to ' \
                          '0.0 due to no predicted samples.'
                warnings.filterwarnings('ignore', message = f1_warn)

                if self.multiprocessing:
                    with Pool(processes = self.workers) as pool:

                        if self.progress_bars >= 2:
                            fit_iter = tqdm(
                                zip(unique_indices, pool.imap(self.fitness_fn, 
                                    self.population[unique_indices])), 
                                total = unique_indices.size
                                )
                            fit_iter.set_description("Computing fitness")
                        else:
                            fit_iter = zip(
                                unique_indices,
                                pool.imap(self.fitness_fn, 
                                    self.population[unique_indices])
                                )
                        for (i, new_fitness) in fit_iter:
                            self.population[i].fitness = new_fitness
                else:
                    if self.progress_bars >= 2:
                        fit_iter = tqdm(
                            zip(unique_indices, map(self.fitness_fn, 
                                self.population[unique_indices])),
                            total = unique_indices.size
                            )
                        fit_iter.set_description("Computing fitness")
                    else:
                        fit_iter = zip(
                            unique_indices, 
                            map(self.fitness_fn, 
                                self.population[unique_indices])
                            )
                    for (i, new_fitness) in fit_iter:
                        self.population[i].fitness = new_fitness


        # Copy out the fitness values to the other organisms with same genome
        for (i, org) in enumerate(self.population):
            if i not in unique_indices and i not in past_indices:
                prev_unique_idx = np.min(np.array([idx
                    for idx in unique_indices
                    if self.immute_dict(org.get_genome()) == \
                        self.immute_dict(self.population[idx].get_genome())
                    ]))
                self.population[i].fitness = \
                    self.population[prev_unique_idx].fitness


    def sample(self, amount = 1):
        ''' Sample a fixed amount of organisms from the population,
            where the fitter an organism is, the more it's likely
            to be chosen. 
    
        INPUT
            (int) amount = 1: number of organisms to sample

        OUTPUT
            (ndarray) sample of population
        '''

        pop = self.population

        # Convert fitness values into probabilities
        fitnesses = np.array([org.fitness for org in pop])
        probs = np.divide(fitnesses, sum(fitnesses))

        # Sort the probabilities in descending order and sort the
        # population in the same way
        sorted_idx = np.argsort(probs)[::-1]
        probs = probs[sorted_idx]
        pop = pop[sorted_idx]

        # Get random numbers between 0 and 1 
        indices = np.random.random(amount)

        for i in range(amount):
            # Find the index of the fitness value whose accumulated
            # sum exceeds the value of the i'th random number.
            fn = lambda x, y: (x[0], x[1] + y[1]) \
                              if x[1] + y[1] > indices[i] \
                              else (x[0] + y[0], x[1] + y[1])
            (idx, _) = reduce(fn, map(lambda x: (1, x), probs))
            indices[i] = idx - 1
        
        # Return the organisms indexed at the indices found above
        return pop[indices.astype(int)]

    def evolve(self, generations = 1, goal = None):
        ''' Evolve the population.

        INPUT
            (int) generations = 1: number of generations to evolve
            (float) goal = None: stop when fitness is above or equal to this
                    value
        '''

        history = History(
            population = self,
            generations = generations,
            memory = self.memory
            )

        if self.progress_bars:
            gen_iter = trange(generations)
            gen_iter.set_description("Evolving population")
        else:
            gen_iter = range(generations)

        for generation in gen_iter:

            if goal and self.fittest.fitness >= goal:
                # Close tqdm iterator
                if self.progress_bars:
                    gen_iter.close()
                self.logger.info('Reached goal, stopping evolution...')
                break

            # Compute fitness values
            self.get_fitness(history = history)
            fitnesses = np.array([org.fitness for org in self.population])
            
            self.logger.debug('Updating fitness values...')

            if max(fitnesses) > self.fittest.fitness:
                self.fittest = self.population[np.argmax(fitnesses)]

            # Store genomes and fitness values in history
            history.add_entry(
                genomes = self.get_genomes(),
                fitnesses = fitnesses,
                generation = generation
                )

            self.logger.debug("Fitness values: {}"\
                .format(np.around(fitnesses, 2)))

            # Select elites 
            elites_amt = np.ceil(self.size * self.elitism_rate).astype(int)
            if self.elitism_rate:
                elites = self.sample(amount = elites_amt)

                self.logger.debug("Elite pool, of size {}:"\
                    .format(elites_amt))
                self.logger.debug(np.array([org.get_genome()
                    for org in elites]))

            breeders_amt = max(2, np.ceil(self.size * self.breeding_rate)\
                .astype(int))
            breeders = self.sample(amount = breeders_amt)

            self.logger.debug("Breeding pool, of size {}:"\
                .format(breeders_amt))
            self.logger.debug(np.array([org.get_genome() for org in breeders]))
            self.logger.debug("Breeding...")

            # Breed until we reach the same size
            children_amt = self.size - elites_amt
            parents = np.random.choice(breeders, (self.size, 2))
            children = np.array([parents[i, 0].breed(parents[i, 1])
                for i in range(children_amt)])

            # Find the mutation pool
            mutators = np.less(np.random.random(children_amt), 
                self.mutation_rate)

            self.logger.debug("Mutation pool, of size {}:"\
                .format(children[mutators].size))
            self.logger.debug(np.array([child.get_genome() for child in
                children[mutators]]))
            self.logger.debug("Mutating...")

            # Mutate the children
            for mutator in children[mutators]:
                mutator.mutate(mutation_factor = self.mutation_factor)

            # The children constitutes our new generation
            if self.elitism_rate:
                self.population = np.append(children, elites)
            else:
                self.population = children
            
            self.logger.debug("New population, of size {}:"\
                .format(self.population.size))
            self.logger.debug(self.get_genomes())
            self.logger.debug("Mean fitness: {}".format(np.mean(fitnesses)))
            self.logger.debug("Std fitness: {}".format(np.std(fitnesses)))

            self.logger.info("Fittest so far, with fitness {}:"\
                .format(self.fittest.fitness))
            self.logger.info(self.fittest.get_genome())

        if self.progress_bars >= 2:
            print("")

        return history

class History():
    ''' History of a population's evolution.
        
    INPUT
        (Population) population
        (int) generations
        (int or string) memory = 20: how many generations the
                        population can look back to avoid redundant
                        fitness computations, where 'inf' means unlimited
                        memory.
    '''

    def __init__(self, population, generations, memory = 20):

        if memory == 'inf' or memory > generations:
            self.memory = generations
        else:
            self.memory = memory

        pop_size = population.size
        self.generations = generations
        self.genome_history = np.empty((self.memory, pop_size), dict)
        self.fitness_history = np.empty((self.memory, pop_size), float)
        self.population = population
        self.fittest = {'genome' : None, 'fitness' : 0}
    
    def add_entry(self, genomes, fitnesses, generation):
        ''' Add genomes and fitnesses to the history. 

        INPUT
            (ndarray) genomes: array of genomes
            (ndarray) fitnesses: array of fitnesses
        '''

        if max(fitnesses) > self.fittest['fitness']:
            self.fittest['genome'] = genomes[np.argmax(fitnesses)]
            self.fittest['fitness'] = max(fitnesses)

        self.genome_history = np.roll(self.genome_history, 1, axis = 0)
        self.genome_history[0, :] = genomes

        self.fitness_history = np.roll(self.fitness_history, 1, axis = 0)
        self.fitness_history[0, :] = fitnesses

        return self

    def plot(self, title = 'Fitness by generation', xlabel = 'Generation',
        ylabel = 'Fitness', file_name = None, show_plot = True,
        show_max = True, discrete = False, legend = True,
        legend_location = 'lower right'):
        ''' Plot the fitness values.

        INPUT
            (string) title = 'Fitness by generation'
            (string) xlabel = 'Generations': label on the x-axis
            (string) ylabel = 'Fitness': label on the y-axis
            (string) file_name = None: file name to save the plot to
            (bool) show_plot = True: show plot as a pop-up
            (bool) show_max = True: show max value line on plot
            (bool) discrete = False: make the error plot discrete
            (bool) legend = True: show legend
            (string or int) legend_location = 'lower right': legend location, 
                            either as e.g. 'lower right' or as an integer
                            between 0 and 10
        '''
        
        fits = self.fitness_history[::-1]
        gens = self.generations
        mem = self.memory
        means = np.mean(fits, axis = 1)
        stds = np.std(fits, axis = 1)
        xs = np.arange(mem)
        if show_max:
            maxs = np.array([np.max(fits[x, :]) for x in xs])

        plt.style.use("ggplot")
        plt.figure()
        plt.xlim(gens - mem, gens - 1)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        xs_shift = xs + (gens - mem)

        if show_max:
            plt.plot(xs_shift, maxs[xs], '--', color = 'blue', label = 'max')

        if discrete:
            plt.errorbar(xs_shift, means[xs], stds[xs], fmt = 'ok', 
                label = 'mean and std')
        else:
            plt.plot(xs_shift, means[xs], '-', color = 'black', label = 'mean')
            plt.fill_between(
                xs_shift, 
                means[xs] - stds[xs],
                means[xs] + stds[xs], 
                alpha = 0.2, 
                color = 'gray', 
                label = 'std'
                )

        if legend:
            plt.legend(loc = legend_location)

        if file_name:
            plt.savefig(file_name)

        if show_plot:
            plt.show()


def __main__():
    pass
