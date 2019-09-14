import numpy as np
import sys
import os
from functools import reduce, partial
import logging
import warnings
from inspect import getfullargspec
import multiprocessing as mp

# Plots
import matplotlib.pyplot as plt

# Progress bars
from tqdm import tqdm, trange

class Genus():
    ''' Storing information about all the possible gene combinations.

    INPUT
        (kwargs) genomes
    '''

    def __init__(self, **genomes):
        self.__dict__.update(
            {key: np.asarray(val) for (key, val) in genomes.items()}
            )

    def create_organism(self):
        rnd_genes = {key: val[np.random.choice(range(val.shape[0]))]
            for (key, val) in self.__dict__.items()}
        return Organism(genus = self, **rnd_genes)

    def create_organisms(self, amount = 1):
        ''' Create organisms of this genus.
        
        INPUT
            (int) amount = 1
        '''
        return np.array([self.create_organism() for _ in range(amount)])

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

        def in_arr(val, arr):
            ''' Check if value occurs as a row in a 1d or 2d array. '''

            if np.asarray(val).size == 1:
                return val in arr
            else:
                results = (True for i in range(arr.shape[0]) 
                                if (arr[i, :] == np.asarray(val)).all())
                try:
                    result = next(results)
                except StopIteration:
                    result = False
                return result

        # Check that the input parameters match with the genus type,
        # and if any parameters are missing then add random values
        genome = {key: val for (key, val) in genome.items() if key in
            genus.__dict__.keys() and in_arr(val, genus.__dict__[key])}

        for key in genus.__dict__.keys() - genome.keys():
            val_idx = np.random.choice(range(genus.__dict__[key].shape[0]))
            genome[key] = genus.__dict__[key][val_idx]

        self.__dict__.update(genome)
        self.genus = genus
        self.fitness = 0

    def get_genome(self):
        return {key: val for (key, val) in self.__dict__.items()
                          if key not in {'genus', 'fitness'}}

    def breed(self, other):
        ''' Breed organism with another organism, returning a new
            organism of the same genus.

        INPUT
            (Organism) other organism
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

        # Get keys that have more than one possible gene value
        all_keys = self.get_genome().keys()
        keys = np.array(
            [k for k in all_keys if self.genus.__dict__[k].size > 1]
            )

        if mutation_factor == 'default':
            mutation_factor = np.divide(1, keys.size)

        # Get the genes that are to be mutated
        mut_keys = keys[np.less(np.random.random(keys.size), mutation_factor)]

        # Mutation loop
        mut_dict = {}
        for key in mut_keys:

            # Get the index of the current gene values in the array of all
            # possible gene values for that gene
            gene_vals = self.genus.__dict__[key]
            gene_type = gene_vals.dtype.type
            
            # If the gene values are numeric then choose the mutated gene
            # value following a normal distribution, otherwise a uniform one
            if issubclass(gene_type, np.integer) or \
               issubclass(gene_type, np.floating):
                
                # Get a new index for a gene value for the given gene, taken
                # from a normal distribution centered on gene_idx and with
                # the above standard deviation
                if len(gene_vals.shape) == 1:

                    # Sort them to make the normal distribution more effective
                    sorted_genes = np.sort(gene_vals)
        
                    # Find the index of the current gene value
                    gene_idx = np.where(
                        sorted_genes == self.get_genome()[key])
                    gene_idx = gene_idx[0][0]

                    # Set a standard deviation
                    scale = np.around(sorted_genes.shape[0] / 2, 0)
                    scale = scale.astype(int)
                    scale = max(scale, 1)

                    rnd_idx = gene_idx
                    while rnd_idx == gene_idx and sorted_genes.size != 1:
                        rnd_idx = np.random.normal(
                            loc = gene_idx, 
                            scale = scale
                            )
                        rnd_idx = np.around(rnd_idx, 0)
                        rnd_idx = abs(rnd_idx)
                        rnd_idx = min(rnd_idx, len(sorted_genes) - 1)
                        rnd_idx = int(rnd_idx)

                    # Save the new gene value
                    mut_dict[key] = sorted_genes[rnd_idx]

                else:
                    dim = gene_vals.shape[1]

                    gene_val_arr = []
                    gene_idx = np.empty(dim)
                    scales = np.empty(dim)
                    for i in range(dim):
                        gene_val_arr.append(np.unique(gene_vals[:, i]))
                        gene_val_arr[i] = np.sort(gene_val_arr[i])
                        gene_idx[i] = np.where(gene_val_arr[i] == 
                            self.get_genome()[key][i])[0][0]
                        scales[i] = max(gene_val_arr[i].shape[0] / 2, 1)

                    def eligible(idx):
                        ''' Check if idx indexes a valid gene value. '''
                        gene_val = np.array([gene_val_arr[i][idx[i]]
                            for i in range(dim)])
                        results = (True for i in range(gene_vals.shape[0])
                            if (gene_vals[i, :] == gene_val).all())
                        try:
                            result = next(results)
                        except StopIteration:
                            result = False
                        return result

                    for _ in range(10):
                        cov = np.zeros((dim, dim))
                        for i in range(dim):
                            cov[i,i] = scales[i]
                        rnd_idx = np.random.multivariate_normal(
                            mean = gene_idx, 
                            cov = cov
                            )
                        rnd_idx = np.around(rnd_idx, 0)
                        rnd_idx = np.absolute(rnd_idx)
                        for i in range(dim):
                            rnd_idx[i] = np.minimum(rnd_idx[i], 
                                len(gene_val_arr[i]) - 1)
                        rnd_idx = rnd_idx.astype(int)

                        rnd_val = np.empty(dim)
                        for i in range(dim):
                            rnd_val[i] = gene_val_arr[i][rnd_idx[i]]

                    rnd_idx = gene_idx
                    while (np.array(rnd_idx) == np.array(gene_idx)).all() \
                        or not eligible(rnd_idx):

                        cov = np.zeros((dim, dim))
                        for i in range(dim):
                            cov[i,i] = scales[i]
                        rnd_idx = np.random.multivariate_normal(
                            mean = gene_idx, 
                            cov = cov
                            )
                        rnd_idx = np.around(rnd_idx, 0)
                        rnd_idx = np.absolute(rnd_idx)
                        for i in range(dim):
                            rnd_idx[i] = np.minimum(rnd_idx[i], 
                                len(gene_val_arr[i]) - 1)
                        rnd_idx = rnd_idx.astype(int)

                    # Save the new gene value
                    rnd_val = np.empty(dim)
                    for i in range(dim):
                        rnd_val[i] = gene_val_arr[i][rnd_idx[i]]
                    mut_dict[key] = rnd_val

            else:
                rnd_idx = np.random.choice(range(gene_vals.size))
                # Save the new gene value
                mut_dict[key] = gene_vals[rnd_idx]

        # Replace the old gene values by the new ones
        self.__dict__.update(mut_dict)
        return self

class Population():
    ''' Population of organisms, all of the same genus.

    INPUT
        (Genus) genus
        (int) size: size of the population
        (function) fitness_fn: fitness function
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
        (int) workers = mp.cpu_count(): how many workers to use if
              multiprocessing is True
        (int) progress_bars = 1: number of progress bars to show, where 1
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
        elitism_rate = 0.05, multiprocessing = False, workers = mp.cpu_count(),
        progress_bars = 1, memory = 'inf', allow_repeats = True,
        verbose = 0):

        self.genus = genus
        self.size = size
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

        if 'worker_idx' not in getfullargspec(fitness_fn).args:
            def new_fitness_fn(*args, worker_idx = None, **kwargs):
                return fitness_fn(*args, **kwargs)
            self.fitness_fn = new_fitness_fn
        else:
            self.fitness_fn = fitness_fn

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
        return np.asarray([org.get_genome() for org in self.population])

    def get_fitnesses(self):
        return np.asarray([org.fitness for org in self.population])

    def update_fitness(self, history = None):
        ''' Compute and update fitness values of the population.

        INPUT
            (History) history = None: previous population history
        '''

        # Duck typing function to make things immutable
        def make_immutable(x):
            try:
                if not isinstance(x, str):
                    x = tuple(x)
            except TypeError:
                pass
            return x

        def immute_dict(d):
            return {key: make_immutable(val) for (key, val) in d.items()}

        unique_genomes = np.array(
            [dict(gene) for gene in set(frozenset(immute_dict(genome).items())
            for genome in self.get_genomes())]
            )

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
            map(immute_dict, self.get_genomes())))
        imm_unique_genomes = np.array(list(
            map(immute_dict, unique_genomes)))
        unique_indices = np.array([np.argmin(imm_genomes != genome) 
            for genome in imm_unique_genomes])

        # Compute fitness values if there are any that needs to be computed
        if unique_indices.size:
            with warnings.catch_warnings():

                # Ignore warning related to F1-scores
                f1_warn = 'F-score is ill-defined and being set to ' \
                          '0.0 due to no predicted samples.'
                warnings.filterwarnings('ignore', message = f1_warn)

                if self.multiprocessing:

                    # Define queues to organise the parallelising
                    todo = mp.Queue(unique_indices.size + self.workers)
                    done = mp.Queue(unique_indices.size)
                    for idx in unique_indices:
                        todo.put(idx)
                    for _ in range(self.workers):
                        todo.put(-1)

                    def worker(todo, done):
                        ''' Fitness computing worker. '''
                        from queue import Empty
                        while True:
                            try:
                                idx = todo.get(timeout = 1)
                            except Empty:
                                continue
                            if idx == -1:
                                break
                            else:
                                org = self.population[idx]
                                worker_idx = mp.current_process()._identity[0]
                                fitness = self.fitness_fn(org,
                                    worker_idx = worker_idx)
                                done.put((idx, fitness))

                    # Define our processes
                    processes = [mp.Process(target = worker,
                        args = (todo, done)) for _ in range(self.workers)]

                    # Daemonise the processes, meaning they close when they
                    # they finish, and start them
                    for p in processes:
                        p.daemon = True
                        p.start()

                    # This is the iterable with (idx, fitness) values
                    idx_fits = (done.get() for _ in unique_indices)

                else:
                    # This is the iterable with (idx, fitness) values,
                    # obtained without any parallelising
                    idx_fits = self.population[unique_indices]
                    idx_fits = map(self.fitness_fn, idx_fits)
                    idx_fits = zip(unique_indices, idx_fits)
        
                # Set up a progress bar
                if self.progress_bars >= 2:
                    idx_fits = tqdm(idx_fits, total = unique_indices.size)
                    idx_fits.set_description("Computing fitness")

                # Compute the fitness values
                for (idx, new_fitness) in idx_fits:
                    self.population[idx].fitness = new_fitness

                # Join up the processes
                if self.multiprocessing:
                    for p in processes:
                        p.join()
               
                # Close the progress bar 
                if self.progress_bars >= 2:
                    idx_fits.close()


        # Copy out the fitness values to the other organisms with same genome
        for (i, org) in enumerate(self.population):
            if i not in unique_indices and i not in past_indices:
                prev_unique_idx = np.min(np.array(
                    [idx for idx in unique_indices
                         if immute_dict(org.get_genome()) == \
                         immute_dict(self.population[idx].get_genome())]
                    ))
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

        # Convert fitness values into probabilities
        fitnesses = self.get_fitnesses()
        probs = np.divide(fitnesses, sum(fitnesses))
        
        # Copy the population to a new variable
        pop = self.population

        # Sort the probabilities in descending order and sort pop (not
        # the actual population) in the same way
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
            (float) goal = None: stop when fitness is not below this value
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

        for gen in gen_iter:

            if goal and self.fittest.fitness >= goal:
                # Close tqdm iterator
                if self.progress_bars:
                    gen_iter.close()
            
                # Truncate history for plotting
                history.generations = gen
                history.fitness_history = history.fitness_history[:gen, :]
                history.genome_history = history.genome_history[:gen, :]
                if history.memory == 'inf' or history.memory > gen:
                    history.memory = gen

                if self.verbose == 2:
                    if self.progress_bars:
                        print("")
                    if self.progress_bars >= 2:
                        print("")
                self.logger.info('Reached goal, stopping evolution...')
                break

            if self.verbose == 2:
                if self.progress_bars:
                    print("")
                if self.progress_bars >= 2:
                    print("")
            self.logger.debug("Current population, of size {}:"\
                .format(self.population.size))
            self.logger.debug(np.array([(org.get_genome(), org.fitness)
                for org in self.population]))

            # Compute and update fitness values
            self.update_fitness(history = history)
            fitnesses = self.get_fitnesses()
            
            self.logger.debug('Updating fitness values...')

            # Update the fittest organism
            if max(fitnesses) > self.fittest.fitness:
                self.fittest = self.population[np.argmax(fitnesses)]

            self.logger.debug("Current population with fitness values:")
            self.logger.debug(np.array([(org.get_genome(), org.fitness)
                for org in self.population]))

            # Store current population in history
            history.add_entry(self, generation = gen)

            # Select elites 
            elites_amt = np.ceil(self.size * self.elitism_rate).astype(int)
            if self.elitism_rate:
                elites = self.sample(amount = elites_amt)

                self.logger.debug("Elite pool, of size {}:"\
                    .format(elites_amt))
                self.logger.debug(np.array([(org.get_genome(), org.fitness)
                    for org in elites]))

            # Select breeders
            breeders_amt = max(2, np.ceil(self.size * self.breeding_rate)\
                .astype(int))
            breeders = self.sample(amount = breeders_amt)

            self.logger.debug("Breeding pool, of size {}:"\
                .format(breeders_amt))
            self.logger.debug(np.array([(org.get_genome(), org.fitness)
                for org in breeders]))
            self.logger.debug("Breeding...")

            # Breed until we reach the same size
            children_amt = self.size - elites_amt
            parents = np.random.choice(breeders, (self.size, 2))
            children = np.array([parents[i, 0].breed(parents[i, 1])
                for i in range(children_amt)])

            # Select mutators
            mutators = np.less(np.random.random(children_amt), 
                self.mutation_rate)

            self.logger.debug("Mutation pool, of size {}:"\
                .format(children[mutators].size))
            self.logger.debug(np.array([(child.get_genome(), child.fitness)
                for child in children[mutators]]))
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
            self.logger.debug(np.array([(org.get_genome(), org.fitness)
                for org in self.population]))

            if self.verbose == 1:
                if self.progress_bars:
                    print("")
                if self.progress_bars >= 2:
                    print("")
            self.logger.info("Median: {}".format(np.median(fitnesses)))
            self.logger.info("IQR: {}".format(
                np.percentile(fitnesses, 75) - np.percentile(fitnesses, 25)))
            self.logger.info("Fittest, with fitness {}:"\
                .format(self.fittest.fitness))
            self.logger.info(self.fittest.get_genome())

        gen_iter.close()

        if self.progress_bars >= 2:
            print("")

        return history

class History():
    ''' History of a population's evolution.
        
    INPUT
        (Population) population
        (int) generations
        (int or string) memory = 'inf': how many generations the
                        population can look back to avoid redundant
                        fitness computations, where 'inf' means unlimited
                        memory.
    '''

    def __init__(self, population, generations, memory = 'inf'):

        if memory == 'inf' or memory > generations:
            self.memory = min(int(1e5), generations)
        else:
            self.memory = memory

        pop_size = population.size
        self.generations = generations
        self.genome_history = np.empty((self.memory, pop_size), dict)
        self.fitness_history = np.empty((self.memory, pop_size), float)
        self.population = population
        self.fittest = {'genome': None, 'fitness': 0}
    
    def add_entry(self, population, generation):
        ''' Add population to the history. 

        INPUT
            (Population) population
            (int) generation
        '''

        genomes = population.get_genomes()
        fitnesses = population.get_fitnesses()

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
        show_max = True, only_show_max = False, show_min = False,
        show_minmax_area = False, show_quartile_area = True,
        show_lower_quartile = False, show_upper_quartile = False,
        show_median = True, legend = True, legend_location = 'lower right',
        show_all_lines = False, show_all_areas = False):
        ''' Plot the fitness values.

        INPUT
            (string) title = 'Fitness by generation'
            (string) xlabel = 'Generations': label on the x-axis
            (string) ylabel = 'Fitness': label on the y-axis
            (string) file_name = None: file name to save the plot to
            (bool) show_plot = True: show plot as a pop-up
            (bool) show_median = True: show a median value line on plot
            (bool) show_max = True: show a maximum value line on plot
            (bool) show_min = False: show a minimum value line on plot
            (bool) show_lower_quartile = False: show a lower quartile
                   value line on plot
            (bool) show_upper_quartile = False: show an upper quartile
                   value line on plot
            (bool) show_all_lines = False: show minimum, lower quartile,
                   median, upper quartile and maximum lines on plot
            (bool) show_minmax_area = False: show a filled area between the
                   minima and the maxima on plot
            (bool) show_quartile_area = True: show a filled area between the
                   lower and upper quartiles on plot
            (bool) show_all_areas = False: show both minmax and quartile area
            (bool) only_show_max = False: only show the max value line
            (bool) legend = True: show legend
            (string or int) legend_location = 'lower right': legend location, 
                            either as e.g. 'lower right' or as an integer
                            between 0 and 10
        '''
        
        fits = self.fitness_history[::-1]
        gens = self.generations
        mem = self.memory
        xs = np.arange(mem)
        xs_shift = xs + (gens - mem)

        meds = np.median(fits, axis = 1)
        lower_quartiles = np.percentile(fits, 25, axis = 1)
        upper_quartiles = np.percentile(fits, 75, axis = 1)
        mins = np.array([np.min(fits[x, :]) for x in xs])
        maxs = np.array([np.max(fits[x, :]) for x in xs])

        if gens == 1:
            discrete = True

        plt.style.use("ggplot")
        plt.figure()
        plt.xlim(gens - mem - 1, gens)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if show_max or only_show_max or show_all_lines:
            plt.plot(xs_shift, maxs[xs], '-', color = 'blue', label = 'max')

        if not only_show_max:
            if show_median or show_all_lines:
                plt.plot(xs_shift, meds[xs], '-', color = 'black', 
                    label = 'median')
            if show_min or show_all_lines:
                plt.plot(xs_shift, mins[xs], '-', color = 'red', 
                    label = 'min')
            if show_lower_quartile or show_all_lines:
                plt.plot(xs_shift, lower_quartiles[xs], ':', color = 'red',
                    label = 'lower quartile')
            if show_upper_quartile or show_all_lines:
                plt.plot(xs_shift, upper_quartiles[xs], ':', color = 'blue',
                    label = 'upper quartile')
            if show_minmax_area or show_all_areas:
                plt.fill_between(xs_shift, mins[xs], maxs[xs], alpha = 0.1, 
                    color = 'gray', label = 'min-max')
            if show_quartile_area or show_all_areas:
                plt.fill_between(xs_shift, lower_quartiles[xs],
                    upper_quartiles[xs], alpha = 0.3, color = 'gray', 
                    label = 'quartiles')

        if legend:
            plt.legend(loc = legend_location)

        if file_name:
            plt.savefig(file_name)

        if show_plot:
            plt.show()


def __main__():
    pass
