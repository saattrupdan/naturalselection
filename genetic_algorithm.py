import numpy as np
import itertools
from tqdm import tqdm

class Genus():

    def __init__(self, dna_range):
        self.dna_range = dna_range

    def create_organisms(self, amount = 1):
        orgs = [Organism(genus = self, dna =
            {key : np.random.choice(self.dna_range[key])
            for key in self.dna_range.keys()}) for i in range(amount)]
        return np.array(orgs)

    def create_genus(self, add = None, remove = None):
        dna_range = np.append(np.array([param for param in self.dna_range
                     if param.key not in remove]), add)
        return Genus(dna_range)

class Organism():

    def __init__(self, genus: Genus, dna: dict):
        if not genus.dna_range.keys() == dna.keys():
            raise TypeError('Genus keys do not match input keys.')
        self.genus = genus
        self.dna = dna

    def breed(self, other):
        if not self.genus == other.genus:
            raise TypeError('The two organisms are not of the same genus.')
        child_dna = {
            key : np.random.choice([self.dna[key], other.dna[key]])
                  for key in self.dna.keys()
            }
        return Organism(self.genus, child_dna)

    def mutate(self):
        keys = np.array([*self.genus.dna_range])
        random_key = np.random.choice(keys)
        random_val = np.random.choice(self.genus.dna_range[random_key])
        self.dna[random_key] = random_val
        return self

class Population():

    def __init__(self, genus, size, fitness_fn):
        self.genus = genus
        self.size = size
        self.fitness_fn = fitness_fn
        self.population = self.genus.create_organisms(self.size)

    def get_fittest(self, top_n = 1):
        fitnesses = np.array([self.fitness_fn(org) for org in self.population])
        fittest_idx = np.argpartition(fitnesses, -top_n)[-top_n:]
        fittest_orgs = self.population[fittest_idx]
        return fittest_orgs[0] if top_n == 1 else fittest_orgs

    def evolve(self, generations = 1, keep = 0.30, mutate = 0.10):
        for generation in tqdm(range(generations)):
            # keep the fittest organisms of the population
            keep_amount = max(2, np.ceil(self.size * keep).astype(int))
            self.population = self.get_fittest(keep_amount)

            # breed until we reach the same size
            remaining = self.size - keep_amount
            for i in range(remaining):
                orgs = np.random.choice(self.population, 2)
                new_org = orgs[0].breed(orgs[1])
                self.population = np.append(self.population, new_org)

            # mutate randomly
            mutate_amount = np.ceil(self.size * mutate).astype(int)
            mutators = np.random.choice(self.population, mutate_amount)
            for mutator in mutators:
                mutator.mutate()
        return self


if __name__ == '__main__':
    Number = Genus({'val' : range(0, 100000)})
    fn = lambda number: number.dna['val']
    numbers = Population(genus = Number, size = 3, fitness_fn = fn)
    numbers.evolve(generations = 500)
    print(numbers.get_fittest().dna['val'])
