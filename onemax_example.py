import naturalselection as ns
import numpy as np

# Length of number lists
N = 10

# Possible values in number lists
K = 50000

BitString = ns.Genus(**{f'x{n}' : range(K) for n in range(N)})

def sum_bits(bitstring):
  return sum(bitstring.get_genome().values())

bitstrings = ns.Population(
    genus = BitString, 
    size = 100,
    fitness_fn = sum_bits,
    verbose = 1
    )

history = bitstrings.evolve(
    generations = int(1e6),
    progress_bars = 1,
    goal = 499990
    )

print('Most fit:', history.fittest)
history.plot()
