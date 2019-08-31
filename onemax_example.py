import naturalselection as ns
import numpy as np

N = 100
BitString = ns.Genus(**{'x{}'.format(n) : range(100) for n in range(N)})

def sum_bits(bitstring):
  return sum(bitstring.get_genome().values())

bitstrings = ns.Population(
    genus = BitString, 
    size = 2, 
    fitness_fn = sum_bits,
    )
history = bitstrings.evolve(
    generations = 10000000, 
    goal = 10000, 
    progress_bars = 1,
    )
history.plot(show_plot = False, file_name = 'onemax2.png')


bitstrings = ns.Population(
    genus = BitString, 
    size = 10, 
    fitness_fn = sum_bits,
    )
history = bitstrings.evolve(
    generations = 10000000, 
    goal = 10000, 
    progress_bars = 1,
    )
history.plot(show_plot = False, file_name = 'onemax10.png')


bitstrings = ns.Population(
    genus = BitString, 
    size = 100, 
    fitness_fn = sum_bits,
    )
history = bitstrings.evolve(
    generations = 10000000, 
    goal = 10000, 
    progress_bars = 1,
    )
history.plot(show_plot = False, file_name = 'onemax100.png')
