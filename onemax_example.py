import naturalselection as ns

BitString = ns.Genus(**{f'x{n}' : (0,1) for n in range(100)})

def sum_bits(bitstring):
  return sum(bitstring.get_genome().values())

bitstrings = ns.Population(genus = BitString, size = 2, fitness_fn = sum_bits)
history = bitstrings.evolve(generations = 10000, progress_bars = 1, goal = 100)
history.plot()
