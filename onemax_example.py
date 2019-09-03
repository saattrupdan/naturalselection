import naturalselection as ns

BitString = ns.Genus(**{f'x{n}' : (0,1) for n in range(100)})

def sum_bits(bitstring):
  return sum(bitstring.get_genome().values())

bitstrings = ns.Population(
    genus = BitString, 
    size = 5,
    fitness_fn = sum_bits,
    )

history = bitstrings.evolve(generations = 5000, goal = 100)

print(f"Number of ones achieved: {history.fittest['fitness']}")

history.plot(only_show_max = True)
