import naturalselection as ns

Pair = ns.Genus(x = range(1, 10000), y = range(1, 10000))

def division(number):
  return number.x / number.y

pairs = ns.Population(genus = Pair, size = 100, fitness_fn = division)
history = pairs.evolve(generations = 50, progress_bars = 1)

print(history.fittest)

history.plot()
