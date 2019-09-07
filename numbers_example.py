import naturalselection as ns

Pair = ns.Genus(x = range(1, 10000), y = range(1, 10000))

pairs = ns.Population(
    genus = Pair, 
    size = 100, 
    fitness_fn = lambda n: n.x/n.y,
    )

history = pairs.evolve(generations = 100)

print(history.fittest)

history.plot()
