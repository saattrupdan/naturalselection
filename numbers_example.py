import naturalselection as ns

Pair = ns.Genus(x = range(1, 10000), y = range(1, 10000))

pairs = ns.Population(
    genus = Pair, 
    size = 100, 
    fitness_fn = lambda n: max(n.x - n.y, 0),
    )

history = pairs.evolve(generations = 50)

print(history.fittest)

history.plot()
