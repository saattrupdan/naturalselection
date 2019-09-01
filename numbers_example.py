import naturalselection as ns
import time

Pair = ns.Genus(x = range(1, 10000), y = range(1, 10000))

def division(number):
    #time.sleep(1)
    return number.x / number.y

pairs = ns.Population(
    genus = Pair, 
    size = 500, 
    fitness_fn = division,
    )

history = pairs.evolve(generations = 30)

print(history.fittest)

history.plot()
