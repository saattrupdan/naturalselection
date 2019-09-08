# NaturalSelection <img src="https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/naturalselection_data/logo.png" width="50" height="50" alt="Logo of green flower"/>

[![PyPI version](https://badge.fury.io/py/naturalselection.svg)](https://badge.fury.io/py/naturalselection)
[![PyPI weekly downloads](https://img.shields.io/pypi/dw/naturalselection)](https://img.shields.io/pypi/dw/naturalselection)

A general-purpose pythonic genetic algorithm, which includes built-in hyperparameter tuning support for neural networks.


## Installation

```
$ pip install naturalselection
```


## Usage

Here is a toy example looking for a pair of numbers with large first coordinate and small second coordinate:

```python
>>> import naturalselection as ns
>>>
>>> Pair = ns.Genus(x = range(1, 10000), y = range(1, 10000))
>>>
>>> pairs = ns.Population(
...   genus = Pair, 
...   size = 100, 
...   fitness_fn = lambda n: max(n.x - n.y, 0)
...   )
...
>>> history = pairs.evolve(generations = 100)
Evolving population: 100%|██████████████████| 100/100 [00:02<00:00,  35.76it/s]
>>>
>>> history.fittest
{'genome': {'x': 9999, 'y': 1}, 'fitness': 9998}
>>>
>>> history.plot()
```

![Plot showing fitness value over 100 generations.](https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/naturalselection_data/numbers_example.png)


We can also easily solve the classical [OneMax problem](http://tracer.lcc.uma.es/problems/onemax/onemax.html), which is about maximising the number of ones in a bit-string of a given length. Here we set `goal = 100` in the `evolve` function to allow for early stopping if we reach our goal before the maximum number of generations, which we here set to 5,000. Note that it only takes nine seconds.

```python3
>>> import naturalselection as ns
>>>
>>> BitString = ns.Genus(**{f'x{n}': (0,1) for n in range(100)})
>>>
>>> def sum_bits(bitstring):
...   return sum(bitstring.get_genome().values())
...
>>> bitstrings = ns.Population(
...   genus = BitString,
...   size = 5,
...   fitness_fn = sum_bits
...   )
... 
>>> history = bitstrings.evolve(generations = 5000, goal = 100)
Evolving population: 30%|█████            | 1499/5000 [00:08<00:19, 180.93it/s]
>>> 
>>> history.plot(only_show_max = True)
```

![Plot showing fitness value over 1805 generations, converging steadily to the optimal filled out sequence of ones.](https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/naturalselection_data/onemax_example.png)


Lastly, here is an example of finding a fully connected feedforward neural network to model [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist).

Note that the models are trained in parallel, so it is loading in a copy of the data set for every CPU core in your computer, each of which takes up ~600MB of memory. If this causes you to run into memory trouble then you can set the `workers` parameter to something small like 2, or set `multiprocessing = False` to turn parallelism off completely. I've marked these in the code below.

```python3
>>> import naturalselection as ns
>>>
>>> def preprocessing(X):
...   ''' Basic normalisation and scaling preprocessing. '''
...   import numpy as np
...   X = X.reshape((-1, np.prod(X.shape[1:])))
...   X = (X - X.min()) / (X.max() - X.min())
...   X -= X.mean(axis = 0)
...   return X
... 
>>> def fashion_mnist_train_val_sets():
...   ''' Get normalised and scaled Fashion-MNIST train- and val sets. '''
...   from tensorflow.keras.utils import to_categorical
...   from tensorflow.keras.datasets import fashion_mnist
...   (X_train, Y_train), (X_val, Y_val) = fashion_mnist.load_data()
...   X_train = preprocessing(X_train)
...   Y_train = to_categorical(Y_train)
...   X_val = preprocessing(X_val)
...   Y_val = to_categorical(Y_val)
...   return (X_train, Y_train, X_val, Y_val)
...
>>> nns = ns.NNs(
...   size = 20,
...   train_val_sets = fashion_mnist_train_val_sets(),
...   loss_fn = 'categorical_crossentropy',
...   score = 'accuracy',
...   output_activation = 'softmax',
...   max_epochs = 1,
...   max_training_time = 120,
...   # workers = 2, # If you want to reduce parallelism
...   # multiprocessing = False # If you want to disable parallelism
...   )
...
>>> history = nns.evolve(generations = 20)
Evolving population: 100%|███████████████████| 20/20 [1:32:00<00:00, 73.22s/it]
Computing fitness: 100%|███████████████████████| 11/11 [05:37<00:00, 22.52s/it]
>>> 
>>> history.fittest
{'genome': {'optimizer': 'adamax', 'hidden_activation': 'relu',
'batch_size': 32, 'initializer': 'glorot_uniform', 'input_dropout': 0.0,
'neurons0': 256, 'dropout0': 0.3, 'neurons1': 512, 'dropout1': 0.3,
'neurons2': 128, 'dropout2': 0.0, 'neurons3': 0, 'dropout3': 0.1,
'neurons4': 1024, 'dropout4': 0.2}, 'fitness': 0.855400025844574}
>>> 
>>> history.plot(
...   title = "Validation accuracy by generation",
...   ylabel = "Validation accuracy"
...   )
```

![Plot showing fitness value (which is accuracy in this case) over 20 generations](https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/naturalselection_data/fashion_mnist_example.png)

We can then train the best performing model and save it locally:

```python3
>>> # Training the best model and saving it to fashion_mnist_model.h5
>>> best_score = nns.train_best(file_name = 'fashion_mnist_model')
Epoch 0 - val_acc: 0.853: 100%|█████████| 60000/60000 [00:21<00:00, 146.25it/s]
(...)
Epoch 44 - val_acc: 0.900: 100%|████████| 60000/60000 [00:21<00:00, 284.68it/s]
>>>
>>> best_score
0.9029
```

## Algorithmic details

The algorithm follows the standard blueprint for a genetic algorithm as e.g. described on this [Wikipedia page](https://en.wikipedia.org/wiki/Genetic_algorithm), which roughly goes like this:

1. An initial population is constructed
2. Fitness values for all organisms in the population are computed
3. A subset of the population (the *elite pool*) is selected
4. A subset of the population (the *breeding pool*) is selected
5. Pairs from the breeding pool are chosen, who will breed to create a new "child" organism with genome a combination of the "parent" organisms. Continue breeding until the children and the elites constitute a population of the same size as the original
6. A subset of the children (the *mutation pool*) is selected
7. Every child in the mutation pool is mutated, meaning that they will have their genome altered in some way
8. Go back to step 2

We now describe the individual steps in this particular implementation in more detail. Note that step 3 is sometimes left out completely, but since that just corresponds to an empty elite pool I decided to keep it in, for generality.

### Step 1: Constructing the initial population

The population is a uniformly random sample of the possible genome values as dictated by the genus, which is run when a new `Population` object is created. Alternatively, you may set the `initial_genome` to whatever genome you would like, which will create a population consisting of organisms similar to this genome (the result of starting with a population all equal to the organism and then mutating 80% of them).

```python3
>>> pairs = ns.Population(
...   genus = Pair,
...   size = 100,
...   fitness_fn = lambda n: n.x/n.y,
...   initial_genome = {'x': 9750, 'y': 15}
...   )
...
>>> history = pairs.evolve(generations = 100)
Evolving population: 100%|██████████████████| 100/100 [00:05<00:00,  19.47it/s]
>>> 
>>> history.fittest
{'genome': {'x': 9989, 'y': 3}, 'fitness': 3329.66666666665}
```

### Step 2: Compute fitness values

This happens in the `update_fitness` function which is called by the `evolve` function. These computations will by default be computed in parallel when dealing with neural networks and serialised otherwise, as the benefits are only reaped when fitness computations take up a significant part of the algorithm (in the examples above not concerning neural networks we would actually slow down the algorithm non-trivially by introducing parallelism).

### Steps 3 & 4: Selecting elite pool and breeding pool

These two pools are selected in exactly the same way, using the `sample` function. They only differ in the amount of organisms sampled, where the default `elitism_rate` is 5% and `breeding_rate` is 80%. In the pool selection it chooses the population based on the distribution with density function the fitness value divided by the sum of all fitness values of the population. This means that the higher fitness score an organism has, the more likely it is for it to be chosen to be a part of the pool. The precise implementation of this is based on the algorithm specified on this [Wikipedia page](https://en.wikipedia.org/wiki/Selection_(genetic_algorithm)).

### Step 5: Breeding

In this implementation the parent organisms are chosen uniformly at random from the breeding pool. When determining the value of the child's genome we apply the "single-point crossover" method, where we choose an index uniformly at random among the attributes, and the child will then inherit all attributes to the left of this index from one parent and the attributes to the right of this index from the other parent.See more on [this Wikipedia page](https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)).

### Step 6: Selection of mutation pool

The mutation pool is chosen uniformly at random in contrast to the other two pools, as otherwise we would suddenly be more likely to "mutate away" many of the good genes of our fittest organisms. The default `mutation_rate` is 20%.

### Step 7: Mutation

This implementation is roughly the [bit string mutation](https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm)), where every gene of the organism has a 1/n chance of being uniformly randomly replaced by another gene, with n being the number of genes in the organism's genome. This means that, on average, mutation causes one gene to be altered. The amount of genes altered in a mutation can be modified by changing thè `mutation_factor` parameter, which by default is the above 1/n.


## Possible future extensions

These are the ideas that I have thought of implementing in the future. Check the ongoing process on the `dev` branch.

* Enable support for CNNs
* Enable support for RNNs and in particular LSTMs
* Include an option to have dependency relations between genes. In a neural network setting this could include the topology as a gene on which all the layer-specific genes depend upon, which would be similar to the approach taken in [this paper](https://arxiv.org/pdf/1703.00548/).


## License

This project is licensed under the [MIT License](https://github.com/saattrupdan/naturalselection/blob/master/LICENSE).
