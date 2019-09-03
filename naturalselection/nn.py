import numpy as np
import os
from functools import partial 
from multiprocessing import cpu_count, current_process
from naturalselection.core import Genus, Population, Organism
import logging

class FNN(Genus):
    ''' Feedforward fully connected neural network genus.

    INPUT:
        (int) max_nm_hidden_layers
        (bool) uniform_layers: whether all hidden layers should
               have the same amount of neurons and dropout
        (iterable) dropout: values for dropout
        (iterable) neurons_per_hidden_layer = neurons in hidden layers
        (iterable) optimizer: keras optimizers
        (iterable) hidden_activation: keras activation functions
        (iterable) batch_size: batch sizes
        (iterable) initializer: keras initializers
        '''
    def __init__(self,
        max_nm_hidden_layers = 5,
        uniform_layers = False,
        input_dropout = np.arange(0, 0.6, 0.1),
        hidden_dropout = np.arange(0, 0.6, 0.1),
        neurons = np.array([2 ** n for n in range(4, 11)]),
        optimizer = np.array(['sgd', 'rmsprop', 'adagrad', 'adadelta',
                              'adamax', 'adam', 'nadam']),
        hidden_activation = np.array(['relu', 'elu', 'softplus', 'softsign']),
        batch_size = np.array([2 ** n for n in range(4, 7)]),
        initializer = np.array(['lecun_uniform', 'lecun_normal',
                                'glorot_uniform', 'glorot_normal',
                                'he_uniform', 'he_normal'])):

        self.optimizer = np.unique(np.asarray(optimizer))
        self.hidden_activation = np.unique(np.asarray(hidden_activation))
        self.batch_size = np.unique(np.asarray(batch_size))
        self.initializer = np.unique(np.asarray(initializer))
        self.input_dropout = np.unique(np.asarray(input_dropout))

        if uniform_layers:
            self.neurons = np.unique(np.asarray(neurons))
            self.dropout = np.unique(np.asarray(hidden_dropout))
            self.nm_hidden_layers = \
                np.arange(1, max_nm_hidden_layers + 1)
        else:
            neurons = np.unique(np.append(neurons, 0))
            dropout = np.around(np.unique(np.append(hidden_dropout, 0)), 2)
            layer_info = {}
            for layer_idx in range(max_nm_hidden_layers):
                layer_info["neurons{}".format(layer_idx)] = neurons
                layer_info["dropout{}".format(layer_idx)] = dropout
            self.__dict__.update(layer_info)

class FNNs(Population):
    def __init__(self, 
        train_val_sets,
        size = 50, 
        initial_genome = None,
        breeding_rate = 0.8,
        mutation_rate = 0.2,
        mutation_factor = 'default',
        elitism_rate = 0.05,
        multiprocessing = True,
        workers = cpu_count(),
        loss_fn = 'binary_crossentropy',
        nm_features = 'infer', 
        nm_labels = 'infer',
        score = 'accuracy', 
        output_activation = 'sigmoid',
        max_epochs = 1000000, 
        patience = 5, 
        min_change = 1e-4,
        max_training_time = None, 
        max_nm_hidden_layers = 5,
        uniform_layers = False,
        input_dropout = np.arange(0, 0.6, 0.1),
        hidden_dropout = np.arange(0, 0.6, 0.1),
        neurons = np.array([2 ** n for n in range(4, 11)]),
        optimizer = np.array(['sgd', 'rmsprop', 'adagrad', 'adadelta',
                              'adamax', 'adam', 'nadam']),
        hidden_activation = np.array(['relu', 'elu', 'softplus',
                                      'softsign']),
        batch_size = np.array([2 ** n for n in range(4, 7)]),
        initializer = np.array(['lecun_uniform', 'lecun_normal',
                                'glorot_uniform', 'glorot_normal',
                                'he_uniform', 'he_normal']),
        verbose = 0):

        self.train_val_sets       = train_val_sets
        self.size                 = size
        self.initial_genome       = initial_genome
        self.breeding_rate        = breeding_rate
        self.mutation_rate        = mutation_rate
        self.mutation_factor      = mutation_factor
        self.elitism_rate         = elitism_rate
        self.multiprocessing      = multiprocessing
        self.workers              = workers
        self.loss_fn              = loss_fn
        self.nm_features          = nm_features
        self.nm_labels            = nm_labels
        self.score                = score
        self.output_activation    = output_activation
        self.max_epochs           = max_epochs 
        self.patience             = patience
        self.min_change           = min_change
        self.max_training_time    = max_training_time
        self.max_nm_hidden_layers = max_nm_hidden_layers
        self.uniform_layers       = uniform_layers
        self.input_dropout        = input_dropout
        self.hidden_dropout       = hidden_dropout
        self.neurons              = neurons
        self.optimizer            = optimizer
        self.hidden_activation    = hidden_activation
        self.batch_size           = batch_size
        self.initializer          = initializer
        self.verbose              = verbose

        logging.basicConfig(format = '%(levelname)s: %(message)s')
        self.logger = logging.getLogger()

        if not verbose:
            self.logger.setLevel(logging.WARNING)
        elif verbose == 1:
            self.logger.setLevel(logging.INFO)
        elif verbose == 2:
            self.logger.setLevel(logging.DEBUG)

        self.logger.info("Creating population...")

        # Hard coded values for neural networks
        self.allow_repeats = False
        self.memory = 'inf'
        self.progress_bars = 2
        self.chunksize = np.ceil(self.size / self.workers).astype(int)
        
        self.genus = FNN(
            max_nm_hidden_layers = self.max_nm_hidden_layers,
            uniform_layers       = self.uniform_layers,
            input_dropout        = self.input_dropout,
            hidden_dropout       = self.hidden_dropout,
            neurons              = self.neurons,
            optimizer            = self.optimizer,
            hidden_activation    = self.hidden_activation,
            batch_size           = self.batch_size,
            initializer          = self.initializer
            )

        self.fitness_fn = partial(
            self.train_fnn,
            max_epochs          = self.max_epochs,
            patience            = self.patience,
            min_change          = self.min_change,
            max_training_time   = self.max_training_time,
            verbose             = 0,
            file_name           = None
            )

        # If user has supplied an initial genome then construct a population
        # which is very similar to that
        if initial_genome:

            # Create a population of organisms all with the initial genome
            self.population = np.array(
                [Organism(self.genus, **initial_genome) for _ in range(size)])

            # Mutate 80% of the population
            rnd = np.random.random(self.population.shape)
            for (i, org) in enumerate(self.population):
                if rnd[i] > 0.2:
                    org.mutate()
        else:
            self.population = self.genus.create_organisms(size)

        # We do not have access to fitness values yet, so choose the 'fittest
        # organism' to just be a random one
        self.fittest = np.random.choice(self.population)

    def train_best(self, max_epochs = 1000000, min_change = 1e-4,
        patience = 5, max_training_time = None, file_name = None):

        best_fnn = self.fittest        
        fitness = self.train_fnn(
            fnn                 = best_fnn,
            max_epochs          = max_epochs,
            patience            = patience,
            min_change          = min_change,
            max_training_time   = max_training_time,
            verbose             = 1,
            file_name           = file_name
            )
        return fitness

    def train_fnn(self, fnn, max_epochs = 1000000, patience = 5,
        min_change = 1e-4, max_training_time = None, verbose = False,
        file_name = None):
        ''' Train a feedforward neural network and output the score.
        
        INPUT
            (FNN) fnn: a feedforward neural network genus
            (int) max_epochs: maximum number of epochs to train for
            (int) patience: number of epochs with no progress above min_change
            (float) min_change: everything below this number won't count as a
                    change in the score
            (int) max_training_time: maximum number of seconds to train for,
                  also training the final epoch after the time has run out
            (int) verbose: verbosity mode

        OUTPUT
            (float) the score of the neural network
        '''

        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Dropout
        from tensorflow.keras import backend as K
        from tensorflow.python.util import deprecation
        from tensorflow import set_random_seed
        
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import f1_score, precision_score, recall_score

        # Custom callbacks
        from .callbacks import TQDMCallback, EarlierStopping

        # Used when building network
        from itertools import count

        # Suppress tensorflow warnings
        deprecation._PRINT_DEPRECATION_WARNINGS = False
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # Set random seeds for reproducibility
        np.random.seed(0)
        set_random_seed(0)

        X_train, Y_train, X_val, Y_val = self.train_val_sets

        if self.nm_features == 'infer':
            self.nm_features = X_val.shape[1]
        if self.nm_labels == 'infer':
            self.nm_labels = Y_val.shape[1]

        inputs = Input(shape = (self.nm_features,))
        x = Dropout(fnn.input_dropout)(inputs)

        if self.uniform_layers:
            for _ in range(fnn.nm_hidden_layers):
                x = Dense(fnn.neurons, activation = fnn.hidden_activation,
                    kernel_initializer = fnn.initializer)(x)
                x = Dropout(fnn.dropout)(x)
        else:
            for i in count():
                try:
                    neurons = fnn.__dict__["neurons{}".format(i)]
                    if neurons:
                        x = Dense(neurons, activation = fnn.hidden_activation,
                            kernel_initializer = fnn.initializer)(x)
                    dropout = fnn.__dict__["dropout{}".format(i)]
                    if dropout:
                        x = Dropout(dropout)(x)
                except:
                    break

        outputs = Dense(self.nm_labels,
            activation = self.output_activation,
            kernel_initializer = fnn.initializer)(x)

        nn = Model(inputs = inputs, outputs = outputs)

        nn.compile(
            loss = self.loss_fn,
            optimizer = fnn.optimizer,
            )

        early_stopping = EarlierStopping(
            monitor = 'val_loss',
            patience = patience,
            min_delta = min_change,
            restore_best_weights = True,
            seconds = max_training_time
            )

        callbacks = [early_stopping]
        if verbose:
            tqdm_callback = TQDMCallback(show_outer = False)
            callbacks.append(tqdm_callback)

        nn.fit(
            X_train,
            Y_train,
            batch_size = fnn.batch_size,
            validation_data = (X_val, Y_val),
            epochs = max_epochs,
            callbacks = callbacks,
            verbose = 0
            )

        if file_name:
            nn.save("{}.h5".format(file_name))

        if self.nm_labels > 1:
            average = 'micro'
        else:
            average = None

        Y_hat = nn.predict(X_val, batch_size = 32)
        if self.score == 'accuracy':
            Y_hat = np.greater(Y_hat, 0.5)
            fitness = accuracy_score(Y_val, Y_hat)
        elif self.score == 'f1':
            Y_hat = np.greater(Y_hat, 0.5)
            fitness = f1_score(Y_val, Y_hat, 
                average = average)
        elif self.score == 'precision':
            Y_hat = np.greater(Y_hat, 0.5)
            fitness = precision_score(Y_val, Y_hat, 
                average = average)
        elif self.score == 'recall':
            Y_hat = np.greater(Y_hat, 0.5)
            fitness = recall_score(Y_val, Y_hat, 
                average = average)
        elif self.score == 'loss':
            fitness = np.divide(1, nn.evaluate(X_val, Y_val))
        else:
            # Custom scoring function
            fitness = self.score(Y_val, Y_hat)
        
        # Clear tensorflow session to avoid memory leak
        K.clear_session()
        
        if verbose:
            print("")
        
        return fitness


def __main__():
    pass
