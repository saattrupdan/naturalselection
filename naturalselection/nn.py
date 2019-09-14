import numpy as np
import os
from functools import partial 
from itertools import product
import logging
from multiprocessing import cpu_count

import naturalselection as ns

class NN(ns.Genus):
    ''' Feedforward fully connected neural network genus.

    INPUT:
        (int) max_nm_hidden_layers
        (bool) uniform_layers: whether all hidden layers should
               have the same amount of neurons and dropout
        (iterable) : values for dropout before hidden layers
        (iterable) neurons_per_hidden_layer = neurons in hidden layers
        (iterable) hidden_activation: keras activation functions
        (iterable) batch_size: batch sizes
        '''
    def __init__(self, max_nm_hidden_layers, uniform_layers, 
        dropout, neurons, hidden_activation, batch_size,
        learning_rate, fst_moment, snd_moment, decay,
        nesterov):

        self.hidden_activation = np.unique(np.asarray(hidden_activation))
        self.batch_size = np.unique(np.asarray(batch_size))
        self.input_dropout = np.around(np.unique(np.append(dropout, 0)), 2)

        learning_rate = np.unique(np.append(learning_rate, 1))
        decay = np.unique(np.append(decay, 0))
        self.lr_and_decay = np.array(list(product(learning_rate, decay)))

        self.snd_moment = np.unique(np.append(snd_moment, 0))
        fst_moment = np.unique(np.append(fst_moment, 0))
        self.fst_moment_and_nesterov = np.array([
            (m, n) for (m, n) in product(fst_moment, nesterov)
            if not (m == 0 and n)
            ])

        neurons = np.unique(np.append(neurons, 0))
        dropouts = np.around(np.unique(np.append(dropout, 0)), 2)
        if uniform_layers:
            self.neurons = neurons
            self.dropout = dropouts
            self.nm_hidden_layers = \
                np.arange(1, max_nm_hidden_layers + 1)
        else:
            layer_info = {}
            for i in range(max_nm_hidden_layers):
                layer_info[f'neurons_and_dropout{i}'] = np.array([
                    (neuron, dropout)
                    for (neuron, dropout) in product(neurons, dropouts)
                    if not (neuron == 0 and dropout != 0)
                    ])

            self.__dict__.update(layer_info)


class NNs(ns.Population):
    def __init__(self, 
        train_val_sets,
        size = 30, 
        initial_genome = {},
        breeding_rate = 0.8,
        mutation_rate = 0.2,
        mutation_factor = 'default',
        elitism_rate = 0.05,
        multiprocessing = True,
        workers = cpu_count(),
        progress_bars = 3,
        loss_fn = 'binary_crossentropy',
        nm_features = 'infer', 
        nm_labels = 'infer',
        score = 'accuracy', 
        output_activation = 'sigmoid',
        max_epochs = 1000000, 
        patience = 3, 
        min_change = 1e-4,
        max_training_time = None, 
        max_epoch_time = None, 
        max_nm_hidden_layers = 5,
        batch_norm = True,
        uniform_layers = False,
        dropout = np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        neurons = np.array([2 ** n for n in range(5, 11)]),
        learning_rate = np.array([5, 2, 1, 5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 
                                  5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]),
        fst_moment = np.array([0.5, 0.8, 0.9, 0.95, 0.98, 0.99,
                             0.995, 0.998, 0.999]),
        snd_moment = np.array([0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 
                                  0.995, 0.998, 0.999]),
        decay = np.array([5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2,
                          5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4,
                          5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]),
        nesterov = np.array([0, 1]),
        hidden_activation = np.array(['relu', 'elu']),
        batch_size = np.array([32]),
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
        self.progress_bars        = progress_bars
        self.loss_fn              = loss_fn
        self.nm_features          = nm_features
        self.nm_labels            = nm_labels
        self.score                = score
        self.output_activation    = output_activation
        self.max_epochs           = max_epochs 
        self.patience             = patience
        self.min_change           = min_change
        self.max_training_time    = max_training_time
        self.max_epoch_time       = max_epoch_time
        self.max_nm_hidden_layers = max_nm_hidden_layers
        self.uniform_layers       = uniform_layers
        self.dropout              = dropout
        self.neurons              = neurons
        self.hidden_activation    = hidden_activation
        self.batch_size           = batch_size
        self.verbose              = verbose
        self.learning_rate        = learning_rate
        self.fst_moment           = fst_moment
        self.snd_moment           = snd_moment
        self.decay                = decay
        self.nesterov             = nesterov
        self.batch_norm           = batch_norm

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
        
        self.genus = NN(
            max_nm_hidden_layers = self.max_nm_hidden_layers,
            uniform_layers       = self.uniform_layers,
            dropout              = self.dropout,
            neurons              = self.neurons,
            hidden_activation    = self.hidden_activation,
            batch_size           = self.batch_size,
            learning_rate        = self.learning_rate,
            fst_moment           = self.fst_moment,
            snd_moment           = self.snd_moment,
            decay                = self.decay,
            nesterov             = self.nesterov
            )

        self.fitness_fn = partial(
            self.train_nn,
            max_epochs          = self.max_epochs,
            patience            = self.patience,
            min_change          = self.min_change,
            max_training_time   = self.max_training_time,
            max_epoch_time      = self.max_epoch_time,
            file_name           = None,
            )

        # If user has supplied an initial genome then construct a population
        # which is very similar to that, and if not then start with a shallow
        # network
        if not 'lr_and_decay' in initial_genome.keys():
            # Start with a large learning rate, following the advice in
            # Bengio's "Practical recommendations for gradient-based training
            # of deep architectures". I start with second highest to allow 
            # mutation options in both directions
            penultimate_lr = np.partition(learning_rate, -2)[-2]
            initial_genome = {
                'lr_and_decay': np.array((penultimate_lr, 0.))
                }
        for i in range(self.max_nm_hidden_layers):
            if not f'neurons_and_dropout{i}' in initial_genome.keys():
                initial_genome[f'neurons_and_dropout{i}'] = np.array((0, 0.))

        # Create a population of organisms all with the initial genome
        self.population = np.array([
            ns.Organism(self.genus, **initial_genome)
            for _ in range(size)
            ])

        # Mutate 50% of the genes in 80% of the population
        rnd = np.random.random(self.population.shape)
        for (i, org) in enumerate(self.population):
            if rnd[i] < 0.80:
                org.mutate(mutation_factor = .50)

        # We do not have access to fitness values yet, so choose the 'fittest
        # organism' to just be a random one
        self.fittest = np.random.choice(self.population)

    def train_best(self, max_epochs = 1000000, min_change = 1e-4,
        patience = 10, max_training_time = None, max_epoch_time = None,
        file_name = None):

        best_nn = self.fittest        
        fitness = self.train_nn(
            nn                  = best_nn,
            max_epochs          = max_epochs,
            patience            = patience,
            min_change          = min_change,
            max_training_time   = max_training_time,
            max_epoch_time      = max_epoch_time,
            file_name           = file_name
            )
        return fitness

    def train_nn(self, nn, max_epochs = 1000000, patience = 3,
        min_change = 1e-4, max_training_time = None, max_epoch_time = None,
        file_name = None, worker_idx = None):
        ''' Train a feedforward neural network and output the score.
        
        INPUT
            (NN) nn: a neural network genus
            (int) max_epochs = 1000000: maximum number of epochs to train for
            (int) patience = 3: number of epochs allowed with no progress
                  above min_change
            (float) min_change = 1e-4: everything below this number will
                    not count as a change in the score
            (int) max_training_time = None: maximum number of seconds to
                  train for
            (int) max_epoch_time = None: maximum number of seconds to
                  spend training for a single epoch
            (int) worker_idx = None: what worker is currently training this
                  network, with enumeration starting from 1

        OUTPUT
            (float) the score of the neural network
        '''

        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Dropout
        from tensorflow.keras.optimizers import SGD, Adam, Nadam, RMSprop
        from tensorflow.keras.initializers import VarianceScaling
        from tensorflow.keras import backend as K
        from tensorflow.python.util import deprecation
        from tensorflow import set_random_seed
        from sklearn.metrics import f1_score, precision_score, recall_score

        # Custom callbacks
        from naturalselection.callbacks import TQDMCallback, EarlierStopping

        # Used when building network
        from itertools import count

        # Suppress tensorflow warnings
        deprecation._PRINT_DEPRECATION_WARNINGS = False
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # Set random seeds to enable better comparison of scores
        np.random.seed(0)
        set_random_seed(0)

        X_train, Y_train, X_val, Y_val = self.train_val_sets

        if self.nm_features == 'infer':
            self.nm_features = X_val.shape[1]
        if self.nm_labels == 'infer':
            self.nm_labels = Y_val.shape[1]

        if nn.hidden_activation == 'relu':
            # This initialisation is specific to ReLU to ensure that the
            # variances of the weights and gradients neither vanishes
            # nor explodes.
            # Source: He et al, "Delving Deep into Rectifiers: Surpassing
            # Human-Level Performance on ImageNet Classification"
            initializer = 'he_uniform'
        elif nn.hidden_activation == 'elu':
            # This initialisation is specific to ELU, where the the 1.55
            # value is derived as in the He et al paper above, but further-
            # more assumes that the input values follow a standard
            # normal distribution, so this is only an approximation.
            # Source: https://stats.stackexchange.com/a/320443/255420
            initializer = VarianceScaling(scale = 1.55,
                distribution = 'uniform', mode = 'fan_in')
        else:
            # This initialisation is good for activations that are symmetric
            # around zero, like sigmoid, softmax and tanh.
            # Source: Glorot and Bengio, "Understanding the difficulty of
            # training deep feedforward neural networks"
            initializer = 'glorot_uniform'

        inputs = Input(shape = (self.nm_features,))
        x = inputs
    
        if nn.input_dropout:
            x = Dropout(nn.input_dropout)(x)

        if self.uniform_layers:
            for _ in range(nn.nm_hidden_layers):
                x = Dense(nn.neurons, activation = nn.hidden_activation,
                    kernel_initializer = initializer)(x)
                if nn.dropout:
                    x = Dropout(nn.dropout)(x)
        else:
            for i in range(self.max_nm_hidden_layers):
                (neurons, dropout) = nn.__dict__\
                    ["neurons_and_dropout{}".format(i)]
                if neurons:
                    x = Dense(neurons, activation = nn.hidden_activation,
                        kernel_initializer = initializer)(x)
                if dropout:
                    x = Dropout(dropout)(x)

        outputs = Dense(self.nm_labels,
            activation = self.output_activation,
            kernel_initializer = 'glorot_uniform')(x)

        model = Model(inputs = inputs, outputs = outputs)

        learning_rate, decay = nn.lr_and_decay
        fst_moment, nesterov = nn.fst_moment_and_nesterov
        if fst_moment and nn.snd_moment:
            if nesterov:
                optimizer = Nadam(lr = learning_rate, schedule_decay = decay,
                    beta_1 = fst_moment, beta_2 = nn.snd_moment)
            else:
                optimizer = Adam(lr = learning_rate, decay = decay,
                    beta_1 = fst_moment, beta_2 = nn.snd_moment)
        elif nn.snd_moment:
            optimizer = RMSprop(lr = learning_rate, decay = decay,
                rho = nn.snd_moment)
        else:
            optimizer = SGD(lr = learning_rate, decay = decay,
                momentum = fst_moment, nesterov = nesterov)

        if self.score == 'accuracy':
            metrics = ['accuracy']        
        elif self.score == 'categorical accuracy':
            metrics = ['categorical accuracy']
        else:
            metrics = []

        model.compile(
            loss = self.loss_fn,
            optimizer = optimizer,
            metrics = metrics
            )

        if self.score == 'accuracy' or self.score == 'categorical_accuracy':
            monitor = 'val_acc'
        else:
            monitor = 'val_loss'

        earlier_stopping = EarlierStopping(
            monitor = monitor,
            patience = patience,
            min_delta = min_change,
            restore_best_weights = True,
            max_training_time = max_training_time,
            max_epoch_time = max_epoch_time
            )

        callbacks = [earlier_stopping]
        if self.progress_bars >= 3:
            if worker_idx:
                desc = f'Worker {(worker_idx - 1) % self.workers}, '
                tqdm_callback = TQDMCallback(
                    show_outer = False, 
                    inner_position = ((worker_idx - 1) % self.workers) + 2,
                    leave_inner = False,
                    inner_description_update = desc + 'Epoch {epoch}',
                    inner_description_initial = desc + 'Epoch {epoch}'
                    )
            else:
                tqdm_callback = TQDMCallback(
                    show_outer = False, 
                    inner_position = 0
                    )
            callbacks.append(tqdm_callback)

        model.fit(
            X_train,
            Y_train,
            batch_size = nn.batch_size,
            validation_data = (X_val, Y_val),
            epochs = max_epochs,
            callbacks = callbacks,
            verbose = 0
            )

        if file_name:
            model.save("{}.h5".format(file_name))

        if self.nm_labels > 1:
            average = 'micro'
        else:
            average = None

        Y_hat = model.predict(X_val, batch_size = 128)
        if self.score == 'accuracy' or self.score == 'categorical accuracy':
            fitness = model.evaluate(X_val, Y_val, verbose = 0)[1]
        elif self.score == 'f1':
            Y_hat = np.greater(Y_hat, 0.5)
            fitness = f1_score(Y_val, Y_hat, average = average)
        elif self.score == 'precision':
            Y_hat = np.greater(Y_hat, 0.5)
            fitness = precision_score(Y_val, Y_hat, average = average)
        elif self.score == 'recall':
            Y_hat = np.greater(Y_hat, 0.5)
            fitness = recall_score(Y_val, Y_hat, average = average)
        elif self.score == 'loss':
            fitness = np.divide(1, model.evaluate(X_val, Y_val, verbose = 0))
        else:
            # Custom scoring function
            fitness = self.score(Y_val, Y_hat)

        # Clear tensorflow session to avoid memory leak
        K.clear_session()

        return fitness


def __main__():
    pass
