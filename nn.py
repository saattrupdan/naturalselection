import numpy as np
import os
import time
from functools import partial
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping

from core import Genus, Population

# Suppress deprecation warnings
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# Suppress tensorflow warnings and infos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TimeStopping(Callback):
    '''Stop training when enough time has passed.

    INPUT
        (int) seconds: maximum time before stopping.
        (int) verbose: verbosity mode.
    '''
    def __init__(self, seconds = None, verbose = 0):
        super(Callback, self).__init__()
        self.start_time = 0
        self.seconds = seconds
        self.verbose = verbose

    def on_train_begin(self, logs = {}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs = {}):
        if time.time() - self.start_time > self.seconds:
            self.model.stop_training = True
            if self.verbose:
                print(f'Stopping after {self.seconds} seconds.')

def fnn_fitness(fnn, train_val_sets, loss_fn, number_of_inputs,
    number_of_outputs = 1, output_activation = 'sigmoid',
    max_epochs = 1000000, patience = 5, min_change = 1e-4,
    max_training_time = None, verbose = False):

    X_train, Y_train, X_val, Y_val = train_val_sets

    inputs = Input(shape = (number_of_inputs,))
    x = Dropout(fnn.genome['input_dropout'])(inputs)
    for i in range(fnn.genome['number_of_hidden_layers']):
        x = Dense(fnn.genome['neurons_per_hidden_layer'],
            activation = fnn.genome['hidden_activation'],
            kernel_initializer = fnn.genome['initializer'])(x)
        x = Dropout(fnn.genome['hidden_dropout'])(x)
    outputs = Dense(number_of_outputs, activation = output_activation,
        kernel_initializer = fnn.genome['initializer'])(x)
    nn = Model(inputs = inputs, outputs = outputs)

    nn.compile(
        loss = loss_fn,
        optimizer = fnn.genome['optimizer']
        )

    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        patience = patience,
        min_delta = min_change,
        restore_best_weights = True,
        verbose = verbose
        )

    time_stopping = TimeStopping(
        seconds = max_training_time,
        verbose = verbose
        )

    H = nn.fit(
        X_train,
        Y_train,
        batch_size = fnn.genome['batch_size'],
        validation_data = (X_val, Y_val),
        epochs = max_epochs,
        callbacks = [early_stopping, time_stopping],
        verbose = verbose
        )
        
    return 1 / nn.evaluate(X_val, Y_val, verbose = verbose)

def optimize_fnn(
    train_val_sets,
    population_size = 10,
    generations = 5,
    keep = 0.20,
    mutate = 0.5,
    loss_fn = 'binary_crossentropy',
    output_activation = 'sigmoid',
    number_of_inputs = 'infer',
    number_of_outputs = 'infer',
    patience = 3,
    max_training_time = None,
    min_change = 0.1,
    max_epochs = 20,
    number_of_hidden_layers = range(1, 5),
    input_dropout = np.arange(0, 1, 0.1),
    hidden_dropout = np.arange(0, 1, 0.1),
    neurons_per_hidden_layer = [2 ** n for n in range(4, 12)],
    optimizer = ['adam', 'nadam', 'rmsprop'],
    hidden_activation = ['relu', 'elu', 'tanh'],
    batch_size = [2 ** n for n in range(4, 12)],
    initializer = ['lecun_uniform', 'lecun_normal', 'glorot_uniform',
                   'glorot_normal', 'he_uniform', 'he_normal'],
    multiprocessing = False,
    verbose = False):

    if number_of_inputs == 'infer':
        number_of_inputs = train_val_sets[0].shape[1]
    if number_of_outputs == 'infer':
        number_of_outputs = train_val_sets[1].shape[1]

    FNN = Genus({
        'number_of_hidden_layers'   : number_of_hidden_layers,
        'input_dropout'             : input_dropout,
        'hidden_dropout'            : hidden_dropout,
        'neurons_per_hidden_layer'  : neurons_per_hidden_layer,
        'optimizer'                 : optimizer,
        'hidden_activation'         : hidden_activation,
        'batch_size'                : batch_size,
        'initializer'               : initializer,
        })

    fitness_fn = partial(
        fnn_fitness,
        train_val_sets      = train_val_sets,
        loss_fn             = loss_fn,
        number_of_inputs    = number_of_inputs,
        number_of_outputs   = number_of_outputs,
        output_activation   = output_activation,
        max_epochs          = max_epochs,
        patience            = patience,
        max_training_time   = max_training_time,
        verbose             = verbose
        )
    
    fnns = Population(
        genus       = FNN,
        size        = population_size,
        fitness_fn  = fitness_fn
        )

    history = fnns.evolve(
        generations = generations,
        keep = keep,
        mutate = mutate,
        multiprocessing = multiprocessing,
        verbose = verbose
        )

    return history


if __name__ == '__main__':
    
    import mnist
    from tensorflow.keras.utils import to_categorical
    
    X_train = ((mnist.train_images() / 255) - 0.5).reshape((-1, 784))
    Y_train = to_categorical(mnist.train_labels())
    X_val = ((mnist.test_images() / 255) - 0.5).reshape((-1, 784))
    Y_val = to_categorical(mnist.test_labels())

    past = time.time()
    history = optimize_fnn(
        (X_train, Y_train, X_val, Y_val),
        loss_fn = 'categorical_crossentropy',
        number_of_outputs = 10,
        output_activation = 'softmax',
        population_size = 20,
        generations = 10,
        max_training_time = 60
        )
    duration = time.time() - past

    print(f"Evolution time: {duration}")
    print("Fittest genome across all generations:")
    print(history.fittest)

    history.plot()
