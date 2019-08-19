import numpy as np
import os
import time
from functools import partial, reduce
from itertools import permutations, chain

# Neural network packages
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score

# Local packages
from core import Genus, Population

# Suppress deprecation warnings
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# Suppress tensorflow warnings and infos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class FNN(Genus):
    ''' Feed-forward neural network genus.

    INPUT:
        (iterable) number_of_hidden_layers: numbers of hidden layers
        (iterable) dropout: values for input dropout
        (iterable) neurons_per_hidden_layer = neurons in hidden layers
        (iterable) optimizer: keras optimizers
        (iterable) hidden_activation: keras activation functions
        (iterable) batch_size: batch sizes
        (iterable) initializer: keras initializers
        '''
    def __init__(self,
        number_of_hidden_layers = np.arange(1, 4),
        dropout = np.arange(0, 0.6, 0.1),
        neurons_per_hidden_layer = np.array([2 ** n for n in range(4, 11)]),
        optimizer = np.array(['adam', 'nadam']),
        hidden_activation = np.array(['relu', 'elu']),
        batch_size = np.array([2 ** n for n in range(4, 12)]),
        initializer = np.array(['lecun_uniform', 'lecun_normal',
                                'glorot_uniform', 'glorot_normal',
                                'he_uniform', 'he_normal'])):

        self.optimizer = optimizer
        self.hidden_activation = hidden_activation
        self.batch_size = batch_size
        self.initializer = initializer
        self.input_dropout = dropout

        self.hidden_dropout = np.asarray(list(reduce(lambda x, y: chain(x, y),
            [permutations(dropout, int(n)) for n in number_of_hidden_layers])))

        self.layers = np.asarray(list(reduce(lambda x, y: chain(x, y),
            [permutations(neurons_per_hidden_layer, int(n))
            for n in number_of_hidden_layers])))


class TimeStopping(Callback):
    ''' Callback to stop training when enough time has passed.

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

def train_fnn(fnn, train_val_sets, loss_fn = 'binary_crossentropy',
    number_of_inputs = 'infer', number_of_outputs = 'infer',
    output_activation = 'sigmoid', score = 'accuracy',
    max_epochs = 1000000, patience = 5, min_change = 1e-4,
    max_training_time = None, verbose = False):
    ''' Train a feed-forward neural network and output the score.
    
    INPUT
        (FNN) fnn: a feed-forward neural network genus
        (tuple) train_val_sets: a quadruple of the form
                (X_train, Y_train, X_val, Y_val)
        (string) loss_fn: keras loss function
        (int or string) number_of_inputs: number of input features,
                        will infer from X_train if it's set to 'infer'
        (int or string) number_of_outputs: number of output features,
                        will infer from Y_train if it's set to 'infer'
        (string) output_activation: keras activation to be used on output
        (string) the scoring used. Can be 'accuracy', 'f1', 'precision' or
                 'recall', where the micro-average will be taken if there
                 are multiple outputs
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

    X_train, Y_train, X_val, Y_val = train_val_sets

    if number_of_inputs == 'infer':
        number_of_inputs = X_train.shape[1]
    if number_of_outputs == 'infer':
        number_of_outputs = Y_train.shape[1]

    inputs = Input(shape = (number_of_inputs,))
    x = Dropout(fnn.input_dropout)(inputs)
    for (layer, dropout) in zip(fnn.layers, fnn.hidden_dropout):
        x = Dense(layer, activation = fnn.hidden_activation,
            kernel_initializer = fnn.initializer)(x)
        x = Dropout(dropout)(x)
    outputs = Dense(number_of_outputs, activation = output_activation,
        kernel_initializer = fnn.initializer)(x)
    nn = Model(inputs = inputs, outputs = outputs)

    nn.compile(
        loss = loss_fn,
        optimizer = fnn.optimizer
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

    nn.fit(
        X_train,
        Y_train,
        batch_size = fnn.batch_size,
        validation_data = (X_val, Y_val),
        epochs = max_epochs,
        callbacks = [early_stopping, time_stopping],
        verbose = verbose
        )

    if Y_val.shape[1] > 1:
        average = 'micro'
    else:
        average = None

    Y_hat = np.greater(np.asarray(nn.predict(X_val, batch_size = 32)), 0.5)
    if score == 'accuracy':
        fitness = accuracy_score(Y_val, Y_hat)
    if score == 'f1':
        fitness = f1_score(Y_val, Y_hat, average = average)
    if score == 'precision':
        fitness = precision_score(Y_val, Y_hat, average = average)
    if score == 'recall':
        fitness = recall_score(Y_val, Y_hat, average = average)
    
    # Clear tensorflow session to avoid memory leak
    K.clear_session()
        
    return fitness

def get_fitness_fn(train_val_sets, loss_fn, number_of_inputs = 'infer',
    number_of_outputs = 'infer', output_activation = 'sigmoid',
    score = 'accuracy', max_epochs = 1000000, patience = 5,
    min_change = 1e-4, max_training_time = None, verbose = False,
    kind = 'fnn'):
    ''' Return a neural network fitness function of the specified kind.
    
    INPUT
        (tuple) train_val_sets: a quadruple of the form
                (X_train, Y_train, X_val, Y_val)
        (string) loss_fn: keras loss function
        (int or string) number_of_inputs: number of input features,
                        will infer from X_train if it's set to 'infer'
        (int or string) number_of_outputs: number of output features,
                        will infer from Y_train if it's set to 'infer'
        (string) output_activation: keras activation to be used on output
        (string) the scoring used. Can be 'accuracy', 'f1', 'precision' or
                 'recall', where the micro-average will be taken if there
                 are multiple outputs
        (int) max_epochs: maximum number of epochs to train for
        (int) patience: number of epochs with no progress above min_change
        (float) min_change: everything below this number won't count as a
                change in the score
        (int) max_training_time: maximum number of seconds to train for,
              also training the final epoch after the time has run out
        (int) verbose: verbosity mode
        (string) kind: type of neural network, can only be 'fnn' at the moment

    OUTPUT
        (function) fitness function
    '''

    if kind == 'fnn':
        fitness_fn = partial(
            train_fnn,
            train_val_sets      = train_val_sets,
            loss_fn             = loss_fn,
            number_of_inputs    = number_of_inputs,
            number_of_outputs   = number_of_outputs,
            output_activation   = output_activation,
            score               = score,
            max_epochs          = max_epochs,
            patience            = patience,
            min_change          = min_change,
            max_training_time   = max_training_time,
            verbose             = verbose
            )
    
    return fitness_fn


if __name__ == '__main__':
    
    import mnist
    from tensorflow.keras.utils import to_categorical
    
    X_train = ((mnist.train_images() / 255) - 0.5).reshape((-1, 784))
    Y_train = to_categorical(mnist.train_labels())
    X_val = ((mnist.test_images() / 255) - 0.5).reshape((-1, 784))
    Y_val = to_categorical(mnist.test_labels())

    fitness_fn = get_fitness_fn(
        kind                = 'fnn',
        train_val_sets      = (X_train, Y_train, X_val, Y_val),
        loss_fn             = 'binary_crossentropy',
        score               = 'accuracy',
        output_activation   = 'softmax',
        max_training_time   = 120
        )

    fnns = Population(
        genus       = FNN(),
        fitness_fn  = fitness_fn,
        size        = 50
        )

    past = time.time()
    history = fnns.evolve(generations = 50)
    duration = time.time() - past

    print(f"Evolution time: {duration}")
    print("Fittest genome across all generations:")
    print(history.fittest)

    history.plot(
        title = "Average accuracy by generation",
        ylabel = "Average accuracy",
        save_to = 'mnist_plot.png'
        )
