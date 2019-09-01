import naturalselection as ns
from tensorflow.keras.utils import to_categorical
import mnist

def mnist_preprocessing(X):
    ''' Basic normalisation and scaling preprocessing. '''
    X = X.reshape((-1, 784))
    X = (X - X.min()) / (X.max() - X.min())
    X -= X.mean(axis = 0)
    return X

X_train = mnist_preprocessing(mnist.train_images())
Y_train = to_categorical(mnist.train_labels())
X_val = mnist_preprocessing(mnist.test_images())
Y_val = to_categorical(mnist.test_labels())

fnns = ns.FNNs(
    size = 50,
    train_val_sets = (X_train, Y_train, X_val, Y_val),
    loss_fn = 'categorical_crossentropy',
    score = 'accuracy',
    output_activation = 'softmax',
    max_training_time = 60,
    multiprocessing = False,
    initial_genome = {
        'optimizer' : 'adam',
        'hidden_activation' : 'relu',
        'batch_size' : 1024,
        'initializer' : 'he_normal',
        'input_dropout' : 0.0,
        'neurons0' : 512,
        'neurons1' : 512,
        'neurons2' : 512,
        'neurons3' : 512,
        'neurons4' : 512,
        'dropout0' : 0.0,
        'dropout1' : 0.0,
        'dropout2' : 0.0,
        'dropout3' : 0.0,
        'dropout4' : 0.0
        }
    )

history = fnns.evolve(generations = 20)
print("Best overall genome:", history.fittest)

history.plot(
    title = "Validation accuracy by generation",
    ylabel = "Validation accuracy",
    )

best_score = fnns.train_best()
print("Best score:", best_score)
