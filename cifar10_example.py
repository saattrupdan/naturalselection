import naturalselection as ns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

def cifar10_preprocessing(X):
    ''' Basic normalisation and scaling preprocessing. '''
    X = X.reshape((-1, 3072))
    X = (X - X.min()) / (X.max() - X.min())
    X -= X.mean(axis = 0)
    return X

(X_train, Y_train), (X_val, Y_val) = cifar10.load_data()
X_train = cifar10_preprocessing(X_train)
Y_train = to_categorical(Y_train)
X_val = cifar10_preprocessing(X_val)
Y_val = to_categorical(Y_val)

fnns = ns.FNNs(
    size = 50,
    train_val_sets = (X_train, Y_train, X_val, Y_val),
    loss_fn = 'categorical_crossentropy',
    score = 'accuracy',
    output_activation = 'softmax',
    max_training_time = 60,
    )

history = fnns.evolve(generations = 20)
print("Best overall genome:", history.fittest)

history.plot(
    title = "Validation accuracy by generation",
    ylabel = "Validation accuracy",
    )

best_score = fnns.train_best()
print("Best score:", best_score)
