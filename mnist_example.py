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
    )

history = fnns.evolve(generations = 20)
print("Best overall genome:", history.fittest)

history.plot(
    title = "Validation accuracy by generation",
    ylabel = "Validation accuracy",
    )

best_score = fnns.train_best()
print("Best score:", best_score)
