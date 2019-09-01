import naturalselection as ns

def preprocessing(X):
    ''' Basic normalisation and scaling preprocessing. '''
    import numpy as np
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X = (X - X.min()) / (X.max() - X.min())
    X -= X.mean(axis = 0)
    return X

def mnist_train_val_sets():
    ''' Get normalised and scaled MNIST train- and val sets. '''
    from tensorflow.keras.utils import to_categorical
    import mnist
    X_train = preprocessing(mnist.train_images())
    Y_train = to_categorical(mnist.train_labels())
    X_val = preprocessing(mnist.test_images())
    Y_val = to_categorical(mnist.test_labels())
    return (X_train, Y_train, X_val, Y_val)

fnns = ns.FNNs(
    size = 50,
    train_val_sets = mnist_train_val_sets(),
    loss_fn = 'categorical_crossentropy',
    score = 'accuracy',
    output_activation = 'softmax',
    max_training_time = 60,
    multiprocessing = False
    )

history = fnns.evolve(generations = 20)
print("Best overall genome:", history.fittest)

history.plot(
    title = "Validation accuracy by generation",
    ylabel = "Validation accuracy",
    )

best_score = fnns.train_best()
print("Best score:", best_score)
