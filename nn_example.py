import naturalselection as ns

def preprocessing(X):
    ''' Basic normalisation and scaling preprocessing. '''
    import numpy as np
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X = X.astype('float32')
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

def cifar10_train_val_sets():
    ''' Get normalised and scaled CIFAR-10 train- and val sets. '''
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.datasets import cifar10
    (X_train, Y_train), (X_val, Y_val) = cifar10.load_data()
    X_train = preprocessing(X_train)
    Y_train = to_categorical(Y_train)
    X_val = preprocessing(X_val)
    Y_val = to_categorical(Y_val)
    return (X_train, Y_train, X_val, Y_val)


print("\n~~~ Now evolving MNIST ~~~")

nns = ns.NNs(
    size = 50,
    train_val_sets = mnist_train_val_sets(),
    loss_fn = 'categorical_crossentropy',
    score = 'accuracy',
    output_activation = 'softmax',
    max_training_time = 200,
    max_epochs = 3,
    )

history = nns.evolve(generations = 10)
print("Best overall genome:", history.fittest)

history.plot(
    title = "Validation accuracy by generation",
    ylabel = "Validation accuracy",
    show_plot = False,
    file_name = "mnist_plot.png"
    )

best_score = nns.train_best()
print("Best score:", best_score)


#print("\n~~~ Now evolving CIFAR-10 ~~~")
#
#nns = ns.NNs(
#    size = 50,
#    train_val_sets = cifar10_train_val_sets(),
#    loss_fn = 'categorical_crossentropy',
#    score = 'accuracy',
#    output_activation = 'softmax',
#    max_training_time = 300,
#    max_epochs = 3,
#    )
#
#history = nns.evolve(generations = 10)
#print("Best overall genome:", history.fittest)
#
#history.plot(
#    title = "Validation accuracy by generation",
#    ylabel = "Validation accuracy",
#    show_plot = False,
#    file_name = "cifar10_plot.png"
#    )
#
#best_score = nns.train_best()
#print("Best score:", best_score)


