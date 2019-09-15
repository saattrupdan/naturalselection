import naturalselection as ns

def image_preprocessing(X, normalisation = 'normal'):
    ''' Basic normalisation and scaling preprocessing. '''
    import numpy as np
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X = X.astype('float32')
    if normalisation == 'normal':
        X = (X - X.mean(axis = 0)) / X.std(axis = 0)
    elif normalisation == 'minmax':
        X = (X - X.min()) / (X.max() - X.min())
        X -= X.mean(axis = 0)
    return X

def train_val_sets(kind = 'mnist'):
    ''' Get normalised and scaled train- and val sets. '''

    from tensorflow.keras.utils import to_categorical
    if kind == 'mnist':
        import tensorflow.keras.datasets.mnist as data
    elif kind == 'fashion_mnist':
        import tensorflow.keras.datasets.fashion_mnist as data
    elif kind == 'cifar10':
        import tensorflow.keras.datasets.cifar10 as data
    elif kind == 'cifar100':
        import tensorflow.keras.datasets.cifar100 as data
    else:
        raise NameError(f'Dataset not recognised: {kind}')

    (X_train, Y_train), (X_val, Y_val) = data.load_data()
    X_train = image_preprocessing(X_train)
    Y_train = to_categorical(Y_train)
    X_val = image_preprocessing(X_val)
    Y_val = to_categorical(Y_val)
    return (X_train, Y_train, X_val, Y_val)

def evolve_nn(kind = 'mnist', pop_size = 30, gens = 30, max_epochs = 20,
    max_epoch_time = 90, verbose = 0):

    print(f"\n~~~ Now evolving {kind} ~~~")

    nns = ns.NNs(
        train_val_sets = train_val_sets(kind),
        loss_fn = 'categorical_crossentropy',
        score = 'accuracy',
        output_activation = 'softmax',
        size = pop_size,
        max_epochs = max_epochs,
        max_epoch_time = max_epoch_time,
        verbose = verbose,
        )

    history = nns.evolve(generations = gens)
    print("Best overall genome:", history.fittest)

    history.plot(
        title = f"Evolution of {kind}",
        ylabel = "Validation accuracy",
        show_plot = False,
        file_name = f'{kind}_plot.png'
        )

    best_score = nns.train_best()
    print("Best score:", best_score)


if __name__ == '__main__':
    from sys import argv

    pop_size = 30
    max_epochs = 10
    max_epoch_time = 90
    gens = 20
    verbose = 1

    if len(argv) > 1:
        for arg in argv[1:]:
            evolve_nn(arg, pop_size = pop_size, gens = gens, 
                max_epochs = max_epochs, max_epoch_time = max_epoch_time,
                verbose = verbose)
    else:
        evolve_nn(pop_size = pop_size, gens = gens, max_epochs = max_epochs,
            max_epoch_time = max_epoch_time, verbose = verbose)
