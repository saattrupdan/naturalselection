import naturalselection as ns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10

(X_train, Y_train), (X_val, Y_val) = cifar10.load_data()

X_train = ((X_train / 255) - 0.5).reshape((-1, 3072))
Y_train = to_categorical(Y_train)
X_val = ((X_val / 255) - 0.5).reshape((-1, 3072))
Y_val = to_categorical(Y_val)

fnns = ns.FNNs(
    size = 20,
    train_val_sets = (X_train, Y_train, X_val, Y_val),
    loss_fn = 'binary_crossentropy',
    score = 'accuracy',
    output_activation = 'softmax',
    patience = 5,
    min_change = 0,
    neurons = [64, 128, 256, 512, 768, 1024],
    uniform_layers = True,
    max_number_of_hidden_layers = 4,
    hidden_activation = ['relu', 'elu', 'tanh', 'sigmoid'],
    optimizer = ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta',
                  'adamax', 'nadam'],
    dropout = [0],
    batch_size = [64],
    initializer = ['glorot_uniform']
    )

history = fnns.evolve(
    generations = 10, 
    multiprocessing = False,
    verbose = 2
    )
print("Best overall genome:", history.fittest)

history.plot(
    title = "Validation accuracy by generation",
    ylabel = "Validation accuracy",
    show_plot = False,
    file_name = "cifar10_plot.png"
    )

best_score = fnns.train_best()
print("Best score:", best_score)
