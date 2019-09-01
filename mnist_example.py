import naturalselection as ns
from tensorflow.keras.utils import to_categorical
import mnist

X_train = ((mnist.train_images() / 255) - 0.5).reshape((-1, 784))
Y_train = to_categorical(mnist.train_labels())
X_val = ((mnist.test_images() / 255) - 0.5).reshape((-1, 784))
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
    show_plot = False,
    file_name = "mnist_plot.png"
    )

best_score = fnns.train_best()
print("Best score:", best_score)
