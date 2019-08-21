import naturalselection as ns
from tensorflow.keras.utils import to_categorical
import mnist

X_train = ((mnist.train_images() / 255) - 0.5).reshape((-1, 784))
Y_train = to_categorical(mnist.train_labels())
X_val = ((mnist.test_images() / 255) - 0.5).reshape((-1, 784))
Y_val = to_categorical(mnist.test_labels())

fitness_fn = ns.get_nn_fitness_fn(
    kind = 'fnn',
    train_val_sets = (X_train, Y_train, X_val, Y_val),
    loss_fn = 'binary_crossentropy',
    score = 'accuracy',
    output_activation = 'softmax',
    max_training_time = 60
    )

fnns = ns.Population(
    genus = ns.FNN(),
    fitness_fn = fitness_fn,
    size = 50
    )

history = fnns.evolve(generations = 20)

print("Best overall genome is:")
print(history.fittest)

history.save_log(file_name = '/home/dn16382/pCloudDrive/mnist_log.txt')

history.plot(
    title = "Average 1 / (1 - val_accuracy) by generation",
    file_name = '/home/dn16382/pCloudDrive/mnist_plot.png',
    show_plot = False
  )
