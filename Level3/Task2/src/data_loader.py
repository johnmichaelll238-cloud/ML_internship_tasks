from tensorflow import keras


def load_mnist_data():
    return keras.datasets.mnist.load_data()
    