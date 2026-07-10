import numpy as np

def flatten_images(images):
    return images.reshape(-1, 784)
    

def normalise_images(images):
    return images.astype(np.float32) / 255.0


def preprocess_data(X_train_raw, 
X_test_raw, 
y_train_raw, 
y_test_raw):
    X_train = flatten_images(X_train_raw)
    X_test = flatten_images(X_test_raw)

    X_train = normalise_images(X_train)
    X_test = normalise_images(X_test)

    y_train = y_train_raw
    y_test = y_test_raw

    return X_train, X_test, y_train, y_test
    