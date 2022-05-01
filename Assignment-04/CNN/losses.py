import numpy as np


def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


def binary_cross_entropy(y_true, y_pred):
    # check if log is not going to be valid

    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


def cross_entropy(y, X):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
        Note that y is not one-hot encoded vector. 
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    p = softmax(X)
    # We use multidimensional array indexing to extract
    # softmax probability of the correct label for each sample.
    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m
    return loss


def cross_entropy_prime(X, y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
        Note that y is not one-hot encoded vector. 
        It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m), y] -= 1
    grad = grad/m
    return grad
