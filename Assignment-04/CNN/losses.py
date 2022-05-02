import numpy as np


def cross_entropy(y_true, y_pred):
    """
    Computes the cross entropy between y_true and y_pred.
    y_true is a one-hot encoded vector of shape (batch_size, num_classes).
    y_pred is a vector of shape (batch_size, num_classes).

    """
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def cross_entropy_prime(y_true, y_pred):
    """
    Computes the derivative of cross entropy with respect to y_pred.
    y_true is a one-hot encoded vector of shape (batch_size, num_classes).
    y_pred is a vector of shape (batch_size, num_classes).

    """
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
