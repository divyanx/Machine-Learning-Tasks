import numpy as np


def cross_entropy(true, pred):
    """
    Computes the cross entropy between true and pred.
    true is a one-hot encoded vector of shape (batch_size, num_classes).
    pred is a vector of shape (batch_size, num_classes).

    """
    return np.mean(-true * np.log(pred) - (1 - true) * np.log(1 - pred))


def cross_entropy_derivative(true, pred):
    """
    Computes the derivative of cross entropy with respect to pred.
    true is a one-hot encoded vector of shape (batch_size, num_classes).
    pred is a vector of shape (batch_size, num_classes).

    """
    return ((1 - true) / (1 - pred) - true / pred) / np.size(true)
