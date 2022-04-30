import numpy as np

# loss function and its derivative
def mse(y_true, y_pred):
    print("-------------------")
    print(np.mean(np.power(y_true-y_pred, 2)))
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    print("-------------------")
    print(2*(y_pred-y_true)/y_true.size)
    return 2*(y_pred-y_true)/y_true.size

# cross entropy function and its derivative
def cross_entropy(y_true, y_pred):
    return -np.mean(y_true*np.log(y_pred))

def cross_entropy_prime(y_true, y_pred):
    return -y_true/y_pred