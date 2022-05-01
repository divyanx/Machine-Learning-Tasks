import numpy as np

# loss function and its derivative
def mse(y_true, y_pred):
    # print("-------------------")
    # print(y_true, y_pred)
    # print(np.mean(np.power(y_true-y_pred, 2)))
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    # print("-------------------")
    # print(y_true, y_pred)
    # print(2*(y_pred-y_true)/y_true.size)
    return 2*(y_pred-y_true)/y_true.size

# cross entropy function and its derivative
def cross_entropy(y_true, y_pred):
    # remove zeros and add some noise
    y_pred[y_pred < 1e-8] = 1e-8
    # print(y_pred)
    # print(y_pred)
    return -np.mean(y_true*np.log(y_pred))

def cross_entropy_prime(y_true, y_pred):
    y_pred[y_pred < 1e-8] = 1e-8
    lossGrad = -y_true/(y_pred*y_true.shape[0])
        # print(lossGrad)
    return lossGrad