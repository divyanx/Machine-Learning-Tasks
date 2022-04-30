import numpy as np

from network import Network

from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime,softmax,softmax_prime,sigmoid,sigmoid_prime,relu,relu_prime,linear,linear_prime
from losses import mse, mse_prime, cross_entropy, cross_entropy_prime
from sklearn.datasets import load_boston


# training data
from sklearn.datasets import load_boston
# training data
x_train, y_train = load_boston(return_X_y=True)
# standarise x and y

# get 10 data out of it
x_train = x_train[:10]
y_train = y_train[:10]
print(x_train)
print(y_train)
# x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
# y_train = (y_train - y_train.mean(axis=0)) / y_train.std(axis=0)

x_train = np.expand_dims(x_train,axis=1)
y_train = y_train.reshape(y_train.shape[0],1)
y_train = np.expand_dims(y_train,axis=1)

print(x_train[0])
print(y_train[0])

# network
net = Network()
net.add(FCLayer(13, 30))
net.add(ActivationLayer(tanh, tanh_prime))
# net.add(FCLayer(30, 30))
# net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(30, 1))
net.add(ActivationLayer(tanh, tanh_prime))


# train
# net.use(mse, mse_prime)
net.use(cross_entropy, cross_entropy_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=1)

# test
out = net.predict(x_train)
print(out)