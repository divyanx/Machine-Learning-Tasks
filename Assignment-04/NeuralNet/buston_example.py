from cv2 import line
import numpy as np

from network import Network

from fc_layer import Layer
from activation_layer import Activation
from activations import tanh, tanh_prime,softmax,softmax_prime,sigmoid,sigmoid_prime,relu,relu_prime,linear,linear_prime
from losses import mse, mse_prime, cross_entropy, cross_entropy_prime
from sklearn.datasets import load_boston


# training data
from sklearn.datasets import load_boston
# training data
x_train, y_train = load_boston(return_X_y=True)
# standarise x and y


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
net.add(Layer(13, 13))
net.add(Activation(linear, linear_prime))
# net.add(Layer(30, 30))
# net.add(Activation(tanh, tanh_prime))
net.add(Layer(13, 1))
net.add(Activation(linear, linear_prime))


# train
# net.use(mse, mse_prime)
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=1)

# test
out = net.predict(x_train)
print(out)