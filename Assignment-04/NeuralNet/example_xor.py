import numpy as np

from network import Network

from fc_layer import Layer
from activation_layer import Activation
from activations import tanh, tanh_prime,softmax,softmax_prime,sigmoid,sigmoid_prime,relu,relu_prime,linear,linear_prime
from losses import mse, mse_prime, cross_entropy, cross_entropy_prime

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(Layer(2, 3))
net.add(Activation(tanh, tanh_prime))
net.add(Layer(3, 3))
net.add(Activation(tanh, tanh_prime))
net.add(Layer(3, 1))
net.add(Activation(tanh, tanh_prime))


# train
# net.use(mse, mse_prime)
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)