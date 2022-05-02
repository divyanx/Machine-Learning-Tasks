import numpy as np

from network import Network
from fullyconnlayer import FullyConnLayer
from activation_layer import ActivationFuncLayerFuncLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
obj = Network()
obj.insert(FullyConnLayer(2, 3))
obj.insert(ActivationFuncLayerFuncLayer(tanh, tanh_prime))
obj.insert(FullyConnLayer(3, 1))
obj.insert(ActivationFuncLayerFuncLayer(tanh, tanh_prime))

# train
obj.employ(mse, mse_prime)
obj.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = obj.predict(x_train)
print(out)