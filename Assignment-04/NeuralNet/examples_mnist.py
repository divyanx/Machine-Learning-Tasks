
import numpy as np

from network import Network
from fullyconnlayer import FullyConnLayer
from activation_layer import ActivationFuncLayer
from activations import tanh, tanh_prime,softmax,softmax_prime,sigmoid,sigmoid_prime,relu,relu_prime,linear,linear_prime
from losses import mse, mse_prime, cross_entropy, cross_entropy_prime
from sklearn.datasets import load_boston


from keras.datasets import mnist
from keras.utils import np_utils

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)
# insert noice to categories
y_train[y_train == 0] = np.random.randint(0,9,(y_train[y_train == 0].shape[0],))
# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

# Network
net = Network()
net.insert(FullyConnLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.insert(ActivationFuncLayer(tanh, tanh_prime))
net.insert(FullyConnLayer(100, 10))  
              # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.insert(ActivationFuncLayer(linear, linear_prime))


# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
net.use(cross_entropy, cross_entropy_prime)
net.fit(x_train[0:1000], y_train[0:100], epochs=1000, learning_rate=0.001)

# test on 3 samples
out = net.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])