#     Create a Neural Network class. This class takes in list of Layer objects. It
# also takes a loss object. You can also allow it to take a seed as input. The
# seed is used for reproduciblity across runs. Each layer is characterized by its
# activation function and count of output neuron. Examples of activation include
# linear (W x + b), sigmoid, tanh etc.
import numpy as np
import random
import math
class NNet:
    def __init__(self, nlayers, layerActivation = None, loss = None, seed=None):
        self.nlayers = nlayers # array of number of neurons in each layer excluding input layer
        self.loss = loss
        self.delLayers = []
        self.delLoss = None
        self.seed = seed # for reproducibility
        if layerActivation is None:
            self.layerActivation = [NNet.sigmoid] * (len(self.nlayers) - 1) + [NNet.linear]
        else:
            self.layerActivation = layerActivation
        if loss is None:
            self.loss = NNet.mse
        else :
            self.loss = loss
        
        # set delLoss Array based on activation of layers
        self.delLayers.append(NNet.del_linear)
        for i in range(len(self.layerActivation)):
            if self.layerActivation[i] == NNet.sigmoid:
                self.delLayers.append(NNet.del_sigmoid)
            elif self.layerActivation[i] == NNet.tanh:
                self.delLayers.append(NNet.del_tanh)
            elif self.layerActivation[i] == NNet.linear:
                self.delLayers.append(NNet.del_linear)

        # set delLoss based on loss function
        if self.loss == NNet.mse:
            self.delLoss = NNet.del_mse
        elif self.loss == NNet.cross_entropy:
            self.delLoss = NNet.del_cross_entropy

        self.weights = []

        self.layerActivation = layerActivation # array of name of activation function for each layer
        #  set random seed
        if seed is not None:
            np.random.seed(seed)
        #  initialize weights
        self.weights = {}
        for i in range(1, len(self.nlayers)):
            self.weights[i] = np.random.randn(self.nlayers[i], self.nlayers[i-1])
        for k, v in self.weights.items():
            print("weights", k, v.shape)

    # activation functions
    @staticmethod
    def sigmoid(X):
        # X can be a matrix or a scalar
        return 1 / (1 + np.exp(-X))

    def tanh(X):
        return np.tanh(X)

    def linear(X):
        return X
        
    # derivatives of activation functions

    # error functions
    def mse(y, y_hat):
        return np.mean((y - y_hat)**2)

    def cross_entropy(y, y_hat):
        return -np.mean(y * np.log(y_hat) )

    def del_sigmoid(X):
        return NNet.sigmoid(X) * (1 - NNet.sigmoid(X))

    def del_tanh(X):
        return 1 - np.tanh(X)**2

    def del_linear(X):
        return np.ones(X.shape)

    def del_mse(y, y_hat):
        return -2 * (y - y_hat) / len(y)

    def del_cross_entropy(self, y, y_hat):
        return -y / y_hat
    
    
    
    # forward propagation
    def forward(self, X):
        #  initialize activations
        print("Shape of input", X.shape[0])
        self.weights[0] = np.random.randn( self.nlayers[0], X.shape[0])
        self.a = {}
        self.z = {}
        print(self.weights[0].shape)
        self.a[0] = X
        for i in range(0 , len(self.nlayers)):
            self.z[i+1] = np.dot(self.weights[i], self.a[i])
            self.a[i+1] = self.layerActivation[i](self.z[i+1])
          
        
    def error(self, y):
        yhat = self.a[len(self.nlayers)]
        print("shape pf yhat", yhat.shape)
        print("shape of y", y.shape)
        return self.loss(y, self.a[len(self.nlayers)])



    # back propagation
    def backprop(self, y ):
        #  initialize deltas
        self.delta = {}
        self.delta[len(self.nlayers)] = self.delLoss(y, self.a[len(self.nlayers)]) * self.delLayers[len(self.nlayers)-1](self.z[len(self.nlayers)])
        print("delta shape", self.delta[len(self.nlayers)].shape) 
        for i in range(len(self.nlayers)-1, -1, -1):
         

            print("----------\n",i,"\n")
            print("weights_shape::",self.weights[i].shape)
            print("other thing", self.delLayers[i](self.z[i]).shape)
            self.delta[i] = np.dot(self.weights[i].T, self.delta[i+1]) * self.delLayers[i](self.z[i])
            print("delta shape i :",i," ::", self.delta[i].shape) 
           
        #  initialize gradient
        self.grad = {}
        for i in range(1, len(self.nlayers)):
            self.grad[i] = np.dot(self.delta[i+1], self.a[i].T)
        #  update weights
        for i in range(1, len(self.nlayers)):
            self.weights[i] -= self.grad[i] * self.learning_rate
        #  return error
        return self.error(y)
    
    def train(self, X, y, epochs, learning_rate):
        self.weights[0] = np.random.randn( self.nlayers[0], X.shape[0])
        self.learning_rate = learning_rate
        for i in range(epochs):
            self.forward(X)
            self.backprop(y)
            print("Epoch: ", i, "Error: ", self.error(y))
        return self.weights


if __name__ == "__main__":
    #  create a neural network with 2 hidden layers of size 3 and 1 output layer
    # of size 2
    myNet = NNet([100, 90, 80,70,60, 7], [NNet.sigmoid, NNet.sigmoid,NNet.sigmoid, NNet.sigmoid, NNet.sigmoid, NNet.sigmoid], NNet.mse)
    #  create a training
    X = np.array([1, 2, 3, 4,6,7,8])
    y = np.array([1, 0, 1, 0,1,0,1])
    #  train the neural network
    myNet.train(X, y, epochs = 100, learning_rate = 0.1)
    #  evaluate the neural network
