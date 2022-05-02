from layer import Layer
import numpy as np

# inherit from base class Layer
class FullyConnLayer(Layer):
    # in_sample_size = # of input neurons
    # out_sample_size = # of output neurons
    def __init__(self, in_sample_size, out_sample_size):
        self.weights = np.random.rand(in_sample_size, out_sample_size) - 0.5
        self.bias = np.random.rand(1, out_sample_size) - 0.5

    # returns output for a given input
    def frwd_pass(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output error=dE/dY. Returns input error=dE/dX.
    def back_pass(self, out_error, learning_rate):
        in_error = np.dot(out_error, self.weights.T)
        err_in_weights = np.dot(self.input.T, out_error)
        # dBias = out_error

        # update parameters
        self.weights -= learning_rate * err_in_weights
        self.bias -= learning_rate * out_error
        return in_error