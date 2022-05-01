import numpy as np
from layer import Layer
from activation import Activation


class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)
# implement linear activation


# class Linear(Activation):
#     def __init__(self):
#         def linear(x):
#             return x

#         def linear_prime(x):
#             if type(x) == np.ndarray:
#                 return np.ones(x.shape)
#             else:
#                 return 1

#         super().__init__(linear, linear_prime)
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


class Linear(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # This version is faster than the one presented in the video
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        # Original formula:
        # tmp = np.tile(self.output, n)
        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)

# # linear function
# def linear(x):
#     """
#     Calculate linear
#     """
#     return x

# # derivative of linear function
# def linear_prime(x):
#     """
#     Derivative of linear function
#     """
#     if type(x) == np.ndarray:
#         return np.ones(x.shape)
#     else:
#         return 1
