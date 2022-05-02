import numpy as np


class Activation():
    def __init__(self, activation, activation_derivative):
        """
        Initializes an activation layer.
        :param activation: activation function
        :param activation_derivative: derivative of activation function

        """
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward_pass(self, input):
        """
        :param input: input data
        :return: output of the activation layer

        """
        self.input = input
        return self.activation(self.input)

    def backward_pass(self, output_gradient, learning_rate):
        """
        :param output_gradient: gradient of the loss function with respect to the output of the activation layer
        :param learning_rate: learning rate
        :return: gradient of the loss function with respect to the input of the activation layer

        """
        return np.multiply(output_gradient, self.activation_derivative(self.input))

# Activation Layers


class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            """
            :param x: input data
            :return: output of the tanh activation function
            """
            return np.tanh(x)

        def tanh_derivative(x):
            """
            :param x: input data
            :return: derivative of the tanh activation function

            """
            return 1 - np.tanh(x) ** 2
        # call the parent class's constructor
        super().__init__(tanh, tanh_derivative)
# implement linear activation


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            """
            :param x: input data
            :return: output of the sigmoid activation function
            """
            return 1 / (1 + np.exp(-x))

        def sigmoid_derivative(x):
            """
            :param x: input data
            :return: derivative of the sigmoid activation function

            """
            s = sigmoid(x)
            return s * (1 - s)
        # call the parent class's constructor
        super().__init__(sigmoid, sigmoid_derivative)


class Linear(Activation):
    def __init__(self):
        def sigmoid(x):
            """
            :param x: input data
            :return: output of the sigmoid activation function
            """
            return 1 / (1 + np.exp(-x))

        def sigmoid_derivative(x):
            """
            :param x: input data
            :return: derivative of the sigmoid activation function
            """
            s = sigmoid(x)
            return s * (1 - s)
        # call the parent class's constructor
        super().__init__(sigmoid, sigmoid_derivative)


class Flatten():

    def __init__(self, input_shape, output_shape):
        """
        Initializes a flatten layer.
        :param input_shape: shape of the input data
        :param output_shape: shape of the output data

        """
        self.input_shape = input_shape
        # get the product of all dimensions of input_shape
        self.output_shape = output_shape

    def forward_pass(self, input):
        """
        :param input: input data
        :return: output of the flatten layer

        """
        return np.reshape(input, self.output_shape)

    def backward_pass(self, output_gradient, learning_rate):
        """
        :param output_gradient: gradient of the loss function with respect to the output of the flatten layer
        :param learning_rate: learning rate
        :return: gradient of the loss function with respect to the input of the flatten layer
        """
        return np.reshape(output_gradient, self.input_shape)

# Fully Connected Layer


class FullyConnected():
    def __init__(self, input_size, output_size):
        """
        Initializes a fully connected layer.
        :param input_size: number of input neurons
        :param output_size: number of output neurons

        """
        # initialize weights and bias
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward_pass(self, input):
        """
        :param input: input data
        :return: output of the fully connected layer

        """
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward_pass(self, output_gradient, learning_rate):
        """
        :param output_gradient: gradient of the loss function with respect to the output of the fully connected layer
        :param learning_rate: learning rate
        :return: gradient of the loss function with respect to the input of the fully connected layer

        """
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
