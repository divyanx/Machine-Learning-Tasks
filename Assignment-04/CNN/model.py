import numpy as np
# create a CNN model class


class CNN():

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def get_model(self):
        return self.layers

    def predict(self, network, input):
        """
        :param network: list of layers
        :param input: input data
        :return: output of the network

        """
        output = input
        for layer in network:
            output = layer.forward_pass(output)
        return output

    def fit(self,  loss, loss_derivative, x_fit, y_fit, learning_rate, epochs):
        """
        :param network: list of layers
        :param loss: loss function
        :param loss_derivative: derivative of loss function
        :param x_fit: input data
        :param y_fit: output data
        :param learning_rate: learning rate
        :param epochs: number of epochs
        :param verbose: print loss after each epoch
        :return: None

        """
        network = self.get_model()
        for e in range(epochs):
            error = 0
            for x, y in zip(x_fit, y_fit):
                # find the output of the network after one forward pass
                output = self.predict(network, x)
                # calculate the error
                error += loss(y, output)
                # do the backward pass
                grad = loss_derivative(y, output)
                for layer in reversed(network):
                    grad = layer.backward_pass(grad, learning_rate)

            error /= len(x_fit)

            print(f"epoch - {e + 1}, ----- error for this epoch is={error}")

    def accuracy(self,  x_test, y_test):
        # test
        test = []
        pred = []
        for x, y in zip(x_test, y_test):
            output = self.predict(self.get_model(), x)
            # apply softmax to the output and get the predicted class
            test.append(np.argmax(output))
            pred.append(np.argmax(y))
        # find the accuracy
        print('accuracy:', np.sum(np.array(test) == np.array(pred))*100/len(test))
