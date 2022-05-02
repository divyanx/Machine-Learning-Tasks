def predict(network, input):
    """
    :param network: list of layers
    :param input: input data
    :return: output of the network

    """
    output = input
    for layer in network:
        output = layer.forward_pass(output)
    return output


def train(network, loss, loss_derivative, x_train, y_train, learning_rate, epochs):
    """
    :param network: list of layers
    :param loss: loss function
    :param loss_derivative: derivative of loss function
    :param x_train: input data
    :param y_train: output data
    :param learning_rate: learning rate
    :param epochs: number of epochs
    :param verbose: print loss after each epoch
    :return: None

    """
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward_pass
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_derivative(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)

        print(f"{e + 1}/{epochs}, ----- error for this epoch is={error}")
