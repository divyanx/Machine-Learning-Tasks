class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.der_loss = None

    # use layer to network
    def insert(self, layer):
        self.layers.append(layer)

    # set loss to employ
    def employ(self, loss, der_loss):
        self.loss = loss
        self.der_loss = der_loss

    # predict out for given input
    def predict(self, input_data):
        # sample dimension first
        in_data = len(input_data)
        ans = []

        # run network over all input data(sample data)
        for i in range(in_data):
            # forward propagation
            out = input_data[i]
            for layer in self.layers:
                out = layer.frwd_pass(out)
            ans.append(out)

        return ans

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        in_data = len(x_train)

        # training loop
        for i in range(epochs):
            mistake = 0
            for j in range(in_data):
                # forward propagation
                out = x_train[j]
                for layer in self.layers:
                    out = layer.frwd_pass(out)

                # compute loss (for display purpose only)
                mistake += self.loss(y_train[j], out)

                # backward propagation
                error = self.der_loss(y_train[j], out)
                for layer in reversed(self.layers):
                    error = layer.back_pass(error, learning_rate)

            # calculate average error on all input data
            mistake /= in_data
            print('epoch %d/%d   error=%f' % (i+1, epochs, mistake))