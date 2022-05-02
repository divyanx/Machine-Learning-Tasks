from layer import Layer

# inherit from base class Layer
class ActivationFuncLayer(Layer):
    def __init__(self, activ, der_activ):
        self.activ = activ
        self.der_activ = der_activ

    # returns the activated input
    # foraward pass function
    def frwd_pass(self, sample_data):
        self.in_set = sample_data
        self.out_set = self.activ(self.in_set)
        return self.out_set

    # Returns in_set_error=dE/dX for a given out_set_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def back_pass(self, out_set_error, learning_rate):
        return self.der_activ(self.in_set) * out_set_error