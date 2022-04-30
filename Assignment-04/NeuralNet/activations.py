import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

# sigmoid function
def sigmoid(x):
    """
    Calculate sigmoid :-
            g(x) = 1 / (1 + e^{-x})
    
    """
    return 1 / (1 + np.exp(-x))



# derivative of sigmoind function
def sigmoid_prime(x):
    """
    Derivative of sigmoid function
    """
    return sigmoid(x) * (1 - sigmoid(x))

# ReLU function
def relu(x):
    """
    Calculate relu
    """
    return np.maximum(0, x)

# derivative of relu function
def relu_prime(x):
    """
    Derivative of relu function
    """
    return np.where(x > 0, 1, 0)

# linear function
def linear(x):
    """
    Calculate linear
    """
    return x

# derivative of linear function
def linear_prime(x):
    """
    Derivative of linear function
    """
 
    # x will be a vector of shape (n,1)
    # return derivative of linear function
    return np.ones(x.shape)
    

# soft max function
def softmax(x):
    """
    Calculate softmax
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# derivative of softmax function
# def softmax_prime(x):
#     """
#     Derivative of softmax function
#     """
#     # return softmax(x) * (1 - softmax(x))                                           
#     # return softmax(x) * np.identity(softmax(x).size) - softmax(x).transpose() @ softmax(x)

    
    
def softmax_prime(x):
    s = softmax(x)
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = s.reshape(-1,1)
    a =  np.diagflat(s) - np.dot(s, s.T)
  
    return a

