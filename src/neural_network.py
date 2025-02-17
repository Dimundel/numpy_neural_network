import numpy as np

class Layer:
    def __init__ (self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def forward(self, input):
        return np.matmul(input, self.weights) + self.bias

class ReLU:
    def __init__(self):
        pass
    
    def forward(self, input):
        return np.maximum(0, input)

class Sigmoid:
    def __init__(self, weights):
        self.weights = weights

    def forward(self, input):
        return 1 / (1 + np.exp(-self.weights * input))

    def back(self, forward_value, forward_gradient):
        return forward_value * (1 - forward_value) * forward_gradient

def calculate_log_loss(y_true, y_prob):
    return -1*y_true*np.log(y_prob)-(1-y_true)*np.log(1-y_prob)

def calculate_gradient(y_true, y_prob):
    return (y_true - y_prob)/ y_prob / (1 - y_prob)
