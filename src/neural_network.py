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
