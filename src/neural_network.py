import numpy as np

class Layer:
    def __init__ (self, weights_size, input_size):
        self.weights = np.zeros(shape=(weights_size, input_size))
        self.bias = np.zeros(shape=weights_size)
    
    def forward(self, input):
        return np.matmul(self.weights, input) + self.bias

class ReLU:
    def __init__(self):
        pass
    
    def forward(self, input):
        return np.maximum(0, input)