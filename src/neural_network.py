import numpy as np

class Layer:
    def __init__ (self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def forward(self, input):
        return np.matmul(input, self.weights) + self.bias
    
    def calculate_gradient(self, input_value, forward_gradient):
        return np.matmul(input_value.transpose(), forward_gradient)
    
    def back(self, forward_gradient):
        return np.matmul(forward_gradient, self.weights.transpose())
    
    def gradient_descent(self, gradient, learning_rate):
        self.weights -= learning_rate*gradient

class ReLU:
    def __init__(self):
        pass
    
    def forward(self, input):
        return np.maximum(0, input)
    
    def back(self, input_value, forward_gradient):
        return (input_value > 0).astype(int) * forward_gradient

class Sigmoid:
    def forward(self, input):
        return 1 / (1 + np.exp(-input))

    def back(self, input_value, forward_gradient):
        return self.forward(input_value) * (1 - self.forward(input_value)) * forward_gradient

def calculate_log_loss(y_true, y_prob):
    return -np.mean(y_true*np.log(y_prob)+(1-y_true)*np.log(1-y_prob))

def calculate_gradient(y_true, y_prob):
    return (y_prob - y_true)/ (y_prob * (1 - y_prob))

x = np.array([[2, 3], [-2, 1]])
sigmoid = Sigmoid()
W = np.array([[1], [-0.2]])
layer = Layer(W, np.zeros(W.shape[1]))
res1 = layer.forward(x)
# print(res1)
res2 = sigmoid.forward(res1)
# print(res2)
print("Loss before backpropagation:")
print(calculate_log_loss(np.array([[1], [0]]), res2))
first = sigmoid.back(res1, calculate_gradient(np.array([[1], [0]]), res2))
# print(first)
second = layer.calculate_gradient(x, first)
# print(second)
layer.weights -= 0.1*second
# print(layer.weights)
print("Loss after backpropagation:")
print(calculate_log_loss(np.array([[1], [0]]), sigmoid.forward(layer.forward(x))))

# print(sigmoid.forward(np.array([[0], [1]])))