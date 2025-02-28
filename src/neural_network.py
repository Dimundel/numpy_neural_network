import numpy as np
import matplotlib.pyplot as plt

class Linear:
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

    
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.intermediate_outputs = np.empty(len(layers), dtype=object)
    
    def forward_propagation(self, X, y_true):
        self.intermediate_outputs[0] = self.layers[0].forward(X)
        for i in range(1, len(self.layers)):
            self.intermediate_outputs[i] = self.layers[i].forward(self.intermediate_outputs[i-1])
        
        y_prob = self.intermediate_outputs[-1]
        
        loss = calculate_log_loss(y_true, y_prob)
        return y_prob, loss

    def back_propagation(self, X, y_true, y_prob, learning_rate):
        gradient = calculate_gradient(y_true, y_prob)
        for i in range(len(self.layers) - 1, 0, -1):
            if isinstance(self.layers[i], Linear):
                self.layers[i].gradient_descent(self.layers[i].calculate_gradient(self.intermediate_outputs[i-1], gradient), learning_rate)
                gradient = self.layers[i].back(gradient)
            else:
                gradient = self.layers[i].back(self.intermediate_outputs[i-1], gradient)
        self.layers[0].gradient_descent(self.layers[0].calculate_gradient(X, gradient), learning_rate)
    
    def learn(self, X, y_true, num_of_iterations, learning_rate):
        for _ in range(num_of_iterations):
            y_prob, loss = self.forward_propagation(X, y_true)
            print(f"Current loss is {loss}")
            self.back_propagation(X, y_true, y_prob, learning_rate)
    
    def predict(self, X):
        x = X
        for layer in self.layers:
            x = layer.forward(x)
        return x


def generate_xor_data(num_samples):
    X = np.random.rand(num_samples, 2)

    # XOR operation for output
    y = ((X[:, 0] >= 0.5) & (X[:, 1] <= 0.5)) | ((X[:, 0] <= 0.5) & (X[:, 1] >= 0.5)).astype(int)
    y = y.reshape(-1, 1)

    return X, y

X, y = generate_xor_data(1000)

print("First 5 samples of X:")
print(X[:5])
print("First 5 corresponding outputs (y):")
print(y[:5])

sigmoid = Sigmoid()
layer1 = Linear(np.array([[-0.2, 0.1, -0.5, 0.2, 0.3], [0.6, -0.2, -0.8, -0.5, -0.4]]), np.zeros(5))
layer2 = Linear(np.array([[0.3], [0.2], [-0.5], [0.2], [-0.8]]), np.zeros(1))

nn = NeuralNetwork([layer1, sigmoid, layer2, sigmoid])
nn.learn(X, y, 300000, 0.001)



x_min, x_max = 0, 1
y_min, y_max = 0, 1
grid_size = 100 

x = np.linspace(x_min, x_max, grid_size)
y = np.linspace(y_min, y_max, grid_size)
X, Y = np.meshgrid(x, y)
xy_pairs = np.column_stack([X.ravel(), Y.ravel()])

def f(x, y):
    z = nn.predict(np.array([x, y]))
    return z

Z = np.array([f(x, y) for x, y in xy_pairs])

plt.figure(figsize=(6, 5))
plt.scatter(xy_pairs[:, 0], xy_pairs[:, 1], c=Z, cmap="coolwarm", s=10)
plt.colorbar(label="Function Value (Probability)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("2D Function Visualization")
plt.show()