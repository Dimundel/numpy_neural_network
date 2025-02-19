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
layer1 = Layer(np.array([[-0.2, 0.1, 0.5, 0.2, 0.3], [0.6, -0.2, -0.8, -0.5, -0.4]]), np.zeros(5))
layer2 = Layer(np.array([[0.3], [0.2], [-0.5], [0.2], [-0.8]]), np.zeros(1))

def learn(X, Y):
     for i in range(250000):
        res1 = layer1.forward(X)
        res2 = sigmoid.forward(res1)
        res3 = layer2.forward(res2)
        res4 = sigmoid.forward(res3)
        loss = calculate_log_loss(Y, res4)

        print(f"Loss function is : {loss}")

        grad1 = calculate_gradient(Y, res4)
        grad2 = sigmoid.back(res3, grad1)
        layer2.gradient_descent(layer2.calculate_gradient(res2, grad2), 0.002)
        grad3 = layer2.back(grad2)
        grad4 = sigmoid.back(res1, grad3)
        layer1.gradient_descent(layer1.calculate_gradient(X, grad4), 0.002)

learn(X, y)

x_min, x_max = 0, 1
y_min, y_max = 0, 1
grid_size = 100 

x = np.linspace(x_min, x_max, grid_size)
y = np.linspace(y_min, y_max, grid_size)
X, Y = np.meshgrid(x, y)
xy_pairs = np.column_stack([X.ravel(), Y.ravel()])

def f(x, y):
    return sigmoid.forward(layer2.forward(sigmoid.forward(layer1.forward(np.array([[x, y]])))))

Z = np.array([f(x, y) for x, y in xy_pairs])

# print(sigmoid.forward(np.array([[0], [1]])))