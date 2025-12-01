import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# ---------------------------------------------------------
# 1. IMPLEMENT YOUR CLASSES HERE
# ---------------------------------------------------------

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights=.01*np.random.randn(n_inputs,n_neurons)
        self.biases=np.zeros((1,n_neurons))
        pass

    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output=np.dot(inputs,self.weights)+self.biases

class Activation_ReLU:
    def forward(self, inputs):
        # Calculate output values from inputs
        self.output=np.maximum(0,inputs)

class Activation_Softmax:
    def forward(self, inputs):
        # Calculate output values from inputs
        exp_values=np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        self.output=exp_values/np.sum(exp_values,axis=1,keepdims=True)

class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        # Calculate loss
        samples=len(y_pred)
        y_pred_clipped=np.clip(y_pred,1e-7,1-1e-7)
        log_likelihoods=-np.log(y_pred_clipped[range(samples),y_true])
        average_loss=np.mean(log_likelihoods)
        return average_loss

# ---------------------------------------------------------
# 2. EXECUTION SCRIPT
# ---------------------------------------------------------

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()

# Create second Dense layer with 3 input features (as we take output of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)

# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossentropy()

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)

# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Perform a forward pass through activation function
# takes the output of second dense layer here
activation2.forward(dense2.output)

# Let's see output of the first few samples:
print(activation2.output[:5])

# Calculate the loss value
loss = loss_function.forward(activation2.output, y)

print('loss:', loss)
