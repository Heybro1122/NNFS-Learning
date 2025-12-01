import numpy as np
import matplotlib.pyplot as plt

# Input range
x = np.linspace(-5, 5, 100)

# ---------------------------------------------------------
# TASK 1: Implement Sigmoid
# ---------------------------------------------------------
def sigmoid(inputs):
    return 1/(1+np.exp(-inputs))

# ---------------------------------------------------------
# TASK 2: Implement Leaky ReLU
# ---------------------------------------------------------
def leaky_relu(inputs):
    return np.maximum(0.01*inputs, inputs)

# ---------------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------------
y_sigmoid = sigmoid(x)
y_leaky = leaky_relu(x)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x, y_sigmoid)
plt.title("Sigmoid (Expected: S-curve 0 to 1)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x, y_leaky)
plt.title("Leaky ReLU (Expected: Linear with slight negative slope)")
plt.grid(True)

plt.show()
