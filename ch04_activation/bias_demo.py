"""
Bias Shift Demo
Visualizing why bias shifts X sometimes and Y other times
"""
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

# Create x values
x = np.linspace(-3, 3, 200)

plt.figure(figsize=(12, 5))

# --- PLOT 1: Bias on First Neuron (Horizontal Shift) ---
plt.subplot(1, 2, 1)
plt.title("Neuron 1 Bias: Left/Right Shift\ny = ReLU(-x + b)")

# Base case (b=0)
y_base = relu(-x + 0)
plt.plot(x, y_base, label='Bias = 0', linewidth=2)

# Shifted case (b=1)
y_shifted = relu(-x - 1) # Changed bias to -1
plt.plot(x, y_shifted, label='Bias = -1', linestyle='--')

plt.legend()
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# --- PLOT 2: Bias on Second Neuron (Vertical Shift) ---
plt.subplot(1, 2, 2)
plt.title("Neuron 2 Bias: Up/Down Shift\ny = ReLU(Neuron1) + b")

# Neuron 1 output (fixed)
neuron1_out = relu(-x)

# Base case (b=0)
y2_base = relu(neuron1_out + 0)
plt.plot(x, y2_base, label='Bias2 = 0', linewidth=2)

# Shifted case (b=1)
# This adds 1 to the output of neuron 1
y2_shifted = relu(neuron1_out + 1)
plt.plot(x, y2_shifted, label='Bias2 = 1', linestyle='--')

plt.legend()
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.show()
