import numpy as np

# ---------------------------------------------------------
# CHALLENGE: FIX THE SHAPE ERRORS! 🔧
# ---------------------------------------------------------

# 1. Input Data: Batch of 10 samples, 4 features each
inputs = np.random.randn(10, 4)
print(f"Input shape: {inputs.shape}")

# ---------------------------------------------------------
# LAYER 1: 4 inputs -> 5 neurons
# ERROR 1: Something is wrong with the weight shape...
# Hint: Weights should be (inputs, neurons)
weights1 = np.random.randn(4, 5) 
biases1 = np.zeros((1, 5))

# Forward pass
# ERROR 2: Dot product might fail if shapes don't align
layer1_output = np.dot(inputs, weights1) + biases1
print(f"Layer 1 output shape: {layer1_output.shape}")

# ---------------------------------------------------------
# LAYER 2: 5 inputs -> 3 neurons
# ERROR 3: Check these shapes carefully
weights2 = np.random.randn(5, 3)
biases2 = np.zeros((1, 3))

# Forward pass
layer2_output = np.dot(layer1_output, weights2) + biases2
print(f"Layer 2 output shape: {layer2_output.shape}")

# ---------------------------------------------------------
# CHECK
if layer2_output.shape == (10, 3):
    print("\n✅ SUCCESS! Shapes are correct.")
else:
    print(f"\n❌ FAIL! Expected (10, 3), got {layer2_output.shape}")
