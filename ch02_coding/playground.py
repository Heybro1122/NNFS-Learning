"""
Chapter 2: Playground
Code your experiments and examples from Chapter 2 here
"""

import numpy as np

print("Chapter 2 - Coding Our First Neurons")
print("=" * 50)

# Example from the book: A single neuron
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1]
bias = 2

output = (inputs[0] * weights[0] + 
          inputs[1] * weights[1] + 
          inputs[2] * weights[2] + 
          inputs[3] * weights[3] + bias)

print(f"Single neuron output: {output}")

print("\n" + "=" * 50)

# Example: A layer of neurons
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

# Output of current layer
layer_outputs = []
# For each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
    # Zeroed output of given neuron
    neuron_output = 0
    # For each input and weight to the neuron
    for n_input, weight in zip(inputs, neuron_weights):
        # Multiply this input by associated weight
        # and add to the neuron's output variable
        neuron_output += n_input * weight
    # Add bias
    neuron_output += neuron_bias
    # Put neuron's result to the layer's output list
    layer_outputs.append(neuron_output)

print(f"Layer outputs (manual): {layer_outputs}")

print("\n" + "=" * 50)

# With NumPy (much cleaner!)
inputs = np.array([1, 2, 3, 2.5])
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]])
biases = np.array([2, 3, 0.5])

layer_outputs = np.dot(weights, inputs) + biases
print(f"Layer outputs (NumPy): {layer_outputs}")

print("\n✅ Chapter 2 Complete!")
print("-" * 50)

# YOUR CODE BELOW:
# Try your own experiments here!
