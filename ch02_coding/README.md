# Chapter 2: Coding Our First Neurons

## What This Chapter Covered
- Coding a single neuron using pure Python
- Coding a layer of neurons
- Understanding the dot product for neural network computations
- Introduction to batches of data
- NumPy fundamentals: arrays, matrices, and operations
- **Key topic**: Understanding axis in NumPy operations

## Key Concepts
- **Weights**: Parameters that multiply inputs
- **Biases**: Parameters that shift the output
- **Dot Product**: Core operation for computing neuron outputs (w⃗ · x⃗ + b)
- **Batches**: Processing multiple samples at once
- **NumPy Arrays**: Efficient multi-dimensional data structures
- **Axis**: The dimension along which operations are performed

## Main Code Pattern

A single neuron:
```python
output = sum(inputs * weights) + bias
```

A layer of neurons (the manual way):
```python
layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = sum(inputs * neuron_weights) + neuron_bias
    layer_outputs.append(neuron_output)
```

With NumPy (the efficient way):
```python
output = np.dot(inputs, weights.T) + biases
```

## Files in This Chapter
- **expand_dims_demo.py**: Understanding np.expand_dims() and dimensions
- **axis_test.py**: Testing different axis values
- **axis_explained.py**: Comprehensive explanation of what axis means in NumPy
- **playground.py**: Your experiments and code from the chapter

## Aha Moments! 💡

### Understanding NumPy Axes
**What I initially thought:**
- axis=0 is the X-axis (horizontal)
- axis=1 is the Y-axis (vertical)
- axis=2 is the Z-axis (depth)

**What it actually is:**
- axis=0 is the **first dimension** (rows, vertical ↓)
- axis=1 is the **second dimension** (columns, horizontal →)
- axis=2 is the **third dimension** (depth ⊙)

The axis number refers to the **indexing order** (arr[axis0, axis1, axis2]), NOT Cartesian coordinates!

**Key insight:** When you operate along an axis, that dimension "collapses" or disappears from the result.
- Shape (2, 3) with axis=0 → Result shape (3,)
- Shape (2, 3) with axis=1 → Result shape (2,)

## Next Steps
Ready for Chapter 3: Adding Layers! This is where we start building deeper networks by stacking multiple layers together.
