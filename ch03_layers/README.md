# Chapter 3: Adding Layers

## What This Chapter Covers
- Building multiple layers in a neural network
- Connecting layers together (output of one → input of next)
- Creating a complete forward pass through multiple layers
- Understanding network depth
- Layer initialization and weight matrices

## Key Concepts
- **Layer stacking**: Passing outputs from one layer as inputs to the next
- **Hidden layers**: Layers between input and output
- **Forward propagation**: Data flowing from input → hidden → output
- **Network depth**: Number of layers in the network
- **Deep learning**: Neural networks with multiple hidden layers

## The Core Pattern

```python
# Layer 1
layer1_output = np.dot(inputs, weights1.T) + biases1

# Layer 2 (uses output of layer 1 as input)
layer2_output = np.dot(layer1_output, weights2.T) + biases2

# Layer 3 (uses output of layer 2 as input)
layer3_output = np.dot(layer2_output, weights3.T) + biases3
```

## Why Multiple Layers?
- **Representation learning**: Each layer learns different features
- **Non-linearity** (coming in Ch 4): Allows complex patterns
- **Hierarchical features**: Early layers learn simple patterns, later layers learn complex ones
- **More capacity**: Deeper networks can learn more complex functions

## Files in This Chapter
- **playground.py**: Your experiments with multiple layers
- **README.md**: This file - chapter notes and concepts

## Questions to Think About
- How does the output shape of one layer relate to the input shape of the next?
- What happens to the weights matrix dimensions when stacking layers?
- Why can't we just make one giant layer instead of many smaller ones?

## Notes
Add your insights as you work through the chapter:

---

## Next Steps
After Chapter 3, you'll move to **Chapter 4: Activation Functions** - this is where neural networks get their real power through non-linearity!
