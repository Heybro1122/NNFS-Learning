# Chapter 4: Activation Functions

## What This Chapter Covers
- Why activation functions are essential
- Different types of activation functions
- Step, Sigmoid, ReLU, and Softmax
- Where and when to use each activation function
- The vanishing gradient problem

## Key Concepts
- **Activation Function**: Non-linear function applied to neuron outputs
- **Non-linearity**: Allows networks to learn complex patterns (without it, layers just collapse to one layer!)
- **ReLU** (Rectified Linear Unit): Most common hidden layer activation
- **Softmax**: Used for output layer in classification
- **Dead neurons**: Problem with ReLU when outputs become permanently zero

## Common Activation Functions

### Step Function
```python
output = 1 if x > 0 else 0
```
- Too simple for modern networks
- Not differentiable (problem for backpropagation)

### Sigmoid
```python
output = 1 / (1 + np.exp(-x))
```
- Outputs between 0 and 1
- Useful for binary classification
- Suffers from vanishing gradient

### ReLU (Rectified Linear Unit)
```python
output = max(0, x)
```
- Most popular for hidden layers
- Fast to compute
- Helps with vanishing gradient problem
- Can cause "dead neurons"

### Softmax
```python
exp_values = np.exp(x - np.max(x))
output = exp_values / np.sum(exp_values)
```
- Used for **output layer** in multi-class classification
- Outputs sum to 1 (probability distribution)
- Interprets as confidence scores

## Why Non-Linearity Matters

**Without activation functions:**
```
Layer1: y = W1·x + b1
Layer2: y = W2·y + b2 = W2·(W1·x + b1) + b2
```
This is still just a linear function! Multiple layers don't help.

**With activation functions:**
```
Layer1: y = activation(W1·x + b1)
Layer2: y = activation(W2·y + b2)
```
Now the network can learn complex, non-linear patterns! 🎉

## Files in This Chapter
- **playground.py**: Your experiments with activation functions
- **bias_demo.py**: Visual comparison of horizontal vs vertical bias shifts
- **bias_demo_interactive.py**: Interactive sliders to experiment with weights and biases
- **bias_explanation.md**: Detailed explanation of why bias moves graphs differently
- **README.md**: This file - chapter notes and concepts

## Questions to Think About
- Why does ReLU work so well despite being so simple?
- When would you use Sigmoid vs Softmax?
- What happens if you don't use activation functions at all?

## Notes
Add your insights as you work through the chapter:

---

## Next Steps
After Chapter 4, you'll have a complete forward pass with activations! Then Chapter 5 introduces **loss functions** - how to measure how wrong your network is.
