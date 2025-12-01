# Chapter 9: Backpropagation

## Overview
This chapter covers **backpropagation** — the algorithm that trains neural networks by computing gradients and updating weights.

## Key Concepts
1. **Backward Pass**: Computing gradients layer by layer (reverse of forward pass)
2. **Chain Rule**: How gradients flow through nested functions
3. **Gradient Descent**: Using gradients to update weights and biases

## Files
- `playground.py` - Experiments with backpropagation implementation
- `README.md` - This file

## What You'll Learn
- How to implement `backward()` methods for each layer
- Why we save inputs/outputs during the forward pass
- How gradients accumulate through the chain rule
- The relationship between loss gradients and parameter updates

## Related Challenges
See `exercises/challenge_5_backprop/` for hands-on practice implementing backprop from scratch.
