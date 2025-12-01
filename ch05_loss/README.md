# Chapter 5: Calculating Loss

## What This Chapter Covers
- What loss functions are and why we need them
- Categorical Cross-Entropy Loss (the most common for classification)
- How to measure how "wrong" the network is
- Accuracy vs Loss (different but related metrics)
- One-hot encoding vs sparse labels

## Key Concepts
- **Loss Function**: Measures how far the network's predictions are from the truth
- **Categorical Cross-Entropy**: Loss function for multi-class classification
- **One-hot encoding**: Converting labels to vectors (e.g., class 2 → [0, 0, 1, 0, 0])
- **Sparse labels**: Using integers directly (e.g., class 2 → 2)
- **Lower loss = Better**: We want to minimize loss during training

## Why Loss Functions Matter

Activation functions give us outputs, but how do we know if they're **good** outputs?

**Loss functions answer:** "How wrong is the network?"

- **Low loss** → Predictions close to correct answers ✅
- **High loss** → Predictions far from correct answers ❌

The goal of training is to **minimize the loss**.

## Categorical Cross-Entropy Formula

For a single sample:
```
Loss = -log(predicted_probability_of_correct_class)
```

For multiple samples (average):
```
Loss = -1/N * Σ log(predicted_probability_of_correct_class)
```

### Why Logarithm?

- **Correct prediction** (probability = 1.0): log(1) = 0 → Loss = 0 ✅
- **Wrong prediction** (probability → 0): log(0) → -∞ → Loss = ∞ ❌
- **Uncertain** (probability = 0.5): log(0.5) ≈ -0.69 → Loss moderate

The log heavily penalizes confident wrong predictions!

## Code Pattern

```python
# After softmax activation
softmax_output = [0.7, 0.2, 0.1]  # Predictions for 3 classes
target = [1, 0, 0]                 # True class is 0

# Cross-entropy loss
loss = -np.sum(target * np.log(softmax_output))
```

## Files in This Chapter
- **playground.py**: Your experiments with loss calculation
- **README.md**: This file - chapter notes and concepts

## Questions to Think About
- Why do we use log instead of just (prediction - target)²?
- What happens if the network is very confident but wrong?
- How does one-hot encoding relate to the loss calculation?

## Notes
Add your insights as you work through the chapter:

---

## Next Steps
After Chapter 5, you'll move to **Chapter 6: Optimization** - learning how to actually adjust weights to reduce the loss!
