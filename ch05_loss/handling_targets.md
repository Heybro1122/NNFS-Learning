# Handling Targets: Sparse vs One-Hot Encoded

This page explains how to make your loss function flexible enough to handle both types of target labels.

## The Two Types of Targets

### 1. Sparse Labels (1D Array)
This is what we used before. It's just a list of the correct class indices.
```python
class_targets = [0, 1, 1]  # Dog, Cat, Cat
# Shape: (3,) -> 1 Dimension
```

### 2. One-Hot Encoded Labels (2D Array)
This is a matrix where each row is all zeros except for a `1` at the correct class index.
```python
class_targets = [[1, 0, 0],  # Dog (index 0 is 1)
                 [0, 1, 0],  # Cat (index 1 is 1)
                 [0, 1, 0]]  # Cat (index 1 is 1)
# Shape: (3, 3) -> 2 Dimensions
```

---

## The Logic: "Check Dimensions"

The code uses `len(class_targets.shape)` to check which format we have.

### Case 1: Sparse Labels (`shape` length is 1)
If we have `[0, 1, 1]`, we use the **fancy indexing** method we just learned:
```python
if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[
        range(len(softmax_outputs)),
        class_targets
    ]
```
*   **Action:** Directly pick the values at indices 0, 1, and 1.

### Case 2: One-Hot Encoded (`shape` length is 2)
If we have the matrix of 0s and 1s, we use **multiplication**.

```python
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(
        softmax_outputs * class_targets,
        axis=1
    )
```

#### How the Math Works (One-Hot)
Imagine Sample 1:
*   Prediction: `[0.7, 0.1, 0.2]`
*   Target: `[1, 0, 0]`

1.  **Multiply:** `[0.7*1, 0.1*0, 0.2*0]` = `[0.7, 0, 0]`
    *   *Notice how the wrong answers become 0!*
2.  **Sum:** `0.7 + 0 + 0` = `0.7`
    *   *We recovered the confidence of the correct class!*

---

## Summary Table

| Label Type | Shape | Example | How to get Correct Confidence |
| :--- | :--- | :--- | :--- |
| **Sparse** | 1D | `[0, 1, 1]` | **Indexing:** Pick the value at the index. |
| **One-Hot** | 2D | `[[1,0,0]...]` | **Math:** Multiply by target (zeros out wrong answers) then Sum. |

This code block ensures your loss function works no matter how the data is formatted!
