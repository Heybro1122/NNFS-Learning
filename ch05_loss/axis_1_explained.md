# Understanding `axis=1` in One-Hot Loss Calculation

You asked about this specific block of code:
```python
correct_confidences = np.sum(
    softmax_outputs * class_targets,
    axis=1
)
```

## The Context
We are handling **One-Hot Encoded** targets.
*   **Prediction:** `[[0.7, 0.1, 0.2], ...]` (3 values per sample)
*   **Target:** `[[1, 0, 0], ...]` (3 values per sample)

## Step 1: The Multiplication
First, we do `softmax_outputs * class_targets`. This is element-wise multiplication.

**Sample 1:**
*   Prediction: `[0.7, 0.1, 0.2]`
*   Target: `[1, 0, 0]`
*   Result: `[0.7, 0.0, 0.0]`

We now have a matrix that looks like this:
```
[[0.7, 0.0, 0.0],   ← Sample 1
 [0.0, 0.5, 0.0],   ← Sample 2
 [0.0, 0.9, 0.0]]   ← Sample 3
```

## Step 2: The Sum with `axis=1`

We want to turn that matrix into a simple list of correct scores: `[0.7, 0.5, 0.9]`.

### What `axis=1` Means
*   **axis=0** = Vertical direction (Down the rows)
*   **axis=1** = Horizontal direction (Across the columns)

When we say `np.sum(..., axis=1)`, we are telling NumPy:
**"Squash the columns together. Sum up the numbers in each ROW."**

### Visualizing the Sum

```
Row 0: [0.7, 0.0, 0.0]  → Sum across → 0.7
Row 1: [0.0, 0.5, 0.0]  → Sum across → 0.5
Row 2: [0.0, 0.9, 0.0]  → Sum across → 0.9
```

### What if we used `axis=0`? (The Mistake)
If we used `axis=0`, we would sum **down**:
```
Col 0   Col 1   Col 2
 0.7     0.0     0.0
 0.0     0.5     0.0
 0.0     0.9     0.0
 ↓       ↓       ↓
 0.7     1.4     0.0  (Total nonsense!)
```
This would give us "total confidence for class 0 across all samples", which is not what we want for calculating loss per sample.

## Conclusion
**`axis=1`** ensures we calculate a single score **per sample** (per row). It collapses the "class probabilities" dimension into a single "correct confidence" value.
