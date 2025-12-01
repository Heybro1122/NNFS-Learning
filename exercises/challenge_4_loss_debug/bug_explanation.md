# Post-Mortem: The "Cheating Loss" Bug 🕵️‍♀️

In Challenge 4, we found a critical bug in how the loss was calculated. Here is exactly what went wrong.

## The Scenario
We had a sample where the network was **WRONG**.
*   **True Class**: 1 (Cat)
*   **Prediction**: `[0.1, 0.3, 0.6]` (10% Dog, 30% Cat, 60% Bird)

## The Bug: `np.max()`
The original code used:
```python
correct_confidences = np.max(y_pred, axis=1)
```
*   **What it did**: Looked at `[0.1, 0.3, 0.6]` and picked the biggest number: **0.6**.
*   **The Logic**: "I am 60% confident about... *something*."
*   **The Result**: The loss was calculated based on 0.6 (Low Loss).

**Why this is bad**: The network was confident about the **WRONG** class (Bird), but we rewarded it as if it were confident about the **RIGHT** class! We effectively told the network: *"Good job! You were very confident!"* even though it was wrong.

## The Fix: Fancy Indexing
We changed it to:
```python
correct_confidences = y_pred[range(len(y_pred)), y_true]
```
*   **What it did**: Looked at `[0.1, 0.3, 0.6]` and picked the value at the **True Class Index (1)**: **0.3**.
*   **The Logic**: "I am only 30% confident about the *correct answer*."
*   **The Result**: The loss was calculated based on 0.3 (High Loss).

**Why this is correct**: We punished the network for having low confidence in the correct answer.

## Summary

| Method | Value Picked | Meaning | Effect on Training |
| :--- | :--- | :--- | :--- |
| **`np.max`** | 0.6 | "Highest confidence (ignoring truth)" | **Disaster.** Network learns to be confident about *anything*, even wrong answers. |
| **Indexing** | 0.3 | "Confidence in the TRUE class" | **Success.** Network learns to increase confidence specifically for the right answer. |
