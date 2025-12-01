# Why We Clip Values (The "Infinity" Problem)

This page solves a specific mathematical problem with the log function.

## The Problem: Log(0) is Undefined
The loss formula is:
$$ \text{Loss} = -\log(\text{Correct Probability}) $$

*   If the network is **100% wrong**, it predicts `0` for the correct class.
*   $\log(0) = -\infty$ (Negative Infinity)
*   $-\log(0) = \infty$ (Positive Infinity)

Computers hate infinity. It crashes the training.

## The Solution: Clipping
We need to ensure the probability is never *exactly* 0.
So we "clip" the values to be slightly larger than 0.

### Attempt 1: Just add a tiny number?
If we just add `1e-7` (0.0000001) to everything:
*   `0` becomes `0.0000001` (Good! No infinity.)
*   `1` becomes `1.0000001` (Wait...)

**The Issue:**
If we have a perfect prediction (`1.0`), it becomes `1.0000001`.
$$ -\log(1.0000001) \approx -0.00000004 $$
We get a **negative loss**. Loss should range from 0 to infinity. Negative loss makes no sense conceptually (you can't be "better than perfect").

### Attempt 2: Clip Both Sides (The Correct Way)
To keep things balanced and valid, we clip the range to be between:
*   **Minimum:** `1e-7` (Just above 0)
*   **Maximum:** `1 - 1e-7` (Just below 1)

```python
y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
```

**What this does:**
1.  `0` $\rightarrow$ `1e-7` (Prevents infinity error)
2.  `1` $\rightarrow$ `0.9999999` (Prevents negative loss)
3.  `0.5` $\rightarrow$ `0.5` (Unchanged)

This keeps our math safe and our loss values positive!
