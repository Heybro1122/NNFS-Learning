# Understanding Target Selection (The "Confidence Check")

You are looking at a crucial step in calculating loss: **Finding out how confident the network was about the CORRECT answer.**

## The Problem
The network gives you probabilities for **ALL** classes (e.g., Dog, Cat, Bird).
But to calculate loss (how "wrong" it is), we only care about the probability it assigned to the **TRUE** class.

## The Data
Imagine we have 3 samples:

1.  **Sample 1**: True answer is **Dog** (Index 0)
    *   Network said: `[0.7, 0.1, 0.2]` (70% Dog, 10% Cat, 20% Bird)
    *   We want: **0.7**

2.  **Sample 2**: True answer is **Cat** (Index 1)
    *   Network said: `[0.1, 0.5, 0.4]` (10% Dog, 50% Cat, 40% Bird)
    *   We want: **0.5**

3.  **Sample 3**: True answer is **Cat** (Index 1)
    *   Network said: `[0.02, 0.9, 0.08]` (2% Dog, 90% Cat, 8% Bird)
    *   We want: **0.9**

## The Python Code (The "Manual" Way)

The code in the image uses a loop to grab these specific numbers.

```python
softmax_outputs = [[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]]

class_targets = [0, 1, 1]  # The correct indices

# zip() pairs them up:
# 1. (target=0, distribution=[0.7, 0.1, 0.2])
# 2. (target=1, distribution=[0.1, 0.5, 0.4])
# 3. (target=1, distribution=[0.02, 0.9, 0.08])

for targ_idx, distribution in zip(class_targets, softmax_outputs):
    print(distribution[targ_idx])
```

**What happens inside the loop:**
1.  **First iteration**: `targ_idx` is 0. It looks at `[0.7, 0.1, 0.2]` and grabs index 0. **Result: 0.7**
2.  **Second iteration**: `targ_idx` is 1. It looks at `[0.1, 0.5, 0.4]` and grabs index 1. **Result: 0.5**
3.  **Third iteration**: `targ_idx` is 1. It looks at `[0.02, 0.9, 0.08]` and grabs index 1. **Result: 0.9**

## The NumPy Way (The "Fast" Way)

The book is leading up to this. Instead of a slow loop, we can use NumPy's **fancy indexing**.

```python
import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([0, 1, 1])

# NumPy syntax: array[row_indices, column_indices]
correct_confidences = softmax_outputs[
    range(len(softmax_outputs)),  # [0, 1, 2] (The rows)
    class_targets                 # [0, 1, 1] (The columns we want)
]

print(correct_confidences)
# Output: [0.7 0.5 0.9]
```

### Breaking Down the NumPy Indexing

**Step 1: Understanding what we need**
We need to grab:
- Row 0, Column 0 → `0.7`
- Row 1, Column 1 → `0.5`
- Row 2, Column 1 → `0.9`

**Step 2: Creating the row indices**
```python
range(len(softmax_outputs))  # Creates: range(3) which is [0, 1, 2]
```
This gives us the row numbers for each sample.

**Step 3: The column indices (already given)**
```python
class_targets = [0, 1, 1]  # These are the column positions we want
```

**Step 4: NumPy's "Paired Indexing"**
When you write `array[rows, cols]` where both are lists/arrays, NumPy **pairs them up**:

```
array[
    [0, 1, 2],  # Row indices
    [0, 1, 1]   # Column indices
]

Pairs up as:
- (row 0, col 0) → softmax_outputs[0, 0] = 0.7
- (row 1, col 1) → softmax_outputs[1, 1] = 0.5
- (row 2, col 1) → softmax_outputs[2, 1] = 0.9
```

**Visual representation:**
```
         Col 0   Col 1   Col 2
Row 0  [ 0.7     0.1     0.2  ]  ← Pick index 0 (0.7) ✓
Row 1  [ 0.1     0.5     0.4  ]  ← Pick index 1 (0.5) ✓
Row 2  [ 0.02    0.9     0.08 ]  ← Pick index 1 (0.9) ✓
```

### Why This Works

**The key insight:** NumPy lets you provide **two arrays of the same length** for indexing, and it will **zip them together** automatically.

Think of it like this in pseudo-code:
```python
for i in range(len(rows)):
    result[i] = array[rows[i], cols[i]]
```

But NumPy does this **all at once** (vectorized), which is **much faster** than a Python loop!

### Complete Example with Print Statements

```python
import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([0, 1, 1])

# Create row indices
row_indices = range(len(softmax_outputs))
print("Row indices:", list(row_indices))  # [0, 1, 2]
print("Column indices:", class_targets)    # [0, 1, 1]

# Paired indexing
print("\nWhat we're grabbing:")
print("softmax_outputs[0, 0] =", softmax_outputs[0, 0])  # 0.7
print("softmax_outputs[1, 1] =", softmax_outputs[1, 1])  # 0.5
print("softmax_outputs[2, 1] =", softmax_outputs[2, 1])  # 0.9

# All at once!
correct_confidences = softmax_outputs[row_indices, class_targets]
print("\nResult:", correct_confidences)  # [0.7 0.5 0.9]
```

## Why do we need this?
These numbers (`0.7`, `0.5`, `0.9`) are the **Predicted Probabilities of the Correct Class**.
We will pass THESE numbers into the negative log function to calculate loss.

*   `0.7` (Good guess) → Low Loss
*   `0.5` (Meh guess) → Medium Loss
*   `0.9` (Great guess) → Very Low Loss
