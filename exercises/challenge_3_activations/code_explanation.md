# Explaining Activation Functions Code

Here is a line-by-line breakdown of your implementation in Challenge 3.

## 1. Sigmoid Implementation
```python
def sigmoid(inputs):
    return 1 / (1 + np.exp(-inputs))
```

*   **`np.exp(-inputs)`**: Calculates $e^{-x}$ for every value in the input array.
    *   If $x$ is big positive (e.g., 10), $e^{-10}$ is tiny ($\approx 0$).
    *   If $x$ is big negative (e.g., -10), $e^{10}$ is huge ($\approx 22026$).
*   **`1 + ...`**: Adds 1 to the result.
    *   Big positive $x \rightarrow 1 + 0 = 1$.
    *   Big negative $x \rightarrow 1 + \text{huge} = \text{huge}$.
*   **`1 / ...`**: Inverts it.
    *   Big positive $x \rightarrow 1 / 1 = 1$.
    *   Big negative $x \rightarrow 1 / \text{huge} \approx 0$.
*   **Result**: Squashes everything between 0 and 1.

## 2. Leaky ReLU Implementation
```python
def leaky_relu(inputs):
    return np.maximum(0.01 * inputs, inputs)
```

*   **`0.01 * inputs`**: Creates a "shadow" version of the inputs that is scaled down to 1%.
*   **`np.maximum(a, b)`**: Compares two arrays element-wise and picks the larger value.

### Case A: Positive Input ($x = 5$)
*   `a = 0.01 * 5 = 0.05`
*   `b = 5`
*   `max(0.05, 5)` is **5**.
*   **Result**: Returns the original value (Linear).

### Case B: Negative Input ($x = -5$)
*   `a = 0.01 * -5 = -0.05`
*   `b = -5`
*   `max(-0.05, -5)` is **-0.05**.
    *   *Note: -0.05 is "larger" (closer to 0) than -5.*
*   **Result**: Returns the scaled-down value (The "Leak").

This clever one-liner avoids using `if/else` statements, making it very fast!

## 3. Visualization (Matplotlib)
```python
plt.figure(figsize=(10, 5))
```
*   Creates a blank canvas that is 10 inches wide and 5 inches tall.

```python
plt.subplot(1, 2, 1)
```
*   **Grid System**: `(Rows, Cols, Index)`
*   `(1, 2, 1)` means: "I have a grid with **1 Row** and **2 Columns**. I am drawing in the **1st** slot (Left)."

```python
plt.plot(x, y_sigmoid)
plt.title("Sigmoid")
plt.grid(True)
```
*   Draws the Sigmoid curve on the left plot.
*   Adds a title and grid lines.

```python
plt.subplot(1, 2, 2)
```
*   `(1, 2, 2)` means: "Same grid (1 Row, 2 Cols), but now I am drawing in the **2nd** slot (Right)."

```python
plt.plot(x, y_leaky)
plt.show()
```
*   Draws the Leaky ReLU curve on the right plot.
*   **`plt.show()`**: Actually opens the window to display the final image. Without this, nothing happens!
