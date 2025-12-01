# 🏎️ Speedrun: Chapters 7 & 8 (Derivatives & Gradients)

Since you already know calculus, you can skip 90% of the text. Here is the only code you need to care about.

## Chapter 7: Derivatives (Numerical)
**Goal:** Understand that we *can* calculate derivatives numerically, but it's slow.
**Key Code:** A simple numerical gradient checker.

```python
def f(x):
    return 2 * x**2

def numerical_derivative(f, x, epsilon=1e-4):
    return (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)

print(numerical_derivative(f, 2)) # Should be approx 8.0
```
*   **Takeaway:** Useful for debugging (Gradient Check), but too slow for training.

## Chapter 8: Partial Derivatives & Chain Rule
**Goal:** Understand how to apply Chain Rule to Neural Networks.
**Key Concept:**
*   **Gradient**: Vector of partial derivatives.
*   **Chain Rule**: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w}$

**Key Code:** None! It's all theory.

## 🏁 Verdict
**SKIP STRAIGHT TO CHAPTER 9.**
Chapter 9 is where we actually implement the `backward()` method using the Chain Rule.

### Next Step:
Start **Challenge 5: Backpropagation** (Chapter 9).
We will implement the `backward` pass for:
1.  ReLU
2.  Dense Layer
3.  Softmax + Loss
