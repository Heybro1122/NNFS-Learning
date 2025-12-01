# Learning Progress Tracker

Track your journey through Neural Networks from Scratch!

## Part I - Foundation

### Chapter 1: Introducing Neural Networks
- [ ] Read chapter
- [ ] Understand biological inspiration
- [ ] Grasp high-level NN concepts
- [ ] **Key Concepts:** Neurons, layers, weights, biases
- [ ] **Notes:**

---

### Chapter 2: Coding Our First Neurons
- [x] Read chapter
- [x] Code: Single neuron
- [x] Code: Layer of neurons
- [x] Code: Batch processing
- [x] **Key Concepts:** Dot product, matrix multiplication, NumPy basics
- [x] **Exercises Completed:** Understanding axis, expand_dims demos
- [x] **Notes:** Learned about NumPy axis indexing - it's NOT Cartesian coordinates!

---

### Chapter 3: Adding Layers
- [x] Read chapter
- [x] Code: Multiple layers
- [x] Code: Forward pass
- [x] **Key Concepts:** Deep networks, layer composition
- [x] **Exercises Completed:** Spiral dataset visualization
- [x] **Notes:** Learned layer stacking and nnfs package usage

---

### Chapter 4: Activation Functions
- [x] Read chapter
- [x] Code: Step function
- [x] Code: Sigmoid
- [x] Code: ReLU
- [x] Code: Softmax
- [x] **Key Concepts:** Non-linearity, output interpretation
- [x] **Exercises Completed:** Interactive bias sliders, visual bias demo
- [x] **Notes:** Documented the "Inside vs Outside" bias rule

---

## Part II - Training

### Chapter 5: Calculating Loss
- [x] Read chapter
- [ ] Code: Categorical Cross-Entropy
- [ ] Code: Loss calculation
- [x] **Key Concepts:** Loss functions, optimization goals, target selection
- [ ] **Exercises Completed:**
- [x] **Notes:** Understanding how to select correct class probabilities from softmax outputs

---

### Chapter 6: Optimization
- [ ] Read chapter
- [ ] Code: Gradient descent
- [ ] Code: Learning rate experiments
- [ ] **Key Concepts:** Optimization, learning rate, gradient descent
- [ ] **Exercises Completed:**
- [ ] **Notes:**

---

### Chapter 7: Derivatives
- [ ] Read chapter
- [ ] Code: Numerical derivatives
- [ ] Understand chain rule
- [ ] **Key Concepts:** Derivatives, gradients, chain rule
- [ ] **Exercises Completed:**
- [ ] **Notes:**

---

### Chapter 8: Gradients, Partial Derivatives, and the Chain Rule
- [ ] Read chapter
- [ ] Code: Backpropagation for ReLU
- [ ] Code: Backpropagation for Softmax
- [ ] **Key Concepts:** Backpropagation, partial derivatives
- [ ] **Exercises Completed:**
- [ ] **Notes:**

---

## Part III - Advanced Topics

*Update as you progress through later chapters...*

---

## 🎯 Milestones
- [x] First neuron coded
- [x] First layer coded
- [x] First forward pass
- [ ] First backpropagation
- [ ] First complete training loop
- [ ] First working neural network on real data

---

## 💡 Insights & Aha Moments

### Chapter 2: Understanding NumPy Axes 🎯
**Initial misconception:**
I thought NumPy axes worked like Cartesian coordinates:
- axis=0 = X-axis (horizontal)
- axis=1 = Y-axis (vertical)
- axis=2 = Z-axis (depth)

**The reality:**
NumPy axes are about **indexing order**, not spatial coordinates!
- axis=0 = first dimension (rows, vertical ↓)
- axis=1 = second dimension (columns, horizontal →)
- axis=2 = third dimension (depth ⊙)

**Key insight:** When you operate along an axis, that dimension "collapses"
- `(2, 3)` with `axis=0` → `(3,)` ← rows disappeared
- `(2, 3)` with `axis=1` → `(2,)` ← columns disappeared

This is crucial for understanding batch operations in neural networks!

### Chapter 4: The "Inside vs Outside" Bias Rule 💡
**The Confusion:**
Why does changing bias sometimes move the graph Left/Right and other times Up/Down?

**The Aha Moment:**
It depends on WHERE the bias is added relative to the activation function!

1. **Inside the function** ($f(x+b)$): Shifts the **INPUT** (Left/Right)
   - "The Alarm Clock": Changing bias changes *when* the neuron activates.
   - This happens in the first layer: `ReLU(x + b)`

2. **Outside the function** ($f(x)+b$): Shifts the **OUTPUT** (Up/Down)
   - "The Platform": Changing bias lifts the entire result up or down.
   - This happens in subsequent layers: `ReLU(prev_layer) + b`

**Key Takeaway:** Layer 1 bias positions the "shape", Layer 2+ bias positions the "height".

See [bias_explanation.md](file:///d:/projects/NNFS/ch04_activation/bias_explanation.md) for the full analogy.

