# Challenge 1: The Full Forward Pass 🏗️

**Objective:** Build a complete neural network forward pass from scratch using Object-Oriented Programming (OOP).

## The Architecture
You will build a network to classify the **Spiral Dataset**.

*   **Input Data**: 2 features (x, y coordinates)
*   **Hidden Layer 1**: 3 neurons, ReLU activation
*   **Output Layer**: 3 neurons (one for each class), Softmax activation
*   **Loss Function**: Categorical Cross-Entropy

## Tasks

### 1. Create the Classes
Implement the following classes in `main.py`. Do NOT use the `nnfs` library for the classes (only for data generation). Use `numpy`.

*   **`Layer_Dense`**
    *   `__init__(self, n_inputs, n_neurons)`: Initialize weights (random) and biases (zeros).
    *   `forward(self, inputs)`: Calculate `dot(inputs, weights) + biases`. Store result in `self.output`.

*   **`Activation_ReLU`**
    *   `forward(self, inputs)`: Apply ReLU (`max(0, x)`). Store result in `self.output`.

*   **`Activation_Softmax`**
    *   `forward(self, inputs)`: Apply Softmax (exponentiate, normalize). Store result in `self.output`.

*   **`Loss_CategoricalCrossentropy`**
    *   `forward(self, y_pred, y_true)`: Calculate the negative log likelihood. Handle both sparse and one-hot targets. Clip values to avoid infinity. Return the average loss.

### 2. The Execution Script
In the main block:
1.  Generate spiral data: `X, y = spiral_data(samples=100, classes=3)`
2.  Create an instance of `Layer_Dense` (2 inputs, 3 neurons).
3.  Create an instance of `Activation_ReLU`.
4.  Create an instance of `Layer_Dense` (3 inputs, 3 neurons).
5.  Create an instance of `Activation_Softmax`.
6.  Create an instance of `Loss_CategoricalCrossentropy`.
7.  **Run the Forward Pass**: Pass data through each object sequentially.
8.  **Calculate Loss**: Compare final output to `y`.
9.  **Print**: The final loss value.

## Hints
*   Remember `np.random.randn` for weights (multiply by 0.01 to keep them small).
*   Remember `np.zeros` for biases.
*   Softmax needs `axis=1` and `keepdims=True`.
*   Loss needs clipping (`1e-7`, `1 - 1e-7`).

**Good luck! This is the real deal!** 🚀
