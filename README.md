# Neural Networks from Scratch (NNFS) 🧠

> Building neural networks from scratch in NumPy to deeply understand how they work under the hood.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.3.5-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📚 What This Is

My hands-on journey through **"Neural Networks from Scratch in Python"** by Sentdex. Instead of just using PyTorch/TensorFlow, I'm implementing every component from scratch to build deep intuition about:

- How forward propagation really works
- The math behind activation functions
- Why we need loss functions and how they're calculated
- Backpropagation and the Chain Rule
- Gradient descent and optimization

## 🏗️ What I've Built

### ✅ Completed
- **Forward Pass**: Dense layers with weights and biases
- **Activation Functions**: ReLU, Softmax (with numerical stability)
- **Loss Functions**: Categorical Cross-Entropy with proper clipping
- **Challenges**: 4 hands-on coding challenges to reinforce concepts

### 🚧 In Progress
- **Backpropagation**: Implementing the backward pass (Chapter 9)
- **Optimization**: Gradient descent with learning rate

### 📂 Project Structure
```
NNFS/
├── ch01_intro/          # Introduction concepts
├── ch02_coding/         # First neurons
├── ch03_layers/         # Adding layers
├── ch04_activation/     # ReLU, Sigmoid, Softmax
│   ├── bias_demo.py
│   └── bias_explanation.md
├── ch05_loss/           # Loss calculation
│   ├── playground.py
│   ├── clipping_explained.md
│   └── understanding_targets.md
├── ch09_backprop/       # Backpropagation (current)
│   └── playground.py
├── exercises/           # Hands-on challenges
│   ├── challenge_1_forward_pass/
│   ├── challenge_2_shapes/
│   ├── challenge_3_activations/
│   ├── challenge_4_loss_debug/
│   └── challenge_5_backprop/
└── progress.md          # Learning journal
```

## 🔑 Key Learnings

### Bias Shifts (Ch 4)
- **Inside** the activation: Shifts the function horizontally
- **Outside** the activation: Shifts the function vertically

### Loss Calculation (Ch 5)
- **Clipping**: Prevent `log(0)` errors by clipping predictions to `[1e-7, 1-1e-7]`
- **Target Selection**: Use fancy indexing `y_pred[range(samples), y_true]` to extract correct class confidences
- **One-hot vs Sparse**: Handle both target formats with conditional logic

### Gradient Flow (Ch 9)
- **Chain Rule**: Gradients flow backward through nested functions
- **Parameter Updates**: `w_new = w_old - learning_rate * gradient`

## 🚀 Running the Code

```bash
# Install dependencies
pip install numpy matplotlib nnfs

# Run playground for any chapter
cd ch05_loss
python playground.py

# Try challenges
cd exercises/challenge_1_forward_pass
python main.py
```

## 📊 Progress Tracking

**Current Chapter**: 9 - Backpropagation  
**Chapters Completed**: 5/18  
**Challenges Completed**: 4/5

See [progress.md](progress.md) for detailed notes and insights.

## 🎯 What's Next

1. Finish implementing backpropagation
2. Add optimization algorithms (SGD, Adam)
3. Build a complete training loop
4. Train on real datasets (MNIST, Fashion-MNIST)
5. Implement CNNs and Transformers

## 🙏 Resources

- **Book**: [Neural Networks from Scratch](https://nnfs.io) by Sentdex
- **Video Series**: [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- **Challenges**: Custom exercises to reinforce concepts

---

**Learning in public** 🌱

*Built with ❤️ while understanding how AI actually works*
