# Python Primer for NNFS

A focused guide on Python concepts you'll need for Neural Networks from Scratch.

## 🐍 Python Basics

### Variables and Types
```python
# Python is dynamically typed (no type declarations needed)
x = 5           # int
y = 3.14        # float
name = "Alice"  # string
ready = True    # boolean

# Type checking
print(type(x))  # <class 'int'>
```

### Lists (Python's basic arrays)
```python
# Creating lists
numbers = [1, 2, 3, 4, 5]
mixed = [1, "two", 3.0, True]

# Indexing (0-based)
first = numbers[0]      # 1
last = numbers[-1]      # 5

# Slicing
subset = numbers[1:4]   # [2, 3, 4] (start:end, end is exclusive)

# List operations
numbers.append(6)       # Add to end
length = len(numbers)   # Get length
```

### Loops
```python
# For loop
for num in [1, 2, 3]:
    print(num)

# Range (useful for iterations)
for i in range(5):      # 0, 1, 2, 3, 4
    print(i)

for i in range(2, 7):   # 2, 3, 4, 5, 6
    print(i)

# While loop
count = 0
while count < 5:
    print(count)
    count += 1
```

### Functions
```python
# Define a function
def add(a, b):
    return a + b

# Call it
result = add(3, 5)  # 8

# Multiple return values
def get_stats(numbers):
    return sum(numbers), len(numbers)

total, count = get_stats([1, 2, 3])
```

### List Comprehensions (Pythonic!)
```python
# Instead of:
squares = []
for i in range(5):
    squares.append(i ** 2)

# Write:
squares = [i ** 2 for i in range(5)]  # [0, 1, 4, 9, 16]
```

---

## 🔢 NumPy Essentials

NumPy is THE library for numerical computing in Python. It's what makes NNFS possible!

### Why NumPy?
- **Fast**: Operations on arrays are 10-100x faster than Python lists
- **Convenient**: Built-in matrix operations
- **Memory efficient**: Stores data more compactly

### Creating Arrays
```python
import numpy as np

# From lists
arr = np.array([1, 2, 3, 4])
matrix = np.array([[1, 2], [3, 4]])

# Special arrays
zeros = np.zeros((3, 4))      # 3x4 matrix of zeros
ones = np.ones((2, 3))        # 2x3 matrix of ones
identity = np.eye(3)          # 3x3 identity matrix
random = np.random.rand(2, 3) # 2x3 random values [0, 1)

# Ranges
arr = np.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
arr = np.linspace(0, 1, 5)    # [0.0, 0.25, 0.5, 0.75, 1.0]
```

### Array Attributes
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr.shape)    # (2, 3) - dimensions
print(arr.ndim)     # 2 - number of dimensions
print(arr.size)     # 6 - total elements
print(arr.dtype)    # dtype('int64') - data type
```

### Indexing and Slicing
```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Element access
element = arr[0, 1]        # 2 (row 0, col 1)

# Row/column access
row = arr[1]               # [4, 5, 6]
col = arr[:, 2]            # [3, 6, 9] (all rows, col 2)

# Slicing
sub = arr[0:2, 1:3]        # [[2, 3], [5, 6]]
```

### Array Operations (Element-wise)
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Arithmetic (element-wise!)
c = a + b       # [5, 7, 9]
c = a * b       # [4, 10, 18]
c = a ** 2      # [1, 4, 9]
c = a + 10      # [11, 12, 13] - broadcasting!

# Boolean operations
mask = a > 1    # [False, True, True]
filtered = a[a > 1]  # [2, 3]
```

### Matrix Operations (The Important Ones!)
```python
# Dot product (crucial for neural networks!)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32

# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.dot(A, B)  # or A @ B (Python 3.5+)
# [[19, 22],
#  [43, 50]]

# Transpose
A_T = A.T
```

### Reshaping
```python
arr = np.array([1, 2, 3, 4, 5, 6])

# Reshape to 2x3
matrix = arr.reshape(2, 3)
# [[1, 2, 3],
#  [4, 5, 6]]

# Reshape to 3x2
matrix = arr.reshape(3, 2)
# [[1, 2],
#  [3, 4],
#  [5, 6]]

# Flatten
flat = matrix.reshape(-1)  # or matrix.flatten()
```

### Useful Functions
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Aggregations
print(np.sum(arr))         # 21 (all elements)
print(np.sum(arr, axis=0)) # [5, 7, 9] (sum along rows)
print(np.sum(arr, axis=1)) # [6, 15] (sum along columns)

print(np.mean(arr))        # 3.5
print(np.max(arr))         # 6
print(np.min(arr))         # 1

# Apply functions
print(np.exp(arr))         # e^x for each element
print(np.sqrt(arr))        # Square root
print(np.log(arr))         # Natural log
```

---

## 🎯 Key Concepts for NNFS

### Broadcasting
NumPy can operate on arrays of different shapes:
```python
# Add a scalar to an array
arr = np.array([1, 2, 3])
result = arr + 10  # [11, 12, 13]

# Add a row vector to each row of a matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
row = np.array([10, 20, 30])
result = matrix + row
# [[11, 22, 33],
#  [14, 25, 36]]
```

### Axis Understanding
- **axis=0**: Along rows (vertically, ↓)
- **axis=1**: Along columns (horizontally, →)

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

# Sum along axis=0 (collapse rows)
print(np.sum(matrix, axis=0))  # [5, 7, 9]

# Sum along axis=1 (collapse columns)
print(np.sum(matrix, axis=1))  # [6, 15]
```

### Random Numbers (for initialization)
```python
# Random values in [0, 1)
np.random.rand(3, 4)

# Random integers
np.random.randint(0, 10, size=(3, 4))

# Random from normal distribution
np.random.randn(3, 4)  # Mean=0, Std=1

# Set seed for reproducibility
np.random.seed(42)
```

---

## 💡 Tips for Learning

1. **Try everything in the Python REPL**: Just type `python` in your terminal
2. **Use `help()`**: `help(np.dot)` shows documentation
3. **Check shapes often**: `print(arr.shape)` prevents many bugs
4. **Visualize**: Draw out matrix dimensions on paper

---

## 🔗 Quick Syntax Comparison

| Math Notation | NumPy Code |
|---------------|------------|
| x⃗ · y⃗ | `np.dot(x, y)` |
| AB | `np.dot(A, B)` or `A @ B` |
| Aᵀ | `A.T` |
| Σxᵢ | `np.sum(x)` |
| max(x) | `np.max(x)` |
| e^x | `np.exp(x)` |

---

> [!TIP]
> You don't need to memorize everything! Keep this guide handy and refer back as you code through the book.

> [!NOTE]
> Your strong linear algebra background will map directly to NumPy. Think of arrays as vectors and matrices!
