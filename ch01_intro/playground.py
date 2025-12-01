import numpy as np

# Create the same numbers in different shapes
a = np.array([1, 2, 3])           # 1D
b = np.array([[1, 2, 3]])         # Row vector (2D)
c = np.array([[1], [2], [3]])     # Column vector (2D)

print("1D array:")
print(a)
print("Shape:", a.shape, "\n")

print("Row vector:")
print(b)
print("Shape:", b.shape, "\n")

print("Column vector:")
print(c)
print("Shape:", c.shape, "\n")

# Try expand_dims
d = np.expand_dims(a, axis=0)
print("Expanded (axis=0):")
print(d)
print("Shape:", d.shape)