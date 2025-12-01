"""
Understanding np.expand_dims() - Visual Demo
"""
import numpy as np

print("=" * 60)
print("UNDERSTANDING np.expand_dims()")
print("=" * 60)

# Start with a simple 1D array
a = np.array([1, 2, 3])

print("\n1. ORIGINAL ARRAY:")
print("   a =", a)
print("   Shape:", a.shape)
print("   This is 1D - just a list of 3 numbers")

print("\n" + "-" * 60)

# Expand at axis=0 (add dimension at the FRONT)
b = np.expand_dims(a, axis=0)

print("\n2. AFTER expand_dims(a, axis=0):")
print("   b =", b)
print("   Shape:", b.shape)
print("   This is 2D - a matrix with 1 row and 3 columns")
print("   Think: We wrapped the array in ANOTHER set of brackets")
print("   [1, 2, 3] became [[1, 2, 3]]")

print("\n" + "-" * 60)

# Expand at axis=1 (add dimension at the END)
c = np.expand_dims(a, axis=1)

print("\n3. AFTER expand_dims(a, axis=1):")
print("   c =")
print(c)
print("   Shape:", c.shape)
print("   This is 2D - a matrix with 3 rows and 1 column")
print("   Think: Each number got its own brackets")
print("   [1, 2, 3] became [[1], [2], [3]]")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Original:  {a}  shape {a.shape}")
print(f"axis=0:    {b}  shape {b.shape}  ← ROW vector")
print(f"axis=1:    [[1] [2] [3]]  shape {c.shape}  ← COLUMN vector")

print("\n" + "=" * 60)
print("THE KEY INSIGHT:")
print("=" * 60)
print("axis=0 means: 'Add a dimension at position 0 (the front)'")
print("  - Shape (3,) becomes (1, 3)")
print("  - You get a ROW vector")
print()
print("axis=1 means: 'Add a dimension at position 1 (the end)'")
print("  - Shape (3,) becomes (3, 1)")
print("  - You get a COLUMN vector")
