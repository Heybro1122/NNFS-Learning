"""
Understanding Axes in NumPy - What Does Axis Mean?
"""
import numpy as np

print("=" * 70)
print("UNDERSTANDING AXIS IN NUMPY")
print("=" * 70)

# Create a 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print("\nOriginal Array:")
print(arr)
print(f"Shape: {arr.shape}")
print("\n        Column 0  Column 1  Column 2")
print("        ↓         ↓         ↓")
print("Row 0:  1         2         3")
print("Row 1:  4         5         6")

print("\n" + "=" * 70)
print("OPERATIONS ALONG DIFFERENT AXES")
print("=" * 70)

# Sum along axis=0 (collapse ROWS)
print("\n1. np.sum(arr, axis=0)  → Sum along axis=0 (↓ down rows)")
result = np.sum(arr, axis=0)
print(f"   Result: {result}")
print("   Explanation:")
print("   - Column 0: 1 + 4 = 5")
print("   - Column 1: 2 + 5 = 7")
print("   - Column 2: 3 + 6 = 9")
print("   You 'collapsed' the rows (axis 0) by summing DOWN")

# Sum along axis=1 (collapse COLUMNS)
print("\n2. np.sum(arr, axis=1)  → Sum along axis=1 (→ across columns)")
result = np.sum(arr, axis=1)
print(f"   Result: {result}")
print("   Explanation:")
print("   - Row 0: 1 + 2 + 3 = 6")
print("   - Row 1: 4 + 5 + 6 = 15")
print("   You 'collapsed' the columns (axis 1) by summing ACROSS")

# Sum with no axis (total sum)
print("\n3. np.sum(arr)  → Sum everything (no axis specified)")
result = np.sum(arr)
print(f"   Result: {result}")
print("   Explanation: 1 + 2 + 3 + 4 + 5 + 6 = 21")

print("\n" + "=" * 70)
print("OTHER OPERATIONS WITH AXIS")
print("=" * 70)

print("\n4. Maximum along each axis:")
print(f"   np.max(arr, axis=0) = {np.max(arr, axis=0)}  ← max of each column")
print(f"   np.max(arr, axis=1) = {np.max(arr, axis=1)}  ← max of each row")

print("\n5. Mean along each axis:")
print(f"   np.mean(arr, axis=0) = {np.mean(arr, axis=0)}  ← mean of each column")
print(f"   np.mean(arr, axis=1) = {np.mean(arr, axis=1)}  ← mean of each row")

print("\n" + "=" * 70)
print("THE KEY INSIGHT")
print("=" * 70)
print("""
When you specify axis=N, you're telling NumPy:
"Perform this operation ALONG this axis (direction)"

axis=0 → Operate along rows (↓) → Result has COLUMNS left
axis=1 → Operate along columns (→) → Result has ROWS left

Think of it as: "The axis you specify DISAPPEARS from the result"
- Original shape: (2, 3)
- axis=0: (2, 3) → (3,)  [rows disappeared]
- axis=1: (2, 3) → (2,)  [columns disappeared]
""")

print("\n" + "=" * 70)
print("3D EXAMPLE (For Completeness)")
print("=" * 70)

# Create a 3D array
arr_3d = np.array([[[1, 2],
                    [3, 4]],
                   
                   [[5, 6],
                    [7, 8]]])

print(f"\n3D Array shape: {arr_3d.shape}  → (2, 2, 2)")
print("Think of it as: 2 matrices, each 2×2")
print("\nMatrix 0:")
print(arr_3d[0])
print("\nMatrix 1:")
print(arr_3d[1])

print("\nOperations:")
print(f"axis=0: {np.sum(arr_3d, axis=0).shape}  ← Sum across matrices")
print(f"axis=1: {np.sum(arr_3d, axis=1).shape}  ← Sum across rows")
print(f"axis=2: {np.sum(arr_3d, axis=2).shape}  ← Sum across columns")

print("\n" + "=" * 70)
