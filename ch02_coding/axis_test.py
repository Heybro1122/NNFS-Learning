"""
Testing expand_dims with different axis values
"""
import numpy as np

print("=" * 60)
print("TESTING DIFFERENT AXIS VALUES")
print("=" * 60)

# 1D array
a = np.array([1, 2, 3])
print("\n1D Array:")
print("a =", a)
print("Shape:", a.shape, "← Only 1 dimension")

print("\n" + "-" * 60)

# axis=0 works
try:
    result = np.expand_dims(a, axis=0)
    print("\n✅ axis=0 WORKS:")
    print("   Result:", result)
    print("   Shape:", result.shape)
except Exception as e:
    print(f"\n❌ axis=0 failed: {e}")

# axis=1 works
try:
    result = np.expand_dims(a, axis=1)
    print("\n✅ axis=1 WORKS:")
    print("   Result:")
    print(result)
    print("   Shape:", result.shape)
except Exception as e:
    print(f"\n❌ axis=1 failed: {e}")

# axis=2 doesn't work for 1D array!
try:
    result = np.expand_dims(a, axis=2)
    print("\n✅ axis=2 WORKS:")
    print("   Result:", result)
    print("   Shape:", result.shape)
except Exception as e:
    print(f"\n❌ axis=2 FAILED: {e}")

print("\n" + "=" * 60)
print("BUT... axis=2 WORKS on 2D arrays!")
print("=" * 60)

# 2D array (matrix)
b = np.array([[1, 2, 3],
              [4, 5, 6]])
print("\n2D Array:")
print("b =")
print(b)
print("Shape:", b.shape, "← 2 dimensions (rows, cols)")

print("\n" + "-" * 60)

# Now axis=2 works!
try:
    result = np.expand_dims(b, axis=2)
    print("\n✅ axis=2 WORKS on 2D array:")
    print("   Result:")
    print(result)
    print("   Shape:", result.shape)
    print("   Explanation: Added a 3rd dimension!")
except Exception as e:
    print(f"\n❌ axis=2 failed: {e}")

print("\n" + "=" * 60)
print("THE RULE:")
print("=" * 60)
print("For an N-dimensional array, axis can be 0 to N (inclusive)")
print("- 1D array (shape: (3,))     → axis can be 0 or 1")
print("- 2D array (shape: (2, 3))   → axis can be 0, 1, or 2")
print("- 3D array (shape: (2, 3, 4)) → axis can be 0, 1, 2, or 3")
