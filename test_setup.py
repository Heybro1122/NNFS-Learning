"""Quick test to verify Python and NumPy are working"""
import sys

print("Python version:", sys.version)
print("✅ Python is working!")

try:
    import numpy as np
    print(f"✅ NumPy {np.__version__} is installed!")
    
    # Quick test
    arr = np.array([1, 2, 3, 4, 5])
    print(f"✅ NumPy test array: {arr}")
    print(f"✅ Sum: {np.sum(arr)}")
    
except ImportError:
    print("❌ NumPy is not installed yet")
    print("Run: py -m pip install numpy matplotlib")

try:
    import matplotlib
    print(f"✅ Matplotlib {matplotlib.__version__} is installed!")
except ImportError:
    print("❌ Matplotlib is not installed yet")
    print("Run: py -m pip install matplotlib")

print("\n🎉 Setup check complete!")
