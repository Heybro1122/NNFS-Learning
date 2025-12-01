# Python Environment Setup for NNFS

## Prerequisites
- Python 3.8 or higher installed
- Basic command line familiarity

## Step 1: Verify Python Installation

Open PowerShell and run:
```powershell
python --version
```

You should see something like `Python 3.x.x`. If not, download Python from [python.org](https://www.python.org/downloads/).

## Step 2: Create a Virtual Environment

Navigate to your NNFS directory:
```powershell
cd d:\projects\NNFS
```

Create a virtual environment:
```powershell
python -m venv venv
```

## Step 3: Activate the Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` appear in your prompt.

> [!TIP]
> If you get an execution policy error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

## Step 4: Install Required Packages

Install NumPy (the main package needed for NNFS):
```powershell
pip install numpy matplotlib
```

- **NumPy**: For numerical computations and arrays
- **Matplotlib**: For visualizing data and results

## Step 5: Verify Installation

Create a test file:
```python
import numpy as np
import matplotlib.pyplot as plt

print("NumPy version:", np.__version__)
print("Setup successful! ✅")

# Quick test
arr = np.array([1, 2, 3, 4, 5])
print("Test array:", arr)
```

Run it:
```powershell
python test.py
```

## Optional: Install Jupyter Notebook

For interactive coding (highly recommended for learning):
```powershell
pip install jupyter
```

Launch Jupyter:
```powershell
jupyter notebook
```

## Daily Workflow

1. Open PowerShell
2. Navigate to NNFS directory: `cd d:\projects\NNFS`
3. Activate virtual environment: `.\venv\Scripts\Activate.ps1`
4. Start coding! 🚀

## Deactivating

When you're done:
```powershell
deactivate
```

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python script.py` | Run a Python script |
| `python -i script.py` | Run and stay in interactive mode |
| `pip list` | Show installed packages |
| `pip install package_name` | Install a package |

---

> [!NOTE]
> Throughout the book, you'll mainly use NumPy. The virtual environment keeps your NNFS packages separate from your system Python.
