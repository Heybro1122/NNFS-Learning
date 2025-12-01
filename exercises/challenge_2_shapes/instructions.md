# Challenge 2: The Shape Master 📐

**Objective:** Master NumPy shapes and matrix multiplication dimensions. This is the #1 source of bugs in Deep Learning!

## The Mission
Fix the broken code in `main.py`. It has multiple shape mismatch errors.

## Concepts to Master
*   Matrix Multiplication Rule: `(A, B) dot (B, C) -> (A, C)`
*   Transposing: `weights.T` swaps rows and columns.
*   Broadcasting: Adding a bias vector `(1, N)` to a matrix `(M, N)`.

## Tasks
1.  Run the code and read the error message.
2.  Fix the shape mismatch in Layer 1.
3.  Fix the shape mismatch in Layer 2.
4.  Ensure the final output shape is correct for 10 samples and 3 classes.

## Expected Output
A successful run printing the final output shape: `(10, 3)`.
