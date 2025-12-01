# Challenge 4: The Loss Detective 🕵️

**Objective:** Debug a broken loss calculation.

## The Mission
The code calculates accuracy and loss, but the results don't make sense. Find the logic errors!

## The Scenario
We have a batch of 3 samples.
*   Sample 1: Correct
*   Sample 2: Wrong
*   Sample 3: Correct

## Tasks
1.  **Fix the Accuracy Calculation**: It's currently calculating the mean of the raw outputs, not the *matches*.
2.  **Fix the Loss Calculation**: It's not clipping values, causing a potential infinity error.
3.  **Fix the Target Selection**: It's grabbing the wrong indices.

## Expected Output
*   Accuracy: ~0.66 (2 out of 3 correct)
*   Loss: A reasonable positive number (not inf, not negative)
