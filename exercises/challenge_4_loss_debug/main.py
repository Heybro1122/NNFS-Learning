import numpy as np

# Predictions (Softmax output)
# Sample 1: Class 0 (Correct)
# Sample 2: Class 2 (Wrong, True is 1)
# Sample 3: Class 1 (Correct)
y_pred = np.array([[0.9, 0.05, 0.05],
                   [0.1, 0.3, 0.6],
                   [0.1, 0.8, 0.1]])

# True targets (Sparse)
y_true = np.array([0, 1, 1])

# ---------------------------------------------------------
# BUG 1: Accuracy is wrong
# ---------------------------------------------------------
# Currently just averaging the raw numbers... that's not accuracy!
# TODO: Find the predicted class (argmax) and compare to y_true
predictions = np.argmax(y_pred, axis=1) # Hint: This part is actually correct
accuracy = np.mean(predictions == y_true) # Is this right?
print(f"Accuracy: {accuracy:.2f}")

# ---------------------------------------------------------
# BUG 2: Target Selection is broken
# ---------------------------------------------------------
# We want the confidence of the CORRECT class
# Currently it's just taking the max confidence... cheating!
correct_confidences = y_pred[range(len(y_pred)), y_true] # <--- BUG! Fix this line using fancy indexing
print(f"Confidences used for loss: {correct_confidences}")

# ---------------------------------------------------------
# BUG 3: No Clipping
# ---------------------------------------------------------
# What if confidence is 0? Log(0) = -inf
# TODO: Clip the values between 1e-7 and 1-1e-7
clipped_confidences = np.clip(correct_confidences, 1e-7, 1-1e-7) # <--- Add clipping here

# Calculate Loss
loss = -np.log(clipped_confidences)
average_loss = np.mean(loss)

print(f"Average Loss: {average_loss:.4f}")
