import numpy as np 
softmax_outputs = np.array([[0.7, 0.1, 0.2],[0.1, 0.5, 0.4],[0.02, 0.9, 0.08]])  # Convert to NumPy array!
neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), [0,1,1]])
mean= np.mean(neg_log)
print(mean)



